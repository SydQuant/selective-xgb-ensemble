
import argparse, numpy as np, pandas as pd, os, json
# CHANGED: Added logging and real data support
import logging
import yaml
from cv.wfo import wfo_splits
from model.xgb_drivers import generate_xgb_specs, fold_train_predict, stratified_xgb_bank, fold_train_predict_tiered, generate_deep_xgb_specs
from model.feature_selection import apply_feature_selection
from ensemble.combiner import build_driver_signals, combine_signals, softmax
from ensemble.selection import pick_top_n_greedy_diverse
from ensemble.gating import apply_pvalue_gating
# ADDED: Stability ensemble support
from ensemble.stability_selection import stability_driver_selection_and_combination, StabilityConfig
from opt.grope import grope_optimize
from opt.weight_objective import weight_objective_factory
from metrics.dapy import hit_rate
from metrics.simple_metrics import information_ratio
from eval.target_shuffling import shuffle_pvalue
from metrics.objective_registry import OBJECTIVE_REGISTRY, create_composite_objective
# CHANGED: Added real data loading support
from data.data_utils_simple import prepare_real_data_simple
from data.symbol_loader import get_default_symbols
# CHANGED: Added comprehensive performance reporting
from metrics.performance_report import (
    calculate_returns_metrics, format_performance_report, 
    format_symbol_breakdown, save_performance_csv
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_dapy_fn(style: str):
    style = (style or "hits").lower()
    if style == "hits":
        from metrics.dapy import dapy_from_binary_hits as fn
        return fn
    elif style == "eri_long":
        from metrics.dapy import dapy_eri_long as fn
        return fn
    elif style == "eri_short":
        from metrics.dapy import dapy_eri_short as fn
        return fn
    elif style == "eri_both":
        from metrics.dapy import dapy_eri_both as fn
        return fn
    elif style in ['adjusted_sharpe', 'cb_ratio', 'information_ratio', 'predictive_icir_logscore']:
        # Support all 5 universal objective functions in driver selection
        try:
            return OBJECTIVE_REGISTRY.get(style)
        except:
            logger.warning(f"Unknown driver_selection objective: {style}, falling back to hits")
            from metrics.dapy import dapy_from_binary_hits as fn
            return fn
    else:
        from metrics.dapy import dapy_from_binary_hits as fn
        return fn

def get_grope_objective_function(grope_objective: str):
    """Get GROPE weight optimization objective function."""
    if not grope_objective:
        return None
    
    grope_objective = grope_objective.lower()
    
    if grope_objective == "hits":
        from metrics.dapy import dapy_from_binary_hits
        return dapy_from_binary_hits
    elif grope_objective == "eri_both":
        from metrics.dapy import dapy_eri_both
        return dapy_eri_both
    elif grope_objective in ['adjusted_sharpe', 'cb_ratio', 'information_ratio', 'predictive_icir_logscore']:
        try:
            return OBJECTIVE_REGISTRY.get(grope_objective)
        except:
            logger.warning(f"Unknown GROPE objective: {grope_objective}, using default")
            return None
    else:
        logger.warning(f"Unknown GROPE objective: {grope_objective}, using default")
        return None

def get_objective_functions(args):
    """Create objective functions from config - STRICT MODE: NO FALLBACKS."""
    from metrics.objective_registry import OBJECTIVE_REGISTRY
    
    driver_selection_obj = None
    weight_optimization_obj = None
    
    # Try new config structure first
    if hasattr(args, 'driver_selection_objective') and args.driver_selection_objective:
        driver_selection_obj = create_composite_objective(args.driver_selection_objective, OBJECTIVE_REGISTRY)
    elif args.driver_selection:
        # Use CLI argument directly with OBJECTIVE_REGISTRY
        driver_selection_obj = OBJECTIVE_REGISTRY.get(args.driver_selection)
    else:
        raise ValueError("No driver_selection objective specified")
    
    if hasattr(args, 'grope_weight_objective') and args.grope_weight_objective:
        weight_optimization_obj = create_composite_objective(args.grope_weight_objective, OBJECTIVE_REGISTRY)
    elif hasattr(args, 'grope_objective') and args.grope_objective:
        weight_optimization_obj = OBJECTIVE_REGISTRY.get(args.grope_objective)
    else:
        # Default to same as driver selection
        weight_optimization_obj = driver_selection_obj
    
    # STRICT MODE: No fallbacks, require valid objectives
    if not driver_selection_obj:
        raise ValueError(f"Invalid or missing driver_selection objective: {getattr(args, 'driver_selection', None)}")
    if not weight_optimization_obj:
        raise ValueError(f"Invalid or missing grope_objective: {getattr(args, 'grope_objective', None)}")
    
    # No dapy_fn in strict mode
    return None, driver_selection_obj, weight_optimization_obj

def setup_xgb_specs(X_columns, args):
    """Setup XGBoost specifications based on architecture choice.
    
    Returns:
        tuple: (specs, col_slices) where col_slices is None for non-tiered approaches
    """
    # Choose XGBoost architecture: tiered (stratified features), deep (complex trees), or standard
    if args.tiered_xgb:
        specs, col_slices = stratified_xgb_bank(X_columns.tolist(), n_models=args.n_models, seed=7)
        return specs, col_slices
    elif args.deep_xgb:
        specs = generate_deep_xgb_specs(n_models=args.n_models, seed=7)
        return specs, None
    else:
        # Standard architecture - most commonly used and reliable
        specs = generate_xgb_specs(n_models=args.n_models, seed=7)
        return specs, None

def train_fold_models(X_tr, y_tr, X_te, specs, col_slices, args):
    """Train XGBoost models and return predictions based on architecture choice."""
    # Train models based on architecture: tiered uses column slicing, standard uses all features
    if args.tiered_xgb and col_slices is not None:
        return fold_train_predict_tiered(X_tr, y_tr, X_te, specs, col_slices)
    else:
        return fold_train_predict(X_tr, y_tr, X_te, specs)

def select_and_optimize_drivers(signals_tr, signals_te_or_full, y_tr, args, driver_selection_obj, weight_optimization_obj, seed=1234):
    """Unified driver selection and weight optimization logic.
    
    Args:
        signals_tr: Training signals list
        signals_te_or_full: Test signals (for CV) or full signals (for production)
        y_tr: Training target
        args: Arguments object
        driver_selection_obj: Driver selection objective function
        weight_optimization_obj: Weight optimization objective function
        seed: Random seed for GROPE optimization
        
    Returns:
        tuple: (combined_signal, selection_info) where selection_info contains chosen_idx, weights, tau, J_star
    """
    # CHECK: Use stability ensemble instead of GROPE?
    if getattr(args, 'use_stability_ensemble', False):
        logger.info("Using stability ensemble method instead of GROPE optimization")
        
        # Create stability configuration from args - handle nested config dictionary
        stability_section = getattr(args, 'stability', {})
        stability_config = StabilityConfig(
            metric_name=stability_section.get('metric_name', 'sharpe'),
            top_k=stability_section.get('top_k', 5),
            alpha=stability_section.get('alpha', 1.0),
            lam_gap=stability_section.get('lam_gap', 0.3),
            relative_gap=stability_section.get('relative_gap', False),
            eta_quality=stability_section.get('eta_quality', 0.0),
            quality_halflife=stability_section.get('quality_halflife', 63),
            inner_val_frac=stability_section.get('inner_val_frac', 0.2),
            costs_per_turn=stability_section.get('costs_per_turn', 0.0001)
        )
        
        # Stability method handles both selection and combination
        combined_signal, diagnostics = stability_driver_selection_and_combination(
            signals_tr, signals_te_or_full, y_tr, stability_config
        )
        
        if combined_signal is None or len(combined_signal) == 0:
            return None, None
            
        # Format selection info to match GROPE format
        selection_info = {
            "method": "stability_ensemble",
            "chosen_idx": diagnostics.get("selected_indices", []),
            "weights": [1.0/len(diagnostics.get("selected_indices", [1]))] * len(diagnostics.get("selected_indices", [1])),  # Equal weights
            "tau": 1.0,  # Not applicable for stability
            "J_star": diagnostics.get("mean_stability", 0.0),  # Use stability score instead
            "selection_diagnostics": diagnostics
        }
        
        return combined_signal, selection_info
    
    # ORIGINAL GROPE METHOD (default behavior)
    # P-value gating function - filters out statistically insignificant signals
    gate = lambda sig, y_local: apply_pvalue_gating(sig, y_local, args)
    
    # Greedy diverse selection - picks best n_select signals with diversity penalty
    chosen_idx, selection_diagnostics = pick_top_n_greedy_diverse(
        signals_tr, y_tr, n=args.n_select, pval_gate=gate,
        objective_fn=driver_selection_obj, diversity_penalty=args.diversity_penalty,
        objective_name=args.driver_selection
    )
    
    if len(chosen_idx) == 0:
        return None, None
        
    train_sel = [signals_tr[i] for i in chosen_idx]
    output_sel = [signals_te_or_full[i] for i in chosen_idx]
    
    # Weight optimization - either equal weights or GROPE optimization
    if args.equal_weights:
        w = np.ones(len(train_sel))
        tau = 1.0
        ww = softmax(w, temperature=tau)
        J_star = 0.0
    else:
        # GROPE optimization - optimize ensemble weights and temperature parameter
        bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
        fobj = weight_objective_factory(
            train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, 
            w_dapy=args.w_dapy, w_ir=args.w_ir, objective_fn=weight_optimization_obj
        )
        # Global optimization to find best weights (w0,w1,...) and temperature (tau)
        theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=seed)
        
        # Extract optimized weights and apply softmax normalization
        w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
        tau = float(theta_star["tau"])
        ww = softmax(w, temperature=tau)  # Convert raw weights to normalized probabilities
    
    # Combine selected signals using optimized weights
    combined_signal = combine_signals(output_sel, ww)
    
    selection_info = {
        "chosen_idx": chosen_idx,
        "weights": ww.tolist(),
        "tau": tau,
        "J_train": J_star
    }
    
    return combined_signal, selection_info

def save_timeseries(path: str, signal: pd.Series, y: pd.Series):
    """Save trading timeseries with proper 1-day lag to avoid look-ahead bias."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    # CRITICAL: Lag signal by 1 day to avoid look-ahead bias
    pnl = (signal.shift(1).fillna(0.0) * y.reindex_like(signal)).astype(float)
    eq = pnl.cumsum()  # Cumulative equity curve
    out = pd.DataFrame({'signal': signal, 'target_ret': y.reindex_like(signal), 'pnl': pnl, 'equity': eq})
    out.to_csv(path)
    return out

def save_oos_artifacts(signal: pd.Series, y: pd.Series, block: int, final_shuffles: int, dapy_fn):
    """Save out-of-sample artifacts and perform statistical significance testing."""
    os.makedirs("artifacts", exist_ok=True)
    out = save_timeseries("artifacts/oos_timeseries.csv", signal, y)
    # Statistical significance testing via block shuffling (preserves autocorrelation)
    pval, obs, _ = shuffle_pvalue(signal, y, dapy_fn, n_shuffles=final_shuffles, block=block)
    print(f"OOS Shuffling p-value (DAPY): {pval:.10f} (obs={obs:.2f})")
    return out

def main(args):
    """Main dispatcher - handles single or multiple targets."""
    dapy_fn, driver_selection_obj, weight_optimization_obj = get_objective_functions(args)
    
    target_symbols = [s.strip() for s in args.target_symbol.split(',')]
    logger.info(f"Starting analysis for targets: {target_symbols}")
    
    if args.symbols:
        # Handle both string and list formats
        if isinstance(args.symbols, str):
            symbols = args.symbols.split(',')
        else:
            symbols = args.symbols  # Already a list from YAML
        # Clean up symbol names
        symbols = [s.strip() for s in symbols]
    else:
        symbols = get_default_symbols()
    
    results = {}
    for target_symbol in target_symbols:
        logger.info(f"Processing {target_symbol}...")
        result = run_single_target(target_symbol, args, symbols, dapy_fn, driver_selection_obj, weight_optimization_obj)
        results[target_symbol] = result
    
    successful = sum(1 for r in results.values() if r is not None)
    logger.info(f"Completed {successful}/{len(target_symbols)} targets")
    return results

def run_single_target(target_symbol: str, args, symbols: list, dapy_fn, driver_selection_obj=None, weight_optimization_obj=None):
    """Run analysis for a single target symbol."""
    original_target = args.target_symbol
    args.target_symbol = target_symbol
    
    try:
        df = prepare_real_data_simple(
            target_symbol=target_symbol,
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            n_hours=args.n_hours,
            signal_hour=args.signal_hour
        )
        
        if df.empty:
            logger.error(f"No data loaded for {target_symbol}")
            return None
        
        target_col = f"{target_symbol}_target_return"
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found")
            return None
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        logger.info(f"Data loaded: {X.shape}, date range {X.index.min()} to {X.index.max()}")
        return run_training_pipeline(X, y, args, dapy_fn, driver_selection_obj, weight_optimization_obj)
    
    finally:
        args.target_symbol = original_target

def run_training_pipeline(X, y, args, dapy_fn, driver_selection_obj=None, weight_optimization_obj=None):
    """Run the XGB training pipeline."""
    
    # STRICT MODE: Require valid objectives
    if not driver_selection_obj:
        raise ValueError("driver_selection_obj is required")
    if not weight_optimization_obj:
        raise ValueError("weight_optimization_obj is required")
    
    # Adjust folds for limited data
    if len(X) < args.folds * 100:
        args.folds = max(2, len(X) // 100)
        logger.info(f"Adjusted to {args.folds} folds for limited data")

    # Apply block-wise feature selection BEFORE training
    # For large datasets, reduce features more aggressively to prevent XGB overfitting
    original_feature_count = X.shape[1]
    features_after_selection = X.shape[1]
    
    if X.shape[1] > 200:  # Only apply if we have many features
        # Use CLI max_features if provided, otherwise use adaptive calculation
        target_features = args.max_features if args.max_features else max(30, min(50, len(X) // 20))
        
        X = apply_feature_selection(X, y, method='block_wise', 
                                   block_size=100, features_per_block=15, 
                                   max_total_features=target_features, corr_threshold=args.corr_threshold)
        features_after_selection = X.shape[1]
        logger.info(f"Feature selection: {original_feature_count} -> {features_after_selection} features")
    
    if args.train_test_split:
        n_data = len(X)
        train_size = int(0.7 * n_data)
        splits = [(np.arange(0, train_size), np.arange(train_size, n_data))]
    else:
        # UPDATED: wfo_splits now returns generator, convert to list for compatibility
        splits = list(wfo_splits(len(X), k_folds=args.folds, min_train=max(252, len(X)//(args.folds*2))))
    
    specs, col_slices = setup_xgb_specs(X.columns, args)
        
    os.makedirs("artifacts", exist_ok=True)

    oos_signal = pd.Series(0.0, index=X.index)
    fold_summaries = []

    for f, (tr, te) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te = X.iloc[te]  # Removed unused y_te

        # Train models using the unified helper function
        train_preds, test_preds = train_fold_models(X_tr, y_tr, X_te, specs, col_slices, args)
        
        
        s_tr, s_te = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

        # Add comprehensive OOS diagnostics for all models
        try:
            from ensemble.oos_diagnostics import compute_oos_model_diagnostics
            y_te = y.iloc[te]  # Get actual test target for OOS analysis
            compute_oos_model_diagnostics(
                test_signals=s_te,
                y_test=y_te,
                objective_fn=driver_selection_obj,
                objective_name=args.driver_selection,
                fold_id=f,
                n_shuffles=200,
                show_top_n=75,
                pvalue_threshold=args.pmax if hasattr(args, 'pmax') and args.pmax is not None else 0.05
            )
        except Exception as e:
            logger.warning(f"OOS diagnostics failed: {e}")

        # Use unified driver selection and optimization
        s_fold, selection_info = select_and_optimize_drivers(
            s_tr, s_te, y_tr, args, driver_selection_obj, weight_optimization_obj, seed=1234+f
        )
        
        if s_fold is None:
            continue
            
        oos_signal.iloc[te] = s_fold
        fold_summaries.append({"fold": f, **selection_info})

    if len(fold_summaries) == 0 or oos_signal.abs().sum() < 1e-10:
        logger.error("No valid signal generated")
        return None

    # Final OOS metrics
    dapy_val = driver_selection_obj(oos_signal, y)
    ir = information_ratio(oos_signal, y)
    hr = hit_rate(oos_signal, y)
    logger.info(f"OOS DAPY({args.driver_selection}): {dapy_val:.2f} | OOS IR: {ir:.2f} | OOS hit-rate: {hr:.3f}")
    save_oos_artifacts(oos_signal, y, block=args.block, final_shuffles=args.final_shuffles, dapy_fn=driver_selection_obj)
    
    performance_metrics = calculate_returns_metrics(oos_signal, y, freq=252)
    performance_report = format_performance_report(performance_metrics, "OUT-OF-SAMPLE PERFORMANCE")
    print(performance_report)
    
    symbol_results = {'ENSEMBLE': performance_metrics}
    save_performance_csv(performance_metrics, symbol_results, "artifacts/performance_summary.csv")

    with open("artifacts/fold_summaries.json", "w", encoding="utf-8") as fsum:
        json.dump(fold_summaries, fsum, indent=2)
    
    # Save diagnostics if available
    try:
        from diagnostics.diagnostic_output import save_comprehensive_diagnostics
        feature_info = []
        for i, col in enumerate(X.columns):
            feature_info.append({
                "name": col, "index": i,
                "min": float(X[col].min()), "max": float(X[col].max()),
                "std": float(X[col].std()), "mean": float(X[col].mean())
            })
        
        model_performance = {
            "total_return": performance_metrics.get('total_return', 0),
            "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0),
            "hit_rate": performance_metrics.get('hit_rate', 0),
            "max_drawdown": performance_metrics.get('max_drawdown', 0)
        }
        
        # Pass ensemble information to diagnostics
        ensemble_info = {
            "fold_summaries": fold_summaries,
            "total_folds": len(fold_summaries),
            "dapy_value": dapy_val,
            "information_ratio": ir,
            "hit_rate": hr,
            "feature_selection": {
                "original_features": original_feature_count,
                "features_after_selection": features_after_selection,
                "reduction_count": original_feature_count - features_after_selection,
                "reduction_percentage": round((original_feature_count - features_after_selection) / original_feature_count * 100, 1) if original_feature_count > 0 else 0
            }
        }
        
        save_comprehensive_diagnostics(vars(args), X.shape, feature_info, model_performance, ensemble_info)
    except Exception:
        pass

    # COMMENTED OUT: Misleading production model that doesn't change validation methodology
    # if args.train_production:
    #     print("\n[Production] Training production model on all data (for deployment only, not for backtest)...")
    #     train_production_model(X, y, args)

    # Random past-train then test from test_start
    if args.random_train_pct > 0.0:
        print(f"\n[RandTrain] Training on {args.random_train_pct*100:.1f}% of pre-{args.test_start} data, then backtesting from {args.test_start}...")
        random_past_train_then_backtest(X, y, args)

    # Return success
    return True


def train_production_model(X: pd.DataFrame, y: pd.Series, args):
    import numpy as np
    dapy_fn = get_dapy_fn(args.driver_selection)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict, stratified_xgb_bank, fold_train_predict_tiered, generate_deep_xgb_specs
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    # Generate XGB specs and train models
    specs, col_slices = setup_xgb_specs(X.columns, args)
    train_preds, test_preds = train_fold_models(X, y, X, specs, col_slices, args)  # predict on full dataset
    s_tr, s_full = build_driver_signals(train_preds, test_preds, y, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        if args.bypass_pvalue_gating:
            return True
        pval, _, _ = shuffle_pvalue(sig, yy, driver_selection_obj, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx, selection_diagnostics = pick_top_n_greedy_diverse(s_tr, y, n=args.n_select, pval_gate=gate, objective_fn=driver_selection_obj, diversity_penalty=args.diversity_penalty, objective_name=args.driver_selection)
    if len(chosen_idx) == 0:
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    full_sel  = [s_full[i] for i in chosen_idx]

    if args.equal_weights:
        w = np.ones(len(train_sel))
        tau = 1.0
        ww = softmax(w, temperature=tau)
        J_star = 0.0
    else:
        bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
        grope_obj = get_grope_objective_function(getattr(args, 'grope_objective', None))
        fobj = weight_objective_factory(train_sel, y, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn, objective_fn=grope_obj)
        theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=5678)

        w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
        tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)
    s_full_ens = combine_signals(full_sel, ww)

    save_timeseries("artifacts/production_timeseries.csv", s_full_ens, y)

    prod = {
        "chosen_driver_indices": [int(i) for i in chosen_idx],
        "weights_softmax": ww.tolist(), "temperature": float(tau),
        "n_models_total": args.n_models, "n_select": args.n_select,
        "z_win": args.z_win, "beta_pre": args.beta_pre, "lambda_to": args.lambda_to,
        "w_dapy": args.w_dapy, "w_ir": args.w_ir, "pmax": args.pmax, "weight_budget": args.weight_budget,
        "diversity_penalty": args.diversity_penalty, "driver_selection": args.driver_selection
    }
    with open("artifacts/production_model.json", "w", encoding="utf-8") as f:
        json.dump(prod, f, indent=2)

def random_past_train_then_backtest(X: pd.DataFrame, y: pd.Series, args):
    dapy_fn = get_dapy_fn(args.driver_selection)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict, stratified_xgb_bank, fold_train_predict_tiered, generate_deep_xgb_specs
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    import numpy as np, pandas as pd
    os.makedirs("artifacts", exist_ok=True)

    ts = pd.Timestamp(args.test_start)
    mask_train_time = X.index < ts
    if mask_train_time.sum() < 50:
        return
    
    rng = np.random.default_rng(args.randtrain_seed)
    idx_pre = np.where(mask_train_time)[0]
    n_sample = max(10, int(len(idx_pre) * args.random_train_pct))
    idx_train = np.sort(rng.choice(idx_pre, size=n_sample, replace=False))

    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    X_te = X[X.index >= ts]
    
    # Generate XGB specs and train models
    specs, col_slices = setup_xgb_specs(X_tr.columns, args)
    train_preds, test_preds = train_fold_models(X_tr, y_tr, X_te, specs, col_slices, args)
    s_tr, s_te = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        if args.bypass_pvalue_gating:
            return True
        pval, _, _ = shuffle_pvalue(sig, yy, driver_selection_obj, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx, selection_diagnostics = pick_top_n_greedy_diverse(s_tr, y_tr, n=args.n_select, pval_gate=gate, objective_fn=driver_selection_obj, diversity_penalty=args.diversity_penalty, objective_name=args.driver_selection)
    if len(chosen_idx) == 0:
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    test_sel  = [s_te[i] for i in chosen_idx]

    if args.equal_weights:
        w = np.ones(len(train_sel))
        tau = 1.0
        ww = softmax(w, temperature=tau)
    else:
        bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
        grope_obj = get_grope_objective_function(getattr(args, 'grope_objective', None))
        fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn, objective_fn=grope_obj)
        theta_star, _, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=2468)

        w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
        tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)

    s_test_ens = combine_signals(test_sel, ww)
    y_test = y[y.index >= ts]
    save_timeseries("artifacts/randomtrain_timeseries.csv", s_test_ens, y_test)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"📄 Loaded configuration from {config_path}")
    return config

def merge_args_with_config(args, config: dict, cli_args=None):
    """Merge command line arguments with config file, CLI takes precedence."""
    # Get list of arguments that were explicitly provided via CLI
    explicitly_set = set()
    if cli_args:
        # Parse the actual CLI arguments to see what was explicitly provided
        import sys
        for i, arg in enumerate(sys.argv[1:]):
            if arg.startswith('--'):
                param_name = arg[2:].replace('-', '_')
                explicitly_set.add(param_name)
    
    for key, value in config.items():
        # Handle boolean flags specially
        if key == 'on_target_only' and isinstance(value, bool):
            if hasattr(args, key):
                # If not explicitly set in CLI, use config value
                if not getattr(args, key, False):
                    setattr(args, key, value)
            else:
                setattr(args, key, value)
        elif hasattr(args, key):
            current_value = getattr(args, key)
            # Only override if CLI value wasn't explicitly provided
            if key not in explicitly_set and current_value is None:
                setattr(args, key, value)
            elif key not in explicitly_set:
                # For parameters with defaults, also allow config override
                setattr(args, key, value)
        else:
            setattr(args, key, value)
    return args

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="XGB ensemble with GROPE optimization - supports config files and CLI overrides")
    
    # Configuration file option
    ap.add_argument("--config", type=str, help="Path to YAML configuration file (e.g., configs/fast_test.yaml)")
    
    # Data source options
    ap.add_argument("--target_symbol", type=str, help="Target symbol(s) for prediction - comma-separated for multiple")
    ap.add_argument("--symbols", type=str, help="Comma-separated list of symbols (default: from symbols.yaml)")
    
    # CHANGED: Added real data parameters
    ap.add_argument("--start_date", type=str, help="Start date for data loading (YYYY-MM-DD)")
    ap.add_argument("--end_date", type=str, help="End date for data loading (YYYY-MM-DD)")
    ap.add_argument("--signal_hour", type=int, help="Hour for signal generation")
    ap.add_argument("--n_hours", type=int, help="Lookahead hours for target return")
    ap.add_argument("--max_features", type=int, help="Limit features for testing")
    
    # Feature selection options
    ap.add_argument("--on_target_only", action="store_true", help="Use only target symbol features")
    ap.add_argument("--corr_threshold", type=float, default=0.7, help="Correlation threshold for feature clustering")
    
    # Model parameters
    ap.add_argument("--folds", type=int, default=6, help="Walk-forward CV folds")
    ap.add_argument("--n_models", type=int, default=50, help="Number of XGB models")
    ap.add_argument("--n_select", type=int, default=12, help="Number of drivers to select")
    # Removed multiprocessing option for original alignment
    
    # Signal processing
    ap.add_argument("--z_win", type=int, default=100, help="Z-score window")
    ap.add_argument("--beta_pre", type=float, default=1.0, help="Tanh squash beta")
    ap.add_argument("--lambda_to", type=float, default=0.05, help="Turnover penalty")
    
    # Optimization parameters
    ap.add_argument("--weight_budget", type=int, default=80, help="GROPE optimization budget")
    ap.add_argument("--w_dapy", type=float, default=1.0, help="DAPY weight")
    ap.add_argument("--w_ir", type=float, default=1.0, help="Information ratio weight")
    ap.add_argument("--diversity_penalty", type=float, default=0.2, help="Diversity penalty")
    ap.add_argument("--pmax", type=float, default=0.8, help="P-value threshold")
    ap.add_argument("--final_shuffles", type=int, default=200, help="Final shuffle tests")
    ap.add_argument("--block", type=int, default=30, help="Block size for permutation")
    
    # Output options
    # COMMENTED OUT: Misleading flag that doesn't change validation methodology
    # ap.add_argument("--train_production", action="store_true", help="Train production model")
    ap.add_argument("--train_test_split", action="store_true", help="Use single train-test split instead of 6-fold cross-validation")
    ap.add_argument("--tiered_xgb", action="store_true", help="Use tiered XGBoost architecture (Tier A/B/C with different complexity)")
    ap.add_argument("--deep_xgb", action="store_true", help="Use deeper XGBoost trees (8-10 depth vs baseline 2-6)")
    ap.add_argument("--driver_selection", type=str, default="hits", help="Driver selection objective function: 'hits', 'eri_both', 'adjusted_sharpe', 'cb_ratio', 'information_ratio', 'predictive_icir_logscore'")
    ap.add_argument("--bypass_pvalue_gating", action="store_true", help="[DEPRECATED] Use --p_value_gating instead") 
    ap.add_argument("--p_value_gating", type=str, help="P-value gating method: 'dapy', 'predictive_icir_logscore', 'adjusted_sharpe', 'information_ratio', 'cb_ratio', or null to disable")
    ap.add_argument("--grope_objective", type=str, help="GROPE weight optimization objective: 'hits', 'eri_both', 'adjusted_sharpe', 'cb_ratio', 'information_ratio', 'predictive_icir_logscore', or null for default DAPY+IR")
    ap.add_argument("--equal_weights", action="store_true", help="Use equal weights instead of GROPE optimization for ensemble combination")
    
    # Random training options
    ap.add_argument("--test_start", type=str, help="Test start date for random training")
    ap.add_argument("--random_train_pct", type=float, help="Random training percentage")
    ap.add_argument("--randtrain_seed", type=int, help="Random training seed")
    
    
    args = ap.parse_args()
    
    # Load configuration file if provided
    config = {}
    if args.config:
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
        config = load_config(args.config)
        args = merge_args_with_config(args, config, cli_args=True)
    
    
    main(args)




        
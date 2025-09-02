
import argparse, numpy as np, pandas as pd, os, json
# CHANGED: Added logging and real data support
import logging
import yaml
from cv.wfo import wfo_splits
from model.xgb_drivers import generate_xgb_specs, fold_train_predict
from model.feature_selection import apply_feature_selection
from ensemble.combiner import build_driver_signals, combine_signals, softmax
from ensemble.selection import pick_top_n_greedy_diverse
from opt.grope import grope_optimize
from opt.weight_objective import weight_objective_factory
from metrics.dapy import hit_rate
from metrics.perf import information_ratio
from eval.target_shuffling import shuffle_pvalue
# CHANGED: Added real data loading support
from data.data_utils import prepare_real_data
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
        from metrics.dapy_eri import dapy_eri_long as fn
        return fn
    elif style == "eri_short":
        from metrics.dapy_eri import dapy_eri_short as fn
        return fn
    elif style == "eri_both":
        from metrics.dapy_eri import dapy_eri_both as fn
        return fn
    else:
        from metrics.dapy import dapy_from_binary_hits as fn
        return fn

def save_timeseries(path: str, signal: pd.Series, y: pd.Series):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pnl = (signal.shift(1).fillna(0.0) * y.reindex_like(signal)).astype(float)
    eq = pnl.cumsum()
    out = pd.DataFrame({'signal': signal, 'target_ret': y.reindex_like(signal), 'pnl': pnl, 'equity': eq})
    out.to_csv(path)
    return out

def save_oos_artifacts(signal: pd.Series, y: pd.Series, block: int, final_shuffles: int, dapy_fn):
    os.makedirs("artifacts", exist_ok=True)
    out = save_timeseries("artifacts/oos_timeseries.csv", signal, y)
    pval, obs, _ = shuffle_pvalue(signal, y, dapy_fn, n_shuffles=final_shuffles, block=block)
    print(f"OOS Shuffling p-value (DAPY): {pval:.10f} (obs={obs:.2f})")
    return out

def main(args):
    """Main dispatcher - handles single or multiple targets."""
    dapy_fn = get_dapy_fn(args.dapy_style)
    
    # Parse target symbols
    target_symbols = [s.strip() for s in args.target_symbol.split(',')]
    logger.info(f"🚀 Starting XGB Ensemble Analysis")
    logger.info(f"🎯 Target symbols: {target_symbols}")
    
    # Load symbols from config
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = get_default_symbols()
        logger.info(f"📋 Using default symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")
    
    # Process each target
    results = {}
    for i, target_symbol in enumerate(target_symbols, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"🔄 PROCESSING TARGET {i}/{len(target_symbols)}: {target_symbol}")
        logger.info(f"{'='*80}")
        
        try:
            result = run_single_target(target_symbol, args, symbols, dapy_fn)
            results[target_symbol] = result
            if result:
                logger.info(f"✅ Completed {target_symbol}")
            else:
                logger.warning(f"⚠️ Failed {target_symbol}")
        except Exception as e:
            logger.error(f"❌ Error processing {target_symbol}: {e}")
            results[target_symbol] = None
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None)
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 FINAL SUMMARY: {successful}/{len(target_symbols)} targets completed successfully")
    logger.info(f"{'='*80}")
    
    return results

def run_single_target(target_symbol: str, args, symbols: list, dapy_fn):
    """Run analysis for a single target symbol."""
    # Set the target in args temporarily for compatibility 
    original_target = args.target_symbol
    args.target_symbol = target_symbol
    
    try:
        # CHANGED: Removed synthetic data fallback - only real market data supported
        logger.info(f"🔄 Loading real market data for target: {target_symbol}")
        
        # Prepare real data with feature engineering
        df = prepare_real_data(
            target_symbol=target_symbol,
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            n_hours=args.n_hours,
            signal_hour=args.signal_hour,
            max_features=args.max_features,
            on_target_only=args.on_target_only,
            corr_threshold=args.corr_threshold
        )
        
        if df.empty:
            logger.error(f"❌ No data loaded for target symbol: {target_symbol}")
            return None
        
        # Split into features and target
        target_col = f"{target_symbol}_target_return"
        if target_col not in df.columns:
            logger.error(f"❌ Target column {target_col} not found in data")
            return None
        
        y = df[target_col]
        X = df.drop(columns=[target_col])
        
        logger.info(f"✅ Loaded data: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"📅 Date range: {X.index.min()} to {X.index.max()}")
        
        # Continue with existing logic...
        return run_training_pipeline(X, y, args, dapy_fn)
    
    finally:
        # Restore original target
        args.target_symbol = original_target

def run_training_pipeline(X, y, args, dapy_fn):
    """Run the XGB training pipeline."""
    
    logger.info(f"Loaded real data: X shape {X.shape}, y shape {y.shape}")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
    
    # Ensure we have enough data
    if len(X) < args.folds * 100:
        logger.warning(f"Limited data: {len(X)} observations for {args.folds} folds")
        args.folds = max(2, len(X) // 100)  # Adjust folds for limited data
        logger.info(f"Reduced to {args.folds} folds")

    # Apply block-wise feature selection BEFORE training
    # For large datasets, reduce features more aggressively to prevent XGB overfitting
    if X.shape[1] > 200:  # Only apply if we have many features
        # Reduce features much more aggressively to prevent XGB overfitting
        # For financial data, use ~20-50 features max to ensure sufficient observations per feature
        target_features = max(30, min(50, len(X) // 20))  # 20:1 observation-to-feature ratio
        logger.info(f"Applying smart block-wise feature selection: {X.shape[1]} -> {target_features} features")
        X = apply_feature_selection(X, y, method='block_wise', 
                                   block_size=100, features_per_block=15, 
                                   max_total_features=target_features, corr_threshold=args.corr_threshold)
        logger.info(f"Feature selection complete: {X.shape}")
    
    splits = wfo_splits(len(X), k_folds=args.folds, min_train=max(252, len(X)//(args.folds*2)))
    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    os.makedirs("artifacts", exist_ok=True)

    oos_signal = pd.Series(0.0, index=X.index)
    fold_summaries = []

    for f, (tr, te) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te = X.iloc[te]  # Removed unused y_te

        logger.info(f"[Fold {f}] Training data: X_tr{X_tr.shape}, y_tr{y_tr.shape}")
        logger.info(f"[Fold {f}] Target stats: mean={y_tr.mean():.6f}, std={y_tr.std():.6f}")
        logger.info(f"[Fold {f}] Feature stats: {X_tr.select_dtypes(include=[np.number]).mean().abs().sum():.6f} total magnitude")
        
        # Check for degenerate cases that could cause Fold 0 to fail
        if y_tr.std() < 1e-10:
            logger.error(f"[Fold {f}] Target has zero variance - skipping fold")
            continue
        
        if X_tr.select_dtypes(include=[np.number]).std().sum() < 1e-6:
            logger.error(f"[Fold {f}] Features have near-zero variance - skipping fold") 
            continue
        
        train_preds, test_preds = fold_train_predict(X_tr, y_tr, X_te, specs)
        
        # Debug: Check XGBoost predictions before signal generation
        logger.info(f"Fold {f} XGBoost predictions: train_preds count={len(train_preds)}, test_preds count={len(test_preds)}")
        if test_preds and len(test_preds) > 0:
            pred_stats = []
            for i, pred in enumerate(test_preds[:3]):
                if pred is not None:
                    pred_stats.append(f"model_{i}: mean={pred.mean():.6f}, std={pred.std():.6f}, range=[{pred.min():.6f}, {pred.max():.6f}]")
            logger.info(f"Fold {f} test prediction stats: {'; '.join(pred_stats)}")
        
        s_tr, s_te = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

        def gate(sig, y_local):
            if args.bypass_pvalue_gating:
                return True
            pval, _, _ = shuffle_pvalue(sig, y_local, dapy_fn, n_shuffles=200, block=args.block)
            return pval <= args.pmax

        chosen_idx = pick_top_n_greedy_diverse(
            s_tr, y_tr, n=args.n_select, pval_gate=gate,
            w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn
        )
        if len(chosen_idx) == 0:
            logger.warning(f"[Fold {f}] No drivers passed the p-value gate (threshold={args.pmax:.3f}); ensemble is zero.")
            continue
        train_sel = [s_tr[i] for i in chosen_idx]
        test_sel  = [s_te[i] for i in chosen_idx]

        bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
        fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
        theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=1234+f)

        w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
        tau = float(theta_star["tau"])
        ww = softmax(w, temperature=tau)
        
        # Show GROPE results for this fold
        logger.info(f"Fold {f}: Selected {len(chosen_idx)} models, weights: {[f'{w:.3f}' for w in ww]}, tau: {tau:.3f}")
        
        # Debug: Check individual test signals before combining
        logger.info(f"Fold {f} individual test signals: count={len(test_sel)}, lengths={[len(s) for s in test_sel[:3]]}, magnitudes={[s.abs().sum() for s in test_sel[:3]]}")
        
        s_fold = combine_signals(test_sel, ww)
        oos_signal.iloc[te] = s_fold
        
        # Debug: Show fold signal statistics
        logger.info(f"Fold {f} signal stats: mean={s_fold.mean():.6f}, std={s_fold.std():.6f}, sum={s_fold.sum():.6f}, magnitude={s_fold.abs().sum():.6f}")

        fold_summaries.append({"fold": f, "chosen_idx": chosen_idx, "weights": ww.tolist(), "tau": tau, "J_train": J_star})

    # Check if we have any valid signal
    logger.info(f"📋 Processing complete: {len(fold_summaries)} fold summaries created")
    logger.info(f"📊 OOS signal magnitude: {oos_signal.abs().sum():.10f}")
    if len(fold_summaries) == 0:
        logger.error("❌ No fold summaries created. All folds failed p-value gating.")
        return None
    elif oos_signal.abs().sum() < 1e-10:
        logger.error("❌ OOS signal has zero magnitude.")
        return None

    # Final OOS metrics - ORIGINAL LOGIC PRESERVED
    dapy_val = dapy_fn(oos_signal, y)
    ir = information_ratio(oos_signal, y)
    hr = hit_rate(oos_signal, y)
    logger.info(f"OOS DAPY({args.dapy_style}): {dapy_val:.2f} | OOS IR: {ir:.2f} | OOS hit-rate: {hr:.3f}")
    save_oos_artifacts(oos_signal, y, block=args.block, final_shuffles=args.final_shuffles, dapy_fn=dapy_fn)
    
    # ENHANCED: Comprehensive performance analysis
    logger.info("Calculating comprehensive performance metrics...")
    
    # Calculate enhanced performance metrics
    performance_metrics = calculate_returns_metrics(oos_signal, y, freq=252)
    
    # Display comprehensive performance report
    performance_report = format_performance_report(performance_metrics, "OUT-OF-SAMPLE PERFORMANCE")
    print(performance_report)
    
    # Individual symbol analysis (if we have fold summaries with individual signals)
    # For now, just show the ensemble performance
    symbol_results = {'ENSEMBLE': performance_metrics}
    
    # Save performance summary
    save_performance_csv(performance_metrics, symbol_results, "artifacts/performance_summary.csv")

    with open("artifacts/fold_summaries.json", "w", encoding="utf-8") as fsum:
        json.dump(fold_summaries, fsum, indent=2)
    
    # ENHANCED: Save comprehensive diagnostics
    try:
        from diagnostics.diagnostic_output import save_comprehensive_diagnostics
        
        # Collect feature information from the loaded data
        feature_info = []
        if 'X' in locals() and X is not None:
            for i, col in enumerate(X.columns):
                feature_info.append({
                    "name": col,
                    "index": i,
                    "min": float(X[col].min()) if not X[col].empty else 0,
                    "max": float(X[col].max()) if not X[col].empty else 0,
                    "std": float(X[col].std()) if not X[col].empty else 0,
                    "mean": float(X[col].mean()) if not X[col].empty else 0
                })
        
        # Collect model performance info from fold summaries
        model_performance = {
            "total_models": sum([len(fs.get('models', [])) for fs in fold_summaries]),
            "pvalue_range": f"{min([fs.get('min_pvalue', 1.0) for fs in fold_summaries]):.4f} - {max([fs.get('max_pvalue', 1.0) for fs in fold_summaries]):.4f}",
            "constant_models": sum([fs.get('constant_predictions', 0) for fs in fold_summaries]),
            "variable_models": sum([fs.get('variable_predictions', 0) for fs in fold_summaries]),
            "total_return": performance_metrics.get('total_return', 0),
            "sharpe_ratio": performance_metrics.get('sharpe_ratio', 0),
            "hit_rate": performance_metrics.get('hit_rate', 0),
            "max_drawdown": performance_metrics.get('max_drawdown', 0)
        }
        
        data_shape = X.shape if 'X' in locals() else (0, 0)
        save_comprehensive_diagnostics(vars(args), data_shape, feature_info, model_performance)
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to save comprehensive diagnostics: {e}")

    # Production model (unrealistic backtest)
    if args.train_production:
        print("\n[Production] Training production model on all data (for deployment only, not for backtest)...")
        train_production_model(X, y, args)

    # Random past-train then test from test_start
    if args.random_train_pct > 0.0:
        print(f"\n[RandTrain] Training on {args.random_train_pct*100:.1f}% of pre-{args.test_start} data, then backtesting from {args.test_start}...")
        random_past_train_then_backtest(X, y, args)

    # Return success
    return True


def train_production_model(X: pd.DataFrame, y: pd.Series, args):
    import numpy as np
    dapy_fn = get_dapy_fn(args.dapy_style)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    train_preds, test_preds = fold_train_predict(X, y, X, specs)  # predict on full
    s_tr, s_full = build_driver_signals(train_preds, test_preds, y, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        if args.bypass_pvalue_gating:
            return True
        pval, _, _ = shuffle_pvalue(sig, yy, dapy_fn, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx = pick_top_n_greedy_diverse(s_tr, y, n=args.n_select, pval_gate=gate, w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn)
    if len(chosen_idx) == 0:
        print("[Production] No drivers passed p-value gate.")
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    full_sel  = [s_full[i] for i in chosen_idx]

    bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
    fobj = weight_objective_factory(train_sel, y, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
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
        "diversity_penalty": args.diversity_penalty, "dapy_style": args.dapy_style
    }
    with open("artifacts/production_model.json", "w", encoding="utf-8") as f:
        json.dump(prod, f, indent=2)
    print("[Production] Saved artifacts/production_model.json")

def random_past_train_then_backtest(X: pd.DataFrame, y: pd.Series, args):
    dapy_fn = get_dapy_fn(args.dapy_style)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict
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
        print("[RandTrain] Not enough pre-test data.")
        return
    rng = np.random.default_rng(args.randtrain_seed)
    idx_pre = np.where(mask_train_time)[0]
    n_sample = max(10, int(len(idx_pre) * args.random_train_pct))
    idx_train = np.sort(rng.choice(idx_pre, size=n_sample, replace=False))

    # CHANGED: Fixed bug - use proper test set (post test_start) instead of full dataset
    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    X_te = X[X.index >= ts]  # Test data only (post test_start)
    
    # CHANGED: Added data validation to prevent NaN/infinity errors
    if X_tr.isnull().any().any() or y_tr.isnull().any():
        print(f"[RandTrain] Warning: NaN values in training data. X_tr NaN: {X_tr.isnull().sum().sum()}, y_tr NaN: {y_tr.isnull().sum()}")
        # Drop NaN rows
        mask_valid = ~(X_tr.isnull().any(axis=1) | y_tr.isnull())
        X_tr, y_tr = X_tr[mask_valid], y_tr[mask_valid]
        print(f"[RandTrain] After cleaning: {len(X_tr)} valid samples")
    
    if (y_tr.abs() > 1e6).any() or not np.isfinite(y_tr).all():
        print(f"[RandTrain] Warning: Extreme/infinite values in y_tr. Range: {y_tr.min():.6f} to {y_tr.max():.6f}")
        # Clip extreme values
        y_tr = y_tr.clip(-10, 10)
    
    if len(X_tr) < 10:
        print(f"[RandTrain] Not enough valid training samples: {len(X_tr)}")
        return
    
    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    train_preds, test_preds = fold_train_predict(X_tr, y_tr, X_te, specs)
    s_tr, s_te = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        if args.bypass_pvalue_gating:
            return True
        pval, _, _ = shuffle_pvalue(sig, yy, dapy_fn, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx = pick_top_n_greedy_diverse(s_tr, y_tr, n=args.n_select, pval_gate=gate, w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn)
    if len(chosen_idx) == 0:
        print("[RandTrain] No drivers passed p-value gate.")
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    test_sel  = [s_te[i] for i in chosen_idx]

    bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
    fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
    theta_star, _, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=2468)

    w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
    tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)

    s_test_ens = combine_signals(test_sel, ww)
    y_test = y[y.index >= ts]
    save_timeseries("artifacts/randomtrain_timeseries.csv", s_test_ens, y_test)
    print("[RandTrain] Saved artifacts/randomtrain_timeseries.csv and metadata.")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"📄 Loaded configuration from {config_path}")
    return config

def merge_args_with_config(args, config: dict):
    """Merge command line arguments with config file, CLI takes precedence."""
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
            # Only override if CLI value is None/default
            if current_value is None:
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
    ap.add_argument("--corr_threshold", type=float, help="Correlation threshold for feature clustering")
    
    # Model parameters
    ap.add_argument("--folds", type=int, help="Walk-forward CV folds")
    ap.add_argument("--n_models", type=int, help="Number of XGB models")
    ap.add_argument("--n_select", type=int, help="Number of drivers to select")
    # Removed multiprocessing option for original alignment
    
    # Signal processing
    ap.add_argument("--z_win", type=int, help="Z-score window")
    ap.add_argument("--beta_pre", type=float, help="Tanh squash beta")
    ap.add_argument("--lambda_to", type=float, help="Turnover penalty")
    
    # Optimization parameters
    ap.add_argument("--weight_budget", type=int, help="GROPE optimization budget")
    ap.add_argument("--w_dapy", type=float, help="DAPY weight")
    ap.add_argument("--w_ir", type=float, help="Information ratio weight")
    ap.add_argument("--diversity_penalty", type=float, help="Diversity penalty")
    ap.add_argument("--pmax", type=float, help="P-value threshold")
    ap.add_argument("--final_shuffles", type=int, help="Final shuffle tests")
    ap.add_argument("--block", type=int, help="Block size for permutation")
    
    # Output options
    ap.add_argument("--train_production", action="store_true", help="Train production model")
    ap.add_argument("--dapy_style", type=str, help="DAPY style")
    ap.add_argument("--bypass_pvalue_gating", action="store_true", help="Bypass p-value gating for testing purposes")
    
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
        args = merge_args_with_config(args, config)
    
    # Log configuration source
    if args.config:
        logger.info(f"📄 Using configuration: {args.config}")
    else:
        logger.info("🔧 Using CLI arguments and defaults")
    
    main(args)

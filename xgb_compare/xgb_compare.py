#!/usr/bin/env python3
"""
XGBoost Comparison Framework - Production Ready
Complete model comparison with Q-score tracking and production backtesting.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import multiprocessing as mp

# Optimize NumExpr threads
os.environ['NUMEXPR_MAX_THREADS'] = str(min(16, mp.cpu_count()))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parse_config
from metrics_utils import QualityTracker, calculate_model_metrics, normalize_predictions, calculate_metric_pvalue
from visualization_clean import create_clean_visualizations
from full_timeline_backtest import FullTimelineBacktester
from backtest_visualization import log_detailed_backtest_summary
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice
from cv.wfo import wfo_splits, wfo_splits_rolling

def export_production_models(all_fold_results, xgb_specs, selected_features, backtest_results, config, logger, col_slices=None):
    """Export ONLY the selected models from final fold as single consolidated file per symbol."""
    import pickle
    from pathlib import Path

    # Create production directory
    prod_dir = Path(__file__).parent.parent / "PROD"
    models_dir = prod_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("EXPORTING PRODUCTION MODELS (Consolidated File)")
    logger.info("="*60)

    try:
        # Debug backtest results structure
        logger.info(f"Backtest results keys: {list(backtest_results.keys())}")

        # Get the final fold's selected model indices from backtest results
        model_selection_history = backtest_results.get('model_selection_history', [])
        logger.info(f"Model selection history length: {len(model_selection_history)}")

        if not model_selection_history:
            logger.error("No model selection history found in backtest results")
            return False

        final_selection = model_selection_history[-1]  # Last fold selection
        logger.info(f"Final selection keys: {list(final_selection.keys())}")
        selected_model_indices = final_selection.get('selected_models', [])

        logger.info(f"Final fold selected models: {selected_model_indices}")

        if not selected_model_indices:
            logger.error("No selected models found in final fold")
            return False

        # Debug all_fold_results structure
        logger.info(f"All fold results type: {type(all_fold_results)}")
        logger.info(f"All fold results keys/length: {list(all_fold_results.keys()) if isinstance(all_fold_results, dict) else len(all_fold_results)}")

        # Get the final fold (handle both list and dict cases)
        if isinstance(all_fold_results, dict):
            # If dict, get the highest fold number
            final_fold_idx = max(all_fold_results.keys())
            final_fold = all_fold_results[final_fold_idx]
            logger.info(f"Using fold {final_fold_idx} as final fold")
        else:
            # If list, get last item
            final_fold = all_fold_results[-1]
            logger.info(f"Using last fold from list")

        logger.info(f"Final fold keys: {list(final_fold.keys())}")

        all_trained_models = final_fold.get('trained_models', {})
        logger.info(f"Trained models available: {list(all_trained_models.keys())}")

        if not all_trained_models:
            logger.error("No trained models found in final fold")
            return False

        # Extract ONLY the selected models with their specific feature slices
        selected_models = {}
        model_feature_slices = {}
        for model_idx in selected_model_indices:
            if model_idx in all_trained_models:
                model_key = f"model_{len(selected_models)+1:02d}"
                selected_models[model_key] = all_trained_models[model_idx]

                # Store actual feature usage (all features, matching current framework behavior)
                model_feature_slices[model_key] = selected_features
                logger.info(f"Added model M{model_idx:02d} -> {model_key} with all {len(selected_features)} features (framework standard)")

        logger.info(f"Extracted {len(selected_models)} selected models")

        if not selected_models:
            logger.error("No selected models could be extracted")
            return False

        # Create consolidated production package
        production_package = {
            'symbol': config.target_symbol,
            'models': selected_models,
            'model_feature_slices': model_feature_slices,  # Store model-specific feature slices
            'selected_features': selected_features,
            'selected_model_indices': selected_model_indices,
            'binary_signal': config.binary_signal,
            'metadata': {
                'n_models': config.n_models,
                'n_folds': config.n_folds,
                'max_features': config.max_features,
                'q_metric': config.q_metric,
                'xgb_type': config.xgb_type,
                'export_timestamp': datetime.now().isoformat(),
                'log_label': config.log_label
            }
        }

        # Save consolidated file
        production_file = models_dir / f"{config.target_symbol}_production.pkl"
        with open(production_file, 'wb') as f:
            pickle.dump(production_package, f)

        logger.info(f"SUCCESS: Exported {len(selected_models)} selected models to {production_file}")
        logger.info(f"File size: {production_file.stat().st_size:,} bytes")
        logger.info(f"Selected features: {len(selected_features)} features")
        logger.info(f"Selected models: {selected_model_indices}")

        return True

    except Exception as e:
        logger.error(f"Production export failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def export_signal_distribution(backtest_results, config, logger):
    """Export daily signals, returns, and PnL distribution to CSV."""
    import pandas as pd
    from pathlib import Path

    try:
        logger.info("="*60)
        logger.info("EXPORTING SIGNAL DISTRIBUTION")
        logger.info("="*60)

        # Get backtest data
        full_timeline_returns = backtest_results.get('full_timeline_returns', [])
        full_timeline_predictions = backtest_results.get('full_timeline_predictions', [])
        full_timeline_dates = backtest_results.get('full_timeline_dates', [])

        if not full_timeline_returns or not full_timeline_predictions:
            logger.error("No timeline data found for signal distribution export")
            return False

        logger.info(f"Timeline data: {len(full_timeline_returns)} points")

        # Create signal distribution dataframe
        signal_data = []
        for i, (date, pred, ret) in enumerate(zip(full_timeline_dates, full_timeline_predictions, full_timeline_returns)):
            # Calculate PnL using the continuous prediction value (as requested)
            pnl = pred * ret

            # Determine signal direction for classification
            signal = 1 if pred > 0 else (-1 if pred < 0 else 0)

            signal_data.append({
                'date': date,
                'signal_direction': signal,  # Binary classification for analysis
                'signal': pred,  # Actual continuous tanh signal used for PnL
                'prediction': pred,  # Keep for backward compatibility
                'target_return': ret,
                'pnl': pnl,
                'cumulative_pnl': 0,  # Will calculate below
                'period': 'training' if i < backtest_results.get('cutoff_fold', 0) * len(full_timeline_returns) // len(backtest_results.get('fold_results', [])) else 'production'
            })

        # Convert to DataFrame
        signal_df = pd.DataFrame(signal_data)

        # Calculate cumulative PnL
        signal_df['cumulative_pnl'] = signal_df['pnl'].cumsum()

        # Add additional metrics
        signal_df['abs_return'] = signal_df['target_return'].abs()
        signal_df['hit'] = ((signal_df['signal_direction'] > 0) & (signal_df['target_return'] > 0)) | ((signal_df['signal_direction'] < 0) & (signal_df['target_return'] < 0))

        # Export to CSV
        results_dir = Path(__file__).parent / 'results'
        csv_file = results_dir / f"{config.log_label}_signal_distribution.csv"

        signal_df.to_csv(csv_file, index=False)

        logger.info(f"SUCCESS: Exported signal distribution to {csv_file}")
        logger.info(f"Records: {len(signal_df):,}")
        logger.info(f"Date range: {signal_df['date'].min()} to {signal_df['date'].max()}")
        logger.info(f"Signal directions: {signal_df['signal_direction'].value_counts().to_dict()}")
        logger.info(f"Signal range: {signal_df['signal'].min():.6f} to {signal_df['signal'].max():.6f}")
        logger.info(f"Hit rate: {signal_df['hit'].mean():.3f}")
        logger.info(f"Total PnL: {signal_df['pnl'].sum():.6f}")

        return True

    except Exception as e:
        logger.error(f"Signal distribution export failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def setup_logging(config):
    """Initialize logging and result directories."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"{timestamp}_xgb_compare_{config.log_label}.log")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                       handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()])

    logger = logging.getLogger(__name__)
    config.log_config(logger)
    return logger, results_dir, timestamp

def load_and_prepare_data(config, logger):
    """Load financial data and apply feature selection."""
    df = prepare_real_data_simple(config.target_symbol, start_date=config.start_date, end_date=config.end_date)
    target_col = f"{config.target_symbol}_target_return"
    X, y = df[[c for c in df.columns if c != target_col]], df[target_col]
    logger.info(f"Loaded: X={X.shape}, y={y.shape}")

    selected_features = list(X.columns)  # Track original or selected features

    if not config.no_feature_selection:
        max_features = config.max_features if config.max_features > 0 else -1
        X = apply_feature_selection(X, y, method='block_wise', max_total_features=max_features)
        selected_features = list(X.columns)  # Update with selected features
        logger.info(f"Selected: {X.shape[1]} features")

    return X, y, selected_features

def train_single_model(model_idx, spec, X_train, y_train, X_inner_train, y_inner_train, 
                      X_inner_val, y_inner_val, X_test, y_test, config, use_gpu=False):
    """Train single XGBoost model and calculate IS/IV/OOS metrics."""
    # Train model
    model = fit_xgb_on_slice(X_train, y_train, spec, force_cpu=not use_gpu)
    
    # Generate normalized predictions  
    pred_inner_train = normalize_predictions(pd.Series(model.predict(X_inner_train.values), index=X_inner_train.index), config.binary_signal)
    pred_inner_val = normalize_predictions(pd.Series(model.predict(X_inner_val.values), index=X_inner_val.index), config.binary_signal)
    pred_test = normalize_predictions(pd.Series(model.predict(X_test.values), index=X_test.index), config.binary_signal)
    
    # Calculate performance metrics
    is_metrics = calculate_model_metrics(pred_inner_train, y_inner_train, shifted=False)
    iv_metrics = calculate_model_metrics(pred_inner_val, y_inner_val, shifted=False)
    oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=False)
    
    # Statistical significance testing
    p_sharpe = calculate_metric_pvalue(pred_test, y_test, 'sharpe', oos_metrics['sharpe'], config.n_bootstraps)
    
    result_dict = {
        'Model': f"M{model_idx:02d}",
        'IS_Sharpe': is_metrics['sharpe'],
        'IV_Sharpe': iv_metrics['sharpe'],
        'OOS_Sharpe': oos_metrics['sharpe'],
        'OOS_Hit_Rate': oos_metrics['hit_rate'],
        'OOS_Sharpe_p': p_sharpe,
        'Q_Sharpe': 0.0,  # Updated by quality tracker
        'OOS_Predictions': pred_test  # For backtesting
    }

    # Include trained model if export is requested
    if config.export_production_models:
        result_dict['trained_model'] = model

    return result_dict, oos_metrics

def _train_single_model_mp(args):
    """Train single model in multiprocessing worker."""
    model_idx, spec, X_train_vals, y_train_vals, X_inner_train_vals, y_inner_train_vals, \
    X_inner_val_vals, y_inner_val_vals, X_test_vals, y_test_vals, binary_signal, \
    train_idx, inner_train_idx, inner_val_idx, test_idx = args
    
    import pandas as pd
    from model.xgb_drivers import fit_xgb_on_slice
    from metrics_utils import normalize_predictions, calculate_model_metrics, calculate_metric_pvalue
    
    # Setup data with correct indices
    X_train_df = pd.DataFrame(X_train_vals, index=train_idx)
    y_train_series = pd.Series(y_train_vals, index=train_idx)
    
    # Train and predict
    model = fit_xgb_on_slice(X_train_df, y_train_series, spec, force_cpu=True)
    pred_inner_train = normalize_predictions(pd.Series(model.predict(X_inner_train_vals), index=inner_train_idx), binary_signal)
    pred_inner_val = normalize_predictions(pd.Series(model.predict(X_inner_val_vals), index=inner_val_idx), binary_signal)
    pred_test = normalize_predictions(pd.Series(model.predict(X_test_vals), index=test_idx), binary_signal)
    
    # Calculate all metrics
    is_metrics = calculate_model_metrics(pred_inner_train, pd.Series(y_inner_train_vals, index=inner_train_idx), shifted=False)
    iv_metrics = calculate_model_metrics(pred_inner_val, pd.Series(y_inner_val_vals, index=inner_val_idx), shifted=False)
    oos_metrics = calculate_model_metrics(pred_test, pd.Series(y_test_vals, index=test_idx), shifted=False)
    p_sharpe = calculate_metric_pvalue(pred_test, pd.Series(y_test_vals, index=test_idx), 'sharpe', oos_metrics['sharpe'], 50)
    
    return {
        'Model': f"M{model_idx:02d}", 'IS_Sharpe': is_metrics['sharpe'], 'IV_Sharpe': iv_metrics['sharpe'], 
        'OOS_Sharpe': oos_metrics['sharpe'], 'OOS_Hit_Rate': oos_metrics['hit_rate'], 
        'OOS_Sharpe_p': p_sharpe, 'Q_Sharpe': 0.0, 'OOS_Predictions': pred_test
    }, oos_metrics

def train_models_multiprocessing(xgb_specs, X_train, y_train, X_inner_train, y_inner_train, 
                                X_inner_val, y_inner_val, X_test, y_test, config):
    """Train models using multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    # Prepare arguments
    args_list = [(i, spec, X_train.values, y_train.values, X_inner_train.values, y_inner_train.values,
                  X_inner_val.values, y_inner_val.values, X_test.values, y_test.values, config.binary_signal,
                  X_train.index.tolist(), X_inner_train.index.tolist(), X_inner_val.index.tolist(), X_test.index.tolist())
                 for i, spec in enumerate(xgb_specs)]
    
    # Execute in parallel
    with ProcessPoolExecutor(max_workers=min(mp.cpu_count() - 1, len(xgb_specs))) as executor:
        results = list(executor.map(_train_single_model_mp, args_list))
    
    fold_results, model_metrics = zip(*results) if results else ([], [])
    return list(fold_results), list(model_metrics)

def process_single_fold(fold_idx, train_idx, test_idx, X, y, xgb_specs, quality_tracker, config, logger, use_gpu=False, use_multiprocessing=False):
    """Process single fold with all models."""
    # Prepare data splits
    inner_split_point = int(len(train_idx) * (1 - config.inner_val_frac))
    inner_train_idx, inner_val_idx = train_idx[:inner_split_point], train_idx[inner_split_point:]
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_inner_train, y_inner_train = X.iloc[inner_train_idx], y.iloc[inner_train_idx]
    X_inner_val, y_inner_val = X.iloc[inner_val_idx], y.iloc[inner_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Train models based on processing mode
    if use_multiprocessing and not use_gpu and len(xgb_specs) > 1:
        logger.info(f"  Using multiprocessing with {min(mp.cpu_count() - 1, len(xgb_specs))} workers")
        fold_results, model_metrics = train_models_multiprocessing(
            xgb_specs, X_train, y_train, X_inner_train, y_inner_train,
            X_inner_val, y_inner_val, X_test, y_test, config)
    else:
        processing_type = "GPU sequential" if use_gpu else "CPU sequential"
        logger.info(f"  Using {processing_type} processing")
        fold_results, model_metrics = [], []
        for model_idx, spec in enumerate(xgb_specs):
            model_result, metrics = train_single_model(
                model_idx, spec, X_train, y_train, X_inner_train, y_inner_train,
                X_inner_val, y_inner_val, X_test, y_test, config, use_gpu=use_gpu)
            fold_results.append(model_result)
            model_metrics.append(metrics)
    
    # Update quality tracker
    quality_tracker.update_quality(fold_idx, model_metrics)
    
    # Create results dataframe
    fold_df = pd.DataFrame(fold_results)
    
    # Update Q-scores based on configured metric
    if config.q_metric == 'combined':
        q_scores = quality_tracker.get_combined_q_scores(fold_idx, config.ewma_alpha, use_zscore=config.q_use_zscore)
        q_column, q_label = 'Q_Combined', 'Q-Combined'
    elif config.q_metric == 'sharpe_hit':
        q_scores = quality_tracker.get_sharpe_hit_combined_q_scores(fold_idx, config.ewma_alpha, 
                                                                    config.q_sharpe_weight, config.q_use_zscore)
        q_column, q_label = 'Q_Combined', 'Q-Sharpe_Hit'
    else:
        all_q_scores = quality_tracker.get_q_scores(fold_idx, config.ewma_alpha)
        q_scores = all_q_scores[config.q_metric]
        q_column, q_label = f'Q_{config.q_metric.title()}', f"Q-{config.q_metric.title()}"
    
    fold_df[q_column] = q_scores[:len(fold_df)]
    
    # Log summary
    best_oos = fold_df.loc[fold_df['OOS_Sharpe'].idxmax()]
    best_q = fold_df.loc[fold_df[q_column].idxmax()]
    mean_sharpe = fold_df['OOS_Sharpe'].mean()
    mean_hit = fold_df['OOS_Hit_Rate'].mean()
    
    logger.info(f"Fold {fold_idx+1} Summary:")
    logger.info(f"  Best OOS Sharpe: {best_oos['Model']} ({best_oos['OOS_Sharpe']:.3f}, p={best_oos['OOS_Sharpe_p']:.3f})")
    logger.info(f"  Best {q_label}:  {best_q['Model']} ({best_q[q_column]:.3f}, OOS_Sharpe={best_q['OOS_Sharpe']:.3f}, OOS_Hit={best_q['OOS_Hit_Rate']:.3f})")
    logger.info(f"  Mean OOS Sharpe: {mean_sharpe:.3f}, Mean Hit: {mean_hit:.3f}")
    logger.info("")
    
    return fold_df, model_metrics

def choose_optimal_processing_mode(config, logger):
    """Choose optimal processing mode: GPU if available, otherwise CPU multiprocessing."""
    from model.xgb_drivers import detect_gpu
    
    gpu_available = detect_gpu() == "cuda"
    total_models = config.n_models * config.n_folds
    
    if gpu_available:
        use_gpu, use_multiprocessing = True, False
        mode_desc = "GPU sequential"
    elif total_models > 50:
        use_gpu, use_multiprocessing = False, True  # Re-enabled multiprocessing with fix
        mode_desc = "CPU multiprocessing"
    else:
        use_gpu, use_multiprocessing = False, False
        mode_desc = "CPU sequential"
    
    logger.info(f"Processing mode: {mode_desc} (GPU: {gpu_available}, Total models: {total_models})")
    return use_gpu, use_multiprocessing

def run_cross_validation(X, y, config, logger):
    """Run cross-validation analysis with intelligent processing mode selection."""
    
    # Choose optimal processing mode
    use_gpu, use_multiprocessing = choose_optimal_processing_mode(config, logger)
    
    if config.xgb_type == 'deep':
        xgb_specs = generate_deep_xgb_specs(config.n_models, seed=13)
        col_slices = [X.columns.tolist()] * len(xgb_specs)  # Use all features for each model
    elif config.xgb_type == 'tiered':
        # Use stratified_xgb_bank for tiered architecture (returns specs and col_slices)
        xgb_specs, col_slices = stratified_xgb_bank(X.columns.tolist(), n_models=config.n_models, seed=13)
    else:
        xgb_specs = generate_xgb_specs(config.n_models, seed=13)
        col_slices = [X.columns.tolist()] * len(xgb_specs)  # Use all features for each model
    # Choose between expanding and rolling window splits
    if config.rolling_days > 0:
        fold_splits = list(wfo_splits_rolling(len(X), k_folds=config.n_folds, rolling_days=config.rolling_days))
        logger.info(f"Using rolling window: {config.rolling_days} days")
    else:
        fold_splits = list(wfo_splits(len(X), k_folds=config.n_folds))
        logger.info("Using expanding window")
    quality_tracker = QualityTracker(config.n_models, config.quality_halflife)
    all_fold_results = {}
    
    logger.info(f"Starting analysis: {config.n_models} {config.xgb_type} models x {len(fold_splits)} folds")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"Processing Fold {fold_idx+1}/{len(fold_splits)}")
        fold_df, model_metrics = process_single_fold(fold_idx, train_idx, test_idx, X, y, 
                                                   xgb_specs, quality_tracker, config, logger,
                                                   use_gpu=use_gpu, use_multiprocessing=use_multiprocessing)
        # Extract OOS predictions for backtesting
        oos_predictions = {int(row['Model'][1:]): row['OOS_Predictions'] for _, row in fold_df.iterrows()}

        fold_result = {
            'results_df': fold_df,
            'model_metrics': model_metrics,
            'oos_predictions': oos_predictions,  # Store predictions for backtesting
            'test_idx': test_idx  # Store test indices for proper alignment
        }

        # Store trained models if export is requested
        if config.export_production_models:
            trained_models = {}
            model_specs = {}
            for _, row in fold_df.iterrows():
                if 'trained_model' in row:
                    model_idx = int(row['Model'][1:])
                    trained_models[model_idx] = row['trained_model']
                    # Store the XGB spec used for this model (includes feature info)
                    if model_idx < len(xgb_specs):
                        model_specs[model_idx] = xgb_specs[model_idx]
            fold_result['trained_models'] = trained_models
            fold_result['model_specs'] = model_specs
            logger.info(f"Stored {len(trained_models)} trained models for production export")

        all_fold_results[f'fold_{fold_idx+1}'] = fold_result
    
    return all_fold_results, quality_tracker, xgb_specs, fold_splits, col_slices

def run_production_backtest(X, y, all_fold_results, xgb_specs, fold_splits, config, logger, quality_tracker):
    """Run production backtesting."""
    logger.info("Starting full timeline backtesting...")
    
    backtester = FullTimelineBacktester(
        top_n_models=config.top_n_models,
        q_metric=config.q_metric,
        cutoff_fraction=config.cutoff_fraction
    )
    
    backtest_results = backtester.run_full_timeline_backtest(
        X, y, all_fold_results, fold_splits, xgb_specs, quality_tracker, config
    )
    
    # Log results
    for period, key in [('Training', 'training_metrics'), ('Production', 'production_metrics'), ('Full Timeline', 'full_timeline_metrics')]:
        if key in backtest_results and backtest_results[key]:
            metrics = backtest_results[key]
            logger.info(f"{period}: Sharpe={metrics.get('sharpe', 0):.3f} | Hit={metrics.get('hit_rate', 0):.1%} | Return={metrics.get('ann_ret', 0):.2%}")
    
    return backtest_results

def main():
    """Main execution pipeline."""
    config = parse_config()
    logger, results_dir, timestamp = setup_logging(config)

    # Data preparation
    X, y, selected_features = load_and_prepare_data(config, logger)

    # Cross-validation analysis
    all_fold_results, quality_tracker, xgb_specs, fold_splits, col_slices = run_cross_validation(X, y, config, logger)

    # Production backtesting
    backtest_results = run_production_backtest(X, y, all_fold_results, xgb_specs, fold_splits, config, logger, quality_tracker)

    # Visualizations
    logger.info("Generating visualizations...")
    image_paths = create_clean_visualizations(all_fold_results, quality_tracker, backtest_results, config, results_dir, timestamp)

    # Final summary
    log_detailed_backtest_summary(backtest_results, logger)

    logger.info("="*80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Generated {len(image_paths)} visualization files")
    for path in image_paths:
        logger.info(f"  - {os.path.basename(path)}")

    if 'training_metrics' in backtest_results and backtest_results['training_metrics']:
        train_metrics = backtest_results['training_metrics']
        logger.info(f"Training Final: Sharpe={train_metrics.get('sharpe', 0):.3f} | Hit={train_metrics.get('hit_rate', 0):.1%} | Return={train_metrics.get('ann_ret', 0):.2%} | CB={train_metrics.get('cb_ratio', 0):.3f}")

    if 'production_metrics' in backtest_results and backtest_results['production_metrics']:
        prod_metrics = backtest_results['production_metrics']
        logger.info(f"Production Final: Sharpe={prod_metrics.get('sharpe', 0):.3f} | Hit={prod_metrics.get('hit_rate', 0):.1%} | Return={prod_metrics.get('ann_ret', 0):.2%} | CB={prod_metrics.get('cb_ratio', 0):.3f}")

    if 'full_timeline_metrics' in backtest_results:
        full_metrics = backtest_results['full_timeline_metrics']
        logger.info(f"Full Timeline Final: Sharpe={full_metrics.get('sharpe', 0):.3f} | Hit={full_metrics.get('hit_rate', 0):.1%} | Return={full_metrics.get('ann_ret', 0):.2%} | CB={full_metrics.get('cb_ratio', 0):.3f}")

    # Export production models if requested
    if config.export_production_models:
        export_production_models(all_fold_results, xgb_specs, selected_features, backtest_results, config, logger, col_slices)

    # Export signal distribution if requested
    if config.export_signal_distribution:
        export_signal_distribution(backtest_results, config, logger)

if __name__ == "__main__":
    main()
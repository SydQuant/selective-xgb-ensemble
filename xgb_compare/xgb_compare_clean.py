#!/usr/bin/env python3
"""
XGBoost Comparison Framework - Clean & Optimized Final Version

Complete XGBoost model comparison with Q-score tracking and production backtesting.
All redundant code removed, optimized for clarity and performance.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Framework imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parse_config
from metrics_utils import QualityTracker, calculate_model_metrics, normalize_predictions, calculate_metric_pvalue
from visualization_clean import create_clean_visualizations
from full_timeline_backtest import FullTimelineBacktester
from backtest_visualization import log_detailed_backtest_summary
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, fit_xgb_on_slice
from cv.wfo import wfo_splits

def setup_logging(config):
    """Setup logging and directories."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"xgb_compare_{config.log_label}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    config.log_config(logger)
    return logger, results_dir

def load_and_prepare_data(config, logger):
    """Load data with feature selection."""
    df = prepare_real_data_simple(config.target_symbol, start_date=config.start_date, end_date=config.end_date)
    target_col = f"{config.target_symbol}_target_return"
    X, y = df[[c for c in df.columns if c != target_col]], df[target_col]
    logger.info(f"Loaded: X={X.shape}, y={y.shape}")
    
    if not config.no_feature_selection:
        max_features = config.max_features if config.max_features > 0 else -1
        X = apply_feature_selection(X, y, method='block_wise', max_total_features=max_features)
        logger.info(f"Selected: {X.shape[1]} features")
    
    return X, y

def train_single_model(model_idx, spec, X_train, y_train, X_inner_train, y_inner_train, 
                      X_inner_val, y_inner_val, X_test, y_test, config):
    """Train single model and return metrics."""
    model = fit_xgb_on_slice(X_train, y_train, spec)
    
    # Predictions with normalization
    pred_inner_train = normalize_predictions(pd.Series(model.predict(X_inner_train.values), index=X_inner_train.index))
    pred_inner_val = normalize_predictions(pd.Series(model.predict(X_inner_val.values), index=X_inner_val.index))
    pred_test = normalize_predictions(pd.Series(model.predict(X_test.values), index=X_test.index))
    
    # Metrics calculation
    is_metrics = calculate_model_metrics(pred_inner_train, y_inner_train, shifted=False)
    iv_metrics = calculate_model_metrics(pred_inner_val, y_inner_val, shifted=False)
    oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=True)
    
    # Statistical significance
    p_sharpe = calculate_metric_pvalue(pred_test, y_test, 'sharpe', oos_metrics['sharpe'], config.n_bootstraps)
    
    return {
        'Model': f"M{model_idx:02d}",
        'IS_Sharpe': is_metrics['sharpe'],
        'IV_Sharpe': iv_metrics['sharpe'], 
        'OOS_Sharpe': oos_metrics['sharpe'],
        'OOS_Hit_Rate': oos_metrics['hit_rate'],
        'OOS_Sharpe_p': p_sharpe,
        'Q_Sharpe': 0.0  # Will be calculated by quality tracker
    }, oos_metrics

def process_single_fold(fold_idx, train_idx, test_idx, X, y, xgb_specs, quality_tracker, config, logger):
    """Process single fold with all models."""
    # Inner validation split
    inner_split_point = int(len(train_idx) * (1 - config.inner_val_frac))
    inner_train_idx = train_idx[:inner_split_point]
    inner_val_idx = train_idx[inner_split_point:]
    
    # Data slicing
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_inner_train, y_inner_train = X.iloc[inner_train_idx], y.iloc[inner_train_idx]
    X_inner_val, y_inner_val = X.iloc[inner_val_idx], y.iloc[inner_val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    fold_results = []
    model_metrics = []
    
    for model_idx, spec in enumerate(xgb_specs):
        model_result, metrics = train_single_model(
            model_idx, spec, X_train, y_train, X_inner_train, y_inner_train,
            X_inner_val, y_inner_val, X_test, y_test, config
        )
        fold_results.append(model_result)
        model_metrics.append(metrics)
    
    # Update quality tracker
    quality_tracker.update_quality(fold_idx, model_metrics)
    
    # Create results dataframe
    fold_df = pd.DataFrame(fold_results)
    
    # Update Q-scores
    q_scores = quality_tracker.get_q_scores(fold_idx, config.ewma_alpha)['sharpe']
    fold_df['Q_Sharpe'] = q_scores[:len(fold_df)]
    
    # Log summary
    best_oos = fold_df.loc[fold_df['OOS_Sharpe'].idxmax()]
    best_q = fold_df.loc[fold_df['Q_Sharpe'].idxmax()]
    mean_sharpe = fold_df['OOS_Sharpe'].mean()
    
    logger.info(f"Fold {fold_idx+1} Summary:")
    logger.info(f"  Best OOS Sharpe: {best_oos['Model']} ({best_oos['OOS_Sharpe']:.3f}, p={best_oos['OOS_Sharpe_p']:.3f})")
    logger.info(f"  Best Q-Sharpe:   {best_q['Model']} ({best_q['Q_Sharpe']:.3f}, OOS={best_q['OOS_Sharpe']:.3f})")
    logger.info(f"  Mean OOS Sharpe: {mean_sharpe:.3f}")
    logger.info("")
    
    return fold_df, model_metrics

def run_cross_validation(X, y, config, logger):
    """Run cross-validation analysis."""
    if config.xgb_type == 'deep':
        xgb_specs = generate_deep_xgb_specs(config.n_models)
    else:
        xgb_specs = generate_xgb_specs(config.n_models)
    fold_splits = list(wfo_splits(len(X), k_folds=config.n_folds))
    quality_tracker = QualityTracker(config.n_models, config.quality_halflife)
    all_fold_results = {}
    
    logger.info(f"Starting analysis: {config.n_models} {config.xgb_type} models x {len(fold_splits)} folds")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"Processing Fold {fold_idx+1}/{config.n_folds}")
        fold_df, model_metrics = process_single_fold(fold_idx, train_idx, test_idx, X, y, 
                                                   xgb_specs, quality_tracker, config, logger)
        all_fold_results[f'fold_{fold_idx+1}'] = {
            'results_df': fold_df,
            'model_metrics': model_metrics
        }
    
    return all_fold_results, quality_tracker, xgb_specs, fold_splits

def run_production_backtest(X, y, all_fold_results, xgb_specs, fold_splits, config, logger, quality_tracker):
    """Run production backtesting."""
    logger.info("Starting full timeline backtesting...")
    
    backtester = FullTimelineBacktester(
        top_n_models=config.top_n_models,
        q_metric=config.q_metric,
        cutoff_fraction=config.cutoff_fraction
    )
    
    backtest_results = backtester.run_full_timeline_backtest(
        X, y, all_fold_results, fold_splits, xgb_specs, quality_tracker
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
    logger, results_dir = setup_logging(config)
    
    # Data preparation
    X, y = load_and_prepare_data(config, logger)
    
    # Cross-validation analysis
    all_fold_results, quality_tracker, xgb_specs, fold_splits = run_cross_validation(X, y, config, logger)
    
    # Production backtesting
    backtest_results = run_production_backtest(X, y, all_fold_results, xgb_specs, fold_splits, config, logger, quality_tracker)
    
    # Visualizations
    logger.info("Generating visualizations...")
    image_paths = create_clean_visualizations(all_fold_results, quality_tracker, backtest_results, config, results_dir)
    
    # Final summary
    log_detailed_backtest_summary(backtest_results, logger)
    
    logger.info("="*80)
    logger.info("ANALYSIS COMPLETED SUCCESSFULLY!")
    logger.info("="*80)
    logger.info(f"Generated {len(image_paths)} visualization files")
    for path in image_paths:
        logger.info(f"  - {os.path.basename(path)}")
    
    if 'full_timeline_metrics' in backtest_results:
        metrics = backtest_results['full_timeline_metrics']
        logger.info(f"\nFinal Results: Sharpe={metrics.get('sharpe', 0):.3f} | Hit={metrics.get('hit_rate', 0):.1%} | Return={metrics.get('ann_ret', 0):.2%}")

if __name__ == "__main__":
    main()
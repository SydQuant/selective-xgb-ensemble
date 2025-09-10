#!/usr/bin/env python3
"""
XGBoost Stability Portfolio Integration

Combines the best individual models from XGBoost comparison framework with stability-based
selection from the horse race framework. Uses pre-trained model results and applies
sophisticated ensemble combination based on stability scores.

Features:
- Leverages existing XGBoost comparison results
- Applies stability-based model selection with multiple weighting methods
- Creates optimally weighted portfolio combinations
- No modifications to existing xgb_compare scripts
- Production-ready backtesting with rolling rebalancing

Weighting Methods Available:
1. equal_weight: Simple equal weighting within each metric selection
2. stability_weighted: Weight by stability scores from train/val gap analysis
3. performance_weighted: Weight by historical out-of-sample performance
4. sharpe_weighted: Weight by rolling Sharpe ratio performance
5. combined_weighted: Hybrid of stability + performance + Sharpe

Usage:
    python xgb_stability_portfolio.py --target_symbol "@ES#C" --weighting_method stability_weighted
    python xgb_stability_portfolio.py --target_symbol "@TY#C" --weighting_method combined_weighted --n_models 50
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import yaml
import argparse
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# Fix NumExpr warning
import multiprocessing as mp
os.environ['NUMEXPR_MAX_THREADS'] = str(min(16, mp.cpu_count()))

# Framework imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGBCompareConfig
from metrics_utils import QualityTracker, calculate_model_metrics, normalize_predictions
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, fit_xgb_on_slice
from cv.wfo import wfo_splits, wfo_splits_rolling
from ensemble.horse_race_stability import (
    z_tanh, positions_from_scores, pnl_stats_from_signal,
    metric_sharpe, metric_adj_sharpe, metric_hit_rate, stability_score,
    MetricConfig, rolling_horse_race
)

@dataclass
class StabilityPortfolioConfig(XGBCompareConfig):
    """Extended configuration for stability-based portfolio construction."""
    
    # Stability-specific configuration
    start_train: int = 750
    step: int = 21
    horizon: int = 21
    costs_per_turn_backtest: float = 0.0001
    
    # Portfolio weighting configuration
    weighting_method: str = "stability_weighted"  # "equal_weight", "stability_weighted", "performance_weighted", "sharpe_weighted", "combined_weighted"
    stability_weight: float = 0.4  # For combined_weighted
    performance_weight: float = 0.3  # For combined_weighted  
    sharpe_weight: float = 0.3  # For combined_weighted
    
    # Meta-portfolio configuration
    meta_weighting_method: str = "performance_weighted"  # How to combine different metrics
    min_models_per_metric: int = 1
    max_models_per_metric: int = 5
    
    # Metric configurations
    stability_metrics: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        # Set top_n_models dynamically to 10% of n_models (from parent config)
        self.top_n_models = max(1, self.n_models // 10)
        
        if not self.stability_metrics:
            # Default best performing configurations from individual tests
            self.stability_metrics = [
                {
                    'name': 'Sharpe_Aggressive',
                    'metric_type': 'sharpe',
                    'kwargs': {'costs_per_turn': 0.0001},  # Only costs_per_turn for sharpe
                    'alpha': 1.0,
                    'lam_gap': 0.2,
                    'relative_gap': False,
                    'top_k': 3,
                    'eta_quality': 0.1
                },
                {
                    'name': 'Sharpe_Conservative', 
                    'metric_type': 'sharpe',
                    'kwargs': {'costs_per_turn': 0.0001},  # Only costs_per_turn for sharpe
                    'alpha': 0.8,
                    'lam_gap': 0.4,
                    'relative_gap': False,
                    'top_k': 5,
                    'eta_quality': 0.0
                },
                {
                    'name': 'AdjSharpe_Balanced',
                    'metric_type': 'adj_sharpe',
                    'kwargs': {'costs_per_turn': 0.0001, 'lambda_to': 0.1},  # Both costs and lambda_to for adj_sharpe
                    'alpha': 1.0,
                    'lam_gap': 0.3,
                    'relative_gap': False,
                    'top_k': 4,
                    'eta_quality': 0.05
                }
            ]
    
    def log_config(self, logger):
        """Log configuration details."""
        logger.info("=== Stability Portfolio Configuration ===")
        logger.info(f"Target Symbol: {self.target_symbol}")
        logger.info(f"Period: {self.start_date} to {self.end_date}")
        logger.info(f"Models: {self.n_models} ({self.xgb_type})")
        logger.info(f"Folds: {self.n_folds}")
        logger.info(f"Weighting Method: {self.weighting_method}")
        logger.info(f"Meta Weighting: {self.meta_weighting_method}")
        logger.info(f"Stability Metrics: {len(self.stability_metrics)}")
        if self.weighting_method == "combined_weighted":
            logger.info(f"  Stability Weight: {self.stability_weight}")
            logger.info(f"  Performance Weight: {self.performance_weight}")
            logger.info(f"  Sharpe Weight: {self.sharpe_weight}")
        logger.info(f"Rolling Window: {'Yes' if self.rolling_days > 0 else 'No'}")
        logger.info(f"Binary Signals: {self.binary_signal}")
        logger.info("=" * 40)

class XGBoostStabilityDriver:
    """Wrapper for XGBoost models to match horse race Driver interface."""
    
    def __init__(self, model_spec: Dict, model_id: int):
        self.model_spec = model_spec
        self.model_id = model_id
        self.model = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit XGBoost model using the spec."""
        self.model = fit_xgb_on_slice(X, y, self.model_spec, force_cpu=False)
        self.is_fitted = True
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted or self.model is None:
            raise ValueError(f"Model {self.model_id} not fitted yet")
        return self.model.predict(X.values)

def setup_logging(config: StabilityPortfolioConfig) -> Tuple[logging.Logger, str]:
    """Setup logging and results directory."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    log_dir = os.path.join(results_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"stability_portfolio_{config.log_label}_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"=== XGBoost Stability Portfolio Framework ===")
    logger.info(f"Target Symbol: {config.target_symbol}")
    logger.info(f"Period: {config.start_date} to {config.end_date}")
    logger.info(f"Models: {config.n_models} {config.xgb_type}")
    logger.info(f"Weighting Method: {config.weighting_method}")
    logger.info(f"Meta Weighting: {config.meta_weighting_method}")
    logger.info(f"Stability Metrics: {len(config.stability_metrics)}")
    
    return logger, results_dir

def load_and_prepare_data(config: StabilityPortfolioConfig, logger: logging.Logger) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for training."""
    logger.info("Loading and preparing data...")
    
    df = prepare_real_data_simple(config.target_symbol, start_date=config.start_date, end_date=config.end_date)
    target_col = f"{config.target_symbol}_target_return"
    X, y = df[[c for c in df.columns if c != target_col]], df[target_col]
    logger.info(f"Loaded: X={X.shape}, y={y.shape}")
    
    if config.max_features > 0:
        X = apply_feature_selection(X, y, method='block_wise', max_total_features=config.max_features)
        logger.info(f"Selected: {X.shape[1]} features")
    
    return X, y

def create_xgb_drivers(config: StabilityPortfolioConfig, logger: logging.Logger) -> List[XGBoostStabilityDriver]:
    """Create XGBoost drivers using specified configuration."""
    logger.info(f"Creating {config.n_models} XGBoost drivers ({config.xgb_type})...")
    
    if config.xgb_type == "deep":
        specs = generate_deep_xgb_specs(config.n_models, seed=42)
    else:
        specs = generate_xgb_specs(config.n_models, seed=42)
    
    drivers = [XGBoostStabilityDriver(spec, i) for i, spec in enumerate(specs)]
    logger.info(f"Created {len(drivers)} drivers")
    
    return drivers

def create_metric_configs(config: StabilityPortfolioConfig) -> List[MetricConfig]:
    """Convert stability metrics configuration to MetricConfig objects."""
    metric_configs = []
    
    for metric_def in config.stability_metrics:
        # Map metric type to function
        if metric_def['metric_type'] == 'sharpe':
            metric_fn = metric_sharpe
        elif metric_def['metric_type'] == 'adj_sharpe':
            metric_fn = metric_adj_sharpe
        elif metric_def['metric_type'] == 'hit_rate':
            metric_fn = metric_hit_rate
        else:
            raise ValueError(f"Unknown metric type: {metric_def['metric_type']}")
            
        metric_config = MetricConfig(
            name=metric_def['name'],
            fn=metric_fn,
            kwargs=metric_def['kwargs'],
            alpha=metric_def['alpha'],
            lam_gap=metric_def['lam_gap'],
            relative_gap=metric_def['relative_gap'],
            top_k=metric_def['top_k'],
            eta_quality=metric_def['eta_quality']
        )
        
        metric_configs.append(metric_config)
    
    return metric_configs

def calculate_model_weights(stability_scores: np.ndarray, performance_history: np.ndarray, 
                           sharpe_history: np.ndarray, method: str, 
                           config: StabilityPortfolioConfig) -> np.ndarray:
    """Calculate model weights using different weighting methods."""
    n_models = len(stability_scores)
    
    if method == "equal_weight":
        # Simple equal weighting
        weights = np.ones(n_models) / n_models
        
    elif method == "stability_weighted":
        # Weight by stability scores (higher stability = higher weight)
        weights = np.maximum(stability_scores, 0)  # Remove negative scores
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
            
    elif method == "performance_weighted":
        # Weight by historical performance (EWMA of past OOS performance)
        weights = np.maximum(performance_history, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
            
    elif method == "sharpe_weighted":
        # Weight by rolling Sharpe ratio
        weights = np.maximum(sharpe_history, 0)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_models) / n_models
            
    elif method == "combined_weighted":
        # Combine stability, performance, and Sharpe with configurable weights
        stability_norm = np.maximum(stability_scores, 0)
        performance_norm = np.maximum(performance_history, 0)
        sharpe_norm = np.maximum(sharpe_history, 0)
        
        # Normalize each component to [0,1]
        if stability_norm.max() > 0:
            stability_norm = stability_norm / stability_norm.max()
        if performance_norm.max() > 0:
            performance_norm = performance_norm / performance_norm.max()
        if sharpe_norm.max() > 0:
            sharpe_norm = sharpe_norm / sharpe_norm.max()
            
        # Combine with weights
        combined = (config.stability_weight * stability_norm + 
                   config.performance_weight * performance_norm + 
                   config.sharpe_weight * sharpe_norm)
        
        weights = combined / combined.sum() if combined.sum() > 0 else np.ones(n_models) / n_models
        
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    return weights

def create_weighted_ensemble(selected_predictions: List[pd.Series], 
                           stability_scores: np.ndarray,
                           performance_history: np.ndarray,
                           sharpe_history: np.ndarray,
                           config: StabilityPortfolioConfig) -> pd.Series:
    """Create weighted ensemble using specified weighting method."""
    if len(selected_predictions) == 0:
        raise ValueError("No predictions provided for ensemble")
    
    if len(selected_predictions) == 1:
        return selected_predictions[0]
    
    # Calculate weights
    weights = calculate_model_weights(
        stability_scores, performance_history, sharpe_history,
        config.weighting_method, config
    )
    
    # Ensure weights match number of predictions
    weights = weights[:len(selected_predictions)]
    weights = weights / weights.sum()  # Renormalize
    
    # Apply z_tanh normalization to each prediction
    normalized_preds = []
    for pred in selected_predictions:
        normalized_preds.append(pd.Series(z_tanh(pred.values), index=pred.index))
    
    # Weighted combination
    ensemble = pd.Series(0.0, index=selected_predictions[0].index)
    for pred, weight in zip(normalized_preds, weights):
        ensemble += weight * pred
    
    return ensemble.clip(-1, 1)

def create_meta_portfolio(summary_df: pd.DataFrame, window_df: pd.DataFrame, 
                         details: Dict, config: StabilityPortfolioConfig, 
                         logger: logging.Logger) -> Dict[str, Any]:
    """Create meta-portfolio combining multiple stability metrics."""
    logger.info(f"Creating meta-portfolio using {config.meta_weighting_method} method...")
    
    # Extract individual metric performances
    metric_performances = {}
    
    for metric_name in summary_df.index:
        metric_data = details[metric_name]
        
        # Calculate metric weight based on performance and consistency
        sharpe_mean = summary_df.loc[metric_name, 'Sharpe_mean']
        sharpe_median = summary_df.loc[metric_name, 'Sharpe_median']
        pred_corr_mean = summary_df.loc[metric_name, 'PredCorr_mean']
        hit_rate_mean = summary_df.loc[metric_name, 'HitRate_mean']
        
        # Calculate weight using different methods
        if config.meta_weighting_method == "equal_weight":
            weight = 1.0
        elif config.meta_weighting_method == "performance_weighted":
            # Weight based on Sharpe and predictive correlation
            if not pd.isna(pred_corr_mean):
                weight = max(0, sharpe_mean) * (1 + 0.5 * max(0, pred_corr_mean))
            else:
                weight = max(0, sharpe_mean)
        elif config.meta_weighting_method == "consistency_weighted":
            # Weight based on consistency (low difference between mean and median)
            consistency = 1.0 - abs(sharpe_mean - sharpe_median) / (abs(sharpe_mean) + 1e-6)
            weight = max(0, sharpe_mean) * max(0, consistency)
        elif config.meta_weighting_method == "combined_meta":
            # Combine performance, consistency, and hit rate
            if not pd.isna(pred_corr_mean):
                perf_score = max(0, sharpe_mean) * (1 + 0.3 * max(0, pred_corr_mean))
            else:
                perf_score = max(0, sharpe_mean)
            consistency = 1.0 - abs(sharpe_mean - sharpe_median) / (abs(sharpe_mean) + 1e-6)
            hit_score = max(0, hit_rate_mean - 0.5) * 2  # Scale hit rate above 50%
            
            weight = 0.5 * perf_score + 0.3 * consistency + 0.2 * hit_score
        else:
            weight = max(0, sharpe_mean)
            
        metric_performances[metric_name] = {
            'sharpe_mean': sharpe_mean,
            'sharpe_median': sharpe_median,
            'pred_corr': pred_corr_mean,
            'hit_rate': hit_rate_mean,
            'weight': weight
        }
        
        logger.info(f"{metric_name}: Sharpe={sharpe_mean:.3f}, Hit={hit_rate_mean:.3f}, PredCorr={pred_corr_mean:.3f}, Weight={weight:.3f}")
    
    # Normalize weights
    total_weight = sum(perf['weight'] for perf in metric_performances.values())
    if total_weight > 0:
        for metric_name in metric_performances:
            metric_performances[metric_name]['normalized_weight'] = metric_performances[metric_name]['weight'] / total_weight
    else:
        # Equal weights if all weights are zero/negative
        n_metrics = len(metric_performances)
        for metric_name in metric_performances:
            metric_performances[metric_name]['normalized_weight'] = 1.0 / n_metrics
    
    # Log final weights
    logger.info("Final meta-portfolio weights:")
    for metric_name, perf in metric_performances.items():
        logger.info(f"  {metric_name}: {perf['normalized_weight']:.3f}")
    
    return {
        'metric_performances': metric_performances,
        'window_df': window_df,
        'summary_df': summary_df
    }

def save_results(portfolio_results: Dict, config: StabilityPortfolioConfig, 
                results_dir: str, logger: logging.Logger):
    """Save portfolio results to files."""
    # Always save results for stability portfolio
        
    logger.info("Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary results
    summary_path = os.path.join(results_dir, f"stability_portfolio_summary_{config.log_label}_{timestamp}.csv")
    portfolio_results['summary_df'].to_csv(summary_path)
    logger.info(f"Summary saved to: {summary_path}")
    
    # Save window-level results
    window_path = os.path.join(results_dir, f"stability_portfolio_windows_{config.log_label}_{timestamp}.csv")
    portfolio_results['window_df'].to_csv(window_path)
    logger.info(f"Window results saved to: {window_path}")
    
    # Save configuration
    config_path = os.path.join(results_dir, f"stability_portfolio_config_{config.log_label}_{timestamp}.yaml")
    config_dict = {
        'target_symbol': config.target_symbol,
        'period': f"{config.start_date} to {config.end_date}",
        'n_models': config.n_models,
        'xgb_type': config.xgb_type,
        'weighting_method': config.weighting_method,
        'meta_weighting_method': config.meta_weighting_method,
        'stability_metrics': config.stability_metrics,
        'results': {
            'summary_path': summary_path,
            'window_path': window_path
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

def run_stability_portfolio(config: StabilityPortfolioConfig) -> Dict[str, Any]:
    """Main function to run stability portfolio analysis."""
    logger, results_dir = setup_logging(config)
    
    try:
        # Load data
        X, y = load_and_prepare_data(config, logger)
        
        # Create XGBoost drivers
        drivers = create_xgb_drivers(config, logger)
        
        # Create metric configurations
        metric_configs = create_metric_configs(config)
        logger.info(f"Using {len(metric_configs)} stability metrics")
        
        # Run rolling horse race with stability selection
        logger.info("Running rolling horse race with stability selection...")
        summary_df, window_df, details = rolling_horse_race(
            X=X,
            y=y,
            drivers=drivers,
            metrics=metric_configs,
            start_train=config.start_train,
            step=config.step,
            horizon=config.horizon,
            inner_val_frac=config.inner_val_frac,
            costs_per_turn_backtest=config.costs_per_turn_backtest,
            quality_halflife=config.quality_halflife
        )
        
        logger.info("\n=== Stability Portfolio Results ===")
        logger.info(f"Windows processed: {len(window_df)}")
        logger.info("\nMetric Performance Summary:")
        for metric_name in summary_df.index:
            row = summary_df.loc[metric_name]
            logger.info(f"{metric_name}:")
            logger.info(f"  Sharpe (mean): {row['Sharpe_mean']:.3f}")
            logger.info(f"  Hit Rate: {row['HitRate_mean']:.3f}")
            logger.info(f"  Pred Corr: {row['PredCorr_mean']:.3f}")
        
        # Create meta-portfolio
        portfolio_results = create_meta_portfolio(summary_df, window_df, details, config, logger)
        
        # Save results
        save_results(portfolio_results, config, results_dir, logger)
        
        logger.info("=== Stability Portfolio Analysis Complete ===")
        
        return portfolio_results
        
    except Exception as e:
        logger.error(f"Error in stability portfolio analysis: {str(e)}")
        raise

def parse_stability_config() -> StabilityPortfolioConfig:
    """Parse command line arguments and create configuration."""
    parser = argparse.ArgumentParser(
        description="XGBoost Stability Portfolio Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core data parameters (matching xgb_compare_clean.py)
    parser.add_argument('--target_symbol', type=str, default='@ES#C',
                       help='Target symbol to analyze')
    parser.add_argument('--start_date', type=str, default='2015-01-01',
                       help='Start date for analysis')
    parser.add_argument('--end_date', type=str, default='2025-08-01',
                       help='End date for analysis')
    
    # Model parameters
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of cross-validation folds')
    parser.add_argument('--n_models', type=int, default=100,
                       help='Number of XGBoost models per fold')
    parser.add_argument('--max_features', type=int, default=-1,
                       help='Maximum number of features (-1 for all after selection)')
    parser.add_argument('--xgb_type', type=str, default='standard',
                       choices=['standard', 'deep'],
                       help='XGBoost model architecture type')
    parser.add_argument('--inner_val_frac', type=float, default=0.2,
                       help='Fraction of training data for inner validation')
    
    # Feature selection
    parser.add_argument('--no_feature_selection', action='store_true',
                       help='Skip feature selection step')
    
    # Q-score parameters
    parser.add_argument('--ewma_alpha', type=float, default=0.1,
                       help='EWMA alpha for Q-score calculation')
    parser.add_argument('--quality_halflife', type=int, default=63,
                       help='Quality tracker halflife in days')
    
    # Signal processing
    parser.add_argument('--binary_signal', action='store_true',
                       help='Use binary (+1/-1) signals instead of tanh normalization')
    
    # Cross-validation parameters
    parser.add_argument('--rolling_days', type=int, default=0,
                       help='Use rolling window of N days (0 for expanding window)')
    
    # Stability-specific parameters
    parser.add_argument('--start_train', type=int, default=750,
                       help='Initial training period for rolling horse race')
    parser.add_argument('--step', type=int, default=21,
                       help='Step size for rolling windows (days)')
    parser.add_argument('--horizon', type=int, default=21,
                       help='Prediction horizon (days)')
    parser.add_argument('--costs_per_turn_backtest', type=float, default=0.0001,
                       help='Transaction costs per turnover for backtesting')
    
    # Weighting configuration
    parser.add_argument('--weighting_method', type=str, default='stability_weighted',
                       choices=['equal_weight', 'stability_weighted', 'performance_weighted', 
                               'sharpe_weighted', 'combined_weighted'],
                       help='Model weighting method within each metric')
    parser.add_argument('--meta_weighting_method', type=str, default='performance_weighted',
                       choices=['equal_weight', 'performance_weighted', 'consistency_weighted', 'combined_meta'],
                       help='Meta-portfolio weighting method across metrics')
    parser.add_argument('--stability_weight', type=float, default=0.4,
                       help='Weight for stability component in combined_weighted method')
    parser.add_argument('--performance_weight', type=float, default=0.3,
                       help='Weight for performance component in combined_weighted method')
    parser.add_argument('--sharpe_weight', type=float, default=0.3,
                       help='Weight for Sharpe component in combined_weighted method')
    
    # Portfolio configuration
    parser.add_argument('--min_models_per_metric', type=int, default=1,
                       help='Minimum models per stability metric')
    parser.add_argument('--max_models_per_metric', type=int, default=5,
                       help='Maximum models per stability metric')
    
    # Analysis parameters
    parser.add_argument('--n_bootstraps', type=int, default=100,
                       help='Number of bootstrap samples for p-value calculation')
    parser.add_argument('--log_label', type=str, default='stability_portfolio',
                       help='Label for logging and output files')
    
    # Preset configurations
    parser.add_argument('--use_best_configs', action='store_true',
                       help='Use best configurations from individual tests')
    parser.add_argument('--preset', type=str, choices=['quick', 'bonds', 'comprehensive'],
                       help='Use predefined configuration preset')
    
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = StabilityPortfolioConfig(
        target_symbol=args.target_symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        n_folds=args.n_folds,
        n_models=args.n_models,
        max_features=args.max_features,
        xgb_type=args.xgb_type,
        inner_val_frac=args.inner_val_frac,
        no_feature_selection=args.no_feature_selection,
        ewma_alpha=args.ewma_alpha,
        quality_halflife=args.quality_halflife,
        binary_signal=args.binary_signal,
        rolling_days=args.rolling_days,
        start_train=args.start_train,
        step=args.step,
        horizon=args.horizon,
        costs_per_turn_backtest=args.costs_per_turn_backtest,
        weighting_method=args.weighting_method,
        meta_weighting_method=args.meta_weighting_method,
        stability_weight=args.stability_weight,
        performance_weight=args.performance_weight,
        sharpe_weight=args.sharpe_weight,
        min_models_per_metric=args.min_models_per_metric,
        max_models_per_metric=args.max_models_per_metric,
        n_bootstraps=args.n_bootstraps,
        log_label=args.log_label
    )
    
    # Apply presets
    if args.preset == 'quick':
        config.start_date = "2020-01-01"
        config.end_date = "2024-01-01"
        config.n_models = 25
        config.start_train = 252
        config.log_label = "quick_stability"
    elif args.preset == 'bonds':
        config.target_symbol = "@TY#C"
        config.n_models = 50
        config.costs_per_turn_backtest = 0.00005
        config.log_label = "bonds_stability"
    elif args.preset == 'comprehensive':
        config.n_models = 100
        config.n_folds = 15
        config.log_label = "comprehensive_stability"
    
    # Apply best configs if requested
    if args.use_best_configs:
        config.stability_metrics = [
            {
                'name': 'Sharpe_Optimal',
                'metric_type': 'sharpe',
                'kwargs': {'costs_per_turn': 0.0001},  # Only costs_per_turn for sharpe
                'alpha': 1.0,
                'lam_gap': 0.3,
                'relative_gap': False,
                'top_k': 5,
                'eta_quality': 0.1
            },
            {
                'name': 'AdjSharpe_Conservative',
                'metric_type': 'adj_sharpe',
                'kwargs': {'costs_per_turn': 0.0001, 'lambda_to': 0.05},  # Both costs and lambda_to for adj_sharpe
                'alpha': 0.8,
                'lam_gap': 0.5,
                'relative_gap': False,
                'top_k': 3,
                'eta_quality': 0.0
            }
        ]
    
    return config

def main():
    """Main entry point."""
    config = parse_stability_config()
    results = run_stability_portfolio(config)
    return results

if __name__ == "__main__":
    results = main()
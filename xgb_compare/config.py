#!/usr/bin/env python3
"""
Configuration management for XGBoost comparison framework.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class XGBCompareConfig:
    """Configuration for XGBoost comparison framework."""
    
    # Data parameters
    target_symbol: str = '@ES#C'
    start_date: str = '2015-01-01'
    end_date: str = '2025-08-01'
    
    # Model training
    n_folds: int = 10
    n_models: int = 50
    max_features: int = -1  # -1 for auto selection
    xgb_type: str = 'standard'
    inner_val_frac: float = 0.2
    no_feature_selection: bool = False
    
    # Quality tracking
    ewma_alpha: float = 0.1
    quality_halflife: int = 63
    
    # Backtesting
    cutoff_fraction: float = 0.6  # Training/production split
    top_n_models: int = 5  # Auto-calculated from model_selection_pct
    model_selection_pct: float = 0.05  # Top 5% of models
    q_metric: str = 'sharpe'
    reselection_frequency: int = 1
    
    # Combined Q-score parameters
    q_sharpe_weight: float = 0.5  # Weight for Sharpe in combined metrics (hit_rate gets 1-this)
    q_use_zscore: bool = True  # Use z-score normalization for combined metrics
    q_metric_weights: dict = None  # Custom metric weights dict (overrides q_sharpe_weight)
    
    # Signal processing
    binary_signal: bool = False  # Use +1/-1 signals instead of tanh
    
    # Cross-validation parameters
    rolling_days: int = 0  # If > 0, use rolling window of this many days instead of expanding
    
    # Analysis parameters
    n_bootstraps: int = 100
    log_label: str = 'comparison'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'target_symbol': self.target_symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'n_folds': self.n_folds,
            'n_models': self.n_models,
            'max_features': self.max_features,
            'xgb_type': self.xgb_type,
            'inner_val_frac': self.inner_val_frac,
            'no_feature_selection': self.no_feature_selection,
            'ewma_alpha': self.ewma_alpha,
            'quality_halflife': self.quality_halflife,
            'cutoff_fraction': self.cutoff_fraction,
            'top_n_models': self.top_n_models,
            'q_metric': self.q_metric,
            'reselection_frequency': self.reselection_frequency,
            'n_bootstraps': self.n_bootstraps,
            'log_label': self.log_label
        }
    
    def log_config(self, logger):
        """Log configuration."""
        logger.info("="*80)
        logger.info("XGBoost Comparison Framework")
        logger.info("="*80)
        logger.info("Configuration:")
        logger.info(f"  Target Symbol: {self.target_symbol}")
        logger.info(f"  Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"  Models: {self.n_models}, Type: {self.xgb_type}")
        logger.info(f"  Folds: {self.n_folds}")
        logger.info(f"  Max Features: {self.max_features} ({'Limited' if self.max_features > 0 else 'Auto'})")
        logger.info(f"  Feature Selection: {'Disabled' if self.no_feature_selection else 'Enabled'}")
        logger.info(f"  Inner Val Fraction: {self.inner_val_frac}")
        logger.info(f"  EWMA Alpha: {self.ewma_alpha}, Quality Halflife: {self.quality_halflife} days")
        logger.info(f"  Signal Type: {'Binary (+1/-1)' if self.binary_signal else 'Tanh Normalized'}")
        logger.info(f"  Cross-validation: {'Rolling ' + str(self.rolling_days) + ' days' if self.rolling_days > 0 else 'Expanding window'}")
        logger.info(f"  Production: Cutoff={self.cutoff_fraction}, Top N={self.top_n_models}, Q-Metric={self.q_metric}")
        logger.info(f"  Reselection: Every {self.reselection_frequency} fold(s)")
        logger.info("")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(description='XGBoost Comparison Framework', 
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Core parameters - concise format
    parser.add_argument('--target_symbol', type=str, default='@ES#C', help='Target symbol')
    parser.add_argument('--start_date', type=str, default='2015-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-08-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--n_folds', type=int, default=10, help='Cross-validation folds')
    parser.add_argument('--n_models', type=int, default=50, help='XGBoost models per fold')
    parser.add_argument('--max_features', type=int, default=-1, help='Max features (-1=auto)')
    parser.add_argument('--xgb_type', type=str, default='standard', choices=['standard', 'deep', 'tiered'], help='XGBoost type')
    parser.add_argument('--inner_val_frac', type=float, default=0.2, help='Inner validation fraction')
    parser.add_argument('--no_feature_selection', action='store_true', help='Skip feature selection')
    parser.add_argument('--ewma_alpha', type=float, default=0.1, help='EWMA alpha for Q-scores')
    parser.add_argument('--quality_halflife', type=int, default=63, help='Quality halflife (days)')
    parser.add_argument('--cutoff_fraction', type=float, default=0.6, help='Training/production split fraction')
    parser.add_argument('--top_n_models', type=int, default=5, help='Top models for production')
    parser.add_argument('--model_selection_pct', type=float, default=0.05, help='Top percentage of models to select (0.05=5%)')
    parser.add_argument('--q_metric', type=str, default='sharpe', 
                       choices=['sharpe', 'hit_rate', 'cb_ratio', 'adj_sharpe', 'combined', 'sharpe_hit'], help='Q-score metric')
    parser.add_argument('--reselection_frequency', type=int, default=1, help='Reselection frequency (folds)')
    
    # Optional parameters  
    parser.add_argument('--q_sharpe_weight', type=float, default=0.5, help='Sharpe weight in combined Q-score')
    parser.add_argument('--q_use_zscore', action='store_true', default=True, help='Use z-score normalization')
    parser.add_argument('--n_bootstraps', type=int, default=100, help='Bootstrap iterations for p-values')
    parser.add_argument('--log_label', type=str, default='comparison', help='Output filename label')
    parser.add_argument('--binary_signal', action='store_true', help='Use binary +1/-1 signals')
    parser.add_argument('--rolling', type=int, default=0, dest='rolling_days', help='Rolling window days (0=expanding)')
    
    return parser

def parse_config() -> XGBCompareConfig:
    """Parse command line arguments and return configuration."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Dynamic top_n_models: calculate based on model_selection_pct
    top_n_models = args.top_n_models
    if top_n_models == 5:  # Default value, not explicitly set
        # Use model_selection_pct: 0.05 = top 5%, 0.10 = top 10%, etc.
        # Use round() for standard rounding (2.5 → 3, 2.4 → 2)
        top_n_models = max(1, round(args.model_selection_pct * args.n_models))
    
    return XGBCompareConfig(
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
        cutoff_fraction=args.cutoff_fraction,
        top_n_models=top_n_models,
        model_selection_pct=args.model_selection_pct,
        q_metric=args.q_metric,
        reselection_frequency=args.reselection_frequency,
        q_sharpe_weight=args.q_sharpe_weight,
        q_use_zscore=args.q_use_zscore,
        n_bootstraps=args.n_bootstraps,
        log_label=args.log_label,
        binary_signal=args.binary_signal,
        rolling_days=args.rolling_days
    )
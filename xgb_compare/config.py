#!/usr/bin/env python3
"""
Configuration management for XGBoost Comparison Framework.
Centralizes argument parsing and configuration logic.
"""

import argparse
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class XGBCompareConfig:
    """Configuration class for XGBoost comparison analysis."""
    
    # Core data parameters
    target_symbol: str = '@ES#C'
    start_date: str = '2015-01-01'
    end_date: str = '2025-08-01'
    
    # Model parameters
    n_folds: int = 10
    n_models: int = 50
    max_features: int = -1  # -1 for all after cluster reduction
    xgb_type: str = 'standard'
    inner_val_frac: float = 0.2
    
    # Feature selection
    no_feature_selection: bool = False
    
    # Q-score parameters
    ewma_alpha: float = 0.1
    quality_halflife: int = 63
    
    # Production backtesting
    cutoff_fraction: float = 0.7
    top_n_models: int = 5  # Will be dynamically set to 10% of n_models
    q_metric: str = 'sharpe'  # 'sharpe', 'hit_rate', 'combined', 'sharpe_hit', 'cb_ratio', 'adj_sharpe'
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
        """Log configuration in a clean format."""
        logger.info("="*80)
        logger.info("XGBoost Comparison Framework")
        logger.info("="*80)
        logger.info("Configuration:")
        logger.info(f"  Target Symbol: {self.target_symbol}")
        logger.info(f"  Date Range: {self.start_date} to {self.end_date}")
        logger.info(f"  Models: {self.n_models}, Type: {self.xgb_type}")
        logger.info(f"  Folds: {self.n_folds}")
        logger.info(f"  Max Features: {self.max_features} ({'All after cluster reduction' if self.max_features == -1 else 'Limited'})")
        logger.info(f"  Feature Selection: {'Disabled' if self.no_feature_selection else 'Enabled'}")
        logger.info(f"  Inner Val Fraction: {self.inner_val_frac}")
        logger.info(f"  EWMA Alpha: {self.ewma_alpha}, Quality Halflife: {self.quality_halflife} days")
        logger.info(f"  Signal Type: {'Binary (+1/-1)' if self.binary_signal else 'Tanh Normalized'}")
        logger.info(f"  Cross-validation: {'Rolling ' + str(self.rolling_days) + ' days' if self.rolling_days > 0 else 'Expanding window'}")
        logger.info(f"  Production: Cutoff={self.cutoff_fraction}, Top N={self.top_n_models}, Q-Metric={self.q_metric}")
        if self.q_metric in ['combined', 'sharpe_hit']:
            logger.info(f"  Combined Q-Score: Sharpe={self.q_sharpe_weight:.1f}, Hit={1-self.q_sharpe_weight:.1f}, Z-score={self.q_use_zscore}")
        logger.info(f"  Reselection: Every {self.reselection_frequency} fold(s)")
        logger.info("")

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description='XGBoost Comparison Framework - Comprehensive model analysis with Q-scores and production backtesting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core configuration
    parser.add_argument('--target_symbol', type=str, default='@ES#C', 
                       help='Target symbol for analysis')
    parser.add_argument('--start_date', type=str, default='2015-01-01', 
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2025-08-01', 
                       help='End date (YYYY-MM-DD)')
    
    # Model parameters
    parser.add_argument('--n_folds', type=int, default=10, 
                       help='Number of cross-validation folds')
    parser.add_argument('--n_models', type=int, default=50, 
                       help='Number of XGBoost models per fold')
    parser.add_argument('--max_features', type=int, default=-1, 
                       help='Maximum features after selection (-1 for all after cluster reduction)')
    parser.add_argument('--xgb_type', type=str, default='standard', 
                       choices=['standard', 'deep', 'tiered'],
                       help='XGBoost architecture type')
    parser.add_argument('--inner_val_frac', type=float, default=0.2, 
                       help='Inner validation fraction for IS/IV split')
    
    # Feature selection
    parser.add_argument('--no_feature_selection', action='store_true', 
                       help='Skip feature selection entirely')
    
    # Q-score parameters
    parser.add_argument('--ewma_alpha', type=float, default=0.1, 
                       help='EWMA alpha parameter for Q-score calculation')
    parser.add_argument('--quality_halflife', type=int, default=63, 
                       help='Quality tracking halflife (days)')
    
    # Production backtesting
    parser.add_argument('--cutoff_fraction', type=float, default=0.7, 
                       help='Fraction of folds for model selection vs backtesting')
    parser.add_argument('--top_n_models', type=int, default=5, 
                       help='Number of top models to select for production')
    parser.add_argument('--q_metric', type=str, default='sharpe', 
                       choices=['sharpe', 'hit_rate', 'cb_ratio', 'adj_sharpe', 'combined', 'sharpe_hit'],
                       help='Q-score metric for model selection')
    parser.add_argument('--reselection_frequency', type=int, default=1, 
                       help='Model reselection frequency (in folds)')
    
    # Combined Q-score parameters
    parser.add_argument('--q_sharpe_weight', type=float, default=0.5,
                       help='Weight for Sharpe in combined Q-scores (hit_rate gets 1-this)')
    parser.add_argument('--q_use_zscore', action='store_true', default=True,
                       help='Use z-score normalization for combined Q-scores')
    parser.add_argument('--q_no_zscore', dest='q_use_zscore', action='store_false',
                       help='Disable z-score normalization (use simple weighted average)')
    
    # Analysis parameters
    parser.add_argument('--n_bootstraps', type=int, default=100, 
                       help='Number of bootstrap iterations for p-values')
    parser.add_argument('--log_label', type=str, default='comparison', 
                       help='Label for output filenames')
    parser.add_argument('--binary_signal', action='store_true',
                       help='Use binary +1/-1 signals instead of tanh normalization')
    
    # Cross-validation parameters
    parser.add_argument('--rolling', type=int, default=0, dest='rolling_days',
                       help='Use rolling window of N days instead of expanding window (0 = expanding)')
    
    return parser

def parse_config() -> XGBCompareConfig:
    """Parse command line arguments and return configuration."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Dynamic top_n_models: if not explicitly provided, use 10% of n_models
    top_n_models = args.top_n_models
    if top_n_models == 5:  # Default value, not explicitly set
        top_n_models = max(1, int(0.1 * args.n_models))  # 10% of n_models, minimum 1
    
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
        q_metric=args.q_metric,
        reselection_frequency=args.reselection_frequency,
        q_sharpe_weight=args.q_sharpe_weight,
        q_use_zscore=args.q_use_zscore,
        n_bootstraps=args.n_bootstraps,
        log_label=args.log_label,
        binary_signal=args.binary_signal,
        rolling_days=args.rolling_days
    )
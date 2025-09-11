"""
XGBoost Comparison Framework

A comprehensive system for comparing XGBoost models across cross-validation folds
with Q-score tracking, visualization, and production backtesting.
"""

from .metrics_utils import (
    calculate_annualized_sharpe, calculate_hit_rate, calculate_adjusted_sharpe,
    calculate_cb_ratio, calculate_model_metrics, calculate_model_metrics_from_pnl, QualityTracker
)

__version__ = "1.0.0"
__all__ = [
    "calculate_annualized_sharpe", "calculate_hit_rate", "calculate_adjusted_sharpe",
    "calculate_cb_ratio", "calculate_model_metrics", "calculate_model_metrics_from_pnl", "QualityTracker"
]
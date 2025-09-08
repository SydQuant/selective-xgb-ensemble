#!/usr/bin/env python3
"""
Metrics calculation utilities for XGBoost comparison framework.
Extracted and enhanced from existing analysis scripts.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_annualized_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def calculate_hit_rate(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Calculate hit rate (directional accuracy)."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actual_returns)
    return np.mean(pred_signs == actual_signs)

def calculate_information_ratio(returns: pd.Series) -> float:
    """Calculate annualized information ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

def calculate_adjusted_sharpe(returns: pd.Series, predictions: pd.Series, lambda_turnover: float = 0.1) -> float:
    """Calculate adjusted Sharpe ratio with turnover penalty."""
    sharpe = calculate_annualized_sharpe(returns)
    if len(predictions) > 1:
        turnover = np.mean(np.abs(predictions.diff().fillna(0)))
        return sharpe - lambda_turnover * turnover
    return sharpe

def calculate_cb_ratio(returns: pd.Series) -> float:
    """Calculate Calmar-Burke ratio (annual return / max drawdown)."""
    if len(returns) == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    cumulative = returns.cumsum()
    max_dd = (cumulative.expanding().max() - cumulative).max()
    return ann_ret / max_dd if max_dd > 0 else 0.0

def calculate_dapy_binary(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Calculate DAPY binary score."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    correct = np.sign(predictions) == np.sign(actual_returns)
    return (np.mean(correct) - 0.5) * 252 * 100

def calculate_dapy_both(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Calculate DAPY both score (direction + magnitude)."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    direction_acc = np.mean(np.sign(predictions) == np.sign(actual_returns))
    magnitude_corr = np.corrcoef(predictions, actual_returns)[0,1] if len(predictions) > 1 else 0.0
    combined_score = 0.7 * direction_acc + 0.3 * abs(magnitude_corr)
    return (combined_score - 0.5) * 252 * 100

def bootstrap_pvalue(actual_metric: float, returns: pd.Series, predictions: pd.Series, 
                    metric_func, n_bootstraps: int = 100) -> float:
    """Calculate bootstrap p-value for a metric."""
    if len(returns) < 10:
        return 0.5
    
    bootstrap_metrics = []
    for _ in range(n_bootstraps):
        shuffled_returns = returns.sample(frac=1, replace=False).reset_index(drop=True)
        shuffled_returns.index = returns.index
        try:
            boot_metric = metric_func(predictions, shuffled_returns)
            bootstrap_metrics.append(boot_metric)
        except:
            bootstrap_metrics.append(0.0)
    
    if len(bootstrap_metrics) == 0:
        return 0.5
    
    # Two-tailed p-value
    better_count = sum(1 for x in bootstrap_metrics if abs(x) >= abs(actual_metric))
    return better_count / len(bootstrap_metrics)

def normalize_predictions(predictions: pd.Series) -> pd.Series:
    """Normalize predictions using z-score + tanh transformation."""
    if predictions.std() == 0:
        return pd.Series(np.zeros_like(predictions), index=predictions.index)
    
    z_scores = (predictions - predictions.mean()) / predictions.std()
    normalized = np.tanh(z_scores)
    return pd.Series(normalized, index=predictions.index)

def calculate_ewma_quality(series: pd.Series, alpha: float = 0.1) -> float:
    """Calculate EWMA quality metric."""
    if len(series) == 0:
        return 0.0
    ewma_vals = series.ewm(alpha=alpha, adjust=False).mean()
    return ewma_vals.iloc[-1] if len(ewma_vals) > 0 else 0.0

def calculate_model_metrics(predictions: pd.Series, returns: pd.Series, 
                          shifted: bool = False) -> Dict[str, float]:
    """Calculate comprehensive metrics for a model's predictions."""
    if shifted:
        # For OOS evaluation, shift predictions by 1 to avoid look-ahead
        signal = predictions.shift(1).fillna(0.0)
    else:
        signal = predictions
    
    pnl = signal * returns
    
    metrics = {
        'ann_ret': pnl.mean() * 252,
        'ann_vol': pnl.std() * np.sqrt(252),
        'sharpe': calculate_annualized_sharpe(pnl),
        'adj_sharpe': calculate_adjusted_sharpe(pnl, signal),
        'hit_rate': calculate_hit_rate(signal, returns),
        'cb_ratio': calculate_cb_ratio(pnl),
        'information_ratio': calculate_information_ratio(pnl),
        'dapy_binary': calculate_dapy_binary(signal, returns),
        'dapy_both': calculate_dapy_both(signal, returns)
    }
    
    return metrics

def calculate_metric_pvalue(predictions: pd.Series, returns: pd.Series, 
                           metric_name: str, actual_value: float, 
                           n_bootstraps: int = 100) -> float:
    """Calculate p-value for a specific metric using bootstrap."""
    
    def sharpe_func(pred, ret):
        return calculate_annualized_sharpe(pred * ret)
    
    def hit_func(pred, ret):
        return calculate_hit_rate(pred, ret)
    
    def cb_func(pred, ret):
        return calculate_cb_ratio(pred * ret)
    
    def adj_sharpe_func(pred, ret):
        return calculate_adjusted_sharpe(pred * ret, pred)
    
    metric_functions = {
        'sharpe': sharpe_func,
        'hit_rate': hit_func,
        'cb_ratio': cb_func,
        'adj_sharpe': adj_sharpe_func,
        'information_ratio': lambda p, r: calculate_information_ratio(p * r)
    }
    
    if metric_name in metric_functions:
        return bootstrap_pvalue(actual_value, returns, predictions, 
                              metric_functions[metric_name], n_bootstraps)
    else:
        return 0.5

class QualityTracker:
    """Track quality momentum for models using EWMA."""
    
    def __init__(self, n_models: int, quality_halflife: int = 63):
        self.n_models = n_models
        self.quality_history = {
            'sharpe': [[] for _ in range(n_models)],
            'hit_rate': [[] for _ in range(n_models)],
            'cb_ratio': [[] for _ in range(n_models)],
            'adj_sharpe': [[] for _ in range(n_models)]
        }
        self.q_decay = np.exp(np.log(0.5) / max(1, quality_halflife))
        
    def update_quality(self, fold_idx: int, model_metrics: list):
        """Update quality history for all models in current fold."""
        for model_idx, metrics in enumerate(model_metrics):
            if model_idx < self.n_models:
                self.quality_history['sharpe'][model_idx].append(metrics.get('sharpe', 0.0))
                self.quality_history['hit_rate'][model_idx].append(metrics.get('hit_rate', 0.0))
                self.quality_history['cb_ratio'][model_idx].append(metrics.get('cb_ratio', 0.0))
                self.quality_history['adj_sharpe'][model_idx].append(metrics.get('adj_sharpe', 0.0))
    
    def get_q_scores(self, fold_idx: int, ewma_alpha: float = 0.1) -> Dict[str, list]:
        """Calculate Q-scores for all models up to current fold."""
        q_scores = {
            'sharpe': [0.0] * self.n_models,
            'hit_rate': [0.0] * self.n_models,
            'cb_ratio': [0.0] * self.n_models,
            'adj_sharpe': [0.0] * self.n_models
        }
        
        for metric in q_scores.keys():
            for model_idx in range(self.n_models):
                history = self.quality_history[metric][model_idx]
                if fold_idx == 0 or len(history) <= 1:
                    # First fold: Q = 0.0 baseline
                    q_scores[metric][model_idx] = 0.0
                else:
                    # Subsequent folds: EWMA of historical performance (exclude current fold)
                    historical_series = pd.Series(history[:-1]) if len(history) > 1 else pd.Series([0.0])
                    q_scores[metric][model_idx] = calculate_ewma_quality(historical_series, ewma_alpha)
        
        return q_scores
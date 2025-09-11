#!/usr/bin/env python3
"""
Metrics utilities for XGBoost comparison framework.
Performance calculations, quality tracking, and signal processing.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def calculate_annualized_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0:
        return 0.0
    
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    # Handle NaN and zero volatility cases
    if std_ret == 0 or np.isnan(mean_ret) or np.isnan(std_ret):
        return 0.0
    
    sharpe = (mean_ret / std_ret) * np.sqrt(252)
    return sharpe if not np.isnan(sharpe) else 0.0

def calculate_hit_rate(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Calculate hit rate (directional accuracy)."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    # Use values to avoid index alignment issues
    pred_signs = np.sign(predictions.values)
    actual_signs = np.sign(actual_returns.values)
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
    # Ensure index alignment by using values
    correct = np.sign(predictions.values) == np.sign(actual_returns.values)
    return (np.mean(correct) - 0.5) * 252 * 100

def calculate_dapy_both(predictions: pd.Series, actual_returns: pd.Series) -> float:
    """Calculate DAPY both score (direction + magnitude)."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    # Ensure index alignment by using values
    direction_acc = np.mean(np.sign(predictions.values) == np.sign(actual_returns.values))
    magnitude_corr = np.corrcoef(predictions.values, actual_returns.values)[0,1] if len(predictions) > 1 else 0.0
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
        boot_metric = metric_func(predictions, shuffled_returns)
        bootstrap_metrics.append(boot_metric)
    
    if len(bootstrap_metrics) == 0:
        return 0.5
    
    # Two-tailed p-value
    better_count = sum(1 for x in bootstrap_metrics if abs(x) >= abs(actual_metric))
    return better_count / len(bootstrap_metrics)

def normalize_predictions(predictions: pd.Series, binary_signal: bool = False) -> pd.Series:
    """Normalize predictions: z-score + tanh or binary +1/-1 signals."""
    if predictions.std() == 0:
        return pd.Series(np.zeros_like(predictions), index=predictions.index)
    
    z_scores = (predictions - predictions.mean()) / predictions.std()
    
    if binary_signal:
        # Binary: +1 for positive, -1 for negative z-scores
        normalized = np.where(z_scores > 0, 1.0, -1.0)
    else:
        # Continuous: tanh normalization
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
    """Calculate comprehensive performance metrics for predictions."""
    if shifted:
        # Shift predictions for OOS evaluation (avoid look-ahead bias)
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
    """Track quality momentum for models using EWMA with combined metrics support."""
    
    def __init__(self, n_models: int, quality_halflife: int = 63):
        self.n_models = n_models
        self.quality_history = {
            'sharpe': [[] for _ in range(n_models)],
            'hit_rate': [[] for _ in range(n_models)],
            'cb_ratio': [[] for _ in range(n_models)],
            'adj_sharpe': [[] for _ in range(n_models)]
        }
        self.q_decay = np.exp(np.log(0.5) / max(1, quality_halflife))
        self.halflife = quality_halflife
        
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
                if len(history) == 0:
                    # No data available: Q = 0.0 baseline
                    q_scores[metric][model_idx] = 0.0
                else:
                    # Subsequent folds: EWMA of historical performance up to and including fold_idx
                    # Use all history up to and including fold_idx for model selection
                    historical_series = pd.Series(history[:fold_idx+1]) if fold_idx < len(history) else pd.Series(history)
                    q_scores[metric][model_idx] = calculate_ewma_quality(historical_series, ewma_alpha)
        
        return q_scores
    
    def get_combined_q_scores(self, fold_idx: int, ewma_alpha: float = 0.1, 
                             metric_weights: Dict[str, float] = None, 
                             use_zscore: bool = True) -> list:
        """
        Calculate combined Q-scores using multiple metrics.
        
        CORRECTED APPROACH: Z-score normalization applied to raw historical values,
        then EWMA applied to normalized series.
        
        Args:
            fold_idx: Current fold index
            ewma_alpha: EWMA alpha parameter  
            metric_weights: Dict of metric weights, e.g. {'sharpe': 0.5, 'hit_rate': 0.5}
            use_zscore: Whether to z-score normalize metrics before combining
            
        Returns:
            List of combined Q-scores for all models
        """
        if metric_weights is None:
            # Default: 50/50 Sharpe and Hit Rate
            metric_weights = {'sharpe': 0.5, 'hit_rate': 0.5}
        
        combined_scores = [0.0] * self.n_models
        
        if use_zscore and len(metric_weights) > 1:
            # Z-score normalization on raw history, then EWMA
            combined_q_scores = {}
            
            # For each metric, calculate z-score normalized EWMA Q-scores
            for metric_name, weight in metric_weights.items():
                if weight <= 0 or metric_name not in self.quality_history:
                    continue
                
                # Collect all raw values across all models/folds for z-score stats
                all_values = []
                for model_idx in range(self.n_models):
                    history = self.quality_history[metric_name][model_idx]
                    historical_data = history[:fold_idx+1] if fold_idx < len(history) else history
                    all_values.extend(historical_data)
                
                # Calculate z-score stats
                if len(all_values) > 0:
                    mean_val = np.mean(all_values)
                    std_val = np.std(all_values) if np.std(all_values) > 1e-9 else 1.0
                else:
                    mean_val, std_val = 0.0, 1.0
                
                # Calculate z-score normalized EWMA for each model
                combined_q_scores[metric_name] = []
                for model_idx in range(self.n_models):
                    history = self.quality_history[metric_name][model_idx]
                    if len(history) == 0:
                        combined_q_scores[metric_name].append(0.0)
                    else:
                        historical_data = history[:fold_idx+1] if fold_idx < len(history) else history
                        # Z-score normalize, then EWMA
                        normalized_data = [(x - mean_val) / std_val for x in historical_data]
                        ewma_result = calculate_ewma_quality(pd.Series(normalized_data), ewma_alpha)
                        combined_q_scores[metric_name].append(ewma_result)
            
            # Weighted combination
            total_weight = sum(weight for metric_name, weight in metric_weights.items() 
                             if weight > 0 and metric_name in combined_q_scores)
            
            for model_idx in range(self.n_models):
                combined_score = 0.0
                for metric_name, weight in metric_weights.items():
                    if weight > 0 and metric_name in combined_q_scores:
                        combined_score += (weight / total_weight) * combined_q_scores[metric_name][model_idx]
                combined_scores[model_idx] = combined_score
                
        else:
            # Simple weighted average approach (no z-score normalization)
            all_q_scores = self.get_q_scores(fold_idx, ewma_alpha)
            
            # Extract valid metrics to combine
            metrics_to_combine = {}
            for metric_name, weight in metric_weights.items():
                if metric_name in all_q_scores and weight > 0:
                    metrics_to_combine[metric_name] = (all_q_scores[metric_name], weight)
            
            if not metrics_to_combine:
                # Fallback to Sharpe if no valid metrics
                return all_q_scores.get('sharpe', [0.0] * self.n_models)
            
            total_weight = sum(weight for _, weight in metrics_to_combine.values())
            for model_idx in range(self.n_models):
                combined_score = 0.0
                for metric_name, (scores, weight) in metrics_to_combine.items():
                    combined_score += (weight / total_weight) * scores[model_idx]
                combined_scores[model_idx] = combined_score
        
        return combined_scores
    
    def get_sharpe_hit_combined_q_scores(self, fold_idx: int, ewma_alpha: float = 0.1, 
                                        sharpe_weight: float = 0.5, use_zscore: bool = True) -> list:
        """
        Convenience method for Sharpe + Hit Rate combination (most common use case).
        
        Args:
            fold_idx: Current fold index
            ewma_alpha: EWMA alpha parameter
            sharpe_weight: Weight for Sharpe (Hit Rate gets 1 - sharpe_weight)
            use_zscore: Whether to use z-score normalization
            
        Returns:
            List of combined Q-scores for all models
        """
        hit_weight = 1.0 - sharpe_weight
        metric_weights = {'sharpe': sharpe_weight, 'hit_rate': hit_weight}
        return self.get_combined_q_scores(fold_idx, ewma_alpha, metric_weights, use_zscore)
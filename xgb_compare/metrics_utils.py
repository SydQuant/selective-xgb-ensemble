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
    """Calculate hit rate (directional accuracy), excluding zero signals (no trade days)."""
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    
    # Use values to avoid index alignment issues
    pred_values = predictions.values
    return_values = actual_returns.values
    
    # Exclude days with zero signals (no trade decisions)
    non_zero_mask = pred_values != 0
    
    if not np.any(non_zero_mask):
        return 0.0  # All signals are zero
    
    # Calculate hit rate only for non-zero signals
    pred_signs = np.sign(pred_values[non_zero_mask])
    actual_signs = np.sign(return_values[non_zero_mask])
    
    return np.mean(pred_signs == actual_signs)

def calculate_information_ratio(returns: pd.Series) -> float:
    """Calculate annualized information ratio (essentially same as Sharpe for daily returns)."""
    # Information ratio is the same as Sharpe ratio for absolute returns
    return calculate_annualized_sharpe(returns)

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
    peak = cumulative.expanding().max()
    drawdown = peak - cumulative
    max_dd = drawdown.max()
    
    if max_dd > 1e-6:  # Avoid division by very small numbers
        return ann_ret / max_dd
    else:
        return 0.0  # No meaningful drawdown

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

def normalize_predictions(predictions: pd.Series) -> pd.Series:
    """Normalize predictions: Simple sign-based approach aligned with PROD signal_engine.py."""

    # Convert each prediction to +1/-1 based on sign (no z-score, no tanh)
    # This matches PROD/common/signal_engine.py line 135-148 logic
    normalized = np.where(predictions > 0, 1.0, np.where(predictions < 0, -1.0, 0.0))
    return pd.Series(normalized, index=predictions.index)

def combine_binary_signals(binary_predictions_list: list) -> pd.Series:
    """
    Combine binary signals using simple voting: sum of +1/-1 votes.
    If tied (even number, equal +1 and -1), result is 0.
    
    Example:
    - Model predictions: [0.1, -0.3, 0.6] → Binary signals: [+1, -1, +1] → Sum: +1
    - Model predictions: [0.1, -0.3] → Binary signals: [+1, -1] → Sum: 0 (tie)
    """
    if not binary_predictions_list:
        return pd.Series(dtype=float)
    
    # Ensure all predictions have same index
    index = binary_predictions_list[0].index
    
    # Sum the binary votes (+1/-1)
    vote_sum = sum(binary_predictions_list)
    
    # Convert sum to final binary signal
    final_signal = np.where(vote_sum > 0, 1.0, np.where(vote_sum < 0, -1.0, 0.0))
    
    return pd.Series(final_signal, index=index)

def calculate_ewma_quality(series: pd.Series, alpha: float = 0.1) -> float:
    """Calculate EWMA quality metric."""
    if len(series) == 0:
        return 0.0
    ewma_vals = series.ewm(alpha=alpha, adjust=False).mean()
    return ewma_vals.iloc[-1] if len(ewma_vals) > 0 else 0.0

def filter_covid_period(pnl_series: pd.Series, signal: pd.Series = None, returns: pd.Series = None) -> tuple:
    """
    Filter out COVID period (Mar 2020 - May 2020) from PnL calculation.

    Args:
        pnl_series: PnL series to filter
        signal: Optional signal series to filter
        returns: Optional returns series to filter

    Returns:
        Tuple of filtered (pnl_series, signal, returns)
    """
    # Define COVID period
    covid_start = pd.Timestamp('2020-03-01')
    covid_end = pd.Timestamp('2020-05-31 23:59:59')

    # Create mask for non-COVID periods
    if hasattr(pnl_series.index, 'to_pydatetime'):
        covid_mask = ~((pnl_series.index >= covid_start) & (pnl_series.index <= covid_end))
    else:
        # Fallback for different index types
        covid_mask = pd.Series(True, index=pnl_series.index)
        try:
            covid_mask = ~((pnl_series.index >= covid_start) & (pnl_series.index <= covid_end))
        except:
            # If date filtering fails, return original series (safer than crashing)
            return pnl_series, signal, returns

    # Filter all series consistently
    filtered_pnl = pnl_series[covid_mask]
    filtered_signal = signal[covid_mask] if signal is not None else None
    filtered_returns = returns[covid_mask] if returns is not None else None

    return filtered_pnl, filtered_signal, filtered_returns

def calculate_model_metrics_from_pnl(pnl_series: pd.Series,
                                   signal: pd.Series = None,
                                   returns: pd.Series = None,
                                   skip_covid: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics from pre-calculated PnL series.
    This eliminates redundant PnL calculations across the system.

    Args:
        pnl_series: Pre-calculated PnL series (signal * returns)
        signal: Optional signal series for hit_rate and adjusted_sharpe calculations
        returns: Optional returns series for hit_rate calculations
        skip_covid: If True, exclude Mar 2020 - May 2020 from calculations
    """
    # Apply COVID filtering if requested
    if skip_covid:
        pnl_series, signal, returns = filter_covid_period(pnl_series, signal, returns)
    metrics = {
        'ann_ret': pnl_series.mean() * 252,
        'ann_vol': pnl_series.std() * np.sqrt(252),
        'sharpe': calculate_annualized_sharpe(pnl_series),
        'cb_ratio': calculate_cb_ratio(pnl_series),
        'information_ratio': calculate_information_ratio(pnl_series),
    }
    
    # Add signal-dependent metrics if signal and returns are provided
    if signal is not None and returns is not None:
        metrics.update({
            'adj_sharpe': calculate_adjusted_sharpe(pnl_series, signal),
            'hit_rate': calculate_hit_rate(signal, returns),
            'dapy_binary': calculate_dapy_binary(signal, returns),
            'dapy_both': calculate_dapy_both(signal, returns)
        })
    else:
        # Default values when signal/returns not available
        metrics.update({
            'adj_sharpe': metrics['sharpe'],  # Fallback to regular sharpe
            'hit_rate': 0.5,  # Neutral hit rate
            'dapy_binary': 0.0,
            'dapy_both': 0.0
        })
    
    return metrics

def calculate_model_metrics(predictions: pd.Series, returns: pd.Series, 
                          shifted: bool = False) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for predictions.
    DEPRECATED: Use calculate_model_metrics_from_pnl for better performance.
    
    NOTE: shifted parameter removed - signals and targets are properly aligned.
    No artificial lag needed since both represent the same time period.
    """
    # Direct calculation - no shift needed (signals and targets properly aligned)
    signal = predictions
    pnl = signal * returns
    
    # Use the new PnL-based function internally
    return calculate_model_metrics_from_pnl(pnl, signal, returns)

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
        Simplified: Apply z-score normalization then EWMA for cleaner logic.
        """
        if metric_weights is None:
            metric_weights = {'sharpe': 0.5, 'hit_rate': 0.5}
        
        combined_scores = [0.0] * self.n_models
        
        if use_zscore and len(metric_weights) > 1:
            # Simplified z-score normalization approach
            normalized_scores = {}
            
            for metric_name, weight in metric_weights.items():
                if weight <= 0 or metric_name not in self.quality_history:
                    continue
                
                # Get all historical values for z-score calculation
                all_historical = []
                for model_idx in range(self.n_models):
                    history = self.quality_history[metric_name][model_idx]
                    relevant_history = history[:fold_idx+1] if fold_idx < len(history) else history
                    all_historical.extend(relevant_history)
                
                # Calculate normalization parameters
                if all_historical:
                    mean_val = np.mean(all_historical)
                    std_val = np.std(all_historical)
                    std_val = std_val if std_val > 1e-9 else 1.0
                else:
                    mean_val, std_val = 0.0, 1.0
                
                # Normalize and calculate EWMA for each model
                normalized_scores[metric_name] = []
                for model_idx in range(self.n_models):
                    history = self.quality_history[metric_name][model_idx]
                    if not history:
                        normalized_scores[metric_name].append(0.0)
                    else:
                        relevant_history = history[:fold_idx+1] if fold_idx < len(history) else history
                        # Normalize then apply EWMA
                        normalized = [(x - mean_val) / std_val for x in relevant_history]
                        ewma_score = calculate_ewma_quality(pd.Series(normalized), ewma_alpha)
                        normalized_scores[metric_name].append(ewma_score)
            
            # Combine weighted scores
            if normalized_scores:
                total_weight = sum(w for m, w in metric_weights.items() if m in normalized_scores)
                for model_idx in range(self.n_models):
                    for metric_name, weight in metric_weights.items():
                        if metric_name in normalized_scores:
                            combined_scores[model_idx] += (weight / total_weight) * normalized_scores[metric_name][model_idx]
        else:
            # Simple weighted average without z-score
            q_scores = self.get_q_scores(fold_idx, ewma_alpha)
            valid_metrics = {m: q_scores[m] for m in metric_weights if m in q_scores and metric_weights[m] > 0}
            
            if valid_metrics:
                total_weight = sum(metric_weights[m] for m in valid_metrics)
                for model_idx in range(self.n_models):
                    for metric_name, scores in valid_metrics.items():
                        combined_scores[model_idx] += (metric_weights[metric_name] / total_weight) * scores[model_idx]
        
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
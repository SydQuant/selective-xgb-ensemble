"""
Alternative performance metrics for XGBoost trading system validation
Implements adjusted Sharpe ratio and CB_ratio as described in Step 8
"""
import numpy as np
import pandas as pd
from scipy import stats

def compute_adjusted_sharpe(sharpe, num_years, num_points, adj_sharpe_n):
    """
    Adjusted Sharpe ratio with multiple testing correction
    
    Args:
        sharpe: Original Sharpe ratio
        num_years: Number of years in the dataset
        num_points: Number of data points  
        adj_sharpe_n: Number of tests for multiple testing correction
    
    Returns:
        Adjusted Sharpe ratio accounting for data mining bias
    """
    if sharpe == 0 or num_years <= 0 or num_points <= 1:
        return 0.0
    
    # T-statistic 
    t_ratio = sharpe * np.sqrt(num_years)
    
    # Two-tailed p-value
    p_val = stats.t.sf(abs(t_ratio), num_points - 1) * 2
    
    # Adjust for multiple testing
    adj_p_val = 1 - (1 - p_val) ** adj_sharpe_n
    
    # Adjusted critical t-statistic
    if adj_p_val >= 1.0:
        return 0.0
    
    adj_t_ratio = stats.t.ppf(1 - adj_p_val / 2, num_points - 1)
    
    # Convert back to adjusted Sharpe
    return abs(adj_t_ratio) / np.sqrt(num_years)

def cb_ratio(sharpe, max_drawdown, l1_penalty=0.0, weights=None):
    """
    CB_ratio with L1 regularization penalty
    
    Args:
        sharpe: Sharpe ratio
        max_drawdown: Maximum drawdown (negative value)
        l1_penalty: L1 regularization strength  
        weights: Model weights for L1 penalty calculation
    
    Returns:
        CB_ratio = sharpe * r2 / (abs(max_dd) + 1e-6) - l1_penalty * sum(abs(weights))
    """
    r2 = 1.0  # Risk adjustment factor
    
    # Risk-adjusted performance
    adjusted = sharpe * r2 / (abs(max_drawdown) + 1e-6)
    
    # L1 penalty on weights
    penalty = 0.0
    if l1_penalty > 0 and weights is not None:
        penalty = l1_penalty * np.sum(np.abs(weights))
    
    return adjusted - penalty

def sharpe_ratio(signal, returns):
    """Calculate Sharpe ratio from signal and returns"""
    pnl = (signal.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    if pnl.std() == 0:
        return 0.0
    return pnl.mean() / pnl.std() * np.sqrt(252)

def max_drawdown(signal, returns):
    """Calculate maximum drawdown from signal and returns"""
    pnl = (signal.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    equity = pnl.cumsum()
    peak = equity.expanding().max()
    drawdown = equity - peak
    return float(drawdown.min())
"""
Predictive Objective Function for XGBoost Trading System

Combines ICIR (Information Coefficient Information Ratio) and calibrated LogScore
to evaluate predictive quality of trading signals.

Key components:
- ICIR: Spearman correlation consistency across time eras 
- LogScore: Calibrated probability scoring for directional prediction
- HitRate: Optional gating mechanism for minimum directional accuracy
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, Dict, Any, Tuple, Optional
from scipy.stats import spearmanr

# ---- Era helpers (simplified) -----------------------------------------------

def make_monthly_eras(index: pd.DatetimeIndex) -> pd.Series:
    """Return monthly era labels per timestamp."""
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    return pd.Series(index, index=index).dt.to_period("M").astype(str)

def era_ic_list(signal: pd.Series, returns: pd.Series, eras: pd.Series, min_obs: int = 5) -> np.ndarray:
    """Calculate Spearman IC per era - optimized for speed"""
    # Apply temporal lag once
    lagged_signal = signal.shift(1).fillna(0.0).reindex_like(returns)
    
    # Pre-filter valid data
    valid_mask = ~(np.isnan(lagged_signal) | np.isnan(returns))
    if not np.any(valid_mask):
        return np.array([])
    
    lagged_signal = lagged_signal[valid_mask]
    returns_clean = returns[valid_mask] 
    eras_clean = eras[valid_mask]
    
    ics = []
    # Use unique() on cleaned data for efficiency
    for era in np.unique(eras_clean):
        era_mask = (eras_clean == era)
        if era_mask.sum() < min_obs:
            continue
        
        try:
            # Direct array indexing is faster
            era_sig = lagged_signal.values[era_mask]
            era_ret = returns_clean.values[era_mask]
            
            rho, _ = spearmanr(era_sig, era_ret)
            if np.isfinite(rho):
                ics.append(rho)
        except:
            continue
    
    return np.array(ics, dtype=float)

def icir(signal: pd.Series, returns: pd.Series, eras: pd.Series) -> float:
    """ICIR = mean(IC_era) / std(IC_era). Returns 0 if undefined."""
    ics = era_ic_list(signal, returns, eras)
    if len(ics) == 0:
        return 0.0
    
    ic_mean = float(np.mean(ics))
    ic_std = float(np.std(ics, ddof=0))
    
    return 0.0 if ic_std == 0 else ic_mean / ic_std

# ---- Direction calibration (simplified) ------------------------------------

def fit_simple_calibrator(train_scores: np.ndarray, train_directions: np.ndarray):
    """Fast calibrator fitting with optimized fallback."""
    try:
        from sklearn.linear_model import LogisticRegression
        # Use liblinear for small datasets - much faster than lbfgs
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=100, random_state=42)
        clf.fit(train_scores.reshape(-1, 1), train_directions.astype(int))
        return lambda x: np.clip(
            clf.predict_proba(x.reshape(-1, 1))[:, 1], 
            1e-12, 1 - 1e-12
        )
    except:
        # Fast fallback using base rate
        pos_rate = float(np.mean(train_directions))
        return lambda x: np.full(x.shape, pos_rate, dtype=float)

def calibrated_logscore(
    train_signal: pd.Series,
    train_returns: pd.Series,
    val_signal: pd.Series, 
    val_returns: pd.Series
) -> float:
    """Fast calibrated LogScore with minimal data copying"""
    
    # Apply temporal lag with efficient alignment
    train_sig_lag = train_signal.shift(1).fillna(0.0)
    train_ret_align = train_returns.reindex_like(train_signal)
    val_sig_lag = val_signal.shift(1).fillna(0.0)
    val_ret_align = val_returns.reindex_like(val_signal)
    
    # Fast valid data filtering using numpy
    train_valid = ~(np.isnan(train_sig_lag) | np.isnan(train_ret_align))
    val_valid = ~(np.isnan(val_sig_lag) | np.isnan(val_ret_align))
    
    if not np.any(train_valid) or not np.any(val_valid):
        return 0.0
    
    # Direct numpy operations - much faster
    train_signals = train_sig_lag.values[train_valid]
    train_directions = (train_ret_align.values[train_valid] > 0).astype(int)
    val_signals = val_sig_lag.values[val_valid]
    val_directions = (val_ret_align.values[val_valid] > 0).astype(int)
    
    try:
        calibrate = fit_simple_calibrator(train_signals, train_directions)
        predicted_probs = calibrate(val_signals)
        
        # Vectorized LogScore calculation
        clipped_probs = np.clip(predicted_probs, 1e-12, 1-1e-12)
        logscore = np.mean(
            val_directions * np.log(clipped_probs) + 
            (1 - val_directions) * np.log(1 - clipped_probs)
        )
        
        return float(logscore)
    except:
        return 0.0

# ---- Hit rate (simplified) -------------------------------------------------

def directional_hit_rate(signal: pd.Series, returns: pd.Series) -> float:
    """Calculate fraction of correct directional predictions."""
    # CRITICAL FIX: Apply temporal lag to avoid look-ahead bias
    # Signal on day T-1 should predict return on day T
    lagged_signal = signal.shift(1).fillna(0.0).reindex_like(returns)
    
    signal_signs = np.sign(lagged_signal.values)
    return_signs = np.sign(returns.values)
    
    valid_mask = (~np.isnan(signal_signs)) & (~np.isnan(return_signs))
    if not np.any(valid_mask):
        return 0.0
    
    return float(np.mean(signal_signs[valid_mask] == return_signs[valid_mask]))

# ---- Main predictive objective ---------------------------------------------

def predictive_quality_score(
    signal: pd.Series,
    returns: pd.Series, 
    train_indices: Optional[Iterable] = None,
    val_indices: Optional[Iterable] = None,
    icir_weight: float = 1.0,
    logscore_weight: float = 1.0,
    min_hit_rate: Optional[float] = None,
    **kwargs
) -> float:
    """Optimized predictive objective combining ICIR and calibrated LogScore."""
    
    if len(signal) == 0 or len(returns) == 0:
        return 0.0
    
    # Fast train/val split using integer indices
    if train_indices is None or val_indices is None:
        n = len(signal)
        split_point = int(0.7 * n)
        train_indices = slice(0, split_point)
        val_indices = slice(split_point, n)
    
    # Efficient slicing
    train_signal = signal.iloc[train_indices]
    train_returns = returns.iloc[train_indices] 
    val_signal = signal.iloc[val_indices]
    val_returns = returns.iloc[val_indices]
    
    if len(val_signal) == 0 or len(val_returns) == 0:
        return 0.0
    
    # Early exit with hit rate gate if specified
    if min_hit_rate is not None:
        hit_rate = directional_hit_rate(val_signal, val_returns)
        if hit_rate < min_hit_rate:
            return -1e9
    
    # Skip ICIR calculation if weight is 0 for speed
    icir_score = 0.0
    if icir_weight > 0:
        val_eras = make_monthly_eras(val_returns.index)
        icir_score = icir(val_signal, val_returns, val_eras)
    
    # Skip LogScore if weight is 0 for speed
    logscore = 0.0
    if logscore_weight > 0:
        logscore = calibrated_logscore(train_signal, train_returns, val_signal, val_returns)
    
    return float(icir_weight * icir_score + logscore_weight * logscore)

# ---- Wrapper for registry compatibility -----------------------------------

def predictive_icir_logscore(signal: pd.Series, returns: pd.Series, **kwargs) -> float:
    """
    Registry-compatible wrapper for predictive objective.
    
    This is the main function that will be registered in the objective registry.
    Name: Predictive ICIR+LogScore - combines era-based information coefficient 
    consistency with calibrated directional probability scoring.
    """
    # Extract specific parameters and pass the rest
    icir_weight = kwargs.pop('icir_weight', 1.0)
    logscore_weight = kwargs.pop('logscore_weight', 1.0)
    min_hit_rate = kwargs.pop('min_hit_rate', None)
    
    return predictive_quality_score(
        signal=signal,
        returns=returns,
        icir_weight=icir_weight,
        logscore_weight=logscore_weight, 
        min_hit_rate=min_hit_rate,
        **kwargs
    )
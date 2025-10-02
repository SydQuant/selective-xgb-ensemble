"""
Stability-based driver selection for main pipeline integration.
Provides a drop-in replacement for GROPE optimization using stability horse race methodology.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

def z_tanh_transform(scores: np.ndarray) -> np.ndarray:
    """Apply z-score normalization followed by tanh squashing."""
    eps = 1e-9
    z_scores = (scores - scores.mean()) / (scores.std(ddof=0) + eps)
    return np.tanh(z_scores)

def calculate_sharpe_metric(signal: pd.Series, returns: pd.Series, costs_per_turn: float = 0.0) -> float:
    """Calculate Sharpe ratio with proper temporal alignment."""
    if len(signal) == 0 or len(returns) == 0:
        return 0.0
    
    # Ensure temporal alignment - signal[t-1] predicts return[t]
    aligned_signal = signal.shift(1).fillna(0.0)
    aligned_returns = returns.reindex_like(aligned_signal).fillna(0.0)
    
    # Remove any remaining NaN pairs
    clean_data = pd.DataFrame({'signal': aligned_signal, 'returns': aligned_returns}).dropna()
    if len(clean_data) < 2:
        return 0.0
    
    # Apply z-tanh transformation to signal
    positions = z_tanh_transform(clean_data['signal'].values)
    
    # Calculate PnL with transaction costs
    pnl = positions * clean_data['returns'].values
    if costs_per_turn > 0:
        turnover = np.abs(np.diff(np.concatenate([[0], positions])))
        costs = turnover * costs_per_turn
        pnl = pnl - costs
    
    # Calculate annualized Sharpe ratio
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(252)
    return float(sharpe) if np.isfinite(sharpe) else 0.0

def calculate_adjusted_sharpe_metric(signal: pd.Series, returns: pd.Series, 
                                   costs_per_turn: float = 0.0, lambda_to: float = 0.0) -> float:
    """Calculate adjusted Sharpe with turnover penalty."""
    base_sharpe = calculate_sharpe_metric(signal, returns, costs_per_turn=0.0)  # No double-counting costs
    
    if lambda_to > 0:
        # Calculate turnover penalty
        aligned_signal = signal.shift(1).fillna(0.0)
        positions = z_tanh_transform(aligned_signal.values)
        turnover = np.mean(np.abs(np.diff(np.concatenate([[0], positions]))))
        return float(base_sharpe - lambda_to * turnover)
    
    return base_sharpe

def calculate_hit_rate_metric(signal: pd.Series, returns: pd.Series) -> float:
    """Calculate hit rate (directional accuracy) with proper temporal alignment."""
    if len(signal) == 0 or len(returns) == 0:
        return 0.5  # Random baseline
    
    # Temporal alignment
    aligned_signal = signal.shift(1).fillna(0.0)
    aligned_returns = returns.reindex_like(aligned_signal).fillna(0.0)
    
    # Remove NaN pairs
    clean_data = pd.DataFrame({'signal': aligned_signal, 'returns': aligned_returns}).dropna()
    if len(clean_data) == 0:
        return 0.5
    
    # Calculate directional accuracy
    signal_signs = np.sign(clean_data['signal'].values)
    return_signs = np.sign(clean_data['returns'].values)
    
    hit_rate = np.mean(signal_signs == return_signs)
    return float(hit_rate) if np.isfinite(hit_rate) else 0.5

def calculate_stability_score(train_metric: float, val_metric: float, 
                            alpha: float = 1.0, lam_gap: float = 0.3, 
                            relative_gap: bool = False) -> float:
    """Calculate stability score penalizing train-validation performance gaps."""
    if relative_gap and abs(train_metric) > 1e-9:
        gap = max(0.0, (train_metric - val_metric) / abs(train_metric))
    else:
        gap = max(0.0, train_metric - val_metric)
    
    stability = alpha * val_metric - lam_gap * gap
    return float(stability) if np.isfinite(stability) else 0.0

@dataclass
class StabilityConfig:
    """Configuration for stability-based driver selection."""
    metric_name: str = "sharpe"
    top_k: int = 5
    alpha: float = 1.0
    lam_gap: float = 0.3
    relative_gap: bool = False
    eta_quality: float = 0.0
    quality_halflife: int = 63
    inner_val_frac: float = 0.2
    costs_per_turn: float = 0.0001

def stability_driver_selection_and_combination(
    train_signals: List[pd.Series], 
    test_signals: List[pd.Series], 
    y_train: pd.Series,
    config: StabilityConfig
) -> Tuple[pd.Series, Dict]:
    """
    Stability-based driver selection and equal-weight combination.
    
    This function replaces both driver selection and GROPE optimization
    with stability-based ensemble methodology.
    
    Args:
        train_signals: List of training signals from XGBoost models
        test_signals: List of test signals from XGBoost models  
        y_train: Training target returns
        config: Stability configuration
        
    Returns:
        combined_signal: Equal-weight ensemble signal for test period
        diagnostics: Selection diagnostics and metadata
    """
    
    if len(train_signals) == 0 or len(test_signals) == 0:
        logger.error("No signals provided for stability selection")
        return pd.Series(dtype=float), {}
    
    n_drivers = len(train_signals)
    logger.info(f"Stability selection: evaluating {n_drivers} drivers with {config.metric_name} metric")
    
    # Split training data for inner train/validation
    train_end = len(y_train)
    val_start = int(train_end * (1.0 - config.inner_val_frac))
    
    inner_train_idx = slice(0, val_start)
    inner_val_idx = slice(val_start, train_end)
    
    y_inner_train = y_train.iloc[inner_train_idx]
    y_inner_val = y_train.iloc[inner_val_idx]
    
    # Evaluate each driver on inner splits
    stability_scores = []
    train_metrics = []
    val_metrics = []
    
    # Select metric function
    metric_functions = {
        "sharpe": calculate_sharpe_metric,
        "adj_sharpe": calculate_adjusted_sharpe_metric,
        "hit_rate": calculate_hit_rate_metric
    }
    
    if config.metric_name not in metric_functions:
        logger.warning(f"Unknown metric {config.metric_name}, using sharpe")
        metric_fn = metric_functions["sharpe"]
    else:
        metric_fn = metric_functions[config.metric_name]
    
    for i, signal in enumerate(train_signals):
        try:
            # Split signal for inner evaluation
            inner_train_signal = signal.iloc[inner_train_idx]
            inner_val_signal = signal.iloc[inner_val_idx]
            
            # Calculate metrics on inner splits
            if config.metric_name == "adj_sharpe":
                train_metric = metric_fn(inner_train_signal, y_inner_train, config.costs_per_turn, 0.05)
                val_metric = metric_fn(inner_val_signal, y_inner_val, config.costs_per_turn, 0.05)
            elif config.metric_name == "hit_rate":
                train_metric = metric_fn(inner_train_signal, y_inner_train)
                val_metric = metric_fn(inner_val_signal, y_inner_val)
            else:
                train_metric = metric_fn(inner_train_signal, y_inner_train, config.costs_per_turn)
                val_metric = metric_fn(inner_val_signal, y_inner_val, config.costs_per_turn)
            
            # Calculate stability score
            stability = calculate_stability_score(
                train_metric, val_metric, 
                config.alpha, config.lam_gap, config.relative_gap
            )
            
            stability_scores.append(stability)
            train_metrics.append(train_metric)
            val_metrics.append(val_metric)
            
        except Exception as e:
            logger.warning(f"Error evaluating driver {i}: {e}")
            stability_scores.append(0.0)
            train_metrics.append(0.0)
            val_metrics.append(0.0)
    
    stability_scores = np.array(stability_scores)
    
    # Select top-k drivers based on stability scores
    top_k = min(config.top_k, n_drivers, len([s for s in stability_scores if s > 0]))
    if top_k == 0:
        logger.error("No drivers with positive stability scores")
        return pd.Series(dtype=float), {}
    
    selected_indices = np.argsort(-stability_scores)[:top_k]
    logger.info(f"Selected top {top_k} drivers: {selected_indices.tolist()}")
    
    # Create equal-weight ensemble from selected drivers
    ensemble_signals = []
    for idx in selected_indices:
        # Apply z-tanh transformation to each selected signal
        raw_signal = test_signals[idx]
        normalized_signal = pd.Series(
            z_tanh_transform(raw_signal.values), 
            index=raw_signal.index
        )
        ensemble_signals.append(normalized_signal)
    
    # Equal-weight combination
    if len(ensemble_signals) == 1:
        combined_signal = ensemble_signals[0]
    else:
        combined_signal = pd.concat(ensemble_signals, axis=1).mean(axis=1)
    
    # Final clipping to [-1, 1] range
    combined_signal = combined_signal.clip(-1.0, 1.0).fillna(0.0)
    
    # Prepare diagnostics
    diagnostics = {
        "method": "stability_ensemble",
        "metric_used": config.metric_name,
        "selected_indices": selected_indices.tolist(),
        "n_selected": len(selected_indices),
        "stability_scores": stability_scores.tolist(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "mean_stability": float(np.mean(stability_scores[selected_indices])),
        "config": {
            "top_k": config.top_k,
            "alpha": config.alpha,
            "lam_gap": config.lam_gap,
            "metric_name": config.metric_name
        }
    }
    
    logger.info(f"Stability ensemble: mean stability = {diagnostics['mean_stability']:.3f}")
    logger.info(f"Selected drivers: {selected_indices.tolist()}")
    
    return combined_signal, diagnostics
"""
Automatic Scale Normalization for Objective Functions

NEW FEATURE: This module enables seamless combination of different objective functions
that operate on vastly different scales (e.g., DAPY ~20 units vs Adjusted Sharpe ~0.4 units).

Automatically normalizes objectives to [0,1] range based on empirical scale estimates,
eliminating the need for manual scale-aware weighting when combining objectives.

Usage:
    # Instead of manual scale weighting:
    objective:
      dapy: {weight: 1.0}
      adjusted_sharpe: {weight: 50.0}  # Manual scale compensation

    # Use auto-normalized objectives:
    objective:
      dapy: {weight: 1.0, auto_normalize: true}
      adjusted_sharpe: {weight: 1.0, auto_normalize: true}  # Equal weights work!
"""

import numpy as np
import pandas as pd
from typing import Dict, Callable, Tuple, Optional
import warnings

# NEW FEATURE: Dynamic scale normalization - no hard-coded ranges needed!
# Auto-calculates min-max from actual data for seamless objective combination

def normalize_objective_score(
    score: float, 
    obj_name: str = None,
    custom_range: Optional[Tuple[float, float]] = None,
    data_range: Optional[Tuple[float, float]] = None
) -> float:
    """
    NEW FEATURE: Dynamic normalization using actual data min-max instead of hard-coded ranges.
    
    Normalize objective score to [0,1] range based on actual data distribution.
    
    Args:
        score: Raw objective function score
        obj_name: Name of objective function (optional, for documentation)
        custom_range: Optional custom (min, max) range override
        data_range: (min, max) calculated from actual data - REQUIRED if custom_range not provided
        
    Returns:
        Normalized score in [0,1] range (higher is always better)
    """
    # Determine range source
    range_tuple = custom_range or data_range
    if range_tuple is None:
        warnings.warn(f"No range provided for normalization of '{obj_name}', using raw score")
        return score
    
    min_val, max_val = range_tuple
    if max_val == min_val:
        return 0.5
    
    # Normalize and clip to [0,1]
    normalized = (score - min_val) / (max_val - min_val)
    return float(np.clip(normalized, 0.0, 1.0))

def create_auto_normalized_objective(
    base_objective: Callable[[pd.Series, pd.Series], float],
    obj_name: str,
    custom_range: Optional[Tuple[float, float]] = None,
    calibration_data: Optional[Tuple[pd.Series, pd.Series]] = None,
    n_samples: int = 50
) -> Callable[[pd.Series, pd.Series], float]:
    """
    NEW FEATURE: Create auto-normalized version using dynamic data-based scaling.
    
    Args:
        base_objective: Original objective function
        obj_name: Name for documentation
        custom_range: Optional custom (min, max) range override
        calibration_data: Optional (signal, returns) tuple for pre-computing scale
        n_samples: Number of samples for dynamic range estimation
        
    Returns:
        Normalized objective function that returns values in [0,1]
    """
    # Pre-compute range if calibration data provided
    precomputed_range = (
        estimate_dynamic_range(base_objective, calibration_data[0], calibration_data[1], n_samples)
        if calibration_data is not None and custom_range is None else None
    )
    
    def normalized_objective(signal: pd.Series, returns: pd.Series, **kwargs) -> float:
        raw_score = base_objective(signal, returns, **kwargs)
        
        # Determine normalization range
        data_range = (
            custom_range or
            precomputed_range or
            estimate_dynamic_range(base_objective, signal, returns, n_samples)
        )
        
        return normalize_objective_score(raw_score, obj_name, data_range=data_range)
    
    # Add docstring with normalization info
    base_doc = getattr(base_objective, '__doc__', None)
    normalized_objective.__doc__ = (
        f"DYNAMIC AUTO-NORMALIZED VERSION: {base_doc}\n"
        f"Returns normalized score in [0,1] range using dynamic data-based scaling."
        if base_doc else f"Dynamic auto-normalized {obj_name} objective (returns [0,1] range)"
    )
    
    return normalized_objective

def estimate_dynamic_range(
    objective_fn: Callable,
    signal: pd.Series,
    returns: pd.Series,
    n_samples: int = 50,
    bootstrap_fraction: float = 0.8
) -> Tuple[float, float]:
    """
    NEW FEATURE: Estimate objective range dynamically from actual data.
    
    Uses bootstrap sampling of actual signal/returns data to estimate objective range,
    eliminating need for hard-coded scale ranges.
    
    Args:
        objective_fn: Objective function to test
        signal: Actual signal data for range estimation
        returns: Actual returns data for range estimation
        n_samples: Number of bootstrap samples for range estimation
        bootstrap_fraction: Fraction of data to use in each bootstrap sample
        
    Returns:
        (min_score, max_score) tuple based on actual data distribution
    """
    if len(signal) == 0 or len(returns) == 0:
        return (0.0, 1.0)  # Fallback range
    
    scores = []
    n_obs = len(signal)
    sample_size = max(10, int(n_obs * bootstrap_fraction))
    
    # Bootstrap sampling with reproducible seeds
    for i in range(n_samples):
        try:
            np.random.seed(i)  # Reproducible sampling
            indices = np.random.choice(n_obs, size=sample_size, replace=True)
            
            sample_signal = signal.iloc[indices] if hasattr(signal, 'iloc') else signal[indices]
            sample_returns = returns.iloc[indices] if hasattr(returns, 'iloc') else returns[indices]
            
            score = objective_fn(sample_signal, sample_returns)
            if np.isfinite(score):
                scores.append(score)
        except:
            continue  # Skip problematic samples
    
    if len(scores) == 0:
        return (0.0, 1.0)  # Fallback range
    
    scores = np.array(scores)
    
    # Use 10th and 90th percentiles for robust range estimation
    min_score = float(np.percentile(scores, 10))
    max_score = float(np.percentile(scores, 90))
    
    # Ensure valid range (min < max)
    if min_score >= max_score:
        score_mean = np.mean(scores)
        range_width = max(0.1, np.std(scores) * 2.0)  # Use 2-sigma range
        min_score = score_mean - range_width / 2
        max_score = score_mean + range_width / 2
    
    return (min_score, max_score)

def create_scale_aware_composite_objective(
    objectives_config: Dict[str, Dict],
    auto_normalize: bool = True,
    calibration_data: Optional[Tuple[pd.Series, pd.Series]] = None
) -> Callable[[pd.Series, pd.Series], float]:
    """
    NEW FEATURE: Create composite objective with dynamic scale normalization.
    
    Enables seamless combination of different objective functions by automatically
    normalizing them to [0,1] range using actual data distribution instead of hard-coded ranges.
    
    Args:
        objectives_config: Dict of {obj_name: {weight: float, auto_normalize: bool, ...}}
        auto_normalize: Global flag to enable/disable auto-normalization
        calibration_data: Optional (signal, returns) for pre-computing normalization ranges
        
    Returns:
        Composite objective function
        
    Example:
        config = {
            'dapy_hits': {'weight': 1.0, 'auto_normalize': True},
            'adjusted_sharpe': {'weight': 1.0, 'auto_normalize': True},  # Equal weights work!
            'information_ratio': {'weight': 2.0, 'auto_normalize': True}
        }
        composite_fn = create_scale_aware_composite_objective(config)
    """
    from metrics.objective_registry import OBJECTIVE_REGISTRY
    
    # Pre-compute ranges for normalization
    precomputed_ranges = {}
    if calibration_data and auto_normalize:
        signal_cal, returns_cal = calibration_data
        for obj_name, config in objectives_config.items():
            if config.get('auto_normalize', auto_normalize) and 'custom_range' not in config:
                try:
                    obj_fn = OBJECTIVE_REGISTRY.get(obj_name)
                    precomputed_ranges[obj_name] = estimate_dynamic_range(obj_fn, signal_cal, returns_cal)
                except:
                    pass  # Skip failed pre-computation
    
    def composite_objective(signal: pd.Series, returns: pd.Series, **kwargs) -> float:
        total_score = 0.0
        total_weight = 0.0
        
        for obj_name, config in objectives_config.items():
            weight = config.get('weight', 1.0)
            use_normalize = config.get('auto_normalize', auto_normalize)
            
            try:
                obj_fn = OBJECTIVE_REGISTRY.get(obj_name)
                raw_score = obj_fn(signal, returns, **kwargs)
                
                # Apply normalization if enabled
                if use_normalize:
                    data_range = (
                        config.get('custom_range') or
                        precomputed_ranges.get(obj_name) or
                        estimate_dynamic_range(obj_fn, signal, returns)
                    )
                    final_score = normalize_objective_score(raw_score, obj_name, data_range=data_range)
                else:
                    final_score = raw_score
                
                total_score += weight * final_score
                total_weight += weight
                
            except Exception as e:
                warnings.warn(f"Error in objective '{obj_name}': {e}")
                continue
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    composite_objective.__doc__ = (
        f"Dynamic scale-aware composite objective combining: {list(objectives_config.keys())}\n"
        f"Auto-normalization: {'Enabled (Dynamic)' if auto_normalize else 'Disabled'}"
    )
    
    return composite_objective

# Example usage and testing functions
def test_dynamic_normalization():
    """Test the dynamic scale normalization functionality"""
    from metrics.objective_registry import OBJECTIVE_REGISTRY
    
    # Create test data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    signal = pd.Series(np.random.normal(0, 0.5, 200), index=dates)
    returns = pd.Series(np.random.normal(0, 0.01, 200), index=dates)
    
    print("=== DYNAMIC SCALE NORMALIZATION TEST ===")
    print("Raw scores vs Dynamically Normalized scores:")
    
    for obj_name in OBJECTIVE_REGISTRY.list_functions():
        try:
            obj_fn = OBJECTIVE_REGISTRY.get(obj_name)
            raw_score = obj_fn(signal, returns)
            data_range = estimate_dynamic_range(obj_fn, signal, returns)
            norm_score = normalize_objective_score(raw_score, obj_name, data_range=data_range)
            
            print(f"{obj_name:>25}: {raw_score:>8.4f} → {norm_score:>6.4f} (range: {data_range[0]:.3f} to {data_range[1]:.3f})")
            
        except Exception as e:
            print(f"{obj_name:>25}: ERROR - {e}")
    
    print("\n=== DYNAMIC COMPOSITE OBJECTIVE TEST ===")
    
    # Test normalized composite objective
    config = {
        'dapy_hits': {'weight': 1.0, 'auto_normalize': True},
        'adjusted_sharpe': {'weight': 1.0, 'auto_normalize': True},
        'information_ratio': {'weight': 1.0, 'auto_normalize': True}
    }
    
    composite_fn = create_scale_aware_composite_objective(config, calibration_data=(signal, returns))
    composite_score = composite_fn(signal, returns)
    print(f"Composite score (dynamic normalized): {composite_score:.4f}")
    
    # Test raw composite for comparison
    config_raw = {k: {**v, 'auto_normalize': False} for k, v in config.items()}
    composite_fn_raw = create_scale_aware_composite_objective(config_raw)
    composite_score_raw = composite_fn_raw(signal, returns)
    print(f"Composite score (raw scales):       {composite_score_raw:.4f}")
    
    print("\nNEW FEATURE: Dynamic normalization uses actual data distribution")
    print("             No more hard-coded ranges - adapts to any dataset!")
    print("             Normalized version gives balanced contribution from all objectives")

if __name__ == "__main__":
    test_dynamic_normalization()
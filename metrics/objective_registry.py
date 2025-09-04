"""
Objective Function Registry for XGBoost Trading System

Provides a centralized registry for different objective functions that can be used
for driver selection and weight optimization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from metrics.dapy import dapy_from_binary_hits, dapy_eri_long, dapy_eri_short, dapy_eri_both
from metrics.simple_metrics import information_ratio, sharpe_ratio, max_drawdown, compute_adjusted_sharpe, cb_ratio
from metrics.predictive_objective import predictive_icir_logscore


class ObjectiveRegistry:
    """Registry for objective functions used in driver selection and weight optimization."""
    
    def __init__(self):
        self._functions = {}
        self._register_builtin_functions()
    
    def _register_builtin_functions(self):
        """Register built-in objective functions."""
        
        # DAPY variants
        self.register("dapy_hits", dapy_from_binary_hits)
        self.register("dapy_eri_both", dapy_eri_both)
        
        # Information Ratio
        self.register("information_ratio", information_ratio)
        
        # Advanced metrics
        self.register("adjusted_sharpe", self._adjusted_sharpe_wrapper)
        self.register("cb_ratio", self._cb_ratio_wrapper)
        self.register("predictive_icir_logscore", predictive_icir_logscore)
    
    def _adjusted_sharpe_wrapper(self, signal: pd.Series, returns: pd.Series, **kwargs) -> float:
        """Wrapper for adjusted Sharpe ratio."""
        sharpe = sharpe_ratio(signal, returns)
        num_years = len(signal) / 252.0  # Assume daily data
        num_points = len(signal)
        adj_sharpe_n = kwargs.get('adj_sharpe_n', 10)  # Default number of tests
        return compute_adjusted_sharpe(sharpe, num_years, num_points, adj_sharpe_n)
    
    def _cb_ratio_wrapper(self, signal: pd.Series, returns: pd.Series, **kwargs) -> float:
        """Wrapper for CB ratio."""
        sharpe = sharpe_ratio(signal, returns)
        max_dd = max_drawdown(signal, returns)
        l1_penalty = kwargs.get('l1_penalty', 0.0)
        weights = kwargs.get('weights', None)
        return cb_ratio(sharpe, max_dd, l1_penalty, weights)
    
    def register(self, name: str, function: Callable):
        """Register a new objective function."""
        self._functions[name] = function
    
    def get(self, name: str) -> Callable:
        """Get an objective function by name."""
        if name not in self._functions:
            raise ValueError(f"Unknown objective function: {name}. Available: {list(self._functions.keys())}")
        return self._functions[name]
    
    def list_functions(self) -> list:
        """List all available objective functions."""
        return list(self._functions.keys())


def create_composite_objective(config: Dict[str, Any], registry: ObjectiveRegistry) -> Callable:
    """
    Create a composite objective function from configuration.
    
    NEW FEATURE: Supports automatic scale normalization to seamlessly combine
    objectives with vastly different scales (e.g., DAPY ~20 units vs Adjusted Sharpe ~0.4 units).
    
    Args:
        config: Configuration dict with metric names and parameters
        registry: ObjectiveRegistry instance
        
    Returns:
        Callable that computes the composite objective
        
    Example config:
        {
            'dapy': {'dapy_style': 'hits', 'weight': 1.0, 'auto_normalize': True},
            'information_ratio': {'weight': 1.0, 'auto_normalize': True}, 
            'adjusted_sharpe': {'weight': 1.0, 'auto_normalize': True}  # Equal weights now work!
        }
        
    Traditional config (manual scale weighting):
        {
            'dapy': {'dapy_style': 'hits', 'weight': 1.0},
            'adjusted_sharpe': {'weight': 50.0}  # Manual scale compensation
        }
    """
    if isinstance(config, str):
        # Simple case: single objective function
        return registry.get(config)
    
    if not isinstance(config, dict):
        raise ValueError("Objective config must be a string or dict")
    
    def composite_objective(signal: pd.Series, returns: pd.Series, **kwargs) -> float:
        # Import here to avoid circular imports
        from metrics.scale_normalization import normalize_objective_score
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric_name, metric_config in config.items():
            if isinstance(metric_config, dict):
                weight = metric_config.get('weight', 1.0)
                auto_normalize = metric_config.get('auto_normalize', False)  # NEW FEATURE
                custom_range = metric_config.get('custom_range', None)       # NEW FEATURE
                metric_kwargs = {k: v for k, v in metric_config.items() 
                               if k not in ['weight', 'auto_normalize', 'custom_range']}
            else:
                weight = 1.0
                auto_normalize = False
                custom_range = None
                metric_kwargs = {}
            
            try:
                # Handle special cases for DAPY variants
                if metric_name == 'dapy':
                    dapy_style = metric_kwargs.get('dapy_style', 'hits')
                    if dapy_style == 'hits':
                        func = registry.get('dapy_hits')
                        obj_name = 'dapy_hits'
                    elif dapy_style == 'eri_both':
                        func = registry.get('dapy_eri_both')
                        obj_name = 'dapy_eri_both'
                    else:
                        func = registry.get('dapy_hits')  # fallback
                        obj_name = 'dapy_hits'
                    
                    # Remove dapy_style from kwargs for DAPY functions
                    metric_kwargs_clean = {k: v for k, v in metric_kwargs.items() if k != 'dapy_style'}
                    all_kwargs = {**kwargs, **metric_kwargs_clean}
                else:
                    func = registry.get(metric_name)
                    obj_name = metric_name
                    all_kwargs = {**kwargs, **metric_kwargs}
                
                # Calculate raw score
                raw_score = func(signal, returns, **all_kwargs)
                
                # Apply dynamic automatic scale normalization if enabled
                if auto_normalize:
                    custom_range = metric_config.get('custom_range', None) if isinstance(metric_config, dict) else None
                    if custom_range is not None:
                        final_score = normalize_objective_score(raw_score, obj_name, custom_range=custom_range)
                    else:
                        # Use dynamic range estimation from actual data
                        from metrics.scale_normalization import estimate_dynamic_range
                        data_range = estimate_dynamic_range(func, signal, returns)
                        final_score = normalize_objective_score(raw_score, obj_name, data_range=data_range)
                else:
                    final_score = raw_score
                
                # Add weighted contribution
                total_score += weight * final_score
                total_weight += weight
                
            except Exception as e:
                import warnings
                warnings.warn(f"Error in objective '{metric_name}': {e}")
                continue
        
        # Return weighted average if any objectives succeeded
        return total_score / total_weight if total_weight > 0 else 0.0
    
    return composite_objective


# Global registry instance
OBJECTIVE_REGISTRY = ObjectiveRegistry()
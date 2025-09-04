"""
P-value gating module for statistical significance testing.
"""
from typing import Callable, Optional
from eval.target_shuffling import shuffle_pvalue
from metrics.objective_registry import OBJECTIVE_REGISTRY


def apply_pvalue_gating(signal, y_local, args, objective_fn: Optional[Callable] = None) -> bool:
    """
    Apply p-value gating to determine if signal passes statistical significance test.
    
    Args:
        signal: Signal series to test
        y_local: Target returns
        args: Command line arguments containing gating configuration
        objective_fn: Optional objective function, uses args.p_value_gating if None
        
    Returns:
        bool: True if signal passes gating (or gating is bypassed), False otherwise
    """
    # Bypass gating if explicitly requested
    if args.bypass_pvalue_gating:
        return True
    
    # Enable gating if pmax is set (even without explicit p_value_gating)
    if hasattr(args, 'pmax') and args.pmax is not None:
        # Use default objective if p_value_gating not specified
        pass
    elif not hasattr(args, 'p_value_gating') or args.p_value_gating is None:
        # No gating configuration - bypass
        return True
    
    # Determine objective function
    if objective_fn is None:
        if hasattr(args, 'p_value_gating') and args.p_value_gating is not None:
            if args.p_value_gating == 'dapy':
                from metrics.dapy import dapy_from_binary_hits
                objective_fn = dapy_from_binary_hits
            else:
                from metrics.dapy import dapy_from_binary_hits
                objective_fn = OBJECTIVE_REGISTRY.get(args.p_value_gating, dapy_from_binary_hits)
        else:
            # Default to DAPY when pmax is set but p_value_gating is not specified
            from metrics.dapy import dapy_from_binary_hits
            objective_fn = dapy_from_binary_hits
    
    # Run significance test
    pval, _, _ = shuffle_pvalue(signal, y_local, objective_fn, n_shuffles=200, block=args.block)
    return pval <= args.pmax
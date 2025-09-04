
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Optional, Dict, Any
from metrics.simple_metrics import information_ratio


def pick_top_n_greedy_diverse(
    train_signals: List[pd.Series],
    y_tr: pd.Series,
    n: int,
    pval_gate: Callable[[pd.Series, pd.Series], bool],
    objective_fn: Callable,
    diversity_penalty: float = 0.2,
) -> List[int]:
    """
    Simplified greedy diverse driver selection algorithm.
    
    Logic (same as before, but cleaner implementation):
    1. Pre-compute correlation matrix between all signals
    2. Greedy loop: Pick signal with highest (objective_score - diversity_penalty)
    3. Diversity penalty = max correlation with already-selected signals
    4. Continue until n signals selected or no more valid signals
    
    Args:
        train_signals: List of candidate signals to select from
        y_tr: Target returns for evaluation
        n: Number of signals to select  
        pval_gate: Statistical significance filter
        objective_fn: Objective function for driver scoring
        diversity_penalty: Penalty for correlation with selected signals
        
    Returns:
        List of indices of selected signals
    """
    num_signals = len(train_signals)
    selected_indices = []
    available_indices = set(range(num_signals))
    
    # Pre-compute correlation matrix (same logic as before)
    correlation_matrix = np.zeros((num_signals, num_signals))
    for i in range(num_signals):
        for j in range(i, num_signals):
            # Handle constant signals gracefully
            if (np.std(train_signals[i].values) < 1e-10 or 
                np.std(train_signals[j].values) < 1e-10):
                corr_val = 0.0
            else:
                corr_val = np.corrcoef(train_signals[i].values, train_signals[j].values)[0,1]
                if not np.isfinite(corr_val):
                    corr_val = 0.0
            correlation_matrix[i, j] = correlation_matrix[j, i] = corr_val
    
    # Greedy selection loop (same logic as before)
    for _ in range(n):
        if not available_indices:
            break
            
        best_score = -1e18
        best_index = None
        
        for candidate_idx in available_indices:
            signal = train_signals[candidate_idx]
            
            # Apply statistical significance gate (same as before)
            if not pval_gate(signal, y_tr):
                continue
                
            # Calculate base objective score
            base_score = objective_fn(signal, y_tr)
            
            # Calculate diversity penalty (same logic as before)
            if len(selected_indices) == 0:
                diversity_penalty_val = 0.0
            else:
                # Penalty = max correlation with any already-selected signal
                correlations_with_selected = [abs(correlation_matrix[candidate_idx, sel_idx]) 
                                            for sel_idx in selected_indices]
                diversity_penalty_val = max(correlations_with_selected)
            
            # Final score = base - penalty (same formula as before)
            final_score = base_score - diversity_penalty * diversity_penalty_val
            
            if final_score > best_score:
                best_score = final_score
                best_index = candidate_idx
        
        # Add best signal to selection (same logic as before)
        if best_index is not None:
            selected_indices.append(best_index)
            available_indices.remove(best_index)
        else:
            break  # No more valid signals
    
    return selected_indices

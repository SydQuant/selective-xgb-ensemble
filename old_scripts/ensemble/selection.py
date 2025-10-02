
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Callable, Optional, Dict, Any
from metrics.simple_metrics import information_ratio

# Setup logging
logger = logging.getLogger(__name__)


def pick_top_n_greedy_diverse(
    train_signals: List[pd.Series],
    y_tr: pd.Series,
    n: int,
    pval_gate: Callable[[pd.Series, pd.Series], bool],
    objective_fn: Callable,
    diversity_penalty: float = 0.2,
    objective_name: str = "unknown",
) -> Tuple[List[int], Dict[str, Any]]:
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
        Tuple of (selected indices, diagnostics dict)
    """
    num_signals = len(train_signals)
    selected_indices = []
    available_indices = set(range(num_signals))
    
    # Initialize diagnostics tracking
    diagnostics = {
        'objective_name': objective_name,
        'total_signals': num_signals,
        'pvalue_passed': 0,
        'pvalue_failed': 0,
        'selected_indices': [],
        'selected_objective_scores': [],
        'selected_diversity_penalties': [],
        'selected_final_scores': [],
        'min_objective_score': None,
        'max_objective_score': None,
        'pvalue_pass_rate': 0.0,
        'objective_threshold': None,
        'pvalue_failures': []  # Track which models failed p-value gating
    }
    
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
    
    # Pre-compute p-value results for all models (do this once, not in greedy loop)
    pval_results = {}
    for idx in range(num_signals):
        signal = train_signals[idx]
        pval_passed = pval_gate(signal, y_tr)
        pval_results[idx] = pval_passed
        
        # Count p-value results (once per model, not per greedy iteration)
        if pval_passed:
            diagnostics['pvalue_passed'] += 1
        else:
            diagnostics['pvalue_failed'] += 1
            diagnostics['pvalue_failures'].append(idx)
    
    # Greedy selection loop (same logic as before)
    for _ in range(n):
        if not available_indices:
            break
            
        best_score = -1e18
        best_index = None
        
        for candidate_idx in available_indices:
            signal = train_signals[candidate_idx]
            
            # Check pre-computed p-value result
            if not pval_results[candidate_idx]:
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
        
        # Add best signal to selection and record diagnostics
        if best_index is not None:
            selected_indices.append(best_index)
            available_indices.remove(best_index)
            
            # Record detailed diagnostics for selected model
            signal = train_signals[best_index]
            obj_score = objective_fn(signal, y_tr)
            
            # Calculate diversity penalty for this selection
            if len(selected_indices) == 1:
                div_penalty = 0.0
            else:
                prev_selected = selected_indices[:-1]  # All except the just-added one
                correlations = [correlation_matrix[best_index, idx] for idx in prev_selected]
                div_penalty = max(correlations) if correlations else 0.0
            
            final_score = obj_score - diversity_penalty * div_penalty
            
            diagnostics['selected_indices'].append(best_index)
            diagnostics['selected_objective_scores'].append(obj_score)
            diagnostics['selected_diversity_penalties'].append(div_penalty)
            diagnostics['selected_final_scores'].append(final_score)
        else:
            break  # No more valid signals
    
    # Compute final diagnostics
    if diagnostics['selected_objective_scores']:
        diagnostics['min_objective_score'] = min(diagnostics['selected_objective_scores'])
        diagnostics['max_objective_score'] = max(diagnostics['selected_objective_scores'])
        diagnostics['objective_threshold'] = diagnostics['min_objective_score']
    
    total_evaluated = diagnostics['pvalue_passed'] + diagnostics['pvalue_failed']
    if total_evaluated > 0:
        diagnostics['pvalue_pass_rate'] = diagnostics['pvalue_passed'] / total_evaluated
    
    # Log detailed diagnostics
    logger.info(f"🎯 Driver Selection Diagnostics ({objective_name}):")
    logger.info(f"  📊 Total signals: {diagnostics['total_signals']}")
    logger.info(f"  ✅ P-value passed: {diagnostics['pvalue_passed']}/{total_evaluated} ({diagnostics['pvalue_pass_rate']:.1%})")
    logger.info(f"  ❌ P-value failed: {diagnostics['pvalue_failed']} models {diagnostics['pvalue_failures'][:10]}...")
    logger.info(f"  🔍 Selected models: {len(selected_indices)} → {selected_indices}")
    # Handle objective threshold and range logging defensively
    threshold = diagnostics.get('objective_threshold')
    min_score = diagnostics.get('min_objective_score') 
    max_score = diagnostics.get('max_objective_score')
    
    if threshold is not None and min_score is not None and max_score is not None:
        logger.info(f"  📈 Objective ({objective_name}) threshold: {threshold:.4f}")
        logger.info(f"  📊 Objective range: [{min_score:.4f}, {max_score:.4f}]")
    else:
        logger.info(f"  📈 Objective ({objective_name}) threshold: N/A (no models selected)")
        logger.info(f"  📊 Objective range: N/A (no models selected)")
    
    # Log detailed selected model information
    if len(selected_indices) > 0:
        logger.info(f"  📋 Selected Model Details:")
        for i, (idx, obj_score, div_penalty, final_score) in enumerate(zip(
            diagnostics['selected_indices'], 
            diagnostics['selected_objective_scores'],
            diagnostics['selected_diversity_penalties'],
            diagnostics['selected_final_scores']
        )):
            logger.info(f"    #{i+1}: Model[{idx}] | {objective_name}={obj_score:.4f} | DivPenalty={div_penalty:.3f} | Final={final_score:.4f}")
    
    return selected_indices, diagnostics

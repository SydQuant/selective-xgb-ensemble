"""
Out-of-Sample Diagnostics Module for XGBoost Ensemble

This module provides comprehensive diagnostics for all 75 models on out-of-sample data,
including individual p-values and performance metrics for each fold.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Callable, Dict, Any
from eval.target_shuffling import shuffle_pvalue
from metrics.simple_metrics import information_ratio, sharpe_ratio
from metrics.dapy import hit_rate

# Setup logging
logger = logging.getLogger(__name__)


def compute_oos_model_diagnostics(
    test_signals: List[pd.Series],
    y_test: pd.Series,
    objective_fn: Callable,
    objective_name: str = "unknown",
    fold_id: int = None,
    n_shuffles: int = 200,
    show_top_n: int = 75
) -> Dict[str, Any]:
    """
    Comprehensive OOS diagnostics for all models in a fold.
    
    Args:
        test_signals: List of OOS test signals (75 models)
        y_test: OOS test returns
        objective_fn: Objective function for scoring
        objective_name: Name of objective function
        fold_id: Fold number for logging
        n_shuffles: Number of shuffles for p-value calculation
        show_top_n: Number of top models to display in logs
    
    Returns:
        Dictionary containing all model diagnostics
    """
    num_signals = len(test_signals)
    model_diagnostics = []
    
    fold_str = f" - FOLD {fold_id}" if fold_id is not None else ""
    logger.info(f"🔍 COMPREHENSIVE OOS MODEL DIAGNOSTICS - ALL {num_signals} MODELS{fold_str}")
    logger.info(f"📊 OOS Analysis: Computing p-values and performance metrics for each model...")
    
    # Calculate diagnostics for all models
    for idx in range(num_signals):
        signal = test_signals[idx]
        
        try:
            # Calculate actual p-value using shuffle test on OOS data
            pval, observed_metric, _ = shuffle_pvalue(signal, y_test, objective_fn, n_shuffles=n_shuffles, block=10)
        except:
            pval = 1.0
            observed_metric = 0.0
        
        # Calculate comprehensive performance metrics on OOS data
        obj_score = objective_fn(signal, y_test)
        ir_score = information_ratio(signal, y_test)
        sharpe_score = sharpe_ratio(signal, y_test)
        hr_score = hit_rate(signal, y_test)
        
        # Calculate signal statistics
        signal_std = signal.std()
        signal_mean = signal.mean()
        signal_min = signal.min()
        signal_max = signal.max()
        
        # Calculate PnL statistics
        pnl = signal.shift(1).fillna(0.0) * y_test.reindex_like(signal)
        pnl_mean = pnl.mean() * 252  # Annualized
        pnl_std = pnl.std() * np.sqrt(252)  # Annualized
        pnl_sharpe = pnl_mean / pnl_std if pnl_std > 0 else 0.0
        max_dd = -(pnl.cumsum() - pnl.cumsum().expanding().max()).min()
        
        model_diag = {
            'model_id': idx,
            'fold_id': fold_id,
            'objective_score': obj_score,
            'p_value': pval,
            'p_value_passed': pval <= 0.05,  # Using 5% threshold
            'information_ratio': ir_score,
            'sharpe_ratio': sharpe_score,
            'hit_rate': hr_score,
            'signal_mean': signal_mean,
            'signal_std': signal_std,
            'signal_min': signal_min,
            'signal_max': signal_max,
            'observed_metric': observed_metric,
            'pnl_annualized_return': pnl_mean,
            'pnl_annualized_vol': pnl_std,
            'pnl_sharpe': pnl_sharpe,
            'max_drawdown': max_dd,
            'num_trades': len(signal.dropna()),
            'winning_trades': len(pnl[pnl > 0]),
            'losing_trades': len(pnl[pnl < 0])
        }
        
        model_diagnostics.append(model_diag)
    
    # Sort by objective score for display
    model_diagnostics_sorted = sorted(model_diagnostics, key=lambda x: x['objective_score'], reverse=True)
    
    # Count p-value results
    p_passed = sum(1 for d in model_diagnostics if d['p_value_passed'])
    p_failed = num_signals - p_passed
    
    # Log detailed model diagnostics table
    logger.info(f"📊 P-Value Summary: {p_passed}/{num_signals} models passed")
    logger.info(f"📋 ALL {num_signals} MODELS OOS PERFORMANCE BREAKDOWN (sorted by {objective_name}):")
    logger.info(f"     Model | {objective_name:<12} | P-Value | Sharpe |  PnL% | Hit%  | MaxDD | Status")
    logger.info(f"     ----- | {'-'*12} | ------- | ------ | ----- | ----- | ----- | ------")
    
    for i, diag in enumerate(model_diagnostics_sorted):
        status = "✅PASS" if diag['p_value_passed'] else "❌FAIL"
        pnl_pct = diag['pnl_annualized_return'] * 100
        dd_pct = diag['max_drawdown'] * 100
        logger.info(f"     [{diag['model_id']:2d}] | {diag['objective_score']:12.4f} | {diag['p_value']:7.4f} | {diag['sharpe_ratio']:6.3f} | {pnl_pct:4.1f} | {diag['hit_rate']*100:4.1f} | {dd_pct:4.1f} | {status}")
    
    # Always show all models now
    # if len(model_diagnostics_sorted) > show_top_n:
    #     logger.info(f"     ... (showing top {show_top_n} of {len(model_diagnostics_sorted)} models)")
    
    # Log summary statistics
    avg_sharpe = np.mean([d['pnl_sharpe'] for d in model_diagnostics])
    avg_hit_rate = np.mean([d['hit_rate'] for d in model_diagnostics])
    avg_p_value = np.mean([d['p_value'] for d in model_diagnostics])
    
    logger.info(f"📈 AVERAGE PERFORMANCE ACROSS ALL {num_signals} MODELS:")
    logger.info(f"     Average Sharpe: {avg_sharpe:.3f}")
    logger.info(f"     Average Hit Rate: {avg_hit_rate:.1%}")
    logger.info(f"     Average P-Value: {avg_p_value:.4f}")
    logger.info(f"     Models with Sharpe > 0: {sum(1 for d in model_diagnostics if d['pnl_sharpe'] > 0)}/{num_signals}")
    logger.info(f"     Models with Hit Rate > 50%: {sum(1 for d in model_diagnostics if d['hit_rate'] > 0.5)}/{num_signals}")
    
    return {
        'all_model_diagnostics': model_diagnostics,
        'model_diagnostics_sorted': model_diagnostics_sorted,
        'summary_stats': {
            'avg_sharpe': avg_sharpe,
            'avg_hit_rate': avg_hit_rate,
            'avg_p_value': avg_p_value,
            'p_value_passed': p_passed,
            'p_value_failed': p_failed,
            'positive_sharpe_count': sum(1 for d in model_diagnostics if d['pnl_sharpe'] > 0),
            'positive_hit_rate_count': sum(1 for d in model_diagnostics if d['hit_rate'] > 0.5)
        }
    }
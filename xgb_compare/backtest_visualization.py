#!/usr/bin/env python3
"""
Enhanced backtesting visualization for model selection tracking and PnL curves.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def create_model_selection_timeline(backtest_results: Dict, save_path: str) -> str:
    """
    Create a timeline visualization showing which models were selected at each fold.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    if 'model_selection_history' not in backtest_results or not backtest_results['model_selection_history']:
        logger.warning("No model selection history available")
        return save_path
    
    selection_history = backtest_results['model_selection_history']
    
    # Top plot: Model selection timeline
    folds = [info['fold'] for info in selection_history]
    
    # Create a matrix showing which models were selected when
    all_models = set()
    for info in selection_history:
        all_models.update(info['selected_models'])
    
    all_models = sorted(list(all_models))
    selection_matrix = np.zeros((len(all_models), len(folds)))
    q_scores_matrix = np.zeros((len(all_models), len(folds)))
    
    for fold_idx, info in enumerate(selection_history):
        for model_idx in info['selected_models']:
            if model_idx in all_models:
                row_idx = all_models.index(model_idx)
                selection_matrix[row_idx, fold_idx] = 1
                # Get Q-score if available
                model_pos = info['selected_models'].index(model_idx)
                if 'q_scores' in info and model_pos < len(info['q_scores']):
                    q_scores_matrix[row_idx, fold_idx] = info['q_scores'][model_pos]
    
    # Create heatmap for model selection
    im1 = ax1.imshow(selection_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
    
    # Customize the plot
    ax1.set_xticks(range(len(folds)))
    ax1.set_xticklabels([f"Fold {f}" for f in folds])
    ax1.set_yticks(range(len(all_models)))
    ax1.set_yticklabels([f"M{m:02d}" for m in all_models])
    ax1.set_title('Model Selection Timeline\n(Red = Selected, Blue = Not Selected)')
    ax1.set_xlabel('Fold Number')
    ax1.set_ylabel('Model ID')
    
    # Add text annotations for Q-scores
    for i in range(len(all_models)):
        for j in range(len(folds)):
            if selection_matrix[i, j] == 1:
                q_score = q_scores_matrix[i, j]
                ax1.text(j, i, f'{q_score:.2f}', ha='center', va='center', 
                        color='white', fontweight='bold', fontsize=8)
    
    # Bottom plot: Average Q-scores per fold
    avg_q_scores = [info['avg_q_score'] for info in selection_history]
    ax2.plot(folds, avg_q_scores, marker='o', linewidth=2, markersize=6, color='blue')
    ax2.set_xlabel('Fold Number')
    ax2.set_ylabel(f"Average Q-Score ({selection_history[0].get('q_metric_used', 'sharpe')})")
    ax2.set_title('Average Q-Score of Selected Models Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for fold, q_score in zip(folds, avg_q_scores):
        ax2.annotate(f'{q_score:.3f}', (fold, q_score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model selection timeline saved to: {save_path}")
    return save_path

def create_production_pnl_curve(backtest_results: Dict, save_path: str, 
                               dates: Optional[List] = None) -> str:
    """
    Create PnL curve with model selection markers and performance metrics.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))
    
    if 'production_returns' not in backtest_results or not backtest_results['production_returns']:
        logger.warning("No production returns available for PnL curve")
        return save_path
    
    returns = pd.Series(backtest_results['production_returns'])
    predictions = pd.Series(backtest_results.get('production_predictions', []))
    
    # Calculate cumulative PnL
    cumulative_pnl = returns.cumsum()
    
    # Top plot: Cumulative PnL curve
    if dates and len(dates) == len(cumulative_pnl):
        ax1.plot(dates, cumulative_pnl.values, linewidth=2, color='blue', label='Cumulative PnL')
        ax1.set_xlabel('Date')
    else:
        ax1.plot(cumulative_pnl.values, linewidth=2, color='blue', label='Cumulative PnL')
        ax1.set_xlabel('Trading Period')
    
    ax1.set_ylabel('Cumulative PnL')
    ax1.set_title('Production Backtesting - Cumulative PnL Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add model selection markers if available
    if 'model_selection_history' in backtest_results:
        selection_history = backtest_results['model_selection_history']
        fold_results = backtest_results.get('fold_results', [])
        
        # Map folds to approximate positions in PnL curve
        for selection in selection_history:
            fold_num = selection['fold']
            # Find corresponding fold in backtest results
            fold_data = next((f for f in fold_results if f['fold'] == fold_num + 1), None)
            if fold_data:
                # Estimate position in PnL curve (approximate)
                pos = len(cumulative_pnl) * (fold_num / len(selection_history))
                pos = min(int(pos), len(cumulative_pnl) - 1)
                
                # Add vertical line and annotation
                ax1.axvline(x=pos, color='red', linestyle='--', alpha=0.7)
                models_str = ', '.join([f'M{m:02d}' for m in selection['selected_models']])
                ax1.annotate(f'F{fold_num+1}: {models_str}', 
                           xy=(pos, cumulative_pnl.iloc[pos]), 
                           xytext=(10, 20), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                           fontsize=8)
    
    # Middle plot: Rolling performance metrics
    window_size = max(21, len(returns) // 10)  # Adaptive window
    rolling_sharpe = returns.rolling(window_size).apply(
        lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0
    )
    # Calculate rolling hit rate more safely
    pred_signs = np.sign(predictions)
    ret_signs = np.sign(returns)
    correct_predictions = (pred_signs == ret_signs).astype(float)
    rolling_hit_rate = correct_predictions.rolling(window_size).mean()
    
    ax2.plot(rolling_sharpe.values, label=f'Rolling Sharpe ({window_size}d)', color='green', linewidth=1.5)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(rolling_hit_rate.values, label=f'Rolling Hit Rate ({window_size}d)', 
                 color='orange', linewidth=1.5)
    
    ax2.set_xlabel('Trading Period')
    ax2.set_ylabel('Rolling Sharpe', color='green')
    ax2_twin.set_ylabel('Rolling Hit Rate', color='orange')
    ax2.set_title('Rolling Performance Metrics')
    ax2.grid(True, alpha=0.3)
    
    # Bottom plot: Drawdown
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    
    ax3.fill_between(range(len(drawdown)), drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
    ax3.plot(drawdown.values, color='red', linewidth=1)
    ax3.set_xlabel('Trading Period')
    ax3.set_ylabel('Drawdown')
    ax3.set_title('Drawdown Analysis')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Production PnL curve saved to: {save_path}")
    return save_path

def create_fold_performance_breakdown(backtest_results: Dict, save_path: str) -> str:
    """
    Create detailed breakdown of performance by fold with model details.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    if 'fold_results' not in backtest_results or not backtest_results['fold_results']:
        logger.warning("No fold results available for breakdown")
        return save_path
    
    fold_results = backtest_results['fold_results']
    
    # Extract data
    folds = [f['fold'] for f in fold_results]
    sharpe_values = [f['fold_metrics'].get('sharpe', 0) for f in fold_results]
    hit_rates = [f['fold_metrics'].get('hit_rate', 0) for f in fold_results]
    ann_returns = [f['fold_metrics'].get('ann_ret', 0) for f in fold_results]
    ann_vols = [f['fold_metrics'].get('ann_vol', 0) for f in fold_results]
    
    # Plot 1: Sharpe Ratio by Fold
    bars1 = axes[0].bar(folds, sharpe_values, color='blue', alpha=0.7)
    axes[0].set_xlabel('Fold Number')
    axes[0].set_ylabel('Sharpe Ratio')
    axes[0].set_title('OOS Sharpe Ratio by Fold')
    axes[0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, sharpe_values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Hit Rate by Fold
    bars2 = axes[1].bar(folds, hit_rates, color='green', alpha=0.7)
    axes[1].set_xlabel('Fold Number')
    axes[1].set_ylabel('Hit Rate')
    axes[1].set_title('OOS Hit Rate by Fold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    # Add value labels
    for bar, val in zip(bars2, hit_rates):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Risk-Return Scatter
    axes[2].scatter(ann_vols, ann_returns, c=folds, cmap='viridis', s=100, alpha=0.7)
    for i, fold in enumerate(folds):
        axes[2].annotate(f'F{fold}', (ann_vols[i], ann_returns[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    axes[2].set_xlabel('Annual Volatility')
    axes[2].set_ylabel('Annual Return')
    axes[2].set_title('Risk-Return Profile by Fold')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Model Usage Summary
    model_usage = {}
    for result in fold_results:
        for model in result['selected_models']:
            model_name = f"M{model:02d}"
            model_usage[model_name] = model_usage.get(model_name, 0) + 1
    
    if model_usage:
        models, counts = zip(*sorted(model_usage.items()))
        bars4 = axes[3].bar(models, counts, color='orange', alpha=0.7)
        axes[3].set_xlabel('Model ID')
        axes[3].set_ylabel('Selection Frequency')
        axes[3].set_title('Model Selection Frequency')
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars4, counts):
            axes[3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{val}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Fold performance breakdown saved to: {save_path}")
    return save_path

def create_comprehensive_backtest_report(backtest_results: Dict, save_dir: str) -> List[str]:
    """
    Create a comprehensive set of backtest visualizations.
    
    Returns:
        List of paths to generated visualization files
    """
    visualization_paths = []
    
    # Model selection timeline
    timeline_path = os.path.join(save_dir, "model_selection_timeline.png")
    visualization_paths.append(create_model_selection_timeline(backtest_results, timeline_path))
    
    # Production PnL curve
    pnl_path = os.path.join(save_dir, "production_pnl_curve.png")
    dates = backtest_results.get('production_dates', None)
    visualization_paths.append(create_production_pnl_curve(backtest_results, pnl_path, dates))
    
    # Fold performance breakdown
    breakdown_path = os.path.join(save_dir, "fold_performance_breakdown.png")
    visualization_paths.append(create_fold_performance_breakdown(backtest_results, breakdown_path))
    
    return visualization_paths

def log_detailed_backtest_summary(backtest_results: Dict, logger: logging.Logger):
    """
    Log a detailed text summary of backtest results with model tracking.
    """
    logger.info("="*100)
    logger.info("DETAILED PRODUCTION BACKTEST SUMMARY")
    logger.info("="*100)
    
    # Configuration
    config = backtest_results.get('configuration', {})
    logger.info(f"Configuration:")
    logger.info(f"  - Cutoff Fraction: {config.get('cutoff_fraction', 0.7):.0%}")
    logger.info(f"  - Top N Models: {config.get('top_n_models', 5)}")
    logger.info(f"  - Q-Metric: {config.get('q_metric', 'sharpe')}")
    logger.info(f"  - Reselection Frequency: {config.get('reselection_frequency', 1)} folds")
    logger.info("")
    
    # Overall performance
    if 'performance_metrics' in backtest_results:
        metrics = backtest_results['performance_metrics']
        logger.info("Overall Performance:")
        logger.info(f"  - Sharpe Ratio: {metrics.get('sharpe', 0):.3f}")
        logger.info(f"  - Annual Return: {metrics.get('ann_ret', 0):.2%}")
        logger.info(f"  - Annual Volatility: {metrics.get('ann_vol', 0):.2%}")
        logger.info(f"  - Hit Rate: {metrics.get('hit_rate', 0):.1%}")
        logger.info(f"  - CB Ratio: {metrics.get('cb_ratio', 0):.3f}")
        logger.info(f"  - Total Periods: {metrics.get('total_periods', 0)}")
        logger.info(f"  - Folds Traded: {metrics.get('folds_traded', 0)}")
        logger.info("")
    
    # Model selection history
    if 'model_selection_history' in backtest_results:
        logger.info("Model Selection History:")
        for selection in backtest_results['model_selection_history']:
            models_str = ', '.join([f"M{m:02d}" for m in selection['selected_models']])
            q_metric = selection.get('q_metric_used', 'sharpe')  # Default to sharpe if not found
            logger.info(f"  Fold {selection['fold']}: {models_str} "
                       f"(Avg Q-{q_metric}: {selection['avg_q_score']:.3f})")
        logger.info("")
    
    # Fold-by-fold performance
    if 'fold_results' in backtest_results:
        logger.info("Fold-by-Fold Performance:")
        for fold_result in backtest_results['fold_results']:
            fold_metrics = fold_result['fold_metrics']
            models_str = ', '.join([f"M{m:02d}" for m in fold_result['selected_models']])
            logger.info(f"  Fold {fold_result['fold']}: {models_str} | "
                       f"Sharpe={fold_metrics.get('sharpe', 0):.3f} | "
                       f"Hit={fold_metrics.get('hit_rate', 0):.1%} | "
                       f"Samples={fold_result['n_test_samples']}")
    
    logger.info("="*100)
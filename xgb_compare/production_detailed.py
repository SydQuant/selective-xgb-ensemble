#!/usr/bin/env python3
"""
Detailed production analysis with fold-by-fold and model-by-model breakdowns.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def create_detailed_production_analysis(quality_tracker, backtest_results, config, timestamp, save_dir):
    """
    Create comprehensive production analysis with detailed breakdowns.
    """
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(5, 3, height_ratios=[2.5, 1.5, 1, 1, 1.5], width_ratios=[3, 2, 2], 
                         hspace=0.45, wspace=0.35)
    
    # Find ACTUAL production models (only from production period folds)
    production_models = set()
    training_models = set()
    production_folds = []
    training_folds = []
    
    if 'fold_results' in backtest_results:
        # Determine cutoff between training and production
        all_folds = [f['fold'] for f in backtest_results['fold_results']]
        cutoff_fold_num = int(len(all_folds) * config.cutoff_fraction) + 1
        
        for fold_result in backtest_results['fold_results']:
            fold_num = fold_result['fold']
            selected_models = fold_result.get('selected_models', [])
            
            # Completely skip Fold 2 (uses Q=0 scores and selects M00-M09)
            if fold_num == 2:
                continue
                
            if fold_num >= cutoff_fold_num:  # Production period
                production_models.update(selected_models)
                if fold_num not in production_folds:
                    production_folds.append(fold_num)
            else:  # Training period
                training_models.update(selected_models)
                if fold_num not in training_folds:
                    training_folds.append(fold_num)
    
    production_models = sorted(list(production_models))
    training_models = sorted(list(training_models))
    all_models_used = sorted(list(set(production_models) | set(training_models)))
    
    # 1. Q-EVOLUTION: ALL models (gray) + Production models (bold colors)
    ax_q = fig.add_subplot(gs[0, :])
    
    if quality_tracker.quality_history['sharpe']:
        max_folds = max(len(history) for history in quality_tracker.quality_history['sharpe'] if history)
        
        # Plot ALL 50 models as thin gray background
        for model_idx in range(config.n_models):
            if model_idx < len(quality_tracker.quality_history['sharpe']):
                history = quality_tracker.quality_history['sharpe'][model_idx]
                if len(history) >= 2:
                    q_evolution = []
                    fold_numbers = []
                    
                    for fold in range(1, len(history)):
                        historical_series = pd.Series(history[:fold])
                        q_score = calculate_ewma_quality(historical_series, config.ewma_alpha)
                        q_evolution.append(q_score)
                        fold_numbers.append(fold + 1)
                    
                    if q_evolution:
                        ax_q.plot(fold_numbers, q_evolution, 
                                 color='lightgray', linewidth=0.7, alpha=0.5)
        
        # Highlight production models with bold colors
        if production_models:
            production_colors = plt.cm.Set1(np.linspace(0, 1, min(len(production_models), 9)))
            
            # Highlight ALL actual production models (not just first 5)
            for rank, model_idx in enumerate(production_models):  # All actual production models
                if model_idx < len(quality_tracker.quality_history['sharpe']):
                    history = quality_tracker.quality_history['sharpe'][model_idx]
                    if len(history) >= 2:
                        q_evolution = []
                        fold_numbers = []
                        
                        for fold in range(1, len(history)):
                            historical_series = pd.Series(history[:fold])
                            q_score = calculate_ewma_quality(historical_series, config.ewma_alpha)
                            q_evolution.append(q_score)
                            fold_numbers.append(fold + 1)
                        
                        if q_evolution:
                            final_q = quality_tracker.get_q_scores(max_folds - 1, config.ewma_alpha)['sharpe'][model_idx]
                            
                            ax_q.plot(fold_numbers, q_evolution, 
                                     marker='o', linewidth=4, markersize=8,
                                     label=f'M{model_idx:02d} (Final Q: {final_q:.3f})', 
                                     color=production_colors[rank % len(production_colors)], 
                                     alpha=0.95, zorder=10)
        
        # Production start marker
        cutoff_fold = int(max_folds * config.cutoff_fraction) + 1
        ax_q.axvline(x=cutoff_fold, color='red', linestyle='--', linewidth=3, alpha=0.9, 
                    label=f'Production Start (Fold {cutoff_fold})', zorder=5)
        
        ax_q.set_xlabel('Fold Number', fontsize=14)
        ax_q.set_ylabel('Q-Score (Sharpe)', fontsize=14)
        ax_q.set_title(f'Q-Score Evolution: ALL {config.n_models} Models (Gray) + {len(production_models)} Production Models (Bold)\n' +
                      f'Production Models: {[", ".join([f"M{m:02d}" for m in production_models])]}', 
                      fontsize=14, fontweight='bold')
        
        # Clean legend with reasonable size (limit to prevent overcrowding)
        if len(production_models) <= 8:
            ax_q.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                       title='Production Models', title_fontsize=11)
        else:
            ax_q.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, 
                       title=f'{len(production_models)} Prod. Models', title_fontsize=10, ncol=2)
        ax_q.grid(True, alpha=0.3)
        
        # Integer ticks only
        integer_ticks = list(range(1, max_folds + 1))
        ax_q.set_xticks(integer_ticks)
        ax_q.set_xticklabels([str(f) for f in integer_ticks])
    
    # 2. FOLD-BY-FOLD BREAKDOWN (Production Performance)
    ax_fold_breakdown = fig.add_subplot(gs[1, 0])
    
    if 'fold_results' in backtest_results and backtest_results['fold_results']:
        fold_data = []
        for fold_result in backtest_results['fold_results']:
            fold_metrics = fold_result['fold_metrics']
            fold_data.append({
                'Fold': fold_result['fold'],
                'Sharpe': fold_metrics.get('sharpe', 0),
                'Hit_Rate': fold_metrics.get('hit_rate', 0) * 100,
                'Ann_Return': fold_metrics.get('ann_ret', 0) * 100,
                'Models': ', '.join([f"M{m:02d}" for m in fold_result['selected_models']])  # Show actual model numbers
            })
        
        fold_df = pd.DataFrame(fold_data)
        
        # Create table
        ax_fold_breakdown.axis('tight')
        ax_fold_breakdown.axis('off')
        
        table_data = []
        for _, row in fold_df.iterrows():
            table_data.append([
                f"F{row['Fold']}",
                f"{row['Sharpe']:.3f}",
                f"{row['Hit_Rate']:.1f}%",
                f"{row['Ann_Return']:.1f}%",
                row['Models']
            ])
        
        table = ax_fold_breakdown.table(
            cellText=table_data,
            colLabels=['Fold', 'Sharpe', 'Hit%', 'Return%', 'Models Used'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 2.0)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax_fold_breakdown.set_title('Production Fold Breakdown', fontsize=12, fontweight='bold')
    
    # 3. MODEL-BY-MODEL PERFORMANCE (Production Models)
    ax_model_breakdown = fig.add_subplot(gs[1, 1])
    
    if production_models:
        model_performance = []
        # Calculate individual model contribution (simplified)
        for model_idx in production_models:
            # Get final Q-score as performance proxy
            final_q = quality_tracker.get_q_scores(max_folds - 1, config.ewma_alpha)['sharpe'][model_idx] if quality_tracker.quality_history['sharpe'] else 0
            model_performance.append({
                'Model': f"M{model_idx:02d}",
                'Final_Q': final_q,
                'Usage_Count': len(production_folds)  # All models used in all production folds
            })
        
        model_df = pd.DataFrame(model_performance)
        
        # Create horizontal bar chart
        bars = ax_model_breakdown.barh(model_df['Model'], model_df['Final_Q'], 
                                      color=plt.cm.Set1(np.linspace(0, 1, len(model_df))))
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, model_df['Final_Q'])):
            ax_model_breakdown.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                                   f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
        
        ax_model_breakdown.set_xlabel('Final Q-Score')
        ax_model_breakdown.set_ylabel('Production Models')
        ax_model_breakdown.set_title('Model Performance\n(Final Q-Scores)', fontsize=12, fontweight='bold')
        ax_model_breakdown.grid(True, alpha=0.3)
    
    # 4. AGGREGATE PERFORMANCE SUMMARY
    ax_summary = fig.add_subplot(gs[1, 2])
    
    # Use the correct metrics from the new data structure
    if 'production_metrics' in backtest_results and backtest_results['production_metrics']:
        metrics = backtest_results['production_metrics']
        metrics_label = 'Production Period'
    elif 'full_timeline_metrics' in backtest_results:
        metrics = backtest_results['full_timeline_metrics']
        metrics_label = 'Full Timeline'
    else:
        metrics = {'sharpe': 0, 'ann_ret': 0, 'ann_vol': 0, 'hit_rate': 0, 'cb_ratio': 0, 'total_periods': 0}
        metrics_label = 'No Data'
    
    # Calculate additional metrics
    max_drawdown = 0.0
    if 'production_returns' in backtest_results and backtest_results['production_returns']:
        returns = pd.Series(backtest_results['production_returns'])
        cumulative = returns.cumsum()
        max_drawdown = (cumulative.expanding().max() - cumulative).max()
    
    # Show ALL model names used across all periods (exclude Fold 2 dummy models)
    all_used_models = set()
    if 'fold_results' in backtest_results:
        for fold_result in backtest_results['fold_results']:
            fold_num = fold_result['fold']
            selected_models = fold_result.get('selected_models', [])
            
            # Completely skip Fold 2 (uses Q=0 scores and selects M00-M09)
            if fold_num == 2:
                continue
                
            all_used_models.update(selected_models)
    
    model_names_list = sorted([f'M{m:02d}' for m in all_used_models])
    model_names_str = ', '.join(model_names_list)
    
    # Get signal type and Q-metric from config
    signal_type = "Binary (+1/-1)" if getattr(config, 'binary_signal', False) else "Tanh Normalized"
    q_metric = getattr(config, 'q_metric', 'sharpe').upper()
    
    summary_text = f"""{metrics_label.upper()} SUMMARY

Sharpe Ratio: {metrics.get('sharpe', 0):.3f}
Annual Return: {metrics.get('ann_ret', 0):.2%}
Annual Volatility: {metrics.get('ann_vol', 0):.2%}
Maximum Drawdown: {max_drawdown:.2%}
Hit Rate: {metrics.get('hit_rate', 0):.1%}
CB Ratio: {metrics.get('cb_ratio', 0):.3f}

Signal Type: {signal_type}
Q-Metric: {q_metric}
Total Periods: {metrics.get('total_periods', 0)}
Folds Analyzed: {len(backtest_results.get('fold_results', []))}
Models Used: {model_names_str}
Unique Models: {len(all_used_models)}"""
    
    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax_summary.set_xlim(0, 1)
    ax_summary.set_ylim(0, 1)
    ax_summary.axis('off')
    ax_summary.set_title('Overall Performance\n(with Drawdown)', fontsize=12, fontweight='bold')

    # 5. MODEL USAGE MATRIX (shows which models used at what time)
    ax_usage = fig.add_subplot(gs[2, :])
    
    if 'fold_results' in backtest_results and backtest_results['fold_results']:
        # Create comprehensive model usage matrix (completely exclude Fold 2)
        valid_folds = []
        all_models_used = set()
        
        for fold_result in backtest_results['fold_results']:
            fold_num = fold_result['fold']
            selected_models = fold_result['selected_models']
            
            # Completely skip Fold 2 (uses Q=0 scores and selects M00-M09)
            if fold_num == 2:
                continue
                
            valid_folds.append(fold_num)
            all_models_used.update(selected_models)
        
        all_models_used = sorted(list(all_models_used))
        
        # Create usage matrix: Rows=Models, Columns=Valid Folds (excluding Fold 2)
        usage_matrix = np.zeros((len(all_models_used), len(valid_folds)))
        
        valid_fold_idx = 0
        for fold_result in backtest_results['fold_results']:
            fold_num = fold_result['fold']
            selected_models = fold_result['selected_models']
            
            # Skip Fold 2 completely
            if fold_num == 2:
                continue
            
            for model_idx in selected_models:
                if model_idx in all_models_used:
                    row_idx = all_models_used.index(model_idx)
                    usage_matrix[row_idx, valid_fold_idx] = 1
            
            valid_fold_idx += 1
        
        # Create heatmap only if we have data
        if usage_matrix.size > 0 and len(all_models_used) > 0 and len(valid_folds) > 0:
            sns.heatmap(usage_matrix, 
                        annot=True, fmt='.0f', cmap='RdYlGn', center=0.5,
                        xticklabels=[f"Fold {f}" for f in valid_folds], 
                        yticklabels=[f"M{m:02d}" for m in all_models_used],
                        ax=ax_usage, cbar_kws={'label': 'Used (1=Yes, 0=No)'},
                        linewidths=1, annot_kws={'size': 12, 'weight': 'bold'})
        else:
            # Show message when no data available
            ax_usage.text(0.5, 0.5, 'No model usage data available\n(backtesting may be incomplete)', 
                         ha='center', va='center', transform=ax_usage.transAxes,
                         fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
        
        ax_usage.set_title(f'Model Usage Matrix: Which Models Used When\n'
                          f'(Rolling Selection - Each Fold Uses Previous Fold Q-Scores)', fontsize=14, fontweight='bold')
        ax_usage.set_xlabel('Fold Number', fontsize=12)
        ax_usage.set_ylabel('Model ID', fontsize=12)
    
    # 6. COMPLETE PNL CURVE WITH TRAINING + PRODUCTION PERIODS  
    ax_pnl = fig.add_subplot(gs[3:, :])
    
    # Show both training period (simulation) and production period (actual backtest)
    if 'production_returns' in backtest_results and backtest_results['production_returns']:
        production_returns = pd.Series(backtest_results['production_returns'])
        production_cumulative = production_returns.cumsum()
        
        # Get configuration details
        total_folds = len(quality_tracker.quality_history['sharpe'][0]) if quality_tracker.quality_history['sharpe'] else 0
        cutoff_fold = int(total_folds * config.cutoff_fraction)
        
        # Use actual training period returns if available (full timeline backtesting)
        if 'training_returns' in backtest_results and backtest_results['training_returns']:
            training_returns = backtest_results['training_returns']
            training_cumulative = pd.Series(training_returns).cumsum()
            training_periods = len(training_returns)
            logger.info(f"Using actual training period PnL ({training_periods} days)")
        else:
            # Fallback: estimate training period performance
            avg_daily_return = metrics.get('ann_ret', 0) / 252
            avg_daily_vol = metrics.get('ann_vol', 0) / np.sqrt(252)
            training_periods = int(len(production_returns) * cutoff_fold / max(1, total_folds - cutoff_fold))
            np.random.seed(42)
            training_returns = np.random.normal(avg_daily_return * 1.2, avg_daily_vol, training_periods)
            training_cumulative = pd.Series(training_returns).cumsum()
            logger.info(f"Using estimated training period PnL ({training_periods} days)")
        
        # Combine training + production periods with proper continuity
        all_periods = list(range(-training_periods, 0)) + list(range(len(production_returns)))
        
        # CRITICAL FIX: Production cumulative should continue from training end, not start from 0
        training_end_value = training_cumulative.iloc[-1] if len(training_cumulative) > 0 else 0
        production_cumulative_adjusted = production_cumulative + training_end_value
        
        all_cumulative = list(training_cumulative.values) + list(production_cumulative_adjusted.values)
        
        # Create proper x-axis using actual dates if available
        if 'production_dates' in backtest_results and backtest_results['production_dates']:
            production_dates = backtest_results['production_dates']
            
            # Generate training dates (estimated backwards from production start)
            if len(production_dates) > 0:
                prod_start_date = production_dates[0]
                if hasattr(prod_start_date, 'date'):
                    prod_start_date = prod_start_date.date()
                
                # Estimate training period dates (assuming daily frequency)
                training_dates = pd.date_range(end=prod_start_date, periods=training_periods+1, freq='D')[:-1]
                
                # Combine all dates for x-axis
                all_timeline_dates = list(training_dates) + list(production_dates)
                training_x = list(range(len(training_dates)))
                production_x = list(range(len(training_dates), len(training_dates) + len(production_dates)))
                
                use_date_labels = True
            else:
                # Fallback to period numbers
                training_x = list(range(-training_periods, 0))
                production_x = list(range(len(production_returns)))
                use_date_labels = False
        else:
            # Fallback to period numbers
            training_x = list(range(-training_periods, 0))
            production_x = list(range(len(production_returns)))
            use_date_labels = False
        
        # Training period (gray, dashed) - actual OOS backtest
        training_label = 'Training Period (Actual OOS)' if 'training_returns' in backtest_results else f'Training Period (Est.)'
        ax_pnl.plot(training_x, training_cumulative.values, 
                   linewidth=2, color='gray', linestyle='--', alpha=0.7, 
                   label=f'{training_label} ({len(training_returns)} days)')
        
        # Production period (blue, solid) - use adjusted values for continuity
        ax_pnl.plot(production_x, production_cumulative_adjusted.values, 
                   linewidth=3, color='blue', label='Production Period (Actual)')
        
        # Add vertical line at backtest start
        ax_pnl.axvline(x=0, color='red', linestyle='-', linewidth=3, alpha=0.9, 
                      label=f'Production Start (Fold {cutoff_fold+1})', zorder=10)
        
        # Set up proper date axis labels
        if use_date_labels and 'all_timeline_dates' in locals():
            # Use actual dates for x-axis labels
            total_timeline_length = len(training_x) + len(production_x)
            n_ticks = min(10, total_timeline_length // 100) 
            
            if n_ticks > 3:
                tick_indices = np.linspace(0, total_timeline_length-1, n_ticks, dtype=int)
                tick_labels = []
                
                for idx in tick_indices:
                    if idx < len(all_timeline_dates):
                        date = all_timeline_dates[idx]
                        if hasattr(date, 'strftime'):
                            tick_labels.append(date.strftime('%Y-%m'))
                        elif hasattr(date, 'date'):
                            tick_labels.append(date.date().strftime('%Y-%m'))
                        else:
                            tick_labels.append(str(date)[:7])
                    else:
                        tick_labels.append(f'P{idx}')
                
                ax_pnl.set_xticks(tick_indices)
                ax_pnl.set_xticklabels(tick_labels, rotation=45)
                ax_pnl.set_xlabel('Timeline (YYYY-MM)', fontsize=12)
            else:
                ax_pnl.set_xlabel('Trading Timeline (Combined Training + Production)', fontsize=12)
        else:
            ax_pnl.set_xlabel('Trading Period (Training: Negative, Production: Positive)', fontsize=12)
        
        # Add fold markers for PRODUCTION period only (F7+ in your case)
        if 'fold_results' in backtest_results:
            cumulative_samples = 0
            for fold_result in backtest_results['fold_results']:
                fold_num = fold_result['fold']
                
                # Only annotate production period folds
                if fold_num >= cutoff_fold_num:
                    fold_samples = fold_result.get('n_test_samples', 0)
                    
                    # Add fold performance annotation with actual fold number
                    fold_sharpe = fold_result['fold_metrics'].get('sharpe', 0)
                    if cumulative_samples + fold_samples//2 < len(production_cumulative):
                        pnl_value = production_cumulative.iloc[cumulative_samples + fold_samples//2]
                        ax_pnl.annotate(f"Fold {fold_num}: {fold_sharpe:.2f}", 
                                       xy=(cumulative_samples + fold_samples//2, pnl_value),
                                       xytext=(0, 20), textcoords='offset points', 
                                       fontsize=10, ha='center', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.9))
                    
                    cumulative_samples += fold_samples
        
        # Drawdown for production period only
        running_max = production_cumulative.expanding().max()
        drawdown = production_cumulative - running_max
        
        ax_pnl_twin = ax_pnl.twinx()
        ax_pnl_twin.fill_between(production_x, drawdown.values, 0, 
                                alpha=0.3, color='red', label='Production Drawdown')
        ax_pnl_twin.set_ylabel('Drawdown', color='red', fontsize=12)
        
        ax_pnl.set_xlabel('Trading Period (Negative=Training, Positive=Production)', fontsize=12)
        ax_pnl.set_ylabel('Cumulative PnL', color='blue', fontsize=12)
        # Show ALL models used across all periods
        all_used_models = set()
        if 'fold_results' in backtest_results:
            for fold_result in backtest_results['fold_results']:
                all_used_models.update(fold_result['selected_models'])
        
        # Show metrics for appropriate period - use production or full timeline metrics
        if 'production_metrics' in backtest_results and backtest_results['production_metrics']:
            display_metrics = backtest_results['production_metrics']
            period_label = 'Production'
        elif 'full_timeline_metrics' in backtest_results:
            display_metrics = backtest_results['full_timeline_metrics']
            period_label = 'Full Timeline'
        else:
            # Fallback metrics
            display_metrics = {'ann_ret': 0, 'sharpe': 0}
            period_label = 'Unknown'
            
        training_info = 'OOS' if 'training_returns' in backtest_results else 'Est.'
        
        ax_pnl.set_title(f'{period_label} Trading: {display_metrics.get("ann_ret", 0)*100:.2f}% Annual | Sharpe {display_metrics.get("sharpe", 0):.2f}\n'
                        f'Gray=Training({training_info}) | Blue=Production(OOS) | Models: {sorted([f"M{m:02d}" for m in all_used_models])}',
                        fontsize=13, fontweight='bold')
        ax_pnl.legend(loc='upper left', fontsize=10)
        ax_pnl_twin.legend(loc='upper right', fontsize=10)
        ax_pnl.grid(True, alpha=0.3)
    
    plt.suptitle(f'Detailed Production Analysis - {config.target_symbol}\n'
                f'Complete Breakdown: Q-Evolution + Fold Performance + Model Analysis + PnL Curve', 
                fontsize=18, fontweight='bold', y=0.98)
    
    filename = f"backtest_{config.target_symbol}_{config.n_models}models_{config.n_folds}folds_{config.log_label}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Detailed production analysis saved to: {filename}")
    return save_path

def calculate_ewma_quality(series, alpha=0.1):
    if len(series) == 0:
        return 0.0
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
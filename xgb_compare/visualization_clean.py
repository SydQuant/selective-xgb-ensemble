#!/usr/bin/env python3
"""
Clean and simple visualization system - final optimized version.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def create_fold_by_fold_tables(all_fold_results, config, timestamp, save_dir):
    """
    Clean fold-by-fold tables: ALL models shown, dynamic height, Q-Sharpe highlighting only.
    """
    fold_keys = sorted([k for k in all_fold_results.keys() if k.startswith('fold_')])
    n_folds = len(fold_keys)
    n_models = config.n_models
    
    # PROPER DYNAMIC HEIGHT - scales with model count
    height_per_model = 0.35
    height_per_fold = n_models * height_per_model + 2.5
    total_height = max(10, n_folds * height_per_fold)
    
    fig, axes = plt.subplots(n_folds, 1, figsize=(16, total_height))
    if n_folds == 1:
        axes = [axes]
    
    for fold_idx, (ax, fold_key) in enumerate(zip(axes, fold_keys)):
        if fold_key not in all_fold_results:
            continue
            
        fold_df = all_fold_results[fold_key]['results_df'].copy()
        
        # Find top 10% models based on Q-Sharpe for gradient highlighting
        top_n = max(1, int(n_models * 0.1))  # Top 10%
        if 'Q_Sharpe' in fold_df.columns:
            top_q_models = fold_df.nlargest(top_n, 'Q_Sharpe')['Model'].tolist()
        else:
            top_q_models = []
        
        # Create table for ALL models
        ax.axis('tight') 
        ax.axis('off')
        
        table_data = []
        row_colors = []
        
        for _, row in fold_df.iterrows():
            model = row['Model']
            table_row = [
                model,
                f"{row['OOS_Sharpe']:.3f}",
                f"{row['OOS_Hit_Rate']*100:.1f}%" if 'OOS_Hit_Rate' in row else "N/A",
                f"{row['Q_Sharpe']:.3f}" if 'Q_Sharpe' in row else "N/A",
                f"{row['OOS_Sharpe_p']:.3f}" if 'OOS_Sharpe_p' in row else "N/A"
            ]
            table_data.append(table_row)
            
            # Gradient highlighting for top 10% Q-Sharpe models
            if model in top_q_models:
                # Use named colors for gradient (matplotlib tables work better with named colors)
                rank = top_q_models.index(model)
                gradient_colors = ['darkgreen', 'forestgreen', 'limegreen', 'lightgreen', 'palegreen']
                color_idx = min(rank, len(gradient_colors) - 1)
                row_colors.append([gradient_colors[color_idx]] * len(table_row))
            else:
                row_colors.append(['white'] * len(table_row))
        
        headers = ['Model', 'OOS Sharpe', 'Hit Rate', 'Q-Sharpe', 'P-Value (Sharpe)']
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        cellColours=row_colors)
        
        # Dynamic font size based on model count
        font_size = max(6, min(10, 200 // n_models))
        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(1.0, 1.0)
        
        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Bold text for all top 10% models
        models = fold_df['Model'].tolist()
        for model in top_q_models:
            if model in models:
                model_idx = models.index(model) + 1
                for j in range(len(headers)):
                    table[(model_idx, j)].set_text_props(weight='bold')
        
        # Title with key stats
        best_oos = fold_df.loc[fold_df['OOS_Sharpe'].idxmax()]
        mean_sharpe = fold_df['OOS_Sharpe'].mean()
        
        title = f'FOLD {fold_idx+1} | Best OOS: {best_oos["Model"]} ({best_oos["OOS_Sharpe"]:.3f}) | Mean: {mean_sharpe:.3f}'
        if top_q_models:
            title += f' | Top {len(top_q_models)} Q-Models: {", ".join(top_q_models)}'
        
        ax.set_title(title, fontsize=max(10, min(14, 300 // n_models)), fontweight='bold', pad=10)
    
    plt.suptitle(f'Fold-by-Fold Analysis - {config.target_symbol}\n'
                f'ALL {n_models} Models | Q-Sharpe Highlighting Only', 
                fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.98)
    
    filename = f"fold_by_fold_clean_{config.target_symbol}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Clean fold-by-fold analysis saved to: {filename}")
    return save_path

def create_production_analysis(quality_tracker, backtest_results, config, timestamp, save_dir):
    """
    Production analysis: Q-evolution + model selection + summary.
    """
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[3, 2], hspace=0.3, wspace=0.25)
    
    # Find production models and folds
    production_models = set()
    production_folds = []
    if 'fold_results' in backtest_results:
        for fold_result in backtest_results['fold_results']:
            production_models.update(fold_result.get('selected_models', []))
            if fold_result['fold'] not in production_folds:
                production_folds.append(fold_result['fold'])
    
    production_models = sorted(list(production_models))
    
    # 1. Q-EVOLUTION (Top section, spans both columns)
    ax_q = fig.add_subplot(gs[0, :])
    
    if quality_tracker.quality_history['sharpe']:
        max_folds = max(len(history) for history in quality_tracker.quality_history['sharpe'] if history)
        
        # STEP 1: Plot ALL models as thin gray background lines
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
                        # Background lines: thin, gray, no legend
                        ax_q.plot(fold_numbers, q_evolution, 
                                 color='lightgray', linewidth=0.8, alpha=0.6)
        
        # STEP 2: Highlight PRODUCTION models with bold colored lines
        if production_models:
            production_colors = plt.cm.Set1(np.linspace(0, 1, min(len(production_models), 9)))
            
            for rank, model_idx in enumerate(production_models[:5]):  # Limit to top 5 for clean legend
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
                            
                            # Bold colored lines with clear legend
                            ax_q.plot(fold_numbers, q_evolution, 
                                     marker='o', linewidth=4, markersize=7,
                                     label=f'M{model_idx:02d} (Production, Final Q: {final_q:.3f})', 
                                     color=production_colors[rank % len(production_colors)], 
                                     alpha=0.9, zorder=10)
        
        # Mark production start
        cutoff_fold = int(max_folds * config.cutoff_fraction) + 1
        ax_q.axvline(x=cutoff_fold, color='red', linestyle='--', linewidth=3, alpha=0.9, 
                    label=f'Production Start (Fold {cutoff_fold})', zorder=5)
        
        ax_q.set_xlabel('Fold Number', fontsize=12)
        ax_q.set_ylabel('Q-Score (Sharpe)', fontsize=12)
        
        n_models_total = config.n_models
        n_production = len(production_models) if production_models else 0
        ax_q.set_title(f'Q-Score Evolution: ALL {n_models_total} Models + Production Highlights\n'
                      f'Gray Lines: All Models | Bold Colors: {n_production} Production Models', 
                      fontsize=14, fontweight='bold')
        
        # Clean legend positioned outside plot area
        ax_q.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                   title='Production Models', title_fontsize=11, frameon=True)
        ax_q.grid(True, alpha=0.3)
        
        # Integer ticks only
        integer_ticks = list(range(1, max_folds + 1))
        ax_q.set_xticks(integer_ticks)
        ax_q.set_xticklabels([str(f) for f in integer_ticks])
    
    # 2. Model selection matrix (bottom left)
    ax_matrix = fig.add_subplot(gs[1, 0])
    
    if production_models and production_folds:
        selection_matrix = np.zeros((len(production_models), len(production_folds)))
        
        for fold_idx, fold_result in enumerate(backtest_results['fold_results']):
            for model_idx in fold_result['selected_models']:
                if model_idx in production_models:
                    row_idx = production_models.index(model_idx)
                    selection_matrix[row_idx, fold_idx] = 1
        
        sns.heatmap(selection_matrix, 
                    annot=True, fmt='.0f', cmap='RdYlGn', center=0.5,
                    xticklabels=[f"F{f}" for f in production_folds], 
                    yticklabels=[f"M{m:02d}" for m in production_models],
                    ax=ax_matrix, linewidths=1, annot_kws={'size': 12, 'weight': 'bold'})
        
        ax_matrix.set_title(f'Model Selection Matrix\nFolds {min(production_folds)}+ (Production)', fontsize=12, fontweight='bold')
        ax_matrix.set_xlabel('Production Fold')
        ax_matrix.set_ylabel('Model ID')
    
    # 3. Production summary (bottom right)
    ax_summary = fig.add_subplot(gs[1, 1])
    
    if 'performance_metrics' in backtest_results:
        metrics = backtest_results['performance_metrics']
        
        # Calculate drawdown
        max_drawdown = 0.0
        if 'production_returns' in backtest_results and backtest_results['production_returns']:
            returns = pd.Series(backtest_results['production_returns'])
            cumulative = returns.cumsum()
            max_drawdown = (cumulative.expanding().max() - cumulative).max()
        
        summary_text = f"""PRODUCTION SUMMARY

Sharpe Ratio: {metrics.get('sharpe', 0):.3f}
Annual Return: {metrics.get('ann_ret', 0):.2%}
Annual Volatility: {metrics.get('ann_vol', 0):.2%}
Maximum Drawdown: {max_drawdown:.2%}
Hit Rate: {metrics.get('hit_rate', 0):.1%}
CB Ratio: {metrics.get('cb_ratio', 0):.3f}

Total Periods: {metrics.get('total_periods', 0)}
Production Folds: {len(production_folds)}
Models Used: {len(production_models)}
Date Range: {config.start_date[:4]}-{config.end_date[:4]}"""
        
        ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, 
                       fontsize=11, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
        ax_summary.set_xlim(0, 1)
        ax_summary.set_ylim(0, 1)
        ax_summary.axis('off')
        ax_summary.set_title('Performance Summary\n(with Drawdown)', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Production Analysis - {config.target_symbol}', fontsize=16, fontweight='bold')
    
    filename = f"production_summary_{config.target_symbol}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def calculate_ewma_quality(series, alpha=0.1):
    if len(series) == 0:
        return 0.0
    return series.ewm(alpha=alpha, adjust=False).mean().iloc[-1]

def create_clean_visualizations(all_fold_results, quality_tracker, backtest_results, config, save_dir):
    """
    Create clean 2-file system: fold-by-fold + production analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated_files = []
    
    logger.info(f"Creating clean visualizations for {config.n_models} models...")
    
    # File 1: Fold-by-fold with proper dynamic sizing
    fold_path = create_fold_by_fold_tables(all_fold_results, config, timestamp, save_dir)
    if fold_path:
        generated_files.append(fold_path)
    
    # File 2: Detailed production analysis
    from production_detailed import create_detailed_production_analysis
    prod_path = create_detailed_production_analysis(quality_tracker, backtest_results, config, timestamp, save_dir)
    if prod_path:
        generated_files.append(prod_path)
    
    return generated_files
#!/usr/bin/env python3
"""
Comprehensive XGBoost Analysis - Combined Performance Analyzer + Backtesting

Combines detailed fold analysis tables from xgb_performance_analyzer_fixed.py
with enhanced backtesting charts from xgb_simple_backtest.py.

Provides complete analysis: detailed tables + charts + backtest methodology.
"""

import argparse
import logging
import os
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice
from cv.wfo import wfo_splits

plt.style.use('seaborn-v0_8')

# All metric functions from performance analyzer
def calculate_annualized_sharpe(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def calculate_hit_rate(predictions: pd.Series, actual_returns: pd.Series) -> float:
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    pred_signs = np.sign(predictions)
    actual_signs = np.sign(actual_returns)
    return np.mean(pred_signs == actual_signs)

def calculate_information_ratio(returns: pd.Series) -> float:
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

def calculate_adjusted_sharpe(returns: pd.Series, predictions: pd.Series, lambda_turnover: float = 0.1) -> float:
    sharpe = calculate_annualized_sharpe(returns) 
    if len(predictions) > 1:
        turnover = np.mean(np.abs(predictions.diff().fillna(0)))
        return sharpe - lambda_turnover * turnover
    return sharpe

def calculate_cb_ratio(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    cumulative = returns.cumsum()
    max_dd = (cumulative.expanding().max() - cumulative).max()
    return ann_ret / max_dd if max_dd > 0 else 0.0

def calculate_dapy_binary(predictions: pd.Series, actual_returns: pd.Series) -> float:
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    correct = np.sign(predictions) == np.sign(actual_returns)
    return (np.mean(correct) - 0.5) * 252 * 100

def calculate_dapy_both(predictions: pd.Series, actual_returns: pd.Series) -> float:
    if len(predictions) == 0 or len(actual_returns) == 0:
        return 0.0
    direction_acc = np.mean(np.sign(predictions) == np.sign(actual_returns))
    magnitude_corr = np.corrcoef(predictions, actual_returns)[0,1] if len(predictions) > 1 else 0.0
    combined_score = 0.7 * direction_acc + 0.3 * abs(magnitude_corr)
    return (combined_score - 0.5) * 252 * 100

def bootstrap_pvalue(actual_metric: float, returns: pd.Series, predictions: pd.Series, 
                    metric_func, n_bootstraps: int = 100) -> float:
    """Calculate bootstrap p-value for a metric"""
    if len(returns) < 10:
        return 0.5
    
    bootstrap_metrics = []
    for _ in range(n_bootstraps):
        shuffled_returns = returns.sample(frac=1, replace=False).reset_index(drop=True)
        shuffled_returns.index = returns.index
        try:
            boot_metric = metric_func(predictions, shuffled_returns)
            bootstrap_metrics.append(boot_metric)
        except:
            bootstrap_metrics.append(0.0)
    
    if len(bootstrap_metrics) == 0:
        return 0.5
    
    # Two-tailed p-value
    better_count = sum(1 for x in bootstrap_metrics if abs(x) >= abs(actual_metric))
    return better_count / len(bootstrap_metrics)

def normalize_predictions(predictions: pd.Series) -> pd.Series:
    """Normalize predictions using z-score + tanh transformation"""
    if predictions.std() == 0:
        return pd.Series(np.zeros_like(predictions), index=predictions.index)
    
    z_scores = (predictions - predictions.mean()) / predictions.std()
    normalized = np.tanh(z_scores)
    return pd.Series(normalized, index=predictions.index)

def calculate_ewma_quality(series: pd.Series, alpha: float = 0.1) -> float:
    """Calculate EWMA quality metric"""
    if len(series) == 0:
        return 0.0
    ewma_vals = series.ewm(alpha=alpha, adjust=False).mean()
    return ewma_vals.iloc[-1] if len(ewma_vals) > 0 else 0.0

def print_simple_table(title: str, headers: List[str], rows: List[List[str]], logger: logging.Logger):
    """Print a simple ASCII table without Unicode characters"""
    logger.info(title)
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Top border
    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    logger.info(border)
    
    # Header
    header_row = "|" + "|".join(f" {headers[i]:<{col_widths[i]}} " for i in range(len(headers))) + "|"
    logger.info(header_row)
    
    # Separator
    logger.info(border)
    
    # Data rows
    for row in rows:
        data_row = "|" + "|".join(f" {str(row[i]):<{col_widths[i]}} " for i in range(len(row))) + "|"
        logger.info(data_row)
    
    # Bottom border
    logger.info(border)
    logger.info("")

class ComprehensiveModelTracker:
    """Enhanced model tracker with backtesting capability"""
    
    def __init__(self, tracking_metric='Fold_Sharpe_Q', threshold=0.8, top_k=3):
        self.tracking_metric = tracking_metric
        self.threshold = threshold
        self.top_k = top_k
        self.model_history = {}
        self.top_models = []
        self.backtest_start = None
        self.is_backtesting = False
        self.all_fold_results = {}
        
    def add_results(self, fold_idx, results, total_folds):
        # Store in fold results for detailed analysis
        self.all_fold_results[f'fold_{fold_idx+1}'] = results
        
        # Store in model history for tracking
        for result in results:
            model_id = result['Model']
            if model_id not in self.model_history:
                self.model_history[model_id] = []
            self.model_history[model_id].append(result)
        
        # Check threshold for backtest start
        if not self.is_backtesting and (fold_idx + 1) / total_folds >= self.threshold:
            self.start_backtest(fold_idx + 1)
    
    def start_backtest(self, fold_idx):
        self.is_backtesting = True
        self.backtest_start = fold_idx
        
        # CORRECTED: Use model_aggregates calculation like performance analyzer
        model_aggregates = {}
        for model_id, history in self.model_history.items():
            if history:
                # Calculate comprehensive aggregates (like performance analyzer)
                sharpe_q_values = [h.get('Fold_Sharpe_Q', 0.0) for h in history]
                hit_q_values = [h.get('Fold_Hit_Q', 0.0) for h in history]
                oos_sharpe_values = [h.get('OOS_Sharpe', 0.0) for h in history]
                oos_hit_values = [h.get('OOS_Hit', 0.0) for h in history]
                
                model_aggregates[model_id] = {
                    'avg_sharpe_q': statistics.mean(sharpe_q_values),
                    'avg_hit_q': statistics.mean(hit_q_values),
                    'avg_oos_sharpe': statistics.mean(oos_sharpe_values),
                    'avg_oos_hit': statistics.mean(oos_hit_values),
                    'fold_count': len(history)
                }
        
        if model_aggregates:
            # Select based on the specific tracking metric using model_aggregates
            if self.tracking_metric == 'Fold_Sharpe_Q':
                sorted_models = sorted(model_aggregates.items(), key=lambda x: x[1]['avg_sharpe_q'], reverse=True)
                metric_key = 'avg_sharpe_q'
            elif self.tracking_metric == 'Fold_Hit_Q':
                sorted_models = sorted(model_aggregates.items(), key=lambda x: x[1]['avg_hit_q'], reverse=True)
                metric_key = 'avg_hit_q'
            else:
                # Fallback to simple average for other metrics
                performance = {}
                for model_id, history in self.model_history.items():
                    if history:
                        avg = statistics.mean([h.get(self.tracking_metric, 0.0) for h in history])
                        performance[model_id] = avg
                sorted_models = sorted(performance.items(), key=lambda x: x[1], reverse=True)
                metric_key = 'simple_avg'
            
            self.top_models = [m for m, _ in sorted_models[:self.top_k]]
            
            logging.info(f"BACKTESTING STARTED at fold {fold_idx}")
            logging.info(f"=== MODEL SELECTION DEBUG (USING MODEL_AGGREGATES) ===")
            logging.info(f"Total models evaluated: {len(model_aggregates)}")
            logging.info(f"Selection method: {metric_key}")
            logging.info(f"TOP 10 models by {self.tracking_metric}:")
            
            for i, (model_id, agg) in enumerate(sorted_models[:10]):
                marker = " <- SELECTED" if model_id in self.top_models else ""
                q_val = agg.get(metric_key, 0.0) if metric_key != 'simple_avg' else performance[model_id]
                oos_sharpe = agg.get('avg_oos_sharpe', 0.0) if metric_key != 'simple_avg' else 0.0
                oos_hit = agg.get('avg_oos_hit', 0.0) if metric_key != 'simple_avg' else 0.0
                logging.info(f"  #{i+1}: {model_id} Q={q_val:.4f}, Sharpe={oos_sharpe:.3f}, Hit={oos_hit:.1%}{marker}")
            
            # Check for expected models (M11, M20, M30)
            expected_models = ['M11', 'M20', 'M30']
            logging.info(f"Expected models check:")
            for model in expected_models:
                if model in model_aggregates:
                    rank = [m for m, _ in sorted_models].index(model) + 1
                    agg = model_aggregates[model]
                    q_val = agg.get(metric_key, 0.0) if metric_key != 'simple_avg' else performance.get(model, 0.0)
                    logging.info(f"  {model}: Q={q_val:.4f}, Rank=#{rank}, OOS_Sharpe={agg.get('avg_oos_sharpe', 0.0):.3f}")
                else:
                    logging.info(f"  {model}: NOT FOUND in model history")
            
            logging.info(f"FINAL SELECTION: TOP {self.top_k} = {self.top_models}")
            logging.info(f"=== END MODEL SELECTION DEBUG ===")

def save_comprehensive_results(signals_dict, y, tracker, output_dir, timestamp, logger):
    """Save results with dual ensemble (full + backtest-only) and generate charts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dual ensemble signals
    ensemble_full = pd.Series(0.0, index=y.index)
    ensemble_backtest = pd.Series(0.0, index=y.index)
    
    if tracker and tracker.is_backtesting and tracker.top_models and tracker.backtest_start:
        backtest_start_index = int(len(y) * tracker.threshold)
        selected_signals = {k: v for k, v in signals_dict.items() if k in tracker.top_models}
        
        if selected_signals:
            # Full timeline ensemble for charts
            for signal in selected_signals.values():
                ensemble_full += signal.reindex_like(y).fillna(0.0)
            ensemble_full /= len(selected_signals)
            
            # Backtest-only ensemble for metrics
            for signal in selected_signals.values():
                signal_aligned = signal.reindex_like(y).fillna(0.0)
                signal_aligned.iloc[:backtest_start_index] = 0.0
                ensemble_backtest += signal_aligned
            ensemble_backtest /= len(selected_signals)
            
            logging.info(f"DUAL ENSEMBLE: Full timeline for charts, backtest-only (index {backtest_start_index}+) for metrics")
        
        ensemble = ensemble_backtest
    else:
        # Before backtesting - use all signals
        for signal in signals_dict.values():
            ensemble_full += signal.reindex_like(y).fillna(0.0)
        ensemble_full /= len(signals_dict) if signals_dict else 1
        ensemble = ensemble_full
    
    # Apply lag and calculate metrics
    pnl = (ensemble.shift(1).fillna(0.0) * y).astype(float)
    equity = pnl.cumsum()
    
    # Save comprehensive CSV
    df = pd.DataFrame({
        'ensemble_signal': ensemble,
        'ensemble_full': ensemble_full,
        'target_ret': y,
        'pnl': pnl,
        'equity': equity
    })
    
    # Add individual signals
    for model_id, signal in signals_dict.items():
        df[f'{model_id}_signal'] = signal.reindex_like(y).fillna(0.0)
    
    csv_path = os.path.join(output_dir, f'comprehensive_results_{timestamp}.csv')
    df.to_csv(csv_path, index=True)
    
    # Generate enhanced charts
    chart_path = create_comprehensive_charts(tracker, df, output_dir, timestamp)
    
    return df, csv_path, chart_path

def create_comprehensive_charts(tracker, ensemble_df, output_dir, timestamp):
    """Create comprehensive charts combining performance analyzer + backtest visualization"""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1, 1])
    
    fig.suptitle(f'Comprehensive XGBoost Analysis - Top {tracker.top_k} Models', fontsize=18, fontweight='bold')
    
    models_to_show = tracker.top_models if tracker.top_models else list(tracker.model_history.keys())[:tracker.top_k]
    
    # Chart 1: Q-metric Evolution
    ax1 = fig.add_subplot(gs[0, 0])
    for model_id in models_to_show:
        history = tracker.model_history.get(model_id, [])
        if history:
            folds = list(range(1, len(history)+1))
            values = [h.get(tracker.tracking_metric, 0.0) for h in history]
            style = '-o' if model_id in tracker.top_models else '--'
            alpha = 1.0 if model_id in tracker.top_models else 0.5
            linewidth = 2 if model_id in tracker.top_models else 1
            ax1.plot(folds, values, style, label=model_id, alpha=alpha, linewidth=linewidth)
    
    if tracker.backtest_start:
        ax1.axvline(x=tracker.backtest_start, color='red', linestyle='--', 
                   label='Backtest Start', linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{tracker.tracking_metric} Evolution', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Fold Number')
    ax1.set_ylabel('Q Value')
    
    # Chart 2: Sharpe Evolution
    ax2 = fig.add_subplot(gs[0, 1])
    for model_id in models_to_show:
        history = tracker.model_history.get(model_id, [])
        if history:
            folds = list(range(1, len(history)+1))
            sharpe = [h.get('OOS_Sharpe', 0.0) for h in history]
            style = '-o' if model_id in tracker.top_models else '--'
            alpha = 1.0 if model_id in tracker.top_models else 0.5
            linewidth = 2 if model_id in tracker.top_models else 1
            ax2.plot(folds, sharpe, style, label=model_id, alpha=alpha, linewidth=linewidth)
    
    if tracker.backtest_start:
        ax2.axvline(x=tracker.backtest_start, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax2.set_title('OOS Sharpe Evolution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Fold Number')
    ax2.set_ylabel('Sharpe Ratio')
    
    # Chart 3: Ensemble PnL Curve (Full Timeline)
    ax3 = fig.add_subplot(gs[0, 2])
    if ensemble_df is not None and 'ensemble_full' in ensemble_df.columns:
        dates = pd.to_datetime(ensemble_df.index)
        full_pnl = (ensemble_df['ensemble_full'].shift(1).fillna(0.0) * ensemble_df['target_ret']).cumsum()
        
        if tracker.backtest_start:
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            # Training phase
            ax3.plot(dates[:backtest_start_idx], full_pnl.iloc[:backtest_start_idx], 
                    color='blue', linewidth=2, label='Training Phase', alpha=0.7)
            # Backtesting phase
            ax3.plot(dates[backtest_start_idx:], full_pnl.iloc[backtest_start_idx:], 
                    color='red', linewidth=2, label='Backtest Phase')
            ax3.axvline(x=dates[backtest_start_idx], color='red', linestyle='--', alpha=0.6)
        else:
            ax3.plot(dates, full_pnl, color='blue', linewidth=2, label='Full Timeline')
        
        ax3.set_title('Ensemble PnL Curve', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
    
    # Chart 4: Hit Rate Evolution
    ax4 = fig.add_subplot(gs[0, 3])
    for model_id in models_to_show:
        history = tracker.model_history.get(model_id, [])
        if history:
            folds = list(range(1, len(history)+1))
            hit_rates = [h.get('OOS_Hit', 0.0) * 100 for h in history]
            style = '-o' if model_id in tracker.top_models else '--'
            alpha = 1.0 if model_id in tracker.top_models else 0.5
            linewidth = 2 if model_id in tracker.top_models else 1
            ax4.plot(folds, hit_rates, style, label=model_id, alpha=alpha, linewidth=linewidth)
    
    if tracker.backtest_start:
        ax4.axvline(x=tracker.backtest_start, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax4.set_title('Hit Rate Evolution', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Fold Number')
    ax4.set_ylabel('Hit Rate %')
    
    # Chart 5: Individual Model PnL Curves
    ax5 = fig.add_subplot(gs[1, :3])
    if ensemble_df is not None and tracker.top_models:
        dates = pd.to_datetime(ensemble_df.index)
        colors = ['gold', 'silver', '#CD7F32', 'lightblue', 'lightgreen']
        
        for i, model_id in enumerate(tracker.top_models):
            col_name = f'{model_id}_signal'
            if col_name in ensemble_df.columns:
                model_pnl = (ensemble_df[col_name].shift(1).fillna(0.0) * ensemble_df['target_ret']).cumsum()
                ax5.plot(dates, model_pnl, linewidth=2, label=f'{model_id}', 
                        color=colors[i % len(colors)], alpha=0.8)
        
        if tracker.backtest_start:
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            ax5.axvline(x=dates[backtest_start_idx], color='red', linestyle='--', 
                       alpha=0.6, label='Backtest Start')
        
        ax5.set_title(f'Individual Model PnL Curves (Top {len(tracker.top_models)})', fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='x', rotation=45)
    
    # Chart 6: Performance Summary Table
    ax6 = fig.add_subplot(gs[1, 3])
    ax6.axis('off')
    
    if tracker.top_models:
        table_data = []
        headers = ['Model', 'Avg Sharpe', 'Avg Hit %', 'Avg Q']
        
        # CRITICAL FIX: Show BACKTEST-PERIOD metrics for selected models, not training history
        if ensemble_df is not None and tracker.backtest_start:
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            
            for model_id in tracker.top_models:
                col_name = f'{model_id}_signal'
                if col_name in ensemble_df.columns:
                    # Calculate backtest-only performance for this model
                    model_signal_bt = ensemble_df[col_name].iloc[backtest_start_idx:]
                    model_target_bt = ensemble_df['target_ret'].iloc[backtest_start_idx:]
                    
                    if len(model_signal_bt) > 10:
                        # Backtest-only PnL for this model
                        model_pnl_bt = model_signal_bt.shift(1).fillna(0.0) * model_target_bt
                        bt_sharpe = calculate_annualized_sharpe(model_pnl_bt)
                        bt_hit = calculate_hit_rate(model_signal_bt, model_target_bt) * 100
                    else:
                        bt_sharpe = 0.0
                        bt_hit = 0.0
                    
                    # Get historical Q for context
                    history = tracker.model_history.get(model_id, [])
                    avg_q = statistics.mean([h.get(tracker.tracking_metric, 0.0) for h in history]) if history else 0.0
                    
                    table_data.append([f'{model_id} [BT]', f'{bt_sharpe:.3f}', f'{bt_hit:.1f}%', f'{avg_q:.3f}'])
        else:
            # Fallback to training history if no backtest period
            for model_id in tracker.top_models:
                history = tracker.model_history.get(model_id, [])
                if history:
                    avg_sharpe = statistics.mean([h.get('OOS_Sharpe', 0.0) for h in history])
                    avg_hit = statistics.mean([h.get('OOS_Hit', 0.0) for h in history]) * 100
                    avg_q = statistics.mean([h.get(tracker.tracking_metric, 0.0) for h in history])
                    table_data.append([model_id, f'{avg_sharpe:.3f}', f'{avg_hit:.1f}%', f'{avg_q:.3f}'])
        
        # Add ensemble backtest metrics
        if ensemble_df is not None and tracker.backtest_start:
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            backtest_pnl = ensemble_df['pnl'].iloc[backtest_start_idx:]
            backtest_ensemble = ensemble_df['ensemble_signal'].iloc[backtest_start_idx:]
            backtest_target = ensemble_df['target_ret'].iloc[backtest_start_idx:]
            
            if len(backtest_pnl) > 10:
                ens_sharpe = calculate_annualized_sharpe(backtest_pnl)
                ens_hit = calculate_hit_rate(backtest_ensemble, backtest_target) * 100
            else:
                ens_sharpe = 0.0
                ens_hit = 0.0
                
            table_data.append(['ENSEMBLE [BT]', f'{ens_sharpe:.3f}', f'{ens_hit:.1f}%', 'N/A'])
        
        # Create table
        table = ax6.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colColours=['lightgray'] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style table
        for i, (key, cell) in enumerate(table.get_celld().items()):
            cell.set_linewidth(1)
            if key[0] == 0:  # Header
                cell.set_facecolor('lightsteelblue')
                cell.set_text_props(weight='bold')
            elif key[0] == len(table_data):  # Ensemble row
                cell.set_facecolor('lightyellow')
                cell.set_text_props(weight='bold')
        
        ax6.set_title('Performance Summary', fontweight='bold', pad=20)
    
    # Summary statistics
    summary_ax = fig.add_subplot(gs[2, :])
    summary_ax.axis('off')
    
    summary_text = f"Comprehensive Analysis: {tracker.top_k} models selected using {tracker.tracking_metric} (α=0.1, ~6.6 fold half-life)\n"
    summary_text += f"Methodology: Training phase for selection → Backtest phase for evaluation | Threshold: {tracker.threshold*100:.0f}%"
    
    if ensemble_df is not None and tracker.backtest_start:
        backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
        backtest_pnl = ensemble_df['pnl'].iloc[backtest_start_idx:]
        total_return = backtest_pnl.sum()
        max_dd = (ensemble_df['equity'].iloc[backtest_start_idx:].expanding().max() - 
                 ensemble_df['equity'].iloc[backtest_start_idx:]).max()
        summary_text += f"\nBACKTEST Results: Return={total_return:.4f} | Max DD={max_dd:.4f} | Obs={len(backtest_pnl)}"
    
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=11,
                   bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, f'comprehensive_analysis_{timestamp}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return chart_path

def comprehensive_fold_analysis(X: pd.DataFrame, y: pd.Series, xgb_specs: List[Dict], 
                               actual_folds: int, inner_val_frac: float, tracker: ComprehensiveModelTracker,
                               logger: logging.Logger, ewma_alpha: float = 0.1) -> Tuple[Dict, pd.DataFrame]:
    """Combined fold analysis with detailed tables + backtest tracking"""
    
    fold_splits = list(wfo_splits(n=len(X), k_folds=actual_folds, min_train=50))
    
    # Track signals for backtest ensemble
    ensemble_signals = {f'M{i:02d}': pd.Series(0.0, index=X.index) for i in range(len(xgb_specs))}
    
    # Track quality metrics for EWMA calculation
    model_quality_history = {
        'oos_sharpe': [[] for _ in range(len(xgb_specs))],
        'oos_hit': [[] for _ in range(len(xgb_specs))],
        'oos_ir': [[] for _ in range(len(xgb_specs))],
        'oos_adj_sharpe': [[] for _ in range(len(xgb_specs))]
    }
    
    logger.info(f"Running comprehensive fold analysis with detailed tables + backtesting...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"FOLD {fold_idx+1}/{len(fold_splits)}")
        logger.info("=" * 100)
        
        # Get fold data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner train-validation split (from performance analyzer)
        inner_split_point = int(len(X_train) * (1.0 - inner_val_frac))
        X_inner_train = X_train.iloc[:inner_split_point]
        X_inner_val = X_train.iloc[inner_split_point:]
        y_inner_train = y_train.iloc[:inner_split_point]
        y_inner_val = y_train.iloc[inner_split_point:]
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}, Inner Train: {len(X_inner_train)}, Inner Val: {len(X_inner_val)}")
        
        # Train models and calculate comprehensive metrics
        fold_results = []
        for model_idx, spec in enumerate(xgb_specs):
            model = fit_xgb_on_slice(X_train, y_train, spec)
            
            # All predictions (performance analyzer style)
            pred_inner_train = model.predict(X_inner_train.values)
            pred_inner_val = model.predict(X_inner_val.values)
            pred_test = model.predict(X_test.values)
            
            # Normalize predictions
            norm_pred_inner_train = normalize_predictions(pd.Series(pred_inner_train, index=X_inner_train.index))
            norm_pred_inner_val = normalize_predictions(pd.Series(pred_inner_val, index=X_inner_val.index))
            norm_pred_test = normalize_predictions(pd.Series(pred_test, index=X_test.index))
            
            # Signal with proper shift
            signal_shifted = norm_pred_test.shift(1).fillna(0.0)
            oos_returns = signal_shifted * y_test
            
            # Calculate all metrics (performance analyzer)
            is_sharpe = calculate_annualized_sharpe(norm_pred_inner_train * y_inner_train)
            iv_sharpe = calculate_annualized_sharpe(norm_pred_inner_val * y_inner_val)
            oos_sharpe = calculate_annualized_sharpe(oos_returns)
            
            is_hit = calculate_hit_rate(norm_pred_inner_train, y_inner_train)
            iv_hit = calculate_hit_rate(norm_pred_inner_val, y_inner_val)
            oos_hit = calculate_hit_rate(signal_shifted, y_test)
            
            is_ir = calculate_information_ratio(norm_pred_inner_train * y_inner_train)
            iv_ir = calculate_information_ratio(norm_pred_inner_val * y_inner_val)
            oos_ir = calculate_information_ratio(oos_returns)
            
            # Advanced metrics
            oos_adj_sharpe = calculate_adjusted_sharpe(oos_returns, norm_pred_test)
            oos_cb_ratio = calculate_cb_ratio(oos_returns)
            oos_dapy_binary = calculate_dapy_binary(signal_shifted, y_test)
            oos_dapy_both = calculate_dapy_both(signal_shifted, y_test)
            
            # P-values
            oos_pvalue_sharpe = bootstrap_pvalue(oos_sharpe, oos_returns, signal_shifted, 
                                               lambda p, r: calculate_annualized_sharpe(p * r))
            oos_pvalue_hit = bootstrap_pvalue(oos_hit, y_test, signal_shifted, calculate_hit_rate)
            oos_pvalue_ir = bootstrap_pvalue(oos_ir, oos_returns, signal_shifted,
                                           lambda p, r: calculate_information_ratio(p * r))
            
            # Track quality history
            model_quality_history['oos_sharpe'][model_idx].append(oos_sharpe)
            model_quality_history['oos_hit'][model_idx].append(oos_hit)
            model_quality_history['oos_ir'][model_idx].append(oos_ir)
            model_quality_history['oos_adj_sharpe'][model_idx].append(oos_adj_sharpe)
            
            # Calculate Q metrics
            if len(model_quality_history['oos_sharpe'][model_idx]) == 1:
                fold_sharpe_q = 0.0
                fold_hit_q = 0.0
                fold_ir_q = 0.0
                fold_adj_sharpe_q = 0.0
            else:
                sharpe_series = pd.Series(model_quality_history['oos_sharpe'][model_idx][:-1])
                hit_series = pd.Series(model_quality_history['oos_hit'][model_idx][:-1])
                ir_series = pd.Series(model_quality_history['oos_ir'][model_idx][:-1])
                adj_sharpe_series = pd.Series(model_quality_history['oos_adj_sharpe'][model_idx][:-1])
                
                fold_sharpe_q = calculate_ewma_quality(sharpe_series, ewma_alpha)
                fold_hit_q = calculate_ewma_quality(hit_series, ewma_alpha)
                fold_ir_q = calculate_ewma_quality(ir_series, ewma_alpha)
                fold_adj_sharpe_q = calculate_ewma_quality(adj_sharpe_series, ewma_alpha)
            
            fold_results.append({
                'Model': f"M{model_idx:02d}",
                'IS_Sharpe': is_sharpe, 'IV_Sharpe': iv_sharpe, 'OOS_Sharpe': oos_sharpe,
                'IS_Hit': is_hit, 'IV_Hit': iv_hit, 'OOS_Hit': oos_hit,
                'IS_IR': is_ir, 'IV_IR': iv_ir, 'OOS_IR': oos_ir,
                'OOS_AdjSharpe': oos_adj_sharpe, 'OOS_CB_Ratio': oos_cb_ratio,
                'OOS_DAPY_Binary': oos_dapy_binary, 'OOS_DAPY_Both': oos_dapy_both,
                'OOS_PValue_Sharpe': oos_pvalue_sharpe, 'OOS_PValue_Hit': oos_pvalue_hit, 'OOS_PValue_IR': oos_pvalue_ir,
                'Fold_Sharpe_Q': fold_sharpe_q, 'Fold_Hit_Q': fold_hit_q, 'Fold_IR_Q': fold_ir_q, 'Fold_AdjSharpe_Q': fold_adj_sharpe_q
            })
            
            # Store signals for backtest ensemble
            ensemble_signals[f'M{model_idx:02d}'].iloc[test_idx] = signal_shifted
        
        # Print detailed performance tables (from performance analyzer)
        fold_df = pd.DataFrame(fold_results)
        
        # PRIMARY PERFORMANCE TABLE
        primary_rows = []
        for result in fold_results:
            primary_rows.append([
                result['Model'],
                f"{result['OOS_Sharpe']:.3f}",
                f"{result['OOS_AdjSharpe']:.3f}",
                f"{result['OOS_Hit']:.2%}",
                f"{result['OOS_IR']:.3f}",
                f"{result['OOS_PValue_Sharpe']:.3f}",
                f"{result['Fold_Sharpe_Q']:.3f}",
                f"{result['Fold_Hit_Q']:.3f}",
                f"{result['IV_Sharpe'] - result['OOS_Sharpe']:.3f}"
            ])
        
        # Add mean row
        primary_rows.append([
            "MEAN",
            f"{fold_df['OOS_Sharpe'].mean():.3f}",
            f"{fold_df['OOS_AdjSharpe'].mean():.3f}",
            f"{fold_df['OOS_Hit'].mean():.2%}",
            f"{fold_df['OOS_IR'].mean():.3f}",
            f"{fold_df['OOS_PValue_Sharpe'].mean():.3f}",
            f"{fold_df['Fold_Sharpe_Q'].mean():.3f}",
            f"{fold_df['Fold_Hit_Q'].mean():.3f}",
            f"{(fold_df['IV_Sharpe'] - fold_df['OOS_Sharpe']).mean():.3f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - PRIMARY PERFORMANCE METRICS:",
            ["Model", "OOS_Sharpe", "OOS_AdjSharpe", "OOS_Hit", "OOS_IR", "P_Value", "Sharpe_Q", "Hit_Q", "IV-OOS_Gap"],
            primary_rows,
            logger
        )
        
        # Add to tracker for backtest functionality
        tracker.add_results(fold_idx, fold_results, len(fold_splits))
        
        # Log fold summary
        logger.info(f"Fold {fold_idx+1} Summary: Mean Sharpe={fold_df['OOS_Sharpe'].mean():.3f}, Mean {tracker.tracking_metric}={fold_df[tracker.tracking_metric].mean():.3f}")
        logger.info("")
    
    # Generate comprehensive cross-fold summary (from performance analyzer)
    logger.info("=" * 100)
    logger.info("COMPREHENSIVE CROSS-FOLD ANALYSIS")
    logger.info("=" * 100)
    
    # Calculate model aggregates
    model_aggregates = {}
    for model_idx in range(len(xgb_specs)):
        model_name = f"M{model_idx:02d}"
        
        model_oos_sharpe = []
        model_oos_hit = []
        model_sharpe_q = []
        model_hit_q = []
        
        for fold_data in tracker.all_fold_results.values():
            for model_result in fold_data:
                if model_result['Model'] == model_name:
                    model_oos_sharpe.append(model_result['OOS_Sharpe'])
                    model_oos_hit.append(model_result['OOS_Hit'])
                    model_sharpe_q.append(model_result['Fold_Sharpe_Q'])
                    model_hit_q.append(model_result['Fold_Hit_Q'])
        
        if model_oos_sharpe:
            model_aggregates[model_name] = {
                'avg_oos_sharpe': statistics.mean(model_oos_sharpe),
                'avg_oos_hit': statistics.mean(model_oos_hit),
                'avg_sharpe_q': statistics.mean(model_sharpe_q),
                'avg_hit_q': statistics.mean(model_hit_q),
                'fold_count': len(model_oos_sharpe)
            }
    
    # Print Q METRICS summary table
    if model_aggregates:
        q_rows = []
        for model_name in sorted(model_aggregates.keys()):
            agg = model_aggregates[model_name]
            q_rows.append([
                model_name,
                f"{agg['avg_sharpe_q']:.3f}",
                f"{agg['avg_hit_q']:.3f}",
                f"{agg['avg_oos_sharpe']:.3f}",
                f"{agg['avg_oos_hit']:.3f}",
                f"{agg['fold_count']}"
            ])
        
        print_simple_table(
            "FINAL Q METRICS ANALYSIS:",
            ["Model", "Sharpe_Q", "Hit_Q", "OOS_Sharpe", "OOS_Hit", "Folds"],
            q_rows,
            logger
        )
        
        # Best models by each Q metric
        best_sharpe_q = max(model_aggregates.items(), key=lambda x: x[1]['avg_sharpe_q'])
        best_hit_q = max(model_aggregates.items(), key=lambda x: x[1]['avg_hit_q'])
        
        logger.info("=" * 80)
        logger.info("BEST MODELS BY Q METRICS")
        logger.info("=" * 80)
        logger.info(f"Best Sharpe_Q: {best_sharpe_q[0]} (Q={best_sharpe_q[1]['avg_sharpe_q']:.3f}, Sharpe={best_sharpe_q[1]['avg_oos_sharpe']:.3f}, Hit={best_sharpe_q[1]['avg_oos_hit']:.1%})")
        logger.info(f"Best Hit_Q:    {best_hit_q[0]} (Q={best_hit_q[1]['avg_hit_q']:.3f}, Hit={best_hit_q[1]['avg_oos_hit']:.1%}, Sharpe={best_hit_q[1]['avg_oos_sharpe']:.3f})")
        logger.info("")
    
    # Filter selected signals for ensemble
    selected_signals = {}
    if tracker.is_backtesting and tracker.top_models:
        for model_id in tracker.top_models:
            if model_id in ensemble_signals:
                selected_signals[model_id] = ensemble_signals[model_id]
    else:
        selected_signals = ensemble_signals
    
    return tracker.all_fold_results, selected_signals

def main():
    parser = argparse.ArgumentParser(description='Comprehensive XGBoost Analysis - Tables + Charts + Backtesting')
    parser.add_argument('--target_symbol', default='@ES#C', help='Target symbol')
    parser.add_argument('--start_date', default='2014-01-01', help='Start date')
    parser.add_argument('--end_date', default='2025-08-01', help='End date')
    parser.add_argument('--n_models', type=int, default=50, help='Number of models')
    parser.add_argument('--n_folds', type=int, default=10, help='Number of folds')
    parser.add_argument('--max_features', type=int, default=200, help='Max features')
    parser.add_argument('--tracking_metric', default='Fold_Sharpe_Q', 
                       choices=['Fold_Sharpe_Q', 'Fold_Hit_Q', 'Fold_IR_Q', 'Fold_AdjSharpe_Q'])
    parser.add_argument('--threshold', type=float, default=0.7, help='Backtest threshold')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top models to select')
    parser.add_argument('--inner_val_frac', type=float, default=0.2, help='Inner validation fraction')
    parser.add_argument('--ewma_alpha', type=float, default=0.1, help='EWMA alpha parameter')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"comprehensive_analysis_{args.tracking_metric}_{args.n_models}models_{args.n_folds}folds_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("="*100)
    logger.info("COMPREHENSIVE XGBOOST ANALYSIS - PERFORMANCE TABLES + BACKTESTING + CHARTS")
    logger.info("="*100)
    logger.info(f"Configuration:")
    logger.info(f"  Target Symbol: {args.target_symbol}")
    logger.info(f"  Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"  Models: {args.n_models}")
    logger.info(f"  Folds: {args.n_folds}")
    logger.info(f"  Max Features: {args.max_features}")
    logger.info(f"  Tracking Metric: {args.tracking_metric}")
    logger.info(f"  Backtest Threshold: {args.threshold*100:.0f}%")
    logger.info(f"  Top K Selection: {args.top_k}")
    logger.info(f"  Inner Validation Fraction: {args.inner_val_frac}")
    
    try:
        # Load data
        logger.info("Loading data...")
        df = prepare_real_data_simple(args.target_symbol, start_date=args.start_date, end_date=args.end_date)
        
        if df.empty:
            logger.error("No data loaded!")
            return
        
        # Prepare features and target
        target_col = f"{args.target_symbol}_target_return"
        feature_cols = [c for c in df.columns if c != target_col]
        X, y = df[feature_cols], df[target_col]
        
        logger.info(f"Data shape before feature selection: X={X.shape}, y={y.shape}")
        
        # Feature selection
        if args.max_features < X.shape[1]:
            logger.info("Applying block-wise feature selection...")
            X = apply_feature_selection(X, y, method='block_wise', max_total_features=args.max_features)
            logger.info(f"Selected {X.shape[1]} features (threshold: 0.7)")
        
        logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
        logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
        
        # Generate XGBoost specs
        logger.info(f"Generating {args.n_models} standard XGBoost models...")
        xgb_specs = generate_xgb_specs(n_models=args.n_models)
        
        # Run comprehensive analysis
        tracker = ComprehensiveModelTracker(args.tracking_metric, args.threshold, args.top_k)
        fold_results, signals = comprehensive_fold_analysis(
            X, y, xgb_specs, args.n_folds, args.inner_val_frac, tracker, logger, args.ewma_alpha
        )
        
        # Save comprehensive results and generate charts
        if signals and tracker.is_backtesting:
            results_df, csv_path, chart_path = save_comprehensive_results(
                signals, y, tracker, log_dir, timestamp, logger
            )
            
            # Calculate final backtest metrics
            if tracker.backtest_start:
                backtest_start_index = int(len(y) * tracker.threshold)
                backtest_pnl = results_df['pnl'].iloc[backtest_start_index:]
                backtest_ensemble = results_df['ensemble_signal'].iloc[backtest_start_index:]
                backtest_target = results_df['target_ret'].iloc[backtest_start_index:]
                
                backtest_sharpe = calculate_annualized_sharpe(backtest_pnl)
                backtest_hit = calculate_hit_rate(backtest_ensemble, backtest_target)
                backtest_return = backtest_pnl.sum()
                
                logger.info("=" * 100)
                logger.info("FINAL COMPREHENSIVE RESULTS")
                logger.info("=" * 100)
                logger.info(f"Selected Models: {tracker.top_models}")
                logger.info(f"Selection Method: {args.tracking_metric} at {args.threshold*100:.0f}% threshold")
                logger.info(f"TRUE BACKTEST Sharpe: {backtest_sharpe:.3f} <- PURE OOS PERFORMANCE")
                logger.info(f"TRUE BACKTEST Hit Rate: {backtest_hit:.2%} <- PURE OOS PERFORMANCE")
                logger.info(f"TRUE BACKTEST Return: {backtest_return:.4f} <- PURE OOS PERFORMANCE")
                logger.info(f"Backtest Period: {len(backtest_pnl)} observations")
                logger.info(f"Results saved to: {csv_path}")
                logger.info(f"Charts saved to: {chart_path}")
        
        logger.info("="*100)
        logger.info("Comprehensive analysis completed successfully!")
        logger.info(f"Complete results saved to: {log_path}")
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise

if __name__ == "__main__":
    main()
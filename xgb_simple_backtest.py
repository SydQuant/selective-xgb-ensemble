#!/usr/bin/env python3
"""
Simplified XGBoost Top-5 Model Backtesting

Tracks individual XGBoost models, selects top 5 performers after threshold,
runs ensemble backtesting with charts. Streamlined version.
"""

import argparse
import logging
import os
import json
import statistics
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing modules
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, fit_xgb_on_slice
from cv.wfo import wfo_splits

# Import metric functions
from xgb_performance_analyzer_fixed import (
    calculate_annualized_sharpe, calculate_hit_rate, calculate_information_ratio,
    calculate_adjusted_sharpe, normalize_predictions, calculate_ewma_quality
)

plt.style.use('seaborn-v0_8')

class SimpleModelTracker:
    def __init__(self, tracking_metric='Fold_AdjSharpe_Q', threshold=0.7, top_k=3):
        self.tracking_metric = tracking_metric
        self.threshold = threshold
        self.top_k = top_k
        self.model_history = {}
        self.top_models = []
        self.backtest_start = None
        self.is_backtesting = False
        
    def add_results(self, fold_idx, results, total_folds):
        # Store results
        for result in results:
            model_id = result['Model']
            if model_id not in self.model_history:
                self.model_history[model_id] = []
            self.model_history[model_id].append(result)
        
        # Check threshold
        if not self.is_backtesting and (fold_idx + 1) / total_folds >= self.threshold:
            self.start_backtest(fold_idx + 1)
    
    def start_backtest(self, fold_idx):
        self.is_backtesting = True
        self.backtest_start = fold_idx
        
        # Calculate avg performance and select top K
        performance = {}
        for model_id, history in self.model_history.items():
            if history:
                avg = statistics.mean([h.get(self.tracking_metric, 0.0) for h in history])
                performance[model_id] = avg
        
        if performance:
            sorted_models = sorted(performance.items(), key=lambda x: x[1], reverse=True)
            self.top_models = [m for m, _ in sorted_models[:self.top_k]]
            
            logging.info(f"BACKTESTING STARTED at fold {fold_idx}")
            logging.info(f"=== MODEL SELECTION DEBUG ===")
            logging.info(f"Total models evaluated: {len(performance)}")
            logging.info(f"TOP 10 models by {self.tracking_metric}:")
            for i, (model_id, avg_q) in enumerate(sorted_models[:10]):
                marker = " <- SELECTED" if model_id in self.top_models else ""
                logging.info(f"  #{i+1}: {model_id} = {avg_q:.4f}{marker}")
            
            # Check for expected models (M11, M20, M30)
            expected_models = ['M11', 'M20', 'M30']
            logging.info(f"Expected models check:")
            for model in expected_models:
                if model in performance:
                    rank = [m for m, _ in sorted_models].index(model) + 1
                    logging.info(f"  {model}: Q={performance[model]:.4f}, Rank=#{rank}")
                else:
                    logging.info(f"  {model}: NOT FOUND in model history")
            
            logging.info(f"FINAL SELECTION: TOP {self.top_k} = {self.top_models}")
            logging.info(f"=== END MODEL SELECTION DEBUG ===")

def save_results(signals_dict, y, output_dir, timestamp, tracker=None):
    """Save ensemble results with proper lag and BACKTEST-PERIOD-ONLY ensemble handling."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ensemble signal for VISUALIZATION (full timeline) vs METRICS (backtest-only)
    ensemble_full = pd.Series(0.0, index=y.index)  # For charts - full timeline
    ensemble_backtest = pd.Series(0.0, index=y.index)  # For metrics - backtest-only
    
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
                signal_aligned.iloc[:backtest_start_index] = 0.0  # Zero pre-backtest
                ensemble_backtest += signal_aligned
            ensemble_backtest /= len(selected_signals)
            
            logging.info(f"DUAL ENSEMBLE: Full timeline for charts, backtest-only (index {backtest_start_index}+) for metrics")
        
        # Use backtest-only ensemble as primary for metrics calculation
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
    
    # Save to CSV with both ensemble versions for analysis
    df = pd.DataFrame({
        'ensemble_signal': ensemble,  # Backtest-only for metrics
        'ensemble_full': ensemble_full if 'ensemble_full' in locals() else ensemble,  # Full timeline for charts
        'target_ret': y,
        'pnl': pnl,
        'equity': equity
    })
    
    # Add individual signals (ALL models for analysis)
    for model_id, signal in signals_dict.items():
        df[f'{model_id}_signal'] = signal.reindex_like(y).fillna(0.0)
    
    path = os.path.join(output_dir, f'ensemble_results_{timestamp}.csv')
    df.to_csv(path, index=True)
    return df, path

def create_charts(tracker, ensemble_df, output_dir):
    """Enhanced 2x2 chart layout with PnL curves and performance table."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create custom layout
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.6], width_ratios=[1, 1, 1])
    
    fig.suptitle(f'Top {tracker.top_k} XGBoost Model Performance Analysis', fontsize=16, fontweight='bold')
    
    models_to_show = tracker.top_models if tracker.top_models else list(tracker.model_history.keys())[:tracker.top_k]
    
    # Chart 1: Tracking metric evolution (top left) - Show fold progression not dates
    ax1 = fig.add_subplot(gs[0, 0])
    for model_id in models_to_show:
        history = tracker.model_history.get(model_id, [])
        if history:
            folds = list(range(1, len(history)+1))  # Start from fold 1 for clarity
            values = [h.get(tracker.tracking_metric, 0.0) for h in history]
            style = '-o' if model_id in tracker.top_models else '--'
            alpha = 1.0 if model_id in tracker.top_models else 0.5
            linewidth = 2 if model_id in tracker.top_models else 1
            ax1.plot(folds, values, style, label=model_id, alpha=alpha, linewidth=linewidth)
    
    if tracker.backtest_start:
        ax1.axvline(x=tracker.backtest_start, color='red', linestyle='--', 
                   label='Backtest Start', linewidth=2, alpha=0.8)
    
    ax1.set_title(f'{tracker.tracking_metric} Evolution (Full History)', fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Fold Number')
    ax1.set_ylabel('Q Value')
    
    # Chart 2: Sharpe evolution (top middle) - Show full fold progression
    ax2 = fig.add_subplot(gs[0, 1])
    for model_id in models_to_show:
        history = tracker.model_history.get(model_id, [])
        if history:
            folds = list(range(1, len(history)+1))  # Start from fold 1 for clarity
            sharpe = [h.get('OOS_Sharpe', 0.0) for h in history]
            style = '-o' if model_id in tracker.top_models else '--'
            alpha = 1.0 if model_id in tracker.top_models else 0.5
            linewidth = 2 if model_id in tracker.top_models else 1
            ax2.plot(folds, sharpe, style, label=model_id, alpha=alpha, linewidth=linewidth)
    
    if tracker.backtest_start:
        ax2.axvline(x=tracker.backtest_start, color='red', linestyle='--', 
                   linewidth=2, alpha=0.8)
    
    ax2.set_title('OOS Sharpe Evolution (Full History)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Fold Number')
    ax2.set_ylabel('Sharpe Ratio')
    
    # Chart 3: Ensemble PnL Curve (top right) - FULL TIMELINE for complete history
    ax3 = fig.add_subplot(gs[0, 2])
    if ensemble_df is not None and 'ensemble_full' in ensemble_df.columns:
        dates = pd.to_datetime(ensemble_df.index)
        # Use full timeline ensemble for complete PnL curve visualization
        full_pnl = (ensemble_df['ensemble_full'].shift(1).fillna(0.0) * ensemble_df['target_ret']).cumsum()
        equity_curve = full_pnl.values
        
        # Color the curve based on backtest start
        if tracker.backtest_start:
            # DYNAMIC THRESHOLD FIX: Use actual threshold instead of hardcoded 0.6
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            
            # Training phase
            ax3.plot(dates[:backtest_start_idx], equity_curve[:backtest_start_idx], 
                    color='blue', linewidth=2, label='Training Phase', alpha=0.7)
            
            # Backtesting phase
            ax3.plot(dates[backtest_start_idx:], equity_curve[backtest_start_idx:], 
                    color='red', linewidth=2, label='Backtest Phase')
            
            ax3.axvline(x=dates[backtest_start_idx], color='red', linestyle='--', 
                       alpha=0.6, label='Backtest Start')
        else:
            ax3.plot(dates, equity_curve, color='blue', linewidth=2, label='Equity Curve')
        
        ax3.set_title('Ensemble Equity Curve', fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative PnL')
        
        # Format x-axis
        ax3.tick_params(axis='x', rotation=45)
    
    # Chart 4: Individual Model PnL Curves (bottom left and middle, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    if ensemble_df is not None and tracker.top_models:
        dates = pd.to_datetime(ensemble_df.index)
        colors = ['gold', 'silver', '#CD7F32']  # Top 3 colors
        
        for i, model_id in enumerate(tracker.top_models):  # Show all 3 selected models
            col_name = f'{model_id}_signal'
            if col_name in ensemble_df.columns:
                # CORRECTED: Show FULL timeline PnL for complete visibility (training + backtest)
                # This gives full context while performance table shows backtest-only metrics
                model_pnl = (ensemble_df[col_name].shift(1).fillna(0.0) * ensemble_df['target_ret']).cumsum()
                ax4.plot(dates, model_pnl, linewidth=2, label=f'{model_id}', 
                        color=colors[i % len(colors)], alpha=0.8)
        
        if tracker.backtest_start:
            # DYNAMIC THRESHOLD FIX: Use actual threshold instead of hardcoded 0.6
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            ax4.axvline(x=dates[backtest_start_idx], color='red', linestyle='--', 
                       alpha=0.6, label='Backtest Start')
        
        ax4.set_title(f'Individual Model PnL Curves (Top {len(tracker.top_models)} Selected)', fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Cumulative PnL')
        ax4.tick_params(axis='x', rotation=45)
    
    # Chart 5: Performance Summary Table (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')  # Turn off axis for table
    
    if tracker.top_models:
        # Calculate performance stats for each selected model
        table_data = []
        headers = ['Model', 'Avg Sharpe', 'Avg Hit %', 'Avg Q']
        
        for model_id in tracker.top_models:
            history = tracker.model_history.get(model_id, [])
            if history:
                avg_sharpe = statistics.mean([h.get('OOS_Sharpe', 0.0) for h in history])
                avg_hit = statistics.mean([h.get('OOS_Hit', 0.0) for h in history]) * 100
                avg_q = statistics.mean([h.get(tracker.tracking_metric, 0.0) for h in history])
                table_data.append([model_id, f'{avg_sharpe:.3f}', f'{avg_hit:.1f}%', f'{avg_q:.3f}'])
        
        # Add ensemble row - BACKTEST PERIOD ONLY for performance table
        if ensemble_df is not None and tracker.backtest_start:
            # Calculate BACKTEST-ONLY metrics for performance table
            backtest_start_idx = int(len(ensemble_df) * tracker.threshold)
            backtest_pnl = ensemble_df['pnl'].iloc[backtest_start_idx:]
            backtest_ensemble = ensemble_df['ensemble_signal'].iloc[backtest_start_idx:]
            backtest_target = ensemble_df['target_ret'].iloc[backtest_start_idx:]
            
            if len(backtest_pnl) > 10:  # Ensure sufficient data
                ens_sharpe = calculate_annualized_sharpe(backtest_pnl)
                ens_hit = calculate_hit_rate(backtest_ensemble, backtest_target) * 100
            else:
                ens_sharpe = 0.0
                ens_hit = 0.0
                
            table_data.append(['ENSEMBLE [BACKTEST]', f'{ens_sharpe:.3f}', f'{ens_hit:.1f}%', 'N/A'])
        
        # Create table
        table = ax5.table(cellText=table_data, colLabels=headers, 
                         cellLoc='center', loc='center',
                         colColours=['lightgray'] * len(headers))
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        for i, (key, cell) in enumerate(table.get_celld().items()):
            cell.set_linewidth(1)
            if key[0] == 0:  # Header row
                cell.set_facecolor('lightsteelblue')
                cell.set_text_props(weight='bold')
            elif key[0] == len(table_data):  # Ensemble row
                cell.set_facecolor('lightyellow')
                cell.set_text_props(weight='bold')
        
        ax5.set_title('Performance Summary', fontweight='bold', pad=20)
    
    # Add summary statistics at the bottom
    summary_ax = fig.add_subplot(gs[2, :])
    summary_ax.axis('off')
    
    summary_text = f"Analysis: {tracker.top_k} models selected at fold {tracker.backtest_start} using {tracker.tracking_metric}\n"
    summary_text += f"Q-Metric Half-Life: ~6.6 periods (α=0.1) | Backtest Threshold: {tracker.threshold*100:.0f}%"
    
    if ensemble_df is not None:
        total_return = ensemble_df['pnl'].sum()
        max_dd = (ensemble_df['equity'].expanding().max() - ensemble_df['equity']).max()
        summary_text += f" | Total Return: {total_return:.4f} | Max DD: {max_dd:.4f}"
    
    summary_ax.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'enhanced_performance.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    return chart_path

def run_analysis(X, y, n_models, n_folds, tracker, logger):
    """Main analysis loop - simplified."""
    specs = generate_xgb_specs(n_models)
    fold_splits = list(wfo_splits(n=len(X), k_folds=n_folds, min_train=50))
    
    # Track signals for ensemble
    ensemble_signals = {f'M{i:02d}': pd.Series(0.0, index=X.index) for i in range(n_models)}
    
    # History for EWMA calculation  
    model_quality_history = {
        'oos_sharpe': [[] for _ in range(n_models)],
        'oos_hit': [[] for _ in range(n_models)],
        'oos_ir': [[] for _ in range(n_models)],
        'oos_adj_sharpe': [[] for _ in range(n_models)]
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info(f"FOLD {fold_idx+1}/{len(fold_splits)}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        fold_results = []
        for model_idx, spec in enumerate(specs):
            # Train and predict
            model = fit_xgb_on_slice(X_train, y_train, spec)
            pred_test = model.predict(X_test.values)
            norm_pred = normalize_predictions(pd.Series(pred_test, index=X_test.index))
            signal_shifted = norm_pred.shift(1).fillna(0.0)
            
            # Calculate metrics
            oos_returns = signal_shifted * y_test
            oos_sharpe = calculate_annualized_sharpe(oos_returns)
            oos_hit = calculate_hit_rate(signal_shifted, y_test)
            oos_ir = calculate_information_ratio(oos_returns)
            oos_adj_sharpe = calculate_adjusted_sharpe(oos_returns, norm_pred)
            
            # Track all quality histories
            model_quality_history['oos_sharpe'][model_idx].append(oos_sharpe)
            model_quality_history['oos_hit'][model_idx].append(oos_hit)
            model_quality_history['oos_ir'][model_idx].append(oos_ir)
            model_quality_history['oos_adj_sharpe'][model_idx].append(oos_adj_sharpe)
            
            # Calculate Q metrics for all types
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
                
                fold_sharpe_q = calculate_ewma_quality(sharpe_series, 0.1)
                fold_hit_q = calculate_ewma_quality(hit_series, 0.1)
                fold_ir_q = calculate_ewma_quality(ir_series, 0.1)
                fold_adj_sharpe_q = calculate_ewma_quality(adj_sharpe_series, 0.1)
            
            fold_results.append({
                'Model': f'M{model_idx:02d}',
                'OOS_Sharpe': oos_sharpe,
                'OOS_Hit': oos_hit,
                'OOS_IR': oos_ir,
                'OOS_AdjSharpe': oos_adj_sharpe,
                'Fold_Sharpe_Q': fold_sharpe_q,
                'Fold_Hit_Q': fold_hit_q,
                'Fold_IR_Q': fold_ir_q,
                'Fold_AdjSharpe_Q': fold_adj_sharpe_q
            })
            
            # Store signal for ensemble - CRITICAL FIX: Store signals for ALL models throughout entire timeline
            # This ensures we capture full performance history and don't lose 60% of trading opportunities
            ensemble_signals[f'M{model_idx:02d}'].iloc[test_idx] = signal_shifted
        
        # Add to tracker
        tracker.add_results(fold_idx, fold_results, len(fold_splits))
        
        # Log fold summary
        df = pd.DataFrame(fold_results)
        logger.info(f"  Mean Sharpe: {df['OOS_Sharpe'].mean():.3f}, Mean {tracker.tracking_metric}: {df[tracker.tracking_metric].mean():.3f}")
    
    # Filter signals to only selected models for ensemble creation
    # Use signals from backtest phase onwards based on when models were selected
    selected_signals = {}
    if tracker.is_backtesting and tracker.top_models:
        for model_id in tracker.top_models:
            if model_id in ensemble_signals:
                selected_signals[model_id] = ensemble_signals[model_id]
        
        # If no models selected yet, return all signals for analysis
        if not selected_signals:
            selected_signals = ensemble_signals
    else:
        # Before backtesting, return all signals for analysis purposes
        selected_signals = ensemble_signals
    
    return selected_signals

def main():
    parser = argparse.ArgumentParser(description='Simplified Top-5 XGBoost Backtesting')
    parser.add_argument('--target_symbol', default='@ES#C', help='Target symbol')
    parser.add_argument('--start_date', default='2022-01-01', help='Start date')
    parser.add_argument('--end_date', default='2024-01-01', help='End date')
    parser.add_argument('--n_models', type=int, default=10, help='Number of models')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of folds')
    parser.add_argument('--max_features', type=int, default=50, help='Max features')
    parser.add_argument('--tracking_metric', default='Fold_Sharpe_Q', 
                       choices=['Fold_Sharpe_Q', 'Fold_Hit_Q', 'Fold_IR_Q', 'Fold_AdjSharpe_Q'])
    parser.add_argument('--threshold', type=float, default=0.7, help='Backtest threshold')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top models to select')
    
    args = parser.parse_args()
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, f'simple_backtest_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=== XGBoost Top-5 Simplified Backtesting ===")
    logger.info(f"Symbol: {args.target_symbol}, Models: {args.n_models}, Folds: {args.n_folds}")
    logger.info(f"Tracking: {args.tracking_metric}, Threshold: {args.threshold*100:.1f}%, Top-K: {args.top_k}")
    
    try:
        # Load and prepare data
        df = prepare_real_data_simple(args.target_symbol, start_date=args.start_date, end_date=args.end_date)
        target_col = f"{args.target_symbol}_target_return"
        X, y = df[[c for c in df.columns if c != target_col]], df[target_col]
        
        if args.max_features < X.shape[1]:
            X = apply_feature_selection(X, y, method='block_wise', max_total_features=args.max_features)
        
        logger.info(f"Data: X={X.shape}, Y={y.shape}")
        
        # Run analysis
        tracker = SimpleModelTracker(args.tracking_metric, args.threshold, args.top_k)
        signals = run_analysis(X, y, args.n_models, args.n_folds, tracker, logger)
        
        # Save results and create charts
        if signals:
            results_df, csv_path = save_results(signals, y, log_dir, timestamp, tracker)
            chart_path = create_charts(tracker, results_df, log_dir)
            
            # CRITICAL FIX: Calculate metrics ONLY on backtest period 
            if tracker and tracker.is_backtesting and tracker.backtest_start:
                backtest_start_index = int(len(y) * tracker.threshold)
                
                # Extract ONLY backtest period data for true performance measurement
                backtest_pnl = results_df['pnl'].iloc[backtest_start_index:]
                backtest_ensemble = results_df['ensemble_signal'].iloc[backtest_start_index:]
                backtest_target = results_df['target_ret'].iloc[backtest_start_index:]
                
                # Calculate TRUE backtest metrics
                backtest_sharpe = calculate_annualized_sharpe(backtest_pnl)
                backtest_hit = calculate_hit_rate(backtest_ensemble, backtest_target)
                backtest_return = backtest_pnl.sum()
                
                # Also calculate full timeline for comparison
                full_sharpe = calculate_annualized_sharpe(results_df['pnl'])
                full_hit = calculate_hit_rate(results_df['ensemble_signal'], results_df['target_ret'])
                
                logger.info("=== FINAL RESULTS ===")
                logger.info(f"Selected Models: {tracker.top_models}")
                logger.info(f"Backtest Start Index: {backtest_start_index} ({tracker.threshold*100:.0f}% threshold)")
                logger.info(f"TRUE BACKTEST Sharpe: {backtest_sharpe:.3f} <- PURE OOS PERFORMANCE")
                logger.info(f"TRUE BACKTEST Hit Rate: {backtest_hit:.2%} <- PURE OOS PERFORMANCE")
                logger.info(f"TRUE BACKTEST Return: {backtest_return:.4f} <- PURE OOS PERFORMANCE")
                logger.info(f"Full Timeline Sharpe: {full_sharpe:.3f} (includes zeros)")
                logger.info(f"Backtest Period Length: {len(backtest_pnl)} observations")
            else:
                # Fallback to full timeline if no backtest period defined
                final_sharpe = calculate_annualized_sharpe(results_df['pnl'])
                final_hit = calculate_hit_rate(results_df['ensemble_signal'], results_df['target_ret'])
                
                logger.info("=== FINAL RESULTS ===")
                logger.info(f"Ensemble Sharpe: {final_sharpe:.3f}")
                logger.info(f"Ensemble Hit Rate: {final_hit:.2%}")
                logger.info(f"Total Return: {results_df['pnl'].sum():.4f}")
            logger.info(f"Results: {csv_path}")
            logger.info(f"Charts: {chart_path}")
        
        logger.info("Analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == "__main__":
    main()
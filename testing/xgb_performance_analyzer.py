#!/usr/bin/env python
"""
XGBoost Performance Analyzer with Configurable Parameters
Shows IIS, IVal, OOS performance with stability scores using clean ASCII tables
"""

import sys
import numpy as np
import pandas as pd
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import xgboost as xgb

# Local imports
from data.data_utils_simple import prepare_real_data_simple
from data.symbol_loader import get_default_symbols
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, detect_gpu
from model.feature_selection import apply_feature_selection
from cv.wfo import wfo_splits

def setup_logging(log_label: str, max_features: int, n_folds: int, xgb_type: str) -> logging.Logger:
    """Setup logging with labeled filename"""
    log_file = f"logs/xgb_performance_{log_label}_{xgb_type}_{max_features}feat_{n_folds}folds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path("logs").mkdir(exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    return logger

def calculate_sharpe(returns: pd.Series) -> float:
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

def calculate_hit_rate(predictions: pd.Series, targets: pd.Series) -> float:
    """Calculate hit rate (percentage of correct direction predictions)"""
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    pred_dir = np.sign(predictions)
    target_dir = np.sign(targets)
    return np.mean(pred_dir == target_dir)

def calculate_information_ratio(returns: pd.Series) -> float:
    """Calculate information ratio (annualized)"""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))

def normalize_predictions(predictions: pd.Series) -> pd.Series:
    """Normalize predictions to [-1,1] range using z-score + tanh"""
    if len(predictions) == 0 or predictions.std() == 0:
        return pd.Series(np.zeros(len(predictions)), index=predictions.index)
    
    # Z-score normalization
    z_scores = (predictions - predictions.mean()) / predictions.std()
    # Tanh squashing to [-1, 1] range
    normalized = np.tanh(z_scores)
    return pd.Series(normalized, index=predictions.index)

def calculate_adjusted_sharpe(returns: pd.Series, predictions: pd.Series, lambda_turnover: float = 0.1) -> float:
    """Calculate adjusted Sharpe ratio with turnover penalty"""
    if len(returns) == 0 or len(predictions) == 0:
        return 0.0
    
    # Calculate base Sharpe
    base_sharpe = calculate_sharpe(returns)
    
    # Calculate turnover penalty
    normalized_preds = normalize_predictions(predictions)
    turnover = normalized_preds.diff().abs().fillna(0.0).mean()
    
    # Adjusted Sharpe = Base Sharpe - λ * Turnover
    return base_sharpe - lambda_turnover * turnover

def calculate_dapy_binary(returns: pd.Series, predictions: pd.Series) -> float:
    """Calculate DAPY using binary hit/miss approach"""
    if len(returns) == 0 or len(predictions) == 0:
        return 0.0
    
    # Binary hit rate approach
    pred_signs = np.sign(predictions)
    ret_signs = np.sign(returns)
    hits = (pred_signs == ret_signs).astype(float)
    
    # Annualized percentage of correct predictions
    return (hits.mean() - 0.5) * 252 * 100  # Excess over random (50%) annualized

def calculate_dapy_both(returns: pd.Series, predictions: pd.Series) -> float:
    """Calculate DAPY using both magnitude and direction"""
    if len(returns) == 0 or len(predictions) == 0:
        return 0.0
    
    # Combine direction accuracy with magnitude correlation
    direction_acc = calculate_hit_rate(predictions, returns) - 0.5  # Excess over 50%
    
    # Magnitude correlation (using absolute values)
    abs_preds = np.abs(predictions)
    abs_rets = np.abs(returns)
    
    if len(abs_preds) > 1 and abs_preds.std() > 0 and abs_rets.std() > 0:
        magnitude_corr = np.corrcoef(abs_preds, abs_rets)[0, 1]
        magnitude_corr = magnitude_corr if not np.isnan(magnitude_corr) else 0.0
    else:
        magnitude_corr = 0.0
    
    # Combined score (direction + magnitude) annualized
    return (direction_acc + 0.3 * magnitude_corr) * 252 * 100

def calculate_cb_ratio(returns: pd.Series) -> float:
    """Calculate Calmar-Bliss ratio (Ann Return / Max Drawdown)"""
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns (equity curve)
    equity = returns.cumsum()
    
    # Calculate running maximum (peak)
    running_max = equity.expanding().max()
    
    # Calculate drawdown
    drawdown = equity - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.0
    
    # Annualized return
    ann_return = returns.mean() * 252
    
    # CB ratio = Annual Return / Max Drawdown
    return ann_return / max_drawdown if max_drawdown > 0 else 0.0

def calculate_oos_pvalue(returns: pd.Series, n_shuffles: int = 100) -> float:
    """Calculate OOS p-value using bootstrap shuffling"""
    if len(returns) <= 10:  # Need minimum samples
        return 1.0
    
    # Actual Sharpe ratio
    actual_sharpe = calculate_sharpe(returns)
    
    # Bootstrap shuffling
    better_count = 0
    for _ in range(n_shuffles):
        shuffled_returns = returns.sample(frac=1.0, replace=True)
        shuffled_sharpe = calculate_sharpe(shuffled_returns)
        if shuffled_sharpe >= actual_sharpe:
            better_count += 1
    
    # P-value = fraction of shuffles that beat actual performance
    return better_count / n_shuffles

def calculate_ann_return(returns: pd.Series) -> float:
    """Calculate annualized return"""
    if len(returns) == 0:
        return 0.0
    return returns.mean() * 252

def calculate_ann_vol(returns: pd.Series) -> float:
    """Calculate annualized volatility"""
    if len(returns) == 0:
        return 0.0
    return returns.std() * np.sqrt(252)

def calculate_ewma_quality(series: pd.Series, alpha: float = 0.1) -> float:
    """Calculate EWMA quality score for time series performance"""
    if len(series) == 0:
        return 0.0
    # Use pandas ewm for proper EWMA calculation
    ewma_vals = series.ewm(alpha=alpha, adjust=False).mean()
    return ewma_vals.iloc[-1] if len(ewma_vals) > 0 else 0.0

def calculate_sharpe_q(returns_series: List[pd.Series], alpha: float = 0.1) -> float:
    """Calculate EWMA of realized Sharpe ratios across time periods"""
    if not returns_series or len(returns_series) == 0:
        return 0.0
    
    sharpe_ratios = []
    for returns in returns_series:
        if len(returns) > 0:
            sharpe_ratios.append(calculate_sharpe(returns))
        else:
            sharpe_ratios.append(0.0)
    
    if len(sharpe_ratios) == 0:
        return 0.0
    
    sharpe_ts = pd.Series(sharpe_ratios)
    return calculate_ewma_quality(sharpe_ts, alpha)

def calculate_hitrate_q(predictions_series: List[pd.Series], targets_series: List[pd.Series], alpha: float = 0.1) -> float:
    """Calculate EWMA of realized hit rates across time periods"""
    if not predictions_series or not targets_series or len(predictions_series) != len(targets_series):
        return 0.0
    
    hit_rates = []
    for preds, targets in zip(predictions_series, targets_series):
        if len(preds) > 0 and len(targets) > 0:
            hit_rates.append(calculate_hit_rate(preds, targets))
        else:
            hit_rates.append(0.0)
    
    if len(hit_rates) == 0:
        return 0.0
    
    hit_ts = pd.Series(hit_rates)
    return calculate_ewma_quality(hit_ts, alpha)

def calculate_ir_q(returns_series: List[pd.Series], alpha: float = 0.1) -> float:
    """Calculate EWMA of realized information ratios across time periods"""
    if not returns_series or len(returns_series) == 0:
        return 0.0
    
    ir_ratios = []
    for returns in returns_series:
        if len(returns) > 0:
            ir_ratios.append(calculate_information_ratio(returns))
        else:
            ir_ratios.append(0.0)
    
    if len(ir_ratios) == 0:
        return 0.0
    
    ir_ts = pd.Series(ir_ratios)
    return calculate_ewma_quality(ir_ts, alpha)


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

def enhanced_fold_analysis(X: pd.DataFrame, y: pd.Series, xgb_specs: List[Dict], 
                          actual_folds: int, inner_val_frac: float, logger: logging.Logger, 
                          ewma_alpha: float = 0.1) -> Dict:
    """
    Perform enhanced fold analysis with IIS, IVal, OOS for each model per fold.
    Returns detailed performance breakdown with stability scores.
    """
    fold_splits = list(wfo_splits(n=len(X), k_folds=actual_folds, min_train=50))
    
    all_fold_results = {}
    
    # Track performance across folds for EWMA quality calculations
    model_performance_tracker = {
        'returns_series': {i: [] for i in range(len(xgb_specs))},
        'predictions_series': {i: [] for i in range(len(xgb_specs))},
        'targets_series': {i: [] for i in range(len(xgb_specs))}
    }
    
    logger.info(f"Running enhanced fold analysis with IIS/IVal/OOS breakdown...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"FOLD {fold_idx+1}/{len(fold_splits)}")
        logger.info("=" * 100)
        
        # Get fold data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        # Inner train/val split
        inner_split_point = int(len(X_train) * (1 - inner_val_frac))
        X_inner_tr, X_inner_val = X_train.iloc[:inner_split_point], X_train.iloc[inner_split_point:]
        y_inner_tr, y_inner_val = y_train.iloc[:inner_split_point], y_train.iloc[inner_split_point:]
        
        logger.info(f"Inner Train: {len(X_inner_tr)} samples, Inner Val: {len(X_inner_val)} samples")
        
        # Collect metrics for all models
        fold_results = []
        
        # For per-fold Q metrics, calculate quality up to this fold for each model
        fold_q_metrics = {}
        
        for model_idx, xgb_spec in enumerate(xgb_specs):
            # FRAMEWORK LOGIC: Train model on FULL training set (like line 139 in horse_race)
            model = xgb.XGBRegressor(**xgb_spec)
            model.fit(X_train.values, y_train.values)  # Train on full training data
            
            # Generate predictions using the SAME model (like lines 140-142 in horse_race)
            pred_inner_tr = pd.Series(model.predict(X_inner_tr.values), index=X_inner_tr.index)
            pred_inner_val = pd.Series(model.predict(X_inner_val.values), index=X_inner_val.index)
            pred_test = pd.Series(model.predict(X_test.values), index=X_test.index)
            
            # Normalize predictions to proper signal range [-1, 1]
            norm_pred_tr = normalize_predictions(pred_inner_tr)
            norm_pred_val = normalize_predictions(pred_inner_val)  
            norm_pred_test = normalize_predictions(pred_test)
            
            # Calculate performance metrics using normalized signals with proper timing
            # IMPORTANT: Shift signals forward to avoid look-ahead bias (like horse_race line 21)
            is_returns = norm_pred_tr.shift(1).fillna(0.0) * y_inner_tr
            iv_returns = norm_pred_val.shift(1).fillna(0.0) * y_inner_val  
            oos_returns = norm_pred_test.shift(1).fillna(0.0) * y_test
            
            # Sharpe ratios
            is_sharpe = calculate_sharpe(is_returns)
            iv_sharpe = calculate_sharpe(iv_returns)
            oos_sharpe = calculate_sharpe(oos_returns)
            
            # Hit rates
            is_hit = calculate_hit_rate(pred_inner_tr, y_inner_tr)
            iv_hit = calculate_hit_rate(pred_inner_val, y_inner_val)
            oos_hit = calculate_hit_rate(pred_test, y_test)
            
            # Annual returns
            is_ann_ret = calculate_ann_return(is_returns)
            iv_ann_ret = calculate_ann_return(iv_returns)
            oos_ann_ret = calculate_ann_return(oos_returns)
            
            # Annual volatility
            is_ann_vol = calculate_ann_vol(is_returns)
            iv_ann_vol = calculate_ann_vol(iv_returns)
            oos_ann_vol = calculate_ann_vol(oos_returns)
            
            # Information ratios
            is_ir = calculate_information_ratio(is_returns)
            iv_ir = calculate_information_ratio(iv_returns)
            oos_ir = calculate_information_ratio(oos_returns)
            
            # Adjusted Sharpe ratios (with turnover penalty)
            is_adj_sharpe = calculate_adjusted_sharpe(is_returns, norm_pred_tr)
            iv_adj_sharpe = calculate_adjusted_sharpe(iv_returns, norm_pred_val)
            oos_adj_sharpe = calculate_adjusted_sharpe(oos_returns, norm_pred_test)
            
            # DAPY metrics
            is_dapy_binary = calculate_dapy_binary(y_inner_tr, norm_pred_tr)
            iv_dapy_binary = calculate_dapy_binary(y_inner_val, norm_pred_val)
            oos_dapy_binary = calculate_dapy_binary(y_test, norm_pred_test)
            
            is_dapy_both = calculate_dapy_both(y_inner_tr, norm_pred_tr)
            iv_dapy_both = calculate_dapy_both(y_inner_val, norm_pred_val)
            oos_dapy_both = calculate_dapy_both(y_test, norm_pred_test)
            
            # CB Ratio (Calmar-Bliss)
            is_cb_ratio = calculate_cb_ratio(is_returns)
            iv_cb_ratio = calculate_cb_ratio(iv_returns)
            oos_cb_ratio = calculate_cb_ratio(oos_returns)
            
            # OOS P-values (statistical significance)
            oos_pvalue_sharpe = calculate_oos_pvalue(oos_returns)
            oos_pvalue_hit = calculate_oos_pvalue(oos_returns)  # Using returns for consistency
            
            # Store data for EWMA calculations
            model_performance_tracker['returns_series'][model_idx].append(oos_returns)
            model_performance_tracker['predictions_series'][model_idx].append(pred_test)
            model_performance_tracker['targets_series'][model_idx].append(y_test)
            
            # Calculate per-fold Q metrics (up to current fold)
            if fold_idx > 0:  # Need at least 2 folds for meaningful EWMA
                current_returns = model_performance_tracker['returns_series'][model_idx]
                current_predictions = model_performance_tracker['predictions_series'][model_idx]
                current_targets = model_performance_tracker['targets_series'][model_idx]
                
                fold_sharpe_q = calculate_sharpe_q(current_returns, ewma_alpha)
                fold_hit_q = calculate_hitrate_q(current_predictions, current_targets, ewma_alpha)
                fold_ir_q = calculate_ir_q(current_returns, ewma_alpha)
                
                fold_q_metrics[model_idx] = {
                    'Sharpe_Q': fold_sharpe_q,
                    'Hit_Q': fold_hit_q,
                    'IR_Q': fold_ir_q
                }
            else:
                # First fold: Q metrics = current metrics
                fold_q_metrics[model_idx] = {
                    'Sharpe_Q': oos_sharpe,
                    'Hit_Q': oos_hit,
                    'IR_Q': oos_ir
                }
            
            fold_results.append({
                'Model': f"M{model_idx:02d}",
                # Core Sharpe metrics
                'IS_Sharpe': is_sharpe,
                'IV_Sharpe': iv_sharpe,
                'OOS_Sharpe': oos_sharpe,
                'OOS_AdjSharpe': oos_adj_sharpe,
                # Hit rate metrics
                'IS_Hit': is_hit,
                'IV_Hit': iv_hit,
                'OOS_Hit': oos_hit,
                # Information Ratio
                'IS_IR': is_ir,
                'IV_IR': iv_ir,
                'OOS_IR': oos_ir,
                # DAPY metrics
                'OOS_DAPY_Binary': oos_dapy_binary,
                'OOS_DAPY_Both': oos_dapy_both,
                # CB Ratio
                'OOS_CB_Ratio': oos_cb_ratio,
                # Return/Vol metrics
                'OOS_Ann_Ret': oos_ann_ret,
                'OOS_Ann_Vol': oos_ann_vol,
                # P-values
                'OOS_PValue_Sharpe': oos_pvalue_sharpe,
                'OOS_PValue_Hit': oos_pvalue_hit,
                # EWMA Quality metrics
                'Fold_Sharpe_Q': fold_q_metrics[model_idx]['Sharpe_Q'],
                'Fold_Hit_Q': fold_q_metrics[model_idx]['Hit_Q'],
                'Fold_IR_Q': fold_q_metrics[model_idx]['IR_Q']
            })
        
        # TABLE 1: Core OOS Performance Metrics
        core_rows = []
        for result in fold_results:
            core_rows.append([
                result['Model'],
                f"{result['OOS_Sharpe']:.3f}",
                f"{result['OOS_AdjSharpe']:.3f}",
                f"{result['OOS_Hit']:.2%}",
                f"{result['OOS_IR']:.3f}",
                f"{result['OOS_Ann_Ret']:.4f}",
                f"{result['OOS_Ann_Vol']:.4f}"
            ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - CORE OOS PERFORMANCE:",
            ["Model", "Sharpe", "AdjSharpe", "Hit", "IR", "Ann_Ret", "Ann_Vol"],
            core_rows,
            logger
        )
        
        # TABLE 2: Advanced Metrics (DAPY, CB, P-values)
        advanced_rows = []
        for result in fold_results:
            advanced_rows.append([
                result['Model'],
                f"{result['OOS_DAPY_Binary']:.1f}",
                f"{result['OOS_DAPY_Both']:.1f}",
                f"{result['OOS_CB_Ratio']:.3f}",
                f"{result['OOS_PValue_Sharpe']:.3f}",
                f"{result['OOS_PValue_Hit']:.3f}"
            ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - ADVANCED METRICS:",
            ["Model", "DAPY_Bin", "DAPY_Both", "CB_Ratio", "P_Sharpe", "P_Hit"],
            advanced_rows,
            logger
        )
        
        # TABLE 3: Quality Metrics (EWMA)
        quality_rows = []
        for result in fold_results:
            quality_rows.append([
                result['Model'],
                f"{result['Fold_Sharpe_Q']:.3f}",
                f"{result['Fold_Hit_Q']:.3f}",
                f"{result['Fold_IR_Q']:.3f}",
                f"{result['OOS_Sharpe'] - result['Fold_Sharpe_Q']:.3f}",  # Quality drift
"↗" if result['Fold_Sharpe_Q'] > result['OOS_Sharpe'] else "↘"
            ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - QUALITY METRICS (EWMA):",
            ["Model", "Sharpe_Q", "Hit_Q", "IR_Q", "Drift", "Trend"],
            quality_rows,
            logger
        )
        
        # TABLE 4: IS/IV/OOS Progression (Compact)
        progression_rows = []
        for result in fold_results:
            progression_rows.append([
                result['Model'],
                f"{result['IS_Sharpe']:.2f}",
                f"{result['IV_Sharpe']:.2f}",
                f"{result['OOS_Sharpe']:.2f}",
                f"{result['IS_Sharpe'] - result['IV_Sharpe']:.2f}",  # IS->IV gap
                f"{result['IV_Sharpe'] - result['OOS_Sharpe']:.2f}"   # IV->OOS gap
            ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - PROGRESSION ANALYSIS:",
            ["Model", "IS", "IV", "OOS", "IS-IV", "IV-OOS"],
            progression_rows,
            logger
        )
        
        # Fold statistics
        fold_df = pd.DataFrame(fold_results)
        stats_rows = [
            ["Mean", f"{fold_df['IS_Sharpe'].mean():.3f}", f"{fold_df['IV_Sharpe'].mean():.3f}", 
             f"{fold_df['OOS_Sharpe'].mean():.3f}", f"{fold_df['IS_Hit'].mean():.2%}", f"{fold_df['IV_Hit'].mean():.2%}",
             f"{fold_df['OOS_Hit'].mean():.2%}", f"{fold_df['IS_IR'].mean():.3f}", f"{fold_df['IV_IR'].mean():.3f}", 
             f"{fold_df['OOS_IR'].mean():.3f}", f"{fold_df['OOS_Ann_Ret'].mean():.3f}", f"{fold_df['OOS_Ann_Vol'].mean():.3f}"],
            ["Std", f"{fold_df['IS_Sharpe'].std():.3f}", f"{fold_df['IV_Sharpe'].std():.3f}", 
             f"{fold_df['OOS_Sharpe'].std():.3f}", f"{fold_df['IS_Hit'].std():.3f}", f"{fold_df['IV_Hit'].std():.3f}",
             f"{fold_df['OOS_Hit'].std():.3f}", f"{fold_df['IS_IR'].std():.3f}", f"{fold_df['IV_IR'].std():.3f}", 
             f"{fold_df['OOS_IR'].std():.3f}", f"{fold_df['OOS_Ann_Ret'].std():.3f}", f"{fold_df['OOS_Ann_Vol'].std():.3f}"]
        ]
        
        print_simple_table(f"FOLD {fold_idx+1} STATISTICS:", ["Metric", "IS_Sharpe", "IV_Sharpe", "OOS_Sharpe", "IS_Hit", "IV_Hit", "OOS_Hit", "IS_IR", "IV_IR", "OOS_IR", "OOS_Ann_Ret", "OOS_Ann_Vol"], stats_rows, logger)
        
        # Performance degradation analysis
        is_iv_gap = fold_df['IS_Sharpe'] - fold_df['IV_Sharpe']
        iv_oos_gap = fold_df['IV_Sharpe'] - fold_df['OOS_Sharpe']
        
        logger.info(f"PERFORMANCE DEGRADATION ANALYSIS (FOLD {fold_idx+1}):")
        logger.info(f"IS->IV Gap:    Mean={is_iv_gap.mean():.3f}, Std={is_iv_gap.std():.3f}, Max={is_iv_gap.max():.3f}")
        logger.info(f"IV->OOS Gap:   Mean={iv_oos_gap.mean():.3f}, Std={iv_oos_gap.std():.3f}, Max={iv_oos_gap.max():.3f}")
        logger.info("")
        
        # Best models
        best_oos = fold_df.loc[fold_df['OOS_Sharpe'].idxmax()]
        best_hit = fold_df.loc[fold_df['OOS_Hit'].idxmax()]
        
        logger.info(f"BEST MODELS (FOLD {fold_idx+1}):")
        logger.info(f"Best OOS Sharpe: {best_oos['Model']} (OOS={best_oos['OOS_Sharpe']:.3f}, Hit={best_oos['OOS_Hit']:.2%})")
        logger.info(f"Best OOS Hit:    {best_hit['Model']} (Hit={best_hit['OOS_Hit']:.2%}, OOS={best_hit['OOS_Sharpe']:.3f})")
        logger.info("")
        logger.info("=" * 100)
        logger.info("")
        
        all_fold_results[f"fold_{fold_idx+1}"] = fold_results
    
    # Calculate EWMA quality metrics across all folds
    logger.info("")
    logger.info("")
    logger.info("="*100)
    logger.info("CROSS-FOLD ANALYSIS WITH EWMA QUALITY METRICS")
    logger.info("="*100)
    logger.info("")
    
    # Calculate quality metrics for each model
    model_quality_results = []
    for model_idx in range(len(xgb_specs)):
        returns_series = model_performance_tracker['returns_series'][model_idx]
        predictions_series = model_performance_tracker['predictions_series'][model_idx]
        targets_series = model_performance_tracker['targets_series'][model_idx]
        
        # EWMA Quality Metrics
        sharpe_q = calculate_sharpe_q(returns_series, ewma_alpha)
        hit_q = calculate_hitrate_q(predictions_series, targets_series, ewma_alpha)
        ir_q = calculate_ir_q(returns_series, ewma_alpha)
        
        # Overall aggregated metrics
        all_oos_returns = pd.concat(returns_series) if returns_series else pd.Series([])
        all_oos_preds = pd.concat(predictions_series) if predictions_series else pd.Series([])
        all_oos_targets = pd.concat(targets_series) if targets_series else pd.Series([])
        
        overall_sharpe = calculate_sharpe(all_oos_returns)
        overall_hit = calculate_hit_rate(all_oos_preds, all_oos_targets)
        overall_ir = calculate_information_ratio(all_oos_returns)
        overall_ann_ret = calculate_ann_return(all_oos_returns)
        overall_ann_vol = calculate_ann_vol(all_oos_returns)
        
        model_quality_results.append({
            'Model': f"M{model_idx:02d}",
            'Overall_Sharpe': overall_sharpe,
            'Overall_Hit': overall_hit,
            'Overall_IR': overall_ir,
            'Overall_Ann_Ret': overall_ann_ret,
            'Overall_Ann_Vol': overall_ann_vol,
            'Sharpe_Q': sharpe_q,
            'Hit_Q': hit_q,
            'IR_Q': ir_q,
            'Folds_Analyzed': len(returns_series)
        })
    
    # Sort and display top models by various quality metrics
    logger.info("\nTOP MODELS BY EWMA QUALITY METRICS:")
    
    # Top by Sharpe_Q
    top_sharpe_q = sorted(model_quality_results, key=lambda x: x['Sharpe_Q'], reverse=True)[:5]
    logger.info(f"\nTop 5 by Sharpe_Q (EWMA α={ewma_alpha}):")
    for i, model in enumerate(top_sharpe_q, 1):
        logger.info(f"  {i}. {model['Model']}: Sharpe_Q={model['Sharpe_Q']:.3f}, Overall_Sharpe={model['Overall_Sharpe']:.3f}, Hit={model['Overall_Hit']:.2%}")
    
    # Top by Hit_Q
    top_hit_q = sorted(model_quality_results, key=lambda x: x['Hit_Q'], reverse=True)[:5]
    logger.info(f"\nTop 5 by Hit_Q (EWMA α={ewma_alpha}):")
    for i, model in enumerate(top_hit_q, 1):
        logger.info(f"  {i}. {model['Model']}: Hit_Q={model['Hit_Q']:.3f}, Overall_Hit={model['Overall_Hit']:.2%}, Sharpe={model['Overall_Sharpe']:.3f}")
    
    # Top by IR_Q
    top_ir_q = sorted(model_quality_results, key=lambda x: x['IR_Q'], reverse=True)[:5]
    logger.info(f"\nTop 5 by IR_Q (EWMA α={ewma_alpha}):")
    for i, model in enumerate(top_ir_q, 1):
        logger.info(f"  {i}. {model['Model']}: IR_Q={model['IR_Q']:.3f}, Overall_IR={model['Overall_IR']:.3f}, Ann_Ret={model['Overall_Ann_Ret']:.3f}")
    
    # Comprehensive table of quality metrics
    quality_table_rows = []
    for result in sorted(model_quality_results, key=lambda x: x['Sharpe_Q'], reverse=True):
        quality_table_rows.append([
            result['Model'],
            f"{result['Overall_Sharpe']:.3f}",
            f"{result['Overall_Hit']:.2%}",
            f"{result['Overall_IR']:.3f}",
            f"{result['Overall_Ann_Ret']:.4f}",
            f"{result['Overall_Ann_Vol']:.4f}",
            f"{result['Sharpe_Q']:.3f}",
            f"{result['Hit_Q']:.3f}",
            f"{result['IR_Q']:.3f}",
            f"{result['Folds_Analyzed']}"
        ])
    
    print_simple_table(
        "COMPREHENSIVE QUALITY ANALYSIS (Sorted by Sharpe_Q):",
        ["Model", "Overall_Sharpe", "Overall_Hit", "Overall_IR", "Overall_Ann_Ret", "Overall_Ann_Vol", "Sharpe_Q", "Hit_Q", "IR_Q", "Folds"],
        quality_table_rows,
        logger
    )
    
    # Add quality results to return data
    all_fold_results['model_quality_analysis'] = model_quality_results
    
    return all_fold_results

def main():
    parser = argparse.ArgumentParser(description='XGBoost Performance Analysis')
    parser.add_argument('--max_features', type=int, default=50, help='Maximum features after selection')
    parser.add_argument('--n_folds', type=int, default=6, help='Number of cross-validation folds')
    parser.add_argument('--n_models', type=int, default=50, help='Number of XGBoost models')
    parser.add_argument('--xgb_type', choices=['standard', 'tiered', 'deep'], default='standard', help='XGBoost architecture type')
    parser.add_argument('--log_label', type=str, default='test', help='Label for log filename')
    parser.add_argument('--target_symbol', type=str, default='@ES#C', help='Target symbol')
    parser.add_argument('--start_date', type=str, default='2014-01-01', help='Start date')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date')
    parser.add_argument('--no_feature_selection', action='store_true', help='Skip feature selection and use all features')
    parser.add_argument('--ewma_alpha', type=float, default=0.1, help='EWMA alpha parameter for quality metrics')
    
    args = parser.parse_args()
    
    # Setup logging with clear filename
    logger = setup_logging(args.log_label, args.max_features, args.n_folds, args.xgb_type)
    
    logger.info("=" * 80)
    logger.info("XGBoost Performance Analysis")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info(f"  Target Symbol: {args.target_symbol}")
    logger.info(f"  Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"  Models: {args.n_models}")
    logger.info(f"  Folds: {args.n_folds}")
    logger.info(f"  Max Features: {args.max_features}")
    logger.info(f"  XGBoost Type: {args.xgb_type}")
    logger.info(f"  Inner Validation Fraction: 0.2")
    
    # Load data
    logger.info("Loading data...")
    symbols = get_default_symbols()
    df = prepare_real_data_simple(args.target_symbol, symbols, args.start_date, args.end_date)
    
    # Prepare features and target
    possible_target_cols = [col for col in df.columns if 'return' in col.lower() and args.target_symbol.replace('@', '').replace('#', '') in col]
    if not possible_target_cols:
        possible_target_cols = [col for col in df.columns if 'return' in col.lower()]
    
    if not possible_target_cols:
        raise ValueError(f"No return column found for {args.target_symbol}")
    
    target_col = possible_target_cols[0]
    logger.info(f"Using target column: {target_col}")
    feature_cols = [col for col in df.columns if col != target_col and col != 'date']
    
    # Handle NaN values
    df_clean = df.dropna(subset=[target_col])
    nan_count = df_clean[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values in feature columns after cleaning")
        df_clean = df_clean.ffill().fillna(0)
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    logger.info(f"Data shape before feature selection: X={X.shape}, y={y.shape}")
    
    # Apply feature selection
    if args.no_feature_selection:
        logger.info("Skipping feature selection - using all features")
        logger.info(f"Using all {X.shape[1]} features")
    else:
        logger.info("Applying block-wise feature selection...")
        X = apply_feature_selection(X, y, method='block_wise', 
                                   max_total_features=args.max_features, 
                                   corr_threshold=0.7)
        logger.info(f"Features after selection: {X.shape[1]}")
    
    logger.info(f"Data shape after feature processing: X={X.shape}, y={y.shape}")
    
    # Date range info
    if 'date' in df_clean.columns:
        logger.info(f"Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    else:
        logger.info(f"Date range: Index {df_clean.index.min()} to {df_clean.index.max()}")
    
    # Generate XGBoost model specifications based on type
    logger.info(f"Generating {args.n_models} {args.xgb_type} XGBoost models...")
    
    if args.xgb_type == 'tiered':
        # Use stratified_xgb_bank for tiered architecture - need feature columns
        xgb_specs, _ = stratified_xgb_bank(X.columns.tolist(), n_models=args.n_models)
    elif args.xgb_type == 'deep':
        xgb_specs = generate_deep_xgb_specs(n_models=args.n_models)
    else:  # standard
        xgb_specs = generate_xgb_specs(n_models=args.n_models)
    
    # GPU detection and update XGB specs for GPU usage
    gpu_device = detect_gpu()
    if gpu_device == "cuda":
        logger.info("Updating XGBoost specs to use GPU")
        # Update all specs to use GPU
        for spec in xgb_specs:
            spec.update({
                'tree_method': 'hist',
                'device': 'cuda'
            })
    else:
        logger.info("Using CPU for XGBoost training")
    
    # Run analysis
    results = enhanced_fold_analysis(X, y, xgb_specs, args.n_folds, 0.2, logger, args.ewma_alpha)
    
    logger.info("Analysis completed successfully!")
    
    # Print log file location
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger.info(f"Log file saved to: {handler.baseFilename}")

if __name__ == "__main__":
    main()
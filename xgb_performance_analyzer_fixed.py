#!/usr/bin/env python3
"""
Fixed XGBoost Performance Analysis with reorganized table structure
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats

# Import existing modules
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice
from cv.wfo import wfo_splits

# Existing metric functions
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

def enhanced_fold_analysis(X: pd.DataFrame, y: pd.Series, xgb_specs: List[Dict], 
                          actual_folds: int, inner_val_frac: float, logger: logging.Logger, 
                          ewma_alpha: float = 0.1) -> Dict:
    """Enhanced fold analysis with reorganized table structure"""
    
    fold_splits = list(wfo_splits(n=len(X), k_folds=actual_folds, min_train=50))
    all_fold_results = {}
    
    logger.info(f"Running enhanced fold analysis with reorganized tables...")
    
    for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
        logger.info("")
        logger.info("=" * 100)
        logger.info(f"FOLD {fold_idx+1}/{len(fold_splits)}")
        logger.info("=" * 100)
        
        # Get fold data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner train-validation split
        inner_split_point = int(len(X_train) * (1.0 - inner_val_frac))
        X_inner_train = X_train.iloc[:inner_split_point]
        X_inner_val = X_train.iloc[inner_split_point:]
        y_inner_train = y_train.iloc[:inner_split_point]
        y_inner_val = y_train.iloc[inner_split_point:]
        
        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Inner Train: {len(X_inner_train)} samples, Inner Val: {len(X_inner_val)} samples")
        
        # Train models and get predictions
        fold_results = []
        for model_idx, spec in enumerate(xgb_specs):
            model = fit_xgb_on_slice(X_train, y_train, spec)
            
            # Predictions
            pred_inner_train = model.predict(X_inner_train.values)
            pred_inner_val = model.predict(X_inner_val.values)
            pred_test = model.predict(X_test.values)
            
            # Normalize predictions
            norm_pred_inner_train = normalize_predictions(pd.Series(pred_inner_train, index=X_inner_train.index))
            norm_pred_inner_val = normalize_predictions(pd.Series(pred_inner_val, index=X_inner_val.index))
            norm_pred_test = normalize_predictions(pd.Series(pred_test, index=X_test.index))
            
            # IMPORTANT: Fix signal shift logic - shift signal first, then multiply
            signal_shifted = norm_pred_test.shift(1).fillna(0.0)
            oos_returns = signal_shifted * y_test
            
            # Calculate all metrics
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
            oos_ann_ret = oos_returns.mean() * 252
            oos_ann_vol = oos_returns.std() * np.sqrt(252)
            oos_adj_sharpe = calculate_adjusted_sharpe(oos_returns, norm_pred_test)
            oos_cb_ratio = calculate_cb_ratio(oos_returns)
            oos_dapy_binary = calculate_dapy_binary(signal_shifted, y_test)
            oos_dapy_both = calculate_dapy_both(signal_shifted, y_test)
            
            # P-values
            oos_pvalue_sharpe = bootstrap_pvalue(oos_sharpe, oos_returns, signal_shifted, 
                                               lambda p, r: calculate_annualized_sharpe(p * r))
            oos_pvalue_hit = bootstrap_pvalue(oos_hit, y_test, signal_shifted, 
                                            calculate_hit_rate)
            oos_pvalue_ir = bootstrap_pvalue(oos_ir, oos_returns, signal_shifted,
                                           lambda p, r: calculate_information_ratio(p * r))
            
            # Quality metrics (for single fold, these will be same as current metrics)
            fold_sharpe_q = oos_sharpe  # For single fold
            fold_hit_q = oos_hit
            fold_ir_q = oos_ir
            
            fold_results.append({
                'Model': f"M{model_idx:02d}",
                'IS_Sharpe': is_sharpe, 'IV_Sharpe': iv_sharpe, 'OOS_Sharpe': oos_sharpe,
                'IS_Hit': is_hit, 'IV_Hit': iv_hit, 'OOS_Hit': oos_hit,
                'IS_IR': is_ir, 'IV_IR': iv_ir, 'OOS_IR': oos_ir,
                'OOS_Ann_Ret': oos_ann_ret, 'OOS_Ann_Vol': oos_ann_vol,
                'OOS_AdjSharpe': oos_adj_sharpe, 'OOS_CB_Ratio': oos_cb_ratio,
                'OOS_DAPY_Binary': oos_dapy_binary, 'OOS_DAPY_Both': oos_dapy_both,
                'OOS_PValue_Sharpe': oos_pvalue_sharpe, 'OOS_PValue_Hit': oos_pvalue_hit, 'OOS_PValue_IR': oos_pvalue_ir,
                'Fold_Sharpe_Q': fold_sharpe_q, 'Fold_Hit_Q': fold_hit_q, 'Fold_IR_Q': fold_ir_q
            })
        
        # Create DataFrame for mean calculations
        fold_df = pd.DataFrame(fold_results)
        
        # TABLE 1: Sharpe Analysis 
        sharpe_rows = []
        for result in fold_results:
            sharpe_rows.append([
                result['Model'],
                f"{result['IS_Sharpe']:.3f}",
                f"{result['IV_Sharpe']:.3f}",
                f"{result['OOS_Sharpe']:.3f}",
                f"{result['OOS_PValue_Sharpe']:.3f}",
                f"{result['Fold_Sharpe_Q']:.3f}",
                f"{result['IS_Sharpe'] - result['IV_Sharpe']:.3f}",
                f"{result['IV_Sharpe'] - result['OOS_Sharpe']:.3f}"
            ])
        
        # Add mean row
        sharpe_rows.append([
            "MEAN",
            f"{fold_df['IS_Sharpe'].mean():.3f}",
            f"{fold_df['IV_Sharpe'].mean():.3f}",
            f"{fold_df['OOS_Sharpe'].mean():.3f}",
            f"{fold_df['OOS_PValue_Sharpe'].mean():.3f}",
            f"{fold_df['Fold_Sharpe_Q'].mean():.3f}",
            f"{(fold_df['IS_Sharpe'] - fold_df['IV_Sharpe']).mean():.3f}",
            f"{(fold_df['IV_Sharpe'] - fold_df['OOS_Sharpe']).mean():.3f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - SHARPE ANALYSIS:",
            ["Model", "IS_Sharpe", "IV_Sharpe", "OOS_Sharpe", "OOS_p", "Sharpe_Q", "IS-IV", "IV-OOS"],
            sharpe_rows,
            logger
        )
        logger.info("")
        
        # TABLE 2: Adjusted Sharpe Analysis
        adj_sharpe_rows = []
        for result in fold_results:
            adj_sharpe_rows.append([
                result['Model'],
                f"{result['IS_Sharpe']:.3f}",  # Use regular Sharpe for IS (no turnover)
                f"{result['IV_Sharpe']:.3f}",  # Use regular Sharpe for IV (no turnover)
                f"{result['OOS_AdjSharpe']:.3f}",
                f"{result['OOS_PValue_Sharpe']:.3f}",  # Same p-value as Sharpe
                f"{result['OOS_AdjSharpe']:.3f}",  # Use AdjSharpe as quality metric
                f"{result['IS_Sharpe'] - result['IV_Sharpe']:.3f}",
                f"{result['IV_Sharpe'] - result['OOS_AdjSharpe']:.3f}"  # IV to OOS AdjSharpe gap
            ])
        
        # Add mean row
        adj_sharpe_rows.append([
            "MEAN",
            f"{fold_df['IS_Sharpe'].mean():.3f}",
            f"{fold_df['IV_Sharpe'].mean():.3f}",
            f"{fold_df['OOS_AdjSharpe'].mean():.3f}",
            f"{fold_df['OOS_PValue_Sharpe'].mean():.3f}",
            f"{fold_df['OOS_AdjSharpe'].mean():.3f}",
            f"{(fold_df['IS_Sharpe'] - fold_df['IV_Sharpe']).mean():.3f}",
            f"{(fold_df['IV_Sharpe'] - fold_df['OOS_AdjSharpe']).mean():.3f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - ADJUSTED SHARPE ANALYSIS:",
            ["Model", "IS_Sharpe", "IV_Sharpe", "OOS_AdjSharpe", "OOS_p", "AdjSharpe_Q", "IS-IV", "IV-OOS"],
            adj_sharpe_rows,
            logger
        )
        logger.info("")
        
        # TABLE 3: Information Ratio Analysis
        ir_rows = []
        for result in fold_results:
            ir_rows.append([
                result['Model'],
                f"{result['IS_IR']:.3f}",
                f"{result['IV_IR']:.3f}",
                f"{result['OOS_IR']:.3f}",
                f"{result['OOS_PValue_IR']:.3f}",
                f"{result['Fold_IR_Q']:.3f}",
                f"{result['IS_IR'] - result['IV_IR']:.3f}",
                f"{result['IV_IR'] - result['OOS_IR']:.3f}"
            ])
        
        # Add mean row
        ir_rows.append([
            "MEAN",
            f"{fold_df['IS_IR'].mean():.3f}",
            f"{fold_df['IV_IR'].mean():.3f}",
            f"{fold_df['OOS_IR'].mean():.3f}",
            f"{fold_df['OOS_PValue_IR'].mean():.3f}",
            f"{fold_df['Fold_IR_Q'].mean():.3f}",
            f"{(fold_df['IS_IR'] - fold_df['IV_IR']).mean():.3f}",
            f"{(fold_df['IV_IR'] - fold_df['OOS_IR']).mean():.3f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - INFORMATION RATIO ANALYSIS:",
            ["Model", "IS_IR", "IV_IR", "OOS_IR", "OOS_p", "IR_Q", "IS-IV", "IV-OOS"],
            ir_rows,
            logger
        )
        logger.info("")
        
        # TABLE 4: Hit Rate Analysis
        hit_rows = []
        for result in fold_results:
            hit_rows.append([
                result['Model'],
                f"{result['IS_Hit']:.2%}",
                f"{result['IV_Hit']:.2%}",
                f"{result['OOS_Hit']:.2%}",
                f"{result['OOS_PValue_Hit']:.3f}",
                f"{result['Fold_Hit_Q']:.3f}"
            ])
        
        # Add mean row
        hit_rows.append([
            "MEAN",
            f"{fold_df['IS_Hit'].mean():.2%}",
            f"{fold_df['IV_Hit'].mean():.2%}",
            f"{fold_df['OOS_Hit'].mean():.2%}",
            f"{fold_df['OOS_PValue_Hit'].mean():.3f}",
            f"{fold_df['Fold_Hit_Q'].mean():.3f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - HIT RATE ANALYSIS:",
            ["Model", "IS_Hit", "IV_Hit", "OOS_Hit", "OOS_p", "Hit_Q"],
            hit_rows,
            logger
        )
        logger.info("")
        
        # TABLE 5: Returns & Risk Analysis
        returns_rows = []
        for result in fold_results:
            returns_rows.append([
                result['Model'],
                f"{result['OOS_Ann_Ret']:.4f}",
                f"{result['OOS_Ann_Vol']:.4f}",
                f"{result['OOS_AdjSharpe']:.3f}",
                f"{result['OOS_CB_Ratio']:.3f}",
                f"{result['OOS_DAPY_Binary']:.1f}",
                f"{result['OOS_DAPY_Both']:.1f}"
            ])
        
        # Add mean row
        returns_rows.append([
            "MEAN",
            f"{fold_df['OOS_Ann_Ret'].mean():.4f}",
            f"{fold_df['OOS_Ann_Vol'].mean():.4f}",
            f"{fold_df['OOS_AdjSharpe'].mean():.3f}",
            f"{fold_df['OOS_CB_Ratio'].mean():.3f}",
            f"{fold_df['OOS_DAPY_Binary'].mean():.1f}",
            f"{fold_df['OOS_DAPY_Both'].mean():.1f}"
        ])
        
        print_simple_table(
            f"FOLD {fold_idx+1} - RETURNS & RISK ANALYSIS:",
            ["Model", "Ann_Ret", "Ann_Vol", "AdjSharpe", "CB_Ratio", "DAPY_Bin", "DAPY_Both"],
            returns_rows,
            logger
        )
        
        # Enhanced best models summary with comprehensive metrics
        best_sharpe = fold_df.loc[fold_df['OOS_Sharpe'].idxmax()]
        best_adj_sharpe = fold_df.loc[fold_df['OOS_AdjSharpe'].idxmax()]
        best_ir = fold_df.loc[fold_df['OOS_IR'].idxmax()]
        best_hit = fold_df.loc[fold_df['OOS_Hit'].idxmax()]
        
        logger.info(f"BEST MODELS (FOLD {fold_idx+1}):")
        logger.info(f"Best OOS Sharpe:     {best_sharpe['Model']} (Sharpe={best_sharpe['OOS_Sharpe']:.3f}, p={best_sharpe['OOS_PValue_Sharpe']:.3f}, Hit={best_sharpe['OOS_Hit']:.2%}, IR={best_sharpe['OOS_IR']:.3f})")
        logger.info(f"Best OOS AdjSharpe:  {best_adj_sharpe['Model']} (AdjSharpe={best_adj_sharpe['OOS_AdjSharpe']:.3f}, p={best_adj_sharpe['OOS_PValue_Sharpe']:.3f}, Hit={best_adj_sharpe['OOS_Hit']:.2%})")
        logger.info(f"Best OOS IR:         {best_ir['Model']} (IR={best_ir['OOS_IR']:.3f}, p={best_ir['OOS_PValue_IR']:.3f}, Sharpe={best_ir['OOS_Sharpe']:.3f}, Hit={best_ir['OOS_Hit']:.2%})")
        logger.info(f"Best OOS Hit:        {best_hit['Model']} (Hit={best_hit['OOS_Hit']:.2%}, p={best_hit['OOS_PValue_Hit']:.3f}, Sharpe={best_hit['OOS_Sharpe']:.3f}, IR={best_hit['OOS_IR']:.3f})")
        
        # Double line spacing between folds
        logger.info("")
        logger.info("")
        logger.info("")
        
        all_fold_results[f'fold_{fold_idx+1}'] = fold_results
    
    return all_fold_results

def main():
    parser = argparse.ArgumentParser(description='Fixed XGBoost Performance Analysis')
    parser.add_argument('--max_features', type=int, default=50, help='Maximum features after selection')
    parser.add_argument('--n_folds', type=int, default=6, help='Number of cross-validation folds')
    parser.add_argument('--n_models', type=int, default=10, help='Number of XGBoost models')
    parser.add_argument('--xgb_type', type=str, default='standard', choices=['standard', 'deep', 'tiered'],
                       help='XGBoost configuration type')
    parser.add_argument('--target_symbol', type=str, default='@ES#C', help='Target symbol')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date YYYY-MM-DD')
    parser.add_argument('--inner_val_frac', type=float, default=0.2, help='Inner validation fraction')
    parser.add_argument('--log_label', type=str, default='fixed_tables', help='Label for log filename')
    parser.add_argument('--ewma_alpha', type=float, default=0.1, help='EWMA alpha parameter')
    parser.add_argument('--no_feature_selection', action='store_true', help='Skip feature selection')
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"xgb_performance_{args.log_label}_{args.xgb_type}_{args.max_features}feat_{args.n_folds}folds_{timestamp}.log"
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
    
    logger.info("="*80)
    logger.info("XGBoost Performance Analysis - Fixed Table Structure")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Target Symbol: {args.target_symbol}")
    logger.info(f"  Date Range: {args.start_date} to {args.end_date}")
    logger.info(f"  Models: {args.n_models}")
    logger.info(f"  Folds: {args.n_folds}")
    logger.info(f"  Max Features: {args.max_features}")
    logger.info(f"  XGBoost Type: {args.xgb_type}")
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
        logger.info(f"Using target column: {target_col}")
        
        if target_col not in df.columns:
            logger.error(f"Target column {target_col} not found!")
            return
        
        feature_cols = [c for c in df.columns if c != target_col]
        X, y = df[feature_cols], df[target_col]
        
        logger.info(f"Data shape before feature selection: X={X.shape}, y={y.shape}")
        
        # Feature selection
        if not args.no_feature_selection:
            logger.info("Applying block-wise feature selection...")
            X = apply_feature_selection(X, y, method='block_wise', max_total_features=args.max_features)
            logger.info(f"Selected {X.shape[1]} features (threshold: 0.7)")
        
        logger.info(f"Features after selection: {X.shape[1]}")
        logger.info(f"Data shape after feature processing: X={X.shape}, y={y.shape}")
        logger.info(f"Date range: Index {X.index[0]} to {X.index[-1]}")
        
        # Generate XGBoost specs
        if args.xgb_type == 'deep':
            logger.info(f"Generating {args.n_models} deep XGBoost models...")
            xgb_specs = generate_deep_xgb_specs(n_models=args.n_models)
        elif args.xgb_type == 'tiered':
            logger.info(f"Generating {args.n_models} tiered XGBoost models...")
            xgb_specs, _ = stratified_xgb_bank(X.columns, n_models=args.n_models)
        else:
            logger.info(f"Generating {args.n_models} standard XGBoost models...")
            xgb_specs = generate_xgb_specs(n_models=args.n_models)
        
        # Run enhanced fold analysis
        results = enhanced_fold_analysis(
            X, y, xgb_specs, args.n_folds, args.inner_val_frac, logger, args.ewma_alpha
        )
        
        logger.info("="*80)
        logger.info("Analysis completed successfully!")
        logger.info(f"Results saved to: {log_path}")
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise

if __name__ == "__main__":
    main()
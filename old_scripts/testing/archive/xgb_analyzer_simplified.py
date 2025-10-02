#!/usr/bin/env python3
"""
Simplified XGBoost Performance Analyzer
Maintains all functionality with cleaner code organization
"""

import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Import existing modules
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice
from cv.wfo import wfo_splits


class XGBoostAnalyzer:
    """
    Simplified XGBoost Performance Analyzer with comprehensive metrics
    """
    
    def __init__(self):
        self.logger = None
        
    def setup_logging(self, log_label: str, target_symbol: str, xgb_type: str, 
                     max_features: int, n_folds: int) -> logging.Logger:
        """Setup logging configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"xgb_analysis_{log_label}_{xgb_type}_{max_features}feat_{n_folds}folds_{timestamp}.log"
        log_path = os.path.join("logs", log_filename)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info("="*80)
        logger.info("XGBoost Performance Analysis - Simplified Framework")
        logger.info("="*80)
        logger.info(f"Configuration:")
        logger.info(f"  Target Symbol: {target_symbol}")
        logger.info(f"  Max Features: {max_features}")
        logger.info(f"  XGBoost Type: {xgb_type}")
        logger.info(f"  Folds: {n_folds}")
        logger.info(f"Results saved to: {log_path}")
        
        self.logger = logger
        return logger

    # Metric calculation functions
    @staticmethod
    def calculate_sharpe(returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252)
    
    @staticmethod 
    def calculate_information_ratio(returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return (returns.mean() * 252) / (returns.std() * np.sqrt(252))
    
    @staticmethod
    def calculate_adjusted_sharpe(returns: pd.Series, predictions: pd.Series, 
                                lambda_turnover: float = 0.1) -> float:
        """Calculate adjusted Sharpe with turnover penalty"""
        sharpe = XGBoostAnalyzer.calculate_sharpe(returns)
        if len(predictions) > 1:
            turnover = np.mean(np.abs(predictions.diff().fillna(0)))
            return sharpe - lambda_turnover * turnover
        return sharpe
    
    @staticmethod
    def calculate_hit_rate(predictions: pd.Series, actual_returns: pd.Series) -> float:
        """Calculate directional hit rate"""
        if len(predictions) == 0 or len(actual_returns) == 0:
            return 0.0
        pred_signs = np.sign(predictions)
        actual_signs = np.sign(actual_returns)
        return np.mean(pred_signs == actual_signs)
    
    @staticmethod
    def calculate_cb_ratio(returns: pd.Series) -> float:
        """Calculate Calmar-Bliss ratio (Annual Return / Max Drawdown)"""
        if len(returns) == 0:
            return 0.0
        ann_ret = returns.mean() * 252
        cumulative = returns.cumsum()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative - rolling_max
        max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0.001
        return ann_ret / max_drawdown if max_drawdown > 0 else 0.0
    
    @staticmethod
    def calculate_dapy_binary(predictions: pd.Series, actual_returns: pd.Series) -> float:
        """Calculate DAPY (Directional Accuracy Per Year) - binary version"""
        hit_rate = XGBoostAnalyzer.calculate_hit_rate(predictions, actual_returns)
        return (hit_rate - 0.5) * 252 * 100  # Convert to annual percentage
    
    @staticmethod
    def calculate_dapy_both(returns: pd.Series) -> float:
        """Calculate DAPY both (magnitude and direction)"""
        if len(returns) == 0:
            return 0.0
        return returns.sum() * 252  # Annualized total return

    @staticmethod
    def bootstrap_p_value(actual_metric: float, returns: pd.Series, 
                         metric_func, n_bootstraps: int = 1000) -> float:
        """Calculate bootstrap p-value for statistical significance"""
        if len(returns) == 0:
            return 1.0
            
        bootstrap_metrics = []
        for _ in range(n_bootstraps):
            shuffled = returns.sample(frac=1.0, replace=True)
            if metric_func == XGBoostAnalyzer.calculate_hit_rate:
                # Special case for hit rate - need predictions too
                shuffled_preds = shuffled.shift(1).fillna(0)
                bootstrap_metric = metric_func(shuffled_preds, shuffled)
            else:
                bootstrap_metric = metric_func(shuffled)
            bootstrap_metrics.append(bootstrap_metric)
        
        # Two-tailed test
        bootstrap_metrics = np.array(bootstrap_metrics)
        p_val = 2 * min(
            np.mean(bootstrap_metrics >= actual_metric),
            np.mean(bootstrap_metrics <= actual_metric)
        )
        return min(p_val, 1.0)

    @staticmethod
    def normalize_predictions(predictions: pd.Series) -> pd.Series:
        """Normalize predictions using z-score + tanh transformation"""
        if predictions.std() == 0:
            return pd.Series(np.zeros(len(predictions)), index=predictions.index)
        z_scores = (predictions - predictions.mean()) / predictions.std()
        normalized = np.tanh(z_scores)
        return pd.Series(normalized, index=predictions.index)

    def calculate_fold_metrics(self, predictions: pd.Series, returns: pd.Series, 
                             fold_num: int, is_train: bool = False) -> Dict[str, float]:
        """Calculate comprehensive metrics for a fold"""
        prefix = "IS" if is_train else "OOS"
        
        # Normalize predictions
        norm_pred = self.normalize_predictions(predictions)
        
        # Shift signal to avoid look-ahead bias
        signal_shifted = norm_pred.shift(1).fillna(0.0)
        pnl_returns = signal_shifted * returns
        
        # Calculate all metrics
        metrics = {
            f"{prefix}_Sharpe": self.calculate_sharpe(pnl_returns),
            f"{prefix}_IR": self.calculate_information_ratio(pnl_returns),
            f"{prefix}_AdjSharpe": self.calculate_adjusted_sharpe(pnl_returns, norm_pred),
            f"{prefix}_Hit": self.calculate_hit_rate(norm_pred, returns),
            f"{prefix}_CB_Ratio": self.calculate_cb_ratio(pnl_returns),
            f"{prefix}_Ann_Ret": pnl_returns.mean() * 252,
            f"{prefix}_Ann_Vol": pnl_returns.std() * np.sqrt(252),
            f"{prefix}_DAPY_Bin": self.calculate_dapy_binary(norm_pred, returns),
            f"{prefix}_DAPY_Both": self.calculate_dapy_both(pnl_returns)
        }
        
        # Calculate p-values for OOS metrics
        if not is_train:
            metrics.update({
                f"{prefix}_PValue_Sharpe": self.bootstrap_p_value(
                    metrics[f"{prefix}_Sharpe"], pnl_returns, self.calculate_sharpe),
                f"{prefix}_PValue_IR": self.bootstrap_p_value(
                    metrics[f"{prefix}_IR"], pnl_returns, self.calculate_information_ratio),
                f"{prefix}_PValue_Hit": self.bootstrap_p_value(
                    metrics[f"{prefix}_Hit"], returns, self.calculate_hit_rate)
            })
        
        return metrics

    def print_metrics_table(self, fold_results: List[Dict[str, Any]], fold_num: int):
        """Print formatted metrics tables for a fold"""
        self.logger.info(f"\nFOLD {fold_num} - COMPREHENSIVE ANALYSIS:")
        
        # Sharpe Analysis Table
        self.logger.info(f"\n{'='*20} SHARPE ANALYSIS {'='*20}")
        header = f"| {'Model':<5} | {'IS_Sharpe':<9} | {'IV_Sharpe':<9} | {'OOS_Sharpe':<10} | {'OOS_p':<5} | {'Sharpe_Q':<8} |"
        self.logger.info(header)
        self.logger.info("|" + "-"*(len(header)-2) + "|")
        
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            self.logger.info(
                f"| M{i:02d}   | {metrics['IS_Sharpe']:8.3f}  | {metrics['IV_Sharpe']:8.3f}  | "
                f"{metrics['OOS_Sharpe']:9.3f}  | {metrics.get('OOS_PValue_Sharpe', 0):4.3f} | "
                f"{metrics['OOS_Sharpe']:7.3f} |"
            )
        
        # Adjusted Sharpe Table  
        self.logger.info(f"\n{'='*20} ADJUSTED SHARPE ANALYSIS {'='*20}")
        header = f"| {'Model':<5} | {'IS_AdjSharpe':<11} | {'IV_AdjSharpe':<11} | {'OOS_AdjSharpe':<12} | {'OOS_p':<5} | {'AdjSharpe_Q':<10} |"
        self.logger.info(header)
        self.logger.info("|" + "-"*(len(header)-2) + "|")
        
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            self.logger.info(
                f"| M{i:02d}   | {metrics['IS_AdjSharpe']:10.3f}  | {metrics['IV_AdjSharpe']:10.3f}  | "
                f"{metrics['OOS_AdjSharpe']:11.3f}  | {metrics.get('OOS_PValue_Sharpe', 0):4.3f} | "
                f"{metrics['OOS_AdjSharpe']:9.3f} |"
            )

        # Information Ratio Table
        self.logger.info(f"\n{'='*20} INFORMATION RATIO ANALYSIS {'='*20}")
        header = f"| {'Model':<5} | {'IS_IR':<6} | {'IV_IR':<6} | {'OOS_IR':<6} | {'OOS_p':<5} | {'IR_Q':<5} |"
        self.logger.info(header)
        self.logger.info("|" + "-"*(len(header)-2) + "|")
        
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            self.logger.info(
                f"| M{i:02d}   | {metrics['IS_IR']:5.3f}  | {metrics['IV_IR']:5.3f}  | "
                f"{metrics['OOS_IR']:5.3f}  | {metrics.get('OOS_PValue_IR', 0):4.3f} | "
                f"{metrics['OOS_IR']:4.3f} |"
            )
        
        # Hit Rate Table
        self.logger.info(f"\n{'='*20} HIT RATE ANALYSIS {'='*20}")
        header = f"| {'Model':<5} | {'IS_Hit':<6} | {'IV_Hit':<6} | {'OOS_Hit':<7} | {'OOS_p':<5} | {'Hit_Q':<5} |"
        self.logger.info(header)
        self.logger.info("|" + "-"*(len(header)-2) + "|")
        
        for i, result in enumerate(fold_results):
            metrics = result['metrics']
            self.logger.info(
                f"| M{i:02d}   | {metrics['IS_Hit']*100:5.1f}% | {metrics['IV_Hit']*100:5.1f}% | "
                f"{metrics['OOS_Hit']*100:6.2f}% | {metrics.get('OOS_PValue_Hit', 0):4.3f} | "
                f"{metrics['OOS_Hit']:4.3f} |"
            )

    def analyze_performance(self, target_symbol: str, start_date: str, end_date: str,
                          max_features: int, n_models: int, n_folds: int,
                          xgb_type: str = "standard", log_label: str = "analysis"):
        """Main analysis function"""
        
        # Setup logging
        logger = self.setup_logging(log_label, target_symbol, xgb_type, max_features, n_folds)
        
        # Load and prepare data
        logger.info("Loading data...")
        data = prepare_real_data_simple(
            target_symbol=target_symbol,
            start_date=start_date,
            end_date=end_date
        )
        
        # Separate features and target
        target_col = f"{target_symbol}_target_return"
        y = data[target_col]
        X = data.drop(columns=[target_col])
        
        # Feature selection
        logger.info("Applying feature selection...")
        X_selected = apply_feature_selection(
            X, y, max_total_features=max_features, corr_threshold=0.7
        )
        
        logger.info(f"Data shape after feature selection: X={X_selected.shape}, y={y.shape}")
        logger.info(f"Date range: {X_selected.index[0]} to {X_selected.index[-1]}")
        
        # Generate XGBoost models
        logger.info(f"Generating {n_models} {xgb_type} XGBoost models...")
        if xgb_type == "deep":
            xgb_specs = generate_deep_xgb_specs(n_models)
        elif xgb_type == "tiered":
            xgb_specs, _ = stratified_xgb_bank(X_selected.columns.tolist(), n_models=n_models, seed=7)
        else:
            xgb_specs = generate_xgb_specs(n_models)
        
        # Cross-validation splits
        splits = wfo_splits(len(X_selected), k_folds=n_folds, min_train=100)
        
        # Analyze each fold
        for fold_idx, (train_idx, test_idx) in enumerate(splits, 1):
            if fold_idx > n_folds:
                break
                
            logger.info(f"\n{'='*50}")
            logger.info(f"FOLD {fold_idx}/{n_folds}")
            logger.info(f"{'='*50}")
            
            # Split data
            X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Inner validation split (20% of training)
            val_size = int(len(X_train) * 0.2)
            X_inner_train, X_inner_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
            y_inner_train, y_inner_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
            
            logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            logger.info(f"Inner Train: {len(X_inner_train)} samples, Inner Val: {len(X_inner_val)} samples")
            
            # Train all models and collect results
            fold_results = []
            for model_idx, spec in enumerate(xgb_specs):
                # Fit model on full training set
                model = fit_xgb_on_slice(X_train, y_train, spec)
                
                # Get predictions for all sets
                pred_inner_train = pd.Series(model.predict(X_inner_train), index=X_inner_train.index)
                pred_inner_val = pd.Series(model.predict(X_inner_val), index=X_inner_val.index) 
                pred_test = pd.Series(model.predict(X_test), index=X_test.index)
                
                # Calculate metrics
                is_metrics = self.calculate_fold_metrics(pred_inner_train, y_inner_train, fold_idx, is_train=True)
                iv_metrics = self.calculate_fold_metrics(pred_inner_val, y_inner_val, fold_idx, is_train=True)
                oos_metrics = self.calculate_fold_metrics(pred_test, y_test, fold_idx, is_train=False)
                
                # Combine all metrics
                combined_metrics = {**is_metrics}
                # Add IV metrics with prefix change
                for key, value in iv_metrics.items():
                    new_key = key.replace("IS_", "IV_")
                    combined_metrics[new_key] = value
                # Add OOS metrics  
                combined_metrics.update(oos_metrics)
                
                fold_results.append({
                    'model_idx': model_idx,
                    'spec': spec,
                    'metrics': combined_metrics
                })
            
            # Print results for this fold
            self.print_metrics_table(fold_results, fold_idx)
            
            # Find and report best models
            best_sharpe = max(fold_results, key=lambda x: x['metrics']['OOS_Sharpe'])
            best_adj_sharpe = max(fold_results, key=lambda x: x['metrics']['OOS_AdjSharpe'])
            best_hit = max(fold_results, key=lambda x: x['metrics']['OOS_Hit'])
            best_ir = max(fold_results, key=lambda x: x['metrics']['OOS_IR'])
            
            logger.info(f"\nBEST MODELS (FOLD {fold_idx}):")
            logger.info(f"Best OOS Sharpe:     M{best_sharpe['model_idx']:02d} "
                       f"(Sharpe={best_sharpe['metrics']['OOS_Sharpe']:.3f}, "
                       f"p={best_sharpe['metrics'].get('OOS_PValue_Sharpe', 0):.3f}, "
                       f"Hit={best_sharpe['metrics']['OOS_Hit']:.2%})")
            logger.info(f"Best OOS AdjSharpe:  M{best_adj_sharpe['model_idx']:02d} "
                       f"(AdjSharpe={best_adj_sharpe['metrics']['OOS_AdjSharpe']:.3f}, "
                       f"p={best_adj_sharpe['metrics'].get('OOS_PValue_Sharpe', 0):.3f}, "
                       f"Hit={best_adj_sharpe['metrics']['OOS_Hit']:.2%})")
            logger.info(f"Best OOS IR:         M{best_ir['model_idx']:02d} "
                       f"(IR={best_ir['metrics']['OOS_IR']:.3f}, "
                       f"p={best_ir['metrics'].get('OOS_PValue_IR', 0):.3f}, "
                       f"Sharpe={best_ir['metrics']['OOS_Sharpe']:.3f})")
            logger.info(f"Best OOS Hit:        M{best_hit['model_idx']:02d} "
                       f"(Hit={best_hit['metrics']['OOS_Hit']:.2%}, "
                       f"p={best_hit['metrics'].get('OOS_PValue_Hit', 0):.3f}, "
                       f"Sharpe={best_hit['metrics']['OOS_Sharpe']:.3f})")
        
        logger.info(f"\n{'='*80}")
        logger.info("Analysis completed successfully!")
        logger.info(f"{'='*80}")


def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description="Simplified XGBoost Performance Analyzer")
    parser.add_argument("--target_symbol", default="@ES#C", help="Target trading symbol")
    parser.add_argument("--start_date", default="2023-01-01", help="Analysis start date")
    parser.add_argument("--end_date", default="2024-01-01", help="Analysis end date")
    parser.add_argument("--max_features", type=int, default=50, help="Maximum features after selection")
    parser.add_argument("--n_models", type=int, default=10, help="Number of XGBoost models")
    parser.add_argument("--n_folds", type=int, default=3, help="Number of cross-validation folds")
    parser.add_argument("--xgb_type", choices=["standard", "deep", "tiered"], default="standard", 
                       help="XGBoost architecture type")
    parser.add_argument("--log_label", default="simplified", help="Label for log file")
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = XGBoostAnalyzer()
    analyzer.analyze_performance(
        target_symbol=args.target_symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        max_features=args.max_features,
        n_models=args.n_models,
        n_folds=args.n_folds,
        xgb_type=args.xgb_type,
        log_label=args.log_label
    )


if __name__ == "__main__":
    main()
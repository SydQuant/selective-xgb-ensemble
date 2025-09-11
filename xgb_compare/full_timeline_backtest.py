#!/usr/bin/env python3
"""
Production backtesting with rolling Q-score model selection.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

from metrics_utils import calculate_model_metrics_from_pnl, QualityTracker

logger = logging.getLogger(__name__)

class FullTimelineBacktester:
    """Backtest using rolling Q-score model selection across training and production periods."""
    
    def __init__(self, top_n_models: int = 5, q_metric: str = 'sharpe', cutoff_fraction: float = 0.7):
        self.top_n_models = top_n_models
        self.q_metric = q_metric
        self.cutoff_fraction = cutoff_fraction
        self.full_timeline_results = []
        self.model_selection_history = []
        
    def run_full_timeline_backtest(self, X: pd.DataFrame, y: pd.Series, 
                                  all_fold_results: List[Dict], 
                                  fold_splits: List[Tuple], 
                                  xgb_specs: List[Dict],
                                  quality_tracker, config=None) -> Dict[str, Any]:
        """
        Run production backtest with rolling Q-score model selection.
        Training and production periods use identical OOS methodology.
        """
        n_folds = len(fold_splits)
        # Calculate training/production cutoff
        cutoff_fold = int(n_folds * self.cutoff_fraction)
        
        logger.info(f"Running full timeline backtest with {n_folds} folds (0-based indexing)")
        logger.info(f"Fold 0 skipped (no prior Q-score history for meaningful selection)")
        logger.info(f"Effective backtest period: Folds 1-{n_folds-1} ({n_folds-1} folds)")
        logger.info(f"Training period: Folds 1-{cutoff_fold}")
        logger.info(f"Production period: Folds {cutoff_fold+1}-{n_folds-1}")
        logger.info(f"Using consistent OOS methodology for all folds")
        
        # Initialize result tracking - keep as lists for now for compatibility
        training_predictions = []
        training_pnl = []
        training_returns = []  # Store actual returns for hit_rate calculation
        training_dates = []
        production_predictions = []
        production_pnl = []
        production_returns = []  # Store actual returns for hit_rate calculation
        production_dates = []
        
        # Copy quality tracker for backtesting
        rolling_quality_tracker = QualityTracker(len(xgb_specs), 63)
        if hasattr(quality_tracker, 'quality_history'):
            rolling_quality_tracker.quality_history = {
                metric: [hist.copy() for hist in quality_tracker.quality_history[metric]] 
                for metric in quality_tracker.quality_history
            }
        
        # Process each fold with same methodology
        # Simplified backtesting: Skip fold 0, start from fold 1 with proper Q-score selection
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            
            # Skip fold 0 - no prior Q-scores for meaningful model selection
            if fold_idx == 0:
                logger.info(f"Fold {fold_idx}: Skipped (no prior Q-score history for meaningful selection)")
                continue
                
            logger.info(f"\n--- Processing fold {fold_idx}/{n_folds-1} ---")
            
            # Model selection: Use previous fold's Q-scores consistently (no fallback)
            selection_fold_idx = fold_idx - 1
            
            # Get Q-scores from previous fold
            if self.q_metric in ['combined', 'sharpe_hit']:
                if hasattr(config, 'q_sharpe_weight') and hasattr(config, 'q_use_zscore'):
                    q_scores = rolling_quality_tracker.get_sharpe_hit_combined_q_scores(
                        selection_fold_idx, ewma_alpha=0.1, 
                        sharpe_weight=config.q_sharpe_weight, use_zscore=config.q_use_zscore)
                else:
                    q_scores = rolling_quality_tracker.get_sharpe_hit_combined_q_scores(
                        selection_fold_idx, ewma_alpha=0.1, sharpe_weight=0.5, use_zscore=True)
            else:
                q_scores = rolling_quality_tracker.get_q_scores(selection_fold_idx, ewma_alpha=0.1)[self.q_metric]
            
            logger.info(f"Fold {fold_idx}: Q-scores from fold {selection_fold_idx}: {[f'M{i:02d}:{score:.3f}' for i, score in enumerate(q_scores[:10]) if not np.isnan(score)][:5]}...")
            
            # Select top N models by Q-score
            model_q_pairs = [(idx, score) for idx, score in enumerate(q_scores) if not np.isnan(score)]
            model_q_pairs.sort(key=lambda x: x[1], reverse=True)
            selected_models = [idx for idx, _ in model_q_pairs[:self.top_n_models]]
            avg_q_score = np.mean([score for _, score in model_q_pairs[:self.top_n_models]]) if model_q_pairs else 0.0
            
            logger.info(f"Fold {fold_idx}: Selected models {selected_models} based on fold {selection_fold_idx} Q-scores, avg Q-score: {avg_q_score:.3f}")
            # This block is now handled above in the unified model selection logic
            
            # Store model selection decision
            selection_info = {
                'fold': fold_idx,
                'selection_based_on_fold': selection_fold_idx if fold_idx > 0 else -1,  # -1 for fold 0 (no prior fold)
                'selected_models': selected_models.copy(),
                'avg_q_score': avg_q_score,
                'period_type': 'training' if fold_idx <= cutoff_fold else 'production'
            }
            self.model_selection_history.append(selection_info)
            
            # Run OOS backtest for this fold using selected models
            if selected_models:
                fold_predictions = {}
                
                # Get stored OOS predictions from training phase (no retraining!)
                fold_key = f'fold_{fold_idx + 1}'  # all_fold_results uses 1-based keys like 'fold_1', 'fold_2'
                logger.debug(f"Fold {fold_idx}: Looking for key '{fold_key}' in {list(all_fold_results.keys())}")
                
                if fold_key in all_fold_results and 'oos_predictions' in all_fold_results[fold_key]:
                    stored_predictions = all_fold_results[fold_key]['oos_predictions']
                    stored_test_idx = all_fold_results[fold_key]['test_idx']
                    logger.debug(f"Fold {fold_idx}: Found stored predictions for {len(stored_predictions)} models")
                    
                    # Verify test indices match (sanity check)
                    if not np.array_equal(stored_test_idx, test_idx):
                        logger.warning(f"Fold {fold_idx}: Test indices mismatch! Stored: {len(stored_test_idx)}, Current: {len(test_idx)}")
                        # Continue anyway - this might happen due to small differences in fold splitting
                    
                    # Retrieve predictions for selected models only
                    for model_idx in selected_models:
                        if model_idx in stored_predictions:
                            fold_predictions[model_idx] = stored_predictions[model_idx]
                            logger.debug(f"Fold {fold_idx}: Model {model_idx} predictions length: {len(stored_predictions[model_idx])}")
                        else:
                            logger.warning(f"Fold {fold_idx}: No stored predictions for model {model_idx}")
                else:
                    logger.error(f"Fold {fold_idx}: No stored predictions found in fold_key={fold_key}")
                    logger.error(f"Available keys: {list(all_fold_results.keys())}")
                    continue
                
                # Combine predictions from selected models (equal weighting)
                if fold_predictions:
                    predictions_list = list(fold_predictions.values())
                    # Ensure all predictions have same length
                    min_length = min(len(pred) for pred in predictions_list)
                    if min_length != max(len(pred) for pred in predictions_list):
                        logger.warning(f"Fold {fold_idx}: Prediction length mismatch, truncating to {min_length}")
                        predictions_list = [pred.iloc[:min_length] for pred in predictions_list]
                    
                    combined_prediction = sum(predictions_list) / len(predictions_list)
                    
                    # CORRECTED LOGIC: Direct calculation (signal and returns already properly aligned)
                    # signal[Monday] predicts Monday→Tuesday return, y[Monday] is actual Monday→Tuesday return
                    signal = combined_prediction
                    actual_returns = y.iloc[test_idx]
                    
                    # Ensure alignment and same length 
                    min_len = min(len(signal), len(actual_returns))
                    if len(signal) != len(actual_returns):
                        logger.warning(f"Fold {fold_idx}: Signal/returns length mismatch ({len(signal)} vs {len(actual_returns)}), truncating to {min_len}")
                        signal = signal.iloc[:min_len]
                        actual_returns = actual_returns.iloc[:min_len]
                    
                    # Direct PnL calculation - no artificial lag needed in backtest
                    fold_pnl = signal * actual_returns
                    
                    # Store results
                    signal_values = signal.values
                    pnl_values = fold_pnl.values
                    returns_values = actual_returns.values
                    
                    # Add to appropriate period first (training vs production)
                    if fold_idx <= cutoff_fold:
                        logger.debug(f"Fold {fold_idx}: Adding {len(signal_values)} to training (current len: {len(training_predictions)})")
                        training_predictions.extend(signal_values)
                        training_pnl.extend(pnl_values)
                        training_returns.extend(returns_values)
                        training_dates.extend(X.index[test_idx].tolist())
                        period_type = 'training'
                    else:
                        logger.debug(f"Fold {fold_idx}: Adding {len(signal_values)} to production (current len: {len(production_predictions)})")
                        production_predictions.extend(signal_values)
                        production_pnl.extend(pnl_values)
                        production_returns.extend(returns_values)
                        production_dates.extend(X.index[test_idx].tolist())
                        period_type = 'production'
                    
                    # Build full timeline from period data (avoid double counting)
                    # Note: full_timeline lists will be built by combining training + production at the end
                    
                    # Calculate fold-level metrics using pre-calculated PnL (no redundant calculation)
                    fold_metrics = calculate_model_metrics_from_pnl(fold_pnl, signal, actual_returns)
                    
                    # No Q-score updates during backtest - use only pre-computed training data
                    # The rolling_quality_tracker was initialized with ALL training fold data
                    
                    # Store fold results
                    fold_result = {
                        'fold': fold_idx,
                        'period_type': period_type,
                        'selected_models': selected_models.copy(),
                        'fold_metrics': fold_metrics,
                        'n_test_samples': len(test_idx),
                        'avg_q_score': selection_info['avg_q_score']
                    }
                    self.full_timeline_results.append(fold_result)
                    
                    period_label = "Training" if fold_idx <= cutoff_fold else "Production"
                    logger.info(f"Fold {fold_idx} ({period_label}): Sharpe={fold_metrics.get('sharpe', 0):.3f}, Hit={fold_metrics.get('hit_rate', 0)*100:.1f}%")
            
            else:
                logger.warning(f"No models selected for fold {fold_idx}")
        
        # Build full timeline data from training + production (avoid double counting)
        full_timeline_predictions = training_predictions + production_predictions
        full_timeline_pnl = training_pnl + production_pnl
        full_timeline_returns = training_returns + production_returns
        full_timeline_dates = training_dates + production_dates
        
        logger.debug(f"Final lengths: training_pred={len(training_predictions)}, training_pnl={len(training_pnl)}")
        logger.debug(f"Final lengths: production_pred={len(production_predictions)}, production_pnl={len(production_pnl)}")
        logger.debug(f"Final lengths: full_pred={len(full_timeline_predictions)}, full_pnl={len(full_timeline_returns)}")
        
        # Calculate overall metrics for each period
        training_metrics = {}
        production_metrics = {}
        full_timeline_metrics = {}
        
        # Calculate aggregate metrics with length validation
        if training_pnl:
            training_pnl_series = pd.Series(training_pnl)
            training_pred_series = pd.Series(training_predictions)
            
            # Debug length mismatch
            if len(training_pred_series) != len(training_pnl_series):
                logger.warning(f"Training data length mismatch: predictions={len(training_pred_series)}, returns={len(training_pnl_series)}")
                min_len = min(len(training_pred_series), len(training_pnl_series))
                training_pred_series = training_pred_series.iloc[:min_len]
                training_pnl_series = training_pnl_series.iloc[:min_len]
            
            
            # Use simplified metrics calculation from PnL with correct returns
            training_returns_series = pd.Series(training_returns)
            training_metrics = calculate_model_metrics_from_pnl(training_pnl_series, training_pred_series, training_returns_series)
            training_metrics['total_periods'] = len(training_pnl)
            logger.info(f"Training Period Overall: Sharpe={training_metrics.get('sharpe', 0):.3f}, Hit={training_metrics.get('hit_rate', 0)*100:.1f}%")
        
        if production_pnl:
            production_pnl_series = pd.Series(production_pnl)
            production_pred_series = pd.Series(production_predictions)
            
            # Debug length mismatch
            if len(production_pred_series) != len(production_pnl_series):
                logger.warning(f"Production data length mismatch: predictions={len(production_pred_series)}, returns={len(production_pnl_series)}")
                min_len = min(len(production_pred_series), len(production_pnl_series))
                production_pred_series = production_pred_series.iloc[:min_len]
                production_pnl_series = production_pnl_series.iloc[:min_len]
            
            
            # Use simplified metrics calculation from PnL with correct returns
            production_returns_series = pd.Series(production_returns)
            production_metrics = calculate_model_metrics_from_pnl(production_pnl_series, production_pred_series, production_returns_series)
            production_metrics['total_periods'] = len(production_pnl)
            logger.info(f"Production Period Overall: Sharpe={production_metrics.get('sharpe', 0):.3f}, Hit={production_metrics.get('hit_rate', 0)*100:.1f}%")
        
        if full_timeline_pnl:
            full_pnl_series = pd.Series(full_timeline_pnl)
            full_pred_series = pd.Series(full_timeline_predictions)
            
            # Debug length mismatch
            if len(full_pred_series) != len(full_pnl_series):
                logger.warning(f"Full timeline data length mismatch: predictions={len(full_pred_series)}, returns={len(full_pnl_series)}")
                min_len = min(len(full_pred_series), len(full_pnl_series))
                full_pred_series = full_pred_series.iloc[:min_len]
                full_pnl_series = full_pnl_series.iloc[:min_len]
            
            # Use simplified metrics calculation from PnL with correct returns
            full_returns_series = pd.Series(full_timeline_returns)
            full_timeline_metrics = calculate_model_metrics_from_pnl(full_pnl_series, full_pred_series, full_returns_series)
            full_timeline_metrics['total_periods'] = len(full_timeline_pnl)
            logger.info(f"Full Timeline Overall: Sharpe={full_timeline_metrics.get('sharpe', 0):.3f}, Hit={full_timeline_metrics.get('hit_rate', 0)*100:.1f}%")
        
        return {
            'full_timeline_metrics': full_timeline_metrics,
            'training_metrics': training_metrics,
            'production_metrics': production_metrics,
            'model_selection_history': self.model_selection_history,
            'fold_results': self.full_timeline_results,
            'full_timeline_returns': full_timeline_pnl,  # PnL for backward compatibility
            'training_returns': training_pnl,  # PnL for visualization
            'production_returns': production_pnl,  # PnL for visualization
            'full_timeline_predictions': full_timeline_predictions,
            'full_timeline_dates': full_timeline_dates if full_timeline_dates else [],
            'cutoff_fold': cutoff_fold + 1,
            'training_folds': list(range(1, cutoff_fold + 1)),
            'production_folds': list(range(cutoff_fold + 1, n_folds + 1))
        }
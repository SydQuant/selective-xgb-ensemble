#!/usr/bin/env python3
"""
Full timeline backtesting with consistent OOS methodology.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple

from metrics_utils import calculate_model_metrics, normalize_predictions, QualityTracker

logger = logging.getLogger(__name__)

class FullTimelineBacktester:
    """Complete timeline backtester using consistent OOS methodology for both training and production periods."""
    
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
        Run complete timeline backtest using same methodology for training and production periods.
        
        Training Period: Folds 1 through cutoff_fold
        Production Period: Folds (cutoff_fold+1) through n_folds
        
        Both periods use identical model selection + OOS backtesting process.
        """
        n_folds = len(fold_splits)
        # Backtest starts from fold 2 (fold 1 has no meaningful Q-scores)
        # Effective folds: 2 through n_folds
        effective_folds = n_folds - 1
        cutoff_fold = int(effective_folds * self.cutoff_fraction) + 1  # +1 to account for skipped fold 1
        
        logger.info(f"Running full timeline backtest with {n_folds} folds (starting from fold 2)")
        logger.info(f"Fold 1 skipped (no Q-score history for meaningful selection)")
        logger.info(f"Effective backtest period: Folds 2-{n_folds} ({effective_folds} folds)")
        logger.info(f"Training period: Folds 2-{cutoff_fold}")
        logger.info(f"Production period: Folds {cutoff_fold+1}-{n_folds}")
        logger.info(f"Using same OOS methodology for both periods")
        
        # Initialize tracking
        full_timeline_predictions = []
        full_timeline_returns = []
        full_timeline_dates = []
        training_predictions = []
        training_returns = []
        production_predictions = []
        production_returns = []
        
        # Create a copy of quality tracker with ALL training fold data
        rolling_quality_tracker = QualityTracker(len(xgb_specs), 63)  # Use default halflife
        # Initialize with ALL training results (all folds from training phase)
        if hasattr(quality_tracker, 'quality_history'):
            # Copy ALL available training data, not just fold 0
            rolling_quality_tracker.quality_history = {
                metric: [hist.copy() for hist in quality_tracker.quality_history[metric]] 
                for metric in quality_tracker.quality_history
            }
        
        # Process each fold with same methodology
        # NOTE: Backtesting starts from fold 2 since fold 1 has no meaningful Q-scores
        for fold_idx, (train_idx, test_idx) in enumerate(fold_splits):
            fold_number = fold_idx + 1
            
            # Skip fold 0 - no meaningful Q-score history for model selection
            if fold_idx == 0:
                logger.info(f"Fold {fold_number}: Skipped (no prior Q-score history for meaningful selection)")
                continue
                
            logger.info(f"Processing fold {fold_number}/{n_folds}")
            
            # Model selection: Use Q-scores up to previous fold (fold x uses data up to x-1)
            selection_fold = fold_idx - 1  # fold_idx-1 gives us data up to fold (fold_idx-1)
            
            # Get latest Q-scores from ROLLING quality tracker (updated with actual backtest results)
            q_scores = rolling_quality_tracker.get_q_scores(selection_fold, ewma_alpha=0.1)[self.q_metric]
            logger.info(f"Fold {fold_number}: Q-scores from fold {selection_fold}: {[f'M{i:02d}:{score:.3f}' for i, score in enumerate(q_scores[:10]) if not np.isnan(score)][:5]}...")
            
            # Get top N models by Q-score
            model_q_pairs = [(idx, score) for idx, score in enumerate(q_scores) if not np.isnan(score)]
            if not model_q_pairs:
                # Fallback if no valid Q-scores
                selected_models = list(range(min(self.top_n_models, len(xgb_specs))))
                logger.info(f"Fold {fold_number}: Using fallback models {selected_models} (no valid Q-scores from fold {selection_fold})")
                avg_q_score = 0.0
            else:
                model_q_pairs.sort(key=lambda x: x[1], reverse=True)
                selected_models = [idx for idx, _ in model_q_pairs[:self.top_n_models]]
                avg_q_score = np.mean([score for _, score in model_q_pairs[:self.top_n_models]])
                
                if fold_idx == 1:
                    logger.info(f"Fold {fold_number}: First meaningful selection based on fold {selection_fold+1} Q-scores, models {selected_models}, avg Q-score: {avg_q_score:.3f}")
                else:
                    logger.info(f"Fold {fold_number}: Selected based on fold {selection_fold+1} Q-scores, models {selected_models}, avg Q-score: {avg_q_score:.3f}")
            # This block is now handled above in the unified model selection logic
            
            # Store model selection decision
            selection_info = {
                'fold': fold_number,
                'selection_based_on_fold': selection_fold,
                'selected_models': selected_models.copy(),
                'q_scores': [q_scores[i] if i < len(q_scores) and not np.isnan(q_scores[i]) else 0.0 for i in selected_models],
                'avg_q_score': np.mean([q_scores[i] for i in selected_models if i < len(q_scores) and not np.isnan(q_scores[i])]) if len(selected_models) > 0 else 0.0,
                'period_type': 'training' if fold_number <= cutoff_fold else 'production'
            }
            self.model_selection_history.append(selection_info)
            
            # Run OOS backtest for this fold using selected models
            if selected_models:
                fold_predictions = {}
                
                # Get stored OOS predictions from training phase (no retraining!)
                fold_key = f'fold_{fold_number}'
                if fold_key in all_fold_results and 'oos_predictions' in all_fold_results[fold_key]:
                    stored_predictions = all_fold_results[fold_key]['oos_predictions']
                    stored_test_idx = all_fold_results[fold_key]['test_idx']
                    
                    # Verify test indices match (sanity check)
                    if not np.array_equal(stored_test_idx, test_idx):
                        logger.warning(f"Fold {fold_number}: Test indices mismatch! Stored: {len(stored_test_idx)}, Current: {len(test_idx)}")
                        # Continue anyway - this might happen due to small differences in fold splitting
                    
                    # Retrieve predictions for selected models only
                    for model_idx in selected_models:
                        if model_idx in stored_predictions:
                            fold_predictions[model_idx] = stored_predictions[model_idx]
                        else:
                            logger.warning(f"Fold {fold_number}: No stored predictions for model {model_idx}")
                else:
                    logger.error(f"Fold {fold_number}: No stored predictions found in fold_key={fold_key}")
                    logger.error(f"Available keys: {list(all_fold_results.keys())}")
                    continue
                
                # Combine predictions from selected models (equal weighting)
                if fold_predictions:
                    predictions_list = list(fold_predictions.values())
                    combined_prediction = sum(predictions_list) / len(predictions_list)
                    
                    # Apply signal lag to avoid look-ahead bias
                    lagged_signal = combined_prediction.shift(1).fillna(0.0)
                    fold_returns = lagged_signal * y.iloc[test_idx]
                    
                    # Store results for full timeline
                    full_timeline_predictions.extend(lagged_signal.values)
                    full_timeline_returns.extend(fold_returns.values)
                    if hasattr(X.index[test_idx], 'tolist'):
                        full_timeline_dates.extend(X.index[test_idx].tolist())
                    
                    # Separate into training vs production periods
                    if fold_number <= cutoff_fold:
                        training_predictions.extend(lagged_signal.values)
                        training_returns.extend(fold_returns.values)
                        period_type = 'training'
                    else:
                        production_predictions.extend(lagged_signal.values)
                        production_returns.extend(fold_returns.values)
                        period_type = 'production'
                    
                    # Calculate fold-level metrics
                    fold_metrics = calculate_model_metrics(lagged_signal, y.iloc[test_idx], shifted=False)
                    
                    # No Q-score updates during backtest - use only pre-computed training data
                    # The rolling_quality_tracker was initialized with ALL training fold data
                    
                    # Store fold results
                    fold_result = {
                        'fold': fold_number,
                        'period_type': period_type,
                        'selected_models': selected_models.copy(),
                        'fold_metrics': fold_metrics,
                        'n_test_samples': len(test_idx),
                        'avg_q_score': selection_info['avg_q_score']
                    }
                    self.full_timeline_results.append(fold_result)
                    
                    period_label = "Training" if fold_number <= cutoff_fold else "Production"
                    logger.info(f"Fold {fold_number} ({period_label}): Sharpe={fold_metrics.get('sharpe', 0):.3f}, Hit={fold_metrics.get('hit_rate', 0)*100:.1f}%")
            
            else:
                logger.warning(f"No models selected for fold {fold_number}")
        
        # Calculate overall metrics for each period
        training_metrics = {}
        production_metrics = {}
        full_timeline_metrics = {}
        
        if training_returns:
            training_series = pd.Series(training_returns)
            training_pred_series = pd.Series(training_predictions)
            training_metrics = calculate_model_metrics(training_pred_series, training_series, shifted=False)
            training_metrics['total_periods'] = len(training_returns)
            logger.info(f"Training Period Overall: Sharpe={training_metrics.get('sharpe', 0):.3f}, Hit={training_metrics.get('hit_rate', 0)*100:.1f}%")
        
        if production_returns:
            production_series = pd.Series(production_returns)
            production_pred_series = pd.Series(production_predictions)
            production_metrics = calculate_model_metrics(production_pred_series, production_series, shifted=False)
            production_metrics['total_periods'] = len(production_returns)
            logger.info(f"Production Period Overall: Sharpe={production_metrics.get('sharpe', 0):.3f}, Hit={production_metrics.get('hit_rate', 0)*100:.1f}%")
        
        if full_timeline_returns:
            full_series = pd.Series(full_timeline_returns)
            full_pred_series = pd.Series(full_timeline_predictions)
            full_timeline_metrics = calculate_model_metrics(full_pred_series, full_series, shifted=False)
            full_timeline_metrics['total_periods'] = len(full_timeline_returns)
            logger.info(f"Full Timeline Overall: Sharpe={full_timeline_metrics.get('sharpe', 0):.3f}, Hit={full_timeline_metrics.get('hit_rate', 0)*100:.1f}%")
        
        return {
            'full_timeline_metrics': full_timeline_metrics,
            'training_metrics': training_metrics,
            'production_metrics': production_metrics,
            'model_selection_history': self.model_selection_history,
            'fold_results': self.full_timeline_results,
            'full_timeline_returns': full_timeline_returns,
            'training_returns': training_returns,
            'production_returns': production_returns,
            'full_timeline_predictions': full_timeline_predictions,
            'full_timeline_dates': full_timeline_dates if full_timeline_dates else [],
            'cutoff_fold': cutoff_fold + 1,
            'training_folds': list(range(1, cutoff_fold + 1)),
            'production_folds': list(range(cutoff_fold + 1, n_folds + 1))
        }
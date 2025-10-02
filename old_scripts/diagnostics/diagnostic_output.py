import json
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_comprehensive_diagnostics(config, data_shape, feature_info, model_performance, ensemble_info=None, xgb_diagnostics=None, grope_diagnostics=None, output_dir="artifacts/diagnostics"):
    """Save simplified diagnostic information for analysis."""
    
    # Create diagnostics directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_symbol = config.get('target_symbol', 'unknown').replace('#', '').replace('@', '')
    
    # Diagnostic data structure
    diagnostics = {
        "timestamp": timestamp,
        "config": dict(config),
        "data_info": {
            "shape": data_shape,
            "feature_count": data_shape[1] if len(data_shape) > 1 else 0,
            "observation_count": data_shape[0] if len(data_shape) > 0 else 0
        },
        "feature_info": feature_info,
        "model_performance": model_performance,
        "ensemble_info": ensemble_info,
        "xgb_diagnostics": xgb_diagnostics,
        "grope_diagnostics": grope_diagnostics
    }
    
    # Save JSON diagnostic
    json_file = f"{output_dir}/diagnostic_{target_symbol}_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    logger.info(f"💾 Saved comprehensive diagnostics to {json_file}")
    
    # Save human-readable summary
    summary_file = f"{output_dir}/summary_{target_symbol}_{timestamp}.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"XGBoost Ensemble Diagnostic Report - {timestamp}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"TARGET SYMBOL: {config.get('target_symbol')}\n")
        f.write(f"DATE RANGE: {config.get('start_date')} to {config.get('end_date')}\n")
        f.write(f"PREDICTION HORIZON: {config.get('n_hours')}h\n")
        f.write(f"SIGNAL HOUR: {config.get('signal_hour')}\n\n")
        
        f.write("DATA CONFIGURATION:\n")
        f.write(f"  Dataset Shape: {data_shape}\n")
        f.write(f"  Features: {data_shape[1] if len(data_shape) > 1 else 0}\n")
        f.write(f"  Observations: {data_shape[0] if len(data_shape) > 0 else 0}\n")
        f.write(f"  On-Target Only: {config.get('on_target_only')}\n")
        f.write(f"  Max Features: {config.get('max_features')}\n")
        f.write(f"  Correlation Threshold: {config.get('corr_threshold')}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  Models: {config.get('n_models')}\n")
        f.write(f"  Folds: {config.get('folds')}\n")
        f.write(f"  Multiprocessing: {config.get('use_multiprocessing')}\n")
        f.write(f"  P-value Threshold: {config.get('pmax')}\n\n")
        
        f.write("FEATURE SUMMARY:\n")
        f.write(f"  Total Features: {len(feature_info)}\n")
        if len(feature_info) > 0:
            f.write(f"  Feature Names: {[f.get('name', f'feat_{i}') for i, f in enumerate(feature_info[:5])]}{'...' if len(feature_info) > 5 else ''}\n")
        
        # Add feature selection information if available
        if ensemble_info and ensemble_info.get('feature_selection'):
            fs_info = ensemble_info['feature_selection']
            f.write(f"\nFEATURE SELECTION DETAILS:\n")
            f.write(f"  Original Features: {fs_info.get('original_features', 'N/A')}\n")
            f.write(f"  Features After Selection: {fs_info.get('features_after_selection', 'N/A')}\n")
            f.write(f"  Features Removed: {fs_info.get('reduction_count', 'N/A')}\n")
            f.write(f"  Reduction Percentage: {fs_info.get('reduction_percentage', 'N/A')}%\n")
        
        f.write(f"\nFINAL PERFORMANCE:\n")
        perf = model_performance
        f.write(f"  DAPY: {perf.get('dapy', perf.get('total_return', 'N/A'))}\n")
        f.write(f"  Information Ratio: {perf.get('information_ratio', perf.get('sharpe_ratio', 'N/A'))}\n")
        f.write(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 'N/A')}\n")
        f.write(f"  Max Drawdown: {perf.get('max_drawdown', 'N/A')}\n")
        f.write(f"  Hit Rate: {perf.get('hit_rate', 'N/A')}\n")
        f.write(f"  Total Return: {perf.get('total_return', 'N/A')}\n")
        
        # Model diagnostics
        f.write(f"\nMODEL DIAGNOSTICS:\n")
        f.write(f"  Selected Models: {perf.get('n_select', config.get('n_select', 'N/A'))}\n")
        f.write(f"  P-value Gating: {'Bypassed' if config.get('bypass_pvalue_gating', False) else 'Enabled'}\n")
        f.write(f"  Ensemble Method: {'GROPE' if not config.get('equal_weights', False) else 'Equal Weights'}\n")
        f.write(f"  Auto-Normalization: {'Yes' if config.get('driver_selection_objective') or config.get('grope_weight_objective') else 'No'}\n")
        
        # Add ensemble and driver selection information
        if ensemble_info:
            f.write(f"\nENSEMBLE DRIVER SELECTION:\n")
            f.write(f"  Total Folds: {ensemble_info.get('total_folds', 'N/A')}\n")
            f.write(f"  DAPY Value: {ensemble_info.get('dapy_value', 'N/A')}\n")
            f.write(f"  Information Ratio: {ensemble_info.get('information_ratio', 'N/A')}\n")
            f.write(f"  Hit Rate: {ensemble_info.get('hit_rate', 'N/A')}\n")
            
            # Show selected drivers for each fold
            fold_summaries = ensemble_info.get('fold_summaries', [])
            if fold_summaries:
                f.write(f"\n  SELECTED DRIVERS BY FOLD:\n")
                for fold_info in fold_summaries:
                    fold_num = fold_info.get('fold', 'N/A')
                    chosen_idx = fold_info.get('chosen_idx', [])
                    weights = fold_info.get('weights', [])
                    tau = fold_info.get('tau', 'N/A')
                    f.write(f"    Fold {fold_num}: Drivers {chosen_idx} (tau={tau:.3f})\n")
                    if weights:
                        weights_str = ', '.join([f"{w:.3f}" for w in weights[:5]])  # First 5 weights
                        if len(weights) > 5:
                            weights_str += f" ... (+{len(weights)-5} more)"
                        f.write(f"      Weights: [{weights_str}]\n")
                
                # Summary of unique drivers across all folds
                all_drivers = set()
                for fold_info in fold_summaries:
                    all_drivers.update(fold_info.get('chosen_idx', []))
                f.write(f"\n  DRIVER SELECTION SUMMARY:\n")
                f.write(f"    Unique Drivers Used: {sorted(all_drivers)}\n")
                f.write(f"    Total Unique Drivers: {len(all_drivers)}\n")
        
        # XGBoost Diagnostics
        if xgb_diagnostics:
            f.write(f"\nXGBOOST MODEL INSIGHTS:\n")
            f.write(f"  Total Models Generated: {xgb_diagnostics.get('total_models', 'N/A')}\n")
            f.write(f"  Models Per Fold: {xgb_diagnostics.get('models_per_fold', 'N/A')}\n")
            f.write(f"  GPU Detection: {xgb_diagnostics.get('gpu_available', 'Unknown')}\n")
            f.write(f"  Model Architecture: {xgb_diagnostics.get('architecture', 'Standard')}\n")
            
            # Hyperparameter insights
            param_stats = xgb_diagnostics.get('hyperparameter_stats', {})
            if param_stats:
                f.write(f"  Hyperparameter Diversity:\n")
                f.write(f"    Learning Rate Range: {param_stats.get('learning_rate_range', 'N/A')}\n")
                f.write(f"    Max Depth Range: {param_stats.get('max_depth_range', 'N/A')}\n")
                f.write(f"    N_Estimators Range: {param_stats.get('n_estimators_range', 'N/A')}\n")
            
            # Model performance diversity
            perf_stats = xgb_diagnostics.get('performance_stats', {})
            if perf_stats:
                f.write(f"  Individual Model Performance:\n")
                f.write(f"    Sharpe Range: {perf_stats.get('sharpe_range', 'N/A')}\n")
                f.write(f"    Best Single Model Sharpe: {perf_stats.get('best_sharpe', 'N/A')}\n")
                f.write(f"    Models with Positive Sharpe: {perf_stats.get('positive_sharpe_count', 'N/A')}\n")
        
        # GROPE Optimization Diagnostics  
        if grope_diagnostics:
            f.write(f"\nGROPE OPTIMIZATION INSIGHTS:\n")
            f.write(f"  Optimization Method: {grope_diagnostics.get('method', 'RBF')}\n")
            f.write(f"  Total Iterations: {grope_diagnostics.get('total_iterations', 'N/A')}\n")
            f.write(f"  Convergence Status: {grope_diagnostics.get('converged', 'Unknown')}\n")
            
            # Weight distribution insights
            weight_stats = grope_diagnostics.get('weight_statistics', {})
            if weight_stats:
                f.write(f"  Weight Distribution:\n")
                f.write(f"    Active Models: {weight_stats.get('active_models', 'N/A')} / {weight_stats.get('total_models', 'N/A')}\n")
                f.write(f"    Weight Concentration: {weight_stats.get('concentration_ratio', 'N/A')}\n")
                f.write(f"    Max Weight: {weight_stats.get('max_weight', 'N/A')}\n")
                f.write(f"    Min Non-Zero Weight: {weight_stats.get('min_nonzero_weight', 'N/A')}\n")
            
            # Temperature parameter insights
            temp_info = grope_diagnostics.get('temperature_info', {})
            if temp_info:
                f.write(f"  Temperature Parameter:\n")
                f.write(f"    Final Temperature: {temp_info.get('final_tau', 'N/A')}\n")
                f.write(f"    Temperature Effect: {temp_info.get('effect', 'N/A')}\n")
            
            # Objective function evolution
            objective_info = grope_diagnostics.get('objective_evolution', {})
            if objective_info:
                f.write(f"  Objective Function Evolution:\n")
                f.write(f"    Initial Score: {objective_info.get('initial_score', 'N/A')}\n")
                f.write(f"    Final Score: {objective_info.get('final_score', 'N/A')}\n")
                f.write(f"    Improvement: {objective_info.get('improvement', 'N/A')}\n")
        
        # Driver Selection Detailed Analysis
        if ensemble_info and ensemble_info.get('fold_summaries'):
            f.write(f"\nDRIVER SELECTION ANALYSIS:\n")
            fold_summaries = ensemble_info.get('fold_summaries', [])
            
            # Analysis across folds
            driver_usage = {}
            weight_distributions = []
            for fold_info in fold_summaries:
                chosen_idx = fold_info.get('chosen_idx', [])
                weights = fold_info.get('weights', [])
                
                for i, driver_idx in enumerate(chosen_idx):
                    if driver_idx not in driver_usage:
                        driver_usage[driver_idx] = []
                    if i < len(weights):
                        driver_usage[driver_idx].append(weights[i])
                
                if weights:
                    weight_distributions.append(weights)
            
            # Driver consistency analysis
            if driver_usage:
                f.write(f"  Driver Consistency Across Folds:\n")
                for driver_idx, weights in driver_usage.items():
                    avg_weight = sum(w for w in weights if w > 0) / len([w for w in weights if w > 0]) if any(w > 0 for w in weights) else 0
                    usage_rate = len([w for w in weights if w > 0]) / len(fold_summaries) if fold_summaries else 0
                    f.write(f"    Driver {driver_idx}: Used in {usage_rate:.1%} folds, Avg Weight: {avg_weight:.3f}\n")
        
    logger.info(f"📄 Saved diagnostic summary to {summary_file}")
    return json_file, summary_file
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def save_comprehensive_diagnostics(config, data_shape, feature_info, model_performance, output_dir="artifacts/diagnostics"):
    """Save comprehensive diagnostic information for analysis."""
    
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
        "model_performance": model_performance
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
        for i, feat_info in enumerate(feature_info[:10]):  # Show first 10 features
            f.write(f"  Feature {i}: {feat_info.get('name', 'unknown')}\n")
            f.write(f"    Range: [{feat_info.get('min', 'N/A'):.6f}, {feat_info.get('max', 'N/A'):.6f}]\n")
            f.write(f"    Std: {feat_info.get('std', 'N/A'):.6f}\n")
        
        if len(feature_info) > 10:
            f.write(f"  ... and {len(feature_info) - 10} more features\n")
        
        f.write(f"\nMODEL PERFORMANCE:\n")
        perf = model_performance
        f.write(f"  Constant Predictions: {perf.get('constant_models', 'N/A')}\n")
        f.write(f"  Variable Predictions: {perf.get('variable_models', 'N/A')}\n")
        f.write(f"  P-value Range: {perf.get('pvalue_range', 'N/A')}\n")
        f.write(f"  Final Return: {perf.get('total_return', 'N/A'):.4f}\n")
        f.write(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 'N/A'):.4f}\n")
        f.write(f"  Hit Rate: {perf.get('hit_rate', 'N/A'):.3f}\n")
        
    logger.info(f"📄 Saved diagnostic summary to {summary_file}")
    return json_file, summary_file
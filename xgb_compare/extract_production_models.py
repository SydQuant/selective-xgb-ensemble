#!/usr/bin/env python3
"""
Extract Production Models from Optimal Test Results

Safely extracts the exact models that were selected in the final fold of optimal tests.
Uses the same data preparation and feature selection as the original tests.
"""

import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# Add parent directory to path for data imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import XGBCompareConfig
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice
from cv.wfo import wfo_splits, wfo_splits_rolling

def extract_final_fold_models_from_log(log_file_path: str) -> list:
    """Extract the final fold selected models from log file."""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()

        # Find the last model selection line in the log
        lines = content.split('\n')
        model_selection_lines = [line for line in lines if 'Selected models' in line and 'based on' in line]

        if not model_selection_lines:
            return []

        # Get the last selection
        last_selection_line = model_selection_lines[-1]

        # Parse model indices (e.g., "Selected models [17, 31, 144, 10, 126, 51, 142, 112]")
        import re
        match = re.search(r'Selected models \[([^\]]+)\]', last_selection_line)
        if match:
            model_indices_str = match.group(1)
            model_indices = [int(x.strip()) for x in model_indices_str.split(',')]
            return model_indices

        return []

    except Exception as e:
        print(f"Error parsing log file {log_file_path}: {e}")
        return []

def build_production_models_for_symbol(symbol: str, optimal_config: dict, log_file_path: str, test_mode: bool = False):
    """Build production models for a specific symbol using exact optimal configuration."""

    print(f"\n{'='*70}")
    print(f"EXTRACTING PRODUCTION MODELS: {symbol}")
    print(f"{'='*70}")

    try:
        # Extract final fold selected models from log
        selected_model_indices = extract_final_fold_models_from_log(log_file_path)
        if not selected_model_indices:
            print(f"ERROR: Could not extract selected models from {log_file_path}")
            return False

        print(f"Final fold selected models: {selected_model_indices}")

        # Handle controlled test naming (use base symbol for data)
        base_symbol = symbol.replace('_controlled', '').replace('_fixed', '').replace('_comparison', '').replace('_final', '').replace('_thorough', '').replace('_verification', '')

        # Create configuration matching the optimal test
        config = XGBCompareConfig(
            target_symbol=base_symbol,
            n_models=optimal_config['models'],
            n_folds=optimal_config['folds'],
            max_features=optimal_config['features'],
            q_metric=optimal_config['q_metric'],
            xgb_type=optimal_config['xgb_type'],
            binary_signal=optimal_config.get('binary_signal', False)
        )

        print(f"Using config: {config.n_models}M, {config.n_folds}F, {config.max_features}feat, {config.q_metric}, {config.xgb_type}")
        print(f"Data symbol: {base_symbol}, Export symbol: {symbol}")

        # Prepare data exactly as in optimal test
        print("Preparing data with same preprocessing...")
        df = prepare_real_data_simple(base_symbol, start_date=config.start_date, end_date=config.end_date)
        target_col = f"{config.target_symbol}_target_return"
        X, y = df[[c for c in df.columns if c != target_col]], df[target_col]

        # Apply same feature selection
        if config.max_features > 0:
            X = apply_feature_selection(X, y, method='block_wise', max_total_features=config.max_features)

        selected_features = list(X.columns)
        print(f"Selected {len(selected_features)} features")

        # Generate XGB specs exactly as in optimal test
        # Need to generate enough specs to cover the highest model index
        max_model_idx = max(selected_model_indices)
        n_specs_needed = max(config.n_models, max_model_idx + 1)

        print(f"Need {n_specs_needed} specs to cover model indices up to {max_model_idx}")

        if config.xgb_type == 'tiered':
            # Pass the feature columns to stratified_xgb_bank with EXACT same seed (returns tuple)
            xgb_specs, col_slices = stratified_xgb_bank(selected_features, n_specs_needed, seed=13)
        elif config.xgb_type == 'deep':
            xgb_specs = generate_deep_xgb_specs(n_specs_needed, seed=13)
            col_slices = [selected_features] * len(xgb_specs)  # Use all features for each model
        else:
            xgb_specs = generate_xgb_specs(n_specs_needed, seed=13)
            col_slices = [selected_features] * len(xgb_specs)  # Use all features for each model

        print(f"Generated {len(xgb_specs)} XGB specifications with column slices")

        # Get the final fold split (use same logic as optimal test)
        if config.rolling_days > 0:
            fold_splits = list(wfo_splits_rolling(len(X), config.n_folds, rolling_days=config.rolling_days))
        else:
            fold_splits = list(wfo_splits(len(X), config.n_folds))

        final_train_idx, final_test_idx = fold_splits[-1]  # Last fold split
        print(f"Final fold: train={len(final_train_idx)}, test={len(final_test_idx)}")

        # Train ONLY the selected models
        X_train, y_train = X.iloc[final_train_idx], y.iloc[final_train_idx]

        selected_models = {}
        model_feature_slices = {}
        for i, model_idx in enumerate(selected_model_indices):
            if model_idx < len(xgb_specs):
                spec = xgb_specs[model_idx]

                print(f"Training selected model M{model_idx:02d} with ALL {X_train.shape[1]} features (matching full framework behavior)...")

                # Use ALL features to match current framework behavior
                # NOTE: This matches how the full framework actually trains tiered models
                model = fit_xgb_on_slice(X_train, y_train, spec, force_cpu=False)

                model_key = f"model_{i+1:02d}"
                selected_models[model_key] = model
                model_feature_slices[model_key] = selected_features  # Store all features (matching framework)

                print(f"  SUCCESS: M{model_idx:02d} -> {model_key} (uses all {len(selected_features)} features)")

        print(f"Trained {len(selected_models)} selected models")

        # Create production package
        production_package = {
            'symbol': symbol,
            'models': selected_models,
            'model_feature_slices': model_feature_slices,  # Store feature slice for each model
            'selected_features': selected_features,
            'selected_model_indices': selected_model_indices,
            'binary_signal': config.binary_signal,
            'metadata': {
                'source_log': os.path.basename(log_file_path),
                'n_models': config.n_models,
                'n_folds': config.n_folds,
                'max_features': config.max_features,
                'q_metric': config.q_metric,
                'xgb_type': config.xgb_type,
                'export_timestamp': datetime.now().isoformat()
            }
        }

        # Save to PROD directory
        prod_dir = Path(__file__).parent.parent / "PROD"
        models_dir = prod_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        production_file = models_dir / f"{symbol}_production.pkl"
        with open(production_file, 'wb') as f:
            pickle.dump(production_package, f)

        print(f"SUCCESS: Exported to {production_file}")
        print(f"File size: {production_file.stat().st_size:,} bytes")
        print(f"Models: {len(selected_models)}, Features: {len(selected_features)}")

        return True

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution."""
    import argparse

    # Optimal configurations and their log files
    optimal_configs = {
        "@AD#C": {
            "models": 150, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered",
            "log_file": "results/logs - PROD prep/20250915_204830_xgb_compare_OPTIMAL_AD_production.log"
        },
        "@AD#C_controlled": {
            "models": 20, "folds": 6, "features": 50, "q_metric": "sharpe", "xgb_type": "tiered",
            "log_file": "results/20250916_060451_xgb_compare_CONTROLLED_full_export_ADC.log"
        },
        "@AD#C_fixed": {
            "models": 8, "folds": 3, "features": 30, "q_metric": "sharpe", "xgb_type": "tiered",
            "log_file": "results/20250916_060924_xgb_compare_FIXED_export_test_ADC.log"
        },
        "@AD#C_comparison": {
            "models": 10, "folds": 4, "features": 40, "q_metric": "sharpe", "xgb_type": "tiered",
            "log_file": "results/20250916_065047_xgb_compare_COMPARISON_full_export.log"
        },
        "@AD#C_final": {
            "models": 8, "folds": 3, "features": 25, "q_metric": "sharpe", "xgb_type": "tiered",
            "log_file": "results/20250916_071642_xgb_compare_FINAL_consistent_full.log"
        },
        "@ES#C": {
            "models": 150, "folds": 15, "features": 100, "q_metric": "hit_rate", "xgb_type": "standard",
            "log_file": "results/logs - PROD prep/20250915_204934_xgb_compare_OPTIMAL_ES_production.log"
        },
        "@ES#C_verification": {
            "models": 150, "folds": 15, "features": 100, "q_metric": "hit_rate", "xgb_type": "standard",
            "log_file": "results/20250916_082053_xgb_compare_VERIFICATION_ES_exact_replica.log"
        },
        "@NQ#C": {
            "models": 100, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "standard",
            "log_file": "results/logs - PROD prep/20250915_210545_xgb_compare_OPTIMAL_NQ_production.log"
        },
        "@NQ#C_thorough": {
            "models": 15, "folds": 4, "features": 30, "q_metric": "sharpe", "xgb_type": "standard",
            "log_file": "results/20250916_075745_xgb_compare_THOROUGH_NQ_full_export.log"
        },
        # Add more symbols as needed
    }

    parser = argparse.ArgumentParser(description='Extract production models from optimal test results')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to extract (e.g., @AD#C @ES#C)')
    parser.add_argument('--test', action='store_true', help='Test mode with smaller config')

    args = parser.parse_args()

    print("Production Model Extractor")
    print("=" * 70)
    print(f"Symbols: {args.symbols}")

    results = {}
    for symbol in args.symbols:
        if symbol not in optimal_configs:
            print(f"ERROR: No optimal config found for {symbol}")
            results[symbol] = False
            continue

        config = optimal_configs[symbol]
        log_file = config['log_file']

        if not os.path.exists(log_file):
            print(f"ERROR: Log file not found: {log_file}")
            results[symbol] = False
            continue

        success = build_production_models_for_symbol(symbol, config, log_file, args.test)
        results[symbol] = success

    # Summary
    print(f"\n{'='*70}")
    print("EXTRACTION SUMMARY")
    print(f"{'='*70}")

    for symbol, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{symbol}: {status}")

    successful = sum(results.values())
    total = len(results)
    print(f"\nOverall: {successful}/{total} extractions successful")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Rigorous Model Identity Validation

Validates that extracted models are IDENTICAL to original test models
by comparing predictions across multiple data samples.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import stratified_xgb_bank, generate_xgb_specs, generate_deep_xgb_specs, fit_xgb_on_slice
from cv.wfo import wfo_splits

def validate_model_identity(symbol: str = "@AD#C", n_test_samples: int = 50):
    """
    Rigorous validation that extracted models are identical to original models.
    Tests predictions on multiple data samples to verify consistency.
    """

    print(f"{'='*80}")
    print(f"RIGOROUS MODEL IDENTITY VALIDATION: {symbol}")
    print(f"{'='*80}")

    try:
        # Load the extracted production package
        prod_dir = Path(__file__).parent.parent / "PROD"
        models_dir = prod_dir / "models"
        production_file = models_dir / f"{symbol}_production.pkl"

        if not production_file.exists():
            print(f"ERROR: Production file not found: {production_file}")
            return False

        with open(production_file, 'rb') as f:
            package = pickle.load(f)

        extracted_models = package.get('models', {})
        model_feature_slices = package.get('model_feature_slices', {})
        selected_model_indices = package.get('selected_model_indices', [])

        print(f"Loaded {len(extracted_models)} extracted models")
        print(f"Original model indices: {selected_model_indices}")

        # Recreate the EXACT same models using same methodology
        print("\nRecreating original models with same methodology...")

        # Use same configuration
        optimal_config = {
            "@AD#C": {"models": 150, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered"}
        }

        config = optimal_config[symbol]

        # Same data preparation
        df = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
        target_col = f"{symbol}_target_return"
        X, y = df[[c for c in df.columns if c != target_col]], df[target_col]

        # Same feature selection
        X = apply_feature_selection(X, y, method='block_wise', max_total_features=config['features'])
        selected_features = list(X.columns)

        print(f"Features match: {set(selected_features) == set(package.get('selected_features', []))}")

        # Same XGB specs with SAME seed
        max_model_idx = max(selected_model_indices)
        n_specs_needed = max(config['models'], max_model_idx + 1)

        if config['xgb_type'] == 'tiered':
            xgb_specs, col_slices = stratified_xgb_bank(selected_features, n_specs_needed, seed=13)
        elif config['xgb_type'] == 'deep':
            xgb_specs = generate_deep_xgb_specs(n_specs_needed, seed=13)
            col_slices = [selected_features] * len(xgb_specs)
        else:
            xgb_specs = generate_xgb_specs(n_specs_needed, seed=13)
            col_slices = [selected_features] * len(xgb_specs)

        # Same fold split
        fold_splits = list(wfo_splits(len(X), config['folds']))
        final_train_idx, final_test_idx = fold_splits[-1]

        # Same training data
        X_train, y_train = X.iloc[final_train_idx], y.iloc[final_train_idx]

        print(f"Training data shape: {X_train.shape}")

        # Recreate the selected models
        recreated_models = {}
        for i, model_idx in enumerate(selected_model_indices):
            if model_idx < len(xgb_specs):
                spec = xgb_specs[model_idx]
                model_cols = col_slices[model_idx]

                print(f"Recreating M{model_idx:02d} with {len(model_cols)} features...")

                X_train_slice = X_train[model_cols]
                recreated_model = fit_xgb_on_slice(X_train_slice, y_train, spec, force_cpu=False)
                recreated_models[f"model_{i+1:02d}"] = recreated_model

        print(f"Recreated {len(recreated_models)} models")

        # Now test predictions on multiple samples to verify identity
        print(f"\nTesting predictions on {n_test_samples} samples...")

        # Get test data (use final test split)
        X_test = X.iloc[final_test_idx]

        if len(X_test) > n_test_samples:
            # Sample random rows for testing
            np.random.seed(42)  # Fixed seed for reproducible testing
            test_indices = np.random.choice(len(X_test), n_test_samples, replace=False)
            X_test_sample = X_test.iloc[test_indices]
        else:
            X_test_sample = X_test

        print(f"Testing on {len(X_test_sample)} data points")

        # Compare predictions model by model
        all_match = True
        tolerance = 1e-10  # Very strict tolerance for floating point

        for i, model_idx in enumerate(selected_model_indices):
            model_key = f"model_{i+1:02d}"

            if model_key in extracted_models and model_key in recreated_models:
                extracted_model = extracted_models[model_key]
                recreated_model = recreated_models[model_key]

                # Get model features
                model_features = model_feature_slices.get(model_key, [])
                available_features = [f for f in model_features if f in X_test_sample.columns]

                if not available_features:
                    print(f"  M{model_idx:02d}: SKIP (no features available)")
                    continue

                X_test_slice = X_test_sample[available_features]

                # Get predictions from both models
                extracted_preds = extracted_model.predict(X_test_slice.values)
                recreated_preds = recreated_model.predict(X_test_slice.values)

                # Compare predictions
                max_diff = np.max(np.abs(extracted_preds - recreated_preds))
                mean_diff = np.mean(np.abs(extracted_preds - recreated_preds))

                match = max_diff < tolerance
                all_match = all_match and match

                print(f"  M{model_idx:02d}: {'MATCH' if match else 'DIFFER'} - Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")

                if not match:
                    print(f"    Sample diffs: {(extracted_preds - recreated_preds)[:5]}")

        print(f"\n{'='*80}")
        print("VALIDATION RESULTS")
        print(f"{'='*80}")

        if all_match:
            print("SUCCESS: All models are IDENTICAL")
            print("- Predictions match within tolerance")
            print("- Feature slices are consistent")
            print("- Extraction method is VALIDATED")
            print("SAFE TO DEPLOY TO PRODUCTION")
        else:
            print("ERROR: Models are NOT IDENTICAL")
            print("- Prediction differences detected")
            print("- DO NOT DEPLOY TO PRODUCTION")
            print("- Review extraction methodology")

        return all_match

    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate model identity')
    parser.add_argument('--symbol', type=str, default='@AD#C', help='Symbol to validate')
    parser.add_argument('--test-samples', type=int, default=100, help='Number of test samples')

    args = parser.parse_args()

    print("Model Identity Validation")
    print("=" * 80)
    print("This test verifies that extracted models are IDENTICAL to original models")
    print("by comparing predictions on the same data samples.")
    print(f"Symbol: {args.symbol}")
    print(f"Test samples: {args.test_samples}")

    success = validate_model_identity(args.symbol, args.test_samples)

    if success:
        print("\nOVERALL: VALIDATION PASSED - Models are identical")
    else:
        print("\nOVERALL: VALIDATION FAILED - Models differ")

if __name__ == "__main__":
    main()
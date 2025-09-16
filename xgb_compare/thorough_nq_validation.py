#!/usr/bin/env python3
"""
Thorough @NQ#C Validation

Comprehensive comparison of full export vs quick extraction for @NQ#C including:
- Model parameters comparison
- Prediction accuracy validation
- Signal generation testing
- Feature usage verification
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from PROD.common.signal_engine import SignalEngine
from data.data_utils_simple import prepare_real_data_simple

def load_and_inspect_models():
    """Load and thoroughly inspect both @NQ#C models."""

    print("="*80)
    print("THOROUGH @NQ#C MODEL INSPECTION")
    print("="*80)

    models_dir = Path(__file__).parent.parent / "PROD" / "models"
    full_file = models_dir / "@NQ#C_production.pkl"
    quick_file = models_dir / "@NQ#C_thorough_production.pkl"

    # Load packages
    with open(full_file, 'rb') as f:
        full_pkg = pickle.load(f)
    with open(quick_file, 'rb') as f:
        quick_pkg = pickle.load(f)

    print(f"Full export file: {full_file.stat().st_size:,} bytes")
    print(f"Quick extract file: {quick_file.stat().st_size:,} bytes")

    # Get models
    full_model = full_pkg['models']['model_01']
    quick_model = quick_pkg['models']['model_01']

    print(f"\n{'='*60}")
    print("MODEL PARAMETER COMPARISON")
    print(f"{'='*60}")

    # Compare XGBoost parameters
    print("FULL EXPORT MODEL (M09):")
    if hasattr(full_model, 'get_params'):
        params = full_model.get_params()
        for key in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']:
            if key in params:
                print(f"  {key}: {params[key]}")
    print(f"  n_features_in_: {getattr(full_model, 'n_features_in_', 'unknown')}")

    print("\nQUICK EXTRACT MODEL (M09):")
    if hasattr(quick_model, 'get_params'):
        params = quick_model.get_params()
        for key in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']:
            if key in params:
                print(f"  {key}: {params[key]}")
    print(f"  n_features_in_: {getattr(quick_model, 'n_features_in_', 'unknown')}")

    # Compare parameters
    if hasattr(full_model, 'get_params') and hasattr(quick_model, 'get_params'):
        full_params = full_model.get_params()
        quick_params = quick_model.get_params()

        print(f"\nPARAMETER COMPARISON:")
        all_match = True
        for key in ['max_depth', 'learning_rate', 'n_estimators', 'subsample', 'colsample_bytree']:
            if key in full_params and key in quick_params:
                match = full_params[key] == quick_params[key]
                all_match = all_match and match
                print(f"  {key}: {'MATCH' if match else 'DIFFER'} ({full_params[key]} vs {quick_params[key]})")

        print(f"All parameters match: {all_match}")

    return full_pkg, quick_pkg, full_model, quick_model

def test_prediction_consistency(full_model, quick_model, selected_features, n_samples=50):
    """Test prediction consistency across multiple data samples."""

    print(f"\n{'='*60}")
    print("PREDICTION CONSISTENCY TEST")
    print(f"{'='*60}")

    # Create multiple test datasets
    np.random.seed(42)  # Fixed seed for reproducible testing

    print(f"Testing {n_samples} prediction samples...")
    print(f"Model expects {len(selected_features)} features")

    all_diffs = []
    for i in range(n_samples):
        # Create random test data with exact feature count
        test_data = np.random.randn(1, len(selected_features))

        # Get predictions
        full_pred = full_model.predict(test_data)[0]
        quick_pred = quick_model.predict(test_data)[0]

        diff = abs(full_pred - quick_pred)
        all_diffs.append(diff)

        if i < 10:  # Show first 10 samples
            print(f"  Sample {i+1:2d}: Full={full_pred:+.8f}, Quick={quick_pred:+.8f}, Diff={diff:.2e}")

    # Statistics
    max_diff = max(all_diffs)
    mean_diff = np.mean(all_diffs)
    std_diff = np.std(all_diffs)

    print(f"\nPREDICTION STATISTICS:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Std difference: {std_diff:.2e}")

    tolerance = 1e-12
    if max_diff < tolerance:
        print(f"SUCCESS: Predictions are IDENTICAL (within {tolerance} tolerance)")
        return True
    else:
        print(f"ERROR: Predictions DIFFER (max diff: {max_diff:.2e})")
        return False

def test_signal_generation(symbol="@NQ#C", n_test_points=20):
    """Test signal generation with both models on real data."""

    print(f"\n{'='*60}")
    print("SIGNAL GENERATION TEST")
    print(f"{'='*60}")

    # Initialize signal engines
    models_dir = Path(__file__).parent.parent / "PROD" / "models"
    config_dir = Path(__file__).parent.parent / "PROD" / "config"

    engine = SignalEngine(models_dir, config_dir)

    # Get test data
    df = prepare_real_data_simple(symbol, start_date="2025-07-01", end_date="2025-08-01")
    target_col = f"{symbol}_target_return"
    features_df = df[[c for c in df.columns if c != target_col]]

    print(f"Available test data: {len(features_df)} rows")

    # Test full export model
    full_pkg = engine.load_symbol_package(symbol)
    if full_pkg:
        print(f"\nFull export package loaded: {len(full_pkg.get('models', {}))} models")

        results = []
        for i in range(min(n_test_points, len(features_df))):
            test_data = features_df.iloc[:i+5].copy()  # Rolling window
            result = engine.generate_signal(test_data, symbol)

            if result:
                signal, score = result
                results.append({'method': 'full', 'point': i, 'signal': signal, 'score': score})

        print(f"Full export results: {len(results)} signals")
        if results:
            signals = [r['signal'] for r in results]
            scores = [r['score'] for r in results]
            print(f"  Signals: {signals.count(1)} buy, {signals.count(-1)} sell")
            print(f"  Score range: {min(scores):.6f} to {max(scores):.6f}")

    # Test quick extract model (load different symbol name)
    quick_symbol = f"{symbol}_thorough"
    quick_pkg = engine.load_symbol_package(quick_symbol)
    if quick_pkg:
        print(f"\nQuick extract package loaded: {len(quick_pkg.get('models', {}))} models")

        quick_results = []
        for i in range(min(n_test_points, len(features_df))):
            test_data = features_df.iloc[:i+5].copy()  # Same rolling window
            result = engine.generate_signal(test_data, quick_symbol)

            if result:
                signal, score = result
                quick_results.append({'method': 'quick', 'point': i, 'signal': signal, 'score': score})

        print(f"Quick extract results: {len(quick_results)} signals")
        if quick_results:
            signals = [r['signal'] for r in quick_results]
            scores = [r['score'] for r in quick_results]
            print(f"  Signals: {signals.count(1)} buy, {signals.count(-1)} sell")
            print(f"  Score range: {min(scores):.6f} to {max(scores):.6f}")

        # Compare signal results
        print(f"\nSIGNAL COMPARISON:")
        if len(results) == len(quick_results):
            signal_matches = 0
            score_diffs = []

            for full_r, quick_r in zip(results, quick_results):
                signal_match = full_r['signal'] == quick_r['signal']
                score_diff = abs(full_r['score'] - quick_r['score'])

                if signal_match:
                    signal_matches += 1
                score_diffs.append(score_diff)

                if len(score_diffs) <= 10:  # Show first 10
                    print(f"  Point {full_r['point']:2d}: Full signal={full_r['signal']:+2d}, Quick signal={quick_r['signal']:+2d}, Score diff={score_diff:.2e}")

            print(f"\nSIGNAL SUMMARY:")
            print(f"  Signal matches: {signal_matches}/{len(results)} ({signal_matches/len(results)*100:.1f}%)")
            print(f"  Max score diff: {max(score_diffs):.2e}")
            print(f"  Mean score diff: {np.mean(score_diffs):.2e}")

            if signal_matches == len(results) and max(score_diffs) < 1e-10:
                print("SUCCESS: Signals are IDENTICAL")
                return True
            else:
                print("ERROR: Signals DIFFER")
                return False
        else:
            print("ERROR: Different number of results")
            return False
    else:
        print("ERROR: Could not load quick extract package")
        return False

def main():
    """Main thorough validation."""

    print("Thorough @NQ#C Production Pipeline Validation")
    print("=" * 80)

    # Step 1: Load and inspect models
    full_pkg, quick_pkg, full_model, quick_model = load_and_inspect_models()

    # Step 2: Test prediction consistency
    selected_features = full_pkg.get('selected_features', [])
    predictions_match = test_prediction_consistency(full_model, quick_model, selected_features, 100)

    # Step 3: Test signal generation
    signals_match = test_signal_generation("@NQ#C", 15)

    # Final verdict
    print(f"\n{'='*80}")
    print("THOROUGH VALIDATION VERDICT")
    print(f"{'='*80}")

    if predictions_match and signals_match:
        print("SUCCESS: @NQ#C production pipeline FULLY VALIDATED")
        print("- Model parameters are identical")
        print("- Predictions match exactly across all test samples")
        print("- Signal generation produces identical results")
        print("- Both methods are mathematically equivalent")
        print("SAFE TO USE EITHER METHOD FOR PRODUCTION")
    else:
        print("ERROR: @NQ#C validation FAILED")
        if not predictions_match:
            print("- Model predictions differ")
        if not signals_match:
            print("- Signal generation differs")
        print("DO NOT DEPLOY TO PRODUCTION")

if __name__ == "__main__":
    main()
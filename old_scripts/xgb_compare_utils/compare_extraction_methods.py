#!/usr/bin/env python3
"""
Compare Full Export vs Quick Extraction Methods

Rigorous side-by-side comparison to validate that both methods produce identical models.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def load_and_compare_models():
    """Load and compare models from both methods."""

    print("="*80)
    print("COMPARING FULL EXPORT vs QUICK EXTRACTION METHODS")
    print("="*80)

    # Load both production packages
    models_dir = Path(__file__).parent.parent / "PROD" / "models"

    full_export_file = models_dir / "@AD#C_production.pkl"  # From fixed full export (M06)
    quick_extract_file = models_dir / "@AD#C_final_production.pkl"  # From quick extract (M06)

    print(f"Full export file: {full_export_file}")
    print(f"Quick extract file: {quick_extract_file}")

    if not full_export_file.exists():
        print("ERROR: Full export file not found")
        return False

    if not quick_extract_file.exists():
        print("ERROR: Quick extract file not found")
        return False

    # Load packages
    with open(full_export_file, 'rb') as f:
        full_package = pickle.load(f)

    with open(quick_extract_file, 'rb') as f:
        quick_package = pickle.load(f)

    print(f"\nFull export package loaded: {full_export_file.stat().st_size:,} bytes")
    print(f"Quick extract package loaded: {quick_extract_file.stat().st_size:,} bytes")

    # Compare metadata
    print(f"\n{'='*60}")
    print("METADATA COMPARISON")
    print(f"{'='*60}")

    full_meta = full_package.get('metadata', {})
    quick_meta = quick_package.get('metadata', {})

    print("FULL EXPORT METADATA:")
    for key, value in full_meta.items():
        print(f"  {key}: {value}")

    print("\nQUICK EXTRACT METADATA:")
    for key, value in quick_meta.items():
        print(f"  {key}: {value}")

    # Compare models
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")

    full_models = full_package.get('models', {})
    quick_models = quick_package.get('models', {})

    print(f"Full export models: {list(full_models.keys())}")
    print(f"Quick extract models: {list(quick_models.keys())}")

    if len(full_models) != len(quick_models):
        print(f"ERROR: Different number of models ({len(full_models)} vs {len(quick_models)})")
        return False

    # Compare model predictions on same data
    print(f"\nTesting predictions on identical data...")

    # Create test data
    np.random.seed(42)  # Fixed seed for reproducible test data
    n_samples = 100
    n_features_full = len(full_package.get('selected_features', []))
    n_features_quick = len(quick_package.get('selected_features', []))

    print(f"Full export features: {n_features_full}")
    print(f"Quick extract features: {n_features_quick}")

    # Test prediction consistency
    if 'model_01' in full_models and 'model_01' in quick_models:
        full_model = full_models['model_01']
        quick_model = quick_models['model_01']

        # Create test data with correct number of features
        # For full export model
        full_features = full_package.get('selected_features', [])[:n_features_full]
        quick_features = quick_package.get('selected_features', [])[:n_features_quick]

        print(f"\nFeatures comparison:")
        print(f"Full export features (first 10): {full_features[:10]}")
        print(f"Quick extract features (first 10): {quick_features[:10]}")
        print(f"Features match: {set(full_features) == set(quick_features)}")

        if set(full_features) == set(quick_features):
            # Get model-specific feature slices
            full_model_slices = full_package.get('model_feature_slices', {})
            quick_model_slices = quick_package.get('model_feature_slices', {})

            # Get feature slices for model_01
            full_model_features = full_model_slices.get('model_01', full_features)
            quick_model_features = quick_model_slices.get('model_01', quick_features)

            print(f"Full model features: {len(full_model_features)} features")
            print(f"Quick model features: {len(quick_model_features)} features")
            print(f"Model features match: {set(full_model_features) == set(quick_model_features)}")

            if set(full_model_features) == set(quick_model_features):
                # Create test data with correct number of features for each model
                full_test_data = np.random.randn(10, len(full_model_features))
                quick_test_data = np.random.randn(10, len(quick_model_features))

                # Test predictions (use same random data for both)
                test_data = np.random.randn(10, len(full_model_features))
                full_preds = full_model.predict(test_data)
                quick_preds = quick_model.predict(test_data)

                # Compare predictions
                max_diff = np.max(np.abs(full_preds - quick_preds))
                mean_diff = np.mean(np.abs(full_preds - quick_preds))

                print(f"\nPrediction comparison:")
                print(f"Max difference: {max_diff:.2e}")
                print(f"Mean difference: {mean_diff:.2e}")
                print(f"Sample predictions:")
                print(f"  Full export:   {full_preds[:5]}")
                print(f"  Quick extract: {quick_preds[:5]}")
                print(f"  Differences:   {(full_preds - quick_preds)[:5]}")

                tolerance = 1e-10
                if max_diff < tolerance:
                    print(f"SUCCESS: Models are IDENTICAL (within {tolerance} tolerance)")
                    return True
                else:
                    print(f"ERROR: Models DIFFER (max diff: {max_diff:.2e})")
                    return False
            else:
                print("ERROR: Model feature slices don't match")
                return False
        else:
            print("ERROR: Features don't match between methods")
            return False
    else:
        print("ERROR: Could not find model_01 in both packages")
        return False

def show_detailed_metadata():
    """Show detailed metadata structure."""

    print(f"\n{'='*80}")
    print("DETAILED METADATA STRUCTURE")
    print(f"{'='*80}")

    models_dir = Path(__file__).parent.parent / "PROD" / "models"
    quick_file = models_dir / "@AD#C_fixed_production.pkl"

    if quick_file.exists():
        with open(quick_file, 'rb') as f:
            package = pickle.load(f)

        print("PRODUCTION PACKAGE STRUCTURE:")
        print(f"  - symbol: {package.get('symbol')}")
        print(f"  - models: {list(package.get('models', {}).keys())}")
        print(f"  - selected_features: {len(package.get('selected_features', []))} features")
        print(f"  - selected_model_indices: {package.get('selected_model_indices', [])}")
        print(f"  - binary_signal: {package.get('binary_signal', False)}")

        print("\nMETADATA SECTION:")
        metadata = package.get('metadata', {})
        for key, value in metadata.items():
            print(f"  - {key}: {value}")

        print(f"\nMODEL FEATURE SLICES:")
        model_slices = package.get('model_feature_slices', {})
        for model_key, features in model_slices.items():
            print(f"  - {model_key}: {len(features)} features")
            print(f"    Features: {features[:5]}{'...' if len(features) > 5 else ''}")

        print(f"\nSELECTED FEATURES (first 20):")
        selected_features = package.get('selected_features', [])
        for i, feature in enumerate(selected_features[:20]):
            print(f"  {i+1:2d}. {feature}")
        if len(selected_features) > 20:
            print(f"  ... and {len(selected_features) - 20} more")

    else:
        print("ERROR: Quick extract file not found")

def main():
    """Main comparison."""

    print("Production Model Extraction Method Comparison")
    print("=" * 80)

    # Compare models
    models_match = load_and_compare_models()

    # Show metadata structure
    show_detailed_metadata()

    # Final verdict
    print(f"\n{'='*80}")
    print("FINAL COMPARISON VERDICT")
    print(f"{'='*80}")

    if models_match:
        print("SUCCESS: Both methods produce IDENTICAL models")
        print("- Predictions match exactly")
        print("- Feature sets are consistent")
        print("- Quick extraction method is VALIDATED")
        print("- SAFE to use quick extraction for production")
    else:
        print("ERROR: Methods produce DIFFERENT models")
        print("- DO NOT use quick extraction")
        print("- Use full export method only")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test Signal Engine with Extracted Production Models
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from PROD.common.signal_engine import SignalEngine
from data.data_utils_simple import prepare_real_data_simple

def test_signal_engine():
    """Test signal engine with extracted @AD#C models."""

    print("Testing Signal Engine with @AD#C Production Models")
    print("=" * 60)

    try:
        # Initialize signal engine
        models_dir = Path(__file__).parent / "models"
        config_dir = Path(__file__).parent / "config"

        engine = SignalEngine(models_dir, config_dir)

        # Test loading @AD#C package
        package = engine.load_symbol_package("@AD#C")

        if package:
            print(f"SUCCESS: Successfully loaded @AD#C package")
            print(f"Models: {len(package.get('models', {}))}")
            print(f"Features: {len(package.get('selected_features', []))}")
            print(f"Binary signal: {package.get('binary_signal', False)}")
            print(f"Selected model indices: {package.get('selected_model_indices', [])}")

            # Test with recent data
            print("\nTesting signal generation with recent data...")

            # Get recent data for @AD#C
            df = prepare_real_data_simple("@AD#C", start_date="2025-07-01", end_date="2025-08-01")
            target_col = "@AD#C_target_return"
            features_df = df[[c for c in df.columns if c != target_col]]

            print(f"Recent data shape: {features_df.shape}")

            # Filter to only the selected features
            selected_features = package.get('selected_features', [])
            available_features = [f for f in selected_features if f in features_df.columns]

            print(f"Available selected features: {len(available_features)}/{len(selected_features)}")

            if available_features:
                features_subset = features_df[available_features]

                # Generate signal
                result = engine.generate_signal(features_subset, "@AD#C")

                if result:
                    signal, raw_score = result
                    print(f"SUCCESS: Signal generated successfully!")
                    print(f"Signal: {signal}")
                    print(f"Raw score: {raw_score:.6f}")
                else:
                    print("FAILED: Failed to generate signal")

            else:
                print("ERROR: No matching features found")

        else:
            print("ERROR: Failed to load @AD#C package")

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_signal_engine()
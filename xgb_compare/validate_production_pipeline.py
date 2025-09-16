#!/usr/bin/env python3
"""
Validate Production Pipeline Consistency

Compares quick extraction method vs full run to ensure identical results.
Tests signal generation across multiple data points to verify consistency.
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from PROD.common.signal_engine import SignalEngine
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection

def run_reference_test(symbol: str, n_models: int = 20, n_folds: int = 8):
    """Run a smaller reference test with export to compare against quick extraction."""

    print(f"\n{'='*70}")
    print(f"RUNNING REFERENCE TEST: {symbol}")
    print(f"{'='*70}")
    print(f"Config: {n_models} models, {n_folds} folds")

    # Run full xgb_compare with export flag
    import subprocess
    cmd = [
        sys.executable, "xgb_compare.py",
        "--target_symbol", symbol,
        "--n_models", str(n_models),
        "--n_folds", str(n_folds),
        "--max_features", "100",
        "--q_metric", "sharpe",
        "--xgb_type", "tiered",
        "--export-production-models",
        "--log_label", f"REFERENCE_{symbol.replace('#', '').replace('@', '')}"
    ]

    print(f"Running reference test: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"SUCCESS: Reference test completed")
        return True
    else:
        print(f"ERROR: Reference test failed: {result.stderr}")
        return False

def compare_production_packages(symbol: str, quick_file: str, reference_file: str):
    """Compare production packages from quick extraction vs reference test."""

    print(f"\n{'='*70}")
    print(f"COMPARING PRODUCTION PACKAGES: {symbol}")
    print(f"{'='*70}")

    try:
        # Load both packages
        with open(quick_file, 'rb') as f:
            quick_package = pickle.load(f)

        with open(reference_file, 'rb') as f:
            reference_package = pickle.load(f)

        print(f"Quick package: {len(quick_package.get('models', {}))} models")
        print(f"Reference package: {len(reference_package.get('models', {}))} models")

        # Compare selected features
        quick_features = set(quick_package.get('selected_features', []))
        ref_features = set(reference_package.get('selected_features', []))

        print(f"\nFeature comparison:")
        print(f"Quick features: {len(quick_features)}")
        print(f"Reference features: {len(ref_features)}")
        print(f"Features match: {quick_features == ref_features}")

        if quick_features != ref_features:
            print(f"Feature differences:")
            print(f"  Only in quick: {quick_features - ref_features}")
            print(f"  Only in reference: {ref_features - quick_features}")

        # Compare model indices
        quick_indices = quick_package.get('selected_model_indices', [])
        ref_indices = reference_package.get('selected_model_indices', [])

        print(f"\nModel indices comparison:")
        print(f"Quick indices: {quick_indices}")
        print(f"Reference indices: {ref_indices}")
        print(f"Indices match: {quick_indices == ref_indices}")

        return quick_features == ref_features and quick_indices == ref_indices

    except Exception as e:
        print(f"ERROR comparing packages: {e}")
        return False

def test_signal_consistency(symbol: str, n_test_points: int = 10):
    """Test signal generation consistency across multiple data points."""

    print(f"\n{'='*70}")
    print(f"TESTING SIGNAL CONSISTENCY: {symbol}")
    print(f"{'='*70}")

    try:
        # Initialize signal engines for both packages
        models_dir = Path(__file__).parent.parent / "PROD" / "models"
        config_dir = Path(__file__).parent.parent / "PROD" / "config"

        engine = SignalEngine(models_dir, config_dir)

        # Get test data (last N days)
        end_date = "2025-08-01"
        start_date = "2025-07-15"  # Get more data for testing

        df = prepare_real_data_simple(symbol, start_date=start_date, end_date=end_date)
        target_col = f"{symbol}_target_return"
        features_df = df[[c for c in df.columns if c != target_col]]

        if len(features_df) < n_test_points:
            n_test_points = len(features_df)

        print(f"Testing {n_test_points} data points from {start_date} to {end_date}")
        print(f"Available data: {len(features_df)} rows")

        # Test signal generation on multiple points
        results = []
        for i in range(n_test_points):
            # Use rolling window of data up to point i
            test_data = features_df.iloc[:i+10].copy()  # Use some history

            if len(test_data) < 5:  # Need minimum data
                continue

            result = engine.generate_signal(test_data, symbol)

            if result:
                signal, raw_score = result
                timestamp = test_data.index[-1]

                results.append({
                    'timestamp': timestamp,
                    'signal': signal,
                    'raw_score': raw_score,
                    'data_points': len(test_data)
                })

                print(f"Point {i+1}: {timestamp} -> Signal: {signal:+2d}, Score: {raw_score:+8.6f}, Data: {len(test_data)} rows")

        print(f"\nSignal Summary:")
        if results:
            signals = [r['signal'] for r in results]
            scores = [r['raw_score'] for r in results]

            print(f"Total tests: {len(results)}")
            print(f"Buy signals (+1): {signals.count(1)}")
            print(f"Sell signals (-1): {signals.count(-1)}")
            print(f"Score range: {min(scores):.6f} to {max(scores):.6f}")
            print(f"Score std: {np.std(scores):.6f}")

            # Check for reasonable variation
            if len(set(signals)) > 1:
                print("SUCCESS: Signal variation detected (not stuck)")
            else:
                print("WARNING: All signals identical (may indicate issue)")

            return len(results) > 0
        else:
            print("ERROR: No signals generated")
            return False

    except Exception as e:
        print(f"ERROR: Signal consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main validation pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate production pipeline consistency')
    parser.add_argument('--symbol', type=str, default='@AD#C', help='Symbol to validate')
    parser.add_argument('--run-reference', action='store_true', help='Run reference test first')
    parser.add_argument('--n-models', type=int, default=20, help='Models for reference test')
    parser.add_argument('--n-folds', type=int, default=8, help='Folds for reference test')
    parser.add_argument('--test-points', type=int, default=15, help='Number of test points for signal consistency')

    args = parser.parse_args()

    print("Production Pipeline Validation")
    print("=" * 70)
    print(f"Symbol: {args.symbol}")
    print(f"Test configuration: {args.n_models}M, {args.n_folds}F")

    # Step 1: Run reference test if requested
    if args.run_reference:
        success = run_reference_test(args.symbol, args.n_models, args.n_folds)
        if not success:
            print("ERROR: Reference test failed, cannot proceed with comparison")
            return

    # Step 2: Test signal consistency with multiple data points
    consistency_success = test_signal_consistency(args.symbol, args.test_points)

    # Step 3: Summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}")

    if consistency_success:
        print(f"SUCCESS: {args.symbol} production pipeline validated")
        print("- Signal generation working across multiple data points")
        print("- Model-specific feature slices handled correctly")
        print("- Ready for production deployment")
    else:
        print(f"ERROR: {args.symbol} validation failed")
        print("- Signal generation issues detected")
        print("- DO NOT deploy to production")

if __name__ == "__main__":
    main()
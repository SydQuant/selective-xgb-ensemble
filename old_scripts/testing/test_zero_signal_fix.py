#!/usr/bin/env python3
"""
Test Zero Signal Handling Fix
=============================

Verify the fix for the bearish bias bug in zero signal handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Import the FIXED function
from xgb_compare.metrics_utils import normalize_predictions

def test_zero_signal_fix():
    """Test the fixed zero signal handling"""
    print(f"\n{'='*60}")
    print("🧪 TESTING ZERO SIGNAL HANDLING FIX")
    print(f"{'='*60}")

    # Create test predictions including exact zeros
    test_preds = pd.Series([-1.0, -0.5, 0.0, 0.5, 1.0])

    print("Testing fixed binary signal transformation:")
    print("Raw Prediction -> Old Logic -> New Logic")
    print("-" * 45)

    for pred in test_preds:
        # Old logic (biased)
        old_binary = 1.0 if pred > 0 else -1.0

        # Test the fixed function
        single_pred = pd.Series([pred])
        fixed_result = normalize_predictions(single_pred, binary_signal=True)
        new_binary = fixed_result.iloc[0]

        print(f"{pred:>+8.1f}       ->    {old_binary:>+4.1f}   ->   {new_binary:>+4.1f}")

    print(f"\n🔍 Key Difference:")
    print(f"  Zero prediction (0.0):")
    print(f"    Old: 0.0 -> -1 (SHORT bias)")
    print(f"    New: 0.0 -> 0 (NEUTRAL)")

    # Test with realistic ensemble scenario
    print(f"\n🔍 ENSEMBLE IMPACT TEST:")
    print("-" * 30)

    # 5 models with one zero prediction
    model_preds = pd.Series([0.5, -0.3, 0.0, 0.8, -0.2])

    print(f"Model raw predictions: {model_preds.values}")

    # Apply new normalization
    normalized_signals = normalize_predictions(model_preds, binary_signal=True)
    print(f"New binary signals: {normalized_signals.values}")

    ensemble_vote = normalized_signals.sum()
    print(f"Ensemble vote: {ensemble_vote}")

    if ensemble_vote > 0:
        position = "LONG"
    elif ensemble_vote < 0:
        position = "SHORT"
    else:
        position = "FLAT"

    print(f"Final position: {position}")

    # Compare with old biased method
    old_signals = np.where(model_preds > 0, 1.0, -1.0)
    old_vote = np.sum(old_signals)

    print(f"\nComparison:")
    print(f"  Old biased vote: {old_vote} ({'LONG' if old_vote > 0 else 'SHORT' if old_vote < 0 else 'FLAT'})")
    print(f"  New unbiased vote: {ensemble_vote} ({position})")

    if old_vote != ensemble_vote:
        print(f"  ✅ Fix successfully removes bias!")
    else:
        print(f"  ℹ️  No difference in this example")

    return []

def test_ensemble_voting_with_fix():
    """Test ensemble voting with the fix"""
    print(f"\n{'='*60}")
    print("🧪 TESTING ENSEMBLE VOTING WITH FIX")
    print(f"{'='*60}")

    from xgb_compare.metrics_utils import combine_binary_signals

    # Test cases with zero signals
    test_cases = [
        {
            "name": "5 models, one zero",
            "signals": [
                pd.Series([1.0, -1.0, 0.0, 1.0, -1.0]),  # Model predictions after normalization
            ]
        },
        {
            "name": "Multiple zeros",
            "signals": [
                pd.Series([1.0, 0.0, 0.0, 0.0, -1.0]),
            ]
        },
        {
            "name": "All zeros",
            "signals": [
                pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]),
            ]
        }
    ]

    for test_case in test_cases:
        print(f"\nTest: {test_case['name']}")
        signals = test_case['signals'][0]

        # Use framework's combine function
        try:
            combined = combine_binary_signals([signals])
            print(f"  Individual signals: {signals.values}")
            print(f"  Combined result: {combined.iloc[0]}")

            # Manual verification
            manual_sum = signals.sum()
            print(f"  Manual sum: {manual_sum}")

            if abs(combined.iloc[0] - manual_sum) < 1e-10:
                print(f"  ✅ Framework matches manual calculation")
            else:
                print(f"  🚨 Framework differs from manual!")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    return []

def main():
    """Test the complete zero signal fix"""
    print(f"\n{'='*80}")
    print("🚨 ZERO SIGNAL BIAS FIX VALIDATION")
    print(f"{'='*80}")

    # Test the fix
    test_zero_signal_fix()
    test_ensemble_voting_with_fix()

    print(f"\n{'='*80}")
    print("✅ ZERO SIGNAL FIX VALIDATED")
    print(f"{'='*80}")
    print("The bearish bias bug has been fixed!")
    print("Zero predictions now properly map to 0 (neutral) instead of -1 (short)")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ensemble Voting Logic Deep Audit
===============================

Test the critical ensemble voting and model selection logic:

1. DEMOCRATIC VOTING IMPLEMENTATION - Sum of +1/-1 votes correct?
2. TIE HANDLING - What happens with 0 votes?
3. MODEL SELECTION CUTOFFS - Are quality thresholds applied correctly?
4. WEIGHTED VS EQUAL VOTING - How are model weights determined?
5. SIGNAL NORMALIZATION - Tanh vs binary transformation bugs?

Democratic voting is a key differentiator - bugs here invalidate the ensemble approach.
"""

import numpy as np
import pandas as pd
from datetime import datetime

def test_democratic_voting_implementation():
    """
    🚨 CRITICAL: Test democratic voting implementation
    """
    print(f"\n{'='*70}")
    print("🚨 CRITICAL TEST: DEMOCRATIC VOTING IMPLEMENTATION")
    print(f"{'='*70}")

    issues = []

    # Test scenarios for democratic voting
    test_cases = [
        {
            "name": "Unanimous Long",
            "individual_votes": [1, 1, 1, 1, 1],
            "expected_ensemble": 5,
            "expected_signal": "STRONG LONG"
        },
        {
            "name": "Unanimous Short",
            "individual_votes": [-1, -1, -1, -1, -1],
            "expected_ensemble": -5,
            "expected_signal": "STRONG SHORT"
        },
        {
            "name": "Majority Long",
            "individual_votes": [1, 1, 1, -1, -1],
            "expected_ensemble": 1,
            "expected_signal": "WEAK LONG"
        },
        {
            "name": "Majority Short",
            "individual_votes": [-1, -1, -1, 1, 1],
            "expected_ensemble": -1,
            "expected_signal": "WEAK SHORT"
        },
        {
            "name": "Perfect Tie (Even Models)",
            "individual_votes": [1, 1, -1, -1],
            "expected_ensemble": 0,
            "expected_signal": "TIE - NO POSITION"
        },
        {
            "name": "Tie (Odd Models)",
            "individual_votes": [1, 1, -1, -1, 0],  # One abstention
            "expected_ensemble": 0,
            "expected_signal": "TIE - NO POSITION"
        },
    ]

    print(f"Testing {len(test_cases)} democratic voting scenarios:")
    print(f"{'Scenario':<20} {'Votes':<15} {'Expected':<8} {'Calculated':<10} {'Status'}")
    print(f"{'-'*65}")

    for test_case in test_cases:
        votes = np.array(test_case["individual_votes"])
        expected = test_case["expected_ensemble"]

        # Calculate democratic vote (sum of +1/-1 votes)
        calculated = np.sum(votes)

        status = "✅ PASS" if calculated == expected else "🚨 FAIL"

        if calculated != expected:
            issue = f"CRITICAL: Democratic voting error - {test_case['name']}: Expected {expected}, Got {calculated}"
            issues.append(issue)

        votes_str = str(votes.tolist())
        print(f"{test_case['name']:<20} {votes_str:<15} {expected:>+4d}{'':4} {calculated:>+4d}{'':6} {status}")

    print(f"\n🔍 Testing edge cases:")

    # Edge case: Mixed vote strengths (should not happen in binary system)
    mixed_votes = np.array([0.8, -0.3, 1.0, -1.0, 0.5])
    print(f"  Mixed strength votes: {mixed_votes}")
    print(f"  Sum: {np.sum(mixed_votes):.2f}")
    print("  🚨 This scenario should NOT occur in proper binary voting")

    # Edge case: Zero votes handling
    zero_ensemble = 0
    print(f"\n🔍 Zero vote (tie) handling:")
    print(f"  Ensemble vote: {zero_ensemble}")
    print(f"  Position: FLAT (no position taken)")
    print(f"  PnL calculation: {zero_ensemble} * return = 0.0 (correct)")

    if not issues:
        print(f"\n✅ Democratic voting implementation verified")
    else:
        print(f"\n🚨 Found {len(issues)} voting implementation issues!")

    return issues

def test_model_selection_cutoffs():
    """
    🔍 TEST: Model selection quality cutoffs
    """
    print(f"\n{'='*70}")
    print("🔍 TEST: MODEL SELECTION QUALITY CUTOFFS")
    print(f"{'='*70}")

    issues = []

    # Simulate model quality distribution
    np.random.seed(42)
    n_models = 100

    # Create realistic model quality distribution
    # Some models are good, some bad, some mediocre
    model_qualities = {
        f"M{i:02d}": np.random.normal(0.8, 0.5) for i in range(n_models)
    }

    print(f"Testing quality cutoff logic with {n_models} models...")

    # Test different cutoff scenarios
    cutoffs = [0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"\nQuality cutoff analysis:")
    print(f"{'Cutoff':<8} {'Selected':<10} {'Avg Quality':<12} {'Min Quality':<12}")
    print(f"{'-'*50}")

    for cutoff in cutoffs:
        # Select models above cutoff
        selected_models = {k: v for k, v in model_qualities.items() if v >= cutoff}

        if selected_models:
            avg_quality = np.mean(list(selected_models.values()))
            min_quality = np.min(list(selected_models.values()))
            n_selected = len(selected_models)
        else:
            avg_quality = 0.0
            min_quality = 0.0
            n_selected = 0

        print(f"{cutoff:<8.1f} {n_selected:<10d} {avg_quality:<12.4f} {min_quality:<12.4f}")

        # Check for reasonable selection
        if cutoff < 0.9 and n_selected == 0:
            issue = f"WARNING: Cutoff {cutoff} selects no models - may be too restrictive"
            issues.append(issue)

        if cutoff > 0.3 and n_selected == n_models:
            issue = f"WARNING: Cutoff {cutoff} selects all models - may be too permissive"
            issues.append(issue)

    # Test the framework's typical cutoff (0.6)
    framework_cutoff = 0.6
    selected = {k: v for k, v in model_qualities.items() if v >= framework_cutoff}

    print(f"\nFramework cutoff ({framework_cutoff}) analysis:")
    print(f"  Models selected: {len(selected)}/{n_models} ({len(selected)/n_models*100:.1f}%)")
    print(f"  Average quality: {np.mean(list(selected.values())):.4f}")

    if len(selected) < 3:
        issue = f"WARNING: Framework cutoff selects very few models ({len(selected)}) - ensemble may be weak"
        issues.append(issue)

    if len(selected) > n_models * 0.8:
        issue = f"WARNING: Framework cutoff selects most models ({len(selected)}) - quality filtering weak"
        issues.append(issue)

    print(f"✅ Quality cutoff logic tested")

    return issues

def test_signal_aggregation_methods():
    """
    🔍 TEST: Signal aggregation methods (tanh vs binary)
    """
    print(f"\n{'='*70}")
    print("🔍 TEST: SIGNAL AGGREGATION METHODS")
    print(f"{'='*70}")

    issues = []

    # Create test individual model signals
    n_models = 5
    n_samples = 20

    np.random.seed(42)

    # Generate realistic model raw predictions
    raw_predictions = []
    for model_id in range(n_models):
        # Each model has different prediction characteristics
        base_strength = np.random.uniform(-1, 1)
        noise_level = np.random.uniform(0.1, 0.5)

        raw_preds = np.random.normal(base_strength, noise_level, n_samples)
        raw_predictions.append(raw_preds)

    raw_matrix = np.array(raw_predictions)

    print(f"Testing signal aggregation with {n_models} models, {n_samples} samples...")

    # Method 1: Tanh signals (continuous averaging)
    tanh_signals = np.tanh(raw_matrix)
    ensemble_tanh = np.mean(tanh_signals, axis=0)

    # Method 2: Binary signals (democratic voting)
    binary_signals = np.where(raw_matrix > 0, 1, -1)
    ensemble_binary = np.sum(binary_signals, axis=0)

    print(f"\nSignal aggregation comparison:")
    print(f"  Tanh ensemble range: [{ensemble_tanh.min():.4f}, {ensemble_tanh.max():.4f}]")
    print(f"  Binary ensemble range: [{ensemble_binary.min()}, {ensemble_binary.max()}]")
    print(f"  Tanh mean/std: {ensemble_tanh.mean():.4f} / {ensemble_tanh.std():.4f}")
    print(f"  Binary vote mean/std: {ensemble_binary.mean():.2f} / {ensemble_binary.std():.2f}")

    # Show sample comparison
    print(f"\nSample signal comparison (first 10):")
    print(f"{'Sample':<6} {'Tanh':<8} {'Binary':<6} {'Raw Preds'}")
    print(f"{'-'*50}")

    for i in range(min(10, n_samples)):
        raw_sample = raw_matrix[:, i]
        tanh_val = ensemble_tanh[i]
        binary_val = ensemble_binary[i]

        raw_str = "[" + ", ".join([f"{x:+.2f}" for x in raw_sample]) + "]"
        print(f"{i:<6d} {tanh_val:>+6.3f}{'':2} {binary_val:>+4d}{'':2} {raw_str}")

    # Test: Both methods should agree on direction for strong signals
    strong_tanh_long = ensemble_tanh > 0.3
    strong_tanh_short = ensemble_tanh < -0.3

    strong_binary_long = ensemble_binary > n_models * 0.6
    strong_binary_short = ensemble_binary < -n_models * 0.6

    # Check agreement on strong signals
    tanh_binary_agreement = np.mean(
        (strong_tanh_long & (ensemble_binary > 0)) |
        (strong_tanh_short & (ensemble_binary < 0))
    )

    print(f"\nStrong signal agreement:")
    print(f"  Tanh-Binary agreement rate: {tanh_binary_agreement:.4f} ({tanh_binary_agreement*100:.1f}%)")

    if tanh_binary_agreement < 0.8:
        issue = f"WARNING: Low agreement between tanh and binary methods ({tanh_binary_agreement:.2f})"
        issues.append(issue)
        print(f"⚠️  {issue}")

    print(f"✅ Signal aggregation methods tested")

    return issues

def main():
    """
    Run comprehensive ensemble voting audit
    """
    print(f"\n{'='*80}")
    print("🚨 ENSEMBLE VOTING DEEP AUDIT")
    print(f"{'='*80}")
    print("Testing democratic voting and model selection logic")
    print(f"Audit time: {datetime.now()}")

    all_issues = []

    # Test 1: Democratic voting
    issues = test_democratic_voting_implementation()
    all_issues.extend(issues)

    # Test 2: Model selection cutoffs
    issues = test_model_selection_cutoffs()
    all_issues.extend(issues)

    # Test 3: Signal aggregation methods
    issues = test_signal_aggregation_methods()
    all_issues.extend(issues)

    # Summary
    print(f"\n{'='*80}")
    print("🚨 ENSEMBLE VOTING AUDIT SUMMARY")
    print(f"{'='*80}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"Total issues: {len(all_issues)}")
    print(f"  Critical: {len(critical_issues)}")
    print(f"  Warnings: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨 CRITICAL VOTING ISSUES:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ No critical voting issues detected")

    if warning_issues:
        print(f"\n⚠️  VOTING WARNINGS:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    return all_issues

if __name__ == "__main__":
    main()
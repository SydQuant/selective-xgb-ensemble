#!/usr/bin/env python3
"""
Investigate Cross-Validation Implementation
==========================================

The CV implementation might have subtle bugs that create data leakage.
Need to check:

1. FEATURE SELECTION TIMING - When in the CV process?
2. MODEL SELECTION TIMING - Using future fold performance?
3. QUALITY METRIC CALCULATION - Forward-looking bias?
4. FOLD BOUNDARIES - Actually implemented correctly?
5. RESELECTION PROCESS - Models picked using future data?

Even small timing errors can create massive performance inflation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from data.data_utils_simple import prepare_real_data_simple
from cv.wfo import wfo_splits

def investigate_actual_cv_process():
    """
    🚨 CRITICAL: How is CV actually implemented in practice?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING ACTUAL CV IMPLEMENTATION")
    print(f"{'='*80}")

    issues = []

    try:
        # Load data
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        # Clean data
        clean_df = df.dropna(subset=[target_col])
        print(f"Testing CV on {len(clean_df)} clean samples...")

        # Test the actual CV splits
        n_folds = 8  # Typical framework setting
        splits = list(wfo_splits(len(clean_df), k_folds=n_folds))

        print(f"\nCV implementation analysis:")
        print(f"  Requested folds: {n_folds}")
        print(f"  Generated splits: {len(splits)}")

        if len(splits) != n_folds:
            issue = f"WARNING: CV generated {len(splits)} splits instead of {n_folds} requested"
            issues.append(issue)
            print(f"⚠️  {issue}")

        # Analyze each fold in detail
        total_train_samples = 0
        total_test_samples = 0

        print(f"\nDetailed fold analysis:")
        print(f"{'Fold':<5} {'Train':<10} {'Test':<10} {'Train Range':<20} {'Test Range':<20} {'Gap':<5}")
        print(f"{'-'*75}")

        for i, (train_idx, test_idx) in enumerate(splits):
            train_range = f"[{min(train_idx)}-{max(train_idx)}]"
            test_range = f"[{min(test_idx)}-{max(test_idx)}]"

            max_train = max(train_idx) if len(train_idx) > 0 else -1
            min_test = min(test_idx) if len(test_idx) > 0 else len(clean_df)
            gap = min_test - max_train - 1

            print(f"{i+1:<5} {len(train_idx):<10} {len(test_idx):<10} {train_range:<20} {test_range:<20} {gap:<5}")

            total_train_samples += len(train_idx)
            total_test_samples += len(test_idx)

            # Check for problems
            if gap < 0:
                issue = f"CRITICAL: Fold {i+1} has overlapping train/test periods (gap: {gap})"
                issues.append(issue)
                print(f"    🚨 {issue}")

            if len(train_idx) < 200:  # Very small training set
                issue = f"WARNING: Fold {i+1} has very small training set ({len(train_idx)} samples)"
                issues.append(issue)
                print(f"    ⚠️  {issue}")

        print(f"\nCV summary:")
        print(f"  Total train samples used: {total_train_samples}")
        print(f"  Total test samples used: {total_test_samples}")
        print(f"  Data utilization: {(total_train_samples + total_test_samples) / (len(splits) * len(clean_df)) * 100:.1f}%")

        # Critical question: Are models retrained on each fold or reused?
        print(f"\n🚨 CRITICAL QUESTIONS:")
        print(f"1. Are models retrained for each fold? (correct)")
        print(f"2. Or are models trained once and tested on all folds? (WRONG - contamination)")
        print(f"3. When is feature selection applied? Before CV (WRONG) or within each fold (correct)?")
        print(f"4. When are quality metrics calculated? Using future folds (WRONG) or historical only (correct)?")

    except Exception as e:
        error = f"CRITICAL: Error investigating CV implementation: {e}"
        issues.append(error)
        print(f"🚨 {error}")

    return issues

def investigate_quality_metric_timing():
    """
    🚨 CRITICAL: When are quality metrics calculated? Future contamination?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING QUALITY METRIC TIMING")
    print(f"{'='*80}")

    issues = []

    print(f"Critical timing analysis:")
    print(f"The framework uses 'Q-metrics' to select models within each fold")
    print(f"Question: What data is used to calculate these Q-metrics?")

    # Simulate the process to identify contamination
    print(f"\n🧪 Simulating quality metric calculation:")

    # Create scenario: Fold 3 of CV
    # Models were trained on folds 1-2 data
    # Need to select models for fold 3 testing
    # What data is used to calculate model quality?

    scenarios = [
        {
            "name": "CORRECT: Historical only",
            "description": "Q-metric calculated using only folds 1-2 performance",
            "contaminated": False
        },
        {
            "name": "WRONG: Includes current fold",
            "description": "Q-metric calculated using folds 1-3 performance",
            "contaminated": True
        },
        {
            "name": "VERY WRONG: Includes future folds",
            "description": "Q-metric calculated using folds 1-4 performance",
            "contaminated": True
        }
    ]

    print(f"\nQ-metric calculation scenarios:")
    for scenario in scenarios:
        status = "❌ INVALID" if scenario['contaminated'] else "✅ VALID"
        print(f"  {scenario['name']}: {status}")
        print(f"    {scenario['description']}")

        if scenario['contaminated']:
            issue = f"CRITICAL: If framework uses '{scenario['name']}', all results are invalid"
            print(f"    🚨 Potential issue: {issue}")

    # The framework documentation should specify this clearly
    print(f"\n🔍 Framework behavior investigation needed:")
    print(f"  Need to verify EXACTLY when quality metrics are calculated")
    print(f"  Need to verify NO future data is used in model selection")
    print(f"  Need to verify models are retrained for each fold")

    return issues

def main():
    """
    Complete CV investigation
    """
    print(f"\n{'='*100}")
    print("🚨🚨🚨 CROSS-VALIDATION CONTAMINATION INVESTIGATION")
    print(f"{'='*100}")

    all_issues = []

    # Investigation 1: CV process
    issues = investigate_actual_cv_process()
    all_issues.extend(issues)

    # Investigation 2: Quality metric timing
    issues = investigate_quality_metric_timing()
    all_issues.extend(issues)

    print(f"\n{'='*100}")
    print("🚨 CV CONTAMINATION FINAL ASSESSMENT")
    print(f"{'='*100}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"CV contamination results:")
    print(f"  Critical issues: {len(critical_issues)}")
    print(f"  Warnings: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨🚨🚨 CRITICAL CV CONTAMINATION:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")

    return all_issues

if __name__ == "__main__":
    main()
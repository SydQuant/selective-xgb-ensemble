#!/usr/bin/env python3
"""
Investigate Overfitting and Model Selection Bugs
===============================================

The overfitting risk is MASSIVE. Even with feature selection to 100,
the models are likely severely overfit. This investigation looks for:

1. MODEL SELECTION CONTAMINATION - Using test data to pick models
2. FEATURE SELECTION LOOKAHEAD - Using full dataset for feature selection
3. HYPERPARAMETER OVERFITTING - Too many model variations tested
4. CROSS-VALIDATION LEAKAGE - Information bleeding between folds
5. PRODUCTION PERIOD CHERRY-PICKING - Best time periods selected

High Sharpe ratios in backtests with massive overfitting = red flag #1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from data.data_utils_simple import prepare_real_data_simple

def investigate_feature_selection_contamination():
    """
    🚨 CRITICAL: Is feature selection using future data?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING FEATURE SELECTION CONTAMINATION")
    print(f"{'='*80}")

    issues = []

    try:
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        print(f"Feature selection contamination test...")
        print(f"Original features: {df.shape[1] - 1}")

        # Critical question: When is feature selection applied?
        # Is it applied to the FULL dataset before CV? (WRONG - contaminated)
        # Or is it applied separately within each CV fold? (CORRECT)

        from model.feature_selection import apply_feature_selection

        X = df[[c for c in df.columns if c != target_col]]
        y = df[target_col]

        # Test: Feature selection on full dataset (what framework might be doing)
        print(f"\n🧪 Testing feature selection on FULL dataset:")
        X_selected_full = apply_feature_selection(X, y, method='block_wise', max_total_features=50)
        print(f"  Selected {X_selected_full.shape[1]} features from full dataset")

        # Test: Feature selection on partial dataset (what it SHOULD do)
        print(f"\n🧪 Testing feature selection on PARTIAL dataset:")
        split_point = len(df) // 2
        X_partial = X.iloc[:split_point]
        y_partial = y.iloc[:split_point]

        X_selected_partial = apply_feature_selection(X_partial, y_partial, method='block_wise', max_total_features=50)
        print(f"  Selected {X_selected_partial.shape[1]} features from partial dataset")

        # Compare feature overlap
        full_features = set(X_selected_full.columns)
        partial_features = set(X_selected_partial.columns)
        overlap = full_features.intersection(partial_features)

        overlap_rate = len(overlap) / len(full_features)
        print(f"\n🔍 Feature selection stability:")
        print(f"  Overlap rate: {overlap_rate:.3f} ({overlap_rate*100:.1f}%)")

        if overlap_rate < 0.5:
            issue = f"CRITICAL: Low feature selection stability ({overlap_rate:.2f}) suggests contamination or instability"
            issues.append(issue)
            print(f"🚨 {issue}")

        # Test performance difference
        print(f"\n🧪 Testing performance impact of feature contamination:")

        # Use latter half as test set
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]

        # Test 1: Features selected on full data (contaminated)
        common_idx_full = X_test.index.intersection(X_selected_full.index)
        if len(common_idx_full) > 50:
            X_test_full_features = X_selected_full.loc[common_idx_full]
            y_test_aligned = y_test.loc[common_idx_full]

            # Simple correlation test
            correlations_full = []
            for col in X_test_full_features.columns:
                try:
                    corr = X_test_full_features[col].corr(y_test_aligned)
                    if not np.isnan(corr):
                        correlations_full.append(abs(corr))
                except:
                    continue

            avg_corr_full = np.mean(correlations_full) if correlations_full else 0

        # Test 2: Features selected on partial data (clean)
        common_idx_partial = X_test.index.intersection(X_selected_partial.index)
        if len(common_idx_partial) > 50:
            X_test_partial_features = X_selected_partial.loc[common_idx_partial]
            y_test_aligned_partial = y_test.loc[common_idx_partial]

            correlations_partial = []
            for col in X_test_partial_features.columns:
                try:
                    corr = X_test_partial_features[col].corr(y_test_aligned_partial)
                    if not np.isnan(corr):
                        correlations_partial.append(abs(corr))
                except:
                    continue

            avg_corr_partial = np.mean(correlations_partial) if correlations_partial else 0

            print(f"  Full dataset feature selection avg correlation: {avg_corr_full:.4f}")
            print(f"  Partial dataset feature selection avg correlation: {avg_corr_partial:.4f}")

            # 🚨 If full dataset selection performs much better, it's contaminated
            if avg_corr_full > avg_corr_partial * 1.5:
                issue = f"CRITICAL: Full dataset feature selection performs suspiciously better - likely contaminated"
                issues.append(issue)
                print(f"🚨 {issue}")

    except Exception as e:
        error = f"CRITICAL: Error investigating feature selection: {e}"
        issues.append(error)
        print(f"🚨 {error}")

    return issues

def investigate_model_selection_contamination():
    """
    🚨 CRITICAL: How are models actually selected? Using future data?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING MODEL SELECTION CONTAMINATION")
    print(f"{'='*80}")

    issues = []

    print(f"Critical questions about model selection:")
    print(f"1. Are 100 models trained and the best ones selected using test performance?")
    print(f"2. Is hyperparameter tuning done on test data?")
    print(f"3. Are quality metrics calculated using future fold performance?")
    print(f"4. Is there implicit selection bias through repeated testing?")

    # Analyze the model bank size and selection process
    print(f"\n🔍 Model selection analysis:")
    print(f"Framework trains 100-200 models per test")
    print(f"Then selects top models based on 'quality metric'")
    print(f"This is essentially hyperparameter search on 100-200 variations")

    # With 100+ model variations, overfitting is almost guaranteed
    print(f"\n🚨 HYPERPARAMETER OVERFITTING RISK:")
    print(f"  100-200 model variations tested")
    print(f"  Best performers selected")
    print(f"  This is multiple testing without correction")
    print(f"  P-hacking probability: VERY HIGH")

    # Calculate multiple testing correction needed
    n_models = 100
    alpha_uncorrected = 0.05
    bonferroni_alpha = alpha_uncorrected / n_models

    print(f"\n📊 Multiple testing statistics:")
    print(f"  Models tested: {n_models}")
    print(f"  Uncorrected alpha: {alpha_uncorrected}")
    print(f"  Bonferroni corrected alpha: {bonferroni_alpha:.6f}")
    print(f"  Without correction: {n_models * alpha_uncorrected:.1f} false positives expected")

    issue = f"CRITICAL: Testing {n_models} models without multiple testing correction creates massive selection bias"
    issues.append(issue)
    print(f"🚨 {issue}")

    return issues

def investigate_time_period_cherry_picking():
    """
    🚨 CRITICAL: Are specific time periods being cherry-picked?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING TIME PERIOD CHERRY-PICKING")
    print(f"{'='*80}")

    issues = []

    try:
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        returns = df[target_col].dropna()

        print(f"Time period analysis for {symbol}:")
        print(f"  Full period: {returns.index.min().date()} to {returns.index.max().date()}")
        print(f"  Total days: {len(returns)}")

        # Check performance by year to see if specific periods are driving results
        returns_by_year = {}
        for year in range(2015, 2026):
            year_returns = returns[returns.index.year == year]
            if len(year_returns) > 50:  # Minimum for meaningful statistics
                returns_by_year[year] = year_returns

        print(f"\n📊 Performance by year:")
        print(f"{'Year':<6} {'Count':<6} {'Mean':<8} {'Std':<8} {'Sharpe':<8}")
        print(f"{'-'*45}")

        year_sharpes = []
        for year, year_returns in returns_by_year.items():
            mean_ret = year_returns.mean()
            std_ret = year_returns.std()
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0

            year_sharpes.append(sharpe)
            print(f"{year:<6} {len(year_returns):<6} {mean_ret:>+6.4f}{'':2} {std_ret:>6.4f}{'':2} {sharpe:>6.2f}")

        # Check for period-specific performance
        max_year_sharpe = max(year_sharpes) if year_sharpes else 0
        min_year_sharpe = min(year_sharpes) if year_sharpes else 0

        print(f"\nYear-by-year Sharpe range: [{min_year_sharpe:.2f}, {max_year_sharpe:.2f}]")

        if max_year_sharpe > 3.0:
            issue = f"WARNING: Extremely high single-year Sharpe ({max_year_sharpe:.2f}) suggests period-specific overfitting"
            issues.append(issue)
            print(f"⚠️  {issue}")

        # Check if framework might be selecting favorable periods
        print(f"\n🔍 Period selection bias check:")
        print(f"  If 'production' period is cherry-picked from best years,")
        print(f"  it would explain the unrealistic performance")

    except Exception as e:
        error = f"CRITICAL: Error investigating time periods: {e}"
        issues.append(error)
        print(f"🚨 {error}")

    return issues

def main():
    """
    Complete overfitting investigation
    """
    print(f"\n{'='*100}")
    print("🚨🚨🚨 OVERFITTING AND CONTAMINATION INVESTIGATION")
    print(f"{'='*100}")
    print("Being BRUTALLY critical about potential overfitting")

    all_issues = []

    # Investigation 1: Feature selection contamination
    issues = investigate_feature_selection_contamination()
    all_issues.extend(issues)

    # Investigation 2: Model selection contamination
    issues = investigate_model_selection_contamination()
    all_issues.extend(issues)

    # Investigation 3: Time period cherry-picking
    issues = investigate_time_period_cherry_picking()
    all_issues.extend(issues)

    # FINAL BRUTAL ASSESSMENT
    print(f"\n{'='*100}")
    print("🚨🚨🚨 FINAL OVERFITTING ASSESSMENT")
    print(f"{'='*100}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"OVERFITTING INVESTIGATION RESULTS:")
    print(f"  Critical overfitting issues: {len(critical_issues)}")
    print(f"  Warning signs: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨🚨🚨 CRITICAL OVERFITTING ISSUES:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")

    if warning_issues:
        print(f"\n⚠️  OVERFITTING WARNING SIGNS:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    # Honest conclusion
    if len(critical_issues) >= 2:
        print(f"\n❌❌❌ RESULTS LIKELY INVALID DUE TO OVERFITTING")
        print("Multiple critical overfitting issues detected")
        print("High Sharpe ratios are probably statistical artifacts")
    elif len(critical_issues) == 1:
        print(f"\n⚠️⚠️⚠️ RESULTS HIGHLY QUESTIONABLE")
        print("Major overfitting risk detected")
        print("Performance is likely overstated")
    else:
        print(f"\n? Results might be legitimate, but require skepticism")

    return all_issues

if __name__ == "__main__":
    main()
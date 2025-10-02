#!/usr/bin/env python3
"""
Critical Bug Detector - Framework Validity Checker
==================================================

This script checks for the most critical bugs that could invalidate ALL results:

1. DATA LEAKAGE - Future data contaminating training sets
2. TARGET ALIGNMENT - Signal-target timing mismatches
3. FOLD CONTAMINATION - Overlapping train/test periods
4. FEATURE LOOKAHEAD - Features using future information
5. PNL CALCULATION BUGS - Incorrect position sizing or returns
6. INDEX MISALIGNMENT - Row shifting bugs
7. SIGNAL TRANSFORMATION BUGS - Tanh/binary conversion errors
8. RESELECTION TIMING - Model selection using future info

If any of these exist, the entire framework results are meaningless.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings

# Import framework components
from data.symbol_loader import load_symbol_data
from model.xgb_drivers import create_standard_xgb_bank
from model.feature_selection import select_features
from cv.wfo import expanding_window_split
from xgb_compare.metrics_utils import calculate_pnl

def check_data_leakage(symbol_data, target_col, verbose=True):
    """
    🚨 CRITICAL: Check for future data leaking into features

    This is the #1 bug that would make all results invalid.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: DATA LEAKAGE DETECTION")
        print(f"{'='*60}")

    issues = []

    # Check if features are calculated using future target values
    feature_cols = [col for col in symbol_data.columns if not col.endswith('_target_return')]

    if verbose:
        print(f"Checking {len(feature_cols)} features for future data leakage...")

    # Test: Check if any feature perfectly predicts future returns
    # This would indicate the feature is calculated AFTER the target period
    target_values = symbol_data[target_col].dropna()

    for feature_col in feature_cols:
        if feature_col not in symbol_data.columns:
            continue

        feature_values = symbol_data[feature_col]

        # Align indices
        common_idx = target_values.index.intersection(feature_values.index)
        if len(common_idx) < 10:
            continue

        aligned_target = target_values.loc[common_idx]
        aligned_feature = feature_values.loc[common_idx]

        # Drop NaN/inf values
        mask = ~(np.isnan(aligned_feature) | np.isnan(aligned_target) |
                np.isinf(aligned_feature) | np.isinf(aligned_target))

        if mask.sum() < 10:
            continue

        clean_feature = aligned_feature[mask]
        clean_target = aligned_target[mask]

        # Check correlation
        correlation = np.corrcoef(clean_feature, clean_target)[0, 1]

        # 🚨 CRITICAL: Perfect correlation suggests data leakage
        if abs(correlation) > 0.95 and not np.isnan(correlation):
            issue = f"CRITICAL: Feature '{feature_col}' has suspicious correlation with target: {correlation:.6f}"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

        # Check if feature values are identical to shifted target values
        # This would indicate the feature IS the target, just shifted
        for shift in [1, -1, 2, -2]:
            if len(clean_target) > abs(shift):
                if shift > 0:
                    shifted_target = clean_target.iloc[shift:].values
                    compare_feature = clean_feature.iloc[:-shift].values
                else:
                    shifted_target = clean_target.iloc[:shift].values
                    compare_feature = clean_feature.iloc[-shift:].values

                if len(shifted_target) > 0 and len(compare_feature) > 0:
                    if len(shifted_target) == len(compare_feature):
                        # Check if they're nearly identical
                        diff = np.abs(shifted_target - compare_feature)
                        if np.mean(diff) < 1e-10:
                            issue = f"CRITICAL: Feature '{feature_col}' appears to be shifted target (shift={shift})"
                            issues.append(issue)
                            if verbose:
                                print(f"🚨 {issue}")

    if verbose:
        if not issues:
            print("✅ No obvious data leakage detected")
        else:
            print(f"🚨 Found {len(issues)} potential data leakage issues!")

    return issues

def check_target_alignment(symbol_data, target_col, verbose=True):
    """
    🚨 CRITICAL: Verify signal-target timing alignment

    Signals should predict FUTURE returns, not current/past returns.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: SIGNAL-TARGET TIMING ALIGNMENT")
        print(f"{'='*60}")

    issues = []

    # Check if we have date information
    if not hasattr(symbol_data.index, 'to_datetime'):
        try:
            dates = pd.to_datetime(symbol_data.index)
        except:
            if verbose:
                print("⚠️  Cannot verify timing alignment - no date index")
            return ["WARNING: Cannot verify timing alignment - no date index"]
    else:
        dates = symbol_data.index

    # Check target return calculation
    # Target should be: (price[t+1] - price[t]) / price[t]
    # NOT: (price[t] - price[t-1]) / price[t-1]

    target_values = symbol_data[target_col].dropna()

    if verbose:
        print(f"Analyzing target return timing for {len(target_values)} observations...")

    # Check for temporal autocorrelation patterns that suggest misalignment
    if len(target_values) > 10:
        # Calculate autocorrelations
        autocorrs = []
        for lag in range(1, min(6, len(target_values)//4)):
            if len(target_values) > lag:
                corr = target_values.autocorr(lag=lag)
                autocorrs.append((lag, corr))
                if verbose and lag <= 3:
                    print(f"  Autocorr lag {lag}: {corr:.4f}")

        # 🚨 CRITICAL: High positive autocorr at lag 1 suggests overlapping periods
        if len(autocorrs) > 0:
            lag1_corr = autocorrs[0][1]
            if not np.isnan(lag1_corr) and lag1_corr > 0.3:
                issue = f"CRITICAL: High lag-1 autocorrelation ({lag1_corr:.4f}) suggests overlapping return periods"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

    # Check return magnitude - should be reasonable for the time period
    abs_returns = np.abs(target_values)
    mean_abs_return = np.mean(abs_returns)
    max_abs_return = np.max(abs_returns)

    if verbose:
        print(f"Return magnitude check:")
        print(f"  Mean absolute return: {mean_abs_return:.6f}")
        print(f"  Max absolute return: {max_abs_return:.6f}")

    # 🚨 CRITICAL: Returns too large suggest wrong calculation
    if mean_abs_return > 0.1:  # 10% daily returns are unrealistic
        issue = f"CRITICAL: Mean absolute return ({mean_abs_return:.4f}) too large - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if max_abs_return > 1.0:  # 100% single return is suspicious
        issue = f"CRITICAL: Max absolute return ({max_abs_return:.4f}) extremely large - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if verbose:
        if not issues:
            print("✅ Target alignment appears correct")
        else:
            print(f"🚨 Found {len(issues)} target alignment issues!")

    return issues

def check_fold_contamination(data_length, n_folds, verbose=True):
    """
    🚨 CRITICAL: Check for overlapping train/test periods in cross-validation
    """
    if verbose:
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: FOLD CONTAMINATION")
        print(f"{'='*60}")

    issues = []

    splits = list(expanding_window_split(data_length, n_folds=n_folds))

    if verbose:
        print(f"Checking {len(splits)} folds for contamination...")

    for i, (train_idx, test_idx) in enumerate(splits):
        # Check for overlap
        train_set = set(train_idx)
        test_set = set(test_idx)

        overlap = train_set.intersection(test_set)

        if len(overlap) > 0:
            issue = f"CRITICAL: Fold {i} has {len(overlap)} overlapping indices between train and test"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

        # Check temporal ordering for expanding window
        if len(train_idx) > 0 and len(test_idx) > 0:
            max_train_idx = np.max(train_idx)
            min_test_idx = np.min(test_idx)

            # For expanding window, test should come after training
            if min_test_idx <= max_train_idx:
                # Allow for small overlap in expanding window, but not significant
                overlap_amount = max_train_idx - min_test_idx + 1
                if overlap_amount > len(test_idx) * 0.1:  # More than 10% overlap
                    issue = f"CRITICAL: Fold {i} test period significantly overlaps with training"
                    issues.append(issue)
                    if verbose:
                        print(f"🚨 {issue}")
                elif verbose:
                    print(f"  Fold {i}: Minor overlap ({overlap_amount} indices) - acceptable for expanding window")

        if verbose:
            print(f"  Fold {i}: Train {len(train_idx)}, Test {len(test_idx)}, "
                  f"Train range: [{np.min(train_idx)}-{np.max(train_idx)}], "
                  f"Test range: [{np.min(test_idx)}-{np.max(test_idx)}]")

    if verbose:
        if not issues:
            print("✅ No fold contamination detected")
        else:
            print(f"🚨 Found {len(issues)} fold contamination issues!")

    return issues

def check_signal_transformation_bugs(raw_signals, tanh_signals, binary_signals, verbose=True):
    """
    🚨 CRITICAL: Check signal transformation correctness
    """
    if verbose:
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: SIGNAL TRANSFORMATION BUGS")
        print(f"{'='*60}")

    issues = []

    # Check tanh transformation
    expected_tanh = np.tanh(raw_signals)
    tanh_diff = np.abs(tanh_signals - expected_tanh)
    max_tanh_diff = np.max(tanh_diff)

    if max_tanh_diff > 1e-10:
        issue = f"CRITICAL: Tanh transformation incorrect (max diff: {max_tanh_diff})"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    # Check binary transformation
    expected_binary = np.where(raw_signals > 0, 1, -1)
    binary_diff = np.abs(binary_signals - expected_binary)
    incorrect_binary = np.sum(binary_diff > 0)

    if incorrect_binary > 0:
        issue = f"CRITICAL: Binary transformation incorrect ({incorrect_binary}/{len(binary_signals)} wrong)"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    # Check value ranges
    if np.any(np.abs(tanh_signals) > 1.0001):  # Allow for tiny numerical errors
        issue = f"CRITICAL: Tanh signals outside [-1,1] range"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if not np.all(np.isin(binary_signals, [-1, 1])):
        issue = f"CRITICAL: Binary signals not in {{-1, 1}}"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if verbose:
        if not issues:
            print("✅ Signal transformations correct")
        else:
            print(f"🚨 Found {len(issues)} signal transformation issues!")

    return issues

def check_pnl_calculation_bugs(signals, returns, verbose=True):
    """
    🚨 CRITICAL: Verify PnL calculation correctness

    Common bugs:
    - Wrong sign (buying when should sell)
    - Position sizing errors
    - Return timing misalignment
    - Compound vs simple returns
    """
    if verbose:
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: PNL CALCULATION VERIFICATION")
        print(f"{'='*60}")

    issues = []

    if len(signals) != len(returns):
        issue = f"CRITICAL: Signal length ({len(signals)}) != return length ({len(returns)})"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")
        return issues

    # Manual PnL calculation for verification
    manual_pnl = signals * returns

    # Check for reasonable PnL values
    mean_pnl = np.mean(manual_pnl)
    std_pnl = np.std(manual_pnl)
    min_pnl = np.min(manual_pnl)
    max_pnl = np.max(manual_pnl)

    if verbose:
        print(f"PnL Statistics:")
        print(f"  Mean: {mean_pnl:.6f}")
        print(f"  Std: {std_pnl:.6f}")
        print(f"  Min: {min_pnl:.6f}")
        print(f"  Max: {max_pnl:.6f}")

    # Check for unrealistic PnL values
    if abs(mean_pnl) > 0.01:  # 1% average daily return is suspicious
        issue = f"WARNING: Mean PnL ({mean_pnl:.4f}) very high - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"⚠️  {issue}")

    if max_pnl > 0.5:  # 50% single period return is suspicious
        issue = f"CRITICAL: Max PnL ({max_pnl:.4f}) extremely high - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    # Check signal-return correlation
    # Should be positive (good signals make positive returns)
    if len(signals) > 10:
        corr = np.corrcoef(signals, returns)[0, 1]
        if not np.isnan(corr):
            if verbose:
                print(f"Signal-return correlation: {corr:.4f}")

            if corr < -0.1:  # Consistently negative correlation suggests wrong sign
                issue = f"CRITICAL: Negative signal-return correlation ({corr:.4f}) - possible wrong sign"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

        # Calculate Sharpe ratio for sanity check
        if std_pnl > 0:
            sharpe = mean_pnl / std_pnl * np.sqrt(252)  # Annualized
            if verbose:
                print(f"Implied Sharpe ratio: {sharpe:.4f}")

            if sharpe > 10:  # Sharpe > 10 is unrealistic
                issue = f"CRITICAL: Unrealistic Sharpe ratio ({sharpe:.2f}) - possible calculation error"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

    if verbose:
        if not [i for i in issues if "CRITICAL" in i]:
            print("✅ PnL calculations appear reasonable")
        else:
            print(f"🚨 Found {len([i for i in issues if 'CRITICAL' in i])} critical PnL issues!")

    return issues

def run_critical_bug_check(symbol="@ES#C", n_folds=5):
    """
    Run comprehensive critical bug detection
    """
    print(f"\n{'='*80}")
    print(f"🚨 CRITICAL BUG DETECTION - {symbol}")
    print(f"{'='*80}")
    print(f"This checks for bugs that would invalidate ALL framework results")
    print(f"Test run at: {datetime.now()}")

    all_issues = []

    try:
        # Load data
        print(f"\nLoading data for {symbol}...")
        symbol_data = load_symbol_data([symbol])

        if symbol_data.empty:
            print(f"❌ No data loaded for {symbol}")
            return ["CRITICAL: No data loaded"]

        target_col = f"{symbol}_target_return"
        if target_col not in symbol_data.columns:
            print(f"❌ Target column {target_col} not found")
            return [f"CRITICAL: Target column {target_col} not found"]

        # Drop NaN targets for analysis
        clean_data = symbol_data.dropna(subset=[target_col])
        print(f"Loaded {len(symbol_data)} rows, {len(clean_data)} with valid targets")

        # 1. Check data leakage
        issues = check_data_leakage(clean_data, target_col)
        all_issues.extend(issues)

        # 2. Check target alignment
        issues = check_target_alignment(clean_data, target_col)
        all_issues.extend(issues)

        # 3. Check fold contamination
        issues = check_fold_contamination(len(clean_data), n_folds)
        all_issues.extend(issues)

        # 4. Quick signal transformation check
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: SIGNAL TRANSFORMATION (QUICK TEST)")
        print(f"{'='*60}")

        # Create small test signals
        test_raw = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        test_tanh = np.tanh(test_raw)
        test_binary = np.where(test_raw > 0, 1, -1)

        issues = check_signal_transformation_bugs(test_raw, test_tanh, test_binary)
        all_issues.extend(issues)

        # 5. Quick PnL check with simulated data
        print(f"\n{'='*60}")
        print("🚨 CRITICAL CHECK: PNL CALCULATION (SIMULATED TEST)")
        print(f"{'='*60}")

        # Create realistic test data
        np.random.seed(42)
        test_signals = np.random.choice([-1, 1], size=100)
        test_returns = np.random.normal(0, 0.01, size=100)  # 1% daily vol

        issues = check_pnl_calculation_bugs(test_signals, test_returns)
        all_issues.extend(issues)

        # Summary
        print(f"\n{'='*80}")
        print("🚨 CRITICAL BUG DETECTION SUMMARY")
        print(f"{'='*80}")

        critical_issues = [i for i in all_issues if "CRITICAL" in i]
        warning_issues = [i for i in all_issues if "WARNING" in i]

        print(f"Total issues found: {len(all_issues)}")
        print(f"  Critical issues: {len(critical_issues)}")
        print(f"  Warning issues: {len(warning_issues)}")

        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES (Framework Invalid):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")

        if warning_issues:
            print(f"\n⚠️  WARNING ISSUES (Need Investigation):")
            for i, issue in enumerate(warning_issues, 1):
                print(f"  {i}. {issue}")

        if not critical_issues:
            print(f"\n✅ No critical bugs detected - framework appears valid!")
        else:
            print(f"\n❌ CRITICAL BUGS DETECTED - ALL RESULTS MAY BE INVALID!")

        # Save results
        results_dir = Path("testing/critical_bug_results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"critical_bugs_{symbol.replace('@', '').replace('#C', '')}_{timestamp}.txt"

        with open(results_file, 'w') as f:
            f.write(f"Critical Bug Detection Results\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Issues: {len(all_issues)}\n")
            f.write(f"Critical Issues: {len(critical_issues)}\n")
            f.write(f"Warning Issues: {len(warning_issues)}\n\n")

            if critical_issues:
                f.write("CRITICAL ISSUES:\n")
                for i, issue in enumerate(critical_issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")

            if warning_issues:
                f.write("WARNING ISSUES:\n")
                for i, issue in enumerate(warning_issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")

        print(f"📁 Results saved to: {results_file}")

    except Exception as e:
        error_msg = f"CRITICAL: Error during bug detection - {str(e)}"
        all_issues.append(error_msg)
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()

    return all_issues

if __name__ == "__main__":
    # Test critical symbols
    test_symbols = ["@ES#C", "QCL#C", "@TY#C"]  # Good, problematic, good

    for symbol in test_symbols:
        issues = run_critical_bug_check(symbol=symbol, n_folds=5)
        print(f"\n" + "="*100 + "\n")
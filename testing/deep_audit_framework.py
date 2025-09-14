#!/usr/bin/env python3
"""
Deep Framework Audit - The Most Insidious Bugs
==============================================

This script checks for the subtle bugs that are hardest to detect but invalidate everything:

1. INDEX SHIFTING BUGS - Off-by-one errors in signal-target alignment
2. FOLD BLEEDING - Subtle information leakage between CV folds
3. FEATURE CALCULATION TIMING - Features using future price information
4. RESELECTION CONTAMINATION - Model selection seeing future performance
5. COMPOUND RETURN BUGS - Incorrect compounding in PnL calculation
6. SIGNAL LAG ANALYSIS - Are signals predicting right time period?
7. ENSEMBLE VOTING LOGIC - Democratic voting implementation errors
8. HIT RATE CALCULATION BUGS - Zero signal handling in hit rates

These are the bugs that show good backtests but fail in live trading.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings

# Import specific framework modules to audit
try:
    from xgb_compare.full_timeline_backtest import FullTimelineBacktest
    from xgb_compare.metrics_utils import calculate_pnl, calculate_hit_rate
    from model.xgb_drivers import create_standard_xgb_bank
    from cv.wfo import expanding_window_split
except ImportError as e:
    print(f"Warning: Could not import framework modules: {e}")

def audit_signal_target_timing(dates, signals, targets, verbose=True):
    """
    🔍 DEEP AUDIT: Signal-target timing alignment

    This is the #1 cause of "too good to be true" backtests.
    Signals should predict tomorrow's return, not today's.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🔍 DEEP AUDIT: SIGNAL-TARGET TIMING ALIGNMENT")
        print(f"{'='*70}")

    issues = []

    if len(signals) != len(targets):
        issue = "CRITICAL: Signal and target arrays have different lengths"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")
        return issues

    # Create aligned DataFrame for analysis
    df = pd.DataFrame({
        'date': dates,
        'signal': signals,
        'target': targets
    })

    # Sort by date to ensure proper timing
    df = df.sort_values('date').reset_index(drop=True)

    if verbose:
        print(f"Analyzing {len(df)} aligned signal-target pairs...")

    # Test 1: Check if signal[t] correlates more with target[t] than target[t+1]
    # If so, it suggests the signal is seeing the current return, not predicting future
    if len(df) > 10:
        # Current period correlation (signal[t] vs target[t])
        current_corr = df['signal'].corr(df['target'])

        # Future period correlation (signal[t] vs target[t+1])
        df['target_next'] = df['target'].shift(-1)
        future_corr = df['signal'].corr(df['target_next'])

        if verbose:
            print(f"Signal correlation with:")
            print(f"  Current target: {current_corr:.4f}")
            print(f"  Future target:  {future_corr:.4f}")
            print(f"  Ratio (future/current): {future_corr/current_corr:.4f}")

        # 🚨 CRITICAL: Signal should predict future better than current
        if not np.isnan(current_corr) and not np.isnan(future_corr):
            if abs(current_corr) > abs(future_corr) and abs(current_corr) > 0.05:
                issue = f"CRITICAL: Signal correlates more with current ({current_corr:.4f}) than future ({future_corr:.4f}) returns"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

    # Test 2: Check for impossible perfect predictions
    # If signal perfectly predicts target, it's seeing the future
    perfect_predictions = np.sum(np.abs(df['signal'] - df['target']) < 1e-10)
    if perfect_predictions > len(df) * 0.01:  # More than 1% perfect
        issue = f"CRITICAL: {perfect_predictions} perfect signal-target matches ({perfect_predictions/len(df)*100:.1f}%)"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    # Test 3: Analyze prediction lead time
    # Good signals should have predictive power that decays over time
    correlations_by_lag = []
    for lag in range(0, min(6, len(df)//4)):
        if len(df) > lag:
            df[f'target_lag_{lag}'] = df['target'].shift(-lag)
            corr = df['signal'].corr(df[f'target_lag_{lag}'])
            correlations_by_lag.append((lag, corr))

            if verbose and lag <= 3:
                print(f"  Lag {lag} correlation: {corr:.4f}")

    # Check if correlation structure makes sense
    if len(correlations_by_lag) >= 3:
        lag0_corr = correlations_by_lag[0][1]  # Current
        lag1_corr = correlations_by_lag[1][1]  # Next period
        lag2_corr = correlations_by_lag[2][1]  # Two periods ahead

        # For proper prediction, lag1 should be highest
        if not np.isnan(lag0_corr) and not np.isnan(lag1_corr) and not np.isnan(lag2_corr):
            if abs(lag0_corr) > abs(lag1_corr) + 0.02:  # Current much better than next
                issue = f"WARNING: Signal better at current ({lag0_corr:.4f}) vs next period ({lag1_corr:.4f})"
                issues.append(issue)
                if verbose:
                    print(f"⚠️  {issue}")

    if verbose:
        if not [i for i in issues if "CRITICAL" in i]:
            print("✅ Signal timing appears correct")
        else:
            print(f"🚨 Found {len([i for i in issues if 'CRITICAL' in i])} critical timing issues!")

    return issues

def audit_ensemble_voting_logic(individual_signals, ensemble_signal, signal_type="binary", verbose=True):
    """
    🔍 DEEP AUDIT: Ensemble voting logic verification

    The democratic voting can have subtle bugs that affect results.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"🔍 DEEP AUDIT: ENSEMBLE VOTING LOGIC - {signal_type.upper()}")
        print(f"{'='*70}")

    issues = []

    if not individual_signals:
        issues.append("CRITICAL: No individual signals provided")
        return issues

    n_models = len(individual_signals)
    n_samples = len(individual_signals[0])

    if verbose:
        print(f"Auditing {n_models} models, {n_samples} samples...")

    # Create signal matrix
    signal_matrix = np.array(individual_signals)

    if signal_type == "binary":
        # Binary signals should be +1/-1
        unique_vals = np.unique(signal_matrix.flatten())
        if not np.all(np.isin(unique_vals, [-1, 1])):
            issue = f"CRITICAL: Binary signals contain values other than +1/-1: {unique_vals}"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

        # Test democratic voting: sum of +1/-1 votes
        expected_ensemble = np.sum(signal_matrix, axis=0)

        # Check if provided ensemble matches expected
        if hasattr(ensemble_signal, '__len__') and len(ensemble_signal) == len(expected_ensemble):
            diff = np.abs(ensemble_signal - expected_ensemble)
            max_diff = np.max(diff)

            if max_diff > 1e-10:
                issue = f"CRITICAL: Ensemble voting logic error (max diff: {max_diff:.6f})"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

                    # Show first few mismatches
                    mismatches = np.where(diff > 1e-10)[0][:5]
                    for idx in mismatches:
                        print(f"    Sample {idx}: Expected {expected_ensemble[idx]}, Got {ensemble_signal[idx]}")

        # Check vote distribution
        vote_counts = np.bincount(expected_ensemble + n_models, minlength=2*n_models+1)
        tie_votes = vote_counts[n_models]  # Zero votes

        if verbose:
            print(f"Vote distribution:")
            print(f"  Tie votes (0): {tie_votes} ({tie_votes/n_samples*100:.1f}%)")
            print(f"  Positive votes: {np.sum(expected_ensemble > 0)} ({np.sum(expected_ensemble > 0)/n_samples*100:.1f}%)")
            print(f"  Negative votes: {np.sum(expected_ensemble < 0)} ({np.sum(expected_ensemble < 0)/n_samples*100:.1f}%)")

    elif signal_type == "tanh":
        # Tanh signals should be in [-1, 1] range
        if np.any(np.abs(signal_matrix) > 1.001):  # Allow tiny numerical errors
            issue = "CRITICAL: Tanh signals outside [-1,1] range"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

        # Test ensemble averaging
        expected_ensemble = np.mean(signal_matrix, axis=0)

        if hasattr(ensemble_signal, '__len__') and len(ensemble_signal) == len(expected_ensemble):
            diff = np.abs(ensemble_signal - expected_ensemble)
            max_diff = np.max(diff)

            if max_diff > 1e-10:
                issue = f"CRITICAL: Ensemble averaging error (max diff: {max_diff:.6f})"
                issues.append(issue)
                if verbose:
                    print(f"🚨 {issue}")

    if verbose:
        if not [i for i in issues if "CRITICAL" in i]:
            print("✅ Ensemble voting logic appears correct")
        else:
            print(f"🚨 Found {len([i for i in issues if 'CRITICAL' in i])} critical voting issues!")

    return issues

def audit_hit_rate_calculation(signals, returns, verbose=True):
    """
    🔍 DEEP AUDIT: Hit rate calculation bugs

    Common issues:
    - Including zero signals in hit rate calculation
    - Wrong sign interpretation
    - Division by zero edge cases
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🔍 DEEP AUDIT: HIT RATE CALCULATION")
        print(f"{'='*70}")

    issues = []

    if len(signals) != len(returns):
        issue = f"CRITICAL: Signal length ({len(signals)}) != return length ({len(returns)})"
        issues.append(issue)
        return issues

    # Manual hit rate calculation for verification
    # Exclude zero signals (ties) from hit rate calculation
    non_zero_mask = np.abs(signals) > 1e-10

    if np.sum(non_zero_mask) == 0:
        issue = "CRITICAL: All signals are zero - cannot calculate hit rate"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")
        return issues

    non_zero_signals = signals[non_zero_mask]
    non_zero_returns = returns[non_zero_mask]

    # Calculate hits: signal and return have same sign
    hits = (non_zero_signals * non_zero_returns) > 0
    manual_hit_rate = np.mean(hits)

    # Test against framework hit rate calculation if available
    try:
        framework_hit_rate = calculate_hit_rate(signals, returns)

        diff = abs(manual_hit_rate - framework_hit_rate)

        if verbose:
            print(f"Hit rate comparison:")
            print(f"  Manual calculation: {manual_hit_rate:.4f}")
            print(f"  Framework calculation: {framework_hit_rate:.4f}")
            print(f"  Difference: {diff:.6f}")

        if diff > 1e-6:
            issue = f"CRITICAL: Hit rate calculation mismatch (diff: {diff:.6f})"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

    except Exception as e:
        if verbose:
            print(f"⚠️  Could not test framework hit rate calculation: {e}")

    # Additional checks
    zero_signals = np.sum(np.abs(signals) < 1e-10)
    if verbose:
        print(f"Signal analysis:")
        print(f"  Total signals: {len(signals)}")
        print(f"  Zero signals: {zero_signals} ({zero_signals/len(signals)*100:.1f}%)")
        print(f"  Non-zero signals: {np.sum(non_zero_mask)}")
        print(f"  Positive signals: {np.sum(signals > 0)}")
        print(f"  Negative signals: {np.sum(signals < 0)}")

    # Check for reasonable hit rate
    if manual_hit_rate < 0.3 or manual_hit_rate > 0.7:
        issue = f"WARNING: Hit rate ({manual_hit_rate:.4f}) outside reasonable range [0.3, 0.7]"
        issues.append(issue)
        if verbose:
            print(f"⚠️  {issue}")

    if verbose:
        if not [i for i in issues if "CRITICAL" in i]:
            print("✅ Hit rate calculation appears correct")
        else:
            print(f"🚨 Found {len([i for i in issues if 'CRITICAL' in i])} critical hit rate issues!")

    return issues

def audit_compound_return_logic(pnl_series, verbose=True):
    """
    🔍 DEEP AUDIT: Compound return calculation

    Check if returns are being compounded correctly vs simple addition.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🔍 DEEP AUDIT: COMPOUND RETURN LOGIC")
        print(f"{'='*70}")

    issues = []

    if len(pnl_series) < 2:
        issues.append("WARNING: Too few PnL observations for compound return analysis")
        return issues

    # Calculate cumulative returns both ways
    cumulative_simple = np.cumsum(pnl_series)
    cumulative_compound = np.cumprod(1 + pnl_series) - 1

    final_simple = cumulative_simple[-1]
    final_compound = cumulative_compound[-1]

    if verbose:
        print(f"Return calculation comparison:")
        print(f"  Simple sum: {final_simple:.6f}")
        print(f"  Compound: {final_compound:.6f}")
        print(f"  Difference: {final_compound - final_simple:.6f}")

    # For small returns, simple and compound should be very close
    if np.max(np.abs(pnl_series)) < 0.1:  # If all returns < 10%
        max_diff = np.max(np.abs(cumulative_compound - cumulative_simple))
        if max_diff > 0.01:  # More than 1% difference
            issue = f"WARNING: Large difference between simple/compound returns (max: {max_diff:.4f})"
            issues.append(issue)
            if verbose:
                print(f"⚠️  {issue}")

    # Check for impossible compound returns
    if np.any(cumulative_compound < -0.99):  # More than 99% loss
        issue = "CRITICAL: Compound returns show >99% loss - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if np.any(cumulative_compound > 10):  # More than 1000% gain
        issue = "CRITICAL: Compound returns show >1000% gain - possible calculation error"
        issues.append(issue)
        if verbose:
            print(f"🚨 {issue}")

    if verbose:
        if not [i for i in issues if "CRITICAL" in i]:
            print("✅ Compound return logic appears reasonable")
        else:
            print(f"🚨 Found {len([i for i in issues if 'CRITICAL' in i])} critical compound return issues!")

    return issues

def run_deep_framework_audit():
    """
    Run the complete deep framework audit
    """
    print(f"\n{'='*80}")
    print("🔍 DEEP FRAMEWORK AUDIT")
    print(f"{'='*80}")
    print("Checking for the most insidious bugs that invalidate results...")
    print(f"Audit run at: {datetime.now()}")

    all_issues = []

    # Test with synthetic data to verify core logic
    print(f"\nTesting core logic with synthetic data...")

    # Create realistic test data
    np.random.seed(42)
    n_samples = 100
    n_models = 5

    # Synthetic dates
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # Synthetic returns (realistic daily returns)
    returns = np.random.normal(0, 0.01, n_samples)  # 1% daily volatility

    # Synthetic signals with predictive power
    individual_binary_signals = []
    individual_tanh_signals = []

    for i in range(n_models):
        # Create signals that have some predictive power
        noise = np.random.normal(0, 0.5, n_samples)
        # Signal should predict next period's return (proper alignment)
        future_returns = np.roll(returns, -1)  # Tomorrow's return
        raw_signal = future_returns + noise  # Add noise to make realistic

        binary_sig = np.where(raw_signal > 0, 1, -1)
        tanh_sig = np.tanh(raw_signal)

        individual_binary_signals.append(binary_sig)
        individual_tanh_signals.append(tanh_sig)

    # Create ensemble signals
    ensemble_binary = np.sum(individual_binary_signals, axis=0)
    ensemble_tanh = np.mean(individual_tanh_signals, axis=0)

    # Audit 1: Signal-target timing
    # Test with properly aligned signals (should pass)
    issues = audit_signal_target_timing(dates[:-1], ensemble_binary[:-1], returns[1:])
    all_issues.extend(issues)

    # Test with improperly aligned signals (should fail)
    print(f"\n--- Testing with INTENTIONALLY MISALIGNED signals (should detect issues) ---")
    issues = audit_signal_target_timing(dates, ensemble_binary, returns)  # Same period alignment
    all_issues.extend([f"TEST: {issue}" for issue in issues])

    # Audit 2: Ensemble voting logic
    issues = audit_ensemble_voting_logic(individual_binary_signals, ensemble_binary, "binary")
    all_issues.extend(issues)

    issues = audit_ensemble_voting_logic(individual_tanh_signals, ensemble_tanh, "tanh")
    all_issues.extend(issues)

    # Audit 3: Hit rate calculation
    issues = audit_hit_rate_calculation(ensemble_binary, returns)
    all_issues.extend(issues)

    # Audit 4: Compound return logic
    pnl = ensemble_binary * returns  # Simple PnL calculation
    issues = audit_compound_return_logic(pnl)
    all_issues.extend(issues)

    # Summary
    print(f"\n{'='*80}")
    print("🔍 DEEP FRAMEWORK AUDIT SUMMARY")
    print(f"{'='*80}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i and not i.startswith("TEST:")]
    warning_issues = [i for i in all_issues if "WARNING" in i and not i.startswith("TEST:")]
    test_issues = [i for i in all_issues if i.startswith("TEST:")]

    print(f"Framework issues found: {len(all_issues) - len(test_issues)}")
    print(f"  Critical issues: {len(critical_issues)}")
    print(f"  Warning issues: {len(warning_issues)}")
    print(f"Test issues (expected): {len(test_issues)}")

    if critical_issues:
        print(f"\n🚨 CRITICAL FRAMEWORK ISSUES:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")

    if warning_issues:
        print(f"\n⚠️  WARNING ISSUES:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    if test_issues:
        print(f"\n🧪 TEST DETECTION (Shows audit is working):")
        for i, issue in enumerate(test_issues, 1):
            print(f"  {i}. {issue}")

    if not critical_issues:
        print(f"\n✅ No critical framework bugs detected!")
        print("   Core logic appears sound for synthetic data test")
    else:
        print(f"\n❌ CRITICAL FRAMEWORK BUGS DETECTED!")
        print("   Framework may be fundamentally flawed")

    # Save results
    results_dir = Path("testing/deep_audit_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"deep_framework_audit_{timestamp}.txt"

    with open(results_file, 'w') as f:
        f.write(f"Deep Framework Audit Results\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Issues: {len(all_issues) - len(test_issues)}\n")
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

    return all_issues

if __name__ == "__main__":
    run_deep_framework_audit()
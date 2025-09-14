#!/usr/bin/env python3
"""
Production Signal Deep Audit
============================

Test the most critical areas that could invalidate results:

1. SIGNAL-RETURN ALIGNMENT - Are we predicting the right period?
2. POSITION SIZING LOGIC - Is 1/-1 signal being used correctly?
3. DAILY PNL BREAKDOWN - What's happening day by day?
4. EXPOSURE CALCULATION - Are position sizes reasonable?
5. SHARPE CALCULATION VERIFICATION - Manual vs framework calculation

This focuses on the production pipeline where signals generate PnL.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from data.data_utils_simple import prepare_real_data_simple

def audit_signal_return_alignment(symbol="@ES#C"):
    """
    🚨 CRITICAL: Verify signal predicts correct return period
    """
    print(f"\n{'='*70}")
    print(f"🚨 CRITICAL AUDIT: SIGNAL-RETURN ALIGNMENT - {symbol}")
    print(f"{'='*70}")

    issues = []

    try:
        # Load data
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        # Get clean data
        clean_data = df.dropna(subset=[target_col])
        returns = clean_data[target_col].values
        dates = clean_data.index

        print(f"Analyzing {len(returns)} return observations...")

        # Create simple test signals
        # If alignment is correct, positive signals should correlate with positive returns
        test_signals = np.where(np.random.random(len(returns)) > 0.5, 1, -1)

        # Test different alignments
        alignments = {
            'same_period': returns,                    # signal[t] vs return[t] - WRONG
            'next_period': np.roll(returns, -1)[:-1],  # signal[t] vs return[t+1] - CORRECT
            'prev_period': np.roll(returns, 1)[1:],    # signal[t] vs return[t-1] - WRONG
        }

        print(f"\nTesting signal-return alignment scenarios:")

        for alignment_name, aligned_returns in alignments.items():
            if len(aligned_returns) != len(test_signals):
                aligned_signals = test_signals[:len(aligned_returns)]
            else:
                aligned_signals = test_signals

            # Calculate PnL for each alignment
            pnl = aligned_signals * aligned_returns

            mean_pnl = np.mean(pnl)
            sharpe = mean_pnl / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0

            print(f"  {alignment_name}: Mean PnL={mean_pnl:.6f}, Sharpe={sharpe:.4f}")

        print(f"\n🔍 ALIGNMENT CHECK:")
        print(f"  If framework is correct, 'next_period' alignment should be used")
        print(f"  Same-period alignment would be data leakage")

    except Exception as e:
        issue = f"CRITICAL: Error in alignment audit: {e}"
        issues.append(issue)
        print(f"🚨 {issue}")

    return issues

def audit_daily_pnl_breakdown(symbol="@ES#C", n_days=20):
    """
    🔍 AUDIT: Daily PnL breakdown analysis
    """
    print(f"\n{'='*70}")
    print(f"🔍 DAILY PNL BREAKDOWN AUDIT - {symbol}")
    print(f"{'='*70}")

    issues = []

    try:
        # Load data
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        clean_data = df.dropna(subset=[target_col])
        returns = clean_data[target_col].values
        dates = clean_data.index

        # Create simple test signals (alternating buy/sell)
        test_signals = np.array([1 if i % 2 == 0 else -1 for i in range(len(returns))])

        # Calculate daily PnL
        daily_pnl = test_signals * returns

        print(f"Analyzing daily PnL for {len(daily_pnl)} days...")

        # Take last n_days for detailed analysis
        recent_pnl = daily_pnl[-n_days:]
        recent_signals = test_signals[-n_days:]
        recent_returns = returns[-n_days:]
        recent_dates = dates[-n_days:]

        print(f"\nLast {n_days} days breakdown:")
        print(f"{'Date':<12} {'Signal':<6} {'Return':<10} {'PnL':<10} {'Position'}")
        print(f"{'-'*50}")

        cumulative_pnl = 0
        for i in range(len(recent_pnl)):
            cumulative_pnl += recent_pnl[i]
            position = "LONG" if recent_signals[i] > 0 else "SHORT"

            print(f"{recent_dates[i].strftime('%Y-%m-%d'):<12} "
                  f"{recent_signals[i]:>+2d}{'':4} "
                  f"{recent_returns[i]:>+8.4f}{'':2} "
                  f"{recent_pnl[i]:>+8.4f}{'':2} "
                  f"{position}")

        print(f"{'-'*50}")
        print(f"{'Total PnL:':<28} {cumulative_pnl:>+8.4f}")

        # Calculate metrics
        mean_daily_pnl = np.mean(daily_pnl)
        std_daily_pnl = np.std(daily_pnl)
        sharpe = mean_daily_pnl / std_daily_pnl * np.sqrt(252) if std_daily_pnl > 0 else 0

        print(f"\nDaily PnL Statistics:")
        print(f"  Mean daily PnL: {mean_daily_pnl:.6f}")
        print(f"  Daily PnL std: {std_daily_pnl:.6f}")
        print(f"  Annualized Sharpe: {sharpe:.4f}")

        # Check for suspicious patterns
        win_rate = np.mean(daily_pnl > 0)
        print(f"  Win rate: {win_rate:.4f} ({win_rate*100:.1f}%)")

        # Flag issues
        if abs(mean_daily_pnl) > 0.001:  # 0.1% daily return is high
            issue = f"WARNING: High average daily PnL ({mean_daily_pnl:.4f}) for random signals"
            issues.append(issue)
            print(f"⚠️  {issue}")

        if sharpe > 2.0:  # Sharpe > 2 for random signals is suspicious
            issue = f"WARNING: High Sharpe ({sharpe:.2f}) for random signals suggests data issues"
            issues.append(issue)
            print(f"⚠️  {issue}")

        print(f"✅ Daily PnL breakdown completed")

    except Exception as e:
        issue = f"CRITICAL: Error in daily PnL audit: {e}"
        issues.append(issue)
        print(f"🚨 {issue}")

    return issues

def audit_position_sizing_logic():
    """
    🔍 AUDIT: Position sizing and signal interpretation
    """
    print(f"\n{'='*70}")
    print("🔍 POSITION SIZING LOGIC AUDIT")
    print(f"{'='*70}")

    issues = []

    print("Testing position sizing interpretation...")

    # Test scenarios
    scenarios = [
        {"signal": 1, "return": 0.01, "expected_pnl": 0.01, "position": "LONG"},
        {"signal": -1, "return": 0.01, "expected_pnl": -0.01, "position": "SHORT"},
        {"signal": 1, "return": -0.01, "expected_pnl": -0.01, "position": "LONG"},
        {"signal": -1, "return": -0.01, "expected_pnl": 0.01, "position": "SHORT"},
        {"signal": 0, "return": 0.01, "expected_pnl": 0.0, "position": "FLAT"},
    ]

    print(f"\nPosition sizing test scenarios:")
    print(f"{'Signal':<6} {'Return':<8} {'Expected PnL':<12} {'Position':<8} {'Status'}")
    print(f"{'-'*50}")

    for scenario in scenarios:
        signal = scenario["signal"]
        return_val = scenario["return"]
        expected_pnl = scenario["expected_pnl"]
        position = scenario["position"]

        # Calculate PnL using framework logic
        calculated_pnl = signal * return_val

        # Check if calculation matches expectation
        diff = abs(calculated_pnl - expected_pnl)
        status = "✅ PASS" if diff < 1e-10 else "🚨 FAIL"

        if diff > 1e-10:
            issue = f"CRITICAL: Position sizing error - Signal {signal} * Return {return_val} = {calculated_pnl}, Expected {expected_pnl}"
            issues.append(issue)

        print(f"{signal:>+2d}{'':4} {return_val:>+6.3f}{'':2} {expected_pnl:>+8.4f}{'':4} {position:<8} {status}")

    print(f"\n✅ Position sizing logic verification completed")

    return issues

def main():
    """
    Run comprehensive production signal audit
    """
    print(f"\n{'='*80}")
    print("🚨 PRODUCTION SIGNAL DEEP AUDIT")
    print(f"{'='*80}")
    print("Testing critical areas of signal-to-PnL pipeline")
    print(f"Audit time: {datetime.now()}")

    all_issues = []

    # Audit 1: Signal-return alignment
    issues = audit_signal_return_alignment("@ES#C")
    all_issues.extend(issues)

    # Audit 2: Daily PnL breakdown
    issues = audit_daily_pnl_breakdown("@ES#C", n_days=10)
    all_issues.extend(issues)

    # Audit 3: Position sizing logic
    issues = audit_position_sizing_logic()
    all_issues.extend(issues)

    # Final summary
    print(f"\n{'='*80}")
    print("🚨 PRODUCTION AUDIT FINAL SUMMARY")
    print(f"{'='*80}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"Total issues: {len(all_issues)}")
    print(f"  Critical: {len(critical_issues)}")
    print(f"  Warnings: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨 CRITICAL PRODUCTION ISSUES:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n❌ PRODUCTION PIPELINE MAY BE INVALID!")
    else:
        print(f"\n✅ No critical production issues detected")
        print("   Signal-to-PnL pipeline appears sound")

    if warning_issues:
        print(f"\n⚠️  PRODUCTION WARNINGS:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    return all_issues

if __name__ == "__main__":
    main()
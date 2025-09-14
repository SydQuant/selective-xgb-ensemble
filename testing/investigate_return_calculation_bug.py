#!/usr/bin/env python3
"""
Critical Return Calculation Bug Investigation
============================================

QCL#C shows impossible returns (1,678% in single day).
This script investigates the root cause of this critical bug.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from data.data_utils_simple import prepare_real_data_simple

def investigate_return_bug(symbol="QCL#C"):
    """
    Investigate the extreme return calculation bug
    """
    print(f"\n{'='*80}")
    print(f"🔍 RETURN CALCULATION BUG INVESTIGATION - {symbol}")
    print(f"{'='*80}")

    try:
        # Load data
        print("1. Loading data and extracting returns...")
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"
        returns = df[target_col].dropna()

        print(f"   Data shape: {df.shape}")
        print(f"   Return observations: {len(returns)}")

        # Analyze extreme returns
        print(f"\n2. Analyzing extreme returns...")

        # Sort returns to find extremes
        sorted_returns = returns.sort_values()

        print(f"   Most extreme returns:")
        print(f"   Top 5 positive: {sorted_returns.tail(5).values}")
        print(f"   Top 5 negative: {sorted_returns.head(5).values}")
        print(f"   Mean: {returns.mean():.6f}")
        print(f"   Std: {returns.std():.6f}")

        # Find dates of extreme returns
        extreme_positive = returns[returns > 5.0]  # > 500% return
        extreme_negative = returns[returns < -5.0]  # < -500% return

        print(f"\n3. Extreme return dates:")
        print(f"   Returns > 500%: {len(extreme_positive)}")
        if len(extreme_positive) > 0:
            print("   Dates and values:")
            for date, ret in extreme_positive.head(10).items():
                print(f"     {date}: {ret:.4f} ({ret*100:.1f}%)")

        print(f"   Returns < -500%: {len(extreme_negative)}")
        if len(extreme_negative) > 0:
            print("   Dates and values:")
            for date, ret in extreme_negative.head(10).items():
                print(f"     {date}: {ret:.4f} ({ret*100:.1f}%)")

        # Check for potential causes
        print(f"\n4. Investigating potential causes...")

        # Check for near-zero prices (division by zero issue)
        # We need to reverse engineer what might cause such returns
        # Return = (price[t+1] - price[t]) / price[t]
        # If return = 16.78, then price[t+1] = price[t] * (1 + 16.78) = price[t] * 17.78

        max_return = returns.max()
        min_return = returns.min()

        print(f"   Max return: {max_return:.6f} ({max_return*100:.1f}%)")
        print(f"   Min return: {min_return:.6f} ({min_return*100:.1f}%)")

        if max_return > 5.0:
            print(f"   🚨 CRITICAL: Max return > 500% indicates price calculation bug")
            print(f"      Possible causes:")
            print(f"      1. Division by near-zero price")
            print(f"      2. Wrong price units (cents vs dollars)")
            print(f"      3. Missing price data causing huge gaps")
            print(f"      4. Corporate actions not handled")

        # Check return distribution
        print(f"\n5. Return distribution analysis...")

        return_bins = [-np.inf, -1, -0.1, -0.05, -0.01, 0.01, 0.05, 0.1, 1, np.inf]
        bin_labels = ['<-100%', '-100% to -10%', '-10% to -5%', '-5% to -1%',
                     '-1% to 1%', '1% to 5%', '5% to 10%', '10% to 100%', '>100%']

        binned = pd.cut(returns, bins=return_bins, labels=bin_labels)
        distribution = binned.value_counts().sort_index()

        print(f"   Return distribution:")
        for bin_label, count in distribution.items():
            pct = count / len(returns) * 100
            print(f"     {bin_label}: {count} ({pct:.1f}%)")

        # Check for suspicious patterns
        print(f"\n6. Checking for suspicious patterns...")

        # Check for repeated extreme values
        extreme_values = returns[abs(returns) > 1.0]
        if len(extreme_values) > 0:
            value_counts = extreme_values.value_counts()
            repeated_extremes = value_counts[value_counts > 1]

            if len(repeated_extremes) > 0:
                print(f"   🚨 Repeated extreme values (suspicious):")
                for value, count in repeated_extremes.head(5).items():
                    print(f"     {value:.4f}: {count} times")

        # Check temporal patterns
        extreme_dates = returns[abs(returns) > 1.0].index
        if len(extreme_dates) > 1:
            date_diffs = pd.Series(extreme_dates).diff().dt.days
            print(f"   Time gaps between extreme returns:")
            print(f"     Mean: {date_diffs.mean():.1f} days")
            print(f"     Min: {date_diffs.min():.1f} days")
            print(f"     Max: {date_diffs.max():.1f} days")

        # Calculate what normal oil returns should be
        print(f"\n7. Expected return magnitude check...")
        print(f"   Crude oil typically has ~2-3% daily volatility")
        print(f"   Observed daily volatility: {returns.std()*100:.1f}%")
        print(f"   Volatility ratio: {returns.std()/0.025:.1f}x normal")

        if returns.std() > 0.1:  # 10% daily vol
            print(f"   🚨 CRITICAL: Volatility is {returns.std()/0.025:.1f}x normal crude oil levels")

        # Sample some raw data around extreme dates
        if len(extreme_positive) > 0:
            print(f"\n8. Examining context around extreme positive return...")
            extreme_date = extreme_positive.index[0]
            extreme_value = extreme_positive.iloc[0]

            print(f"   Date: {extreme_date}")
            print(f"   Return: {extreme_value:.6f} ({extreme_value*100:.1f}%)")

            # Look at surrounding dates
            start_idx = max(0, returns.index.get_loc(extreme_date) - 2)
            end_idx = min(len(returns), returns.index.get_loc(extreme_date) + 3)

            context_returns = returns.iloc[start_idx:end_idx]
            print(f"   Surrounding returns:")
            for date, ret in context_returns.items():
                marker = " <-- EXTREME" if date == extreme_date else ""
                print(f"     {date}: {ret:.6f} ({ret*100:.1f}%){marker}")

    except Exception as e:
        print(f"❌ Error during investigation: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*80}")
    print(f"🔍 INVESTIGATION SUMMARY")
    print(f"{'='*80}")
    print(f"The extreme returns in {symbol} indicate a serious bug in price/return calculation.")
    print(f"This explains the poor model performance (0.087 Sharpe).")
    print(f"Models cannot learn from corrupted target data with impossible return values.")

if __name__ == "__main__":
    investigate_return_bug("QCL#C")
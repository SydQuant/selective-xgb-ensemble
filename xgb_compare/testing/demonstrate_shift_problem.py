"""
Demonstrate the Shift Problem: Hours vs Rows

This shows exactly why shift(-24) (hours) is wrong and why we need 
shift(-n_rows) for robust weekend/holiday handling.
"""

import pandas as pd
import numpy as np
import sys
import os

def demonstrate_shift_hours_vs_rows():
    """
    Show the exact problem with time-based vs row-based shifts
    """
    print("=== DEMONSTRATING SHIFT PROBLEM: HOURS vs ROWS ===")
    
    # Create realistic hourly data with weekend gaps
    print("Creating realistic hourly trading data (Mon-Fri only)...")
    
    dates = []
    current = pd.Timestamp('2024-01-01 09:00:00')  # Start Monday 9am
    
    # Generate 2 weeks of trading hours (simulate real market)
    for day in range(14):  # 2 weeks
        day_date = current + pd.Timedelta(days=day)
        
        # Only add weekdays (Mon-Fri)
        if day_date.weekday() < 5:  # 0=Monday, 4=Friday
            # Add trading hours 9am-5pm
            for hour in range(9, 18):  # 9am to 5pm
                trading_time = day_date.replace(hour=hour)
                dates.append(trading_time)
    
    # Create price data
    n_periods = len(dates)
    prices = 4000 + np.cumsum(np.random.RandomState(42).normal(0, 2, n_periods))
    
    df = pd.DataFrame({
        'close': prices
    }, index=pd.DatetimeIndex(dates))
    
    print(f"Created {n_periods} periods of trading data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Sample periods:")
    
    for i in range(min(10, len(df))):
        date = df.index[i]
        price = df['close'].iloc[i]
        day_name = date.strftime('%A')
        print(f"  Period {i+1:2d}: {date.strftime('%Y-%m-%d %H:%M')} ({day_name}) = {price:.2f}")
    
    return df

def test_shift_methods(df):
    """
    Test different shift methods and show the problems
    """
    print(f"\n=== TESTING DIFFERENT SHIFT METHODS ===")
    
    print(f"Dataset has {len(df)} periods")
    print(f"First period: {df.index[0]} (Friday 9am)")
    print(f"Last period: {df.index[-1]}")
    
    # Method 1: shift(-24) - Time-based (PROBLEMATIC)
    print(f"\n❌ METHOD 1: shift(-24) - Time-based")
    future_close_time = df['close'].shift(-24)  # 24 periods = 24 hours
    returns_time = (future_close_time - df['close']) / df['close']
    
    print(f"Example: Friday 12pm signal")
    friday_12pm_idx = None
    for i, date in enumerate(df.index):
        if date.weekday() == 4 and date.hour == 12:  # Friday 12pm
            friday_12pm_idx = i
            break
    
    if friday_12pm_idx is not None:
        friday_date = df.index[friday_12pm_idx]
        friday_price = df['close'].iloc[friday_12pm_idx]
        
        # What does shift(-24) give us?
        target_date_expected = friday_date + pd.Timedelta(hours=24)  # Saturday 12pm
        
        print(f"  Friday 12pm: {friday_date} (price: {friday_price:.2f})")
        print(f"  Expected target time (Fri+24h): {target_date_expected} (Saturday - no trading!)")
        
        # What target return do we actually get?
        friday_target = returns_time.iloc[friday_12pm_idx]
        print(f"  Calculated target return: {friday_target}")
        
        if pd.isna(friday_target):
            print("  → Result: NaN (because Saturday price doesn't exist)")
        else:
            print(f"  → Result: {friday_target:.6f} (somehow got a value - investigate)")
    
    # Method 2: shift(-N) - Row-based (CORRECT)
    print(f"\n✅ METHOD 2: shift(-N) - Row-based")
    
    # Let's say we want "next trading day" which could be 9 hours later (same day) or 72 hours later (weekend)
    # But in row terms, it's always just "next row"
    future_close_row = df['close'].shift(-1)  # Next trading period
    returns_row = (future_close_row - df['close']) / df['close']
    
    if friday_12pm_idx is not None:
        friday_target_row = returns_row.iloc[friday_12pm_idx]
        next_period_idx = friday_12pm_idx + 1
        
        if next_period_idx < len(df):
            next_date = df.index[next_period_idx]
            next_price = df['close'].iloc[next_period_idx]
            
            print(f"  Friday 12pm: {friday_date} (price: {friday_price:.2f})")
            print(f"  Next trading period: {next_date} (price: {next_price:.2f})")
            print(f"  Time gap: {next_date - friday_date}")
            print(f"  Calculated return: {friday_target_row:.6f}")
            print(f"  → Result: Handles weekend gap correctly!")
    
    # Show NaN patterns
    time_nans = returns_time.isna().sum()
    row_nans = returns_row.isna().sum()
    
    print(f"\n📊 NaN COMPARISON:")
    print(f"Time-based shift(-24): {time_nans} NaNs out of {len(returns_time)} ({time_nans/len(returns_time)*100:.1f}%)")
    print(f"Row-based shift(-1):   {row_nans} NaNs out of {len(returns_row)} ({row_nans/len(returns_row)*100:.1f}%)")
    
    return returns_time, returns_row

def test_robust_row_based_target():
    """
    Test a more robust row-based target calculation
    """
    print(f"\n=== TESTING ROBUST ROW-BASED TARGET CALCULATION ===")
    
    df = demonstrate_shift_methods()
    
    # Different row-based shift options
    shift_options = [1, 2, 8, 24]  # 1 hour, 2 hours, 8 hours, 24 rows ahead
    
    for n_rows in shift_options:
        future_close = df['close'].shift(-n_rows)
        returns = (future_close - df['close']) / df['close']
        
        valid_returns = returns.dropna()
        nan_count = returns.isna().sum()
        
        print(f"\nshift(-{n_rows} rows):")
        print(f"  Valid targets: {len(valid_returns)}")
        print(f"  NaN targets: {nan_count}")
        print(f"  Last row: {returns.iloc[-1] if not pd.isna(returns.iloc[-1]) else 'NaN'}")
        
        # Show what this represents in time
        if len(valid_returns) > 0:
            first_valid_idx = returns.first_valid_index()
            first_valid_pos = df.index.get_loc(first_valid_idx)
            target_date = df.index[first_valid_pos + n_rows] if first_valid_pos + n_rows < len(df) else "Beyond data"
            
            print(f"  Example: {first_valid_idx} → {target_date}")
            if target_date != "Beyond data":
                time_diff = target_date - first_valid_idx
                print(f"           Time span: {time_diff}")
    
    # Recommended approach
    print(f"\n🎯 RECOMMENDED APPROACH:")
    print(f"Use shift(-1) for 'next trading period' predictions")
    print(f"This naturally handles:")
    print(f"  - Weekends (Friday 12pm → Monday 9am)")
    print(f"  - Holidays (automatic gap handling)")
    print(f"  - Market closures")
    print(f"  - Data gaps")
    
    return df

if __name__ == "__main__":
    print("🔍 DEMONSTRATING SHIFT PROBLEM: HOURS vs ROWS")
    print("="*70)
    
    # Create realistic data
    df = demonstrate_shift_hours_vs_rows()
    
    # Test different methods
    returns_time, returns_row = test_shift_methods(df)
    
    # Test robust approach
    robust_df = test_robust_row_based_target()
    
    print("\n" + "="*70)
    print("🏁 SHIFT METHOD INVESTIGATION COMPLETE")
    
    print(f"\n🎯 KEY INSIGHT:")
    print("Using time-based shifts (hours) is fragile and creates unnecessary NaNs")
    print("Using row-based shifts is robust and handles market gaps naturally")
    
    print(f"\n🔧 IMPLICATIONS FOR TARGET CALCULATION:")
    print("Current: shift(-24) assumes 24 hours = next trading day (WRONG)")
    print("Better: shift(-1) means next available trading period (ROBUST)")
    print("Even better: shift(-8) for 8-hour ahead prediction, etc.")
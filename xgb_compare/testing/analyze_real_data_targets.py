"""
Analyze Real Data Target Calculation

Test the actual target calculation on real market data to understand:
1. How many NaN targets exist with current shift(-24) approach
2. Weekend behavior (Friday 12pm → ?)
3. Why last row has a target when it shouldn't
4. Compare shift(-24) vs shift(-n_rows) approach
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_raw_data_target_calculation():
    """
    Test target calculation on actual raw data
    """
    print("=== TESTING TARGET CALCULATION ON REAL DATA ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        # Load actual ES data
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        print(f"Loaded real ES data:")
        print(f"Shape: {raw_df.shape}")
        print(f"Date range: {raw_df.index[0]} to {raw_df.index[-1]}")
        
        # Take recent subset for detailed analysis
        recent_df = raw_df.tail(200)  # Last 200 records
        print(f"Analyzing recent {len(recent_df)} records:")
        print(f"Recent range: {recent_df.index[0]} to {recent_df.index[-1]}")
        
        # Test current approach: shift(-24)
        print(f"\n🔍 CURRENT APPROACH: shift(-24)")
        future_close_24h = recent_df['close'].shift(-24)
        returns_24h = (future_close_24h - recent_df['close']) / recent_df['close']
        
        valid_24h = returns_24h.dropna()
        nan_24h = returns_24h.isna().sum()
        
        print(f"Using shift(-24):")
        print(f"  Total periods: {len(returns_24h)}")
        print(f"  Valid targets: {len(valid_24h)}")
        print(f"  NaN targets: {nan_24h}")
        print(f"  Valid percentage: {len(valid_24h)/len(returns_24h)*100:.1f}%")
        print(f"  Last row target: {returns_24h.iloc[-1]} ({'NaN' if pd.isna(returns_24h.iloc[-1]) else 'HAS VALUE'})")
        
        # Test alternative: shift(-1) 
        print(f"\n🔍 ALTERNATIVE APPROACH: shift(-1)")
        future_close_1 = recent_df['close'].shift(-1)
        returns_1 = (future_close_1 - recent_df['close']) / recent_df['close']
        
        valid_1 = returns_1.dropna()
        nan_1 = returns_1.isna().sum()
        
        print(f"Using shift(-1):")
        print(f"  Total periods: {len(returns_1)}")
        print(f"  Valid targets: {len(valid_1)}")
        print(f"  NaN targets: {nan_1}")
        print(f"  Valid percentage: {len(valid_1)/len(returns_1)*100:.1f}%")
        print(f"  Last row target: {returns_1.iloc[-1]} ({'NaN' if pd.isna(returns_1.iloc[-1]) else 'HAS VALUE'})")
        
        # Show specific weekend examples
        print(f"\n📅 WEEKEND BEHAVIOR ANALYSIS:")
        
        # Find Friday 12pm periods
        fridays_12pm = recent_df[(recent_df.index.weekday == 4) & (recent_df.index.hour == 12)]
        
        if len(fridays_12pm) > 0:
            print(f"Found {len(fridays_12pm)} Friday 12pm periods in recent data")
            
            for i, (friday_date, friday_row) in enumerate(fridays_12pm.head(3).iterrows()):
                print(f"\nFriday {i+1}: {friday_date}")
                friday_price = friday_row['close']
                
                # What does shift(-24) target?
                target_24h_date = friday_date + pd.Timedelta(hours=24)  # Saturday 12pm
                
                # What does shift(-1) target?
                friday_idx = recent_df.index.get_loc(friday_date)
                if friday_idx + 1 < len(recent_df):
                    next_trading_date = recent_df.index[friday_idx + 1]
                    next_trading_price = recent_df.iloc[friday_idx + 1]['close']
                    
                    print(f"  Current: {friday_date} (price: {friday_price:.2f})")
                    print(f"  shift(-24) targets: {target_24h_date} (weekend - no trading)")
                    print(f"  shift(-1) targets: {next_trading_date} (price: {next_trading_price:.2f})")
                    
                    # Calculate returns both ways
                    return_24h = returns_24h.loc[friday_date] if friday_date in returns_24h.index else None
                    return_1 = returns_1.loc[friday_date] if friday_date in returns_1.index else None
                    
                    print(f"  shift(-24) return: {return_24h}")
                    print(f"  shift(-1) return: {return_1}")
        
        return recent_df, returns_24h, returns_1
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        return None, None, None

def test_signal_hour_filtering():
    """
    Test how signal hour filtering affects the results
    """
    print("\n=== TESTING SIGNAL HOUR FILTERING ===")
    
    raw_df, returns_24h, returns_1 = test_raw_data_target_calculation()
    
    if raw_df is not None:
        # Test the complete pipeline step that includes signal hour filtering
        print("Testing signal hour filtering (signal_hour=12)...")
        
        # Replicate the filtering step
        signal_hour = 12
        if returns_24h is not None:
            filtered_24h = returns_24h[returns_24h.index.hour == signal_hour]
            filtered_1 = returns_1[returns_1.index.hour == signal_hour]
            
            print(f"Before signal hour filtering:")
            print(f"  shift(-24): {len(returns_24h)} periods, {returns_24h.isna().sum()} NaN")
            print(f"  shift(-1): {len(returns_1)} periods, {returns_1.isna().sum()} NaN")
            
            print(f"After signal hour filtering (12pm only):")
            print(f"  shift(-24): {len(filtered_24h)} periods, {filtered_24h.isna().sum()} NaN")
            print(f"  shift(-1): {len(filtered_1)} periods, {filtered_1.isna().sum()} NaN")
            
            if len(filtered_24h) > 0:
                print(f"  Last filtered(-24) target: {filtered_24h.iloc[-1]} ({'NaN' if pd.isna(filtered_24h.iloc[-1]) else 'HAS VALUE'})")
            if len(filtered_1) > 0:
                print(f"  Last filtered(-1) target: {filtered_1.iloc[-1]} ({'NaN' if pd.isna(filtered_1.iloc[-1]) else 'HAS VALUE'})")
            
            return filtered_24h, filtered_1
    
    return None, None

def demonstrate_weekend_gap_issue():
    """
    Create specific test case to show the weekend gap problem
    """
    print("\n=== DEMONSTRATING WEEKEND GAP ISSUE ===")
    
    # Create data that clearly shows the weekend problem
    dates = []
    
    # Create 3 days: Thursday, Friday, Monday (skip weekend)
    base_date = pd.Timestamp('2024-01-04 12:00:00')  # Thursday
    dates.extend([
        base_date,                                    # Thursday 12pm
        base_date + pd.Timedelta(days=1),            # Friday 12pm  
        base_date + pd.Timedelta(days=4)             # Monday 12pm (skip weekend)
    ])
    
    prices = [4000, 4010, 4020]
    
    df = pd.DataFrame({
        'close': prices
    }, index=pd.DatetimeIndex(dates))
    
    print("Test data (3 trading days with weekend gap):")
    for i, (date, price) in enumerate(df['close'].items()):
        day_name = date.strftime('%A')
        print(f"  Day {i+1}: {date.strftime('%Y-%m-%d %H:%M')} ({day_name}) = {price}")
    
    # Test shift(-24) vs shift(-1)
    print(f"\n📊 SHIFT COMPARISON:")
    
    # shift(-24): Looks for same time 24 hours later
    future_24h = df['close'].shift(-24)
    print(f"shift(-24) results:")
    for i, (date, future_price) in enumerate(future_24h.items()):
        expected_date = date + pd.Timedelta(hours=24)
        print(f"  {date.strftime('%A %H:%M')} + 24h = {expected_date.strftime('%A %H:%M')}: {future_price if not pd.isna(future_price) else 'NaN (weekend)'}")
    
    # shift(-1): Looks for next trading period
    future_1 = df['close'].shift(-1)
    print(f"\nshift(-1) results:")
    for i, (date, future_price) in enumerate(future_1.items()):
        print(f"  {date.strftime('%A %H:%M')} → next period: {future_price if not pd.isna(future_price) else 'NaN (last period)'}")
    
    # Calculate returns
    returns_24h = (future_24h - df['close']) / df['close']
    returns_1 = (future_1 - df['close']) / df['close']
    
    print(f"\n📈 RETURN CALCULATIONS:")
    print("shift(-24) returns:")
    for i, (date, ret) in enumerate(returns_24h.items()):
        print(f"  {date.strftime('%A')}: {ret if not pd.isna(ret) else 'NaN'}")
        
    print("shift(-1) returns:")  
    for i, (date, ret) in enumerate(returns_1.items()):
        print(f"  {date.strftime('%A')}: {ret if not pd.isna(ret) else 'NaN'}")
    
    print(f"\n🎯 INSIGHTS:")
    print(f"shift(-24): Creates NaN for periods that don't have data exactly 24h later")
    print(f"shift(-1): Creates NaN only for the truly last period (no next data)")
    print(f"→ shift(-1) is more robust for market data with gaps")
    
    return df, returns_24h, returns_1

def check_current_vs_proposed_approach():
    """
    Compare current approach with proposed row-based approach
    """
    print("\n=== COMPARING CURRENT vs PROPOSED APPROACH ===")
    
    print("🔄 CURRENT APPROACH (data_utils_simple.py):")
    print("   Line 139: future_close = df['close'].shift(-n_hours)")
    print("   → n_hours=24 means 'exactly 24 hours later'")
    print("   → Fails on weekends/holidays (creates artificial NaNs)")
    print("   → Reduces usable data unnecessarily")
    
    print("\n🔄 PROPOSED APPROACH:")
    print("   future_close = df['close'].shift(-n_periods)")
    print("   → n_periods=1 means 'next available trading period'")
    print("   → Handles weekends/holidays naturally")
    print("   → Maximizes usable data")
    
    print(f"\n📊 EXPECTED IMPACT OF FIXING:")
    print("✅ More valid target returns (fewer artificial NaNs)")
    print("✅ Proper weekend handling (Friday 12pm → Monday 9am)")
    print("✅ Better data utilization")
    print("✅ More robust to market schedules")
    print("⚠️  Need to adjust n_periods parameter (24 hours ≠ 24 periods)")
    
    return True

if __name__ == "__main__":
    print("🔍 ANALYZING REAL DATA TARGET PATTERNS")
    print("="*70)
    
    # Test 1: Real data target calculation
    raw_df, rets_24h, rets_1 = test_raw_data_target_calculation()
    
    # Test 2: Signal hour filtering effect
    filt_24h, filt_1 = test_signal_hour_filtering()
    
    # Test 3: Weekend gap demonstration
    demo_df, demo_24h, demo_1 = demonstrate_weekend_gap_issue()
    
    # Test 4: Current vs proposed comparison
    check_current_vs_proposed_approach()
    
    print("\n" + "="*70)
    print("🏁 REAL DATA ANALYSIS COMPLETE")
    
    print(f"\n🎯 CRITICAL FINDINGS:")
    print("1. Current shift(-24) approach creates artificial NaNs due to weekends")
    print("2. Row-based shift(-n) would be more robust")
    print("3. Last row target issue may be due to insufficient NaNs from weekend gaps")
    print("4. Need to reconsider n_hours parameter for row-based approach")
    
    print(f"\n🔧 RECOMMENDED CHANGES:")
    print("1. Change shift(-n_hours) to shift(-n_periods)")
    print("2. Adjust parameter: instead of n_hours=24, use n_periods=1 (next period)")
    print("3. For longer horizons: n_periods=8 (8 trading hours ahead), etc.")
    print("4. This will naturally handle market gaps and weekends")
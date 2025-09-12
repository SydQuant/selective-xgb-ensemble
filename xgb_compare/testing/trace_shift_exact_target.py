"""
Trace Exact Target of shift(-24) and Fix Intraday Fill

Based on CME trading hours:
- ES trades Sunday 5pm - Friday 4pm CT (almost 24/7)
- Only gap: Friday 4pm - Sunday 5pm (weekend shutdown)
- Current data should have this gap, but intraday fill might be wrong

This script:
1. Finds exactly which row shift(-24) hits
2. Identifies intraday fill problems
3. Proposes correct weekend gap handling
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def trace_exact_shift_target():
    """
    Find exactly which row shift(-24) is hitting when we expect Friday→Monday
    """
    print("=== TRACING EXACT shift(-24) TARGET ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Get recent data for analysis
        recent_df = raw_df.tail(1000)
        
        # Find Friday 12pm periods
        fridays_12pm = recent_df[(recent_df.index.weekday == 4) & (recent_df.index.hour == 12)]
        
        print(f"Analyzing {len(fridays_12pm)} Friday 12pm periods...")
        
        for i, (friday_date, friday_row) in enumerate(fridays_12pm.tail(2).iterrows()):
            print(f"\n🔍 FRIDAY TEST {i+1}: {friday_date}")
            friday_price = friday_row['close']
            
            # Find Friday's position in the dataset
            friday_idx = recent_df.index.get_loc(friday_date)
            
            # What does shift(-24) actually target?
            if friday_idx + 24 < len(recent_df):
                target_idx = friday_idx + 24
                target_date = recent_df.index[target_idx]
                target_price = recent_df['close'].iloc[target_idx]
                
                print(f"  Friday 12pm: {friday_date} (idx {friday_idx}, price {friday_price:.2f})")
                print(f"  shift(-24) hits: {target_date} (idx {target_idx}, price {target_price:.2f})")
                print(f"  Target day: {target_date.strftime('%A')}")
                print(f"  Time difference: {target_date - friday_date}")
                
                # Is this the intended Monday 12pm?
                intended_monday = friday_date + pd.Timedelta(days=3)  # Friday + 3 days = Monday
                intended_monday = intended_monday.replace(hour=12)
                
                print(f"  Intended Monday 12pm: {intended_monday}")
                
                if target_date == intended_monday:
                    print("  ✅ PERFECT: shift(-24) correctly hits Monday 12pm")
                else:
                    time_diff = abs((target_date - intended_monday).total_seconds())
                    print(f"  ❌ MISMATCH: {time_diff/3600:.1f} hours off from intended Monday")
                
                # Find the actual Monday 12pm for comparison
                monday_12pm_candidates = recent_df[
                    (recent_df.index > friday_date) &
                    (recent_df.index.weekday == 0) &
                    (recent_df.index.hour == 12)
                ]
                
                if len(monday_12pm_candidates) > 0:
                    actual_monday = monday_12pm_candidates.index[0]
                    actual_monday_price = monday_12pm_candidates.iloc[0]['close']
                    
                    print(f"  Actual Monday 12pm: {actual_monday} (price {actual_monday_price:.2f})")
                    
                    if target_date == actual_monday:
                        print("  ✅ shift(-24) correctly targets Monday 12pm")
                    else:
                        print(f"  ❌ shift(-24) targets wrong period!")
                        print(f"     Should target: {actual_monday}")
                        print(f"     Actually targets: {target_date}")
                        
                        # Calculate what the correct return should be
                        correct_return = (actual_monday_price - friday_price) / friday_price
                        actual_return = (target_price - friday_price) / friday_price
                        
                        print(f"     Correct return (Fri→Mon): {correct_return:.6f}")
                        print(f"     Actual return (shift-24): {actual_return:.6f}")
                        print(f"     Error: {abs(correct_return - actual_return):.6f}")
        
        return recent_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def examine_weekend_data_pattern():
    """
    Examine the exact data pattern around weekends
    """
    print("\n=== EXAMINING WEEKEND DATA PATTERN ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Find a recent Friday-Monday transition
        recent_df = raw_df.tail(2000)
        
        # Look for Friday 4pm (market close) and Sunday 5pm (market open) pattern
        print("Looking for CME trading schedule pattern...")
        print("Expected: Friday 4pm close → Sunday 5pm open")
        
        # Find Friday periods
        friday_periods = recent_df[recent_df.index.weekday == 4]  # Friday
        sunday_periods = recent_df[recent_df.index.weekday == 6]  # Sunday
        
        if len(friday_periods) > 0:
            print(f"\nFriday data pattern:")
            last_friday_hours = sorted(set(friday_periods.index.hour))
            print(f"  Available Friday hours: {last_friday_hours}")
            
            # Check if we have Friday 4pm (market close)
            friday_4pm = friday_periods[friday_periods.index.hour == 16]  # 4pm = 16
            if len(friday_4pm) > 0:
                print(f"  Friday 4pm periods found: {len(friday_4pm)}")
                latest_friday_4pm = friday_4pm.index[-1]
                print(f"  Latest Friday 4pm: {latest_friday_4pm}")
                
                # Look for next data after Friday 4pm
                after_friday = recent_df[recent_df.index > latest_friday_4pm]
                if len(after_friday) > 0:
                    next_data = after_friday.index[0]
                    gap_duration = next_data - latest_friday_4pm
                    print(f"  Next data after Friday 4pm: {next_data}")
                    print(f"  Gap duration: {gap_duration}")
                    
                    # Is it Sunday 5pm as expected?
                    if next_data.weekday() == 6 and next_data.hour == 17:  # Sunday 5pm
                        print(f"  ✅ CORRECT: Weekend gap Friday 4pm → Sunday 5pm")
                    else:
                        print(f"  ⚠️  UNEXPECTED: Next data is {next_data.strftime('%A %H:%M')}")
        
        if len(sunday_periods) > 0:
            print(f"\nSunday data pattern:")
            sunday_hours = sorted(set(sunday_periods.index.hour))
            print(f"  Available Sunday hours: {sunday_hours}")
        
        # Check Saturday data (should be minimal/none according to CME schedule)
        saturday_periods = recent_df[recent_df.index.weekday == 5]  # Saturday
        print(f"\nSaturday data: {len(saturday_periods)} periods")
        if len(saturday_periods) > 0:
            saturday_hours = sorted(set(saturday_periods.index.hour))
            print(f"  Saturday hours: {saturday_hours}")
            print(f"  ⚠️  Unexpected Saturday data - should be market closed")
        
        return recent_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_current_intraday_fill():
    """
    Test if current intraday fill is working correctly for the weekend gap
    """
    print("\n=== TESTING CURRENT INTRADAY FILL ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Check if we have proper weekend gaps
        recent_df = raw_df.tail(500)
        
        # Look at time intervals to see gap patterns
        time_diffs = recent_df.index[1:] - recent_df.index[:-1]
        
        # Find large gaps (more than 2 hours = weekend or holiday)
        large_gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
        
        print(f"Data interval analysis:")
        print(f"Total periods: {len(recent_df)}")
        print(f"Large gaps (>2h): {len(large_gaps)}")
        
        if len(large_gaps) > 0:
            print(f"Gap details:")
            for i, gap in enumerate(large_gaps.tail(3)):
                gap_start_idx = recent_df.index.get_loc(recent_df.index[time_diffs.get_loc(gap)])
                gap_start = recent_df.index[gap_start_idx]
                gap_end = recent_df.index[gap_start_idx + 1]
                
                start_day = gap_start.strftime('%A %H:%M')
                end_day = gap_end.strftime('%A %H:%M')
                
                print(f"  Gap {i+1}: {start_day} → {end_day} ({gap})")
                
                # Is this a proper weekend gap?
                if gap_start.weekday() == 4 and gap_end.weekday() == 6:  # Fri→Sun
                    print(f"    ✅ Proper weekend gap (Friday → Sunday)")
                elif gap_start.weekday() == 4 and gap_end.weekday() == 0:  # Fri→Mon
                    print(f"    ⚠️  Extended weekend gap (Friday → Monday)")
                else:
                    print(f"    🔍 Other gap type")
        
        # Test shift(-24) around these gaps
        print(f"\n🔍 shift(-24) BEHAVIOR AROUND GAPS:")
        
        # Find a Friday 12pm before a weekend gap
        fridays_12pm = recent_df[(recent_df.index.weekday == 4) & (recent_df.index.hour == 12)]
        
        if len(fridays_12pm) > 0:
            test_friday = fridays_12pm.index[-1]
            friday_idx = recent_df.index.get_loc(test_friday)
            
            print(f"Test Friday 12pm: {test_friday} (index {friday_idx})")
            
            # Show the next 30 periods to see what shift(-24) would hit
            start_idx = friday_idx
            end_idx = min(friday_idx + 30, len(recent_df))
            
            print(f"Next 30 periods after Friday 12pm:")
            for j in range(start_idx, end_idx):
                date = recent_df.index[j]
                price = recent_df['close'].iloc[j]
                offset = j - friday_idx
                marker = " ← shift(-24)" if offset == 24 else ""
                day_name = date.strftime('%A %H:%M')
                
                print(f"  +{offset:2d}: {day_name} = {price:.2f}{marker}")
                
                if offset == 24:
                    # This is what shift(-24) targets
                    expected_monday_12pm = test_friday + pd.Timedelta(days=3)
                    expected_monday_12pm = expected_monday_12pm.replace(hour=12)
                    
                    print(f"       Expected Monday 12pm: {expected_monday_12pm}")
                    if date == expected_monday_12pm:
                        print(f"       ✅ PERFECT MATCH!")
                    else:
                        print(f"       ❌ MISMATCH: {(date - expected_monday_12pm).total_seconds()/3600:.1f}h off")
        
        return recent_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def propose_corrected_intraday_fill():
    """
    Propose how intraday fill should work for proper Friday→Monday mapping
    """
    print("\n=== PROPOSING CORRECTED INTRADAY FILL ===")
    
    print("🏛️ CME ES TRADING SCHEDULE:")
    print("  Sunday 5pm CT → Friday 4pm CT")
    print("  Weekend gap: Friday 4pm → Sunday 5pm (~ 49 hours)")
    print("  Daily break: 4pm → 5pm CT (1 hour)")
    
    print(f"\n🔧 CORRECT INTRADAY FILL LOGIC:")
    print("1. Fill small gaps (≤2 hours) - daily breaks, minor outages")
    print("2. DON'T fill weekend gap (Friday 4pm → Sunday 5pm)")
    print("3. Result: Friday 12pm + 24 periods = Monday 12pm")
    
    print(f"\n📊 EXPECTED BEHAVIOR:")
    print("With proper weekend gaps:")
    print("  Friday 12pm (idx 100) → +24 periods → Monday 12pm (idx 124)")
    print("  Friday 1pm (idx 101) → +24 periods → Monday 1pm (idx 125)")
    print("  This ensures shift(-24) maps to same time next trading day")
    
    print(f"\n❌ CURRENT PROBLEM:")
    print("If weekends are filled with synthetic data:")
    print("  Friday 12pm (idx 100) → +24 periods → Saturday 12pm (idx 124)")
    print("  This breaks the Friday→Monday mapping")
    
    print(f"\n✅ PROPOSED FIX:")
    print("1. Modify intraday_fill to preserve weekend gaps")
    print("2. Or use more robust period-based logic:")
    print("   def get_next_trading_day_same_hour(date):")
    print("     # Find next weekday at same hour")
    print("     # Handles weekends, holidays naturally")
    
    return True

def create_test_with_proper_gaps():
    """
    Create test data with proper weekend gaps to verify the fix
    """
    print("\n=== CREATING TEST WITH PROPER WEEKEND GAPS ===")
    
    # Create realistic trading schedule
    dates = []
    current = pd.Timestamp('2024-01-01 09:00:00')  # Monday 9am
    
    for day in range(14):  # 2 weeks
        day_date = current.date() + pd.Timedelta(days=day)
        day_timestamp = pd.Timestamp(day_date)
        
        if day_timestamp.weekday() < 5:  # Monday-Friday
            # Add trading hours (simplified: 9am-5pm)
            for hour in range(9, 18):
                trading_time = day_timestamp + pd.Timedelta(hours=hour)
                dates.append(trading_time)
        # Skip weekends (no Saturday/Sunday data)
    
    # Create price data  
    n_periods = len(dates)
    prices = 4000 + np.cumsum(np.random.RandomState(42).normal(0, 2, n_periods))
    
    df = pd.DataFrame({
        'close': prices
    }, index=pd.DatetimeIndex(dates))
    
    print(f"Created proper weekend gap data:")
    print(f"Periods: {n_periods}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Test shift(-24) on this clean data
    future_close = df['close'].shift(-24)
    returns = (future_close - df['close']) / df['close']
    
    # Find Friday 12pm and see what it maps to
    fridays_12pm = df[(df.index.weekday == 4) & (df.index.hour == 12)]
    
    if len(fridays_12pm) > 0:
        test_friday = fridays_12pm.index[0]
        friday_idx = df.index.get_loc(test_friday)
        
        print(f"\nTest Friday 12pm: {test_friday} (idx {friday_idx})")
        
        if friday_idx + 24 < len(df):
            target_date = df.index[friday_idx + 24]
            target_day = target_date.strftime('%A %H:%M')
            
            print(f"shift(-24) targets: {target_date} ({target_day})")
            
            if target_date.weekday() == 0 and target_date.hour == 12:
                print(f"✅ PERFECT: Friday 12pm → Monday 12pm mapping!")
            else:
                print(f"❌ WRONG: Should be Monday 12pm")
        
        # Check return calculation
        friday_return = returns.loc[test_friday]
        print(f"Friday 12pm target return: {friday_return}")
        
        if pd.isna(friday_return):
            print("Return is NaN - check data or calculation")
        else:
            print(f"Return calculated successfully: {friday_return:.6f}")
    
    return df

if __name__ == "__main__":
    print("🔍 TRACING EXACT shift(-24) TARGET AND INTRADAY FILL")
    print("="*70)
    
    # Step 1: Trace exact target
    traced_data = trace_exact_shift_target()
    
    # Step 2: Examine weekend patterns
    weekend_data = examine_weekend_data_pattern()
    
    # Step 3: Propose fix
    propose_corrected_intraday_fill()
    
    # Step 4: Test with proper gaps
    test_data = create_test_with_proper_gaps()
    
    print("\n" + "="*70)
    print("🏁 SHIFT TARGET ANALYSIS COMPLETE")
    
    print(f"\n🎯 DIAGNOSIS:")
    print("1. Current data may have improper weekend fills")
    print("2. shift(-24) might be hitting filled weekend data instead of Monday")
    print("3. Need to fix intraday_fill to preserve weekend gaps")
    print("4. Proper gaps ensure Friday 12pm + 24 periods = Monday 12pm")
    
    print(f"\n🔧 ACTION ITEMS:")
    print("1. Check intraday_fill function implementation")
    print("2. Ensure weekend gaps (Fri 4pm → Sun 5pm) are preserved")
    print("3. Verify shift(-24) maps Friday 12pm → Monday 12pm")
    print("4. Test with corrected data to confirm last row behavior")
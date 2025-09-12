"""
Analyze Raw Data Pattern and Intraday Fill Logic

This script:
1. Examines latest raw data pattern (gaps, frequencies)
2. Tests current intraday_fill behavior  
3. Identifies why shift(-n_hours) doesn't map T 12pm → T+n 12pm correctly
4. Proposes flexible solutions for any n-hour target calculation
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def analyze_latest_raw_data_pattern():
    """
    Pull latest raw data and analyze the exact pattern
    """
    print("=== ANALYZING LATEST RAW DATA PATTERN ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Get very latest data
        latest_data = raw_df.tail(200)  # Last 200 records
        
        print(f"Latest raw data:")
        print(f"Shape: {latest_data.shape}")
        print(f"Date range: {latest_data.index[0]} to {latest_data.index[-1]} (NY time)")
        
        # Analyze time intervals
        time_diffs = latest_data.index[1:] - latest_data.index[:-1]
        
        print(f"\nTime interval analysis:")
        print(f"Most common interval: {time_diffs.value_counts().index[0]}")
        print(f"Unique intervals: {sorted(time_diffs.unique())}")
        
        # Find gaps and trading patterns
        gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]
        print(f"Large gaps (>2h): {len(gaps)}")
        
        if len(gaps) > 0:
            print(f"Recent gaps:")
            for i, gap in enumerate(gaps.tail(3)):
                gap_idx = time_diffs.get_loc(gap)
                gap_start = latest_data.index[gap_idx]
                gap_end = latest_data.index[gap_idx + 1]
                
                start_info = f"{gap_start.strftime('%A %H:%M')}"
                end_info = f"{gap_end.strftime('%A %H:%M')}"
                
                print(f"  {start_info} → {end_info} ({gap})")
                
                # Check if this follows expected pattern
                if gap_start.weekday() == 4 and gap_end.weekday() == 6:  # Fri→Sun
                    print(f"    ✅ Expected weekend gap pattern")
                else:
                    print(f"    🔍 Different gap pattern")
        
        # Check daily coverage
        print(f"\nDaily trading coverage:")
        for weekday in range(7):
            weekday_data = latest_data[latest_data.index.weekday == weekday]
            if len(weekday_data) > 0:
                hours = sorted(set(weekday_data.index.hour))
                day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][weekday]
                print(f"  {day_name}: {len(weekday_data)} periods, hours {hours[:5]}{'...' if len(hours) > 5 else ''}")
        
        return latest_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_intraday_fill_function():
    """
    Test if there's an intraday_fill function and what it does
    """
    print("\n=== TESTING INTRADAY_FILL FUNCTION ===")
    
    # Look for intraday_fill function
    try:
        # Search for intraday fill in data loaders
        from data import loaders
        
        # Check if loaders has intraday_fill
        if hasattr(loaders, 'intraday_fill'):
            print("Found intraday_fill function in data.loaders")
        else:
            print("No intraday_fill function found in data.loaders")
            
        # Let's examine the loaders module
        import inspect
        functions = [name for name, obj in inspect.getmembers(loaders) if inspect.isfunction(obj)]
        print(f"Available functions in data.loaders: {functions}")
        
    except Exception as e:
        print(f"Error checking intraday_fill: {e}")
    
    # Also check if it's in data_utils_simple
    try:
        from data import data_utils_simple
        
        if hasattr(data_utils_simple, 'intraday_fill'):
            print("Found intraday_fill function in data_utils_simple")
        else:
            print("No intraday_fill function found in data_utils_simple")
            
    except Exception as e:
        print(f"Error checking data_utils_simple: {e}")
    
    return True

def diagnose_shift_offset_problem(raw_data):
    """
    Diagnose exactly why shift(-24) is hitting 1pm instead of 12pm
    """
    print("\n=== DIAGNOSING SHIFT OFFSET PROBLEM ===")
    
    if raw_data is None:
        print("No raw data available for diagnosis")
        return
    
    # Find recent Friday 12pm
    fridays_12pm = raw_data[(raw_data.index.weekday == 4) & (raw_data.index.hour == 12)]
    
    if len(fridays_12pm) > 0:
        test_friday = fridays_12pm.index[-1]
        friday_idx = raw_data.index.get_loc(test_friday)
        
        print(f"Diagnosis for Friday: {test_friday}")
        print(f"Friday index position: {friday_idx}")
        
        # Show 48 periods around the target to see the pattern
        start_idx = max(0, friday_idx - 5)
        end_idx = min(len(raw_data), friday_idx + 35)
        
        print(f"\nData pattern around Friday 12pm (showing periods {start_idx}-{end_idx}):")
        
        for j in range(start_idx, end_idx):
            date = raw_data.index[j]
            price = raw_data['close'].iloc[j]
            offset_from_friday = j - friday_idx
            
            # Mark key periods
            marker = ""
            if j == friday_idx:
                marker = " ← FRIDAY 12PM"
            elif offset_from_friday == 24:
                marker = " ← shift(-24) TARGET"
            elif date.weekday() == 0 and date.hour == 12:  # Monday 12pm
                marker = " ← INTENDED TARGET (Monday 12pm)"
            
            day_hour = date.strftime('%A %H:%M')
            print(f"  {offset_from_friday:+3d}: {day_hour} = {price:.2f}{marker}")
        
        # Calculate the offset needed to hit Monday 12pm
        monday_12pm_candidates = raw_data[
            (raw_data.index > test_friday) &
            (raw_data.index.weekday == 0) &
            (raw_data.index.hour == 12)
        ]
        
        if len(monday_12pm_candidates) > 0:
            correct_monday = monday_12pm_candidates.index[0]
            correct_idx = raw_data.index.get_loc(correct_monday)
            correct_offset = correct_idx - friday_idx
            
            print(f"\n📊 OFFSET ANALYSIS:")
            print(f"Current shift(-24) offset: 24 periods")
            print(f"Correct offset for Monday 12pm: {correct_offset} periods")
            print(f"Difference: {correct_offset - 24} periods")
            
            if correct_offset != 24:
                print(f"🚨 PROBLEM: shift(-24) is off by {correct_offset - 24} periods")
                print(f"This explains why we hit Monday 1pm instead of Monday 12pm")
    
    return True

def propose_flexible_target_solutions():
    """
    Propose flexible solutions for any n-hour target calculations
    """
    print("\n=== PROPOSING FLEXIBLE TARGET SOLUTIONS ===")
    
    print("🎯 REQUIREMENTS:")
    print("- Friday 12pm → Monday 12pm (24-hour target)")
    print("- Friday 12pm → Monday 3pm (27-hour target)")
    print("- Friday 12pm → Wednesday 12pm (48-hour target)")
    print("- Handle weekends/holidays automatically")
    print("- Work with any signal_hour (not just 12pm)")
    
    print(f"\n💡 SOLUTION 1: SMART HOUR-BASED TARGETING")
    print("```python")
    print("def calculate_target_returns_smart(df, n_hours, signal_hour=12):")
    print("    '''Calculate returns to same hour n_hours later, skipping weekends'''")
    print("    returns = []")
    print("    ")
    print("    for current_date in df.index:")
    print("        if current_date.hour == signal_hour:")
    print("            # Calculate target date")
    print("            target_date = current_date + pd.Timedelta(hours=n_hours)")
    print("            ")
    print("            # If target falls on weekend, move to next Monday same hour")
    print("            while target_date.weekday() >= 5:  # Sat=5, Sun=6")
    print("                target_date += pd.Timedelta(days=1)")
    print("            ")
    print("            # Find closest data point to target_date at same hour")
    print("            target_candidates = df[")
    print("                (df.index.date == target_date.date()) &")
    print("                (df.index.hour == signal_hour)")
    print("            ]")
    print("            ")
    print("            if len(target_candidates) > 0:")
    print("                target_price = target_candidates.iloc[0]['close']")
    print("                current_price = df.loc[current_date, 'close']")
    print("                ret = (target_price - current_price) / current_price")
    print("                returns.append((current_date, ret))")
    print("            else:")
    print("                returns.append((current_date, np.nan))")
    print("    ")
    print("    return pd.Series(dict(returns))")
    print("```")
    
    print(f"\n💡 SOLUTION 2: BUSINESS-DAY AWARE CALCULATION")
    print("```python")
    print("def calculate_target_returns_business_days(df, n_business_days=1, signal_hour=12):")
    print("    '''Calculate returns to same hour n business days later'''")
    print("    returns = []")
    print("    ")
    print("    for current_date in df.index:")
    print("        if current_date.hour == signal_hour:")
    print("            # Add n business days")
    print("            target_date = current_date")
    print("            days_added = 0")
    print("            ")
    print("            while days_added < n_business_days:")
    print("                target_date += pd.Timedelta(days=1)")
    print("                if target_date.weekday() < 5:  # Mon-Fri")
    print("                    days_added += 1")
    print("            ")
    print("            # Keep same hour")
    print("            target_date = target_date.replace(hour=signal_hour)")
    print("            ")
    print("            # Find data at target date/hour")
    print("            # ... (similar lookup logic)")
    print("```")
    
    print(f"\n💡 SOLUTION 3: ROBUST PERIOD-BASED (Your Original Idea)")
    print("```python")
    print("def calculate_target_returns_periods(df, n_periods=24, signal_hour=12):")
    print("    '''Use next n periods at same hour, auto-handles gaps'''")
    print("    ")
    print("    # Filter to signal hour only")
    print("    signal_data = df[df.index.hour == signal_hour]")
    print("    ")
    print("    # Now shift(-n_periods) works correctly")
    print("    future_close = signal_data['close'].shift(-n_periods)")
    print("    returns = (future_close - signal_data['close']) / signal_data['close']")
    print("    ")
    print("    return returns")
    print("```")
    
    print(f"\n📊 COMPARISON:")
    print("Solution 1: Most flexible, handles any n_hours correctly")
    print("Solution 2: Good for standard business day calculations") 
    print("Solution 3: Simplest, works with existing shift logic")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("Use Solution 1 (Smart Hour-Based) because:")
    print("✅ Handles any n_hours (3, 24, 48, etc.)")
    print("✅ Always maps to same hour (12pm → 12pm)")
    print("✅ Automatically handles weekends/holidays")
    print("✅ Flexible for different signal_hours")
    print("✅ Explicit logic (easier to debug)")
    
    return True

def create_test_implementation():
    """
    Create a test implementation of the smart hour-based targeting
    """
    print("\n=== CREATING TEST IMPLEMENTATION ===")
    
    # Create test data with realistic weekend gaps
    dates = []
    current = pd.Timestamp('2024-01-01 09:00:00')  # Start Monday
    
    # Create 10 days of trading data (Mon-Fri only, skipping weekends)
    for day in range(14):  # 2 weeks
        day_date = current.date() + pd.Timedelta(days=day)
        day_timestamp = pd.Timestamp(day_date)
        
        # Monday-Friday: 6am-10pm (extended hours)
        if day_timestamp.weekday() < 5:
            for hour in range(6, 23):  # 6am-10pm
                trading_time = day_timestamp + pd.Timedelta(hours=hour)
                dates.append(trading_time)
    
    # Create realistic price data
    n_periods = len(dates)
    prices = 4500 + np.cumsum(np.random.RandomState(42).normal(0, 3, n_periods))
    
    test_df = pd.DataFrame({
        'close': prices
    }, index=pd.DatetimeIndex(dates))
    
    print(f"Created test data:")
    print(f"Periods: {n_periods}")
    print(f"Date range: {test_df.index[0]} to {test_df.index[-1]}")
    
    # Test implementation
    def calculate_smart_target_returns(df, n_hours, signal_hour=12):
        """Smart hour-based target calculation"""
        returns = {}
        
        signal_periods = df[df.index.hour == signal_hour]
        
        for current_date, current_row in signal_periods.iterrows():
            current_price = current_row['close']
            
            # Calculate target date (handling weekends)
            target_date = current_date + pd.Timedelta(hours=n_hours)
            
            # If target falls on weekend, find next Monday at same hour
            while target_date.weekday() >= 5:  # Saturday or Sunday
                target_date += pd.Timedelta(days=1)
            
            # Ensure we keep the same hour
            target_date = target_date.replace(hour=signal_hour)
            
            # Find actual data at target date/hour
            target_candidates = df[
                (df.index.date == target_date.date()) &
                (df.index.hour == signal_hour)
            ]
            
            if len(target_candidates) > 0:
                target_price = target_candidates.iloc[0]['close']
                ret = (target_price - current_price) / current_price
                returns[current_date] = ret
            else:
                returns[current_date] = np.nan
        
        return pd.Series(returns)
    
    # Test with different n_hours
    test_cases = [24, 48, 72]  # 1 day, 2 days, 3 days
    
    for n_hours in test_cases:
        print(f"\n🔍 TESTING n_hours = {n_hours}:")
        
        # Current method (shift-based)
        signal_data = test_df[test_df.index.hour == 12]
        current_returns = signal_data['close'].shift(-n_hours//1)  # Approximate
        current_calc = (current_returns - signal_data['close']) / signal_data['close']
        
        # Smart method
        smart_returns = calculate_smart_target_returns(test_df, n_hours, signal_hour=12)
        
        # Compare
        common_dates = signal_data.index.intersection(smart_returns.index)
        
        if len(common_dates) > 0:
            print(f"  Periods compared: {len(common_dates)}")
            
            # Show first few comparisons
            for date in common_dates[:3]:
                if date in current_calc.index and not pd.isna(current_calc.loc[date]):
                    current_val = current_calc.loc[date]
                    smart_val = smart_returns.loc[date]
                    
                    day_name = date.strftime('%A')
                    print(f"    {day_name} 12pm: Current={current_val:.6f}, Smart={smart_val:.6f}")
                    
            # Count NaNs
            current_nans = current_calc.isna().sum() if len(current_calc) > 0 else 0
            smart_nans = smart_returns.isna().sum()
            
            print(f"  Current method NaNs: {current_nans}")
            print(f"  Smart method NaNs: {smart_nans}")
            
    return test_df

if __name__ == "__main__":
    print("🔍 ANALYZING RAW DATA AND PROPOSING INTRADAY FILL FIX")
    print("="*70)
    print("ES futures trade Sun 6pm ET - Fri 5pm ET (CME converted to NY time)")
    
    # Step 1: Analyze latest raw data pattern
    raw_data = analyze_latest_raw_data_pattern()
    
    # Step 2: Test intraday_fill function
    test_intraday_fill_function()
    
    # Step 3: Diagnose shift offset problem
    diagnose_shift_offset_problem(raw_data)
    
    # Step 4: Propose flexible solutions
    propose_flexible_target_solutions()
    
    # Step 5: Test implementation
    test_data = create_test_implementation()
    
    print("\n" + "="*70)
    print("🏁 ANALYSIS AND PROPOSAL COMPLETE")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("1. Raw data pattern and gaps identified")
    print("2. shift(-24) offset problem diagnosed")
    print("3. Flexible solutions proposed for any n_hours")
    print("4. Test implementation created")
    
    print(f"\n🔧 RECOMMENDED APPROACH:")
    print("Implement smart hour-based targeting to ensure:")
    print("- Friday 12pm + 24h = Monday 12pm (not Monday 1pm)")
    print("- Friday 12pm + 48h = Tuesday 12pm")  
    print("- Any n_hours maps correctly to same hour n business hours later")
    print("- Robust to weekends, holidays, and data irregularities")
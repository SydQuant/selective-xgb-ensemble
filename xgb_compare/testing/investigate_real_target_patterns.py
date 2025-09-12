"""
Investigate Real Target Return Patterns

This script examines actual market data to understand:
1. How many NaN targets actually exist when building target df
2. Weekend/holiday handling (Friday 12pm → Monday 12pm)
3. Intraday fill behavior
4. What percentage of targets are valid vs NaN
5. Last row behavior in real production data
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def load_raw_market_data():
    """
    Load raw market data to examine the actual data structure
    """
    print("=== LOADING RAW MARKET DATA ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        # Get connection and load data
        futures_lib = get_arcticdb_connection()
        
        # Load ES data for analysis
        print("Loading @ES#C data...")
        versioned_item = futures_lib.read("@ES#C")
        df = versioned_item.data
        
        print(f"Raw data loaded:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Show data frequency
        if len(df) > 1:
            time_diffs = df.index[1:5] - df.index[0:4]
            print(f"Time intervals (first 4): {time_diffs}")
        
        # Focus on recent period for analysis
        recent_cutoff = df.index[-1000]  # Last 1000 records
        df_recent = df[df.index >= recent_cutoff]
        
        print(f"\nFocusing on recent data:")
        print(f"Recent period: {df_recent.index[0]} to {df_recent.index[-1]}")
        print(f"Records: {len(df_recent)}")
        
        return df_recent
        
    except Exception as e:
        print(f"Error loading real data: {e}")
        print("Creating synthetic hourly data for analysis...")
        
        # Create synthetic hourly data that mimics real trading
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        # Create hourly data for 30 days, excluding weekends
        dates = []
        current = start_date
        
        for _ in range(30 * 24):  # 30 days * 24 hours
            # Skip weekends (Saturday=5, Sunday=6) 
            if current.weekday() < 5:  # Monday=0 to Friday=4
                dates.append(current)
            current += timedelta(hours=1)
        
        dates = pd.DatetimeIndex(dates[:500])  # Take first 500 for analysis
        prices = 4000 + np.cumsum(np.random.normal(0, 2, len(dates)))
        
        df_synthetic = pd.DataFrame({
            'open': prices,
            'high': prices + np.abs(np.random.normal(0, 1, len(dates))),
            'low': prices - np.abs(np.random.normal(0, 1, len(dates))),
            'close': prices + np.random.normal(0, 0.5, len(dates)),
            'volume': np.random.randint(1000, 5000, len(dates))
        }, index=dates)
        
        print(f"Synthetic data created:")
        print(f"Shape: {df_synthetic.shape}")
        print(f"Date range: {df_synthetic.index[0]} to {df_synthetic.index[-1]}")
        
        return df_synthetic

def analyze_target_calculation_patterns(raw_data):
    """
    Analyze how targets are calculated and what patterns exist
    """
    print("\n=== ANALYZING TARGET CALCULATION PATTERNS ===")
    
    from data.data_utils_simple import prepare_target_returns
    
    # Test different n_hours values to see the impact
    n_hours_options = [1, 24, 48]
    signal_hour = 12
    
    raw_data_dict = {"@ES#C": raw_data}
    
    for n_hours in n_hours_options:
        print(f"\n📊 Testing n_hours = {n_hours}:")
        
        target_returns = prepare_target_returns(raw_data_dict, "@ES#C", n_hours=n_hours, signal_hour=signal_hour)
        
        if len(target_returns) == 0:
            print(f"   No data returned (likely no signal_hour={signal_hour} data)")
            continue
        
        # Analyze the results
        total_periods = len(target_returns)
        nan_count = target_returns.isna().sum()
        valid_count = total_periods - nan_count
        
        print(f"   Total periods: {total_periods}")
        print(f"   Valid targets: {valid_count}")
        print(f"   NaN targets: {nan_count}")
        print(f"   Valid percentage: {valid_count/total_periods*100:.1f}%")
        
        # Show first few and last few
        print(f"   First 5 targets: {target_returns.head().values}")
        print(f"   Last 5 targets: {target_returns.tail().values}")
        
        # Check last row specifically
        last_target = target_returns.iloc[-1] if len(target_returns) > 0 else None
        print(f"   Last target: {last_target} ({'NaN' if pd.isna(last_target) else 'HAS VALUE'})")
        
        # Examine specific periods around weekends if we have enough data
        if len(target_returns) > 50:
            print(f"   Weekend transition analysis:")
            friday_periods = target_returns[target_returns.index.weekday == 4]  # Friday
            monday_periods = target_returns[target_returns.index.weekday == 0]  # Monday
            
            if len(friday_periods) > 0 and len(monday_periods) > 0:
                friday_nan_pct = friday_periods.isna().mean() * 100
                monday_nan_pct = monday_periods.isna().mean() * 100
                print(f"     Friday periods NaN: {friday_nan_pct:.1f}%")
                print(f"     Monday periods NaN: {monday_nan_pct:.1f}%")
    
    return target_returns

def examine_weekend_behavior(raw_data):
    """
    Specifically examine Friday→Monday behavior
    """
    print("\n=== EXAMINING WEEKEND BEHAVIOR ===")
    
    # Look at raw data patterns around weekends
    print("Raw data weekend patterns:")
    
    # Find Fridays and Mondays at 12pm
    fridays_12pm = raw_data[(raw_data.index.weekday == 4) & (raw_data.index.hour == 12)]
    mondays_12pm = raw_data[(raw_data.index.weekday == 0) & (raw_data.index.hour == 12)]
    
    print(f"Fridays at 12pm: {len(fridays_12pm)} periods")
    print(f"Mondays at 12pm: {len(mondays_12pm)} periods")
    
    if len(fridays_12pm) > 0 and len(mondays_12pm) > 0:
        print(f"First few Friday 12pm dates: {fridays_12pm.index[:3].tolist()}")
        print(f"First few Monday 12pm dates: {mondays_12pm.index[:3].tolist()}")
        
        # Test the shift calculation manually
        print(f"\n🔍 MANUAL SHIFT CALCULATION TEST:")
        
        # Take first few Fridays and see what shift(-24) gives us
        test_fridays = fridays_12pm.head(3)
        
        for friday_date, friday_row in test_fridays.iterrows():
            print(f"\nFriday {friday_date.strftime('%Y-%m-%d %H:%M')}:")
            friday_close = friday_row['close']
            
            # What should shift(-24) give us?
            expected_monday = friday_date + timedelta(hours=24)
            print(f"  Expected Monday (Friday + 24h): {expected_monday.strftime('%Y-%m-%d %H:%M')} ({'Weekend' if expected_monday.weekday() >= 5 else 'Weekday'})")
            
            # What does the data actually have?
            try:
                monday_data = raw_data.loc[expected_monday]
                monday_close = monday_data['close']
                manual_return = (monday_close - friday_close) / friday_close
                print(f"  Found data at expected time: {monday_close:.2f}")
                print(f"  Manual return calculation: {manual_return:.6f}")
            except KeyError:
                print(f"  No data at expected Monday time (weekend gap)")
                
                # Find next available Monday
                next_monday_candidates = raw_data[raw_data.index > friday_date]
                if len(next_monday_candidates) > 0:
                    next_available = next_monday_candidates.index[0]
                    print(f"  Next available data: {next_available.strftime('%Y-%m-%d %H:%M')}")
                    
                    if next_available.weekday() == 0:  # Is Monday
                        next_close = next_monday_candidates.iloc[0]['close']
                        weekend_return = (next_close - friday_close) / friday_close
                        print(f"  Weekend return (Fri→Mon): {weekend_return:.6f}")
    
    return fridays_12pm, mondays_12pm

def test_full_pipeline_with_real_data():
    """
    Test the full pipeline with real data to see actual behavior
    """
    print("\n=== TESTING FULL PIPELINE WITH REAL DATA ===")
    
    try:
        from data.data_utils_simple import prepare_real_data_simple
        
        # Test with a longer date range to get more data
        print("Testing prepare_real_data_simple with extended date range...")
        
        df_result = prepare_real_data_simple(
            "@ES#C", 
            start_date="2024-01-01", 
            end_date="2024-02-01",  # 1 month of data
            n_hours=24,
            signal_hour=12
        )
        
        target_col = "@ES#C_target_return"
        
        print(f"Pipeline result:")
        print(f"Shape: {df_result.shape}")
        print(f"Date range: {df_result.index[0]} to {df_result.index[-1]}")
        
        if target_col in df_result.columns:
            target_series = df_result[target_col]
            total_targets = len(target_series)
            nan_targets = target_series.isna().sum()
            valid_targets = total_targets - nan_targets
            
            print(f"Target analysis:")
            print(f"  Total periods: {total_targets}")
            print(f"  Valid targets: {valid_targets}")  
            print(f"  NaN targets: {nan_targets}")
            print(f"  Valid percentage: {valid_targets/total_targets*100:.1f}%")
            
            # Critical check: last row
            print(f"\n🔍 LAST ROW CHECK:")
            last_target = target_series.iloc[-1]
            print(f"Last target value: {last_target}")
            print(f"Last target date: {df_result.index[-1]}")
            print(f"Last target is NaN: {pd.isna(last_target)}")
            
            if pd.isna(last_target):
                print("✅ GOOD: Last row has NaN target (production-ready)")
            else:
                print("❌ ISSUE: Last row has target value (future leakage)")
                
            # Show weekend patterns
            print(f"\n📅 WEEKEND PATTERN ANALYSIS:")
            weekday_counts = {}
            weekday_nan_counts = {}
            
            for weekday in range(7):  # 0=Monday, 6=Sunday
                weekday_mask = df_result.index.weekday == weekday
                weekday_data = target_series[weekday_mask]
                
                if len(weekday_data) > 0:
                    weekday_counts[weekday] = len(weekday_data)
                    weekday_nan_counts[weekday] = weekday_data.isna().sum()
            
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            for weekday, count in weekday_counts.items():
                nan_count = weekday_nan_counts[weekday]
                valid_count = count - nan_count
                nan_pct = nan_count / count * 100 if count > 0 else 0
                
                print(f"  {weekday_names[weekday]:9s}: {count:3d} total, {valid_count:3d} valid, {nan_count:3d} NaN ({nan_pct:4.1f}%)")
        
        return df_result
        
    except Exception as e:
        print(f"Error testing full pipeline: {e}")
        return None

def analyze_data_gaps_and_coverage():
    """
    Analyze data gaps and coverage patterns
    """
    print("\n=== ANALYZING DATA GAPS AND COVERAGE ===")
    
    raw_data = load_raw_market_data()
    
    # Look for gaps in the data
    if len(raw_data) > 1:
        time_diffs = raw_data.index[1:] - raw_data.index[:-1]
        
        print(f"Time difference analysis:")
        print(f"Most common interval: {time_diffs.mode().iloc[0] if len(time_diffs.mode()) > 0 else 'N/A'}")
        print(f"Min interval: {time_diffs.min()}")
        print(f"Max interval: {time_diffs.max()}")
        
        # Find large gaps (more than 2 hours)
        large_gaps = time_diffs[time_diffs > timedelta(hours=2)]
        
        print(f"\nLarge gaps (>2 hours): {len(large_gaps)}")
        if len(large_gaps) > 0:
            for i, gap in enumerate(large_gaps.head(5)):
                gap_start = raw_data.index[time_diffs.get_loc(gap)]
                gap_end = gap_start + gap
                print(f"  Gap {i+1}: {gap_start} → {gap_end} ({gap})")
        
        # Weekend gaps specifically  
        weekend_gaps = []
        for i, (start_time, end_time) in enumerate(zip(raw_data.index[:-1], raw_data.index[1:])):
            if start_time.weekday() == 4 and end_time.weekday() == 0:  # Friday → Monday
                weekend_gaps.append((start_time, end_time, end_time - start_time))
        
        print(f"\nWeekend gaps (Friday → Monday): {len(weekend_gaps)}")
        for i, (start, end, duration) in enumerate(weekend_gaps[:3]):
            print(f"  Weekend {i+1}: {start} → {end} ({duration})")
    
    return raw_data

if __name__ == "__main__":
    print("🔍 INVESTIGATING REAL TARGET RETURN PATTERNS")
    print("="*70)
    
    # Step 1: Load and examine raw data
    raw_data = analyze_data_gaps_and_coverage()
    
    # Step 2: Test target calculation with different parameters
    target_returns = analyze_target_calculation_patterns(raw_data)
    
    # Step 3: Examine weekend-specific behavior
    fridays, mondays = examine_weekend_behavior(raw_data)
    
    # Step 4: Test full pipeline
    pipeline_result = test_full_pipeline_with_real_data()
    
    print("\n" + "="*70)
    print("🏁 REAL DATA INVESTIGATION COMPLETE")
    
    print(f"\n🎯 KEY FINDINGS:")
    print("1. Actual data frequency and gaps")
    print("2. Target return calculation success rate")  
    print("3. Weekend handling (Friday 12pm → Monday 12pm)")
    print("4. Last row behavior in production data")
    print("5. NaN vs valid target percentages")
    
    print(f"\n💡 IMPLICATIONS:")
    print("This reveals the true nature of the last row target issue")
    print("and shows how weekend gaps affect target calculation.")
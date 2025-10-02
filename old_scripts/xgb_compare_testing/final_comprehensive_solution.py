"""
Final Comprehensive Solution for Target Return Calculation

Based on analysis:
1. ES data is hourly with weekend gaps
2. shift(-24) hits Monday 1pm instead of Monday 12pm (1-hour offset)
3. Need flexible solution for any n_hours (3, 24, 48, etc.)
4. Must handle weekends robustly

This provides the final recommended solution.
"""

import pandas as pd
import numpy as np

def analyze_current_raw_data_precisely():
    """
    Get precise analysis of current raw data pattern
    """
    print("=== PRECISE RAW DATA ANALYSIS ===")
    
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Get very recent data
        recent = raw_df.tail(100)
        
        print(f"Recent 100 records:")
        print(f"Date range: {recent.index[0]} to {recent.index[-1]}")
        
        # Find the pattern around Friday-Monday
        friday_data = recent[recent.index.weekday == 4]  # Friday
        monday_data = recent[recent.index.weekday == 0]  # Monday
        
        print(f"\nFriday hours available: {sorted(set(friday_data.index.hour)) if len(friday_data) > 0 else 'None'}")
        print(f"Monday hours available: {sorted(set(monday_data.index.hour)) if len(monday_data) > 0 else 'None'}")
        
        # Check specific Friday-Monday transitions
        if len(friday_data) > 0:
            latest_friday = friday_data.index[-1]
            friday_12pm = friday_data[friday_data.index.hour == 12]
            
            if len(friday_12pm) > 0:
                test_friday_12pm = friday_12pm.index[-1]
                print(f"\nAnalyzing Friday 12pm: {test_friday_12pm}")
                
                # Find next Monday 12pm
                next_monday_12pm = monday_data[
                    (monday_data.index > test_friday_12pm) &
                    (monday_data.index.hour == 12)
                ]
                
                if len(next_monday_12pm) > 0:
                    monday_12pm_date = next_monday_12pm.index[0]
                    print(f"Next Monday 12pm: {monday_12pm_date}")
                    print(f"Time span: {monday_12pm_date - test_friday_12pm}")
                    
                    # Calculate period difference
                    friday_idx = recent.index.get_loc(test_friday_12pm)
                    monday_idx = recent.index.get_loc(monday_12pm_date)
                    period_diff = monday_idx - friday_idx
                    
                    print(f"Period difference: {period_diff} periods")
                    print(f"Expected for 24h: 24 periods")
                    print(f"Actual offset: {period_diff - 24} periods")
                    
                    if period_diff == 24:
                        print("✅ PERFECT: 24 periods = Friday 12pm → Monday 12pm")
                    else:
                        print(f"❌ OFFSET: Need shift(-{period_diff}) to hit Monday 12pm")
        
        return recent
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def propose_optimal_solution():
    """
    Propose the optimal solution based on your requirements
    """
    print("\n=== OPTIMAL SOLUTION PROPOSAL ===")
    
    print("🎯 YOUR REQUIREMENTS:")
    print("- Friday 12pm → Monday 12pm (regardless of weekend gap)")
    print("- Flexible for 3h, 24h, 48h, etc.")
    print("- Handle trading hours: Sun 6pm ET - Fri 5pm ET")
    print("- Preserve weekend gap (Fri 5pm - Sun 6pm)")
    
    print(f"\n💡 RECOMMENDED SOLUTION: Business-Hour-Aware Targeting")
    
    solution_code = '''
def calculate_target_returns_business_hours(df, n_business_hours, signal_hour=12):
    """
    Calculate target returns using business hours, skipping weekends.
    
    Args:
        df: Raw hourly data
        n_business_hours: Number of business hours ahead (3, 24, 48, etc.)
        signal_hour: Hour for signal generation (12 for 12pm)
    
    Returns:
        Series with target returns: Friday 12pm → Monday 12pm for n_business_hours=24
    """
    import pandas as pd
    import numpy as np
    
    returns = {}
    
    # Get all signal hour periods
    signal_periods = df[df.index.hour == signal_hour]
    
    for current_date, current_row in signal_periods.iterrows():
        current_price = current_row['close']
        
        # Find target date by adding business hours
        target_date = current_date
        hours_added = 0
        
        while hours_added < n_business_hours:
            target_date += pd.Timedelta(hours=1)
            
            # Skip weekend gap (Fri 5pm ET - Sun 6pm ET)
            if target_date.weekday() == 5:  # Saturday
                # Jump to Sunday 6pm ET
                target_date = target_date.replace(weekday=6, hour=18)  # Sun 6pm
            elif target_date.weekday() == 6 and target_date.hour < 18:  # Sunday before 6pm
                target_date = target_date.replace(hour=18)  # Jump to Sun 6pm
            else:
                hours_added += 1
        
        # Find data at target date
        target_candidates = df[
            (abs(df.index - target_date) < pd.Timedelta(hours=2)) &  # Within 2h window
            (df.index.hour == signal_hour)  # Same hour
        ]
        
        if len(target_candidates) > 0:
            # Take closest match
            closest_idx = (target_candidates.index - target_date).abs().argmin()
            target_price = target_candidates.iloc[closest_idx]['close']
            ret = (target_price - current_price) / current_price
            returns[current_date] = ret
        else:
            returns[current_date] = np.nan
    
    return pd.Series(returns)
'''
    
    print(solution_code)
    
    print(f"\n🎯 ALTERNATIVE: SIMPLER PERIOD-BASED SOLUTION")
    
    simple_solution = '''
def calculate_target_returns_simple(df, n_periods=1, signal_hour=12):
    """
    Simple solution: work on signal_hour data only, then shift by periods.
    This automatically handles weekends since weekend data is excluded.
    """
    
    # Step 1: Filter to signal hour only (removes weekend complexity)
    signal_data = df[df.index.hour == signal_hour]
    
    # Step 2: shift(-n_periods) on daily signal data
    future_close = signal_data['close'].shift(-n_periods)
    returns = (future_close - signal_data['close']) / signal_data['close']
    
    return returns

# Usage examples:
# n_periods=1: Next trading day same hour
# n_periods=2: 2 trading days ahead same hour  
# n_periods=5: 1 week ahead same hour
'''
    
    print(simple_solution)
    
    print(f"\n📊 SOLUTION COMPARISON:")
    print("Business-Hour-Aware:")
    print("  ✅ Handles exact n_hours (24h, 48h)")
    print("  ✅ Weekend logic explicit") 
    print("  ❌ More complex implementation")
    
    print("Simple Period-Based:")
    print("  ✅ Very simple implementation")
    print("  ✅ Automatically handles weekends")
    print("  ❌ Uses trading days, not exact hours")
    
    print(f"\n🎯 RECOMMENDATION:")
    print("Start with Simple Period-Based approach:")
    print("- Change n_hours=24 to n_periods=1 (next trading day)")
    print("- Change n_hours=48 to n_periods=2 (2 trading days ahead)")
    print("- If you need exact hour control, upgrade to Business-Hour-Aware")
    
    return True

def create_implementation_template():
    """
    Create implementation template for the fix
    """
    print("\n=== IMPLEMENTATION TEMPLATE ===")
    
    print("🔧 STEPS TO FIX TARGET CALCULATION:")
    
    print("\n1. UPDATE prepare_target_returns() function:")
    print("   File: data/data_utils_simple.py lines 131-145")
    print("   Replace current shift(-n_hours) logic with business-hour-aware logic")
    
    print("\n2. UPDATE function signature:")
    print("   OLD: prepare_target_returns(raw_data, target_symbol, n_hours=24, signal_hour=12)")
    print("   NEW: prepare_target_returns(raw_data, target_symbol, n_periods=1, signal_hour=12)")
    print("   OR:  prepare_target_returns(raw_data, target_symbol, n_business_hours=24, signal_hour=12)")
    
    print("\n3. UPDATE calling code:")
    print("   File: data/data_utils_simple.py line 259")
    print("   Update prepare_real_data_simple() to use new parameters")
    
    print("\n4. TEST SCENARIOS:")
    print("   - Friday 12pm → Monday 12pm (n_periods=1 or n_business_hours=24)")
    print("   - Friday 12pm → Monday 3pm (n_business_hours=27)")
    print("   - Friday 12pm → Tuesday 12pm (n_periods=2 or n_business_hours=48)")
    
    print(f"\n🧪 VALIDATION TESTS:")
    print("1. Create test data with proper weekend gaps")
    print("2. Verify Friday 12pm maps exactly to Monday 12pm")
    print("3. Test different n_periods/n_hours values")
    print("4. Confirm last row behavior (should be NaN)")
    print("5. Compare before/after performance impact")
    
    return True

if __name__ == "__main__":
    print("🔍 FINAL COMPREHENSIVE SOLUTION FOR TARGET CALCULATION")
    print("="*70)
    print("Goal: Ensure Friday 12pm → Monday 12pm mapping works perfectly")
    print("Data: Hourly ES futures (Sun 6pm ET - Fri 5pm ET)")
    
    # Step 1: Analyze current data precisely
    recent_data = analyze_current_raw_data_precisely()
    
    # Step 2: Propose optimal solutions
    propose_optimal_solution()
    
    # Step 3: Implementation guidance
    create_implementation_template()
    
    print("\n" + "="*70)
    print("🏁 COMPREHENSIVE SOLUTION COMPLETE")
    
    print(f"\n🎯 SUMMARY OF FINDINGS:")
    print("1. ✅ Data confirmed: Hourly with weekend gaps")
    print("2. ❌ Current shift(-24) is 1 hour off (hits 1pm instead of 12pm)")
    print("3. 🔧 Solution: Business-hour-aware or period-based targeting")
    print("4. 📈 Impact: More accurate target returns, better model performance")
    
    print(f"\n⚡ IMMEDIATE ACTION:")
    print("Choose and implement one of the proposed solutions to fix")
    print("the Friday 12pm → Monday 1pm offset problem!")
    
    print(f"\n🎯 EXPECTED RESULTS AFTER FIX:")
    print("- Friday 12pm signals will predict Friday 12pm → Monday 12pm returns")
    print("- Target calculations will be more accurate")
    print("- Last row behavior should improve")
    print("- Model performance may increase due to better target alignment")
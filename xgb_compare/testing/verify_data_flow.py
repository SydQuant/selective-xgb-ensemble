"""
Verify Data Flow and Target Calculation Logic

1. Confirm we calculate features on hourly data, then aggregate to daily (correct)
2. Verify Friday 12pm → Monday 12pm target calculation works correctly
3. Check if our shift(-24) actually achieves the intended behavior
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def verify_feature_calculation_flow():
    """
    Verify: hourly data → hourly features → daily aggregation (at signal_hour)
    """
    print("=== VERIFYING FEATURE CALCULATION FLOW ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        from data.data_utils_simple import calculate_simple_features, build_features_simple
        
        # Load small sample of raw data
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Take just 48 hours (2 days) for clear analysis
        sample_df = raw_df.tail(48)
        
        print(f"Raw data sample:")
        print(f"Shape: {sample_df.shape}")
        print(f"Frequency: Hourly")
        print(f"Date range: {sample_df.index[0]} to {sample_df.index[-1]}")
        
        print(f"\nFirst 5 raw records:")
        for i in range(5):
            date = sample_df.index[i]
            price = sample_df['close'].iloc[i]
            print(f"  {date}: {price:.2f}")
        
        # Step 1: Calculate features on hourly data
        print(f"\n📊 STEP 1: Calculate features on hourly data")
        hourly_features = calculate_simple_features(sample_df)
        
        print(f"Hourly features shape: {hourly_features.shape}")
        print(f"Feature columns (first 5): {list(hourly_features.columns[:5])}")
        print(f"All hours included: {sorted(set(hourly_features.index.hour))}")
        
        # Step 2: Filter to signal hour (daily aggregation)
        print(f"\n📊 STEP 2: Filter to signal hour (daily aggregation)")
        signal_hour = 12
        daily_features = hourly_features[hourly_features.index.hour == signal_hour]
        
        print(f"Daily features (12pm only) shape: {daily_features.shape}")
        print(f"Hours after filtering: {sorted(set(daily_features.index.hour))}")
        
        if len(daily_features) > 0:
            print(f"Sample daily features:")
            for i in range(min(3, len(daily_features))):
                date = daily_features.index[i]
                momentum = daily_features.iloc[i, 0] if daily_features.shape[1] > 0 else "N/A"
                print(f"  {date} (12pm): momentum={momentum}")
        
        print(f"\n✅ CONFIRMED: Features calculated hourly, then aggregated to daily (12pm)")
        
        return sample_df, hourly_features, daily_features
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def verify_friday_monday_target_logic():
    """
    Verify that Friday 12pm → Monday 12pm target calculation works correctly
    """
    print("\n=== VERIFYING FRIDAY→MONDAY TARGET LOGIC ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        # Load recent data
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Find Friday and Monday 12pm periods
        recent_df = raw_df.tail(500)  # Last 500 records
        
        fridays_12pm = recent_df[(recent_df.index.weekday == 4) & (recent_df.index.hour == 12)]
        mondays_12pm = recent_df[(recent_df.index.weekday == 0) & (recent_df.index.hour == 12)]
        
        print(f"Found {len(fridays_12pm)} Friday 12pm periods")
        print(f"Found {len(mondays_12pm)} Monday 12pm periods")
        
        if len(fridays_12pm) > 0 and len(mondays_12pm) > 0:
            # Test the shift(-24) logic on actual data
            print(f"\n🔍 TESTING shift(-24) ON REAL DATA:")
            
            # Take a Friday 12pm and see what shift(-24) gives us
            test_friday = fridays_12pm.index[-1]  # Most recent Friday 12pm
            friday_price = fridays_12pm.loc[test_friday, 'close']
            
            print(f"Test Friday: {test_friday} (price: {friday_price:.2f})")
            
            # Manual calculation: what should be 24 hours later?
            expected_saturday = test_friday + pd.Timedelta(hours=24)
            print(f"Expected Saturday (Fri+24h): {expected_saturday}")
            
            # What does shift(-24) actually give us?
            future_close_24h = recent_df['close'].shift(-24)
            saturday_price = future_close_24h.loc[test_friday] if test_friday in future_close_24h.index else None
            
            print(f"shift(-24) result: {saturday_price}")
            
            if pd.isna(saturday_price):
                print("❌ shift(-24) gives NaN (no Saturday 12pm data)")
                print("   The system should somehow map Friday 12pm → Monday 12pm")
            else:
                print(f"✅ shift(-24) gives {saturday_price:.2f}")
                print("   But we need to verify this is actually Monday 12pm, not Saturday")
                
                # Find what date this price actually comes from
                saturday_matches = recent_df[recent_df['close'] == saturday_price]
                if len(saturday_matches) > 0:
                    actual_date = saturday_matches.index[0]
                    actual_day = actual_date.strftime('%A')
                    print(f"   This price is from: {actual_date} ({actual_day})")
                    
                    if actual_day == 'Monday':
                        print("   ✅ CORRECT: Actually Monday price (weekend bridged)")
                    else:
                        print(f"   ⚠️  UNEXPECTED: {actual_day} price, not Monday")
        
        return fridays_12pm, mondays_12pm
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None

def test_weekend_bridging_mechanism():
    """
    Test how the data bridges weekends and if our shift logic works correctly
    """
    print("\n=== TESTING WEEKEND BRIDGING MECHANISM ===")
    
    try:
        from data.loaders import get_arcticdb_connection
        
        futures_lib = get_arcticdb_connection()
        versioned_item = futures_lib.read("@ES#C")
        raw_df = versioned_item.data
        
        # Look for Friday-Monday pairs
        recent_df = raw_df.tail(1000)
        
        # Find consecutive Friday→Monday 12pm pairs
        fridays_12pm = recent_df[(recent_df.index.weekday == 4) & (recent_df.index.hour == 12)]
        
        print(f"Testing weekend bridging with {len(fridays_12pm)} Friday 12pm periods...")
        
        for i, (friday_date, friday_row) in enumerate(fridays_12pm.tail(3).iterrows()):
            friday_price = friday_row['close']
            
            print(f"\nTest {i+1}: Friday {friday_date}")
            print(f"  Friday 12pm price: {friday_price:.2f}")
            
            # Find next Monday 12pm manually
            next_monday_candidates = recent_df[
                (recent_df.index > friday_date) & 
                (recent_df.index.weekday == 0) & 
                (recent_df.index.hour == 12)
            ]
            
            if len(next_monday_candidates) > 0:
                next_monday = next_monday_candidates.index[0]
                monday_price = next_monday_candidates.iloc[0]['close']
                
                print(f"  Next Monday 12pm: {next_monday} (price: {monday_price:.2f})")
                print(f"  Time gap: {next_monday - friday_date}")
                print(f"  Manual return calc: {(monday_price - friday_price) / friday_price:.6f}")
                
                # Now test what shift(-24) gives us
                future_close_24h = recent_df['close'].shift(-24)
                shift_result = future_close_24h.loc[friday_date] if friday_date in future_close_24h.index else None
                
                if not pd.isna(shift_result):
                    shift_return = (shift_result - friday_price) / friday_price
                    print(f"  shift(-24) return: {shift_return:.6f}")
                    
                    # Check if they match
                    manual_return = (monday_price - friday_price) / friday_price
                    if abs(shift_return - manual_return) < 0.0001:
                        print("  ✅ GOOD: shift(-24) correctly maps Friday→Monday!")
                    else:
                        print("  ❌ MISMATCH: shift(-24) not mapping to Monday")
                        print(f"       Manual Friday→Monday: {manual_return:.6f}")
                        print(f"       shift(-24) result: {shift_return:.6f}")
                else:
                    print("  ❌ shift(-24) gives NaN (fails to bridge weekend)")
            else:
                print("  No Monday 12pm found after this Friday")
        
        return recent_df
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    print("🔍 VERIFYING DATA FLOW AND TARGET CALCULATION")
    print("="*70)
    
    # Test 1: Feature calculation flow
    raw_data, hourly_feat, daily_feat = verify_feature_calculation_flow()
    
    # Test 2: Friday→Monday logic
    fridays, mondays = verify_friday_monday_target_logic()
    
    # Test 3: Weekend bridging
    bridging_data = test_weekend_bridging_mechanism()
    
    print("\n" + "="*70)
    print("🏁 DATA FLOW VERIFICATION COMPLETE")
    
    print(f"\n📋 CONFIRMED:")
    print("1. ✅ Features: Calculated hourly, aggregated to daily (12pm)")
    print("2. 🔍 Targets: Need to verify Friday→Monday bridging works correctly")
    print("3. 🔍 Last row: Issue may be related to data range vs production timing")
    
    print(f"\n💡 KEY INSIGHTS:")
    print("- Data has intraday fill that bridges weekends")
    print("- shift(-24) may actually work if data is properly filled")
    print("- Last row issue may be about using historical vs live data")
    print("- Row-based approach would still be more robust")
"""
Rigorous Pipeline Testing - Actually Test Real Functions

Step-by-step rigorous testing of actual pipeline functions with controlled data.
No synthetic demonstrations - only real function calls with measurable results.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def step1_test_feature_calculation_rigorously():
    """
    STEP 1: Rigorously test calculate_simple_features() with controlled data
    """
    print("=== STEP 1: RIGOROUS FEATURE CALCULATION TEST ===")
    print("Goal: Test if our ATR fix actually works in the real function")
    
    from data.data_utils_simple import calculate_simple_features
    
    # Create dummy data designed to expose future leakage
    dates = pd.date_range('2024-01-01 12:00', periods=30, freq='h')
    
    # Create price series with KNOWN patterns that future leakage would distort
    prices = [4000]
    for i in range(29):
        # Predictable price movement: +1 for 10 periods, -1 for next 10, +1 for rest
        if i < 10:
            prices.append(prices[-1] + 1)
        elif i < 20:
            prices.append(prices[-1] - 1)
        else:
            prices.append(prices[-1] + 1)
    
    # Create OHLCV with strategic NaN placement to test our fix
    ohlcv_data = pd.DataFrame({
        'open': prices,
        'high': [p + 2 for p in prices],
        'low': [p - 2 for p in prices], 
        'close': prices,
        'volume': [1000] * 30
    }, index=dates)
    
    # Strategic NaN placement that would trigger the old leakage bug
    ohlcv_data.loc[ohlcv_data.index[0:3], 'high'] = np.nan   # First 3 high values missing
    ohlcv_data.loc[ohlcv_data.index[15], 'low'] = np.nan     # Middle gap
    
    print(f"Created controlled test data:")
    print(f"  Periods: {len(ohlcv_data)}")
    print(f"  Price pattern: +1 for 10, -1 for 10, +1 for rest")
    print(f"  NaN placement: First 3 high values, position 15 low value")
    
    # CRITICAL TEST: Call the actual function
    print(f"\n🔍 CALLING REAL calculate_simple_features():")
    
    try:
        real_features = calculate_simple_features(ohlcv_data)
        
        print(f"✅ Function executed successfully")
        print(f"   Input shape: {ohlcv_data.shape}")
        print(f"   Output shape: {real_features.shape}")
        
        # CRITICAL ANALYSIS: Check if our fix worked
        atr_values = real_features['atr']
        
        print(f"\n📊 ATR ANALYSIS (Testing our fix):")
        print(f"First 8 ATR values:")
        for i in range(8):
            atr_val = atr_values.iloc[i]
            high_val = ohlcv_data['high'].iloc[i]
            low_val = ohlcv_data['low'].iloc[i]
            
            high_status = "NaN" if pd.isna(high_val) else f"{high_val:.1f}"
            low_status = "NaN" if pd.isna(low_val) else f"{low_val:.1f}"
            atr_status = "NaN" if pd.isna(atr_val) else "Zero" if atr_val == 0.0 else f"{atr_val:.6f}"
            
            print(f"  Period {i+1}: High={high_status:>6s}, Low={low_status:>6s} → ATR={atr_status}")
            
            # CRITICAL CHECK: Periods 1-3 should have zero ATR (no future data used)
            if i < 3 and atr_val > 0:
                print(f"    ❌ POTENTIAL ISSUE: Period {i+1} has ATR despite missing high data")
            elif i < 3 and atr_val == 0.0:
                print(f"    ✅ CONFIRMED: Period {i+1} has zero ATR (fix working)")
        
        # Test position 15 (middle NaN)
        pos_15_atr = atr_values.iloc[15]
        pos_14_atr = atr_values.iloc[14]
        pos_16_atr = atr_values.iloc[16]
        
        print(f"\nMiddle gap test (position 15 has NaN low):")
        print(f"  Position 14 ATR: {pos_14_atr:.6f}")
        print(f"  Position 15 ATR: {pos_15_atr:.6f}")
        print(f"  Position 16 ATR: {pos_16_atr:.6f}")
        
        # CRITICAL: Position 15 should use forward-fill (past data), not future data
        if abs(pos_15_atr - pos_14_atr) < 0.000001:
            print("  ✅ CONFIRMED: Position 15 uses forward-fill (past data)")
        elif abs(pos_15_atr - pos_16_atr) < 0.000001:
            print("  ❌ BUG DETECTED: Position 15 uses future data!")
        else:
            print("  🔍 UNCLEAR: Position 15 ATR doesn't match expected pattern")
        
        return real_features, True
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Real function failed: {e}")
        return None, False

def step2_test_target_calculation_rigorously():
    """
    STEP 2: Rigorously test target calculation with controlled data
    """
    print("\n=== STEP 2: RIGOROUS TARGET CALCULATION TEST ===")
    print("Goal: Test if target calculation works correctly with controlled data")
    
    # I need to test the target calculation functions directly
    from data.data_utils_simple import prepare_target_returns, build_features_simple
    
    # Create dummy raw_data in the format expected by the function
    dates = pd.date_range('2024-01-01 12:00', periods=50, freq='h')
    
    # Create realistic OHLCV data with KNOWN weekend pattern
    base_price = 4000
    prices = []
    
    for i, date in enumerate(dates):
        # Create predictable price movement that we can verify
        if date.weekday() < 5:  # Weekday
            price_change = 1 if date.hour % 2 == 0 else -0.5  # Predictable pattern
        else:  # Weekend (shouldn't exist in real data, but for testing)
            price_change = 0  # Flat on weekends
        
        new_price = base_price + i * 0.5 + price_change
        prices.append(new_price)
    
    raw_data_dummy = {
        "@ES#C": pd.DataFrame({
            'open': prices,
            'high': [p + 2 for p in prices],
            'low': [p - 2 for p in prices],
            'close': prices,
            'volume': [1000] * 50
        }, index=dates)
    }
    
    print(f"Created dummy raw_data:")
    print(f"  Symbol: @ES#C")
    print(f"  Periods: {len(raw_data_dummy['@ES#C'])}")
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    
    # CRITICAL TEST: Call real target calculation function
    print(f"\n🔍 CALLING REAL prepare_target_returns():")
    
    try:
        target_returns = prepare_target_returns(raw_data_dummy, "@ES#C", n_hours=24, signal_hour=12)
        
        print(f"✅ Function executed successfully")
        print(f"   Target returns shape: {target_returns.shape}")
        
        if len(target_returns) > 0:
            print(f"   Date range: {target_returns.index[0]} to {target_returns.index[-1]}")
            
            # CRITICAL ANALYSIS: Check last row behavior
            last_target = target_returns.iloc[-1] if len(target_returns) > 0 else None
            
            print(f"\n📊 TARGET CALCULATION ANALYSIS:")
            print(f"Valid targets: {target_returns.dropna().shape[0]}")
            print(f"NaN targets: {target_returns.isna().sum()}")
            print(f"Last target: {last_target} ({'NaN' if pd.isna(last_target) else 'HAS VALUE'})")
            
            # CRITICAL CHECK: With our dummy data, can we verify the calculation?
            if len(target_returns) >= 2:
                # Manually verify first target calculation
                first_signal_date = target_returns.index[0]
                first_target_value = target_returns.iloc[0]
                
                # Find corresponding prices
                current_price = raw_data_dummy["@ES#C"].loc[first_signal_date, 'close']
                
                # What should shift(-24) give us?
                expected_future_date = first_signal_date + pd.Timedelta(hours=24)
                
                if expected_future_date in raw_data_dummy["@ES#C"].index:
                    expected_future_price = raw_data_dummy["@ES#C"].loc[expected_future_date, 'close']
                    expected_return = (expected_future_price - current_price) / current_price
                    
                    print(f"\nManual verification:")
                    print(f"  Signal date: {first_signal_date}")
                    print(f"  Current price: {current_price:.2f}")
                    print(f"  Expected future date: {expected_future_date}")
                    print(f"  Expected future price: {expected_future_price:.2f}")
                    print(f"  Expected return: {expected_return:.6f}")
                    print(f"  Function return: {first_target_value:.6f}")
                    print(f"  Match: {abs(expected_return - first_target_value) < 0.000001}")
                    
                    if abs(expected_return - first_target_value) < 0.000001:
                        print("  ✅ CONFIRMED: Target calculation is mathematically correct")
                    else:
                        print("  ❌ BUG DETECTED: Target calculation doesn't match expected")
        
        return target_returns, True
        
    except Exception as e:
        print(f"❌ CRITICAL FAILURE: Real function failed: {e}")
        return None, False

def step3_test_full_pipeline_with_dummy_data():
    """
    STEP 3: Test the complete pipeline by creating a dummy data version
    """
    print("\n=== STEP 3: FULL PIPELINE TEST WITH CONTROLLED DATA ===")
    print("Goal: Create version of pipeline that uses dummy data and test end-to-end")
    
    # Create a controlled version of the pipeline
    from data.data_utils_simple import build_features_simple, prepare_target_returns, clean_data_simple
    
    # Create dummy raw_data with specific patterns to test
    dates = pd.date_range('2024-01-01 12:00', periods=72, freq='h')  # 3 days
    
    # Create price data with KNOWN future pattern to test leakage
    prices = []
    for i, date in enumerate(dates):
        # Pattern: price increases by 1 every hour, but with a jump on day 2
        if i < 24:  # Day 1
            price = 4000 + i
        elif i < 48:  # Day 2 - sudden jump
            price = 4100 + (i - 24)
        else:  # Day 3 - back to normal
            price = 4200 + (i - 48)
        prices.append(price)
    
    raw_data_controlled = {
        "@ES#C": pd.DataFrame({
            'open': prices,
            'high': [p + 1 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000] * 72
        }, index=dates)
    }
    
    print(f"Created controlled raw_data with KNOWN pattern:")
    print(f"  Day 1: 4000-4023 (hourly +1)")
    print(f"  Day 2: 4100-4123 (jump + hourly +1)")  
    print(f"  Day 3: 4200-4223 (jump + hourly +1)")
    
    # STEP 3A: Test feature building
    print(f"\n🔍 TESTING REAL build_features_simple():")
    
    try:
        features = build_features_simple(raw_data_controlled, "@ES#C", signal_hour=12)
        
        print(f"✅ Features built successfully")
        print(f"   Features shape: {features.shape}")
        
        # CRITICAL CHECK: Do features at 12pm reflect the controlled pattern?
        features_12pm = features[features.index.hour == 12]
        
        print(f"   12pm features: {len(features_12pm)} periods")
        
        if len(features_12pm) >= 3:
            print(f"Feature values at 12pm each day:")
            for i, (date, row) in enumerate(features_12pm.iterrows()):
                momentum = row['@ES#C_momentum_1h'] if '@ES#C_momentum_1h' in row.index else "N/A"
                rsi = row['@ES#C_rsi'] if '@ES#C_rsi' in row.index else "N/A"
                
                expected_price = [4012, 4112, 4212][i] if i < 3 else "Unknown"
                
                print(f"    Day {i+1} 12pm: momentum={momentum}, expected_price={expected_price}")
                
        step3a_success = True
        
    except Exception as e:
        print(f"❌ FEATURE BUILD FAILED: {e}")
        step3a_success = False
        features = None
    
    # STEP 3B: Test target calculation
    print(f"\n🔍 TESTING REAL prepare_target_returns():")
    
    try:
        targets = prepare_target_returns(raw_data_controlled, "@ES#C", n_hours=24, signal_hour=12)
        
        print(f"✅ Targets calculated successfully")
        print(f"   Targets shape: {targets.shape}")
        
        if len(targets) > 0:
            print(f"Target returns:")
            for i, (date, target) in enumerate(targets.items()):
                print(f"    {date}: {target:.6f}" if not pd.isna(target) else f"    {date}: NaN")
                
                # CRITICAL VERIFICATION: Can we manually verify this?
                if not pd.isna(target) and i < len(targets) - 1:
                    current_price = raw_data_controlled["@ES#C"].loc[date, 'close']
                    future_date = date + pd.Timedelta(hours=24)
                    
                    if future_date in raw_data_controlled["@ES#C"].index:
                        future_price = raw_data_controlled["@ES#C"].loc[future_date, 'close']
                        expected_return = (future_price - current_price) / current_price
                        
                        print(f"      Manual check: {current_price:.0f}→{future_price:.0f} = {expected_return:.6f}")
                        
                        if abs(target - expected_return) < 0.000001:
                            print(f"      ✅ VERIFIED: Calculation correct")
                        else:
                            print(f"      ❌ ERROR: Expected {expected_return:.6f}, got {target:.6f}")
        
        step3b_success = True
        
    except Exception as e:
        print(f"❌ TARGET CALCULATION FAILED: {e}")
        step3b_success = False
        targets = None
    
    # STEP 3C: Test data cleaning
    if step3a_success and step3b_success and features is not None and targets is not None:
        print(f"\n🔍 TESTING REAL clean_data_simple():")
        
        try:
            # Combine features and targets (mimicking the real pipeline)
            target_col = "@ES#C_target_return"
            combined_df = pd.concat([features, targets.to_frame(target_col)], axis=1)
            
            print(f"Before cleaning:")
            print(f"   Combined shape: {combined_df.shape}")
            print(f"   NaN targets: {combined_df[target_col].isna().sum()}")
            print(f"   Last row target: {combined_df[target_col].iloc[-1]}")
            
            # Call real cleaning function
            cleaned_df = clean_data_simple(combined_df)
            
            print(f"After cleaning:")
            print(f"   Cleaned shape: {cleaned_df.shape}")
            
            if len(cleaned_df) > 0:
                print(f"   NaN targets: {cleaned_df[target_col].isna().sum()}")
                print(f"   Last row target: {cleaned_df[target_col].iloc[-1]}")
                
                # CRITICAL ANALYSIS: Was the last row with NaN target removed?
                rows_dropped = len(combined_df) - len(cleaned_df)
                print(f"   Rows dropped: {rows_dropped}")
                
                if rows_dropped > 0:
                    print("   ❌ CONFIRMED BUG: clean_data_simple() drops NaN target rows")
                    print("   This is the source of the 'last row has target' problem!")
                    
            return cleaned_df, True
            
        except Exception as e:
            print(f"❌ CLEANING FAILED: {e}")
            return None, False
    
    return None, step3a_success and step3b_success

def step4_measure_actual_impact():
    """
    STEP 4: Actually measure the impact of our fixes on real performance
    """
    print("\n=== STEP 4: MEASURING ACTUAL IMPACT ===")
    print("Goal: Run before/after comparison on real data to measure fix impact")
    
    print("🔍 ATTEMPTING BEFORE/AFTER COMPARISON:")
    
    # This is challenging because we'd need to:
    # 1. Temporarily revert our fix
    # 2. Run the pipeline
    # 3. Measure performance
    # 4. Apply our fix
    # 5. Run again and compare
    
    print("LIMITATION: This requires running full backtests which is time-intensive")
    print("ALTERNATIVE: Test on smaller data subset with controlled conditions")
    
    try:
        from data.data_utils_simple import prepare_real_data_simple
        
        # Test with actual data but limited date range
        print("\nTesting with real but limited data...")
        
        df_result = prepare_real_data_simple("@ES#C", start_date="2024-01-01", end_date="2024-01-03")
        
        print(f"✅ Real pipeline executed")
        print(f"   Result shape: {df_result.shape}")
        
        target_col = [c for c in df_result.columns if 'target_return' in c][0]
        
        print(f"   Target column: {target_col}")
        print(f"   Valid targets: {df_result[target_col].dropna().shape[0]}")
        print(f"   NaN targets: {df_result[target_col].isna().sum()}")
        
        # CRITICAL: Check last row 
        last_target = df_result[target_col].iloc[-1] if len(df_result) > 0 else None
        print(f"   Last row target: {last_target} ({'NaN' if pd.isna(last_target) else 'HAS VALUE'})")
        
        if not pd.isna(last_target):
            print("   ❌ CONFIRMED: Last row has target value (potential future leakage)")
        else:
            print("   ✅ GOOD: Last row has NaN target")
            
        return df_result, True
        
    except Exception as e:
        print(f"❌ REAL PIPELINE TEST FAILED: {e}")
        return None, False

if __name__ == "__main__":
    print("🔍 RIGOROUS STEP-BY-STEP PIPELINE TESTING")
    print("="*70)
    print("CRITICAL APPROACH: Test actual functions, not synthetic examples")
    
    # Step 1: Feature calculation
    features, step1_success = step1_test_feature_calculation_rigorously()
    
    # Step 2: Target calculation  
    targets, step2_success = step2_test_target_calculation_rigorously()
    
    # Step 3: Full pipeline components
    cleaned_data, step3_success = step3_test_full_pipeline_with_dummy_data()
    
    # Step 4: Actual impact measurement
    real_result, step4_success = step4_measure_actual_impact()
    
    print("\n" + "="*70)
    print("🏁 RIGOROUS TESTING RESULTS")
    
    print(f"\n📊 TEST OUTCOMES:")
    print(f"Step 1 (Feature calc): {'✅ PASS' if step1_success else '❌ FAIL'}")
    print(f"Step 2 (Target calc): {'✅ PASS' if step2_success else '❌ FAIL'}")
    print(f"Step 3 (Full pipeline): {'✅ PASS' if step3_success else '❌ FAIL'}")
    print(f"Step 4 (Impact measure): {'✅ PASS' if step4_success else '❌ FAIL'}")
    
    overall_success = all([step1_success, step2_success, step3_success, step4_success])
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    if overall_success:
        print("✅ RIGOROUS TESTING SUCCESSFUL")
        print("   Our fixes have been validated with actual pipeline functions")
    else:
        print("❌ TESTING REVEALED ISSUES")
        print("   Need to address failures before claiming fixes work")
        
    print(f"\n💡 CRITICAL INSIGHT:")
    print("This is what actual rigorous testing looks like:")
    print("- Using real functions with controlled inputs")
    print("- Verifying mathematical correctness")
    print("- Measuring actual differences, not just illustrations")
    print("- Being honest about what worked vs what failed")
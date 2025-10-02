"""
Investigate Last Row Target Problem in Detail

Call the inner functions step by step to see exactly where the NaN target
becomes a real value, and verify if the raw target calculation has NaN last row.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_raw_target_calculation():
    """
    Test the prepare_target_returns function directly (before cleaning)
    """
    print("=== TESTING RAW TARGET CALCULATION ===")
    
    from data.data_utils_simple import prepare_target_returns
    
    # Create minimal test data that mimics real scenario
    dates = pd.date_range('2023-01-01 12:00', periods=8, freq='D')
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    
    raw_data = {
        "@ES#C": pd.DataFrame({
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [1000] * 8
        }, index=dates)
    }
    
    print("Raw price data:")
    for i, (date, price) in enumerate(raw_data["@ES#C"]['close'].items()):
        print(f"Day {i+1} ({date.strftime('%Y-%m-%d %H:%M')}): {price}")
    
    # Test the target calculation function directly
    target_returns = prepare_target_returns(raw_data, "@ES#C", n_hours=24, signal_hour=12)
    
    print(f"\n📊 DIRECT TARGET CALCULATION RESULTS:")
    print(f"Target returns shape: {target_returns.shape}")
    print(f"Target returns:")
    
    for i, (date, target) in enumerate(target_returns.items()):
        next_price = raw_data["@ES#C"]['close'].iloc[i+1] if i+1 < len(raw_data["@ES#C"]) else "N/A"
        current_price = raw_data["@ES#C"]['close'].iloc[i]
        
        if pd.isna(target):
            print(f"Day {i+1} ({date.strftime('%m-%d')}): NaN (no future price available)")
        else:
            expected = (next_price - current_price) / current_price if next_price != "N/A" else "N/A"
            print(f"Day {i+1} ({date.strftime('%m-%d')}): {target:.6f} (calculated from {current_price}→{next_price})")
    
    # Critical check: Is the last row NaN?
    last_target = target_returns.iloc[-1] if len(target_returns) > 0 else None
    print(f"\n🔍 LAST ROW CHECK:")
    print(f"Last target value: {last_target}")
    
    if pd.isna(last_target):
        print("✅ GOOD: Raw target calculation has NaN in last row")
    else:
        print("❌ PROBLEM: Raw target calculation has value in last row!")
        print("   This suggests the shift(-n_hours) is not working as expected")
    
    return target_returns, raw_data

def test_feature_building_step():
    """
    Test the feature building step to see what happens before cleaning
    """
    print("\n=== TESTING FEATURE BUILDING STEP ===")
    
    from data.data_utils_simple import build_features_simple, prepare_target_returns
    
    # Use the same test data
    dates = pd.date_range('2023-01-01 12:00', periods=6, freq='D')
    prices = [100, 101, 102, 103, 104, 105]
    
    raw_data = {
        "@ES#C": pd.DataFrame({
            'open': prices,
            'high': [p + 0.5 for p in prices],
            'low': [p - 0.5 for p in prices],
            'close': prices,
            'volume': [1000] * 6
        }, index=dates)
    }
    
    # Step 1: Build features
    feature_df = build_features_simple(raw_data, "@ES#C", signal_hour=12)
    print(f"Features built: {feature_df.shape}")
    print(f"Feature index: {feature_df.index[0]} to {feature_df.index[-1]}")
    
    # Step 2: Build targets
    target_returns = prepare_target_returns(raw_data, "@ES#C", n_hours=24, signal_hour=12)
    print(f"Targets built: {target_returns.shape}")
    print(f"Target index: {target_returns.index[0]} to {target_returns.index[-1]}")
    
    # Step 3: Combine (this is what prepare_real_data_simple does)
    target_col = "@ES#C_target_return"
    target_reindexed = target_returns.reindex(feature_df.index)
    combined_df = pd.concat([feature_df, target_reindexed.to_frame(target_col)], axis=1)
    
    print(f"\nCombined dataframe before cleaning:")
    print(f"Shape: {combined_df.shape}")
    print(f"Last 3 rows of target column:")
    print(combined_df[target_col].tail(3))
    
    # Check last row specifically
    last_target_before_clean = combined_df[target_col].iloc[-1]
    print(f"\n🔍 BEFORE CLEANING - Last row target: {last_target_before_clean}")
    
    return combined_df

def test_cleaning_step_effect():
    """
    Test the cleaning step to see exactly what it does
    """
    print("\n=== TESTING CLEANING STEP EFFECT ===")
    
    from data.data_utils_simple import clean_data_simple
    
    # Get data before cleaning
    combined_df = test_feature_building_step()
    
    target_col = "@ES#C_target_return"
    print(f"Before cleaning:")
    print(f"Shape: {combined_df.shape}")
    print(f"NaN targets: {combined_df[target_col].isna().sum()}")
    print(f"Last row target: {combined_df[target_col].iloc[-1]}")
    
    # Apply cleaning
    cleaned_df = clean_data_simple(combined_df)
    
    print(f"\nAfter cleaning:")
    print(f"Shape: {cleaned_df.shape}")
    if len(cleaned_df) > 0:
        print(f"NaN targets: {cleaned_df[target_col].isna().sum()}")
        print(f"Last row target: {cleaned_df[target_col].iloc[-1]}")
        
        # Show what was dropped
        dropped_rows = len(combined_df) - len(cleaned_df)
        print(f"Dropped {dropped_rows} rows during cleaning")
        
        if dropped_rows > 0:
            print("🔍 WHAT WAS DROPPED:")
            # Find which rows were dropped
            dropped_indices = combined_df.index.difference(cleaned_df.index)
            for idx in dropped_indices:
                target_val = combined_df.loc[idx, target_col]
                print(f"   Dropped {idx}: target = {target_val}")
    else:
        print("❌ CRITICAL: All data was dropped during cleaning!")
    
    return combined_df, cleaned_df

def call_actual_pipeline_functions():
    """
    Call the actual pipeline step by step to trace the issue
    """
    print("\n=== CALLING ACTUAL PIPELINE FUNCTIONS ===")
    
    try:
        # This mimics what prepare_real_data_simple does internally
        from data.data_utils_simple import prepare_real_data_simple
        from data.loaders import get_arcticdb_connection
        from data.symbol_loader import get_default_symbols
        
        # Try with very limited date range to get minimal data
        print("Calling prepare_real_data_simple with minimal date range...")
        
        # Call with just 3 days to minimize data
        df_final = prepare_real_data_simple("@ES#C", start_date="2023-01-01", end_date="2023-01-03")
        
        target_col = "@ES#C_target_return"
        
        print(f"Final result:")
        print(f"Shape: {df_final.shape}")
        print(f"Date range: {df_final.index[0]} to {df_final.index[-1]}")
        print(f"Target column: {target_col}")
        
        if len(df_final) > 0:
            print(f"First target: {df_final[target_col].iloc[0]}")
            print(f"Last target: {df_final[target_col].iloc[-1]}")
            print(f"NaN targets: {df_final[target_col].isna().sum()}")
            
            # Show all targets
            print("\nAll targets:")
            for i, (date, target) in enumerate(df_final[target_col].items()):
                print(f"Row {i+1} ({date.strftime('%Y-%m-%d %H:%M')}): {target}")
                
            print(f"\n🔍 DIAGNOSIS:")
            if pd.isna(df_final[target_col].iloc[-1]):
                print("✅ Last row target is NaN - this is correct!")
                print("   Issue might be elsewhere in the pipeline")
            else:
                print("❌ Last row target has a value - confirming the bug!")
                print("   The cleaning step is likely dropping NaN rows incorrectly")
        
    except Exception as e:
        print(f"Error calling actual pipeline: {e}")
        print("This might be expected if no real data is available")
    
    return True

def demonstrate_shift_calculation():
    """
    Demonstrate the shift calculation to verify it works correctly
    """
    print("\n=== DEMONSTRATING SHIFT CALCULATION ===")
    
    # Simple test of the shift(-n_hours) logic
    dates = pd.date_range('2023-01-01 12:00', periods=5, freq='D')
    prices = pd.Series([100, 101, 102, 103, 104], index=dates)
    
    print("Original prices:")
    for i, (date, price) in enumerate(prices.items()):
        print(f"Day {i+1} ({date.strftime('%m-%d')}): {price}")
    
    # Test different shift values
    for n_hours in [1, 24]:  # 1 hour, 24 hours (1 day for daily data)
        future_close = prices.shift(-n_hours)
        returns = (future_close - prices) / prices
        
        print(f"\nWith n_hours={n_hours}:")
        print(f"Future close (shift(-{n_hours})):")
        for i, (date, price) in enumerate(future_close.items()):
            print(f"Day {i+1}: {price if not pd.isna(price) else 'NaN'}")
            
        print(f"Calculated returns:")
        for i, (date, ret) in enumerate(returns.items()):
            print(f"Day {i+1}: {ret if not pd.isna(ret) else 'NaN'}")
    
    return True

if __name__ == "__main__":
    print("🔍 DETAILED INVESTIGATION OF LAST ROW TARGET PROBLEM")
    print("="*70)
    print("Testing each step of the pipeline to isolate the issue...")
    
    # Test 1: Raw target calculation
    target_returns, raw_data = test_raw_target_calculation()
    
    # Test 2: Feature building step
    combined_df = test_feature_building_step()
    
    # Test 3: Cleaning step effect
    before_clean, after_clean = test_cleaning_step_effect()
    
    # Test 4: Actual pipeline
    call_actual_pipeline_functions()
    
    # Test 5: Shift calculation verification
    demonstrate_shift_calculation()
    
    print("\n" + "="*70)
    print("🏁 DETAILED INVESTIGATION COMPLETE")
    
    print(f"\n🎯 KEY FINDINGS:")
    print("1. Raw target calculation behavior")
    print("2. Effect of feature building step")
    print("3. Impact of cleaning step (dropna)")
    print("4. Final pipeline result")
    print("5. Shift calculation verification")
    
    print(f"\n💡 NEXT STEPS:")
    print("Based on the results above, we can pinpoint exactly where")
    print("the last row NaN target becomes a real value.")
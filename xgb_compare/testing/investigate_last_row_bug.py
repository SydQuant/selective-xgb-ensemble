"""
Investigate Last Row Target Bug

The issue: Our pipeline drops rows with NaN targets, which removes the last row
that SHOULD be NaN in production (since we can't know future returns yet).
This creates artificial access to future information.
"""

import pandas as pd
import numpy as np
import sys
import os

def demonstrate_last_row_bug():
    """
    Show exactly why the last row has a target value when it shouldn't
    """
    print("=== DEMONSTRATING LAST ROW TARGET BUG ===")
    
    # Simulate the target calculation process
    dates = pd.date_range('2020-01-01 12:00', periods=10, freq='D')
    prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    
    df = pd.DataFrame({
        'close': prices
    }, index=dates)
    
    print("Original price data:")
    for i, (date, price) in enumerate(df.iterrows()):
        print(f"Day {i+1} ({date.strftime('%m-%d')}): {price['close']}")
    
    # Step 1: Calculate target returns (shift(-24) for 24-hour ahead)
    n_hours = 24  # This is 1 day in our case
    future_close = df['close'].shift(-n_hours)  # This is shift(-1) for daily data
    returns = (future_close - df['close']) / df['close']
    
    print(f"\nStep 1: Target calculation (shift(-{n_hours})):")
    print("future_close (what close will be tomorrow):")
    for i, (date, val) in enumerate(future_close.items()):
        print(f"Day {i+1}: {val if not pd.isna(val) else 'NaN (no future data)'}")
    
    print("\nCalculated target returns:")
    for i, (date, val) in enumerate(returns.items()):
        print(f"Day {i+1}: {val if not pd.isna(val) else 'NaN (no future data)'}")
    
    # This is CORRECT - last row should be NaN
    print(f"\n✅ CORRECT STATE: Last row target = {returns.iloc[-1]} (should be NaN)")
    
    # Step 2: The problematic dropna() operation
    print(f"\nStep 2: Problematic dropna() operation...")
    
    # Combine with dummy features
    feature_df = pd.DataFrame({
        'feature_1': np.random.random(10),
        'feature_2': np.random.random(10),
        'target_return': returns
    }, index=dates)
    
    print("Before dropna():")
    print(f"Shape: {feature_df.shape}")
    print(f"Last 3 rows:")
    print(feature_df.tail(3)[['target_return']])
    
    # This is the problematic line from data_utils_simple.py:207
    feature_df_cleaned = feature_df.dropna(subset=['target_return'])
    
    print(f"\nAfter dropna(subset=['target_return']):")
    print(f"Shape: {feature_df_cleaned.shape}")
    print(f"Last 3 rows:")
    print(feature_df_cleaned.tail(3)[['target_return']])
    
    print(f"\n❌ PROBLEM: Last row now has target = {feature_df_cleaned['target_return'].iloc[-1]:.6f}")
    print("   In production, we shouldn't know this future return!")
    
    return feature_df, feature_df_cleaned

def show_production_vs_backtest_difference():
    """
    Show how this affects production vs backtest
    """
    print("\n=== PRODUCTION vs BACKTEST DIFFERENCE ===")
    
    print("📊 BACKTEST SCENARIO (what we're doing):")
    print("   Data: Jan 1-10 with targets calculated using Jan 2-11 prices")
    print("   After dropna(): Jan 1-9 (removed Jan 10 because target was NaN)")
    print("   Last model input: Jan 9 features → predict Jan 9→10 return")
    print("   Last model target: Jan 9→10 return = KNOWN (from historical data)")
    print("   Result: Model gets 'free' correct answer for Jan 9→10 period")
    
    print("\n🏭 PRODUCTION SCENARIO (what should happen):")
    print("   Current date: Jan 10 (live trading)")
    print("   Available data: Jan 1-10 prices")
    print("   Model input: Jan 10 features → predict Jan 10→11 return") 
    print("   Model target: Jan 10→11 return = UNKNOWN (future!)")
    print("   Result: Model makes real prediction without knowing answer")
    
    print("\n🚨 THE DISCREPANCY:")
    print("   Backtest last prediction: Jan 9→10 return (known answer)")
    print("   Production last prediction: Jan 10→11 return (unknown answer)")
    print("   → Backtest has 1-day advantage (peeking into future)")
    
    return True

def test_fix_approach():
    """
    Test potential fix - don't drop NaN targets in last row
    """
    print("\n=== TESTING FIX APPROACH ===")
    
    # Create test data
    dates = pd.date_range('2020-01-01 12:00', periods=8, freq='D')
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    
    df = pd.DataFrame({'close': prices}, index=dates)
    
    # Calculate targets (last will be NaN)
    future_close = df['close'].shift(-1)  # 1 day ahead
    returns = (future_close - df['close']) / df['close']
    
    # Add features
    feature_df = pd.DataFrame({
        'feature_1': np.arange(8),
        'feature_2': np.arange(8) * 0.1,
        'target_return': returns
    }, index=dates)
    
    print("Original dataset:")
    print(feature_df[['target_return']])
    
    # Current approach (WRONG)
    df_wrong = feature_df.dropna(subset=['target_return'])
    print(f"\nWrong approach (dropna): Shape {df_wrong.shape}")
    print(f"Last row target: {df_wrong['target_return'].iloc[-1]:.6f}")
    
    # Better approach - keep NaN targets but handle them in training
    df_correct = feature_df.copy()
    
    # For training, we can split into train/predict differently
    train_mask = ~df_correct['target_return'].isna()
    
    X_all = df_correct[['feature_1', 'feature_2']]  # All features
    y_all = df_correct['target_return']  # All targets (including NaN)
    
    X_train = X_all[train_mask]  # Features where we have targets
    y_train = y_all[train_mask]  # Non-NaN targets only
    
    X_predict = X_all[~train_mask]  # Features where we don't have targets (production case)
    
    print(f"\nCorrect approach:")
    print(f"X_all shape: {X_all.shape} (all periods)")
    print(f"X_train shape: {X_train.shape} (periods with known targets)")
    print(f"X_predict shape: {X_predict.shape} (periods for prediction)")
    print(f"Last train target: {y_train.iloc[-1]:.6f}")
    print(f"Prediction period: {X_predict.index.tolist()}")
    
    print(f"\n✅ CORRECT BEHAVIOR:")
    print("   - Train on periods 1-7 (where we know future returns)")
    print("   - Predict for period 8 (where future return is unknown)")
    print("   - No access to future information")
    
    return df_wrong, df_correct

def investigate_data_date_range():
    """
    Investigate what date range our test data actually covers
    """
    print("\n=== INVESTIGATING ACTUAL DATA DATE RANGE ===")
    
    # Add parent directories to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    try:
        from data.data_utils_simple import prepare_real_data_simple
        
        # Test with our actual call
        df = prepare_real_data_simple("@ES#C", start_date="2023-01-01", end_date="2023-01-10")
        
        print(f"Actual data loaded:")
        print(f"Shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        target_col = [c for c in df.columns if 'target_return' in c][0]
        
        print(f"\nTarget column: {target_col}")
        print(f"First 3 targets: {df[target_col].head(3).values}")
        print(f"Last 3 targets: {df[target_col].tail(3).values}")
        
        # Check if this is historical data (all targets known) vs production data
        nan_count = df[target_col].isna().sum()
        print(f"NaN targets: {nan_count}")
        
        if nan_count == 0:
            print("🔍 INSIGHT: This appears to be historical data (all targets known)")
            print("   In true production, the last row target should be NaN")
            print("   The dropna() is removing rows that SHOULD exist in production")
        else:
            print(f"🔍 INSIGHT: {nan_count} NaN targets found - might be correct")
            
    except Exception as e:
        print(f"Could not load actual data: {e}")
    
    return True

if __name__ == "__main__":
    print("🔍 INVESTIGATING LAST ROW TARGET BUG")
    print("="*60)
    
    # Investigation 1: Demonstrate the bug
    original_df, cleaned_df = demonstrate_last_row_bug()
    
    # Investigation 2: Production vs backtest difference
    show_production_vs_backtest_difference()
    
    # Investigation 3: Test fix approach
    wrong_df, correct_df = test_fix_approach()
    
    # Investigation 4: Check actual data
    investigate_data_date_range()
    
    print("\n" + "="*60)
    print("🏁 INVESTIGATION COMPLETE")
    
    print(f"\n🎯 ROOT CAUSE IDENTIFIED:")
    print("Line 207 in data_utils_simple.py:")
    print("   df_clean.dropna(subset=[target_col])")
    print("   → This removes the last row that should have NaN target")
    print("   → Creates artificial access to future information")
    
    print(f"\n🔧 RECOMMENDED FIX:")
    print("1. Don't drop NaN targets during data preparation")
    print("2. Handle NaN targets during training (exclude from training set)")
    print("3. Keep NaN target rows for production prediction scenarios")
    print("4. This ensures backtest matches production conditions")
    
    print(f"\n⚠️  IMPACT:")
    print("Current approach gives models access to 1 period of future information")
    print("This likely inflates backtest performance vs production performance")
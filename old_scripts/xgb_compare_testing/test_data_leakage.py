"""
Test Data Preparation for Future Leakage and Last Row Issues

This script performs comprehensive testing of the data pipeline to identify:
1. Future leakage in target calculation  
2. Last row target issues
3. Feature/target temporal alignment problems
4. Cross-validation boundary issues
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_target_calculation_logic():
    """
    Test the target calculation logic for future leakage.
    
    Key issue to test: prepare_target_returns() uses shift(-n_hours) 
    This means we're looking FORWARD to calculate the target, which is correct.
    
    BUT: The last row will have NaN target (no future data available)
    If the last row has a non-NaN target, that's a RED FLAG.
    """
    print("=== TESTING TARGET CALCULATION LOGIC ===")
    
    # Simulate the prepare_target_returns logic with dummy data
    dates = pd.date_range('2020-01-01 12:00', periods=10, freq='H')
    close_prices = np.arange(1, 11)  # 1, 2, 3, ..., 10
    
    df = pd.DataFrame({
        'close': close_prices
    }, index=dates)
    
    print("Original price data:")
    print(df)
    
    # Replicate prepare_target_returns logic
    n_hours = 2  # Looking 2 hours ahead
    future_close = df['close'].shift(-n_hours)  # shift(-2) 
    returns = (future_close - df['close']) / df['close']
    
    print(f"\nTarget calculation (n_hours={n_hours}):")
    print("future_close (shift(-2)):")
    print(future_close)
    print("\nCalculated returns:")
    print(returns)
    
    # Check last row issue
    last_n_rows = returns.tail(n_hours)
    print(f"\n🔍 LAST {n_hours} ROWS CHECK:")
    print(last_n_rows)
    
    nan_in_last_rows = last_n_rows.isna().sum()
    print(f"\nNaN values in last {n_hours} rows: {nan_in_last_rows}")
    
    if nan_in_last_rows == n_hours:
        print("✅ GOOD: Last rows have NaN targets (no future data available)")
    else:
        print("⚠️  WARNING: Some last rows have non-NaN targets!")
        print("   This suggests we might have access to future information")
    
    return df, returns

def test_feature_target_alignment():
    """
    Test temporal alignment between features and targets.
    
    Critical issue: Features at time T should predict target from T to T+n_hours
    If features include information from time > T, that's future leakage.
    """
    print("\n=== TESTING FEATURE-TARGET ALIGNMENT ===")
    
    # Create dummy data with clear timestamps  
    dates = pd.date_range('2020-01-01 12:00', periods=8, freq='H')
    
    # Feature: RSI calculated from past prices (backward looking - GOOD)
    prices = [100, 101, 102, 103, 104, 105, 106, 107]
    rsi_values = [50, 51, 52, 53, 54, 55, 56, 57]  # Simulated RSI
    
    # Target: 2-hour forward return (forward looking - GOOD)  
    n_hours = 2
    target_values = []
    for i in range(len(prices)):
        if i + n_hours < len(prices):
            # Return from current price to price 2 hours later
            ret = (prices[i + n_hours] - prices[i]) / prices[i]
            target_values.append(ret)
        else:
            target_values.append(np.nan)  # No future data available
    
    df = pd.DataFrame({
        'price': prices,
        'rsi': rsi_values,
        'target_return': target_values
    }, index=dates)
    
    print("Feature-Target alignment test:")
    print(df)
    
    # Verify alignment logic
    print(f"\n🔍 ALIGNMENT VERIFICATION:")
    for i in range(min(5, len(df)-2)):
        current_time = df.index[i]
        future_time = df.index[i+n_hours] if i+n_hours < len(df) else "N/A"
        
        feature_val = df['rsi'].iloc[i]
        target_val = df['target_return'].iloc[i] 
        expected_future_price = df['price'].iloc[i+n_hours] if i+n_hours < len(df) else None
        
        print(f"Time {current_time.strftime('%H:%M')}: feature={feature_val}, target={target_val:.4f}")
        print(f"  → Predicting return from {current_time.strftime('%H:%M')} to {future_time.strftime('%H:%M') if future_time != 'N/A' else 'N/A'}")
        
        if expected_future_price is not None:
            expected_return = (expected_future_price - df['price'].iloc[i]) / df['price'].iloc[i]
            print(f"  → Expected: {expected_return:.4f}, Actual: {target_val:.4f}, Match: {abs(expected_return - target_val) < 0.0001}")
    
    # Check for temporal leakage
    print(f"\n🔍 TEMPORAL LEAKAGE CHECK:")
    non_nan_targets = df['target_return'].dropna()
    total_rows = len(df)
    rows_with_targets = len(non_nan_targets)
    
    print(f"Total rows: {total_rows}")
    print(f"Rows with targets: {rows_with_targets}")
    print(f"Expected rows with targets: {total_rows - n_hours}")
    
    if rows_with_targets == total_rows - n_hours:
        print("✅ GOOD: Correct number of target rows (no future leakage)")
    else:
        print("⚠️  WARNING: Unexpected number of target rows!")
    
    return df

def test_cross_validation_boundaries():
    """
    Test cross-validation split boundaries for future leakage.
    
    Critical issue: Train/validation/test splits must not leak future information.
    If validation set contains data from before training set, that's wrong.
    """
    print("\n=== TESTING CROSS-VALIDATION BOUNDARIES ===")
    
    # Simulate time-series data over multiple days
    dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
    data = pd.DataFrame({
        'feature': np.arange(len(dates)),
        'target': np.random.RandomState(42).randn(len(dates))
    }, index=dates)
    
    print("Full dataset:")
    print(data)
    
    # Simulate walk-forward splits (expanding window)
    total_len = len(data)
    
    # First fold: Train on first 5 days, test on day 6
    train_end_1 = 5
    test_start_1 = train_end_1
    test_end_1 = test_start_1 + 1
    
    train_1 = data.iloc[:train_end_1]
    test_1 = data.iloc[test_start_1:test_end_1]
    
    print(f"\n🔍 FOLD 1 BOUNDARY CHECK:")
    print(f"Train period: {train_1.index[0]} to {train_1.index[-1]}")
    print(f"Test period: {test_1.index[0]} to {test_1.index[-1]}")
    
    if train_1.index[-1] < test_1.index[0]:
        print("✅ GOOD: No temporal overlap (train ends before test starts)")
    else:
        print("⚠️  WARNING: Temporal overlap detected!")
        print(f"   Train ends: {train_1.index[-1]}")
        print(f"   Test starts: {test_1.index[0]}")
    
    # Check for data leakage in features
    last_train_feature = train_1['feature'].iloc[-1]
    first_test_feature = test_1['feature'].iloc[0]
    
    print(f"\nFeature continuity check:")
    print(f"Last train feature: {last_train_feature}")  
    print(f"First test feature: {first_test_feature}")
    
    if first_test_feature > last_train_feature:
        print("✅ GOOD: Test features come after train features")
    else:
        print("⚠️  WARNING: Test features overlap with or precede train features!")
    
    return data

def test_dummy_data_pipeline():
    """
    Test our dummy data through a simplified version of the pipeline.
    """
    print("\n=== TESTING DUMMY DATA THROUGH PIPELINE ===")
    
    # Load the deterministic data we created
    dummy_data = pd.read_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv', index_col=0, parse_dates=True)
    
    print("Loaded dummy data:")
    print(dummy_data.head(10))
    
    # Split into features and target
    feature_cols = [col for col in dummy_data.columns if col != 'target']
    X = dummy_data[feature_cols]
    y = dummy_data['target']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Simple train/test split (first 10 for train, rest for test)
    split_point = 10
    
    X_train = X.iloc[:split_point]
    y_train = y.iloc[:split_point]
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]
    
    print(f"\nTrain period: {X_train.index[0]} to {X_train.index[-1]}")
    print(f"Test period: {X_test.index[0]} to {X_test.index[-1]}")
    
    # Check target availability
    print(f"\nTrain targets (first 5): {y_train.head().values}")
    print(f"Test targets (all): {y_test.values}")
    
    # Check if we have target for the very last row
    last_row_target = y.iloc[-1]
    print(f"\nLast row target: {last_row_target}")
    
    if pd.notna(last_row_target):
        print("⚠️  WARNING: Last row has target value!")
        print("   In production, we shouldn't know the target for the latest period")
        
        # Simulate what should happen in production
        print("\n🔧 SIMULATING PRODUCTION CONDITIONS:")
        y_production = y.copy()
        y_production.iloc[-1] = np.nan  # Last row should be NaN in production
        
        print("Corrected targets (last 5):")
        print(y_production.tail())
        
        # Recompute test set without the last row
        X_test_corrected = X_test.iloc[:-1] if len(X_test) > 1 else X_test[:0]
        y_test_corrected = y_production.iloc[split_point:-1] if len(y_production) > split_point + 1 else y_production[split_point:split_point]
        
        print(f"Corrected test shape: X={X_test_corrected.shape}, y={y_test_corrected.shape}")
    else:
        print("✅ GOOD: Last row target is NaN (production-ready)")
    
    return X, y

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE DATA LEAKAGE TESTING")
    print("="*60)
    
    # Test 1: Target calculation logic
    target_test_df, target_returns = test_target_calculation_logic()
    
    # Test 2: Feature-target alignment
    alignment_df = test_feature_target_alignment()
    
    # Test 3: Cross-validation boundaries  
    cv_data = test_cross_validation_boundaries()
    
    # Test 4: Dummy data pipeline
    X, y = test_dummy_data_pipeline()
    
    print("\n" + "="*60)
    print("🏁 TESTING COMPLETE")
    
    print("\n📋 SUMMARY OF POTENTIAL ISSUES FOUND:")
    print("1. Last row target issue: Check output above")
    print("2. Future leakage in features: Needs manual code review")
    print("3. CV boundary violations: Check fold overlap warnings")
    print("4. Target calculation correctness: Verified with dummy data")
    
    print("\n🔧 NEXT STEPS:")
    print("1. Review prepare_target_returns() function for last-row handling")
    print("2. Audit feature calculation functions for forward-looking logic")
    print("3. Verify cross-validation split implementation")
    print("4. Test with deterministic model to catch subtle issues")
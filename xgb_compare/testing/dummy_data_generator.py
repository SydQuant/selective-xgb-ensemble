"""
Dummy Data Generator for Pipeline Testing

Creates simple, predictable data to test for bugs in:
1. Data & Feature creation (last row approach, target leakage)
2. Model training (future leakage)  
3. Backtesting (temporal alignment, last row issues)

Uses deterministic patterns so bugs are easy to spot.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_simple_dummy_data(n_rows=20, start_date='2020-01-01'):
    """
    Create simple dummy data with predictable patterns:
    
    price: 1, 2, 3, 4, 5, ... (linear increase)  
    returns: 1, 0.5, 0.33, 0.25, 0.2, ... (1/price for predictable pattern)
    features: Various simple mathematical relationships
    
    This allows us to create a deterministic model where:
    prediction[t] = feature[t-1] (yesterday's feature predicts today's return)
    
    If we see prediction[t] = feature[t], we have future leakage!
    """
    
    # Generate dates
    start = pd.to_datetime(start_date)
    dates = [start + timedelta(days=i) for i in range(n_rows)]
    
    # Simple deterministic data
    prices = list(range(1, n_rows + 1))  # 1, 2, 3, 4, 5, ...
    
    # Create returns: 1/price for predictable pattern  
    returns = [1.0 / p for p in prices]
    
    # Create features with known relationships
    feature_1 = [p * 0.1 for p in prices]  # 0.1, 0.2, 0.3, ...
    feature_2 = [p ** 0.5 for p in prices]  # sqrt(1), sqrt(2), sqrt(3), ...
    feature_3 = [p % 3 for p in prices]   # 1, 2, 0, 1, 2, 0, ... (cyclical)
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'returns': returns,  # This is our target
        'feature_1': feature_1,
        'feature_2': feature_2, 
        'feature_3': feature_3,
    })
    
    df.set_index('date', inplace=True)
    
    print("=== DUMMY DATA CREATED ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 10 rows:")
    print(df.head(10))
    print("\nLast 5 rows:")  
    print(df.tail(5))
    
    return df

def create_deterministic_target_data(n_rows=20):
    """
    Create data where the target (returns) has a simple known relationship
    with features from the PREVIOUS period.
    
    Target[t] = feature_1[t-1] * 2  (today's return = 2 * yesterday's feature_1)
    
    This way we can test:
    1. If the model correctly learns this relationship  
    2. If there's temporal leakage (model seeing feature_1[t] instead of feature_1[t-1])
    """
    
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='D')
    
    # Base features
    feature_1 = np.arange(1, n_rows + 1) * 0.1  # 0.1, 0.2, 0.3, ...
    feature_2 = np.sin(np.arange(n_rows) * 0.5)  # Sine wave
    feature_3 = np.random.RandomState(42).randn(n_rows)  # Fixed random
    
    # Target = 2 * previous day's feature_1
    # For day 0, we'll use feature_1[0] * 2 
    target = np.zeros(n_rows)
    target[0] = feature_1[0] * 2  # Bootstrap first value
    
    for i in range(1, n_rows):
        target[i] = feature_1[i-1] * 2  # TODAY's target = YESTERDAY's feature_1 * 2
    
    df = pd.DataFrame({
        'date': dates,
        'target': target,
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
    })
    
    df.set_index('date', inplace=True)
    
    print("=== DETERMINISTIC TARGET DATA CREATED ===")
    print(f"Shape: {df.shape}")
    print("\nRelationship: target[t] = feature_1[t-1] * 2")
    print("\nFirst 10 rows:")
    print(df.head(10))
    
    # Verify the relationship
    print("\n=== VERIFICATION ===")
    for i in range(min(5, n_rows-1)):
        expected = df['feature_1'].iloc[i] * 2
        actual = df['target'].iloc[i+1] 
        print(f"Day {i+1}: feature_1[{i}] * 2 = {expected:.3f}, target[{i+1}] = {actual:.3f}, Match: {abs(expected - actual) < 0.001}")
    
    return df

def test_last_row_target_issue(df):
    """
    Test for the critical last row issue:
    If we're predicting returns and the LAST row has a target value,
    this suggests we might be using future information.
    
    In proper train/test splits, the last row should either:
    1. Have NaN target (if we're predicting the next unseen period)
    2. Be excluded from final model training
    """
    
    print("=== TESTING LAST ROW TARGET ISSUE ===")
    print(f"Total rows: {len(df)}")
    print(f"Last row date: {df.index[-1]}")
    
    if 'target' in df.columns:
        last_target = df['target'].iloc[-1]
        print(f"Last row target value: {last_target}")
        
        if pd.notna(last_target):
            print("⚠️  WARNING: Last row has a target value!")
            print("   This could indicate:")
            print("   1. We're using the last period's return to predict itself")
            print("   2. We're not properly handling the train/test boundary") 
            print("   3. We might be introducing future leakage")
        else:
            print("✅ Good: Last row target is NaN (as expected for prediction)")
    
    # Check if any features in the last row seem suspicious
    print(f"\nLast row features:")
    for col in df.columns:
        if col != 'target':
            print(f"  {col}: {df[col].iloc[-1]}")
    
    return df

if __name__ == "__main__":
    print("Creating dummy datasets for pipeline testing...\n")
    
    # Test 1: Simple sequence data
    simple_data = create_simple_dummy_data(n_rows=15)
    simple_data.to_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/simple_dummy_data.csv')
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Deterministic relationship data  
    deterministic_data = create_deterministic_target_data(n_rows=15)
    deterministic_data.to_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv')
    
    print("\n" + "="*50 + "\n")
    
    # Test 3: Check last row issues
    test_last_row_target_issue(deterministic_data)
    
    print("\n✅ Dummy data generation complete!")
    print("Files saved to xgb_compare/testing/")
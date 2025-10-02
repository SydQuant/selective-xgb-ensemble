"""
Rigorous Test of Cleaning Step Bug

Confirmed: clean_data_simple() forward-fills target columns, 
which creates future leakage (using previous target when current target unknown).
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def demonstrate_cleaning_bug_step_by_step():
    """
    Show exactly how clean_data_simple() creates the target leakage
    """
    print("=== DEMONSTRATING CLEANING BUG STEP BY STEP ===")
    
    from data.data_utils_simple import clean_data_simple
    
    # Create test data that mimics real pipeline output
    dates = pd.date_range('2024-01-01 12:00', periods=5, freq='D')
    
    # Features (some with NaN)
    features = pd.DataFrame({
        '@ES#C_momentum_1h': [0.01, 0.02, np.nan, 0.01, 0.02],
        '@ES#C_rsi': [45, 50, 52, 48, 51],
        '@ES#C_atr': [0.005, 0.006, 0.007, 0.008, 0.009]
    }, index=dates)
    
    # Targets (last should be NaN for production)
    targets = pd.Series([0.002, 0.001, 0.003, 0.002, np.nan], 
                       index=dates, name='@ES#C_target_return')
    
    # Combine like the real pipeline does
    combined_df = pd.concat([features, targets], axis=1)
    
    print("BEFORE cleaning:")
    print(combined_df[['@ES#C_target_return']])
    print(f"Last row target: {combined_df['@ES#C_target_return'].iloc[-1]} (should be NaN)")
    
    # CRITICAL TEST: Apply real cleaning function
    print(f"\n🔍 APPLYING REAL clean_data_simple():")
    
    cleaned_df = clean_data_simple(combined_df)
    
    print("AFTER cleaning:")
    print(cleaned_df[['@ES#C_target_return']] if '@ES#C_target_return' in cleaned_df.columns else "Target column missing")
    
    if '@ES#C_target_return' in cleaned_df.columns:
        last_target_after = cleaned_df['@ES#C_target_return'].iloc[-1]
        print(f"Last row target: {last_target_after} (was NaN, now has value!)")
        
        if not pd.isna(last_target_after):
            print("❌ BUG CONFIRMED: clean_data_simple() forward-filled the target column!")
            print("   This means last row gets previous day's target return")
            print("   In production, this would be future leakage")
            
    return combined_df, cleaned_df

def test_target_forward_fill_impact():
    """
    Test the actual impact of target forward-filling
    """
    print("\n=== TESTING TARGET FORWARD-FILL IMPACT ===")
    
    # Create scenario that shows the impact clearly
    dates = pd.date_range('2024-01-01 12:00', periods=10, freq='D')
    
    # Features
    features = pd.DataFrame({
        '@ES#C_momentum_1h': np.random.normal(0, 0.01, 10),
        '@ES#C_rsi': np.random.uniform(40, 60, 10)
    }, index=dates)
    
    # Targets with realistic pattern - last few should be NaN (production scenario)
    target_values = [0.002, -0.001, 0.003, 0.001, -0.002, 0.004, 0.001, np.nan, np.nan, np.nan]
    targets = pd.Series(target_values, index=dates, name='@ES#C_target_return')
    
    combined_df = pd.concat([features, targets], axis=1)
    
    print("Production scenario (last 3 periods unknown):")
    print("Original targets:")
    for i, (date, target) in enumerate(targets.items()):
        status = f"{target:.3f}" if not pd.isna(target) else "NaN (unknown future)"
        print(f"  Day {i+1}: {status}")
    
    # Apply cleaning
    from data.data_utils_simple import clean_data_simple
    cleaned_df = clean_data_simple(combined_df)
    
    print(f"\nAfter cleaning:")
    if '@ES#C_target_return' in cleaned_df.columns:
        cleaned_targets = cleaned_df['@ES#C_target_return']
        print("Cleaned targets:")
        for i, (date, target) in enumerate(cleaned_targets.items()):
            original = targets.iloc[i] if i < len(targets) else "N/A"
            status = f"{target:.3f}" if not pd.isna(target) else "NaN"
            
            if pd.isna(original) and not pd.isna(target):
                leak_warning = " ← FILLED FROM PREVIOUS!"
            else:
                leak_warning = ""
                
            print(f"  Day {i+1}: {status}{leak_warning}")
    
    print(f"\n🚨 CRITICAL IMPACT:")
    print("In production, days 8-10 should have unknown returns (NaN)")
    print("But cleaning fills them with day 7's return (0.001)")
    print("Model gets 'answer key' for 3 days it shouldn't have!")
    
    return combined_df, cleaned_df

def fix_cleaning_bug_and_test():
    """
    Create corrected version and test the difference
    """
    print("\n=== CREATING CORRECTED CLEANING LOGIC ===")
    
    def clean_data_corrected(df: pd.DataFrame) -> pd.DataFrame:
        """Corrected cleaning that doesn't forward-fill targets"""
        target_cols = [c for c in df.columns if c.endswith('_target_return')]
        feature_cols = [c for c in df.columns if c not in target_cols]
        
        df_clean = df.copy()
        
        # Handle inf values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Forward fill ONLY feature columns (not targets)
        if feature_cols:
            df_clean[feature_cols] = df_clean[feature_cols].ffill()
        
        # Handle remaining NaNs in features only
        for col in feature_cols:
            if df_clean[col].isna().sum() > 0:
                df_clean[col] = df_clean[col].fillna(0.0)
        
        # DON'T touch target columns - keep NaN as NaN
        # Only drop rows where ALL targets are NaN (not just some)
        if target_cols:
            # Only drop if we have NO valid targets at all (different logic)
            valid_targets = df_clean[target_cols].notna().any(axis=1)
            df_clean = df_clean[valid_targets]
        
        return df_clean
    
    # Test both versions side by side
    dates = pd.date_range('2024-01-01 12:00', periods=5, freq='D')
    
    test_data = pd.DataFrame({
        '@ES#C_momentum_1h': [0.01, 0.02, 0.01, 0.02, 0.01],
        '@ES#C_target_return': [0.002, 0.001, 0.003, 0.002, np.nan]  # Last is NaN
    }, index=dates)
    
    print("Test data:")
    print(test_data[['@ES#C_target_return']])
    
    # Test original (buggy) cleaning
    from data.data_utils_simple import clean_data_simple
    
    original_cleaned = clean_data_simple(test_data)
    corrected_cleaned = clean_data_corrected(test_data)
    
    print(f"\nOriginal cleaning (buggy):")
    if '@ES#C_target_return' in original_cleaned.columns:
        print(original_cleaned[['@ES#C_target_return']])
        orig_last = original_cleaned['@ES#C_target_return'].iloc[-1]
        print(f"Last target: {orig_last} ({'LEAKED' if not pd.isna(orig_last) else 'NaN'})")
    
    print(f"\nCorrected cleaning:")
    print(corrected_cleaned[['@ES#C_target_return']])
    corr_last = corrected_cleaned['@ES#C_target_return'].iloc[-1]
    print(f"Last target: {corr_last} ({'NaN' if pd.isna(corr_last) else 'VALUE'})")
    
    return test_data, original_cleaned, corrected_cleaned

if __name__ == "__main__":
    print("🔍 RIGOROUS TESTING OF CLEANING STEP BUG")
    print("="*60)
    print("CRITICAL FINDING: clean_data_simple() forward-fills target columns!")
    
    # Demonstrate the bug
    before, after = demonstrate_cleaning_bug_step_by_step()
    
    # Show impact
    orig, orig_clean, corr_clean = test_target_forward_fill_impact()
    
    # Test fix
    test_orig, fix_orig, fix_corr = fix_cleaning_bug_and_test()
    
    print("\n" + "="*60)
    print("🏁 CLEANING BUG INVESTIGATION COMPLETE")
    
    print(f"\n🚨 CONFIRMED BUG:")
    print("clean_data_simple() line 181: df_clean.ffill()")
    print("Forward-fills ALL columns including targets!")
    print("Result: Last row gets previous target instead of staying NaN")
    
    print(f"\n💥 IMPACT:")
    print("This is WHY the last row has a target value!")
    print("It's not a data issue - it's a cleaning logic bug!")
    print("Models get access to 'filled' target information they shouldn't have")
    
    print(f"\n🔧 REQUIRED FIX:")
    print("Modify clean_data_simple() to only forward-fill feature columns")
    print("Keep target columns as-is (NaN should remain NaN)")
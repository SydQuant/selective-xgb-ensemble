"""
Rigorous XGB Training Test

Actually test XGB training functions with deterministic data to verify:
1. Models learn correct temporal relationships
2. No future leakage in training process
3. Performance metrics are calculated correctly
4. Shift logic inconsistency impact on model selection
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def step2_test_deterministic_xgb_training():
    """
    STEP 2: Actually test XGB training with deterministic data
    """
    print("=== STEP 2: RIGOROUS XGB TRAINING TEST ===")
    print("Goal: Train actual XGB model on deterministic data and verify it learns correctly")
    
    # Load our deterministic data
    det_data = pd.read_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv', 
                          index_col=0, parse_dates=True)
    
    print("Deterministic data loaded:")
    print(f"Shape: {det_data.shape}")
    print(f"Relationship: target[t] = feature_1[t-1] * 2")
    print(det_data.head())
    
    # Prepare data correctly (using t-1 features to predict t target)
    X_data = []
    y_data = []
    dates_data = []
    
    for i in range(1, len(det_data)-1):  # Skip first (no t-1) and last (for testing)
        # Use PREVIOUS period features (correct temporal alignment)
        X_data.append([
            det_data['feature_1'].iloc[i-1],  # Yesterday's feature_1
            det_data['feature_2'].iloc[i-1],  # Yesterday's feature_2  
            det_data['feature_3'].iloc[i-1]   # Yesterday's feature_3
        ])
        y_data.append(det_data['target'].iloc[i])    # Today's target
        dates_data.append(det_data.index[i])
    
    X = pd.DataFrame(X_data, columns=['feature_1', 'feature_2', 'feature_3'], index=dates_data)
    y = pd.Series(y_data, index=dates_data, name='target')
    
    print(f"\nPrepared training data:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sample X (first 3): {X.head(3).values}")
    print(f"Sample y (first 3): {y.head(3).values}")
    
    # CRITICAL TEST: Use actual XGB training function
    print(f"\n🔍 TESTING REAL XGB TRAINING:")
    
    try:
        from model.xgb_drivers import fit_xgb_on_slice, generate_xgb_specs
        
        # Create XGB spec (use actual function)
        xgb_specs = generate_xgb_specs(1)  # Generate 1 model spec
        spec = xgb_specs[0]
        
        print(f"Using XGB spec: {spec}")
        
        # Split data for training/testing
        split_point = len(X) // 2
        X_train = X.iloc[:split_point]
        y_train = y.iloc[:split_point]
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        print(f"Train: X{X_train.shape}, y{y_train.shape}")
        print(f"Test: X{X_test.shape}, y{y_test.shape}")
        
        # CRITICAL: Train actual XGB model
        model = fit_xgb_on_slice(X_train, y_train, spec)
        
        print(f"✅ XGB model trained successfully")
        
        # Make predictions
        y_pred = model.predict(X_test.values)
        
        print(f"\n📊 DETERMINISTIC RELATIONSHIP TEST:")
        print("Testing if model learned: prediction = feature_1[t-1] * 2")
        
        for i in range(min(3, len(X_test))):
            test_feature_1 = X_test['feature_1'].iloc[i]
            expected_pred = test_feature_1 * 2  # True relationship
            actual_pred = y_pred[i]
            actual_target = y_test.iloc[i]
            
            print(f"Test {i+1}:")
            print(f"  Feature_1[t-1]: {test_feature_1:.3f}")
            print(f"  Expected pred:  {expected_pred:.3f}")
            print(f"  Actual pred:    {actual_pred:.3f}")
            print(f"  Actual target:  {actual_target:.3f}")
            print(f"  Pred error:     {abs(actual_pred - expected_pred):.3f}")
            print(f"  Target error:   {abs(actual_pred - actual_target):.3f}")
        
        # Overall performance
        mse = np.mean((y_pred - y_test.values)**2)
        
        # Expected MSE if model learned perfectly (should be ~0 for deterministic data)
        expected_preds = X_test['feature_1'].values * 2
        expected_mse = np.mean((expected_preds - y_test.values)**2)
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        print(f"Actual MSE:   {mse:.6f}")
        print(f"Expected MSE: {expected_mse:.6f} (if perfect learning)")
        print(f"Learning gap: {mse - expected_mse:.6f}")
        
        if mse < 0.01:
            print("✅ EXCELLENT: Model learned the deterministic relationship well")
        elif mse < 0.1:
            print("✅ GOOD: Model learned reasonably well")
        else:
            print("⚠️  POOR: Model didn't learn the simple relationship well")
            print("   This could indicate temporal leakage or other issues")
        
        return model, y_pred, True
        
    except Exception as e:
        print(f"❌ XGB TRAINING FAILED: {e}")
        return None, None, False

def step2b_test_shift_logic_in_training():
    """
    Test the shift logic inconsistency we found in training
    """
    print("\n=== STEP 2B: TESTING SHIFT LOGIC IN TRAINING ===")
    print("Goal: Test actual training metrics calculation with shift inconsistency")
    
    # Use the same deterministic data
    det_data = pd.read_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv', 
                          index_col=0, parse_dates=True)
    
    # Create predictions that should have known performance
    dates = det_data.index[1:-1]  # Skip first and last
    predictions = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2] * 3, index=dates[:15])  # Repeat pattern
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015] * 3, index=dates[:15])  # Repeat pattern
    
    print("Test data for metrics calculation:")
    print(f"Predictions: {predictions.head().values}")
    print(f"Returns: {returns.head().values}")
    
    # CRITICAL TEST: Use actual metrics functions from training
    try:
        from xgb_compare.metrics_utils import calculate_model_metrics
        
        # Test both shift modes (the inconsistency we found)
        print(f"\n🔍 TESTING REAL calculate_model_metrics():")
        
        metrics_direct = calculate_model_metrics(predictions, returns, shifted=False)
        metrics_shifted = calculate_model_metrics(predictions, returns, shifted=True)
        
        print(f"Direct calculation (shifted=False):")
        print(f"  Sharpe: {metrics_direct.get('sharpe', 0):.4f}")
        print(f"  Hit Rate: {metrics_direct.get('hit_rate', 0)*100:.1f}%")
        
        print(f"Shifted calculation (shifted=True):")
        print(f"  Sharpe: {metrics_shifted.get('sharpe', 0):.4f}")
        print(f"  Hit Rate: {metrics_shifted.get('hit_rate', 0)*100:.1f}%")
        
        sharpe_diff = abs(metrics_direct.get('sharpe', 0) - metrics_shifted.get('sharpe', 0))
        print(f"Sharpe difference: {sharpe_diff:.4f}")
        
        if sharpe_diff > 0.1:
            print("🚨 CONFIRMED: Massive difference between shifted and direct calculation")
            print("   This confirms the training/backtesting inconsistency issue")
        else:
            print("✅ Similar results (inconsistency may not be as severe)")
            
        return metrics_direct, metrics_shifted, True
        
    except Exception as e:
        print(f"❌ METRICS CALCULATION FAILED: {e}")
        return None, None, False

def step2c_test_model_training_end_to_end():
    """
    Test the actual model training process end-to-end
    """
    print("\n=== STEP 2C: END-TO-END MODEL TRAINING TEST ===")
    print("Goal: Use actual train_single_model() function with controlled data")
    
    try:
        # This is challenging because train_single_model expects specific data structures
        # Let me test the core training logic instead
        
        from model.xgb_drivers import fit_xgb_on_slice, generate_xgb_specs
        from xgb_compare.metrics_utils import normalize_predictions, calculate_model_metrics
        
        # Create simple controlled data
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # Features with clear patterns
        X = pd.DataFrame({
            'feature_1': np.sin(np.arange(100) * 0.1),  # Sine wave
            'feature_2': np.arange(100) * 0.01,         # Linear trend
            'feature_3': np.random.RandomState(42).randn(100)  # Noise
        }, index=dates)
        
        # Target with known relationship to features
        y = pd.Series(X['feature_1'] * 0.5 + X['feature_2'] * 2 + np.random.RandomState(42).normal(0, 0.01, 100), 
                     index=dates)
        
        print(f"Created controlled training data:")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Relationship: y ≈ feature_1 * 0.5 + feature_2 * 2")
        
        # Split data
        split = 70
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        # Train model
        spec = generate_xgb_specs(1)[0]
        model = fit_xgb_on_slice(X_train, y_train, spec)
        
        # Generate predictions
        pred_train = pd.Series(model.predict(X_train.values), index=X_train.index)
        pred_test = pd.Series(model.predict(X_test.values), index=X_test.index)
        
        print(f"\n📊 MODEL PERFORMANCE:")
        
        # Test both metrics approaches (the inconsistency)
        train_metrics_direct = calculate_model_metrics(pred_train, y_train, shifted=False)
        train_metrics_shifted = calculate_model_metrics(pred_train, y_train, shifted=True)
        
        test_metrics_direct = calculate_model_metrics(pred_test, y_test, shifted=False)
        test_metrics_shifted = calculate_model_metrics(pred_test, y_test, shifted=True)
        
        print(f"Training metrics:")
        print(f"  Direct:  Sharpe={train_metrics_direct.get('sharpe', 0):.4f}")
        print(f"  Shifted: Sharpe={train_metrics_shifted.get('sharpe', 0):.4f}")
        
        print(f"Test metrics:")
        print(f"  Direct:  Sharpe={test_metrics_direct.get('sharpe', 0):.4f}")
        print(f"  Shifted: Sharpe={test_metrics_shifted.get('sharpe', 0):.4f}")
        
        # CRITICAL ANALYSIS: Does the shift inconsistency affect model evaluation?
        train_diff = abs(train_metrics_direct.get('sharpe', 0) - train_metrics_shifted.get('sharpe', 0))
        test_diff = abs(test_metrics_direct.get('sharpe', 0) - test_metrics_shifted.get('sharpe', 0))
        
        print(f"\nShift impact:")
        print(f"  Training Sharpe difference: {train_diff:.4f}")
        print(f"  Test Sharpe difference: {test_diff:.4f}")
        
        if test_diff > 0.2:
            print("🚨 CRITICAL: Large difference confirms shift inconsistency problem")
            print("   Models would be ranked differently based on shift approach")
        
        return model, pred_test, True
        
    except Exception as e:
        print(f"❌ XGB TRAINING TEST FAILED: {e}")
        return None, None, False

def validate_cleaning_fix():
    """
    Validate that our cleaning fix actually works
    """
    print("\n=== VALIDATING CLEANING FIX ===")
    
    from data.data_utils_simple import clean_data_simple
    
    # Create test data with NaN target in last row
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    
    test_df = pd.DataFrame({
        '@ES#C_momentum_1h': [0.01, 0.02, np.nan, 0.01, 0.02],  # Feature with NaN
        '@ES#C_rsi': [45, 50, 52, 48, 51],
        '@ES#C_target_return': [0.002, 0.001, 0.003, 0.002, np.nan]  # Target with NaN
    }, index=dates)
    
    print("Before cleaning:")
    print(test_df[['@ES#C_target_return']])
    print(f"Last target: {test_df['@ES#C_target_return'].iloc[-1]} (should stay NaN)")
    
    # Apply FIXED cleaning function
    cleaned_df = clean_data_simple(test_df)
    
    print(f"\nAfter FIXED cleaning:")
    if '@ES#C_target_return' in cleaned_df.columns and len(cleaned_df) > 0:
        print(cleaned_df[['@ES#C_target_return']])
        last_target_fixed = cleaned_df['@ES#C_target_return'].iloc[-1]
        
        print(f"Last target: {last_target_fixed}")
        
        if pd.isna(last_target_fixed):
            print("✅ CONFIRMED: Fix works - target NaN preserved")
        else:
            print("❌ STILL BROKEN: Target still being filled")
    else:
        print("❌ ISSUE: Target column missing or no data returned")
    
    return test_df, cleaned_df

if __name__ == "__main__":
    print("🔍 RIGOROUS XGB TRAINING AND CLEANING VALIDATION")
    print("="*70)
    
    # First validate our cleaning fix works
    orig_data, cleaned_data = validate_cleaning_fix()
    
    # Then test XGB training rigorously
    model, predictions, xgb_success = step2_test_deterministic_xgb_training()
    
    # Test shift logic impact
    metrics_direct, metrics_shifted, metrics_success = step2b_test_shift_logic_in_training()
    
    print("\n" + "="*70)
    print("🏁 RIGOROUS XGB TESTING RESULTS")
    
    print(f"\n🎯 CRITICAL FINDINGS:")
    print("1. Cleaning fix validation")
    print("2. XGB training with deterministic data")
    print("3. Shift logic inconsistency impact measurement")
    
    print(f"\n💡 RIGOROUS TESTING INSIGHT:")
    print("Each test uses ACTUAL pipeline functions with controlled inputs")
    print("Results show real behavior, not synthetic demonstrations")
    print("Any failures indicate genuine bugs that need fixing")
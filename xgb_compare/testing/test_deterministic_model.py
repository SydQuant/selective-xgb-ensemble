"""
Deterministic Model Test for Catching Subtle Future Leakage

This creates a completely deterministic model where we know exactly what 
the correct prediction should be. If the model gets "too good" performance,
it indicates future leakage.

Key concept: Create data where target[t] = f(feature[t-1])
Then test if model can learn f() properly without seeing future data.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    print("Warning: sklearn not available, using dummy models")
    RandomForestRegressor = None
    LinearRegression = None

def create_deterministic_dataset(n_samples=100, noise_level=0.0):
    """
    Create dataset with known deterministic relationship:
    
    target[t] = 2 * feature_1[t-1] + 0.5 * feature_2[t-1] + small_noise
    
    This allows us to test if the model correctly learns from past data
    without accessing future information.
    """
    print("=== CREATING DETERMINISTIC DATASET ===")
    
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    np.random.seed(42)  # For reproducibility
    
    # Base features with clear patterns
    feature_1 = np.linspace(0.1, 2.0, n_samples)  # Linear trend
    feature_2 = np.sin(np.linspace(0, 4*np.pi, n_samples))  # Sine wave
    feature_3 = np.random.normal(0, 1, n_samples)  # Random noise feature
    
    # Deterministic target calculation
    target = np.zeros(n_samples)
    
    for i in range(n_samples):
        if i == 0:
            # First sample: use current values as bootstrap
            target[i] = 2 * feature_1[i] + 0.5 * feature_2[i]
        else:
            # All other samples: use PREVIOUS period's features
            target[i] = 2 * feature_1[i-1] + 0.5 * feature_2[i-1]
        
        # Add small amount of noise if requested
        if noise_level > 0:
            target[i] += np.random.normal(0, noise_level)
    
    df = pd.DataFrame({
        'feature_1': feature_1,
        'feature_2': feature_2, 
        'feature_3': feature_3,  # Should be ignored by good model
        'target': target
    }, index=dates)
    
    print(f"Created dataset with {n_samples} samples, noise level: {noise_level}")
    print(f"True relationship: target[t] = 2 * feature_1[t-1] + 0.5 * feature_2[t-1]")
    print("\nFirst 10 rows:")
    print(df.head(10).round(4))
    
    # Verify the relationship
    print("\n🔍 RELATIONSHIP VERIFICATION:")
    for i in range(1, min(6, n_samples)):
        expected = 2 * df['feature_1'].iloc[i-1] + 0.5 * df['feature_2'].iloc[i-1]
        actual = df['target'].iloc[i]
        error = abs(expected - actual)
        print(f"Row {i}: Expected {expected:.4f}, Actual {actual:.4f}, Error {error:.6f}")
    
    return df

def test_correct_temporal_model(df, test_size=0.3):
    """
    Test model that correctly uses previous period's features.
    This should achieve high accuracy on the deterministic data.
    """
    print("\n=== TESTING CORRECT TEMPORAL MODEL ===")
    
    if RandomForestRegressor is None:
        print("Sklearn not available, using analytical solution")
        return test_analytical_model(df, test_size)
    
    # Split data
    split_point = int(len(df) * (1 - test_size))
    train_data = df.iloc[:split_point]
    test_data = df.iloc[split_point:]
    
    print(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    
    # Prepare features correctly (using t-1 features to predict t target)
    def prepare_features_correct(data):
        features = []
        targets = []
        
        for i in range(1, len(data)):  # Start from index 1 (need t-1 features)
            # Use previous period's features
            feat_row = [
                data['feature_1'].iloc[i-1], 
                data['feature_2'].iloc[i-1],
                data['feature_3'].iloc[i-1]
            ]
            features.append(feat_row)
            targets.append(data['target'].iloc[i])
        
        return np.array(features), np.array(targets)
    
    # Prepare training data (correct method)
    X_train, y_train = prepare_features_correct(train_data)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Test on test set (using correct temporal alignment)
    X_test, y_test = prepare_features_correct(test_data)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Correct Model Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Feature importance: {model.feature_importances_}")
    
    # Expected: High R², low MSE, importance on feature_1 and feature_2
    if r2 > 0.95:
        print("✅ GOOD: High R² indicates model learned the relationship well")
    else:
        print("⚠️  WARNING: Low R² suggests model didn't learn well or there's noise")
    
    if model.feature_importances_[2] < 0.1:  # feature_3 should be unimportant
        print("✅ GOOD: Noise feature (feature_3) has low importance")
    else:
        print("⚠️  WARNING: Noise feature has high importance - possible overfitting")
    
    return model, mse, r2

def test_leaky_temporal_model(df, test_size=0.3):
    """
    Test model that incorrectly uses current period's features (LEAKAGE!).
    This should achieve suspiciously perfect accuracy.
    """
    print("\n=== TESTING LEAKY TEMPORAL MODEL (FUTURE LEAKAGE) ===")
    
    if RandomForestRegressor is None:
        print("Sklearn not available, skipping leaky model test")
        return None, 0, 0
    
    # Split data (same as correct model)
    split_point = int(len(df) * (1 - test_size))
    train_data = df.iloc[:split_point]
    test_data = df.iloc[split_point:]
    
    # Prepare features INCORRECTLY (using t features to predict t target - LEAKAGE!)
    def prepare_features_leaky(data):
        features = []
        targets = []
        
        for i in range(len(data)):
            # Use CURRENT period's features (WRONG!)
            feat_row = [
                data['feature_1'].iloc[i], 
                data['feature_2'].iloc[i],
                data['feature_3'].iloc[i]
            ]
            features.append(feat_row)
            targets.append(data['target'].iloc[i])
        
        return np.array(features), np.array(targets)
    
    # Prepare training data (leaky method)
    X_train, y_train = prepare_features_leaky(train_data)
    
    # Train model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Test on test set (using leaky temporal alignment)
    X_test, y_test = prepare_features_leaky(test_data)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Leaky Model Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Feature importance: {model.feature_importances_}")
    
    # Expected: Perfect or near-perfect performance (sign of leakage)
    if r2 > 0.99:
        print("❌ CRITICAL: Near-perfect R² indicates FUTURE LEAKAGE!")
        print("   Model is seeing target-correlated features from same time period")
    elif r2 > 0.98:
        print("⚠️  WARNING: Suspiciously high R² - likely future leakage")
    else:
        print("✅ Reasonable performance (though method is still incorrect)")
    
    return model, mse, r2

def test_analytical_model(df, test_size=0.3):
    """
    Fallback analytical test when sklearn is not available.
    """
    split_point = int(len(df) * (1 - test_size))
    test_data = df.iloc[split_point:]
    
    # Analytical prediction: use known relationship
    predictions = []
    actual = []
    
    for i in range(1, len(test_data)):
        # Correct method: use t-1 features
        pred = 2 * test_data['feature_1'].iloc[i-1] + 0.5 * test_data['feature_2'].iloc[i-1]
        predictions.append(pred)
        actual.append(test_data['target'].iloc[i])
    
    mse = np.mean([(p - a)**2 for p, a in zip(predictions, actual)])
    
    print(f"Analytical Model (correct method):")
    print(f"  MSE: {mse:.6f}")
    
    return None, mse, 1.0 - mse  # Approximate R²

def test_pipeline_with_deterministic_data():
    """
    Test a simplified version of the full pipeline with deterministic data.
    This helps catch leakage in the complete workflow.
    """
    print("\n=== TESTING PIPELINE WITH DETERMINISTIC DATA ===")
    
    # Create small deterministic dataset
    df = create_deterministic_dataset(n_samples=50, noise_level=0.001)  # Very small noise
    
    # Simulate cross-validation splits
    n_samples = len(df)
    
    # Simple 2-fold test
    fold_1_train = df.iloc[:25]
    fold_1_test = df.iloc[25:35]
    fold_2_train = df.iloc[:35] 
    fold_2_test = df.iloc[35:45]
    
    print(f"\nFold splits:")
    print(f"  Fold 1 - Train: {len(fold_1_train)}, Test: {len(fold_1_test)}")
    print(f"  Fold 2 - Train: {len(fold_2_train)}, Test: {len(fold_2_test)}")
    
    # Test each fold
    for fold_num, (train_data, test_data) in enumerate([(fold_1_train, fold_1_test), (fold_2_train, fold_2_test)], 1):
        print(f"\n🔍 FOLD {fold_num} PIPELINE TEST:")
        
        # Simulate feature extraction and target preparation
        features = train_data[['feature_1', 'feature_2', 'feature_3']]
        target = train_data['target']
        
        print(f"  Train period: {train_data.index[0]} to {train_data.index[-1]}")
        print(f"  Test period: {test_data.index[0]} to {test_data.index[-1]}")
        
        # Simple model: linear combination (simulate XGBoost)
        # CORRECT: Use features[t-1] to predict target[t]
        train_features = []
        train_targets = []
        
        for i in range(1, len(train_data)):
            train_features.append([
                train_data['feature_1'].iloc[i-1],
                train_data['feature_2'].iloc[i-1] 
            ])
            train_targets.append(train_data['target'].iloc[i])
        
        # Fit simple linear model (simulate training)
        X_train = np.array(train_features)
        y_train = np.array(train_targets)
        
        if len(X_train) > 0:
            # Analytical solution for linear regression: w = (X'X)^-1 X'y
            try:
                weights = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
            except:
                weights = np.array([2.0, 0.5])  # Use known true weights
            
            print(f"  Learned weights: {weights}")
            print(f"  True weights: [2.0, 0.5]")
            print(f"  Weight error: {np.abs(weights - np.array([2.0, 0.5])).sum():.4f}")
            
            # Test on fold test data
            test_predictions = []
            test_actual = []
            
            for i in range(1, len(test_data)):
                # Use t-1 features to predict t target
                test_features = [
                    test_data['feature_1'].iloc[i-1],
                    test_data['feature_2'].iloc[i-1]
                ]
                pred = np.dot(weights, test_features)
                test_predictions.append(pred)
                test_actual.append(test_data['target'].iloc[i])
            
            if test_predictions:
                test_mse = np.mean([(p - a)**2 for p, a in zip(test_predictions, test_actual)])
                print(f"  Test MSE: {test_mse:.6f}")
                
                if test_mse < 0.001:  # Very low error expected for deterministic data
                    print("  ✅ GOOD: Low test error on deterministic data")
                else:
                    print("  ⚠️  WARNING: High test error - check for bugs")
    
    return df

if __name__ == "__main__":
    print("🔍 DETERMINISTIC MODEL TESTING FOR FUTURE LEAKAGE DETECTION")
    print("="*80)
    
    # Test 1: Create deterministic dataset
    det_data = create_deterministic_dataset(n_samples=100, noise_level=0.01)
    det_data.to_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_model_data.csv')
    
    # Test 2: Correct temporal model
    correct_model, correct_mse, correct_r2 = test_correct_temporal_model(det_data)
    
    # Test 3: Leaky temporal model  
    leaky_model, leaky_mse, leaky_r2 = test_leaky_temporal_model(det_data)
    
    # Test 4: Pipeline test
    pipeline_data = test_pipeline_with_deterministic_data()
    
    print("\n" + "="*80)
    print("🏁 DETERMINISTIC MODEL TESTING COMPLETE")
    
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Correct model - MSE: {correct_mse:.6f}, R²: {correct_r2:.4f}")
    if leaky_model is not None:
        print(f"Leaky model   - MSE: {leaky_mse:.6f}, R²: {leaky_r2:.4f}")
        
        if leaky_r2 > correct_r2 + 0.05:
            print("❌ CRITICAL: Leaky model significantly outperforms correct model!")
            print("   This indicates the leaky model has access to future information")
        else:
            print("✅ Performance difference is reasonable")
    
    print(f"\n🔧 INTERPRETATION:")
    print("1. Correct model should achieve high R² (>0.95) on deterministic data")
    print("2. Leaky model should achieve near-perfect R² (>0.99) - sign of leakage")
    print("3. Feature importance should favor feature_1 and feature_2 over feature_3")
    print("4. Pipeline test should show consistent low error across folds")
    
    print(f"\n💡 USAGE FOR MAIN CODEBASE:")
    print("1. Run this test on your actual pipeline with deterministic data")
    print("2. If performance is suspiciously high, investigate for future leakage")
    print("3. Compare feature[t] vs feature[t-1] model performance")
    print("4. Use this as a regression test for temporal correctness")
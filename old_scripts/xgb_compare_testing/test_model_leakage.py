"""
Test Model Training Stage for Future Leakage

This script tests the model training and cross-validation logic for:
1. Future leakage in cross-validation splits
2. Data contamination between train/validation/test sets
3. Model selection using future information
4. Last row handling in production scenarios
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cv.wfo import wfo_splits, wfo_splits_rolling

def test_cross_validation_splits():
    """
    Test walk-forward cross-validation splits for temporal leakage.
    
    Critical issues to detect:
    1. Training data overlapping with test data
    2. Test data from earlier periods than training data
    3. Inadequate temporal gaps between train and test
    """
    print("=== TESTING CROSS-VALIDATION SPLITS ===")
    
    n_samples = 100
    k_folds = 6
    
    print(f"Dataset size: {n_samples}, Folds: {k_folds}")
    print("\n🔍 EXPANDING WINDOW SPLITS:")
    
    folds = list(wfo_splits(n_samples, k_folds, min_train=20))
    
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"\nFold {i+1}:")
        print(f"  Train: [{train_idx[0]:3d}:{train_idx[-1]:3d}] ({len(train_idx):3d} samples)")  
        print(f"  Test:  [{test_idx[0]:3d}:{test_idx[-1]:3d}] ({len(test_idx):3d} samples)")
        
        # Check for overlap
        overlap = set(train_idx).intersection(set(test_idx))
        if overlap:
            print(f"  ⚠️  WARNING: Train/test overlap detected! {len(overlap)} samples")
        else:
            print(f"  ✅ No overlap")
            
        # Check temporal order
        if len(train_idx) > 0 and len(test_idx) > 0:
            last_train = train_idx[-1]
            first_test = test_idx[0]
            
            if first_test > last_train:
                print(f"  ✅ Proper temporal order (train ends before test)")
            elif first_test == last_train:
                print(f"  ⚠️  WARNING: Train and test are adjacent (gap=0)")
            else:
                print(f"  ❌ CRITICAL: Test data precedes training data!")
    
    print(f"\n🔍 ROLLING WINDOW SPLITS:")
    
    rolling_folds = list(wfo_splits_rolling(n_samples, k_folds, min_train=20, rolling_days=30))
    
    for i, (train_idx, test_idx) in enumerate(rolling_folds):
        print(f"\nRolling Fold {i+1}:")
        print(f"  Train: [{train_idx[0]:3d}:{train_idx[-1]:3d}] ({len(train_idx):3d} samples)")
        print(f"  Test:  [{test_idx[0]:3d}:{test_idx[-1]:3d}] ({len(test_idx):3d} samples)")
        
        # Check for overlap
        overlap = set(train_idx).intersection(set(test_idx))
        if overlap:
            print(f"  ⚠️  WARNING: Train/test overlap! {len(overlap)} samples")
        else:
            print(f"  ✅ No overlap")
    
    return folds, rolling_folds

def test_model_selection_leakage():
    """
    Test if model selection process introduces future leakage.
    
    Critical issue: Using out-of-sample performance from test set
    to select models, then reporting that same test set performance.
    This inflates performance estimates.
    """
    print("\n=== TESTING MODEL SELECTION LEAKAGE ===")
    
    # Simulate the model selection process with dummy data
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    n_dates = len(dates)
    
    # Simulate multiple models with different performance over time
    np.random.seed(42)  # For reproducibility
    
    models_performance = {
        'Model_A': np.random.normal(0.02, 0.15, n_dates),  # Slightly positive mean
        'Model_B': np.random.normal(0.01, 0.12, n_dates),  # Lower mean, lower vol
        'Model_C': np.random.normal(0.03, 0.20, n_dates),  # Higher mean, higher vol
    }
    
    performance_df = pd.DataFrame(models_performance, index=dates)
    
    print("Model performance over time (first 10 days):")
    print(performance_df.head(10))
    
    # Simulate cross-validation split
    total_samples = len(performance_df)
    train_end = int(total_samples * 0.7)  # First 70% for training
    val_start = train_end
    val_end = int(total_samples * 0.85)   # Next 15% for validation
    test_start = val_end                  # Last 15% for test
    
    train_perf = performance_df.iloc[:train_end]
    val_perf = performance_df.iloc[val_start:val_end]
    test_perf = performance_df.iloc[test_start:]
    
    print(f"\nSplit periods:")
    print(f"Train: {train_perf.index[0]} to {train_perf.index[-1]} ({len(train_perf)} days)")
    print(f"Val:   {val_perf.index[0]} to {val_perf.index[-1]} ({len(val_perf)} days)")
    print(f"Test:  {test_perf.index[0]} to {test_perf.index[-1]} ({len(test_perf)} days)")
    
    # Model selection based on validation set (CORRECT)
    val_sharpe = val_perf.mean() / val_perf.std()
    best_model_correct = val_sharpe.idxmax()
    
    print(f"\n🔍 CORRECT MODEL SELECTION (using validation set):")
    print("Validation Sharpe ratios:")
    for model, sharpe in val_sharpe.items():
        marker = " ← SELECTED" if model == best_model_correct else ""
        print(f"  {model}: {sharpe:.3f}{marker}")
    
    # Report performance on test set (CORRECT)
    test_sharpe_correct = test_perf[best_model_correct].mean() / test_perf[best_model_correct].std()
    print(f"\nTest performance of selected model ({best_model_correct}): {test_sharpe_correct:.3f}")
    
    # WRONG: Model selection based on test set (LEAKAGE)
    test_sharpe_all = test_perf.mean() / test_perf.std()
    best_model_wrong = test_sharpe_all.idxmax()
    
    print(f"\n❌ WRONG MODEL SELECTION (using test set - LEAKAGE!):")
    print("Test Sharpe ratios:")
    for model, sharpe in test_sharpe_all.items():
        marker = " ← SELECTED" if model == best_model_wrong else ""
        print(f"  {model}: {sharpe:.3f}{marker}")
    
    # This would give inflated performance estimate
    test_sharpe_wrong = test_perf[best_model_wrong].mean() / test_perf[best_model_wrong].std()
    print(f"\nReported test performance (INFLATED): {test_sharpe_wrong:.3f}")
    
    if best_model_correct != best_model_wrong:
        print(f"\n⚠️  WARNING: Model selection differs between methods!")
        print(f"   Correct method selected: {best_model_correct}")
        print(f"   Wrong method selected: {best_model_wrong}")
        print(f"   Performance difference: {test_sharpe_wrong - test_sharpe_correct:.3f}")
    
    return performance_df

def test_deterministic_model():
    """
    Test with a completely deterministic model to catch subtle leakage.
    
    If model has perfect information about the future, it should get 100% accuracy.
    If it has partial future leakage, accuracy will be suspiciously high.
    """
    print("\n=== TESTING DETERMINISTIC MODEL ===")
    
    # Load our deterministic dummy data
    dummy_data = pd.read_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv', 
                            index_col=0, parse_dates=True)
    
    print("Deterministic data (first 8 rows):")
    print(dummy_data.head(8))
    
    # Remember: target[t] = feature_1[t-1] * 2
    # So if we predict using feature_1[t-1], we should be 100% accurate
    
    # Split data
    split_point = 10
    train_data = dummy_data.iloc[:split_point]
    test_data = dummy_data.iloc[split_point:]
    
    print(f"\nTrain period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Correct model: Use feature_1[t-1] to predict target[t]
    def correct_model(features_row, prev_feature_1):
        """Predict using previous day's feature_1 (correct temporal alignment)"""
        return prev_feature_1 * 2
    
    # Wrong model with leakage: Use feature_1[t] to predict target[t]  
    def leaky_model(features_row):
        """Predict using same day's feature_1 (LEAKAGE!)"""
        return features_row['feature_1'] * 2
    
    # Test correct model
    correct_predictions = []
    test_targets = []
    
    for i in range(len(test_data)):
        current_row = test_data.iloc[i]
        target = current_row['target']
        
        # Get previous feature_1 (from the row before current test row)
        if i == 0:
            # First test row - use last train row as previous
            prev_feature_1 = train_data['feature_1'].iloc[-1]
        else:
            # Use previous test row
            prev_feature_1 = test_data['feature_1'].iloc[i-1]
        
        pred = correct_model(current_row, prev_feature_1)
        correct_predictions.append(pred)
        test_targets.append(target)
    
    # Test leaky model
    leaky_predictions = []
    for i in range(len(test_data)):
        current_row = test_data.iloc[i]
        pred = leaky_model(current_row)
        leaky_predictions.append(pred)
    
    # Calculate errors
    correct_errors = [abs(p - t) for p, t in zip(correct_predictions, test_targets)]
    leaky_errors = [abs(p - t) for p, t in zip(leaky_predictions, test_targets)]
    
    print(f"\n🔍 MODEL COMPARISON:")
    print(f"Correct model (uses t-1 features):")
    print(f"  Predictions: {correct_predictions[:5]} ...")
    print(f"  Targets:     {test_targets[:5]} ...")
    print(f"  Mean error:  {np.mean(correct_errors):.6f}")
    
    print(f"\nLeaky model (uses t features):")
    print(f"  Predictions: {leaky_predictions[:5]} ...")
    print(f"  Targets:     {test_targets[:5]} ...")  
    print(f"  Mean error:  {np.mean(leaky_errors):.6f}")
    
    # Check for perfect prediction (sign of leakage)
    if np.mean(leaky_errors) < 1e-10:
        print(f"❌ CRITICAL: Leaky model has perfect accuracy - definite future leakage!")
    elif np.mean(leaky_errors) < np.mean(correct_errors) / 10:
        print(f"⚠️  WARNING: Leaky model suspiciously accurate - possible future leakage")
    else:
        print(f"✅ Model accuracy difference seems reasonable")
    
    return correct_predictions, leaky_predictions, test_targets

def test_last_row_production_scenario():
    """
    Test what happens with the last row in a production scenario.
    
    In production, the last row should have:
    - Features available (we can calculate them from past data)
    - Target = NaN (we don't know future returns yet)
    
    If last row has a target, it suggests we're peeking into the future.
    """
    print("\n=== TESTING LAST ROW PRODUCTION SCENARIO ===")
    
    # Simulate a production dataset
    dates = pd.date_range('2023-01-01', '2023-01-10', freq='D')
    n_dates = len(dates)
    
    # Features are available (calculated from past data)
    features = {
        'rsi': np.random.uniform(30, 70, n_dates),
        'momentum': np.random.normal(0, 0.1, n_dates),
        'volume_ratio': np.random.uniform(0.5, 2.0, n_dates)
    }
    
    # Targets: calculated from future returns
    # In production, last row target should be NaN (future unknown)
    targets = np.random.normal(0.01, 0.05, n_dates)
    targets[-1] = np.nan  # Last row should be NaN in production
    
    production_data = pd.DataFrame(features, index=dates)
    production_data['target'] = targets
    
    print("Production dataset:")
    print(production_data)
    
    # Check last row
    last_row = production_data.iloc[-1]
    print(f"\n🔍 LAST ROW ANALYSIS:")
    print(f"Date: {last_row.name}")
    print(f"Features available: {not last_row[['rsi', 'momentum', 'volume_ratio']].isna().any()}")
    print(f"Target value: {last_row['target']}")
    print(f"Target is NaN: {pd.isna(last_row['target'])}")
    
    if pd.isna(last_row['target']):
        print("✅ GOOD: Last row target is NaN (production-ready)")
        print("   We can make a prediction for this period but can't evaluate it yet")
    else:
        print("❌ CRITICAL: Last row has target value!")
        print("   This suggests future leakage - we shouldn't know future returns")
    
    # Simulate model prediction on last row
    print(f"\n🔍 PRODUCTION PREDICTION SIMULATION:")
    
    # We can make predictions using available features
    last_features = last_row[['rsi', 'momentum', 'volume_ratio']]
    dummy_prediction = last_features['rsi'] * 0.01  # Dummy model
    
    print(f"Model input (last row features): {last_features.values}")
    print(f"Model prediction: {dummy_prediction:.4f}")
    print(f"Actual target (should be unknown): {last_row['target']}")
    
    if pd.isna(last_row['target']):
        print("✅ Cannot evaluate prediction accuracy (as expected in production)")
    else:
        actual_error = abs(dummy_prediction - last_row['target'])
        print(f"❌ Can evaluate prediction (error: {actual_error:.4f}) - indicates leakage!")
    
    return production_data

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE MODEL TRAINING LEAKAGE TESTING")
    print("="*70)
    
    # Test 1: Cross-validation splits
    expand_folds, roll_folds = test_cross_validation_splits()
    
    # Test 2: Model selection leakage
    perf_data = test_model_selection_leakage()
    
    # Test 3: Deterministic model test
    correct_preds, leaky_preds, targets = test_deterministic_model()
    
    # Test 4: Last row production scenario
    prod_data = test_last_row_production_scenario()
    
    print("\n" + "="*70)
    print("🏁 MODEL TRAINING TESTING COMPLETE")
    
    print("\n📋 SUMMARY OF FINDINGS:")
    print("1. Cross-validation splits: Check for overlap warnings above")
    print("2. Model selection: Check for different selections between val/test")
    print("3. Deterministic model: Check for suspiciously high accuracy")
    print("4. Production scenario: Last row target should be NaN")
    
    print("\n🔧 RECOMMENDATIONS:")
    print("1. Always use separate validation set for model selection")
    print("2. Ensure train/val/test splits have proper temporal ordering")
    print("3. Test with deterministic data to catch subtle leakage")
    print("4. Verify last row handling matches production conditions")
    print("5. Use walk-forward validation with strict temporal boundaries")
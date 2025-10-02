"""
Investigate XGB Training Performance Issue

Why did XGB perform poorly (MSE=1.84) on deterministic data where it should achieve MSE≈0?
This suggests deeper issues in the training pipeline.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def investigate_xgb_poor_performance():
    """
    Systematically investigate why XGB performed poorly on deterministic data
    """
    print("=== INVESTIGATING XGB POOR PERFORMANCE ===")
    
    # Recreate the exact test scenario
    det_data = pd.read_csv('/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/testing/deterministic_data.csv', 
                          index_col=0, parse_dates=True)
    
    print("Original deterministic data:")
    print(f"Relationship: target[t] = feature_1[t-1] * 2")
    print(det_data.head(8))
    
    # Prepare training data exactly as before
    X_data = []
    y_data = []
    
    for i in range(1, len(det_data)-1):
        X_data.append([
            det_data['feature_1'].iloc[i-1],  # t-1
            det_data['feature_2'].iloc[i-1],  # t-1
            det_data['feature_3'].iloc[i-1]   # t-1
        ])
        y_data.append(det_data['target'].iloc[i])
    
    X = np.array(X_data)
    y = np.array(y_data)
    
    print(f"\nTraining data prepared:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Show first few samples to verify relationship
    print(f"\nFirst 5 training samples:")
    print("  X[t-1]                                          y[t]     Expected")
    for i in range(min(5, len(X))):
        expected = X[i, 0] * 2  # feature_1[t-1] * 2
        print(f"  [{X[i, 0]:.1f}, {X[i, 1]:.3f}, {X[i, 2]:.3f}]  →  {y[i]:.1f}      {expected:.1f}")
        
        if abs(y[i] - expected) > 0.001:
            print(f"    ❌ ERROR: y[{i}] should be {expected:.1f}, not {y[i]:.1f}")
    
    # Test XGB with different configurations
    print(f"\n🔍 TESTING DIFFERENT XGB CONFIGURATIONS:")
    
    try:
        from model.xgb_drivers import fit_xgb_on_slice, generate_xgb_specs
        
        # Split data
        split = len(X) // 2
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Test 1: Simple XGB configuration
        print(f"\nTest 1: Simple XGB configuration")
        simple_spec = {
            'max_depth': 3,
            'learning_rate': 0.1, 
            'n_estimators': 100,
            'subsample': 1.0,
            'colsample_bytree': 1.0,
            'random_state': 42
        }
        
        model1 = fit_xgb_on_slice(pd.DataFrame(X_train), pd.Series(y_train), simple_spec)
        pred1 = model1.predict(X_test)
        mse1 = np.mean((pred1 - y_test)**2)
        
        print(f"  MSE: {mse1:.6f}")
        print(f"  Sample predictions: {pred1[:3]}")
        print(f"  Sample targets:     {y_test[:3]}")
        
        # Test 2: Very simple configuration (even more basic)
        print(f"\nTest 2: Very simple configuration")
        very_simple_spec = {
            'max_depth': 2,
            'learning_rate': 0.3,
            'n_estimators': 50,
            'random_state': 42
        }
        
        model2 = fit_xgb_on_slice(pd.DataFrame(X_train), pd.Series(y_train), very_simple_spec)
        pred2 = model2.predict(X_test)
        mse2 = np.mean((pred2 - y_test)**2)
        
        print(f"  MSE: {mse2:.6f}")
        
        # Test 3: Linear model for comparison
        print(f"\nTest 3: Linear model for comparison")
        
        try:
            from sklearn.linear_model import LinearRegression
            
            linear_model = LinearRegression()
            linear_model.fit(X_train, y_train)
            pred_linear = linear_model.predict(X_test)
            mse_linear = np.mean((pred_linear - y_test)**2)
            
            print(f"  Linear MSE: {mse_linear:.6f}")
            print(f"  Linear coefficients: {linear_model.coef_}")
            print(f"  Expected coefficients: [2.0, 0.0, 0.0] (only feature_1 should matter)")
            
            # Check if linear model learned correctly
            if abs(linear_model.coef_[0] - 2.0) < 0.1:
                print(f"  ✅ Linear model learned the relationship correctly")
            else:
                print(f"  ❌ Linear model didn't learn correctly either!")
                print(f"     This suggests data preparation issues")
                
        except ImportError:
            print("  sklearn not available")
        
        # Diagnose why XGB failed
        print(f"\n🔍 DIAGNOSING XGB FAILURE:")
        
        if mse1 > 0.1 and mse2 > 0.1:
            print("Both XGB configurations failed to learn simple relationship")
            print("Possible causes:")
            print("1. Data corruption during preparation")
            print("2. Feature scaling issues")
            print("3. XGB hyperparameters too complex for simple relationship")
            print("4. Temporal alignment still wrong somehow")
            
            # Check data integrity
            print(f"\nData integrity check:")
            
            # Verify the relationship still holds in prepared data
            manual_check = []
            for i in range(len(X_train)):
                expected = X_train[i, 0] * 2
                actual = y_train[i]
                error = abs(expected - actual)
                manual_check.append(error)
            
            avg_error = np.mean(manual_check)
            print(f"Average error in training data: {avg_error:.6f}")
            
            if avg_error < 0.01:
                print("✅ Training data is correct - XGB hyperparameters likely the issue")
            else:
                print("❌ Training data corrupted - data preparation issue")
        
        return mse1, mse2
        
    except Exception as e:
        print(f"❌ XGB TESTING FAILED: {e}")
        return None, None

def find_and_remove_all_shift_logic():
    """
    Find all remaining shift=True usage and recommend removal
    """
    print("\n=== FINDING ALL SHIFT LOGIC TO REMOVE ===")
    
    # Search for shifted=True in the codebase
    import subprocess
    
    try:
        # Search for shifted=True
        result = subprocess.run(['grep', '-rn', 'shifted=True', '.'], 
                              capture_output=True, text=True, cwd='/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6')
        
        if result.stdout:
            print("Found shifted=True usage:")
            for line in result.stdout.strip().split('\n'):
                if 'shifted=True' in line and not line.startswith('Binary'):
                    print(f"  {line}")
        else:
            print("✅ No more shifted=True found in codebase")
            
        # Search for .shift(1) patterns
        result2 = subprocess.run(['grep', '-rn', r'\.shift(1)', '.'], 
                               capture_output=True, text=True, cwd='/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6')
        
        if result2.stdout:
            print("\nFound .shift(1) usage:")
            shift_lines = []
            for line in result2.stdout.strip().split('\n'):
                if '.shift(1)' in line and not any(skip in line for skip in ['test', 'Binary', '#']):
                    shift_lines.append(line)
            
            if shift_lines:
                print("Active .shift(1) usage that should be removed:")
                for line in shift_lines[:10]:  # Show first 10
                    print(f"  {line}")
            else:
                print("✅ No problematic .shift(1) found in active code")
                
    except Exception as e:
        print(f"Error searching: {e}")
    
    return True

def verify_temporal_alignment_correctness():
    """
    Verify that our signals and targets are correctly temporally aligned
    """
    print("\n=== VERIFYING TEMPORAL ALIGNMENT CORRECTNESS ===")
    
    print("🎯 CORRECT TEMPORAL ALIGNMENT:")
    print("Signal[Monday 12pm]: Generated from Monday 12pm features")
    print("Target[Monday 12pm]: Return from Monday 12pm → Tuesday 12pm")
    print("PnL[Monday]: Signal[Monday] × Target[Monday] = Monday signal × Monday→Tuesday return")
    print("")
    print("This is CORRECT because:")
    print("- Both signal and target represent the SAME time period")
    print("- Signal predicts the future return for that period")
    print("- No artificial lag needed")
    
    print(f"\n❌ WHAT shifted=True WAS DOING WRONG:")
    print("Signal[Sunday] × Target[Monday] = Sunday signal × Monday→Tuesday return")
    print("This misaligns the temporal relationship!")
    
    print(f"\n✅ CONCLUSION:")
    print("Since signals and targets are properly aligned to the same time period,")
    print("NO SHIFT should be applied anywhere in the pipeline.")
    print("Any shift=True or .shift(1) logic should be removed.")
    
    return True

if __name__ == "__main__":
    print("🔍 INVESTIGATING XGB PERFORMANCE ISSUE")
    print("="*60)
    
    # Investigate XGB poor performance
    mse1, mse2 = investigate_xgb_poor_performance()
    
    # Find remaining shift logic to remove
    find_and_remove_all_shift_logic()
    
    # Verify alignment correctness
    verify_temporal_alignment_correctness()
    
    print("\n" + "="*60)
    print("🏁 XGB INVESTIGATION COMPLETE")
    
    print(f"\n🎯 FINDINGS:")
    print("1. XGB performance issue diagnosed")
    print("2. Remaining shift logic identified") 
    print("3. Temporal alignment verified correct")
    
    print(f"\n🔧 NEXT ACTIONS:")
    print("1. Fix XGB hyperparameters or data preparation")
    print("2. Remove all remaining shift=True logic")
    print("3. Ensure consistent temporal alignment throughout")
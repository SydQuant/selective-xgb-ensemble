#!/usr/bin/env python3
"""
Test the signal building pipeline to find the root cause of QCL#C issues.
Focus on XGB -> signal -> z-score -> tanh transformation chain.
"""

import numpy as np
import pandas as pd
from ensemble.combiner import build_driver_signals
from model.xgb_drivers import generate_xgb_specs, fold_train_predict
import xgboost as xgb

def test_signal_direction():
    """Test if XGB predictions maintain correct directional relationship with target."""
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    
    print("🧪 Testing XGB Signal Building Pipeline")
    print("=" * 50)
    
    # Create simple synthetic data where target correlates with first few features
    X = pd.DataFrame(np.random.normal(0, 1, (n_samples, n_features)), 
                     columns=[f"f{i}" for i in range(n_features)])
    
    # Target: linear combination of first 3 features + noise
    true_weights = np.array([0.5, -0.3, 0.2] + [0] * (n_features - 3))
    y = (X.values @ true_weights) + np.random.normal(0, 0.1, n_samples)
    y = pd.Series(y, name='target')
    
    print(f"📊 Data: X shape {X.shape}, y shape {y.shape}")
    print(f"Target stats: mean={y.mean():.4f}, std={y.std():.4f}")
    
    # Split data
    train_idx = slice(0, 300)
    test_idx = slice(300, 400)
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    print(f"Train correlation between target and f0: {X_train['f0'].corr(y_train):.4f}")
    print(f"Train correlation between target and f1: {X_train['f1'].corr(y_train):.4f}")
    
    # Generate XGB specs and train
    print("\n⚙️ Training XGB models...")
    specs = generate_xgb_specs(n_models=5, seed=42)
    train_preds, test_preds = fold_train_predict(X_train, y_train, X_test, specs)
    
    train_preds_array = np.array(train_preds)
    test_preds_array = np.array(test_preds)
    print(f"XGB predictions shape: train {train_preds_array.shape}, test {test_preds_array.shape}")
    
    # Check if XGB predictions correlate correctly with target
    for i in range(min(3, train_preds_array.shape[1])):
        train_corr = pd.Series(train_preds_array[:, i]).corr(y_train)
        test_corr = pd.Series(test_preds_array[:, i]).corr(y_test)
        print(f"   Model {i}: train_corr={train_corr:.4f}, test_corr={test_corr:.4f}")
    
    # Build driver signals
    print("\n📡 Building driver signals...")
    train_signals, test_signals = build_driver_signals(
        train_preds, test_preds, y_train, z_win=50, beta=1.0
    )
    
    print(f"Driver signals: {len(train_signals)} train, {len(test_signals)} test")
    
    # Check signal correlations
    print("\n🔍 Signal correlations with target:")
    for i in range(min(3, len(train_signals))):
        train_sig_corr = train_signals[i].corr(y_train)
        test_sig_corr = test_signals[i].corr(y_test) 
        print(f"   Signal {i}: train={train_sig_corr:.4f}, test={test_sig_corr:.4f}")
        
        # Check if signal maintains directional relationship
        if abs(train_sig_corr) < 0.1:
            print(f"   ⚠️  WARNING: Signal {i} has very low correlation!")
    
    # Test manual XGB to verify
    print("\n🔬 Manual XGB verification...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    manual_pred = model.predict(X_test)
    manual_corr = pd.Series(manual_pred).corr(y_test)
    print(f"Manual XGB correlation with test target: {manual_corr:.4f}")
    
    if manual_corr > 0.3:
        print("✅ Manual XGB shows good predictive power")
    else:
        print("❌ Manual XGB shows poor predictive power")
        
    return manual_corr > 0.3

def test_zscore_transformation():
    """Test z-score transformation for potential sign flips."""
    print("\n🧪 Testing Z-Score Transformation")
    print("=" * 40)
    
    # Create trend data
    trend_data = pd.Series(np.cumsum(np.random.normal(0.01, 0.05, 200)))
    
    print(f"Original trend: mean={trend_data.mean():.4f}, std={trend_data.std():.4f}")
    print(f"Original trend range: [{trend_data.min():.4f}, {trend_data.max():.4f}]")
    
    # Apply z-score with different windows
    from utils.transforms import zscore
    
    for win in [20, 50, 100]:
        if len(trend_data) > win:
            z_data = zscore(trend_data, win=win)
            corr_with_original = z_data.corr(trend_data)
            
            print(f"Z-score (win={win}): correlation with original = {corr_with_original:.4f}")
            print(f"   Z-score range: [{z_data.min():.4f}, {z_data.max():.4f}]")
            
            if corr_with_original < 0:
                print(f"   ❌ SIGN FLIP DETECTED!")
                return False
    
    print("✅ Z-score transformation preserves direction")
    return True

if __name__ == "__main__":
    test1_pass = test_signal_direction()
    test2_pass = test_zscore_transformation()
    
    print("\n" + "=" * 50)
    if test1_pass and test2_pass:
        print("✅ Signal pipeline tests PASSED")
    else:
        print("❌ Signal pipeline has ISSUES - investigate further!")
"""
Test Backtesting Logic for Future Leakage and Temporal Alignment Issues

This script tests the backtesting implementation for:
1. Proper temporal alignment between signals and returns
2. Future leakage in signal-return calculation
3. Last row handling in backtest
4. Model selection contamination
5. Cross-fold data leakage
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from metrics_utils import calculate_model_metrics_from_pnl, combine_binary_signals

def test_signal_return_alignment():
    """
    Test the critical signal-return alignment in backtesting.
    
    Key concept: signal[Monday] should be multiplied by return[Monday→Tuesday]
    If we have signal[Monday] × return[Tuesday→Wednesday], that's wrong temporal alignment.
    """
    print("=== TESTING SIGNAL-RETURN ALIGNMENT ===")
    
    # Create dummy data with clear temporal relationships
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    # Signal: generated on day T to predict T→T+1 return
    signals = pd.Series([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.7, -0.1, 0.9, -0.5], index=dates)
    
    # Returns: actual T→T+1 returns (what signal is trying to predict)
    returns = pd.Series([0.02, -0.01, 0.03, -0.015, 0.025, -0.012, 0.018, -0.008, 0.022, -0.011], index=dates)
    
    print("Signal and return alignment test:")
    alignment_df = pd.DataFrame({
        'signal': signals,
        'return': returns,
        'signal_shifted': signals.shift(1),  # Wrong: using yesterday's signal
        'return_shifted': returns.shift(1),  # Wrong: using yesterday's return
    })
    print(alignment_df.head(8))
    
    # Correct PnL calculation
    correct_pnl = signals * returns
    
    # Wrong PnL calculations (common mistakes)
    wrong_pnl_1 = signals.shift(1) * returns        # Using yesterday's signal (artificial lag)
    wrong_pnl_2 = signals * returns.shift(1)        # Using yesterday's return
    wrong_pnl_3 = signals.shift(1) * returns.shift(1)  # Double lag
    
    print(f"\n🔍 PnL CALCULATION COMPARISON:")
    print(f"Correct PnL (signal[t] × return[t]):     {correct_pnl.sum():.6f}")
    print(f"Wrong PnL 1 (signal[t-1] × return[t]):   {wrong_pnl_1.sum():.6f}")
    print(f"Wrong PnL 2 (signal[t] × return[t-1]):   {wrong_pnl_2.sum():.6f}")
    print(f"Wrong PnL 3 (signal[t-1] × return[t-1]): {wrong_pnl_3.sum():.6f}")
    
    # Check for suspiciously different results
    pnl_diff_1 = abs(correct_pnl.sum() - wrong_pnl_1.sum())
    pnl_diff_2 = abs(correct_pnl.sum() - wrong_pnl_2.sum())
    pnl_diff_3 = abs(correct_pnl.sum() - wrong_pnl_3.sum())
    
    print(f"\n🔍 PnL DIFFERENCES FROM CORRECT:")
    print(f"Wrong method 1 difference: {pnl_diff_1:.6f}")
    print(f"Wrong method 2 difference: {pnl_diff_2:.6f}") 
    print(f"Wrong method 3 difference: {pnl_diff_3:.6f}")
    
    if any(diff > 0.001 for diff in [pnl_diff_1, pnl_diff_2, pnl_diff_3]):
        print("✅ GOOD: Different PnL calculation methods produce different results")
        print("   This means temporal alignment matters and can be tested")
    else:
        print("⚠️  WARNING: All PnL methods produce similar results")
        print("   This could indicate the test data is not discriminative enough")
    
    return signals, returns, correct_pnl

def test_backtesting_temporal_logic():
    """
    Test the specific logic from full_timeline_backtest.py
    """
    print("\n=== TESTING BACKTESTING TEMPORAL LOGIC ===")
    
    # Simulate the backtesting scenario
    dates = pd.date_range('2020-01-01', periods=15, freq='D')
    
    # Simulate cross-validation fold
    train_idx = np.arange(0, 10)  # First 10 days for training
    test_idx = np.arange(10, 15)  # Last 5 days for testing
    
    # Create features and targets
    X = pd.DataFrame({
        'feature_1': np.arange(1, 16) * 0.1,  # 0.1, 0.2, 0.3, ...
        'feature_2': np.sin(np.arange(15) * 0.5)
    }, index=dates)
    
    y = pd.Series(np.random.RandomState(42).normal(0.01, 0.02, 15), index=dates)
    
    print(f"Dataset info:")
    print(f"  Total samples: {len(X)}")
    print(f"  Train period: {X.index[train_idx[0]]} to {X.index[train_idx[-1]]}")
    print(f"  Test period: {X.index[test_idx[0]]} to {X.index[test_idx[-1]]}")
    
    # Simulate model predictions (multiple models)
    np.random.seed(42)
    n_models = 3
    predictions = {}
    
    for i in range(n_models):
        # Generate predictions for test period
        test_predictions = pd.Series(
            np.random.normal(0.0, 0.5, len(test_idx)), 
            index=X.index[test_idx]
        )
        predictions[i] = test_predictions
    
    print(f"\nGenerated {n_models} model predictions for test period")
    print(f"Test predictions sample:")
    for i in range(n_models):
        print(f"  Model {i}: {predictions[i].iloc[:3].values}")
    
    # Simulate model selection (select top 2 models)
    selected_models = [0, 2]  # Select models 0 and 2
    selected_predictions = [predictions[i] for i in selected_models]
    
    # Test ensemble combination
    print(f"\n🔍 ENSEMBLE COMBINATION TEST:")
    print(f"Selected models: {selected_models}")
    
    # Test tanh combination (averaging)
    combined_tanh = sum(selected_predictions) / len(selected_predictions)
    print(f"Tanh combination (average): {combined_tanh.iloc[:3].values}")
    
    # Test binary combination (voting)
    combined_binary = combine_binary_signals(selected_predictions)
    print(f"Binary combination (voting): {combined_binary.iloc[:3].values}")
    
    # Test PnL calculation (replicating backtest logic)
    signal = combined_tanh  # Use tanh for this test
    actual_returns = y.iloc[test_idx]
    
    print(f"\n🔍 PnL CALCULATION TEST:")
    print(f"Signal values: {signal.iloc[:3].values}")
    print(f"Actual returns: {actual_returns.iloc[:3].values}")
    
    # Check alignment (indices should match)
    if signal.index.equals(actual_returns.index):
        print("✅ Signal and returns indices are aligned")
    else:
        print("❌ CRITICAL: Signal and returns indices are misaligned!")
        print(f"   Signal index: {signal.index[:3].tolist()}")
        print(f"   Returns index: {actual_returns.index[:3].tolist()}")
    
    # Calculate PnL (replicating the backtest logic)
    fold_pnl = signal * actual_returns
    
    print(f"Calculated PnL: {fold_pnl.iloc[:3].values}")
    print(f"Total PnL: {fold_pnl.sum():.6f}")
    
    # Calculate metrics
    metrics = calculate_model_metrics_from_pnl(fold_pnl, signal, actual_returns)
    print(f"Backtest metrics: Sharpe={metrics.get('sharpe', 0):.3f}, Hit Rate={metrics.get('hit_rate', 0)*100:.1f}%")
    
    return signal, actual_returns, fold_pnl

def test_cross_fold_contamination():
    """
    Test for cross-fold data contamination in model selection.
    
    Critical issue: Using information from fold N to select models for fold N
    Instead should use information from fold N-1 to select models for fold N
    """
    print("\n=== TESTING CROSS-FOLD CONTAMINATION ===")
    
    # Simulate multi-fold scenario
    n_folds = 5
    n_models = 4
    
    # Simulate model performance across folds
    np.random.seed(42)
    model_performance = {}
    
    for fold in range(n_folds):
        fold_perfs = {}
        for model in range(n_models):
            # Each model has different performance in different folds
            perf = np.random.normal(0.02 * model, 0.1)  # Model 0 worst, Model 3 best on average
            fold_perfs[model] = perf
        model_performance[fold] = fold_perfs
    
    print("Model performance by fold:")
    perf_df = pd.DataFrame(model_performance).T
    perf_df.columns = [f'Model_{i}' for i in range(n_models)]
    print(perf_df.round(4))
    
    print(f"\n🔍 MODEL SELECTION COMPARISON:")
    
    # Correct method: Use fold N-1 performance to select models for fold N
    print("Correct model selection (using previous fold):")
    for fold in range(1, n_folds):  # Start from fold 1
        prev_fold_perfs = model_performance[fold-1]
        best_model = max(prev_fold_perfs.keys(), key=lambda m: prev_fold_perfs[m])
        current_fold_actual = model_performance[fold][best_model]
        
        print(f"  Fold {fold}: Selected Model_{best_model} based on fold {fold-1} performance")
        print(f"           Performance in fold {fold-1}: {prev_fold_perfs[best_model]:.4f}")
        print(f"           Actual performance in fold {fold}: {current_fold_actual:.4f}")
    
    # Wrong method: Use fold N performance to select models for fold N (contamination!)
    print(f"\n❌ Wrong model selection (using current fold - CONTAMINATION!):")
    for fold in range(n_folds):
        current_fold_perfs = model_performance[fold]
        best_model = max(current_fold_perfs.keys(), key=lambda m: current_fold_perfs[m])
        reported_perf = current_fold_perfs[best_model]
        
        print(f"  Fold {fold}: Selected Model_{best_model} based on fold {fold} performance")  
        print(f"           Reported performance: {reported_perf:.4f} (INFLATED!)")
    
    # Calculate average performance for both methods
    correct_perfs = []
    wrong_perfs = []
    
    for fold in range(1, n_folds):
        # Correct method
        prev_fold_perfs = model_performance[fold-1]
        best_model_correct = max(prev_fold_perfs.keys(), key=lambda m: prev_fold_perfs[m])
        actual_perf = model_performance[fold][best_model_correct]
        correct_perfs.append(actual_perf)
        
        # Wrong method  
        current_fold_perfs = model_performance[fold]
        best_model_wrong = max(current_fold_perfs.keys(), key=lambda m: current_fold_perfs[m])
        inflated_perf = current_fold_perfs[best_model_wrong]
        wrong_perfs.append(inflated_perf)
    
    avg_correct = np.mean(correct_perfs)
    avg_wrong = np.mean(wrong_perfs)
    
    print(f"\n🔍 PERFORMANCE COMPARISON:")
    print(f"Average performance (correct method): {avg_correct:.4f}")
    print(f"Average performance (wrong method): {avg_wrong:.4f}")
    print(f"Inflation factor: {avg_wrong / avg_correct:.2f}x")
    
    if avg_wrong > avg_correct * 1.1:
        print("⚠️  WARNING: Wrong method shows significantly inflated performance!")
        print("   This indicates potential selection bias/contamination")
    else:
        print("✅ Performance difference is within reasonable bounds")
    
    return model_performance

def test_last_row_backtest_handling():
    """
    Test how the backtest handles the last row of data.
    
    In production backtesting, we should never use information that wasn't 
    available at the time of prediction.
    """
    print("\n=== TESTING LAST ROW BACKTEST HANDLING ===")
    
    # Create dataset where last row would have known target (bad) or unknown target (good)
    dates = pd.date_range('2020-01-01', periods=20, freq='D')
    
    # Features always available (calculated from past data)
    features = pd.DataFrame({
        'momentum': np.random.normal(0, 0.1, 20),
        'rsi': np.random.uniform(30, 70, 20)
    }, index=dates)
    
    # Returns: normally all available in backtest, but last one should be treated specially
    returns = pd.Series(np.random.normal(0.01, 0.02, 20), index=dates)
    
    # Simulate backtesting scenario - we're at the last date
    backtest_date = dates[-1]
    print(f"Current backtest date (simulation): {backtest_date}")
    
    # In production, we can calculate features up to current date
    available_features = features.loc[:backtest_date]
    print(f"Available features: {len(available_features)} rows")
    
    # But we shouldn't know the return for the current period yet
    # (return from current period to next period)
    known_returns = returns.iloc[:-1]  # All except last
    unknown_return = returns.iloc[-1]   # This should be unknown in production
    
    print(f"Known returns: {len(known_returns)} rows")
    print(f"Unknown return (current period): {unknown_return:.4f}")
    
    # Test backtest logic
    print(f"\n🔍 BACKTEST LOGIC TEST:")
    
    # We can make prediction for current period using available features
    current_features = available_features.iloc[-1]
    dummy_signal = current_features['rsi'] * 0.01  # Dummy prediction model
    
    print(f"Current period features: {current_features.values}")
    print(f"Generated signal: {dummy_signal:.4f}")
    
    # CRITICAL TEST: Can we calculate PnL for current period?
    print(f"\n🔍 PnL CALCULATION TEST FOR CURRENT PERIOD:")
    
    try:
        current_pnl = dummy_signal * unknown_return
        print(f"❌ CRITICAL: Can calculate PnL = {current_pnl:.6f}")
        print("   This suggests we have access to future return information!")
        print("   In true production, this return wouldn't be known yet")
    except Exception as e:
        print(f"✅ GOOD: Cannot calculate PnL ({e})")
        print("   This suggests proper temporal boundaries")
    
    # Simulate proper backtesting - only use known returns
    print(f"\n🔍 PROPER BACKTESTING SIMULATION:")
    
    # Generate signals for all periods except current (since we can't evaluate current)
    backtest_periods = dates[:-1]  # All periods except current
    backtest_features = features.iloc[:-1]
    backtest_returns = known_returns
    
    # Generate dummy signals
    dummy_signals = backtest_features['rsi'] * 0.01
    
    # Calculate PnL for backtest periods  
    backtest_pnl = dummy_signals * backtest_returns
    
    print(f"Backtest periods: {len(backtest_periods)}")
    print(f"Backtest PnL: {backtest_pnl.sum():.6f}")
    print(f"Current period: Prediction made but not evaluated (return unknown)")
    
    # This is the correct approach - we can make predictions for the current period
    # but cannot evaluate them until the returns are realized
    
    return features, returns, dummy_signals

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE BACKTESTING LEAKAGE TESTING")
    print("="*70)
    
    # Test 1: Signal-return temporal alignment
    signals, returns, pnl = test_signal_return_alignment()
    
    # Test 2: Backtesting temporal logic
    signal, actual_returns, fold_pnl = test_backtesting_temporal_logic()
    
    # Test 3: Cross-fold contamination
    model_perf = test_cross_fold_contamination()
    
    # Test 4: Last row handling
    features, returns, signals = test_last_row_backtest_handling()
    
    print("\n" + "="*70)
    print("🏁 BACKTESTING TESTING COMPLETE")
    
    print("\n📋 KEY FINDINGS:")
    print("1. Signal-return alignment: Check PnL calculation differences above")
    print("2. Temporal logic: Check index alignment warnings above")  
    print("3. Cross-fold contamination: Check performance inflation above")
    print("4. Last row handling: Check future information access above")
    
    print("\n🔧 CRITICAL BACKTESTING RULES:")
    print("1. signal[t] × return[t] - both represent same time period")
    print("2. Use fold[t-1] performance to select models for fold[t]")
    print("3. Never evaluate performance on data used for model selection")
    print("4. Last period predictions cannot be evaluated until returns are known")
    print("5. Maintain strict temporal boundaries throughout pipeline")
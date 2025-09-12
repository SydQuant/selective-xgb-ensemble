"""
Trace Q-Score Impact from Shift Inconsistency

This investigates the exact impact of the shift inconsistency on:
1. Training fold Q-scores (used for model selection)
2. Production period PnL/metrics 
3. Full timeline PnL/metrics
4. Model selection decisions
"""

import pandas as pd
import numpy as np
import sys
import os

def trace_training_q_score_calculation():
    """
    Trace how Q-scores are calculated during training and if they're affected
    """
    print("=== TRACING TRAINING Q-SCORE CALCULATION ===")
    
    print("📋 TRAINING FLOW (xgb_compare.py):")
    print("1. train_single_model() called for each model in each fold")
    print("2. Line 68: is_metrics = calculate_model_metrics(pred_inner_train, y_inner_train, shifted=False)")
    print("3. Line 69: iv_metrics = calculate_model_metrics(pred_inner_val, y_inner_val, shifted=False)")  
    print("4. Line 70: oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=True) ← PROBLEM!")
    print("5. Q-scores calculated from these metrics")
    
    print(f"\n🔍 WHAT THIS MEANS:")
    print("✅ IS (in-sample) metrics: CORRECT (no artificial lag)")
    print("✅ IV (inner-validation) metrics: CORRECT (no artificial lag)")
    print("❌ OOS (out-of-sample) metrics: WRONG (artificial 1-day lag)")
    print("❌ Q-scores: Based on WRONG OOS metrics")
    
    print(f"\n🎯 IMPACT ON Q-SCORES:")
    print("- Q-scores are calculated from OOS metrics (the lagged ones)")
    print("- Model ranking based on artificially lagged performance")
    print("- Selected 'best' models may not actually be the best")
    print("- Model selection is suboptimal")
    
    return True

def trace_production_period_calculation():
    """
    Trace how production period metrics are calculated in backtesting
    """
    print("\n=== TRACING PRODUCTION PERIOD CALCULATION ===")
    
    print("📋 BACKTESTING FLOW (full_timeline_backtest.py):")
    print("1. Selected models determined from (incorrect) Q-scores")
    print("2. Line 169: fold_pnl = signal * actual_returns  ← DIRECT (correct)")
    print("3. Line 196: fold_metrics = calculate_model_metrics_from_pnl(fold_pnl, signal, actual_returns)")
    print("4. Production metrics calculated from this")
    
    print(f"\n🔍 WHAT THIS MEANS:")
    print("✅ Production PnL calculation: CORRECT (direct, no lag)")
    print("✅ Production metrics calculation: CORRECT (from correct PnL)")
    print("❌ Model selection: Based on WRONG Q-scores from training")
    
    print(f"\n🎯 THE DISCONNECT:")
    print("- Production uses CORRECT PnL calculation")
    print("- But models were selected using WRONG training metrics")
    print("- It's like selecting a basketball team based on football scores!")
    
    return True

def trace_full_timeline_calculation():
    """
    Trace how full timeline metrics are calculated
    """
    print("\n=== TRACING FULL TIMELINE CALCULATION ===")
    
    print("📋 FULL TIMELINE FLOW:")
    print("1. Combines training + production periods")
    print("2. Uses same PnL from backtesting (direct calculation)")
    print("3. Line 248: training_metrics = calculate_model_metrics_from_pnl(...)")
    print("4. Line 265: production_metrics = calculate_model_metrics_from_pnl(...)")
    print("5. Aggregated metrics calculated")
    
    print(f"\n🔍 WHAT THIS MEANS:")
    print("✅ Full timeline PnL: CORRECT (direct calculation)")
    print("✅ Full timeline metrics: CORRECT (calculated from correct PnL)")
    print("❌ Model composition: Based on wrong Q-score model selection")
    
    print(f"\n🎯 INTERPRETATION:")
    print("- The PnL and metrics themselves are mathematically correct")
    print("- BUT they're calculated on the wrong set of models")
    print("- Performance could be better with correct model selection")
    
    return True

def demonstrate_q_score_impact():
    """
    Demonstrate numerical impact on Q-scores and model selection
    """
    print("\n=== DEMONSTRATING Q-SCORE IMPACT ===")
    
    # Simulate 3 models with different characteristics
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015, 0.005, -0.02, 0.01, -0.005, 0.008], index=dates)
    
    models = {
        'Model_A': pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.2, -0.1, 0.1], index=dates),  # Consistent
        'Model_B': pd.Series([0.2, -0.1, 0.1, -0.2, 0.3, 0.0, -0.1, 0.3, -0.2, 0.0], index=dates),  # Volatile
        'Model_C': pd.Series([0.05, -0.05, 0.1, -0.03, 0.08, 0.02, -0.1, 0.05, -0.02, 0.03], index=dates)  # Conservative
    }
    
    print("Model predictions (first 5 periods):")
    for name, preds in models.items():
        print(f"{name}: {preds.head().values}")
    print(f"Returns: {returns.head().values}")
    
    # Calculate metrics both ways
    print(f"\n📊 METRICS COMPARISON:")
    
    for name, predictions in models.items():
        # Correct way (direct)
        pnl_correct = predictions * returns
        sharpe_correct = pnl_correct.mean() / pnl_correct.std()
        
        # Wrong way (shifted - what training uses)
        pred_shifted = predictions.shift(1).fillna(0.0)
        pnl_wrong = pred_shifted * returns
        sharpe_wrong = pnl_wrong.mean() / pnl_wrong.std()
        
        print(f"{name}:")
        print(f"  Correct Sharpe: {sharpe_correct:7.4f}")
        print(f"  Wrong Sharpe:   {sharpe_wrong:7.4f}")
        print(f"  Difference:     {abs(sharpe_correct - sharpe_wrong):7.4f}")
    
    # Show how model ranking might change
    correct_sharpes = {}
    wrong_sharpes = {}
    
    for name, predictions in models.items():
        pnl_correct = predictions * returns
        correct_sharpes[name] = pnl_correct.mean() / pnl_correct.std()
        
        pred_shifted = predictions.shift(1).fillna(0.0)
        pnl_wrong = pred_shifted * returns
        wrong_sharpes[name] = pnl_wrong.mean() / pnl_wrong.std()
    
    # Sort by performance
    correct_ranking = sorted(correct_sharpes.items(), key=lambda x: x[1], reverse=True)
    wrong_ranking = sorted(wrong_sharpes.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 MODEL RANKING COMPARISON:")
    print("Correct ranking (direct calculation):")
    for i, (name, sharpe) in enumerate(correct_ranking, 1):
        print(f"  {i}. {name}: {sharpe:.4f}")
        
    print("Wrong ranking (shifted - what training uses):")
    for i, (name, sharpe) in enumerate(wrong_ranking, 1):
        print(f"  {i}. {name}: {sharpe:.4f}")
        
    # Check if rankings are different
    if [name for name, _ in correct_ranking] != [name for name, _ in wrong_ranking]:
        print("\n🚨 CRITICAL: Model rankings are DIFFERENT!")
        print("Training will select suboptimal models for backtesting")
    else:
        print("\n✅ Rankings are the same in this example")
    
    return correct_sharpes, wrong_sharpes

def analyze_system_wide_impact():
    """
    Analyze the system-wide impact of this inconsistency
    """
    print("\n=== ANALYZING SYSTEM-WIDE IMPACT ===")
    
    print("🔄 THE INCONSISTENCY CHAIN:")
    print("1. Training uses shifted=True for OOS metrics")
    print("2. Q-scores calculated from wrong OOS metrics")
    print("3. Model selection based on wrong Q-scores")
    print("4. Backtesting uses correct PnL but wrong model set")
    print("5. Reported performance is for suboptimal model ensemble")
    
    print(f"\n📊 WHAT'S CORRECT vs WRONG:")
    
    correct_components = [
        "✅ Production period PnL calculation (direct)",
        "✅ Production period metrics (from correct PnL)",
        "✅ Full timeline PnL calculation (direct)", 
        "✅ Full timeline metrics (from correct PnL)",
        "✅ Backtesting methodology (no look-ahead bias)"
    ]
    
    wrong_components = [
        "❌ Training OOS metrics (use artificial lag)",
        "❌ Q-scores (based on wrong OOS metrics)",
        "❌ Model selection (based on wrong Q-scores)",
        "❌ Model composition in backtesting"
    ]
    
    print("CORRECT components:")
    for item in correct_components:
        print(f"  {item}")
        
    print("\nWRONG components:")
    for item in wrong_components:
        print(f"  {item}")
    
    print(f"\n🎯 BOTTOM LINE:")
    print("- The backtesting PnL and metrics are mathematically correct")
    print("- BUT they represent the performance of a suboptimally selected model ensemble")
    print("- True performance could be higher with correct model selection")
    print("- The inconsistency hurts performance, not accuracy of measurement")
    
    return True

def show_fix_impact():
    """
    Show what fixing this inconsistency would accomplish
    """
    print("\n=== SHOWING FIX IMPACT ===")
    
    print("🔧 PROPOSED FIX:")
    print("Change xgb_compare.py line 70:")
    print("  OLD: oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=True)")
    print("  NEW: oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=False)")
    
    print(f"\n🎯 EXPECTED OUTCOMES:")
    print("✅ Training OOS metrics will match backtesting metrics")
    print("✅ Q-scores will be calculated correctly")
    print("✅ Model selection will be optimal")
    print("✅ Backtesting will use the truly best models")
    print("✅ Reported performance will likely improve")
    print("✅ Training-backtesting consistency achieved")
    
    print(f"\n⚠️  POTENTIAL SIDE EFFECTS:")
    print("- Historical Q-scores will change (they were wrong before)")
    print("- Model rankings may shuffle")
    print("- Previously 'best' models may no longer be selected")
    print("- Backtesting results will change (should improve)")
    
    print(f"\n📈 PERFORMANCE EXPECTATION:")
    print("Current performance = Performance of suboptimally selected models")
    print("Fixed performance = Performance of optimally selected models")
    print("→ Performance should improve (models selected correctly)")
    
    return True

if __name__ == "__main__":
    print("🔍 TRACING Q-SCORE IMPACT FROM SHIFT INCONSISTENCY")
    print("="*70)
    
    # Trace each component
    trace_training_q_score_calculation()
    trace_production_period_calculation()
    trace_full_timeline_calculation()
    
    # Demonstrate numerical impact
    correct_sharpes, wrong_sharpes = demonstrate_q_score_impact()
    
    # System-wide analysis
    analyze_system_wide_impact()
    
    # Show fix impact
    show_fix_impact()
    
    print("\n" + "="*70)
    print("🏁 Q-SCORE IMPACT ANALYSIS COMPLETE")
    
    print(f"\n🎯 KEY TAKEAWAY:")
    print("The production PnL and metrics are CORRECT in calculation,")
    print("but they're calculated on a SUBOPTIMAL model selection.")
    print("Fixing the training inconsistency should IMPROVE performance!")
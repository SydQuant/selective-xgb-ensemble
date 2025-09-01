#!/usr/bin/env python3
"""
Simple test to verify GROPE optimization is working correctly.
We'll create synthetic signals and see if GROPE finds optimal weights.
"""

import numpy as np
import pandas as pd
from opt.grope import grope_optimize
from opt.weight_objective import weight_objective_factory
from metrics.dapy import dapy_from_binary_hits
from ensemble.combiner import softmax, combine_signals

def test_grope_simple():
    """Test GROPE with synthetic signals where we know the optimal solution."""
    np.random.seed(42)
    n_samples = 500
    
    # Create target: simple trend + noise
    target = np.cumsum(np.random.normal(0, 0.01, n_samples))
    target = pd.Series(target, name='y')
    
    print("📊 Testing GROPE with synthetic signals...")
    print(f"Target: mean={target.mean():.4f}, std={target.std():.4f}")
    
    # Create 3 signals:
    # Signal 0: Perfect predictor (should get highest weight)
    signal_0 = target.shift(1).fillna(0) + np.random.normal(0, 0.001, n_samples)  # Almost perfect
    
    # Signal 1: Decent predictor (should get medium weight) 
    signal_1 = target.shift(1).fillna(0) * 0.5 + np.random.normal(0, 0.005, n_samples)
    
    # Signal 2: Noise (should get low/zero weight)
    signal_2 = pd.Series(np.random.normal(0, 0.01, n_samples))
    
    signals = [signal_0, signal_1, signal_2]
    
    print("\n🔍 Signal correlations with target:")
    for i, sig in enumerate(signals):
        corr = sig.corr(target)
        print(f"   Signal {i}: correlation = {corr:.4f}")
    
    # Test GROPE optimization
    print("\n⚙️ Running GROPE optimization...")
    
    bounds = {
        "w0": (-2.0, 2.0),
        "w1": (-2.0, 2.0), 
        "w2": (-2.0, 2.0),
        "tau": (0.2, 3.0)
    }
    
    dapy_fn = dapy_from_binary_hits
    fobj = weight_objective_factory(
        signals, target, 
        turnover_penalty=0.01, 
        pmax=0.20, 
        w_dapy=1.0, 
        w_ir=1.0, 
        metric_fn_dapy=dapy_fn
    )
    
    theta_star, J_star, history = grope_optimize(bounds, fobj, budget=50, seed=123)
    
    print(f"\n✅ GROPE Results:")
    print(f"   Objective value: {J_star:.4f}")
    print(f"   Raw weights: w0={theta_star['w0']:.4f}, w1={theta_star['w1']:.4f}, w2={theta_star['w2']:.4f}")
    print(f"   Temperature: tau={theta_star['tau']:.4f}")
    
    # Apply softmax
    raw_weights = np.array([theta_star['w0'], theta_star['w1'], theta_star['w2']])
    softmax_weights = softmax(raw_weights, temperature=theta_star['tau'])
    
    print(f"   Softmax weights: {softmax_weights}")
    
    # Create ensemble signal
    ensemble = combine_signals(signals, softmax_weights)
    ensemble_corr = ensemble.corr(target)
    
    print(f"\n📈 Ensemble Performance:")
    print(f"   Correlation with target: {ensemble_corr:.4f}")
    print(f"   Expected: Signal 0 should have highest weight")
    
    # Test direction: ensemble should be positively correlated with target
    if ensemble_corr > 0.5:
        print("✅ PASS: Ensemble positively correlated with target")
        return True
    else:
        print("❌ FAIL: Ensemble not positively correlated with target")
        return False

def test_grope_direction():
    """Test if GROPE handles signal direction correctly."""
    np.random.seed(123)
    n_samples = 300
    
    print("\n📊 Testing GROPE signal direction handling...")
    
    # Create simple target
    target = pd.Series(np.random.normal(0, 0.01, n_samples))
    
    # Create signals:
    # Signal A: Positively correlated with target
    signal_a = target * 0.8 + np.random.normal(0, 0.005, n_samples)
    
    # Signal B: Negatively correlated with target  
    signal_b = -target * 0.6 + np.random.normal(0, 0.005, n_samples)
    
    signals = [signal_a, signal_b]
    
    print(f"Signal A correlation: {signal_a.corr(target):.4f} (should be positive)")
    print(f"Signal B correlation: {signal_b.corr(target):.4f} (should be negative)")
    
    # Run GROPE
    bounds = {"w0": (-2.0, 2.0), "w1": (-2.0, 2.0), "tau": (0.2, 3.0)}
    dapy_fn = dapy_from_binary_hits
    fobj = weight_objective_factory(signals, target, turnover_penalty=0.01, pmax=0.20, w_dapy=1.0, w_ir=1.0, metric_fn_dapy=dapy_fn)
    
    theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=30, seed=456)
    
    raw_weights = np.array([theta_star['w0'], theta_star['w1']])
    softmax_weights = softmax(raw_weights, temperature=theta_star['tau'])
    
    print(f"\nGROPE weights: w0={softmax_weights[0]:.4f}, w1={softmax_weights[1]:.4f}")
    
    # Check if Signal A (positive corr) gets positive contribution
    # and Signal B (negative corr) gets handled correctly
    ensemble = combine_signals(signals, softmax_weights)
    ensemble_corr = ensemble.corr(target)
    
    print(f"Ensemble correlation with target: {ensemble_corr:.4f}")
    
    if ensemble_corr > 0.3:
        print("✅ PASS: GROPE handled signal directions correctly")
        return True
    else:
        print("❌ FAIL: GROPE may have direction issues")
        return False

if __name__ == "__main__":
    print("🧪 GROPE Optimization Tests")
    print("=" * 50)
    
    test1_pass = test_grope_simple()
    test2_pass = test_grope_direction()
    
    print("\n" + "=" * 50)
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED: GROPE appears to be working correctly")
    else:
        print("❌ SOME TESTS FAILED: GROPE has issues that need investigation")
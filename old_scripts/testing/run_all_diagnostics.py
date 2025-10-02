#!/usr/bin/env python3
"""
Complete Framework Diagnostic Suite
===================================

Actually RUN all diagnostics using the real framework code.
No assumptions - test everything systematically.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# Use ACTUAL framework imports
from data.data_utils_simple import prepare_real_data_simple
from model.xgb_drivers import create_standard_xgb_bank, fit_xgb_on_slice
from model.feature_selection import apply_feature_selection

def test_actual_xgb_training_pipeline(symbol="@ES#C"):
    """
    🚨 CRITICAL: Test the actual XGB training pipeline for bugs
    """
    print(f"\n{'='*80}")
    print(f"🚨 TESTING ACTUAL XGB TRAINING PIPELINE - {symbol}")
    print(f"{'='*80}")

    issues = []

    try:
        # Step 1: Load data exactly as framework does
        print("1. Loading data with prepare_real_data_simple()...")
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        # Split exactly as xgb_compare.py line 46
        X = df[[c for c in df.columns if c != target_col]]
        y = df[target_col]

        print(f"   Loaded: X={X.shape}, y={y.shape}")

        # Check for NaN in target
        y_clean = y.dropna()
        if len(y_clean) < len(y):
            print(f"   Dropped {len(y) - len(y_clean)} NaN targets")

        # Step 2: Apply feature selection exactly as framework
        print("2. Testing feature selection...")
        X_selected = apply_feature_selection(X, y, method='block_wise', max_total_features=50)
        print(f"   Selected features: {X_selected.shape[1]}")

        # Get clean aligned data
        common_idx = X_selected.index.intersection(y_clean.index)
        X_clean = X_selected.loc[common_idx]
        y_clean_aligned = y_clean.loc[common_idx]

        print(f"   Clean aligned data: X={X_clean.shape}, y={len(y_clean_aligned)}")

        # Step 3: Test actual XGB model training
        print("3. Testing XGB model training...")

        # Create models exactly as framework does
        models_bank = create_standard_xgb_bank(3)  # Just 3 for testing

        # Simple train/test split
        split_point = int(len(X_clean) * 0.8)
        X_train = X_clean.iloc[:split_point]
        y_train = y_clean_aligned.iloc[:split_point]
        X_test = X_clean.iloc[split_point:]
        y_test = y_clean_aligned.iloc[split_point:]

        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

        # Train models and get predictions
        model_results = []

        for i, model_spec in enumerate(models_bank):
            print(f"   Training model {i+1}/3...")

            # Train exactly as framework does
            model = fit_xgb_on_slice(X_train, y_train, model_spec)

            # Get predictions
            train_preds = model.predict(X_train.values)
            test_preds = model.predict(X_test.values)

            # Check prediction quality
            train_corr = np.corrcoef(train_preds, y_train.values)[0,1] if len(y_train) > 1 else 0
            test_corr = np.corrcoef(test_preds, y_test.values)[0,1] if len(y_test) > 1 else 0

            model_results.append({
                'model_idx': i,
                'train_corr': train_corr,
                'test_corr': test_corr,
                'train_preds': train_preds,
                'test_preds': test_preds
            })

            print(f"     Train correlation: {train_corr:.4f}")
            print(f"     Test correlation: {test_corr:.4f}")

            # Flag suspicious correlations
            if abs(train_corr) > 0.8:
                issue = f"CRITICAL: Model {i} has suspiciously high training correlation ({train_corr:.4f})"
                issues.append(issue)
                print(f"🚨 {issue}")

            if abs(test_corr) > abs(train_corr) + 0.3:
                issue = f"CRITICAL: Model {i} test correlation ({test_corr:.4f}) much higher than training ({train_corr:.4f})"
                issues.append(issue)
                print(f"🚨 {issue}")

        # Step 4: Test ensemble combination
        print("4. Testing ensemble signal combination...")

        # Extract test predictions
        test_preds_matrix = np.array([r['test_preds'] for r in model_results])

        # Test tanh transformation
        tanh_signals = np.tanh(test_preds_matrix)
        ensemble_tanh = np.mean(tanh_signals, axis=0)

        # Test binary transformation
        binary_signals = np.where(test_preds_matrix > 0, 1, -1)
        ensemble_binary = np.sum(binary_signals, axis=0)

        print(f"   Ensemble tanh range: [{ensemble_tanh.min():.4f}, {ensemble_tanh.max():.4f}]")
        print(f"   Ensemble binary range: [{ensemble_binary.min()}, {ensemble_binary.max()}]")

        # Test ensemble-target correlation
        ensemble_target_corr = np.corrcoef(ensemble_tanh, y_test.values)[0,1] if len(y_test) > 1 else 0
        print(f"   Ensemble-target correlation: {ensemble_target_corr:.4f}")

        # Step 5: Test PnL calculation
        print("5. Testing PnL calculation...")

        # Calculate PnL using ensemble signals
        pnl_tanh = ensemble_tanh * y_test.values
        pnl_binary = (ensemble_binary / len(model_results)) * y_test.values  # Normalize binary votes

        # Calculate Sharpe ratios
        sharpe_tanh = np.mean(pnl_tanh) / np.std(pnl_tanh) * np.sqrt(252) if np.std(pnl_tanh) > 0 else 0
        sharpe_binary = np.mean(pnl_binary) / np.std(pnl_binary) * np.sqrt(252) if np.std(pnl_binary) > 0 else 0

        print(f"   Tanh Sharpe: {sharpe_tanh:.4f}")
        print(f"   Binary Sharpe: {sharpe_binary:.4f}")

        # Flag unrealistic performance
        if abs(sharpe_tanh) > 5:
            issue = f"WARNING: Unrealistic tanh Sharpe ({sharpe_tanh:.2f}) in test"
            issues.append(issue)
            print(f"⚠️  {issue}")

        if abs(sharpe_binary) > 5:
            issue = f"WARNING: Unrealistic binary Sharpe ({sharpe_binary:.2f}) in test"
            issues.append(issue)
            print(f"⚠️  {issue}")

        print(f"\n✅ XGB training pipeline test completed")

    except Exception as e:
        error = f"CRITICAL: Error in XGB pipeline test: {str(e)}"
        issues.append(error)
        print(f"🚨 {error}")
        import traceback
        traceback.print_exc()

    return issues

def test_cross_validation_implementation():
    """
    🚨 CRITICAL: Test the actual CV implementation for contamination
    """
    print(f"\n{'='*80}")
    print("🚨 TESTING ACTUAL CROSS-VALIDATION IMPLEMENTATION")
    print(f"{'='*80}")

    issues = []

    try:
        # Import actual CV function
        from cv.wfo import wfo_splits

        # Test CV splits
        n_samples = 1000
        n_folds = 5

        print(f"Testing CV splits: {n_samples} samples, {n_folds} folds")

        splits = list(wfo_splits(n_samples, k_folds=n_folds))

        print(f"Generated {len(splits)} splits")

        # Check each split for contamination
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f"\nFold {i+1}:")
            print(f"  Train: {len(train_idx)} samples [{min(train_idx)}-{max(train_idx)}]")
            print(f"  Test: {len(test_idx)} samples [{min(test_idx)}-{max(test_idx)}]")

            # Critical check: No overlap between train and test
            train_set = set(train_idx)
            test_set = set(test_idx)
            overlap = train_set.intersection(test_set)

            if len(overlap) > 0:
                issue = f"CRITICAL: Fold {i+1} has {len(overlap)} overlapping indices"
                issues.append(issue)
                print(f"🚨 {issue}")
            else:
                print(f"  ✅ No overlap detected")

            # Check temporal ordering (for expanding window)
            max_train = max(train_idx)
            min_test = min(test_idx)

            if min_test <= max_train:
                gap = max_train - min_test + 1
                if gap > len(test_idx) * 0.1:  # Significant overlap
                    issue = f"CRITICAL: Fold {i+1} has significant temporal overlap ({gap} indices)"
                    issues.append(issue)
                    print(f"🚨 {issue}")
                else:
                    print(f"  ✅ Minor gap acceptable for expanding window")
            else:
                print(f"  ✅ Perfect temporal separation")

        print(f"\n✅ Cross-validation implementation tested")

    except ImportError as e:
        error = f"CRITICAL: Cannot import CV function: {e}"
        issues.append(error)
        print(f"🚨 {error}")
    except Exception as e:
        error = f"CRITICAL: Error in CV test: {e}"
        issues.append(error)
        print(f"🚨 {error}")
        import traceback
        traceback.print_exc()

    return issues

def test_feature_selection_contamination():
    """
    🚨 CRITICAL: Test feature selection for temporal contamination
    """
    print(f"\n{'='*80}")
    print("🚨 TESTING FEATURE SELECTION FOR CONTAMINATION")
    print(f"{'='*80}")

    issues = []

    try:
        # Load real data
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        X = df[[c for c in df.columns if c != target_col]]
        y = df[target_col]

        print(f"Testing feature selection with {X.shape[1]} features...")

        # Test: Feature selection on partial data vs full data
        # If selection uses future data, results will be different

        split_point = len(df) // 2

        # Partial data (first half only)
        X_partial = X.iloc[:split_point]
        y_partial = y.iloc[:split_point]

        print(f"\nComparing feature selection:")
        print(f"  Full data: {len(y)} samples")
        print(f"  Partial data: {len(y_partial)} samples")

        # Apply feature selection to both
        X_selected_full = apply_feature_selection(X, y, method='block_wise', max_total_features=20)
        X_selected_partial = apply_feature_selection(X_partial, y_partial, method='block_wise', max_total_features=20)

        print(f"  Full data selected: {X_selected_full.shape[1]} features")
        print(f"  Partial data selected: {X_selected_partial.shape[1]} features")

        # Compare selected features
        full_features = set(X_selected_full.columns)
        partial_features = set(X_selected_partial.columns)

        common_features = full_features.intersection(partial_features)
        only_full = full_features - partial_features
        only_partial = partial_features - full_features

        print(f"  Common features: {len(common_features)}")
        print(f"  Only in full: {len(only_full)}")
        print(f"  Only in partial: {len(only_partial)}")

        overlap_rate = len(common_features) / max(len(full_features), len(partial_features))
        print(f"  Feature stability: {overlap_rate:.4f} ({overlap_rate*100:.1f}%)")

        # If feature selection is contaminated, overlap should be very low
        if overlap_rate < 0.3:
            issue = f"WARNING: Low feature selection stability ({overlap_rate:.2f}) - possible contamination"
            issues.append(issue)
            print(f"⚠️  {issue}")

        print(f"✅ Feature selection contamination test completed")

    except Exception as e:
        error = f"CRITICAL: Error in feature selection test: {e}"
        issues.append(error)
        print(f"🚨 {error}")
        import traceback
        traceback.print_exc()

    return issues

def test_actual_signal_generation():
    """
    🚨 CRITICAL: Test actual signal generation with real models
    """
    print(f"\n{'='*80}")
    print("🚨 TESTING ACTUAL SIGNAL GENERATION")
    print(f"{'='*80}")

    issues = []

    try:
        # Use small dataset for quick test
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        # Clean data
        clean_df = df.dropna(subset=[target_col])[-200:]  # Last 200 days
        X = clean_df[[c for c in clean_df.columns if c != target_col]]
        y = clean_df[target_col]

        print(f"Testing with {len(y)} recent samples...")

        # Apply feature selection
        X_selected = apply_feature_selection(X, y, method='block_wise', max_total_features=30)

        # Simple train/test split
        split = int(len(X_selected) * 0.7)
        X_train = X_selected.iloc[:split]
        y_train = y.iloc[:split]
        X_test = X_selected.iloc[split:]
        y_test = y.iloc[split:]

        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        # Train actual XGB models
        models_bank = create_standard_xgb_bank(3)
        trained_models = []

        for i, model_spec in enumerate(models_bank):
            print(f"Training model {i+1}...")

            # Train model exactly as framework does
            model = fit_xgb_on_slice(X_train, y_train, model_spec)

            # Get predictions
            train_preds = model.predict(X_train.values)
            test_preds = model.predict(X_test.values)

            trained_models.append({
                'model': model,
                'train_preds': train_preds,
                'test_preds': test_preds
            })

            # Analyze predictions
            print(f"   Train preds: mean={np.mean(train_preds):.4f}, std={np.std(train_preds):.4f}")
            print(f"   Test preds: mean={np.mean(test_preds):.4f}, std={np.std(test_preds):.4f}")

            # Check for suspicious prediction patterns
            if np.std(train_preds) < 1e-6:
                issue = f"CRITICAL: Model {i} produces constant predictions (std={np.std(train_preds):.8f})"
                issues.append(issue)
                print(f"🚨 {issue}")

            if np.std(test_preds) < 1e-6:
                issue = f"CRITICAL: Model {i} produces constant test predictions"
                issues.append(issue)
                print(f"🚨 {issue}")

        # Test signal transformations
        print(f"\n4. Testing signal transformations...")

        # Get test predictions from all models
        all_test_preds = np.array([m['test_preds'] for m in trained_models])

        # Apply transformations exactly as framework does
        tanh_signals = np.tanh(all_test_preds)
        binary_signals = np.where(all_test_preds > 0, 1, -1)

        # Create ensemble signals
        ensemble_tanh = np.mean(tanh_signals, axis=0)
        ensemble_binary = np.sum(binary_signals, axis=0)

        print(f"   Individual tanh range: [{tanh_signals.min():.4f}, {tanh_signals.max():.4f}]")
        print(f"   Individual binary values: {np.unique(binary_signals.flatten())}")
        print(f"   Ensemble tanh range: [{ensemble_tanh.min():.4f}, {ensemble_tanh.max():.4f}]")
        print(f"   Ensemble binary range: [{ensemble_binary.min()}, {ensemble_binary.max()}]")

        # Test PnL calculation
        pnl_tanh = ensemble_tanh * y_test.values
        pnl_binary = (ensemble_binary / len(trained_models)) * y_test.values

        sharpe_tanh = np.mean(pnl_tanh) / np.std(pnl_tanh) * np.sqrt(252) if np.std(pnl_tanh) > 0 else 0
        sharpe_binary = np.mean(pnl_binary) / np.std(pnl_binary) * np.sqrt(252) if np.std(pnl_binary) > 0 else 0

        print(f"   Tanh PnL Sharpe: {sharpe_tanh:.4f}")
        print(f"   Binary PnL Sharpe: {sharpe_binary:.4f}")

        # Show sample signals
        print(f"\n5. Sample signal analysis (first 10 test samples):")
        print(f"{'Date':<12} {'Target':<8} {'Tanh':<8} {'Binary':<6} {'PnL_T':<8} {'PnL_B':<8}")
        print(f"{'-'*60}")

        for i in range(min(10, len(y_test))):
            date = y_test.index[i].strftime('%Y-%m-%d')
            target = y_test.iloc[i]
            tanh_sig = ensemble_tanh[i]
            binary_sig = ensemble_binary[i]
            pnl_t = pnl_tanh[i]
            pnl_b = pnl_binary[i]

            print(f"{date:<12} {target:>+6.4f}{'':2} {tanh_sig:>+6.3f}{'':2} {binary_sig:>+4d}{'':2} {pnl_t:>+6.4f}{'':2} {pnl_b:>+6.4f}")

        print(f"\n✅ Signal generation test completed")

    except Exception as e:
        error = f"CRITICAL: Error in signal generation test: {e}"
        issues.append(error)
        print(f"🚨 {error}")
        import traceback
        traceback.print_exc()

    return issues

def main():
    """
    Run ALL comprehensive framework tests
    """
    print(f"\n{'='*100}")
    print("🚨🚨🚨 COMPREHENSIVE FRAMEWORK DIAGNOSTIC SUITE")
    print(f"{'='*100}")
    print("ACTUALLY TESTING ALL CRITICAL AREAS OF THE FRAMEWORK")
    print(f"Test run: {datetime.now()}")

    all_issues = []

    # Test 1: Actual XGB training pipeline
    print(f"\n🧪 TEST 1: XGB TRAINING PIPELINE")
    issues = test_actual_xgb_training_pipeline("@ES#C")
    all_issues.extend(issues)

    # Test 2: Cross-validation implementation
    print(f"\n🧪 TEST 2: CROSS-VALIDATION IMPLEMENTATION")
    issues = test_cross_validation_implementation()
    all_issues.extend(issues)

    # Test 3: Feature selection contamination
    print(f"\n🧪 TEST 3: FEATURE SELECTION CONTAMINATION")
    issues = test_feature_selection_contamination()
    all_issues.extend(issues)

    # Test 4: Actual signal generation
    print(f"\n🧪 TEST 4: SIGNAL GENERATION")
    issues = test_actual_signal_generation()
    all_issues.extend(issues)

    # FINAL VERDICT
    print(f"\n{'='*100}")
    print("🚨🚨🚨 FINAL FRAMEWORK VALIDATION VERDICT")
    print(f"{'='*100}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"COMPREHENSIVE TEST RESULTS:")
    print(f"  Total issues: {len(all_issues)}")
    print(f"  Critical issues: {len(critical_issues)}")
    print(f"  Warning issues: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨🚨🚨 CRITICAL FRAMEWORK BUGS DETECTED:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")
        print(f"\n❌❌❌ FRAMEWORK IS INVALID - ALL RESULTS MEANINGLESS! ❌❌❌")
    else:
        print(f"\n✅✅✅ NO CRITICAL BUGS DETECTED IN COMPREHENSIVE TESTING")
        print("Framework appears fundamentally sound")

    if warning_issues:
        print(f"\n⚠️  WARNINGS REQUIRING ATTENTION:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    # Save complete diagnostic report
    results_dir = Path("testing/comprehensive_results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = results_dir / f"comprehensive_framework_validation_{timestamp}.txt"

    with open(report_file, 'w') as f:
        f.write("COMPREHENSIVE FRAMEWORK VALIDATION REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total Issues: {len(all_issues)}\n")
        f.write(f"Critical Issues: {len(critical_issues)}\n")
        f.write(f"Warning Issues: {len(warning_issues)}\n\n")

        if critical_issues:
            f.write("CRITICAL ISSUES:\n")
            for i, issue in enumerate(critical_issues, 1):
                f.write(f"{i}. {issue}\n")
            f.write("\n")

        if warning_issues:
            f.write("WARNING ISSUES:\n")
            for i, issue in enumerate(warning_issues, 1):
                f.write(f"{i}. {issue}\n")
            f.write("\n")

        f.write("CONCLUSION:\n")
        if critical_issues:
            f.write("FRAMEWORK IS INVALID - CRITICAL BUGS DETECTED\n")
        else:
            f.write("FRAMEWORK VALIDATION PASSED - NO CRITICAL BUGS\n")

    print(f"\n📁 Complete diagnostic report saved to: {report_file}")

    return all_issues

if __name__ == "__main__":
    main()
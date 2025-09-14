#!/usr/bin/env python3
"""
Immediate Framework Diagnostics - Critical Bug Hunter
====================================================

This script uses the ACTUAL framework data loading to detect critical bugs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# Import the ACTUAL data loading function used by framework
from data.data_utils_simple import prepare_real_data_simple

def immediate_bug_check(symbol="@ES#C"):
    """
    Run immediate critical bug checks using the actual framework pipeline
    """
    print(f"\n{'='*80}")
    print(f"🚨 IMMEDIATE CRITICAL BUG CHECK - {symbol}")
    print(f"{'='*80}")
    print(f"Using ACTUAL framework data loading pipeline")
    print(f"Check time: {datetime.now()}")

    critical_issues = []
    warnings = []

    try:
        print(f"\n1. Loading data using framework's prepare_real_data_simple()...")

        # Load data exactly as the framework does
        df = prepare_real_data_simple(symbol)

        print(f"   ✅ Data loaded successfully")
        print(f"   Shape: {df.shape}")
        print(f"   Date range: {df.index.min()} to {df.index.max()}")

        # Extract target and features exactly as framework does
        target_col = f"{symbol}_target_return"

        if target_col not in df.columns:
            error = f"CRITICAL: Target column {target_col} not found in data!"
            critical_issues.append(error)
            print(f"🚨 {error}")
            return critical_issues

        # Split exactly as framework does: line 46 in xgb_compare.py
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = df[target_col]

        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Feature columns: {len(feature_cols)}")

        print(f"\n2. 🚨 ULTIMATE CRITICAL CHECK: Target in Features")

        # Check if target column accidentally included in features
        if target_col in feature_cols:
            error = f"ULTIMATE CRITICAL: Target column {target_col} is in feature list!"
            critical_issues.append(error)
            print(f"🚨🚨🚨 {error}")

        # Check for any column containing target values
        target_values = y.dropna().values

        for col in feature_cols[:50]:  # Check first 50 features
            feature_values = X[col].dropna().values

            if len(feature_values) == len(target_values):
                # Check for exact match
                aligned_feature = X[col].reindex(y.dropna().index).values

                if len(aligned_feature) == len(target_values):
                    diff = np.abs(aligned_feature - target_values)
                    perfect_matches = np.sum(diff < 1e-12)

                    if perfect_matches > len(target_values) * 0.9:
                        error = f"ULTIMATE CRITICAL: Feature '{col}' is identical to target variable!"
                        critical_issues.append(error)
                        print(f"🚨🚨🚨 {error}")
                        print(f"    {perfect_matches}/{len(target_values)} values are identical")

        print(f"\n3. 🚨 Target Variable Analysis")

        target_clean = y.dropna()
        print(f"   Non-null targets: {len(target_clean)}")
        print(f"   Mean return: {target_clean.mean():.6f}")
        print(f"   Std return: {target_clean.std():.6f}")
        print(f"   Min return: {target_clean.min():.6f}")
        print(f"   Max return: {target_clean.max():.6f}")

        # Check for suspicious target characteristics
        if target_clean.std() < 1e-8:
            error = f"CRITICAL: Target has zero variance (std: {target_clean.std():.10f})"
            critical_issues.append(error)
            print(f"🚨 {error}")

        if abs(target_clean.mean()) > 0.01:  # 1% daily return average is unrealistic
            warning = f"WARNING: Very high average return ({target_clean.mean():.4f}) - possible calculation error"
            warnings.append(warning)
            print(f"⚠️  {warning}")

        if target_clean.max() > 1.0:  # 100% single day return is suspicious
            error = f"CRITICAL: Extremely large return detected ({target_clean.max():.4f}) - possible calculation error"
            critical_issues.append(error)
            print(f"🚨 {error}")

        print(f"\n4. 🚨 Quick Feature Analysis")

        # Check for suspicious feature names
        suspicious_names = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['return', 'pnl', 'target', 'future', 'profit', 'loss']):
                suspicious_names.append(col)

        if suspicious_names:
            warning = f"WARNING: Suspicious feature names: {suspicious_names[:10]}"
            warnings.append(warning)
            print(f"⚠️  {warning}")

        # Quick correlation check on first 20 features
        high_corr_features = []
        for col in feature_cols[:20]:
            try:
                # Align feature and target
                common_idx = X[col].dropna().index.intersection(y.dropna().index)
                if len(common_idx) < 10:
                    continue

                aligned_feature = X[col].loc[common_idx].values
                aligned_target = y.loc[common_idx].values

                if np.std(aligned_feature) > 1e-10 and np.std(aligned_target) > 1e-10:
                    corr = np.corrcoef(aligned_feature, aligned_target)[0, 1]
                    if abs(corr) > 0.8:
                        high_corr_features.append((col, corr))
            except:
                continue

        if high_corr_features:
            for col, corr in high_corr_features:
                if abs(corr) > 0.95:
                    error = f"ULTIMATE CRITICAL: Feature '{col}' has perfect correlation with target ({corr:.6f})"
                    critical_issues.append(error)
                    print(f"🚨🚨🚨 {error}")
                else:
                    warning = f"WARNING: Feature '{col}' has high correlation with target ({corr:.4f})"
                    warnings.append(warning)
                    print(f"⚠️  {warning}")

        print(f"\n5. 🚨 Data Integrity Checks")

        # Check for NaN/inf in features
        feature_nan_cols = []
        for col in feature_cols:
            if X[col].isnull().sum() > len(X) * 0.5:  # More than 50% missing
                feature_nan_cols.append(col)

        if feature_nan_cols:
            warning = f"WARNING: Features with >50% missing values: {len(feature_nan_cols)} columns"
            warnings.append(warning)
            print(f"⚠️  {warning}")

        # Check for constant features
        constant_features = []
        for col in feature_cols[:100]:  # Check first 100
            values = X[col].dropna()
            if len(values) > 10 and values.std() < 1e-10:
                constant_features.append(col)

        if constant_features:
            warning = f"WARNING: Constant features detected: {constant_features[:5]}"
            warnings.append(warning)
            print(f"⚠️  {warning}")

    except Exception as e:
        error = f"CRITICAL: Error during immediate diagnostics: {str(e)}"
        critical_issues.append(error)
        print(f"🚨 {error}")
        import traceback
        traceback.print_exc()

    # SUMMARY
    print(f"\n{'='*80}")
    print(f"🚨 IMMEDIATE DIAGNOSTICS SUMMARY - {symbol}")
    print(f"{'='*80}")

    print(f"Critical issues: {len(critical_issues)}")
    print(f"Warnings: {len(warnings)}")

    if critical_issues:
        print(f"\n🚨🚨🚨 CRITICAL ISSUES (Framework Invalid):")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")

    if warnings:
        print(f"\n⚠️  WARNINGS (Need Investigation):")
        for i, issue in enumerate(warnings[:10], 1):
            print(f"  {i}. {issue}")

    if not critical_issues:
        print(f"\n✅ No critical bugs detected in immediate check!")
        print(f"   Framework data pipeline appears clean")
    else:
        print(f"\n❌ CRITICAL BUGS DETECTED!")
        print(f"   🚨🚨🚨 ALL FRAMEWORK RESULTS MAY BE INVALID! 🚨🚨🚨")

    return critical_issues, warnings

def main():
    """Run immediate diagnostics on key symbols"""
    print(f"\n{'='*100}")
    print(f"🚨🚨🚨 IMMEDIATE FRAMEWORK DIAGNOSTICS")
    print(f"{'='*100}")
    print("Checking for CRITICAL bugs that would invalidate ALL results")
    print("Using the ACTUAL framework data loading pipeline")

    # Test symbols: good performer, poor performer, standard
    test_symbols = ["@ES#C", "QCL#C"]

    all_critical = []
    all_warnings = []

    for symbol in test_symbols:
        critical, warnings = immediate_bug_check(symbol)
        all_critical.extend(critical)
        all_warnings.extend(warnings)
        print(f"\n" + "="*100 + "\n")

    # FINAL SUMMARY
    print(f"{'='*100}")
    print(f"🚨🚨🚨 FINAL FRAMEWORK VALIDATION")
    print(f"{'='*100}")

    unique_critical = list(set(all_critical))
    unique_warnings = list(set(all_warnings))

    print(f"Total unique critical issues: {len(unique_critical)}")
    print(f"Total unique warnings: {len(unique_warnings)}")

    if unique_critical:
        print(f"\n🚨🚨🚨 FRAMEWORK IS INVALID - CRITICAL BUGS DETECTED:")
        for i, issue in enumerate(unique_critical, 1):
            print(f"  {i}. {issue}")
        print(f"\n❌ ALL BACKTEST RESULTS ARE MEANINGLESS!")
    else:
        print(f"\n✅ FRAMEWORK VALIDATION PASSED!")
        print(f"   No critical bugs detected in immediate diagnostics")
        print(f"   Framework appears sound for basic data integrity")

if __name__ == "__main__":
    main()
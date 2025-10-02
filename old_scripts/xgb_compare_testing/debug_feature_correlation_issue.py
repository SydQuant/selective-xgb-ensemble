#!/usr/bin/env python3
"""
Detailed investigation of NumPy correlation warnings in feature engineering.
Identifies specific problematic features and why they pass clean_df validation.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection

def investigate_feature_pipeline(symbol="BL#C"):
    """Detailed investigation of feature engineering pipeline"""
    print("="*80)
    print(f"FEATURE CORRELATION WARNING INVESTIGATION: {symbol}")
    print("="*80)

    # Step 1: Load raw data
    print("STEP 1: Loading raw data...")
    df_raw = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
    target_col = f"{symbol}_target_return"

    print(f"Raw data shape: {df_raw.shape}")
    print(f"Target column: {target_col}")

    # Extract features
    feature_cols = [c for c in df_raw.columns if c != target_col]
    X_raw = df_raw[feature_cols]
    y = df_raw[target_col]

    print(f"Raw features shape: {X_raw.shape}")
    print(f"Target shape: {y.shape}")

    # Step 2: Analyze features before feature selection
    print("\nSTEP 2: Analyzing raw features...")

    # Check for problematic features in raw data
    zero_var_features = []
    near_zero_var_features = []
    constant_features = []

    for col in X_raw.columns:
        values = X_raw[col].dropna()
        if len(values) == 0:
            continue

        var = values.var()
        nunique = values.nunique()

        if var == 0:
            zero_var_features.append(col)
        elif var < 1e-12:
            near_zero_var_features.append(col)
        elif nunique == 1:
            constant_features.append(col)

    print(f"Raw data analysis:")
    print(f"  Zero variance features: {len(zero_var_features)}")
    print(f"  Near-zero variance features: {len(near_zero_var_features)}")
    print(f"  Constant features: {len(constant_features)}")

    if zero_var_features:
        print(f"  Zero variance examples: {zero_var_features[:5]}")
    if near_zero_var_features:
        print(f"  Near-zero variance examples: {near_zero_var_features[:5]}")
    if constant_features:
        print(f"  Constant feature examples: {constant_features[:5]}")

    # Step 3: Apply feature selection with detailed monitoring
    print("\nSTEP 3: Applying feature selection with monitoring...")

    # Enable warnings capture
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)

        # Apply feature selection (this is where warnings typically occur)
        X_selected = apply_feature_selection(X_raw, y, method='block_wise', max_total_features=100)

        print(f"Feature selection warnings: {len(w)}")

        # Analyze warnings
        warning_details = defaultdict(int)
        for warning in w:
            warning_str = str(warning.message)
            warning_details[warning_str] += 1

        for msg, count in warning_details.items():
            print(f"  '{msg}': {count} times")

    print(f"Selected features shape: {X_selected.shape}")

    # Step 4: Analyze selected features for issues
    print("\nSTEP 4: Analyzing selected features...")

    selected_issues = []
    for col in X_selected.columns:
        values = X_selected[col].dropna()
        if len(values) == 0:
            continue

        var = values.var()
        nunique = values.nunique()

        if var == 0 or var < 1e-12 or nunique <= 1:
            selected_issues.append({
                'feature': col,
                'variance': var,
                'unique_values': nunique,
                'min_val': values.min(),
                'max_val': values.max(),
                'sample_values': values.head(10).tolist()
            })

    print(f"Problematic features in selected set: {len(selected_issues)}")
    for issue in selected_issues[:5]:  # Show first 5
        print(f"  {issue['feature']}: var={issue['variance']:.2e}, unique={issue['unique_values']}")
        print(f"    Range: {issue['min_val']:.6f} to {issue['max_val']:.6f}")
        print(f"    Sample: {issue['sample_values'][:5]}")

    return X_raw, X_selected, selected_issues

def investigate_time_window_variance(symbol="BL#C"):
    """Investigate how feature variance changes across time windows"""
    print("\n" + "="*80)
    print(f"TIME WINDOW VARIANCE INVESTIGATION: {symbol}")
    print("="*80)

    # Load data
    df = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
    target_col = f"{symbol}_target_return"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    # Simulate fold splits like in cross-validation
    print("Analyzing variance across time windows (simulating CV folds)...")

    n_samples = len(X)
    n_folds = 5  # Simulate 5 folds for testing

    problematic_features = defaultdict(list)

    for fold in range(n_folds):
        start_idx = int(fold * n_samples / n_folds)
        end_idx = int((fold + 1) * n_samples / n_folds)

        X_fold = X.iloc[start_idx:end_idx]

        print(f"\nFold {fold+1}: samples {start_idx}-{end_idx} ({len(X_fold)} rows)")

        # Check variance in this time window
        fold_issues = []
        for col in X_fold.columns:
            values = X_fold[col].dropna()
            if len(values) == 0:
                continue

            var = values.var()
            nunique = values.nunique()

            if var == 0 or nunique <= 1:
                fold_issues.append({
                    'feature': col,
                    'variance': var,
                    'unique_values': nunique,
                    'fold': fold + 1
                })
                problematic_features[col].append(fold + 1)

        print(f"  Problematic features in this window: {len(fold_issues)}")

        # Show worst offenders
        for issue in fold_issues[:3]:
            print(f"    {issue['feature']}: var={issue['variance']:.2e}, unique={issue['unique_values']}")

    # Summary of features with issues across multiple folds
    print(f"\nFEATURES WITH ISSUES ACROSS MULTIPLE TIME WINDOWS:")
    recurring_issues = {k: v for k, v in problematic_features.items() if len(v) > 1}

    for feature, folds in list(recurring_issues.items())[:10]:  # Top 10
        print(f"  {feature}: issues in folds {folds}")

def investigate_clean_df_process(symbol="BL#C"):
    """Investigate the clean_df process and why problematic features pass"""
    print("\n" + "="*80)
    print(f"CLEAN_DF PROCESS INVESTIGATION: {symbol}")
    print("="*80)

    # We need to examine the data_utils_simple.py to understand clean_df
    # Let's load and check what cleaning is applied

    print("Loading data and examining cleaning process...")

    try:
        # Import the cleaning function if it exists
        from data.data_utils_simple import clean_df
        print("Found clean_df function, analyzing...")

        # Load raw data before cleaning
        df_raw = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")

        # Check what clean_df does (we need to see its source)
        print(f"Data shape after prepare_real_data_simple: {df_raw.shape}")

        # Look at the actual values of features that cause problems
        target_col = f"{symbol}_target_return"
        feature_cols = [c for c in df_raw.columns if c != target_col]

        # Check specific features that might be problematic
        suspicious_features = []
        for col in feature_cols[:20]:  # Check first 20 features
            values = df_raw[col].dropna()
            var = values.var()
            nunique = values.nunique()

            # These would be flags for clean_df to remove
            if var < 1e-15 or nunique <= 1:
                suspicious_features.append({
                    'feature': col,
                    'variance': var,
                    'unique_values': nunique,
                    'null_count': df_raw[col].isnull().sum(),
                    'sample_values': values.head(10).tolist()
                })

        print(f"Features that should be flagged by clean_df: {len(suspicious_features)}")
        for feat in suspicious_features:
            print(f"  {feat['feature']}: var={feat['variance']:.2e}, unique={feat['unique_values']}, nulls={feat['null_count']}")
            print(f"    Sample values: {feat['sample_values'][:5]}")

    except ImportError:
        print("clean_df function not found, checking data preparation process...")

        # Manually check what kind of cleaning might be applied
        df_raw = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
        print(f"Final data shape: {df_raw.shape}")

        # Check if any obvious constant features remain
        target_col = f"{symbol}_target_return"
        feature_cols = [c for c in df_raw.columns if c != target_col]
        X = df_raw[feature_cols]

        constant_after_clean = []
        for col in X.columns:
            if X[col].nunique() <= 1:
                constant_after_clean.append(col)

        print(f"Constant features after cleaning: {len(constant_after_clean)}")
        if constant_after_clean:
            print(f"Examples: {constant_after_clean[:5]}")

def run_correlation_test_with_specific_features(symbol="BL#C"):
    """Test correlation calculation with specific features to trigger warnings"""
    print("\n" + "="*80)
    print(f"CORRELATION TEST WITH SPECIFIC FEATURES: {symbol}")
    print("="*80)

    # Load data
    df = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
    target_col = f"{symbol}_target_return"
    X = df[[c for c in df.columns if c != target_col]]

    # Create a subset with known problematic features
    print("Testing correlation calculation on subsets...")

    # Test 1: Small subset
    X_small = X.iloc[:100, :20]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)

        print("Test 1: Correlation on small subset...")
        try:
            corr_matrix = X_small.corr()
            print(f"  Warnings: {len(w)}")
            print(f"  NaN values in correlation: {corr_matrix.isnull().sum().sum()}")

            # Find which features caused NaN correlations
            nan_features = []
            for col in corr_matrix.columns:
                if corr_matrix[col].isnull().any():
                    nan_features.append(col)

            print(f"  Features with NaN correlations: {len(nan_features)}")
            for feat in nan_features[:5]:
                var = X_small[feat].var()
                nunique = X_small[feat].nunique()
                print(f"    {feat}: var={var:.2e}, unique={nunique}")

        except Exception as e:
            print(f"  Error in correlation: {e}")

        # Show warning details
        for warning in w:
            print(f"  Warning: {warning.message}")

if __name__ == "__main__":
    # Test with BL#C (where we saw warnings)
    symbols_to_test = ["BL#C", "@BO#C", "QCL#C"]

    for symbol in symbols_to_test:
        print(f"\n{'='*100}")
        print(f"INVESTIGATING {symbol}")
        print(f"{'='*100}")

        try:
            X_raw, X_selected, issues = investigate_feature_pipeline(symbol)
            investigate_time_window_variance(symbol)
            investigate_clean_df_process(symbol)
            run_correlation_test_with_specific_features(symbol)

            print(f"\n{symbol} INVESTIGATION COMPLETE")
            print(f"Issues found: {len(issues)} problematic features in selected set")

        except Exception as e:
            print(f"Error investigating {symbol}: {e}")
            import traceback
            traceback.print_exc()
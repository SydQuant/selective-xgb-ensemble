#!/usr/bin/env python3
"""
Focused investigation of correlation warnings and underperforming symbols.
Targets specific issues quickly.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple

def quick_feature_analysis(symbol):
    """Quick analysis of feature issues"""
    print(f"\n{'='*60}")
    print(f"QUICK ANALYSIS: {symbol}")
    print(f"{'='*60}")

    try:
        # Load data (smaller date range for speed)
        df = prepare_real_data_simple(symbol, start_date="2020-01-01", end_date="2023-01-01")
        target_col = f"{symbol}_target_return"
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]

        print(f"Data loaded: {X.shape}")

        # Quick feature analysis
        zero_var_features = []
        near_zero_var = []
        very_low_var = []

        for col in X.columns:
            var = X[col].var()
            nunique = X[col].nunique()

            if var == 0 or nunique == 1:
                zero_var_features.append((col, var, nunique))
            elif var < 1e-12:
                near_zero_var.append((col, var, nunique))
            elif var < 1e-8:
                very_low_var.append((col, var, nunique))

        print(f"Zero variance features: {len(zero_var_features)}")
        print(f"Near-zero variance features: {len(near_zero_var)}")
        print(f"Very low variance features: {len(very_low_var)}")

        # Show examples
        for name, features in [("Zero variance", zero_var_features[:3]),
                              ("Near-zero variance", near_zero_var[:3]),
                              ("Very low variance", very_low_var[:3])]:
            if features:
                print(f"\n{name} examples:")
                for col, var, nunique in features:
                    sample_vals = X[col].dropna().head(5).tolist()
                    print(f"  {col}: var={var:.2e}, unique={nunique}, sample={sample_vals}")

        # Test correlation warnings
        print(f"\nTesting correlation calculation...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # Test on subset
            X_test = X.sample(min(500, len(X)), random_state=42)
            corr_matrix = X_test.corr()

            print(f"Correlation warnings: {len(w)}")
            if w:
                for warning in w[:3]:
                    print(f"  {warning.message}")

            # Count NaN correlations
            nan_count = corr_matrix.isnull().sum().sum()
            print(f"NaN correlations: {nan_count}")

            if nan_count > 0:
                # Find features causing NaN
                nan_features = []
                for col in corr_matrix.columns:
                    if corr_matrix[col].isnull().any():
                        nan_features.append(col)

                print(f"Features causing NaN correlations: {len(nan_features)}")
                for feat in nan_features[:3]:
                    var = X_test[feat].var()
                    nunique = X_test[feat].nunique()
                    print(f"  {feat}: var={var:.2e}, unique={nunique}")

        # Analyze time window variance (simulate CV splits)
        print(f"\nTime window analysis (simulating CV)...")
        n_folds = 3
        time_issues = defaultdict(int)

        for fold in range(n_folds):
            start_idx = int(fold * len(X) / n_folds)
            end_idx = int((fold + 1) * len(X) / n_folds)
            X_fold = X.iloc[start_idx:end_idx]

            fold_zero_var = 0
            for col in X_fold.columns:
                if X_fold[col].var() == 0 or X_fold[col].nunique() <= 1:
                    time_issues[col] += 1
                    fold_zero_var += 1

            print(f"  Fold {fold+1}: {fold_zero_var} zero-variance features")

        recurring_issues = {k: v for k, v in time_issues.items() if v > 1}
        print(f"Features with recurring issues: {len(recurring_issues)}")
        for feat, count in list(recurring_issues.items())[:5]:
            print(f"  {feat}: issues in {count}/{n_folds} folds")

        return {
            'symbol': symbol,
            'zero_var_count': len(zero_var_features),
            'near_zero_var_count': len(near_zero_var),
            'correlation_warnings': len(w) if 'w' in locals() else 0,
            'nan_correlations': nan_count if 'nan_count' in locals() else 0,
            'recurring_issues': len(recurring_issues)
        }

    except Exception as e:
        print(f"ERROR analyzing {symbol}: {e}")
        return {'symbol': symbol, 'error': str(e)}

def analyze_feature_patterns():
    """Analyze patterns in problematic features"""
    print(f"\n{'='*60}")
    print("FEATURE PATTERN ANALYSIS")
    print(f"{'='*60}")

    # Load one symbol to analyze feature patterns
    symbol = "BL#C"
    df = prepare_real_data_simple(symbol, start_date="2020-01-01", end_date="2023-01-01")
    target_col = f"{symbol}_target_return"
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"Analyzing feature patterns from {len(feature_cols)} features...")

    # Categorize features by type
    feature_types = {
        'momentum': [],
        'velocity': [],
        'rsi': [],
        'atr': [],
        'breakout': [],
        'corr': [],
        'other': []
    }

    for col in feature_cols:
        col_lower = col.lower()
        categorized = False
        for ftype in feature_types.keys():
            if ftype in col_lower:
                feature_types[ftype].append(col)
                categorized = True
                break
        if not categorized:
            feature_types['other'].append(col)

    # Analyze variance by feature type
    print("\nVariance analysis by feature type:")
    for ftype, features in feature_types.items():
        if features:
            variances = [df[col].var() for col in features]
            zero_var_count = sum(1 for v in variances if v == 0)
            near_zero_count = sum(1 for v in variances if 0 < v < 1e-12)

            print(f"  {ftype}: {len(features)} features, {zero_var_count} zero-var, {near_zero_count} near-zero-var")

            if zero_var_count > 0:
                zero_var_features = [f for f in features if df[f].var() == 0]
                print(f"    Zero-var examples: {zero_var_features[:3]}")

    return feature_types

def investigate_cleaning_thresholds():
    """Investigate why problematic features pass cleaning"""
    print(f"\n{'='*60}")
    print("CLEANING THRESHOLD INVESTIGATION")
    print(f"{'='*60}")

    symbol = "BL#C"

    # Load larger dataset to see full picture
    df = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
    target_col = f"{symbol}_target_return"
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"Full dataset: {df.shape}")

    # Check cleaning thresholds from data_utils_simple.py
    print("\nChecking features against cleaning thresholds:")

    features_analysis = []
    for col in feature_cols[:20]:  # Check first 20 for speed
        series = df[col]

        # Check cleaning criteria
        n_total = len(series)
        n_nan = series.isna().sum()
        nan_pct = n_nan / n_total

        finite_vals = series.dropna()
        std_val = finite_vals.std() if len(finite_vals) > 0 else 0

        # Would this feature be dropped by clean_data_simple?
        would_drop_nan = nan_pct > 0.9
        would_drop_constant = std_val < 1e-12

        features_analysis.append({
            'feature': col,
            'nan_pct': nan_pct,
            'std': std_val,
            'would_drop_nan': would_drop_nan,
            'would_drop_constant': would_drop_constant,
            'variance': finite_vals.var() if len(finite_vals) > 0 else 0,
            'unique_values': finite_vals.nunique()
        })

    # Show features that should be dropped but aren't
    print("\nFeatures that pass cleaning but have issues:")
    problematic = [f for f in features_analysis
                  if not f['would_drop_nan'] and not f['would_drop_constant']
                  and (f['variance'] < 1e-10 or f['unique_values'] <= 2)]

    for feat in problematic:
        print(f"  {feat['feature']}: std={feat['std']:.2e}, var={feat['variance']:.2e}, unique={feat['unique_values']}")
        print(f"    nan_pct={feat['nan_pct']:.3f}, would_drop={feat['would_drop_constant'] or feat['would_drop_nan']}")

    return features_analysis

if __name__ == "__main__":
    # Quick analysis of all problematic symbols
    symbols = ["BL#C", "@BO#C", "QCL#C", "QRB#C"]

    results = []
    for symbol in symbols:
        result = quick_feature_analysis(symbol)
        results.append(result)

    # Pattern analysis
    feature_types = analyze_feature_patterns()

    # Cleaning investigation
    cleaning_analysis = investigate_cleaning_thresholds()

    # Summary
    print(f"\n{'='*80}")
    print("INVESTIGATION SUMMARY")
    print(f"{'='*80}")

    for result in results:
        if 'error' not in result:
            print(f"{result['symbol']}: {result['zero_var_count']} zero-var, "
                  f"{result['correlation_warnings']} warnings, "
                  f"{result['recurring_issues']} recurring issues")
        else:
            print(f"{result['symbol']}: ERROR - {result['error']}")

    print(f"\nKey findings will be in the detailed output above.")
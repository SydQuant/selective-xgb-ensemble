#!/usr/bin/env python3
"""Simple debug script to investigate NumPy correlation warnings"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple

def simple_correlation_test():
    """Simple test to reproduce and understand the correlation warnings"""
    print("="*60)
    print("SIMPLE NUMPY CORRELATION WARNING TEST")
    print("="*60)

    # Load BL data (where warnings occurred)
    print("Loading BL#C data (first 500 rows, 50 features)...")
    df = prepare_real_data_simple("BL#C", start_date="2015-01-01", end_date="2025-08-01")
    target_col = "BL#C_target_return"
    X = df[[c for c in df.columns if c != target_col]]

    # Take small subset for testing
    X_small = X.iloc[:500, :50]
    print(f"Test data shape: {X_small.shape}")

    # Check for problematic features
    print("\nAnalyzing feature characteristics...")
    zero_var = (X_small.var() == 0).sum()
    near_zero_var = (X_small.var() < 1e-12).sum()

    print(f"Zero variance features: {zero_var}")
    print(f"Near-zero variance features: {near_zero_var}")

    # Test correlation with warnings captured
    print("\nTesting correlation calculation...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)

        try:
            corr_matrix = X_small.corr()

            print(f"Correlation matrix computed: {corr_matrix.shape}")
            print(f"Warnings generated: {len(w)}")

            if w:
                print("Warning details:")
                for i, warning in enumerate(w[:3]):  # Show first 3
                    print(f"  {i+1}. {warning.message}")
                    print(f"     Category: {warning.category.__name__}")

            # Check for NaN/inf in results
            nan_count = corr_matrix.isna().sum().sum()
            inf_count = np.isinf(corr_matrix.values).sum()

            print(f"NaN values in correlation matrix: {nan_count}")
            print(f"Inf values in correlation matrix: {inf_count}")

            # Show some statistics
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            valid_corr = corr_values[~np.isnan(corr_values) & ~np.isinf(corr_values)]

            print(f"Valid correlation values: {len(valid_corr)}/{len(corr_values)}")
            if len(valid_corr) > 0:
                print(f"Correlation range: {valid_corr.min():.6f} to {valid_corr.max():.6f}")
                print(f"Mean absolute correlation: {np.abs(valid_corr).mean():.6f}")

        except Exception as e:
            print(f"Error during correlation: {e}")

def test_manual_correlation():
    """Test manual correlation calculation to understand the warning source"""
    print("\n" + "="*60)
    print("MANUAL CORRELATION CALCULATION TEST")
    print("="*60)

    # Create synthetic data that will trigger warnings
    np.random.seed(42)
    n_samples = 100

    # Normal feature
    feature1 = np.random.randn(n_samples)

    # Constant feature (zero variance)
    feature2 = np.ones(n_samples)

    # Near-constant feature
    feature3 = np.ones(n_samples) + np.random.randn(n_samples) * 1e-15

    # Create DataFrame
    test_df = pd.DataFrame({
        'normal': feature1,
        'constant': feature2,
        'near_constant': feature3
    })

    print(f"Test data variances:")
    for col in test_df.columns:
        print(f"  {col}: {test_df[col].var():.2e}")

    # Test correlation with warnings
    print("\nTesting correlation on synthetic data...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)

        # Manual correlation calculation
        print("Manual calculation:")
        for i, col1 in enumerate(test_df.columns):
            for j, col2 in enumerate(test_df.columns):
                if i < j:
                    try:
                        corr = test_df[col1].corr(test_df[col2])
                        print(f"  {col1} vs {col2}: {corr}")
                    except Exception as e:
                        print(f"  {col1} vs {col2}: ERROR - {e}")

        # Pandas corr()
        print("\nPandas corr():")
        corr_matrix = test_df.corr()
        print(corr_matrix)

        print(f"\nWarnings generated: {len(w)}")
        for warning in w:
            print(f"  {warning.message}")

def test_impact_on_feature_selection():
    """Test if warnings impact feature selection logic"""
    print("\n" + "="*60)
    print("TESTING IMPACT ON FEATURE SELECTION")
    print("="*60)

    # Load small BL subset
    df = prepare_real_data_simple("BL#C", start_date="2015-01-01", end_date="2016-01-01")  # 1 year only
    target_col = "BL#C_target_return"
    X, y = df[[c for c in df.columns if c != target_col]], df[target_col]

    # Take small subset
    X_small = X.iloc[:200, :20]  # Very small for testing
    y_small = y.iloc[:200]

    print(f"Small test data: {X_small.shape}")

    # Test correlation-based feature selection manually
    print("\nTesting correlation-based feature ranking...")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)

        # Calculate correlations with target
        target_corrs = []
        for col in X_small.columns:
            try:
                corr = X_small[col].corr(y_small)
                target_corrs.append((col, abs(corr) if not pd.isna(corr) else 0))
            except Exception as e:
                target_corrs.append((col, 0))
                print(f"  Error with {col}: {e}")

        # Sort by correlation
        target_corrs.sort(key=lambda x: x[1], reverse=True)

        print(f"Warnings during target correlation: {len(w)}")
        print("Top 5 features by target correlation:")
        for col, corr in target_corrs[:5]:
            print(f"  {col}: {corr:.6f}")

        # Check if any selected features are problematic
        print("\nChecking top features for issues:")
        for col, corr in target_corrs[:5]:
            var = X_small[col].var()
            nunique = X_small[col].nunique()
            print(f"  {col}: var={var:.2e}, unique_vals={nunique}")

if __name__ == "__main__":
    simple_correlation_test()
    test_manual_correlation()
    test_impact_on_feature_selection()
#!/usr/bin/env python3
"""Debug script to investigate NumPy correlation warnings and their impact"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection

def investigate_correlation_warnings():
    """Investigate the NumPy correlation warnings and their impact"""
    print("="*60)
    print("NUMPY CORRELATION WARNINGS INVESTIGATION")
    print("="*60)

    # Enable all warnings to see exactly where they occur
    warnings.filterwarnings('error', category=RuntimeWarning)

    try:
        # Load BL data (where we saw warnings)
        print("Loading BL#C data...")
        df = prepare_real_data_simple("BL#C", start_date="2015-01-01", end_date="2025-08-01")
        target_col = "BL#C_target_return"
        X, y = df[[c for c in df.columns if c != target_col]], df[target_col]

        print(f"Original shape: {X.shape}")

        # Try feature selection with warnings as errors
        print("\nApplying feature selection (with warnings as errors)...")
        X_selected = apply_feature_selection(X, y, method='block_wise', max_total_features=100)
        print(f"Selected shape: {X_selected.shape}")

    except RuntimeWarning as e:
        print(f"\nRuntimeWarning caught: {e}")
        print("This warning occurs during feature selection process!")

        # Now investigate with warnings enabled but not as errors
        warnings.filterwarnings('default', category=RuntimeWarning)

        print("\nRe-running with warnings (not errors) to see impact...")

        # Reload data
        df = prepare_real_data_simple("BL#C", start_date="2015-01-01", end_date="2025-08-01")
        target_col = "BL#C_target_return"
        X, y = df[[c for c in df.columns if c != target_col]], df[target_col]

        print("\nAnalyzing features for zero variance issues...")

        # Check for problematic features
        zero_var_features = []
        near_zero_var_features = []
        constant_features = []

        for col in X.columns:
            var = X[col].var()
            std = X[col].std()
            nunique = X[col].nunique()

            if var == 0:
                zero_var_features.append(col)
            elif var < 1e-15:
                near_zero_var_features.append(col)
            elif nunique <= 1:
                constant_features.append(col)

        print(f"Zero variance features: {len(zero_var_features)}")
        print(f"Near-zero variance features: {len(near_zero_var_features)}")
        print(f"Constant features: {len(constant_features)}")

        if zero_var_features:
            print(f"Examples of zero variance: {zero_var_features[:5]}")

        # Test correlation calculation manually
        print("\nTesting correlation calculation manually...")
        test_subset = X.iloc[:100, :50]  # Small subset for testing

        try:
            corr_matrix = test_subset.corr()
            nan_count = corr_matrix.isna().sum().sum()
            inf_count = np.isinf(corr_matrix.values).sum()
            print(f"Correlation matrix: {corr_matrix.shape}")
            print(f"NaN values in correlation matrix: {nan_count}")
            print(f"Inf values in correlation matrix: {inf_count}")

        except Exception as e:
            print(f"Error in correlation calculation: {e}")

        # Now try feature selection with monitoring
        print("\nApplying feature selection with detailed monitoring...")

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            X_selected = apply_feature_selection(X, y, method='block_wise', max_total_features=100)

            print(f"Number of warnings during feature selection: {len(w)}")
            if w:
                for warning in w[:3]:  # Show first 3 warnings
                    print(f"  Warning: {warning.message}")

        print(f"Feature selection completed: {X.shape} -> {X_selected.shape}")

        # Check if any features in selected set have issues
        print("\nAnalyzing selected features...")
        selected_issues = []
        for col in X_selected.columns:
            var = X_selected[col].var()
            if var == 0 or var < 1e-15:
                selected_issues.append((col, var))

        print(f"Problematic features in selected set: {len(selected_issues)}")
        if selected_issues:
            print(f"Examples: {selected_issues[:3]}")

        return X_selected

def test_impact_on_models():
    """Test if the warnings impact model performance"""
    print("\n" + "="*60)
    print("TESTING IMPACT ON MODEL PERFORMANCE")
    print("="*60)

    try:
        # Simple test with synthetic data that will trigger warnings
        print("Creating synthetic data with zero-variance features...")

        np.random.seed(42)
        n_samples, n_features = 1000, 50

        # Create normal features
        X = np.random.randn(n_samples, n_features)

        # Add some zero-variance features
        X[:, 0] = 1.0  # Constant feature
        X[:, 1] = 1.0  # Another constant feature
        X[:, 2] = np.ones(n_samples) * 0.5  # Another constant

        # Add near-zero variance features
        X[:, 3] = 1.0 + np.random.randn(n_samples) * 1e-10

        y = np.random.randn(n_samples)

        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        y_series = pd.Series(y)

        print(f"Synthetic data shape: {X_df.shape}")
        print(f"Zero variance features: {(X_df.var() == 0).sum()}")
        print(f"Near-zero variance: {(X_df.var() < 1e-8).sum()}")

        # Test correlation with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            corr_matrix = X_df.corr()

            print(f"Warnings in correlation: {len(w)}")
            print(f"NaN in correlation matrix: {corr_matrix.isna().sum().sum()}")
            print(f"Inf in correlation matrix: {np.isinf(corr_matrix.values).sum()}")

        return True

    except Exception as e:
        print(f"Error in synthetic test: {e}")
        return False

if __name__ == "__main__":
    investigate_correlation_warnings()
    test_impact_on_models()
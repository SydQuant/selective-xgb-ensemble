#!/usr/bin/env python3
"""Single symbol investigation - BL#C"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple

def investigate_bl():
    """Investigate BL#C specifically"""
    print("Investigating BL#C correlation warnings...")

    try:
        # Load small subset first
        df = prepare_real_data_simple("BL#C", start_date="2022-01-01", end_date="2023-01-01")
        target_col = "BL#C_target_return"
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]

        print(f"Data shape: {X.shape}")

        # Quick variance check
        zero_var = 0
        near_zero = 0
        for col in X.columns:
            var = X[col].var()
            if var == 0:
                zero_var += 1
            elif var < 1e-12:
                near_zero += 1

        print(f"Zero variance: {zero_var}, Near-zero: {near_zero}")

        # Test correlation with warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # Small test
            X_small = X.iloc[:100, :20]
            corr = X_small.corr()

            print(f"Warnings: {len(w)}")
            for warning in w:
                print(f"  {warning.message}")

            print(f"NaN correlations: {corr.isnull().sum().sum()}")

        # Check specific features
        print("\nChecking first 10 features:")
        for col in X.columns[:10]:
            var = X[col].var()
            nunique = X[col].nunique()
            print(f"  {col}: var={var:.2e}, unique={nunique}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    investigate_bl()
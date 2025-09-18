#!/usr/bin/env python3
"""Debug script to investigate BL data issues"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple
import pandas as pd
import numpy as np

def debug_bl_data():
    """Debug BL#C data loading and processing"""
    print("="*60)
    print("BL#C DATA DEBUGGING")
    print("="*60)

    try:
        # Load BL data
        print("Loading BL#C data...")
        df = prepare_real_data_simple("BL#C", start_date="2015-01-01", end_date="2025-08-01")

        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Check target column
        target_col = "BL#C_target_return"
        if target_col in df.columns:
            target = df[target_col]
            print(f"\nTarget column stats:")
            print(f"  Shape: {target.shape}")
            print(f"  NaN count: {target.isna().sum()}")
            print(f"  Valid values: {(~target.isna()).sum()}")
            print(f"  Min/Max: {target.min():.6f} / {target.max():.6f}")
            print(f"  Mean/Std: {target.mean():.6f} / {target.std():.6f}")

            # Check for zero variance features
            print(f"\nFeature variance analysis:")
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols]

            # Calculate variance for each feature
            variances = X.var()
            zero_var_features = variances[variances == 0].index.tolist()
            very_low_var = variances[(variances > 0) & (variances < 1e-10)].index.tolist()

            print(f"  Total features: {len(feature_cols)}")
            print(f"  Zero variance features: {len(zero_var_features)}")
            print(f"  Very low variance features: {len(very_low_var)}")

            if zero_var_features:
                print(f"  First 5 zero variance: {zero_var_features[:5]}")

            # Check for constant features
            constant_features = []
            for col in feature_cols[:20]:  # Check first 20 features
                unique_vals = X[col].nunique()
                if unique_vals <= 1:
                    constant_features.append(col)

            print(f"  Constant features (first 20 checked): {len(constant_features)}")
            if constant_features:
                print(f"    Examples: {constant_features[:3]}")

        else:
            print(f"ERROR: Target column '{target_col}' not found!")
            print(f"Available columns: {list(df.columns)[:10]}...")

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_bl_data()
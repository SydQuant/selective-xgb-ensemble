#!/usr/bin/env python3
"""
Quick Data Inspection - Immediate Framework Check
================================================

Fast check of the most critical aspects:
1. What features are actually being loaded?
2. What does the target variable look like?
3. Are there any obvious data leakage features?
4. Quick correlation analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from data.symbol_loader import load_symbol_data

def quick_data_inspection(symbol="@ES#C"):
    print(f"\n{'='*60}")
    print(f"QUICK DATA INSPECTION - {symbol}")
    print(f"{'='*60}")

    try:
        # Load data
        print("1. Loading data...")
        symbol_data = load_symbol_data([symbol])

        if symbol_data.empty:
            print(f"❌ No data loaded for {symbol}")
            return

        print(f"   Shape: {symbol_data.shape}")
        print(f"   Date range: {symbol_data.index.min()} to {symbol_data.index.max()}")

        # Check target column
        target_col = f"{symbol}_target_return"
        if target_col in symbol_data.columns:
            target_data = symbol_data[target_col].dropna()
            print(f"\n2. Target variable analysis:")
            print(f"   Target column: {target_col}")
            print(f"   Non-null values: {len(target_data)}")
            print(f"   Mean: {target_data.mean():.6f}")
            print(f"   Std: {target_data.std():.6f}")
            print(f"   Min: {target_data.min():.6f}")
            print(f"   Max: {target_data.max():.6f}")
        else:
            print(f"❌ Target column {target_col} not found!")
            return

        # Check feature columns
        feature_cols = [col for col in symbol_data.columns if not col.endswith('_target_return')]
        print(f"\n3. Feature analysis:")
        print(f"   Total features: {len(feature_cols)}")

        # Look for suspicious feature names
        suspicious = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['return', 'pnl', 'target', 'future']):
                suspicious.append(col)

        if suspicious:
            print(f"   ⚠️  Suspicious feature names: {suspicious[:5]}")
            if len(suspicious) > 5:
                print(f"       ... and {len(suspicious) - 5} more")

        # Sample some feature names
        print(f"   Sample features: {feature_cols[:10]}")

        # Quick correlation check
        print(f"\n4. Quick correlation check (first 20 features):")
        feature_subset = feature_cols[:20]
        target_values = symbol_data[target_col].values

        high_corr_features = []

        for col in feature_subset:
            feature_values = symbol_data[col].values

            # Clean data
            mask = ~(np.isnan(feature_values) | np.isnan(target_values))
            if np.sum(mask) < 10:
                continue

            clean_feature = feature_values[mask]
            clean_target = target_values[mask]

            if np.std(clean_feature) > 1e-10 and np.std(clean_target) > 1e-10:
                corr = np.corrcoef(clean_feature, clean_target)[0, 1]
                if abs(corr) > 0.3:  # High correlation threshold
                    high_corr_features.append((col, corr))

        if high_corr_features:
            print(f"   High correlation features:")
            for col, corr in sorted(high_corr_features, key=lambda x: abs(x[1]), reverse=True):
                print(f"     {col}: {corr:.4f}")
        else:
            print(f"   No high correlations in feature subset")

        # Check for constant features
        print(f"\n5. Feature quality check:")
        constant_features = []
        for col in feature_cols[:50]:  # Check first 50
            values = symbol_data[col].dropna()
            if len(values) > 10 and values.std() < 1e-10:
                constant_features.append(col)

        if constant_features:
            print(f"   ⚠️  Constant features: {constant_features[:5]}")
        else:
            print(f"   ✅ No constant features detected (in sample)")

        print(f"\n✅ Quick inspection completed")

    except Exception as e:
        print(f"❌ Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Quick check on key symbols
    for symbol in ["@ES#C", "QCL#C", "@TY#C"]:
        quick_data_inspection(symbol)
        print()
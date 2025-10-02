#!/usr/bin/env python3
"""Comprehensive investigation of all underperforming symbols"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils_simple import prepare_real_data_simple

def test_symbol_comprehensive(symbol):
    """Comprehensive test of a single symbol"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS: {symbol}")
    print(f"{'='*80}")

    results = {'symbol': symbol}

    try:
        # Test 1: Load full dataset
        print("Test 1: Loading full dataset...")
        df_full = prepare_real_data_simple(symbol, start_date="2015-01-01", end_date="2025-08-01")
        target_col = f"{symbol}_target_return"
        feature_cols = [c for c in df_full.columns if c != target_col]
        X_full = df_full[feature_cols]

        print(f"  Full data shape: {X_full.shape}")
        results['full_shape'] = X_full.shape

        # Test 2: Check variance issues in full dataset
        print("Test 2: Variance analysis on full dataset...")
        zero_var_full = []
        near_zero_var_full = []
        for col in X_full.columns:
            var = X_full[col].var()
            nunique = X_full[col].nunique()
            if var == 0 or nunique <= 1:
                zero_var_full.append(col)
            elif var < 1e-12:
                near_zero_var_full.append(col)

        print(f"  Zero variance features: {len(zero_var_full)}")
        print(f"  Near-zero variance features: {len(near_zero_var_full)}")
        results['zero_var_count'] = len(zero_var_full)
        results['near_zero_var_count'] = len(near_zero_var_full)

        # Test 3: Correlation warnings on full dataset
        print("Test 3: Correlation test on full dataset...")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # Test correlation on random sample
            sample_size = min(1000, len(X_full))
            X_sample = X_full.sample(sample_size, random_state=42)
            corr_matrix = X_sample.corr()

            print(f"  Correlation warnings: {len(w)}")
            results['correlation_warnings'] = len(w)

            if w:
                print(f"  Warning messages:")
                for warning in w[:3]:
                    print(f"    {warning.message}")

            nan_count = corr_matrix.isnull().sum().sum()
            print(f"  NaN correlations: {nan_count}")
            results['nan_correlations'] = nan_count

        # Test 4: Time window analysis (simulate CV)
        print("Test 4: Time window variance analysis...")
        n_windows = 5
        window_issues = defaultdict(int)

        for i in range(n_windows):
            start_idx = int(i * len(X_full) / n_windows)
            end_idx = int((i + 1) * len(X_full) / n_windows)
            X_window = X_full.iloc[start_idx:end_idx]

            window_zero_var = 0
            for col in X_window.columns:
                var = X_window[col].var()
                nunique = X_window[col].nunique()
                if var == 0 or nunique <= 1:
                    window_issues[col] += 1
                    window_zero_var += 1

            print(f"  Window {i+1}: {window_zero_var} zero-variance features")

        recurring = {k: v for k, v in window_issues.items() if v > 1}
        print(f"  Features with recurring issues: {len(recurring)}")
        results['recurring_issues'] = len(recurring)

        # Test 5: Feature type analysis
        print("Test 5: Feature type analysis...")
        feature_types = {
            'momentum': [c for c in feature_cols if 'momentum' in c.lower()],
            'velocity': [c for c in feature_cols if 'velocity' in c.lower()],
            'rsi': [c for c in feature_cols if 'rsi' in c.lower()],
            'atr': [c for c in feature_cols if 'atr' in c.lower()],
            'breakout': [c for c in feature_cols if 'breakout' in c.lower()],
            'corr': [c for c in feature_cols if 'corr' in c.lower()]
        }

        type_issues = {}
        for ftype, features in feature_types.items():
            if features:
                zero_count = sum(1 for f in features if X_full[f].var() == 0)
                type_issues[ftype] = {'total': len(features), 'zero_var': zero_count}
                print(f"  {ftype}: {len(features)} features, {zero_count} zero-variance")

        results['feature_type_issues'] = type_issues

        # Test 6: Extreme values and outliers
        print("Test 6: Extreme value analysis...")
        extreme_features = []
        for col in X_full.columns[:20]:  # Check first 20 for speed
            series = X_full[col].dropna()
            if len(series) > 0:
                q99 = series.quantile(0.99)
                q01 = series.quantile(0.01)
                range_val = q99 - q01
                if range_val < 1e-10:  # Very narrow range
                    extreme_features.append(col)

        print(f"  Features with extremely narrow ranges: {len(extreme_features)}")
        results['narrow_range_features'] = len(extreme_features)

        return results

    except Exception as e:
        print(f"ERROR analyzing {symbol}: {e}")
        results['error'] = str(e)
        return results

def analyze_csv_outputs():
    """Analyze the CSV outputs from the logs- rerun directory"""
    print(f"\n{'='*80}")
    print("CSV OUTPUTS ANALYSIS")
    print(f"{'='*80}")

    csv_files = {
        '@BO#C': '/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs- rerun/20250917_155254_signal_distribution_rerun_BO_multiprocessing.csv',
        'QCL#C': '/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs- rerun/20250917_180458_signal_distribution_rerun_QCL_multiprocessing.csv',
        'QRB#C': '/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs- rerun/20250917_180545_signal_distribution_rerun_QRB_multiprocessing.csv'
    }

    csv_analysis = {}

    for symbol, csv_path in csv_files.items():
        print(f"\nAnalyzing {symbol} CSV...")
        try:
            df = pd.read_csv(csv_path)
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")

            # Analyze signals
            if 'signal_direction' in df.columns:
                signal_dist = df['signal_direction'].value_counts()
                print(f"  Signal distribution: {signal_dist.to_dict()}")

            # Analyze PnL
            if 'pnl' in df.columns:
                total_pnl = df['pnl'].sum()
                mean_pnl = df['pnl'].mean()
                print(f"  Total PnL: {total_pnl:.6f}")
                print(f"  Mean PnL: {mean_pnl:.6f}")

                # Check for extreme PnL values
                extreme_pnl = df['pnl'].abs() > df['pnl'].std() * 3
                print(f"  Extreme PnL points: {extreme_pnl.sum()}")

            # Analyze returns
            if 'target_return' in df.columns:
                ret_stats = df['target_return'].describe()
                print(f"  Return stats: mean={ret_stats['mean']:.6f}, std={ret_stats['std']:.6f}")

            csv_analysis[symbol] = {
                'shape': df.shape,
                'total_pnl': df['pnl'].sum() if 'pnl' in df.columns else None,
                'signal_balance': df['signal_direction'].value_counts().to_dict() if 'signal_direction' in df.columns else None
            }

        except Exception as e:
            print(f"  Error reading {symbol} CSV: {e}")
            csv_analysis[symbol] = {'error': str(e)}

    return csv_analysis

def main():
    """Main investigation function"""
    print("COMPREHENSIVE UNDERPERFORMING SYMBOLS INVESTIGATION")
    print("="*80)

    # Test all problematic symbols
    symbols = ["BL#C", "@BO#C", "QCL#C", "QRB#C"]
    all_results = []

    for symbol in symbols:
        try:
            result = test_symbol_comprehensive(symbol)
            all_results.append(result)
        except Exception as e:
            print(f"Failed to analyze {symbol}: {e}")
            all_results.append({'symbol': symbol, 'error': str(e)})

    # Analyze CSV outputs
    csv_analysis = analyze_csv_outputs()

    # Summary analysis
    print(f"\n{'='*80}")
    print("INVESTIGATION SUMMARY")
    print(f"{'='*80}")

    print("\nSymbol Analysis Summary:")
    for result in all_results:
        symbol = result['symbol']
        if 'error' not in result:
            print(f"\n{symbol}:")
            print(f"  Data shape: {result.get('full_shape', 'N/A')}")
            print(f"  Zero variance features: {result.get('zero_var_count', 0)}")
            print(f"  Correlation warnings: {result.get('correlation_warnings', 0)}")
            print(f"  Recurring issues: {result.get('recurring_issues', 0)}")
            print(f"  Narrow range features: {result.get('narrow_range_features', 0)}")

            if 'feature_type_issues' in result:
                print(f"  Feature type issues:")
                for ftype, issue in result['feature_type_issues'].items():
                    print(f"    {ftype}: {issue['zero_var']}/{issue['total']} zero-variance")
        else:
            print(f"\n{symbol}: ERROR - {result['error']}")

    print(f"\nCSV Analysis Summary:")
    for symbol, analysis in csv_analysis.items():
        if 'error' not in analysis:
            print(f"\n{symbol}:")
            print(f"  Total PnL: {analysis.get('total_pnl', 'N/A')}")
            print(f"  Signal balance: {analysis.get('signal_balance', 'N/A')}")
        else:
            print(f"\n{symbol}: CSV Error - {analysis['error']}")

    return all_results, csv_analysis

if __name__ == "__main__":
    results, csv_results = main()
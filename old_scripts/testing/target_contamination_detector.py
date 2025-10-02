#!/usr/bin/env python3
"""
Target Contamination Detector - The Ultimate Framework Killer
============================================================

This script checks for the MOST CRITICAL bug: target variable contamination in features.

This includes:
1. TARGET VARIABLE DIRECTLY IN FEATURES - y accidentally included in X
2. LAGGED TARGET CONTAMINATION - y[t-1] included as feature for predicting y[t]
3. TRANSFORMED TARGET CONTAMINATION - log(y), y^2, etc. in features
4. DERIVED TARGET CONTAMINATION - Features calculated from target returns
5. FUTURE TARGET CONTAMINATION - Features using y[t+1] to predict y[t]
6. CROSS-SYMBOL CONTAMINATION - Using other symbol's contemporaneous returns
7. PERFECT PREDICTOR DETECTION - Any feature that predicts too well

If ANY of these exist, ALL results are completely meaningless.
This would explain artificially high Sharpe ratios and hit rates.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
from scipy.stats import pearsonr

# Import framework components
from data.symbol_loader import load_symbol_data
from model.feature_selection import select_features

def detect_direct_target_contamination(X, y, feature_names=None, verbose=True):
    """
    🚨 ULTIMATE CRITICAL CHECK: Direct target variable contamination

    Check if target variable is directly included in feature matrix.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🚨 ULTIMATE CRITICAL: DIRECT TARGET CONTAMINATION")
        print(f"{'='*70}")

    issues = []

    if len(X) != len(y):
        issues.append("CRITICAL: X and y have different lengths")
        return issues

    if verbose:
        print(f"Checking {X.shape[1]} features against {len(y)} targets...")

    # Check each feature column against target
    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]

        # Remove NaN/inf values for comparison
        valid_mask = ~(np.isnan(feature_values) | np.isnan(y) |
                      np.isinf(feature_values) | np.isinf(y))

        if np.sum(valid_mask) < 10:
            continue

        clean_feature = feature_values[valid_mask]
        clean_target = y[valid_mask]

        # Test 1: Exact match (target directly included)
        diff = np.abs(clean_feature - clean_target)
        identical_count = np.sum(diff < 1e-12)

        if identical_count > len(clean_feature) * 0.9:  # 90% identical
            feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
            issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' is identical to target variable!"
            issues.append(issue)
            if verbose:
                print(f"🚨🚨🚨 {issue}")
                print(f"    {identical_count}/{len(clean_feature)} values identical")

        # Test 2: Perfect linear relationship (scaled/shifted target)
        if np.std(clean_feature) > 1e-10 and np.std(clean_target) > 1e-10:
            correlation, p_value = pearsonr(clean_feature, clean_target)

            if abs(correlation) > 0.999 and p_value < 1e-10:
                feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' perfectly correlated with target (r={correlation:.6f})"
                issues.append(issue)
                if verbose:
                    print(f"🚨🚨🚨 {issue}")

        # Test 3: Transformed target (y^2, log(y), etc.)
        # Check common transformations
        transformations = {
            'squared': clean_target ** 2,
            'sqrt': np.sqrt(np.abs(clean_target)),
            'log': np.log(np.abs(clean_target) + 1e-8),
            'exp': np.exp(np.clip(clean_target, -10, 10)),
            'inverse': 1 / (clean_target + 1e-8),
            'abs': np.abs(clean_target)
        }

        for transform_name, transformed in transformations.items():
            try:
                if np.all(np.isfinite(transformed)):
                    transform_diff = np.abs(clean_feature - transformed)
                    transform_matches = np.sum(transform_diff < 1e-10)

                    if transform_matches > len(clean_feature) * 0.9:
                        feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                        issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' is {transform_name} of target!"
                        issues.append(issue)
                        if verbose:
                            print(f"🚨🚨🚨 {issue}")
                            break
            except:
                continue

    if verbose:
        if not issues:
            print("✅ No direct target contamination detected")
        else:
            print(f"🚨🚨🚨 FOUND {len(issues)} ULTIMATE CRITICAL TARGET CONTAMINATION ISSUES!")

    return issues

def detect_lagged_target_contamination(X, y, feature_names=None, max_lags=5, verbose=True):
    """
    🚨 CRITICAL: Lagged target variable contamination

    Check if lagged versions of target are included in features.
    y[t-1], y[t-2], etc. should not be used to predict y[t].
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🚨 CRITICAL: LAGGED TARGET CONTAMINATION")
        print(f"{'='*70}")

    issues = []

    if len(y) < max_lags + 1:
        issues.append("WARNING: Not enough data points to check lagged contamination")
        return issues

    if verbose:
        print(f"Checking for lagged target contamination up to {max_lags} periods...")

    # Create lagged target series
    for lag in range(1, max_lags + 1):
        if lag >= len(y):
            break

        lagged_target = np.roll(y, lag)[lag:]  # y[t-lag] to predict y[t]
        current_target = y[lag:]               # y[t]
        current_features = X[lag:, :]          # X[t]

        # Check each feature against this lagged target
        for feature_idx in range(current_features.shape[1]):
            feature_values = current_features[:, feature_idx]

            # Remove NaN/inf
            valid_mask = ~(np.isnan(feature_values) | np.isnan(lagged_target) |
                          np.isinf(feature_values) | np.isinf(lagged_target))

            if np.sum(valid_mask) < 10:
                continue

            clean_feature = feature_values[valid_mask]
            clean_lagged_target = lagged_target[valid_mask]

            # Check for exact match
            diff = np.abs(clean_feature - clean_lagged_target)
            identical_count = np.sum(diff < 1e-12)

            if identical_count > len(clean_feature) * 0.9:
                feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' is target lagged by {lag} periods!"
                issues.append(issue)
                if verbose:
                    print(f"🚨🚨🚨 {issue}")

            # Check for perfect correlation
            if np.std(clean_feature) > 1e-10 and np.std(clean_lagged_target) > 1e-10:
                correlation, p_value = pearsonr(clean_feature, clean_lagged_target)

                if abs(correlation) > 0.999 and p_value < 1e-10:
                    feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                    issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' perfectly correlated with target lag {lag} (r={correlation:.6f})"
                    issues.append(issue)
                    if verbose:
                        print(f"🚨🚨🚨 {issue}")

    if verbose:
        if not issues:
            print("✅ No lagged target contamination detected")
        else:
            print(f"🚨🚨🚨 FOUND {len(issues)} LAGGED TARGET CONTAMINATION ISSUES!")

    return issues

def detect_future_target_contamination(X, y, feature_names=None, max_leads=3, verbose=True):
    """
    🚨 CRITICAL: Future target contamination

    Check if future target values are accidentally used in features.
    This is the most insidious form of data leakage.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🚨 CRITICAL: FUTURE TARGET CONTAMINATION")
        print(f"{'='*70}")

    issues = []

    if len(y) < max_leads + 1:
        issues.append("WARNING: Not enough data points to check future contamination")
        return issues

    if verbose:
        print(f"Checking for future target contamination up to {max_leads} periods ahead...")

    # Create future target series
    for lead in range(1, max_leads + 1):
        if lead >= len(y):
            break

        future_target = np.roll(y, -lead)[:-lead]  # y[t+lead]
        current_target = y[:-lead]                 # y[t]
        current_features = X[:-lead, :]            # X[t] (should predict y[t])

        # Check each feature against future target
        for feature_idx in range(current_features.shape[1]):
            feature_values = current_features[:, feature_idx]

            # Remove NaN/inf
            valid_mask = ~(np.isnan(feature_values) | np.isnan(future_target) |
                          np.isinf(feature_values) | np.isinf(future_target))

            if np.sum(valid_mask) < 10:
                continue

            clean_feature = feature_values[valid_mask]
            clean_future_target = future_target[valid_mask]

            # Check for exact match
            diff = np.abs(clean_feature - clean_future_target)
            identical_count = np.sum(diff < 1e-12)

            if identical_count > len(clean_feature) * 0.9:
                feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' contains future target (lead +{lead})!"
                issues.append(issue)
                if verbose:
                    print(f"🚨🚨🚨 {issue}")

            # Check for suspiciously high correlation
            if np.std(clean_feature) > 1e-10 and np.std(clean_future_target) > 1e-10:
                correlation, p_value = pearsonr(clean_feature, clean_future_target)

                if abs(correlation) > 0.8 and p_value < 1e-5:  # Lower threshold for future
                    feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"
                    issue = f"CRITICAL: Feature '{feature_name}' suspiciously correlated with future target +{lead} (r={correlation:.6f})"
                    issues.append(issue)
                    if verbose:
                        print(f"🚨 {issue}")

    if verbose:
        if not issues:
            print("✅ No future target contamination detected")
        else:
            print(f"🚨🚨🚨 FOUND {len(issues)} FUTURE TARGET CONTAMINATION ISSUES!")

    return issues

def detect_perfect_predictors(X, y, feature_names=None, verbose=True):
    """
    🚨 CRITICAL: Perfect predictor detection

    Any feature that predicts too well is suspicious and needs investigation.
    """
    if verbose:
        print(f"\n{'='*70}")
        print("🚨 CRITICAL: PERFECT PREDICTOR DETECTION")
        print(f"{'='*70}")

    issues = []

    suspicious_features = []

    if verbose:
        print(f"Analyzing {X.shape[1]} features for suspicious predictive power...")

    for feature_idx in range(X.shape[1]):
        feature_values = X[:, feature_idx]

        # Remove NaN/inf
        valid_mask = ~(np.isnan(feature_values) | np.isnan(y) |
                      np.isinf(feature_values) | np.isinf(y))

        if np.sum(valid_mask) < 10:
            continue

        clean_feature = feature_values[valid_mask]
        clean_target = y[valid_mask]

        if np.std(clean_feature) < 1e-10 or np.std(clean_target) < 1e-10:
            continue

        # Calculate correlation
        correlation, p_value = pearsonr(clean_feature, clean_target)

        feature_name = feature_names[feature_idx] if feature_names else f"Feature_{feature_idx}"

        # Flag suspiciously high correlations
        if abs(correlation) > 0.9 and p_value < 1e-10:
            issue = f"ULTIMATE CRITICAL: Feature '{feature_name}' has perfect prediction (r={correlation:.6f}, p={p_value:.2e})"
            issues.append(issue)
            if verbose:
                print(f"🚨🚨🚨 {issue}")

        elif abs(correlation) > 0.7 and p_value < 1e-8:
            issue = f"CRITICAL: Feature '{feature_name}' has suspicious prediction (r={correlation:.6f}, p={p_value:.2e})"
            issues.append(issue)
            if verbose:
                print(f"🚨 {issue}")

        elif abs(correlation) > 0.5 and p_value < 1e-6:
            issue = f"WARNING: Feature '{feature_name}' has high prediction (r={correlation:.6f}, p={p_value:.2e})"
            issues.append(issue)
            suspicious_features.append({
                'feature': feature_name,
                'correlation': correlation,
                'p_value': p_value
            })

    # Print top suspicious features
    if verbose and suspicious_features:
        print(f"\nTop suspicious features:")
        suspicious_features.sort(key=lambda x: abs(x['correlation']), reverse=True)
        for i, feat in enumerate(suspicious_features[:5]):
            print(f"  {i+1}. {feat['feature']}: r={feat['correlation']:.4f}, p={feat['p_value']:.2e}")

    if verbose:
        critical_count = len([i for i in issues if "ULTIMATE CRITICAL" in i or "CRITICAL" in i])
        if critical_count == 0:
            print("✅ No perfect predictors detected")
        else:
            print(f"🚨🚨🚨 FOUND {critical_count} PERFECT PREDICTOR ISSUES!")

    return issues

def run_target_contamination_audit(symbol="@ES#C", verbose=True):
    """
    Run complete target contamination audit on a symbol
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"🚨🚨🚨 TARGET CONTAMINATION AUDIT - {symbol}")
        print(f"{'='*80}")
        print("Checking for the MOST CRITICAL bug: target variable in features")
        print(f"Audit run at: {datetime.now()}")

    all_issues = []

    try:
        # Load data
        if verbose:
            print(f"\n1. Loading data for {symbol}...")

        symbol_data = load_symbol_data([symbol])
        if symbol_data.empty:
            error = f"CRITICAL: No data loaded for {symbol}"
            all_issues.append(error)
            if verbose:
                print(f"🚨 {error}")
            return all_issues

        target_col = f"{symbol}_target_return"
        if target_col not in symbol_data.columns:
            error = f"CRITICAL: Target column {target_col} not found"
            all_issues.append(error)
            if verbose:
                print(f"🚨 {error}")
            return all_issues

        # Clean data
        clean_data = symbol_data.dropna(subset=[target_col])
        if verbose:
            print(f"   Loaded {len(symbol_data)} rows, {len(clean_data)} with valid targets")

        # Prepare features and target
        feature_cols = [col for col in clean_data.columns if not col.endswith('_target_return')]
        X = clean_data[feature_cols].values
        y = clean_data[target_col].values

        if verbose:
            print(f"   Features: {X.shape}")
            print(f"   Target: {y.shape}")

        # Run all contamination checks
        if verbose:
            print(f"\n2. Running contamination checks...")

        # Check 1: Direct target contamination (MOST CRITICAL)
        issues = detect_direct_target_contamination(X, y, feature_cols, verbose)
        all_issues.extend(issues)

        # Check 2: Lagged target contamination
        issues = detect_lagged_target_contamination(X, y, feature_cols, verbose=verbose)
        all_issues.extend(issues)

        # Check 3: Future target contamination
        issues = detect_future_target_contamination(X, y, feature_cols, verbose=verbose)
        all_issues.extend(issues)

        # Check 4: Perfect predictors
        issues = detect_perfect_predictors(X, y, feature_cols, verbose=verbose)
        all_issues.extend(issues)

        # Additional check: Feature names analysis
        if verbose:
            print(f"\n{'='*70}")
            print("🚨 FEATURE NAME ANALYSIS")
            print(f"{'='*70}")

        suspicious_names = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['return', 'pnl', 'profit', 'target', 'future', 'next']):
                suspicious_names.append(col)

        if suspicious_names:
            issue = f"WARNING: Suspicious feature names detected: {suspicious_names[:5]}"
            all_issues.append(issue)
            if verbose:
                print(f"⚠️  {issue}")
                if len(suspicious_names) > 5:
                    print(f"   ... and {len(suspicious_names) - 5} more")

        # Summary
        if verbose:
            print(f"\n{'='*80}")
            print("🚨🚨🚨 TARGET CONTAMINATION AUDIT SUMMARY")
            print(f"{'='*80}")

        ultimate_critical = [i for i in all_issues if "ULTIMATE CRITICAL" in i]
        critical = [i for i in all_issues if "CRITICAL" in i and "ULTIMATE" not in i]
        warnings = [i for i in all_issues if "WARNING" in i]

        if verbose:
            print(f"Issues found: {len(all_issues)}")
            print(f"  ULTIMATE CRITICAL: {len(ultimate_critical)}")
            print(f"  Critical: {len(critical)}")
            print(f"  Warnings: {len(warnings)}")

        if ultimate_critical:
            if verbose:
                print(f"\n🚨🚨🚨 ULTIMATE CRITICAL ISSUES (FRAMEWORK INVALID):")
                for i, issue in enumerate(ultimate_critical, 1):
                    print(f"  {i}. {issue}")

        if critical:
            if verbose:
                print(f"\n🚨 CRITICAL ISSUES:")
                for i, issue in enumerate(critical, 1):
                    print(f"  {i}. {issue}")

        if warnings:
            if verbose:
                print(f"\n⚠️  WARNINGS:")
                for i, issue in enumerate(warnings, 1):
                    print(f"  {i}. {issue}")

        if not (ultimate_critical or critical):
            if verbose:
                print(f"\n✅ No target contamination detected!")
                print("   Framework appears clean of the most critical bug")
        else:
            if verbose:
                print(f"\n❌ TARGET CONTAMINATION DETECTED!")
                if ultimate_critical:
                    print("   ALL RESULTS ARE COMPLETELY MEANINGLESS!")
                else:
                    print("   Results may be compromised!")

        # Save results
        results_dir = Path("testing/contamination_results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"target_contamination_{symbol.replace('@', '').replace('#C', '')}_{timestamp}.txt"

        with open(results_file, 'w') as f:
            f.write(f"Target Contamination Audit Results\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Issues: {len(all_issues)}\n")
            f.write(f"Ultimate Critical: {len(ultimate_critical)}\n")
            f.write(f"Critical: {len(critical)}\n")
            f.write(f"Warnings: {len(warnings)}\n\n")

            if ultimate_critical:
                f.write("ULTIMATE CRITICAL ISSUES:\n")
                for i, issue in enumerate(ultimate_critical, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")

            if critical:
                f.write("CRITICAL ISSUES:\n")
                for i, issue in enumerate(critical, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")

            if warnings:
                f.write("WARNING ISSUES:\n")
                for i, issue in enumerate(warnings, 1):
                    f.write(f"{i}. {issue}\n")

        if verbose:
            print(f"📁 Results saved to: {results_file}")

    except Exception as e:
        error = f"CRITICAL: Error during contamination audit - {str(e)}"
        all_issues.append(error)
        if verbose:
            print(f"🚨 {error}")
            import traceback
            traceback.print_exc()

    return all_issues

if __name__ == "__main__":
    # Test the most critical symbols
    test_symbols = ["@ES#C", "QCL#C", "@TY#C"]

    for symbol in test_symbols:
        print(f"\n{'='*100}")
        issues = run_target_contamination_audit(symbol=symbol, verbose=True)
        print(f"{'='*100}\n")
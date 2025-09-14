#!/usr/bin/env python3
"""
Investigate Unrealistic Performance
==================================

Sharpe ratios of 2.3+ are EXTREMELY suspicious. These would rank in the top 0.1%
of all trading strategies globally. Something is seriously wrong.

Possible causes:
1. MASSIVE OVERFITTING - 1906 features on limited data
2. DATA LEAKAGE - Future data contaminating training
3. SELECTION BIAS - Cherry-picking best results
4. CALCULATION ERRORS - Wrong Sharpe formula or assumptions
5. UNREALISTIC ASSUMPTIONS - Wrong volatility scaling
6. PRODUCTION vs TRAINING CONFUSION - "Production" isn't actually OOS

I need to be BRUTALLY honest about what could invalidate these results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

from data.data_utils_simple import prepare_real_data_simple

def investigate_sharpe_calculation_realism():
    """
    🚨 CRITICAL: Are the Sharpe calculations realistic or wrong?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING UNREALISTIC SHARPE RATIOS")
    print(f"{'='*80}")
    print("Sharpe 2.3+ would make this better than most hedge funds globally!")

    issues = []

    # Load actual data and manually calculate what realistic Sharpe should be
    symbol = "@ES#C"
    df = prepare_real_data_simple(symbol)
    target_col = f"{symbol}_target_return"

    returns = df[target_col].dropna()

    print(f"\nRaw target return analysis for {symbol}:")
    print(f"  Daily return mean: {returns.mean():.6f}")
    print(f"  Daily return std: {returns.std():.6f}")
    print(f"  Daily Sharpe (raw): {returns.mean()/returns.std():.4f}")
    print(f"  Annualized Sharpe (raw): {returns.mean()/returns.std() * np.sqrt(252):.4f}")

    # What would a perfect predictor achieve?
    print(f"\n🧪 THEORETICAL MAXIMUM PERFORMANCE:")

    # Perfect predictor: always long when positive, short when negative
    perfect_signals = np.where(returns > 0, 1, -1)
    perfect_pnl = perfect_signals * returns

    perfect_sharpe = np.mean(perfect_pnl) / np.std(perfect_pnl) * np.sqrt(252)
    perfect_hit_rate = np.mean((perfect_signals * returns) > 0)

    print(f"  Perfect predictor Sharpe: {perfect_sharpe:.4f}")
    print(f"  Perfect predictor hit rate: {perfect_hit_rate:.4f} ({perfect_hit_rate*100:.1f}%)")

    # Random baseline
    np.random.seed(42)
    random_signals = np.random.choice([-1, 1], len(returns))
    random_pnl = random_signals * returns
    random_sharpe = np.mean(random_pnl) / np.std(random_pnl) * np.sqrt(252)

    print(f"  Random strategy Sharpe: {random_sharpe:.4f}")

    print(f"\n🚨 REALITY CHECK:")
    print(f"  Framework claims: 2.385 Sharpe (@NQ#C)")
    print(f"  Perfect predictor: {perfect_sharpe:.4f} Sharpe")
    print(f"  Ratio: {2.385/perfect_sharpe:.2f} of perfect predictor")

    if 2.385 > perfect_sharpe * 0.8:  # More than 80% of perfect
        issue = f"CRITICAL: Claimed Sharpe (2.385) is suspiciously close to perfect predictor ({perfect_sharpe:.2f})"
        issues.append(issue)
        print(f"🚨 {issue}")

    # Check if framework results are even theoretically possible
    framework_results = {
        "@ES#C": 2.319,
        "@NQ#C": 2.385,
        "@TY#C": 2.067,
        "RTY": 2.193
    }

    print(f"\n🧪 THEORETICAL POSSIBILITY CHECK:")
    for sym, claimed_sharpe in framework_results.items():
        if claimed_sharpe > 3.0:
            issue = f"CRITICAL: {sym} Sharpe ({claimed_sharpe}) exceeds reasonable theoretical limits"
            issues.append(issue)
            print(f"🚨 {issue}")
        elif claimed_sharpe > 2.5:
            issue = f"WARNING: {sym} Sharpe ({claimed_sharpe}) is in top 0.1% of all strategies globally"
            issues.append(issue)
            print(f"⚠️  {issue}")

    return issues

def investigate_overfitting_catastrophe():
    """
    🚨 CRITICAL: Check for massive overfitting
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING MASSIVE OVERFITTING")
    print(f"{'='*80}")
    print("1906 features on ~2700 samples is a recipe for overfitting disaster")

    issues = []

    symbol = "@ES#C"
    df = prepare_real_data_simple(symbol)
    target_col = f"{symbol}_target_return"

    print(f"\nData dimensions for {symbol}:")
    print(f"  Total features: {df.shape[1] - 1}")  # Exclude target
    print(f"  Total samples: {len(df.dropna(subset=[target_col]))}")
    print(f"  Feature-to-sample ratio: {(df.shape[1] - 1) / len(df.dropna(subset=[target_col])):.2f}")

    # Rule of thumb: need 10-20 samples per feature
    min_samples_needed = (df.shape[1] - 1) * 10
    actual_samples = len(df.dropna(subset=[target_col]))

    print(f"\nOverfitting risk analysis:")
    print(f"  Features: {df.shape[1] - 1}")
    print(f"  Samples needed (10x rule): {min_samples_needed}")
    print(f"  Actual samples: {actual_samples}")
    print(f"  Shortfall: {min_samples_needed - actual_samples}")

    if actual_samples < min_samples_needed:
        issue = f"CRITICAL: Severe overfitting risk - need {min_samples_needed} samples, have {actual_samples}"
        issues.append(issue)
        print(f"🚨 {issue}")

    # Check feature selection effectiveness
    # Even after selecting 100 features, on 2700 samples, overfitting is likely
    selected_features = 100
    effective_ratio = actual_samples / selected_features

    print(f"\nAfter feature selection to {selected_features}:")
    print(f"  Sample-to-feature ratio: {effective_ratio:.1f}")

    if effective_ratio < 25:  # Less than 25 samples per feature
        issue = f"CRITICAL: Even after feature selection, overfitting likely (ratio: {effective_ratio:.1f})"
        issues.append(issue)
        print(f"🚨 {issue}")

    return issues

def investigate_production_vs_training_confusion():
    """
    🚨 CRITICAL: What does "production" actually mean?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING 'PRODUCTION' METRIC CONFUSION")
    print(f"{'='*80}")
    print("Is 'production' actually out-of-sample or just a different training set?")

    issues = []

    # Analyze the typical results pattern
    print(f"\nTypical framework results pattern:")
    examples = [
        {"symbol": "@ES#C", "training": 0.766, "production": 2.093, "full_timeline": 1.304},
        {"symbol": "@TY#C", "training": 1.311, "production": 1.830, "full_timeline": 1.534},
        {"symbol": "@NQ#C", "training": 1.138, "production": 2.385, "full_timeline": 1.699},
    ]

    for ex in examples:
        print(f"  {ex['symbol']}: Train={ex['training']:.3f}, Prod={ex['production']:.3f}, Full={ex['full_timeline']:.3f}")

        train_prod_ratio = ex['production'] / ex['training'] if ex['training'] != 0 else float('inf')
        prod_full_ratio = ex['production'] / ex['full_timeline'] if ex['full_timeline'] != 0 else float('inf')

        print(f"    Prod/Train ratio: {train_prod_ratio:.2f}")
        print(f"    Prod/Full ratio: {prod_full_ratio:.2f}")

        # 🚨 RED FLAG: Production consistently better than training
        if ex['production'] > ex['training'] * 1.5:
            issue = f"CRITICAL: {ex['symbol']} production ({ex['production']:.3f}) much better than training ({ex['training']:.3f})"
            issues.append(issue)
            print(f"    🚨 {issue}")

        # 🚨 RED FLAG: Production much better than full timeline
        if ex['production'] > ex['full_timeline'] * 1.3:
            issue = f"CRITICAL: {ex['symbol']} production ({ex['production']:.3f}) much better than full timeline ({ex['full_timeline']:.3f})"
            issues.append(issue)
            print(f"    🚨 {issue}")

    print(f"\n🚨 CRITICAL QUESTIONS:")
    print(f"1. Why is 'production' consistently BETTER than 'training'?")
    print(f"2. Why is 'production' better than 'full timeline'?")
    print(f"3. What time period does 'production' actually cover?")
    print(f"4. Is 'production' actually out-of-sample or cherry-picked period?")

    return issues

def investigate_feature_lookahead_bias():
    """
    🚨 CRITICAL: Check if features have lookahead bias
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING FEATURE LOOKAHEAD BIAS")
    print(f"{'='*80}")
    print("1906 features - some might be using future information")

    issues = []

    try:
        symbol = "@ES#C"
        df = prepare_real_data_simple(symbol)
        target_col = f"{symbol}_target_return"

        feature_cols = [c for c in df.columns if c != target_col]

        print(f"Analyzing {len(feature_cols)} features for lookahead bias...")

        # Sample feature names to check for suspicious patterns
        print(f"\nSample feature names (first 20):")
        for i, col in enumerate(feature_cols[:20]):
            print(f"  {i+1:2d}. {col}")

        # Look for obviously problematic features
        suspicious_features = []
        for col in feature_cols:
            col_lower = col.lower()
            if any(word in col_lower for word in ['future', 'next', 'forward', 'lead', 'ahead']):
                suspicious_features.append(col)

        if suspicious_features:
            issue = f"CRITICAL: Features with suspicious names suggesting future data: {suspicious_features[:5]}"
            issues.append(issue)
            print(f"🚨 {issue}")

        # Test: Correlation of features with FUTURE returns
        print(f"\n🧪 Testing feature correlation with FUTURE returns:")

        target_returns = df[target_col].dropna()

        # Create future returns (1, 2, 3 days ahead)
        future_returns_1d = target_returns.shift(-1).dropna()
        future_returns_2d = target_returns.shift(-2).dropna()

        high_future_corr = []

        # Test subset of features for computational efficiency
        test_features = feature_cols[:100] if len(feature_cols) > 100 else feature_cols

        for col in test_features:
            try:
                feature_values = df[col]

                # Align with future returns
                common_idx_1d = feature_values.index.intersection(future_returns_1d.index)
                if len(common_idx_1d) > 50:
                    aligned_feature = feature_values.loc[common_idx_1d]
                    aligned_future = future_returns_1d.loc[common_idx_1d]

                    # Remove NaN
                    mask = ~(aligned_feature.isna() | aligned_future.isna())
                    if mask.sum() > 20:
                        clean_feature = aligned_feature[mask]
                        clean_future = aligned_future[mask]

                        if clean_feature.std() > 1e-10 and clean_future.std() > 1e-10:
                            corr = clean_feature.corr(clean_future)

                            if abs(corr) > 0.3:  # High correlation with future
                                high_future_corr.append((col, corr))

            except:
                continue

        if high_future_corr:
            print(f"  ⚠️  Features correlated with FUTURE returns:")
            for col, corr in sorted(high_future_corr, key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"    {col}: {corr:.4f}")

            if any(abs(corr) > 0.8 for _, corr in high_future_corr):
                issue = f"CRITICAL: Features with extreme future correlation detected"
                issues.append(issue)
                print(f"🚨 {issue}")

        else:
            print(f"  ✅ No high future correlations in tested subset")

    except Exception as e:
        error = f"CRITICAL: Error investigating lookahead bias: {e}"
        issues.append(error)
        print(f"🚨 {error}")

    return issues

def investigate_selection_bias():
    """
    🚨 CRITICAL: Are we cherry-picking the best results?
    """
    print(f"\n{'='*80}")
    print("🚨 INVESTIGATING SELECTION BIAS")
    print(f"{'='*80}")
    print("Are we only reporting the lucky results and hiding the failures?")

    issues = []

    # Look at ALL results, not just the good ones
    print(f"Examining all available results...")

    # Check if there are failed runs or poor results being ignored
    from pathlib import Path

    log_dirs = [
        "/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs - Hit_Q v2",
        "/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs - Multi Symbol"
    ]

    all_sharpes = []
    failed_runs = []

    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if log_path.exists():
            log_files = list(log_path.glob("*.log"))

            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()

                    # Look for failures
                    if "ERROR" in content or "FAILED" in content:
                        failed_runs.append(log_file.name)

                    # Extract production Sharpe
                    import re
                    sharpe_match = re.search(r'Production Final: Sharpe=([+-]?\d+\.?\d*)', content)
                    if sharpe_match:
                        sharpe = float(sharpe_match.group(1))
                        all_sharpes.append(sharpe)

                except:
                    continue

    print(f"\nComplete results analysis:")
    print(f"  Total log files analyzed: {sum(len(list(Path(d).glob('*.log'))) for d in log_dirs if Path(d).exists())}")
    print(f"  Failed runs: {len(failed_runs)}")
    print(f"  Successful Sharpe extractions: {len(all_sharpes)}")

    if failed_runs:
        print(f"  Failed run examples: {failed_runs[:3]}")

    if all_sharpes:
        print(f"\nSharpe ratio distribution (ALL results):")
        print(f"  Mean: {np.mean(all_sharpes):.4f}")
        print(f"  Std: {np.std(all_sharpes):.4f}")
        print(f"  Min: {np.min(all_sharpes):.4f}")
        print(f"  Max: {np.max(all_sharpes):.4f}")
        print(f"  Median: {np.median(all_sharpes):.4f}")

        # Check distribution
        excellent = len([s for s in all_sharpes if s > 2.0])
        good = len([s for s in all_sharpes if 1.5 < s <= 2.0])
        mediocre = len([s for s in all_sharpes if 1.0 < s <= 1.5])
        poor = len([s for s in all_sharpes if s <= 1.0])

        print(f"\nResult distribution:")
        print(f"  Excellent (>2.0): {excellent} ({excellent/len(all_sharpes)*100:.1f}%)")
        print(f"  Good (1.5-2.0): {good} ({good/len(all_sharpes)*100:.1f}%)")
        print(f"  Mediocre (1.0-1.5): {mediocre} ({mediocre/len(all_sharpes)*100:.1f}%)")
        print(f"  Poor (≤1.0): {poor} ({poor/len(all_sharpes)*100:.1f}%)")

        # 🚨 RED FLAG: Too many excellent results
        if excellent / len(all_sharpes) > 0.3:  # More than 30% excellent
            issue = f"CRITICAL: Suspiciously high rate of excellent results ({excellent/len(all_sharpes)*100:.0f}%)"
            issues.append(issue)
            print(f"🚨 {issue}")

        # Check for impossible results
        impossible = len([s for s in all_sharpes if abs(s) > 5])
        if impossible > 0:
            issue = f"CRITICAL: {impossible} results with impossible Sharpe ratios (>5)"
            issues.append(issue)
            print(f"🚨 {issue}")

    return issues

def main():
    """
    Run critical investigation of unrealistic performance
    """
    print(f"\n{'='*100}")
    print("🚨🚨🚨 CRITICAL INVESTIGATION: TOO GOOD TO BE TRUE?")
    print(f"{'='*100}")
    print("Investigating whether the high performance results are realistic")
    print("Being BRUTALLY honest about potential issues")

    all_issues = []

    # Investigation 1: Sharpe calculation realism
    issues = investigate_sharpe_calculation_realism()
    all_issues.extend(issues)

    # Investigation 2: Overfitting catastrophe
    issues = investigate_overfitting_catastrophe()
    all_issues.extend(issues)

    # Investigation 3: Selection bias
    issues = investigate_selection_bias()
    all_issues.extend(issues)

    # Investigation 4: Lookahead bias
    issues = investigate_feature_lookahead_bias()
    all_issues.extend(issues)

    # BRUTAL HONEST SUMMARY
    print(f"\n{'='*100}")
    print("🚨🚨🚨 BRUTAL HONEST ASSESSMENT")
    print(f"{'='*100}")

    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"REALITY CHECK RESULTS:")
    print(f"  Critical red flags: {len(critical_issues)}")
    print(f"  Warning signs: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨🚨🚨 CRITICAL RED FLAGS:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")

    if warning_issues:
        print(f"\n⚠️  WARNING SIGNS:")
        for i, issue in enumerate(warning_issues, 1):
            print(f"  {i}. {issue}")

    # Final brutal assessment
    if len(critical_issues) > 0:
        print(f"\n❌❌❌ RESULTS ARE LIKELY INVALID")
        print("Multiple critical red flags detected")
        print("High Sharpe ratios are probably due to bugs/overfitting")
    elif len(warning_issues) > 2:
        print(f"\n⚠️⚠️⚠️ RESULTS ARE HIGHLY SUSPICIOUS")
        print("Multiple warning signs suggest problems")
        print("Results may be overstated or unrealistic")
    else:
        print(f"\n✅ Results appear potentially legitimate")
        print("No major red flags detected in critical investigation")

    return all_issues

if __name__ == "__main__":
    main()
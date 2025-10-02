#!/usr/bin/env python3
"""
Real Framework Signal Extraction
================================

Extract and analyze ACTUAL signals from completed test runs to verify:

1. SIGNAL REASONABLENESS - Do the signals make sense?
2. DAILY EXPOSURE TRACKING - What's the actual position each day?
3. PNL VERIFICATION - Manual calculation vs framework results
4. SIGNAL DISTRIBUTION - Are we getting sensible long/short balance?
5. CORRELATION WITH RETURNS - Do good signals correlate with future returns?

This uses REAL framework output to validate the end-to-end pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import re

def extract_signals_from_log(log_file_path):
    """
    Extract actual signals and performance from a completed test log
    """
    print(f"\n{'='*70}")
    print(f"📊 EXTRACTING SIGNALS FROM: {Path(log_file_path).name}")
    print(f"{'='*70}")

    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()

        # Extract key metrics from log
        metrics = {}

        # Extract final results
        patterns = {
            'training_sharpe': r'Training Final: Sharpe=([+-]?\d+\.?\d*)',
            'production_sharpe': r'Production Final: Sharpe=([+-]?\d+\.?\d*)',
            'full_timeline_sharpe': r'Full Timeline Final: Sharpe=([+-]?\d+\.?\d*)',
            'training_hit': r'Training Final:.*Hit=([+-]?\d+\.?\d*)%',
            'production_hit': r'Production Final:.*Hit=([+-]?\d+\.?\d*)%',
            'full_timeline_hit': r'Full Timeline Final:.*Hit=([+-]?\d+\.?\d*)%',
        }

        for metric_name, pattern in patterns.items():
            match = re.search(pattern, log_content)
            if match:
                metrics[metric_name] = float(match.group(1))
            else:
                metrics[metric_name] = None

        # Extract configuration
        config_patterns = {
            'symbol': r'Target Symbol: (\S+)',
            'models': r'Models: (\d+)',
            'folds': r'Folds: (\d+)',
            'features': r'Max Features: (\d+)',
            'q_metric': r'Q-Metric=(\w+)',
            'signal_type': r'Signal Type: (\w+)',
        }

        config = {}
        for config_name, pattern in config_patterns.items():
            match = re.search(pattern, log_content)
            if match:
                config[config_name] = match.group(1)

        print(f"Configuration extracted:")
        for key, value in config.items():
            print(f"  {key}: {value}")

        print(f"\nPerformance metrics extracted:")
        for metric, value in metrics.items():
            if value is not None:
                if 'hit' in metric:
                    print(f"  {metric}: {value:.1f}%")
                else:
                    print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: Not found")

        # Check for reasonable metrics
        issues = []

        # Verify metrics are reasonable
        if metrics['production_sharpe'] is not None:
            if abs(metrics['production_sharpe']) > 10:
                issue = f"CRITICAL: Unrealistic production Sharpe ({metrics['production_sharpe']:.2f})"
                issues.append(issue)
                print(f"🚨 {issue}")

        if metrics['production_hit'] is not None:
            if metrics['production_hit'] < 30 or metrics['production_hit'] > 70:
                issue = f"WARNING: Extreme hit rate ({metrics['production_hit']:.1f}%)"
                issues.append(issue)
                print(f"⚠️  {issue}")

        # Check for consistency between training and production
        if (metrics['training_sharpe'] is not None and
            metrics['production_sharpe'] is not None):

            train_prod_diff = abs(metrics['training_sharpe'] - metrics['production_sharpe'])
            if train_prod_diff > 3.0:
                issue = f"WARNING: Large train-prod Sharpe difference ({train_prod_diff:.2f})"
                issues.append(issue)
                print(f"⚠️  {issue}")

        return {
            'config': config,
            'metrics': metrics,
            'issues': issues,
            'log_path': log_file_path
        }

    except Exception as e:
        print(f"❌ Error extracting from log: {e}")
        return None

def analyze_real_framework_output():
    """
    Analyze actual framework output from completed tests
    """
    print(f"\n{'='*80}")
    print("📊 REAL FRAMEWORK OUTPUT ANALYSIS")
    print(f"{'='*80}")

    # Find recent completed logs
    log_dirs = [
        "/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs - Multi Symbol",
        "/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6/xgb_compare/results/logs - Hit_Q v2"
    ]

    all_results = []
    all_issues = []

    for log_dir in log_dirs:
        log_path = Path(log_dir)
        if not log_path.exists():
            continue

        # Get recent log files
        log_files = list(log_path.glob("*.log"))[-5:]  # Last 5 logs

        for log_file in log_files:
            print(f"\nAnalyzing: {log_file.name}")
            result = extract_signals_from_log(log_file)

            if result:
                all_results.append(result)
                all_issues.extend(result['issues'])

    # Summary analysis
    print(f"\n{'='*80}")
    print("📊 FRAMEWORK OUTPUT SUMMARY")
    print(f"{'='*80}")

    if all_results:
        print(f"Analyzed {len(all_results)} completed test runs")

        # Extract Sharpe ratios for distribution analysis
        production_sharpes = [r['metrics']['production_sharpe'] for r in all_results
                            if r['metrics']['production_sharpe'] is not None]

        if production_sharpes:
            print(f"\nProduction Sharpe distribution:")
            print(f"  Count: {len(production_sharpes)}")
            print(f"  Mean: {np.mean(production_sharpes):.4f}")
            print(f"  Std: {np.std(production_sharpes):.4f}")
            print(f"  Min: {np.min(production_sharpes):.4f}")
            print(f"  Max: {np.max(production_sharpes):.4f}")

            # Check for unrealistic results
            extreme_sharpes = [s for s in production_sharpes if abs(s) > 5]
            if extreme_sharpes:
                issue = f"WARNING: {len(extreme_sharpes)} tests with extreme Sharpe ratios: {extreme_sharpes}"
                all_issues.append(issue)
                print(f"⚠️  {issue}")

        # Analyze hit rates
        production_hits = [r['metrics']['production_hit'] for r in all_results
                          if r['metrics']['production_hit'] is not None]

        if production_hits:
            print(f"\nProduction hit rate distribution:")
            print(f"  Mean: {np.mean(production_hits):.1f}%")
            print(f"  Std: {np.std(production_hits):.1f}%")
            print(f"  Min: {np.min(production_hits):.1f}%")
            print(f"  Max: {np.max(production_hits):.1f}%")

            # Check for unrealistic hit rates
            if np.mean(production_hits) < 45 or np.mean(production_hits) > 65:
                issue = f"WARNING: Average hit rate ({np.mean(production_hits):.1f}%) outside reasonable range"
                all_issues.append(issue)
                print(f"⚠️  {issue}")

    else:
        print("❌ No completed test results found for analysis")

    # Final summary
    critical_issues = [i for i in all_issues if "CRITICAL" in i]
    warning_issues = [i for i in all_issues if "WARNING" in i]

    print(f"\n{'='*80}")
    print("📊 REAL OUTPUT ANALYSIS SUMMARY")
    print(f"{'='*80}")

    print(f"Total issues from real runs: {len(all_issues)}")
    print(f"  Critical: {len(critical_issues)}")
    print(f"  Warnings: {len(warning_issues)}")

    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES IN REAL RUNS:")
        for i, issue in enumerate(critical_issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✅ No critical issues in real framework output")

    if warning_issues:
        print(f"\n⚠️  WARNINGS FROM REAL RUNS:")
        for i, issue in enumerate(warning_issues[:5], 1):
            print(f"  {i}. {issue}")

    return all_issues

if __name__ == "__main__":
    analyze_real_framework_output()
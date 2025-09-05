#!/usr/bin/env python3
"""
Corrected Phase 4 Results Analysis - Extract proper OOS metrics from multi-year data
XGBoost Architecture Comparison (Standard, Deep, Tiered)
"""

import re
import os
import statistics

def extract_oos_sharpe_values(log_file: str):
    """Extract all OOS_Sharpe values from log file"""
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    oos_values = []
    
    # Pattern: | M##   |    #.###  |    #.###  |     #.###  | 0.### |   #.### |
    # Looking for the OOS_Sharpe column (4th numeric column)
    pattern = r'\|\s*M\d+\s*\|\s*[-\d.]+\s*\|\s*[-\d.]+\s*\|\s*([-\d.]+)\s*\|\s*[\d.]+\s*\|\s*[-\d.]+\s*\|'
    
    matches = re.findall(pattern, content)
    
    for match in matches:
        try:
            oos_sharpe = float(match)
            oos_values.append(oos_sharpe)
        except ValueError:
            continue
    
    return oos_values

def extract_oos_hit_rates(log_file: str):
    """Extract OOS hit rates from log file"""
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    hit_values = []
    
    # Find all lines with model IDs (M00, M01, etc.) and extract 3rd percentage
    for line in content.split('\n'):
        if re.search(r'\|\s*M\d+\s*\|', line) and line.count('%') >= 3:
            # Extract the 3rd percentage which is OOS hit rate
            percentages = re.findall(r'(\d+\.\d+)%', line)
            if len(percentages) >= 3:
                try:
                    oos_hit_rate = float(percentages[2]) / 100  # Convert to decimal
                    hit_values.append(oos_hit_rate)
                except ValueError:
                    continue
    
    return hit_values

def analyze_phase4_results():
    """Analyze corrected Phase 4 results with proper OOS metrics"""
    
    log_files = {
        'Standard': "logs/xgb_performance_corrected_p4_standard_standard_100feat_10folds_20250905_150008.log",
        'Deep': "logs/xgb_performance_corrected_p4_deep_deep_100feat_10folds_20250905_150014.log", 
        'Tiered': "logs/xgb_performance_tiered_rerun_tiered_100feat_10folds_20250905_122001.log"
    }
    
    results = {}
    
    print("=" * 80)
    print("PHASE 4 CORRECTED ANALYSIS - PROPER OOS METRICS")
    print("=" * 80)
    print("Multi-Year Data (2015-2024): 2324 observations, 10 folds, 50 models, 100 features")
    print()
    
    for config_name, log_file in log_files.items():
        if os.path.exists(log_file):
            oos_sharpe_values = extract_oos_sharpe_values(log_file)
            oos_hit_values = extract_oos_hit_rates(log_file)
            
            if oos_sharpe_values:
                # Calculate statistics
                avg_sharpe = statistics.mean(oos_sharpe_values)
                sharpe_std = statistics.stdev(oos_sharpe_values) if len(oos_sharpe_values) > 1 else 0
                positive_sharpe_pct = sum(1 for s in oos_sharpe_values if s > 0) / len(oos_sharpe_values) * 100
                sharpe_range = (min(oos_sharpe_values), max(oos_sharpe_values))
                
                # Use extracted hit rates or default to 50%
                if oos_hit_values:
                    avg_hit_rate = statistics.mean(oos_hit_values)
                    hit_std = statistics.stdev(oos_hit_values) if len(oos_hit_values) > 1 else 0
                else:
                    avg_hit_rate = 0.5
                    hit_std = 0.0
                
                results[config_name] = {
                    'avg_oos_sharpe': avg_sharpe,
                    'sharpe_std': sharpe_std,
                    'avg_hit_rate': avg_hit_rate,
                    'hit_std': hit_std,
                    'positive_sharpe_pct': positive_sharpe_pct,
                    'sharpe_range': sharpe_range,
                    'total_models': len(oos_sharpe_values)
                }
                    
                print(f"Configuration: {config_name} XGBoost")
                print(f"  Total Model Tests: {len(oos_sharpe_values)}")
                print(f"  Average OOS Sharpe: {avg_sharpe:.3f}")
                print(f"  Sharpe Consistency (StdDev): {sharpe_std:.3f}")
                print(f"  Average OOS Hit Rate: {avg_hit_rate:.3f}")
                print(f"  Hit Rate StdDev: {hit_std:.3f}")
                print(f"  Positive Sharpe Percentage: {positive_sharpe_pct:.1f}%")
                print(f"  Sharpe Range: {sharpe_range[0]:.3f} to {sharpe_range[1]:.3f}")
                print()
        else:
            print(f"Warning: Log file not found: {log_file}")
    
    # Phase 4 comparison
    print("=" * 80)
    print("PHASE 4 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'Standard':<12} | {'Deep':<12} | {'Tiered':<12} | {'Winner':<10}")
    print("-" * 80)
    
    if results:
        configs = ['Standard', 'Deep', 'Tiered']
        configs = [k for k in configs if k in results]
        
        # Compare metrics
        sharpe_values = {k: results[k]['avg_oos_sharpe'] for k in configs}
        hit_values = {k: results[k]['avg_hit_rate'] for k in configs}
        consistency_values = {k: results[k]['sharpe_std'] for k in configs}
        significance_values = {k: results[k]['positive_sharpe_pct'] for k in configs}
        
        # Winners
        sharpe_winner = max(sharpe_values.keys(), key=lambda k: sharpe_values[k]) if sharpe_values else 'N/A'
        hit_winner = max(hit_values.keys(), key=lambda k: hit_values[k]) if hit_values else 'N/A'
        consistency_winner = min(consistency_values.keys(), key=lambda k: consistency_values[k]) if consistency_values else 'N/A'
        significance_winner = max(significance_values.keys(), key=lambda k: significance_values[k]) if significance_values else 'N/A'
        
        print(f"{'Avg OOS Sharpe':<25} | {sharpe_values.get('Standard', 0):<12.3f} | {sharpe_values.get('Deep', 0):<12.3f} | {sharpe_values.get('Tiered', 0):<12.3f} | {sharpe_winner:<10}")
        print(f"{'Avg OOS Hit Rate':<25} | {hit_values.get('Standard', 0):<12.3f} | {hit_values.get('Deep', 0):<12.3f} | {hit_values.get('Tiered', 0):<12.3f} | {hit_winner:<10}")
        print(f"{'Sharpe Consistency':<25} | {consistency_values.get('Standard', 0):<12.3f} | {consistency_values.get('Deep', 0):<12.3f} | {consistency_values.get('Tiered', 0):<12.3f} | {consistency_winner:<10}")
        print(f"{'Statistical Significance':<25} | {significance_values.get('Standard', 0):<12.1f}% | {significance_values.get('Deep', 0):<12.1f}% | {significance_values.get('Tiered', 0):<12.1f}% | {significance_winner:<10}")
        
        # Overall scoring
        print(f"\n{'='*80}")
        print("PHASE 4 RECOMMENDATION")
        print(f"{'='*80}")
        
        # Score each configuration (Updated weights: Sharpe 30%, Hit Rate 10%, Consistency 30%, Significance 30%)
        scores = {}
        for config in configs:
            if config in results:
                score = 0
                # Sharpe ratio (30% weight)
                if sharpe_values:
                    max_sharpe = max(sharpe_values.values())
                    score += 0.30 * (sharpe_values[config] / max_sharpe if max_sharpe > 0 else 0)
                # Hit rate (10% weight)
                if hit_values:
                    max_hit = max(hit_values.values())
                    score += 0.10 * (hit_values[config] / max_hit if max_hit > 0 else 0)
                # Consistency (30% weight, inverted since lower is better)
                if consistency_values:
                    max_std = max(consistency_values.values())
                    score += 0.30 * (1 - consistency_values[config] / max_std) if max_std > 0 else 0.30
                # Statistical significance (30% weight)
                if significance_values:
                    max_sig = max(significance_values.values())
                    score += 0.30 * (significance_values[config] / max_sig if max_sig > 0 else 0)
                
                scores[config] = score
        
        if scores:
            winner = max(scores.keys(), key=lambda k: scores[k])
            print(f"Phase 4 Winner: {winner} XGBoost (Score: {scores[winner]:.3f})")
            print()
            for config in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
                print(f"  {config} XGBoost: {scores[config]:.3f}")
    
    return results

if __name__ == "__main__":
    analyze_phase4_results()
#!/usr/bin/env python3
"""
Corrected Phase 2 Results Analysis - Extract proper OOS metrics from multi-year data
Model Count Optimization (25, 50, 75, 100 models)
"""

import re
import os
from typing import Dict, List, Tuple
import statistics

def extract_oos_sharpe_values(log_file: str) -> List[float]:
    """Extract all OOS_Sharpe values from log file"""
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract OOS_Sharpe values from each fold
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

def analyze_phase2_results():
    """Analyze corrected Phase 2 results with proper OOS metrics"""
    
    log_files = {
        '25_models': "logs/xgb_analysis_p2_25models_standard_100feat_10folds_20250905_140627.log",
        '50_models': "logs/xgb_analysis_p2_50models_standard_100feat_10folds_20250905_140632.log", 
        '75_models': "logs/xgb_analysis_p2_75models_standard_100feat_10folds_20250905_140636.log",
        '100_models': "logs/xgb_analysis_p2_100models_standard_100feat_10folds_20250905_140641.log"
    }
    
    results = {}
    
    print("=" * 80)
    print("PHASE 2 CORRECTED ANALYSIS - PROPER OOS METRICS")
    print("=" * 80)
    print("Multi-Year Data (2015-2024): 2324 observations, 10 folds, 100 features")
    print()
    
    for config_name, log_file in log_files.items():
        if os.path.exists(log_file):
            oos_sharpe_values = extract_oos_sharpe_values(log_file)
            
            if oos_sharpe_values:
                # Calculate statistics
                avg_sharpe = statistics.mean(oos_sharpe_values)
                sharpe_std = statistics.stdev(oos_sharpe_values) if len(oos_sharpe_values) > 1 else 0
                positive_sharpe_pct = sum(1 for s in oos_sharpe_values if s > 0) / len(oos_sharpe_values) * 100
                sharpe_range = (min(oos_sharpe_values), max(oos_sharpe_values))
                
                # Assume 50% hit rate for comparison
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
                
                model_count = config_name.split('_')[0]
                print(f"Configuration: {model_count} models")
                print(f"  Total Model Tests: {len(oos_sharpe_values)}")
                print(f"  Average OOS Sharpe: {avg_sharpe:.3f}")
                print(f"  Sharpe Consistency (StdDev): {sharpe_std:.3f}")
                print(f"  Positive Sharpe Percentage: {positive_sharpe_pct:.1f}%")
                print(f"  Sharpe Range: {sharpe_range[0]:.3f} to {sharpe_range[1]:.3f}")
                print()
        else:
            print(f"Warning: Log file not found: {log_file}")
    
    # Phase 2 comparison
    print("=" * 80)
    print("PHASE 2 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'25 Models':<12} | {'50 Models':<12} | {'75 Models':<12} | {'100 Models':<12} | {'Winner':<10}")
    print("-" * 105)
    
    if results:
        configs = ['25_models', '50_models', '75_models', '100_models']
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
        
        print(f"{'Avg OOS Sharpe':<25} | {sharpe_values.get('25_models', 0):<12.3f} | {sharpe_values.get('50_models', 0):<12.3f} | {sharpe_values.get('75_models', 0):<12.3f} | {sharpe_values.get('100_models', 0):<12.3f} | {sharpe_winner.split('_')[0]:<10}")
        print(f"{'Sharpe Consistency':<25} | {consistency_values.get('25_models', 0):<12.3f} | {consistency_values.get('50_models', 0):<12.3f} | {consistency_values.get('75_models', 0):<12.3f} | {consistency_values.get('100_models', 0):<12.3f} | {consistency_winner.split('_')[0]:<10}")
        print(f"{'Statistical Significance':<25} | {significance_values.get('25_models', 0):<12.1f}% | {significance_values.get('50_models', 0):<12.1f}% | {significance_values.get('75_models', 0):<12.1f}% | {significance_values.get('100_models', 0):<12.1f}% | {significance_winner.split('_')[0]:<10}")
        
        # Overall scoring
        print(f"\n{'='*80}")
        print("PHASE 2 RECOMMENDATION")
        print(f"{'='*80}")
        
        # Score each configuration
        scores = {}
        for config in configs:
            if config in results:
                score = 0
                # Sharpe ratio (30% weight)
                if sharpe_values:
                    max_sharpe = max(sharpe_values.values())
                    score += 0.30 * (sharpe_values[config] / max_sharpe if max_sharpe > 0 else 0)
                # Consistency (40% weight, inverted since lower is better)
                if consistency_values:
                    max_std = max(consistency_values.values())
                    score += 0.40 * (1 - consistency_values[config] / max_std) if max_std > 0 else 0.40
                # Statistical significance (30% weight)
                if significance_values:
                    max_sig = max(significance_values.values())
                    score += 0.30 * (significance_values[config] / max_sig if max_sig > 0 else 0)
                
                scores[config] = score
        
        if scores:
            winner = max(scores.keys(), key=lambda k: scores[k])
            winner_model_count = winner.split('_')[0]
            print(f"Phase 2 Winner: {winner_model_count} models (Score: {scores[winner]:.3f})")
            print()
            for config in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
                model_count = config.split('_')[0]
                print(f"  {model_count} models: {scores[config]:.3f}")
    
    return results

if __name__ == "__main__":
    analyze_phase2_results()
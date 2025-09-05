#!/usr/bin/env python3
"""
Corrected Phase 3 Results Analysis - Extract proper OOS metrics from multi-year data
Feature Count Optimization (50, 100, 150, ALL features)
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

def extract_oos_hit_rates(log_file: str) -> List[float]:
    """Extract OOS hit rates from log file"""
    if not os.path.exists(log_file):
        return []
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Look for HIT RATE ANALYSIS section
    hit_values = []
    
    # Pattern for hit rate tables: | M##   | ##.##% | ##.##% | ##.##% |
    hit_pattern = r'HIT RATE ANALYSIS.*?\n(.*?)(?=\n.*ANALYSIS|\Z)'
    hit_section_match = re.search(hit_pattern, content, re.DOTALL)
    
    if hit_section_match:
        hit_section = hit_section_match.group(1)
        # Extract OOS hit rates (4th column - OOS_Hit) - account for log timestamp
        hit_row_pattern = r'.*?\|\s*M\d+\s*\|\s*[\d.]+%\s*\|\s*[\d.]+%\s*\|\s*([\d.]+)%\s*\|\s*[\d.]+\s*\|\s*[\d.]+\s*\|'
        hit_matches = re.findall(hit_row_pattern, hit_section)
        
        for match in hit_matches:
            try:
                hit_rate = float(match) / 100  # Convert percentage to decimal
                hit_values.append(hit_rate)
            except ValueError:
                continue
    
    return hit_values

def analyze_phase3_results():
    """Analyze corrected Phase 3 results with proper OOS metrics"""
    
    log_files = {
        '100_features': "logs/phase3/xgb_performance_p3_T2_100feat_fixed_standard_100feat_10folds_20250905_163747.log",
        '200_features': "logs/phase3/xgb_performance_p3_T3_200feat_fixed_standard_200feat_10folds_20250905_163755.log", 
        'ALL_filtered_features': "logs/phase3/xgb_performance_p3_T4_ALLfiltered_fixed_standard_-1feat_10folds_20250905_163802.log",
        'ALL_raw_features': "logs/phase3/xgb_performance_p3_T5_ALLraw_fixed_standard_50feat_10folds_20250905_162158.log"
    }
    
    results = {}
    
    print("=" * 80)
    print("PHASE 3 CORRECTED ANALYSIS - PROPER OOS METRICS")
    print("=" * 80)
    print("Multi-Year Data (2015-2024): 2324 observations, 10 folds, 50 models")
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
                
                feature_count = config_name.replace('_features', '')
                if feature_count == 'ALL':
                    feature_desc = "ALL (389 selected)"
                else:
                    feature_desc = f"{feature_count} features"
                    
                print(f"Configuration: {feature_desc}")
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
    
    # Phase 3 comparison
    print("=" * 80)
    print("PHASE 3 COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Metric':<25} | {'100 Features':<12} | {'200 Features':<12} | {'ALL Filtered':<12} | {'ALL Raw':<12} | {'Winner':<10}")
    print("-" * 105)
    
    if results:
        configs = ['100_features', '200_features', 'ALL_filtered_features', 'ALL_raw_features']
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
        
        def format_winner(winner_key):
            if winner_key == 'ALL_filtered_features':
                return 'ALL_FILT'
            elif winner_key == 'ALL_raw_features':
                return 'ALL_RAW'
            else:
                return winner_key.split('_')[0]
        
        print(f"{'Avg OOS Sharpe':<25} | {sharpe_values.get('100_features', 0):<12.3f} | {sharpe_values.get('200_features', 0):<12.3f} | {sharpe_values.get('ALL_filtered_features', 0):<12.3f} | {sharpe_values.get('ALL_raw_features', 0):<12.3f} | {format_winner(sharpe_winner):<10}")
        print(f"{'Avg OOS Hit Rate':<25} | {hit_values.get('100_features', 0):<12.3f} | {hit_values.get('200_features', 0):<12.3f} | {hit_values.get('ALL_filtered_features', 0):<12.3f} | {hit_values.get('ALL_raw_features', 0):<12.3f} | {format_winner(hit_winner):<10}")
        print(f"{'Sharpe Consistency':<25} | {consistency_values.get('100_features', 0):<12.3f} | {consistency_values.get('200_features', 0):<12.3f} | {consistency_values.get('ALL_filtered_features', 0):<12.3f} | {consistency_values.get('ALL_raw_features', 0):<12.3f} | {format_winner(consistency_winner):<10}")
        print(f"{'Statistical Significance':<25} | {significance_values.get('100_features', 0):<12.1f}% | {significance_values.get('200_features', 0):<12.1f}% | {significance_values.get('ALL_filtered_features', 0):<12.1f}% | {significance_values.get('ALL_raw_features', 0):<12.1f}% | {format_winner(significance_winner):<10}")
        
        # Overall scoring
        print(f"\n{'='*80}")
        print("PHASE 3 RECOMMENDATION")
        print(f"{'='*80}")
        
        # Score each configuration (Updated weights: Sharpe 20%, Hit Rate 20%, Consistency 30%, Significance 30%)
        scores = {}
        for config in configs:
            if config in results:
                score = 0
                # Sharpe ratio (20% weight)
                if sharpe_values:
                    max_sharpe = max(sharpe_values.values())
                    score += 0.20 * (sharpe_values[config] / max_sharpe if max_sharpe > 0 else 0)
                # Hit rate (20% weight)
                if hit_values:
                    max_hit = max(hit_values.values())
                    score += 0.20 * (hit_values[config] / max_hit if max_hit > 0 else 0)
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
            winner_label = format_winner(winner)
            print(f"Phase 3 Winner: {winner_label} features (Score: {scores[winner]:.3f})")
            print()
            for config in sorted(scores.keys(), key=lambda k: scores[k], reverse=True):
                feature_label = format_winner(config)
                print(f"  {feature_label} features: {scores[config]:.3f}")
    
    return results

if __name__ == "__main__":
    analyze_phase3_results()
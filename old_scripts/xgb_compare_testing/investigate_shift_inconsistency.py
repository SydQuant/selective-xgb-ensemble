"""
Investigate Shift Logic Inconsistency Across Codebase

This script identifies and analyzes all the different PnL calculation approaches
used throughout the codebase and shows the exact inconsistencies.
"""

import os
import re
import pandas as pd
import numpy as np

def scan_files_for_pnl_calculations():
    """
    Scan all Python files for PnL calculation patterns
    """
    print("=== SCANNING CODEBASE FOR PNL CALCULATION PATTERNS ===")
    
    base_path = "/Users/steven/Projects/SQ/bond_ls_xgb_grope_full_v6"
    
    # Patterns to look for
    patterns = {
        'shifted_pnl': r'\.shift\(1\).*\*.*',  # signal.shift(1) * returns
        'direct_pnl': r'signal\s*\*\s*(?:returns|actual_returns|y)',  # signal * returns (no shift)
        'calculate_model_metrics': r'calculate_model_metrics\(',
        'calculate_model_metrics_from_pnl': r'calculate_model_metrics_from_pnl\(',
        'shifted_true': r'shifted\s*=\s*True',
        'shifted_false': r'shifted\s*=\s*False'
    }
    
    findings = {}
    
    # Key files to check
    key_files = [
        'xgb_compare/xgb_compare.py',
        'xgb_compare/full_timeline_backtest.py', 
        'xgb_compare/metrics_utils.py',
        'main.py',
        'metrics/simple_metrics.py',
        'metrics/dapy.py',
        'metrics/performance_report.py'
    ]
    
    for file_rel_path in key_files:
        file_path = os.path.join(base_path, file_rel_path)
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
            
            file_findings = {}
            
            for pattern_name, pattern in patterns.items():
                matches = []
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line) and not line.strip().startswith('#'):
                        matches.append({
                            'line_num': i,
                            'code': line.strip(),
                            'context': lines[max(0, i-2):i+1] if i > 1 else lines[i-1:i+1]
                        })
                
                if matches:
                    file_findings[pattern_name] = matches
            
            if file_findings:
                findings[file_rel_path] = file_findings
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    return findings

def analyze_inconsistencies(findings):
    """
    Analyze the findings to show inconsistencies
    """
    print("\n=== ANALYZING INCONSISTENCIES ===")
    
    # Group by approach
    shifted_approach_files = []
    direct_approach_files = []
    mixed_approach_files = []
    
    for file_path, file_findings in findings.items():
        has_shifted = any(pattern in file_findings for pattern in ['shifted_pnl', 'shifted_true'])
        has_direct = any(pattern in file_findings for pattern in ['direct_pnl', 'shifted_false'])
        
        if has_shifted and has_direct:
            mixed_approach_files.append(file_path)
        elif has_shifted:
            shifted_approach_files.append(file_path)
        elif has_direct:
            direct_approach_files.append(file_path)
    
    print(f"📊 APPROACH DISTRIBUTION:")
    print(f"Files using SHIFTED approach only: {len(shifted_approach_files)}")
    print(f"Files using DIRECT approach only: {len(direct_approach_files)}")
    print(f"Files using MIXED approaches: {len(mixed_approach_files)}")
    
    return shifted_approach_files, direct_approach_files, mixed_approach_files

def show_detailed_inconsistencies(findings):
    """
    Show detailed inconsistencies with actual code examples
    """
    print("\n=== DETAILED INCONSISTENCY ANALYSIS ===")
    
    for file_path, file_findings in findings.items():
        print(f"\n📁 {file_path}:")
        
        # Check what approaches this file uses
        approaches = []
        
        if 'shifted_pnl' in file_findings or 'shifted_true' in file_findings:
            approaches.append("SHIFTED (artificial lag)")
        
        if 'direct_pnl' in file_findings or 'shifted_false' in file_findings:
            approaches.append("DIRECT (no lag)")
        
        if 'calculate_model_metrics' in file_findings:
            approaches.append("LEGACY FUNCTION")
            
        if 'calculate_model_metrics_from_pnl' in file_findings:
            approaches.append("NEW FUNCTION")
        
        print(f"   Approaches used: {', '.join(approaches) if approaches else 'None detected'}")
        
        # Show specific code examples
        for pattern_name, matches in file_findings.items():
            if pattern_name in ['shifted_pnl', 'direct_pnl', 'calculate_model_metrics', 'shifted_true', 'shifted_false']:
                print(f"   {pattern_name.upper()}:")
                for match in matches[:3]:  # Show first 3 matches
                    print(f"      Line {match['line_num']}: {match['code']}")

def demonstrate_the_actual_problem():
    """
    Demonstrate the actual numerical impact of the inconsistency
    """
    print("\n=== DEMONSTRATING NUMERICAL IMPACT ===")
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    predictions = pd.Series([0.1, -0.2, 0.3, -0.1, 0.2, 0.1, -0.3, 0.2, -0.1, 0.1], index=dates)
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015, 0.005, -0.02, 0.01, -0.005, 0.008], index=dates)
    
    print("Test data (first 5 periods):")
    for i in range(5):
        print(f"Period {i+1}: prediction={predictions.iloc[i]:6.1f}, return={returns.iloc[i]:7.3f}")
    
    # Method 1: Direct calculation (used in backtester)
    pnl_direct = predictions * returns
    sharpe_direct = pnl_direct.mean() / pnl_direct.std()
    
    print(f"\n📊 DIRECT METHOD (backtester approach):")
    print(f"   PnL = prediction[t] × return[t]")
    print(f"   First 5 PnL: {pnl_direct.head().values}")
    print(f"   Sharpe: {sharpe_direct:.4f}")
    
    # Method 2: Shifted calculation (used in training)
    predictions_shifted = predictions.shift(1).fillna(0.0)
    pnl_shifted = predictions_shifted * returns
    sharpe_shifted = pnl_shifted.mean() / pnl_shifted.std()
    
    print(f"\n📊 SHIFTED METHOD (training approach):")
    print(f"   PnL = prediction[t-1] × return[t]")
    print(f"   Shifted predictions: {predictions_shifted.head().values}")
    print(f"   First 5 PnL: {pnl_shifted.head().values}")
    print(f"   Sharpe: {sharpe_shifted:.4f}")
    
    # Show the difference
    sharpe_diff = abs(sharpe_direct - sharpe_shifted)
    pnl_diff = abs(pnl_direct.sum() - pnl_shifted.sum())
    
    print(f"\n🚨 THE INCONSISTENCY:")
    print(f"   Sharpe difference: {sharpe_diff:.4f}")
    print(f"   Total PnL difference: {pnl_diff:.6f}")
    print(f"   → Training metrics != Backtesting metrics")
    print(f"   → This makes it impossible to compare them meaningfully!")

def trace_through_actual_pipeline():
    """
    Trace through the actual pipeline to show where each approach is used
    """
    print("\n=== TRACING THROUGH ACTUAL PIPELINE ===")
    
    pipeline_flow = [
        {
            'stage': 'Training - Model Evaluation',
            'file': 'xgb_compare/xgb_compare.py',
            'lines': [68, 69, 70],
            'approach': 'SHIFTED (calculate_model_metrics with shifted=True)',
            'impact': 'Training metrics use artificial 1-day lag'
        },
        {
            'stage': 'Backtesting - Model Evaluation', 
            'file': 'xgb_compare/full_timeline_backtest.py',
            'line': 196,
            'approach': 'DIRECT (calculate_model_metrics_from_pnl)',
            'impact': 'Backtesting uses direct PnL calculation'
        },
        {
            'stage': 'Legacy Functions',
            'file': 'metrics/simple_metrics.py',
            'approach': 'SHIFTED (signal.shift(1))',
            'impact': 'Old code still uses artificial lag'
        }
    ]
    
    print("Pipeline Flow Analysis:")
    
    for i, stage in enumerate(pipeline_flow, 1):
        print(f"\n{i}. {stage['stage']}:")
        print(f"   📁 File: {stage['file']}")
        print(f"   🔧 Approach: {stage['approach']}")
        print(f"   💥 Impact: {stage['impact']}")
        
        if stage['stage'] == 'Training - Model Evaluation':
            print(f"   📋 Details:")
            print(f"      Line 68: is_metrics = calculate_model_metrics(..., shifted=False)")
            print(f"      Line 69: iv_metrics = calculate_model_metrics(..., shifted=False)")  
            print(f"      Line 70: oos_metrics = calculate_model_metrics(..., shifted=True) ← PROBLEM!")
    
    print(f"\n🎯 KEY INSIGHT:")
    print("The OOS (out-of-sample) metrics during TRAINING use shifted=True")
    print("But the BACKTESTING uses direct calculation")
    print("This means:")
    print("  - Training OOS Sharpe ≠ Backtesting Sharpe")
    print("  - Model selection during training may be suboptimal") 
    print("  - Performance estimates are inconsistent")

def show_fix_recommendation():
    """
    Show the recommended fix
    """
    print("\n=== RECOMMENDED FIX ===")
    
    print("🔧 OPTION 1: Make everything DIRECT (RECOMMENDED)")
    print("   ✓ Change xgb_compare.py line 70 to use shifted=False")
    print("   ✓ Or better: use calculate_model_metrics_from_pnl() directly")
    print("   ✓ This aligns training with backtesting")
    
    print("\n🔧 OPTION 2: Make everything SHIFTED (NOT RECOMMENDED)")
    print("   ✗ Change backtesting to use artificial lag")
    print("   ✗ This doesn't match production conditions")
    
    print("\n📋 SPECIFIC CHANGES NEEDED:")
    print("In xgb_compare.py, replace line 70:")
    print("   OLD: oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=True)")
    print("   NEW: oos_metrics = calculate_model_metrics(pred_test, y_test, shifted=False)")
    print("   OR:  pnl = pred_test * y_test")
    print("        oos_metrics = calculate_model_metrics_from_pnl(pnl, pred_test, y_test)")

if __name__ == "__main__":
    print("🔍 INVESTIGATING SHIFT LOGIC INCONSISTENCY")
    print("="*70)
    
    # Step 1: Scan codebase
    findings = scan_files_for_pnl_calculations()
    
    # Step 2: Analyze inconsistencies  
    shifted_files, direct_files, mixed_files = analyze_inconsistencies(findings)
    
    # Step 3: Show detailed examples
    show_detailed_inconsistencies(findings)
    
    # Step 4: Demonstrate numerical impact
    demonstrate_the_actual_problem()
    
    # Step 5: Trace pipeline
    trace_through_actual_pipeline()
    
    # Step 6: Show fix
    show_fix_recommendation()
    
    print("\n" + "="*70)
    print("🏁 INVESTIGATION COMPLETE")
    
    print(f"\n🚨 CRITICAL FINDING:")
    print("Training and backtesting use fundamentally different PnL calculations!")
    print("This makes performance comparison meaningless and model selection suboptimal.")
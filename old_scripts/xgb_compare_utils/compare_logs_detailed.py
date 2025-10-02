#!/usr/bin/env python3
"""
Compare logs in detail to verify identical framework behavior
"""

import re
from pathlib import Path

def extract_key_metrics_from_log(log_file):
    """Extract key metrics and model selections from log file."""

    with open(log_file, 'r') as f:
        content = f.read()

    metrics = {
        'config': {},
        'fold_summaries': [],
        'final_selection': None,
        'final_metrics': {}
    }

    # Extract configuration
    config_section = re.search(r'Configuration:\s*\n(.*?)\n\s*\n', content, re.DOTALL)
    if config_section:
        config_lines = config_section.group(1).split('\n')
        for line in config_lines:
            if ':' in line:
                key, value = line.split(':', 1)
                metrics['config'][key.strip()] = value.strip()

    # Extract fold summaries
    fold_pattern = r'Fold (\d+) Summary:\s*\n.*?Best OOS Sharpe: (M\d+) \(([\d.]+).*?\n.*?Best.*?: (M\d+) \(([\d.]+).*?\n.*?Mean OOS Sharpe: ([\d.-]+), Mean Hit: ([\d.]+)'
    fold_matches = re.findall(fold_pattern, content, re.DOTALL)

    for match in fold_matches:
        fold_num, best_model, best_sharpe, q_model, q_score, mean_sharpe, mean_hit = match
        metrics['fold_summaries'].append({
            'fold': int(fold_num),
            'best_oos_model': best_model,
            'best_oos_sharpe': float(best_sharpe),
            'q_model': q_model,
            'q_score': float(q_score),
            'mean_sharpe': float(mean_sharpe),
            'mean_hit': float(mean_hit)
        })

    # Extract final metrics
    final_pattern = r'Training Final: Sharpe=([\d.-]+).*?Production Final: Sharpe=([\d.-]+).*?Full Timeline Final: Sharpe=([\d.-]+)'
    final_match = re.search(final_pattern, content, re.DOTALL)
    if final_match:
        metrics['final_metrics'] = {
            'training_sharpe': float(final_match.group(1)),
            'production_sharpe': float(final_match.group(2)),
            'full_timeline_sharpe': float(final_match.group(3))
        }

    # Extract final model selection
    selection_pattern = r'Model Selection History:.*?Fold (\d+): ([M\d, ]+)'
    selections = re.findall(selection_pattern, content, re.DOTALL)
    if selections:
        last_fold, models = selections[-1]
        model_list = [m.strip() for m in models.split(',')]
        metrics['final_selection'] = {
            'fold': int(last_fold),
            'models': model_list
        }

    return metrics

def compare_logs(original_log, new_log):
    """Compare two logs for identical behavior."""

    print("="*80)
    print("DETAILED LOG COMPARISON")
    print("="*80)

    # Extract metrics from both logs
    original_metrics = extract_key_metrics_from_log(original_log)
    new_metrics = extract_key_metrics_from_log(new_log)

    # Compare configurations
    print("CONFIGURATION COMPARISON:")
    print("-" * 40)

    config_match = True
    for key in ['Target Symbol', 'Models', 'Folds', 'Max Features', 'Q-Metric']:
        orig_val = original_metrics['config'].get(key, 'N/A')
        new_val = new_metrics['config'].get(key, 'N/A')
        match = orig_val == new_val
        config_match = config_match and match
        print(f"  {key}: {'MATCH' if match else 'DIFFER'} ({orig_val} vs {new_val})")

    # Compare fold-by-fold results
    print(f"\nFOLD-BY-FOLD COMPARISON:")
    print("-" * 40)

    orig_folds = original_metrics['fold_summaries']
    new_folds = new_metrics['fold_summaries']

    if len(orig_folds) != len(new_folds):
        print(f"ERROR: Different number of folds ({len(orig_folds)} vs {len(new_folds)})")
        return False

    fold_matches = 0
    for i, (orig, new) in enumerate(zip(orig_folds, new_folds)):
        # Compare key metrics
        sharpe_match = abs(orig['best_oos_sharpe'] - new['best_oos_sharpe']) < 1e-6
        model_match = orig['best_oos_model'] == new['best_oos_model']

        if sharpe_match and model_match:
            fold_matches += 1

        if i < 5:  # Show first 5 folds
            print(f"  Fold {orig['fold']:2d}: Model {'MATCH' if model_match else 'DIFFER'} ({orig['best_oos_model']} vs {new['best_oos_model']}), Sharpe {'MATCH' if sharpe_match else 'DIFFER'} ({orig['best_oos_sharpe']:.3f} vs {new['best_oos_sharpe']:.3f})")

    print(f"  Fold matches: {fold_matches}/{len(orig_folds)}")

    # Compare final metrics
    print(f"\nFINAL METRICS COMPARISON:")
    print("-" * 40)

    orig_final = original_metrics['final_metrics']
    new_final = new_metrics['final_metrics']

    metrics_match = True
    for metric in ['training_sharpe', 'production_sharpe', 'full_timeline_sharpe']:
        if metric in orig_final and metric in new_final:
            diff = abs(orig_final[metric] - new_final[metric])
            match = diff < 1e-6
            metrics_match = metrics_match and match
            print(f"  {metric}: {'MATCH' if match else 'DIFFER'} ({orig_final[metric]:.3f} vs {new_final[metric]:.3f}, diff={diff:.2e})")

    # Compare final model selection
    print(f"\nMODEL SELECTION COMPARISON:")
    print("-" * 40)

    orig_selection = original_metrics.get('final_selection', {})
    new_selection = new_metrics.get('final_selection', {})

    if orig_selection and new_selection:
        models_match = set(orig_selection['models']) == set(new_selection['models'])
        print(f"  Final models: {'MATCH' if models_match else 'DIFFER'}")
        print(f"    Original: {orig_selection['models']}")
        print(f"    New:      {new_selection['models']}")

    # Overall verdict
    overall_match = config_match and (fold_matches == len(orig_folds)) and metrics_match and models_match

    print(f"\n{'='*60}")
    print("OVERALL COMPARISON VERDICT")
    print(f"{'='*60}")

    if overall_match:
        print("SUCCESS: Logs are IDENTICAL")
        print("- Configuration matches")
        print("- All fold results match")
        print("- Final metrics match")
        print("- Model selection matches")
        print("FRAMEWORK IS PRODUCING IDENTICAL RESULTS")
    else:
        print("ERROR: Logs DIFFER")
        print("- Framework behavior has changed")
        print("- Cannot guarantee same model extraction")

    return overall_match

def main():
    """Main comparison function."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare log files for identical behavior')
    parser.add_argument('--original', required=True, help='Original log file path')
    parser.add_argument('--new', required=True, help='New log file path')

    args = parser.parse_args()

    if not Path(args.original).exists():
        print(f"ERROR: Original log not found: {args.original}")
        return

    if not Path(args.new).exists():
        print(f"ERROR: New log not found: {args.new}")
        return

    print("Log Comparison Tool")
    print("=" * 60)
    print(f"Original: {args.original}")
    print(f"New:      {args.new}")

    success = compare_logs(args.original, args.new)

    if success:
        print("\nOVERALL: VALIDATION PASSED")
    else:
        print("\nOVERALL: VALIDATION FAILED")

if __name__ == "__main__":
    main()
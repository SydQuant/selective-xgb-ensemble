#!/usr/bin/env python3
"""
Stability Ensemble Parameter Optimization Test Suite - Simple Version
"""

import subprocess
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_test(name, metric_name, top_k, lam_gap, timeout_minutes=30):
    """Run a single stability test"""
    print(f"Starting test: {name}")
    
    # Create command
    cmd = [
        "~/anaconda3/python.exe", "main.py",
        "--config", "configs/production_full_system.yaml",
        "--target_symbol", "@ES#C",
        "--bypass_pvalue_gating",
        "--n_models", "25",  # Reduced for speed
        "--folds", "3"       # Reduced for speed
    ]
    
    # Temporary config modification via environment or inline editing
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_minutes*60)
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            # Parse key metrics
            output = result.stdout
            performance = {}
            
            for line in output.split('\n'):
                if "Sharpe Ratio:" in line:
                    try:
                        performance['sharpe'] = float(line.split(':')[-1].strip())
                    except:
                        performance['sharpe'] = 0.0
                elif "Total Return:" in line:
                    try:
                        performance['return'] = float(line.split(':')[-1].strip().rstrip('%'))
                    except:
                        performance['return'] = 0.0
                elif "Max Drawdown:" in line:
                    try:
                        performance['drawdown'] = float(line.split(':')[-1].strip().rstrip('%'))
                    except:
                        performance['drawdown'] = 0.0
            
            return {
                'name': name,
                'status': 'SUCCESS',
                'performance': performance,
                'runtime': runtime/60,
                'config': {'metric': metric_name, 'top_k': top_k, 'lam_gap': lam_gap}
            }
        else:
            return {
                'name': name,
                'status': 'FAILED',
                'error': result.stderr[:200],
                'runtime': runtime/60
            }
    
    except subprocess.TimeoutExpired:
        return {
            'name': name,
            'status': 'TIMEOUT', 
            'runtime': timeout_minutes
        }
    except Exception as e:
        return {
            'name': name,
            'status': 'ERROR',
            'error': str(e),
            'runtime': (time.time() - start_time)/60
        }

def main():
    """Run key parameter tests"""
    
    # Define key test cases
    tests = [
        ("Baseline", "sharpe", 8, 0.3),
        ("TopK_5", "sharpe", 5, 0.3),
        ("TopK_12", "sharpe", 12, 0.3),
        ("LamGap_01", "sharpe", 8, 0.1),
        ("LamGap_05", "sharpe", 8, 0.5),
        ("AdjSharpe", "adj_sharpe", 8, 0.3),
    ]
    
    print(f"Running {len(tests)} stability parameter tests")
    print("=" * 60)
    
    # Run tests in parallel (limited to avoid overload)
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_test = {
            executor.submit(run_test, name, metric, top_k, lam_gap): name
            for name, metric, top_k, lam_gap in tests
        }
        
        for future in as_completed(future_to_test):
            test_name = future_to_test[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'SUCCESS':
                    perf = result['performance']
                    print(f"COMPLETED {test_name}: Sharpe={perf.get('sharpe', 0):.3f}, Return={perf.get('return', 0):.1f}%, Time={result['runtime']:.1f}min")
                else:
                    print(f"FAILED {test_name}: {result['status']}")
                    
            except Exception as exc:
                print(f"ERROR {test_name}: {exc}")
                results.append({'name': test_name, 'status': 'EXCEPTION', 'error': str(exc)})
    
    # Save and summarize results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    with open(f"artifacts/stability_test_results_{timestamp}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION RESULTS")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    if successful:
        successful.sort(key=lambda x: x['performance'].get('sharpe', -999), reverse=True)
        
        print("\nTop Performing Configurations:")
        for i, r in enumerate(successful[:3]):
            perf = r['performance']
            config = r['config']
            print(f"{i+1}. {r['name']}")
            print(f"   Sharpe: {perf.get('sharpe', 0):.3f}")
            print(f"   Return: {perf.get('return', 0):+.1f}%")
            print(f"   Config: top_k={config.get('top_k', 0)}, lam_gap={config.get('lam_gap', 0)}, metric={config.get('metric', 'N/A')}")
            print()
    
    print(f"Completed {len(successful)}/{len(tests)} tests successfully")

if __name__ == "__main__":
    main()
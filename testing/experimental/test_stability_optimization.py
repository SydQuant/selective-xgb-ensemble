#!/usr/bin/env python3
"""
Stability Ensemble Parameter Optimization Test Suite
Systematically tests different stability parameters to find optimal configuration
"""

import subprocess
import time
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

@dataclass
class StabilityTestConfig:
    """Configuration for a single stability test"""
    name: str
    metric_name: str
    top_k: int
    alpha: float
    lam_gap: float
    relative_gap: bool
    description: str

def create_test_config(base_config_path: str, test_config: StabilityTestConfig, output_path: str):
    """Create a test configuration file with specific stability parameters"""
    # Read base config
    with open(base_config_path, 'r') as f:
        content = f.read()
    
    # Replace stability parameters
    lines = content.split('\n')
    new_lines = []
    in_stability_section = False
    
    for line in lines:
        if line.strip().startswith('stability:'):
            in_stability_section = True
            new_lines.append(line)
        elif in_stability_section and line.startswith('  ') and ':' in line:
            param = line.strip().split(':')[0]
            if param == 'metric_name':
                new_lines.append(f'  metric_name: "{test_config.metric_name}"')
            elif param == 'top_k':
                new_lines.append(f'  top_k: {test_config.top_k}')
            elif param == 'alpha':
                new_lines.append(f'  alpha: {test_config.alpha}')
            elif param == 'lam_gap':
                new_lines.append(f'  lam_gap: {test_config.lam_gap}')
            elif param == 'relative_gap':
                new_lines.append(f'  relative_gap: {str(test_config.relative_gap).lower()}')
            else:
                new_lines.append(line)
        elif in_stability_section and not line.startswith('  '):
            in_stability_section = False
            new_lines.append(line)
        else:
            new_lines.append(line)
    
    # Write test config
    with open(output_path, 'w') as f:
        f.write('\n'.join(new_lines))

def run_stability_test(test_config: StabilityTestConfig, python_path: str = "~/anaconda3/python.exe") -> Dict:
    """Run a single stability test and return results"""
    print(f"Starting test: {test_config.name}")
    
    # Create test config file
    config_path = f"configs/test_{test_config.name.lower().replace(' ', '_')}.yaml"
    create_test_config("configs/production_full_system.yaml", test_config, config_path)
    
    start_time = time.time()
    
    try:
        # Run the test with reduced parameters for faster execution
        cmd = [
            python_path, "main.py",
            "--config", config_path,
            "--target_symbol", "@ES#C",
            "--bypass_pvalue_gating",
            "--n_models", "30",  # Reduced for faster testing
            "--folds", "4"       # Reduced for faster testing
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30min timeout
        
        runtime = time.time() - start_time
        
        if result.returncode == 0:
            # Parse performance from output
            output_lines = result.stdout.split('\n')
            performance = {}
            
            for line in output_lines:
                if "Total Return:" in line:
                    performance['total_return'] = float(line.split(':')[-1].strip().rstrip('%')) / 100
                elif "Sharpe Ratio:" in line:
                    performance['sharpe_ratio'] = float(line.split(':')[-1].strip())
                elif "Max Drawdown:" in line:
                    performance['max_drawdown'] = float(line.split(':')[-1].strip().rstrip('%')) / 100
                elif "Win Rate:" in line:
                    performance['win_rate'] = float(line.split(':')[-1].strip().rstrip('%')) / 100
                elif "p-value (DAPY):" in line:
                    p_val_str = line.split('(')[1].split(')')[0].split('=')[-1].strip()
                    performance['p_value'] = float(p_val_str)
                elif "mean stability =" in line:
                    stability_str = line.split('mean stability =')[-1].strip()
                    performance['mean_stability'] = float(stability_str)
            
            result_data = {
                'test_name': test_config.name,
                'description': test_config.description,
                'config': {
                    'metric_name': test_config.metric_name,
                    'top_k': test_config.top_k,
                    'alpha': test_config.alpha,
                    'lam_gap': test_config.lam_gap,
                    'relative_gap': test_config.relative_gap
                },
                'performance': performance,
                'runtime_minutes': runtime / 60,
                'status': 'SUCCESS'
            }
            
            print(f"✅ {test_config.name}: Sharpe={performance.get('sharpe_ratio', 'N/A'):.3f}, Return={performance.get('total_return', 0)*100:.1f}%")
            
        else:
            result_data = {
                'test_name': test_config.name,
                'description': test_config.description,
                'config': test_config.__dict__,
                'error': result.stderr,
                'runtime_minutes': runtime / 60,
                'status': 'FAILED'
            }
            print(f"❌ {test_config.name}: FAILED - {result.stderr[:100]}...")
    
    except subprocess.TimeoutExpired:
        result_data = {
            'test_name': test_config.name,
            'description': test_config.description,
            'config': test_config.__dict__,
            'error': 'TIMEOUT',
            'runtime_minutes': 30,
            'status': 'TIMEOUT'
        }
        print(f"⏰ {test_config.name}: TIMEOUT after 30 minutes")
    
    except Exception as e:
        result_data = {
            'test_name': test_config.name,
            'description': test_config.description, 
            'config': test_config.__dict__,
            'error': str(e),
            'runtime_minutes': (time.time() - start_time) / 60,
            'status': 'ERROR'
        }
        print(f"💥 {test_config.name}: ERROR - {str(e)}")
    
    # Clean up config file
    try:
        os.remove(config_path)
    except:
        pass
    
    return result_data

def main():
    """Run comprehensive stability parameter optimization tests"""
    
    # Define test configurations
    test_configs = [
        # Baseline (current production settings)
        StabilityTestConfig(
            name="Baseline_Current",
            metric_name="sharpe",
            top_k=8, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="Current production settings baseline"
        ),
        
        # Test different top_k values
        StabilityTestConfig(
            name="TopK_5_Conservative",
            metric_name="sharpe", 
            top_k=5, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="More selective with fewer drivers"
        ),
        StabilityTestConfig(
            name="TopK_12_Diverse",
            metric_name="sharpe",
            top_k=12, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="More diverse ensemble with more drivers"
        ),
        StabilityTestConfig(
            name="TopK_15_VeryDiverse",
            metric_name="sharpe",
            top_k=15, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="Maximum diversity with many drivers"
        ),
        
        # Test different lam_gap values (stability penalty)
        StabilityTestConfig(
            name="LamGap_01_LowPenalty",
            metric_name="sharpe",
            top_k=8, alpha=1.0, lam_gap=0.1, relative_gap=False,
            description="Lower stability penalty, more performance focus"
        ),
        StabilityTestConfig(
            name="LamGap_05_HighPenalty", 
            metric_name="sharpe",
            top_k=8, alpha=1.0, lam_gap=0.5, relative_gap=False,
            description="Higher stability penalty, more robustness focus"
        ),
        
        # Test different metrics
        StabilityTestConfig(
            name="AdjSharpe_Baseline",
            metric_name="adj_sharpe",
            top_k=8, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="Adjusted Sharpe with turnover penalty"
        ),
        StabilityTestConfig(
            name="HitRate_Baseline",
            metric_name="hit_rate", 
            top_k=8, alpha=1.0, lam_gap=0.3, relative_gap=False,
            description="Hit rate metric for directional accuracy"
        ),
        
        # Test relative gap
        StabilityTestConfig(
            name="RelativeGap_True",
            metric_name="sharpe",
            top_k=8, alpha=1.0, lam_gap=0.3, relative_gap=True,
            description="Relative gap penalty instead of absolute"
        ),
        
        # Test alpha variations
        StabilityTestConfig(
            name="Alpha_07_Balanced",
            metric_name="sharpe",
            top_k=8, alpha=0.7, lam_gap=0.3, relative_gap=False,
            description="Balanced train/validation weighting"
        ),
        
        # Best combinations from theory
        StabilityTestConfig(
            name="Optimized_LowGap_HighK",
            metric_name="adj_sharpe",
            top_k=12, alpha=1.0, lam_gap=0.1, relative_gap=False,
            description="Optimized: Low gap penalty + high diversity + adj_sharpe"
        ),
        StabilityTestConfig(
            name="Optimized_Conservative",
            metric_name="sharpe",
            top_k=5, alpha=0.7, lam_gap=0.5, relative_gap=True,
            description="Conservative: High stability focus + relative gap"
        ),
    ]
    
    print(f"Starting stability parameter optimization with {len(test_configs)} tests")
    print(f"Running tests in parallel with reduced parameters for speed")
    print("=" * 80)
    
    # Run tests in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all tests
        future_to_config = {
            executor.submit(run_stability_test, config): config 
            for config in test_configs
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"💥 {config.name} generated an exception: {exc}")
                results.append({
                    'test_name': config.name,
                    'status': 'EXCEPTION',
                    'error': str(exc),
                    'config': config.__dict__
                })
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"artifacts/stability_optimization_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    print("\n" + "=" * 80)
    print("📊 STABILITY PARAMETER OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r['status'] == 'SUCCESS']
    
    if successful_results:
        # Sort by Sharpe ratio
        successful_results.sort(key=lambda x: x['performance'].get('sharpe_ratio', -999), reverse=True)
        
        print("\n🏆 TOP PERFORMING CONFIGURATIONS:")
        for i, result in enumerate(successful_results[:5]):
            perf = result['performance']
            config = result['config']
            print(f"\n{i+1}. {result['test_name']}")
            print(f"   📈 Sharpe: {perf.get('sharpe_ratio', 'N/A'):.3f}")
            print(f"   💰 Return: {perf.get('total_return', 0)*100:+.1f}%")
            print(f"   📉 Drawdown: {perf.get('max_drawdown', 0)*100:.1f}%")
            print(f"   🎯 Win Rate: {perf.get('win_rate', 0)*100:.1f}%")
            print(f"   ⚡ Stability: {perf.get('mean_stability', 0):.3f}")
            print(f"   ⏱️  Runtime: {result['runtime_minutes']:.1f} min")
            print(f"   🔧 Config: top_k={config['top_k']}, lam_gap={config['lam_gap']}, metric={config['metric_name']}")
        
        print(f"\n📁 Full results saved to: {results_file}")
        
        # Create CSV summary
        summary_data = []
        for result in successful_results:
            if result['status'] == 'SUCCESS':
                perf = result['performance']
                config = result['config']
                summary_data.append({
                    'test_name': result['test_name'],
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'total_return': perf.get('total_return', 0) * 100,
                    'max_drawdown': perf.get('max_drawdown', 0) * 100,
                    'win_rate': perf.get('win_rate', 0) * 100,
                    'mean_stability': perf.get('mean_stability', 0),
                    'p_value': perf.get('p_value', 1.0),
                    'runtime_minutes': result['runtime_minutes'],
                    'top_k': config['top_k'],
                    'lam_gap': config['lam_gap'],
                    'metric_name': config['metric_name'],
                    'alpha': config['alpha'],
                    'relative_gap': config['relative_gap']
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            csv_file = f"artifacts/stability_optimization_summary_{timestamp}.csv"
            summary_df.to_csv(csv_file, index=False)
            print(f"📊 CSV summary saved to: {csv_file}")
    
    else:
        print("❌ No successful tests completed!")
    
    failed_count = len([r for r in results if r['status'] != 'SUCCESS'])
    if failed_count > 0:
        print(f"\n⚠️  {failed_count} tests failed or timed out")
    
    print(f"\n✅ Optimization completed! {len(successful_results)}/{len(test_configs)} tests successful")

if __name__ == "__main__":
    main()
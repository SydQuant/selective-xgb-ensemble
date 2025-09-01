#!/usr/bin/env python3
"""
Minimal test script for XGB ensemble with real data.
Tests both synthetic and real data loading with minimal parameters.
"""

import subprocess
import os
import sys

def run_command(cmd, description):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"CMD: {cmd}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode != 0:
            print(f"❌ FAILED with return code: {result.returncode}")
            return False
        else:
            print("✅ SUCCESS")
            return True
            
    except subprocess.TimeoutExpired:
        print("❌ TIMEOUT (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def main():
    """Run minimal tests"""
    print("🚀 Starting minimal XGB ensemble tests...")
    
    # Test 1: Synthetic data (minimal)
    synthetic_cmd = (
        "python main.py --synthetic "
        "--n_obs 200 --n_features 50 --folds 2 --n_models 4 --n_select 2 "
        "--weight_budget 10 --final_shuffles 50 --dapy_style hits"
    )
    
    success1 = run_command(synthetic_cmd, "Synthetic data test (minimal)")
    
    # Test 2: Real data (if Arctic DB is available)
    # Using minimal parameters for speed
    real_cmd = (
        "python main.py "
        "--target_symbol '@ES#C' --folds 2 --n_models 4 --n_select 2 "
        "--weight_budget 10 --final_shuffles 50 --max_features 30 "
        "--start_date 2023-01-01 --end_date 2023-06-30 --dapy_style hits"
    )
    
    success2 = run_command(real_cmd, "Real data test (minimal)")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)
    print(f"Synthetic data test: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Real data test: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    if success1 and success2:
        print("\n🎉 All tests passed! Ready for deployment.")
        return 0
    elif success1:
        print("\n⚠️  Synthetic test passed, real data test failed (check Arctic DB)")
        return 1
    else:
        print("\n💥 Tests failed - check implementation")
        return 1

if __name__ == "__main__":
    exit(main())
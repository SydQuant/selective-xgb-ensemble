#!/usr/bin/env python3
"""
Build Production Models for Specific Symbols
Builds production-ready models using optimal configurations from testing results.
"""

import sys
import os
from pathlib import Path

# Add xgb_compare directory to path
xgb_compare_dir = str(Path(__file__).parent.parent / "xgb_compare")
sys.path.insert(0, xgb_compare_dir)

def get_optimal_config(symbol: str) -> dict:
    """Get optimal configuration for a symbol from test results."""

    # Optimal configurations from comprehensive testing
    optimal_configs = {
        "@AD#C": {"models": 150, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 2.402},
        "@BO#C": {"models": 150, "folds": 15, "features": 250, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": -0.360},
        "@BP#C": {"models": 150, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 1.891},
        "@C#C": {"models": 150, "folds": 10, "features": 250, "q_metric": "hit_rate", "xgb_type": "tiered", "production_sharpe": 1.196},
        "@CT#C": {"models": 150, "folds": 20, "features": 250, "q_metric": "hit_rate", "xgb_type": "tiered", "production_sharpe": 0.847},
        "@ES#C": {"models": 150, "folds": 15, "features": 100, "q_metric": "hit_rate", "xgb_type": "standard", "production_sharpe": 1.975},
        "@EU#C": {"models": 200, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 1.537},
        "@FV#C": {"models": 150, "folds": 10, "features": 250, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 1.919},
        "@JY#C": {"models": 150, "folds": 20, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 2.560},
        "@KW#C": {"models": 150, "folds": 15, "features": 250, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.446},
        "@NQ#C": {"models": 100, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.851},
        "@RTY#C": {"models": 100, "folds": 10, "features": 100, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.480},
        "@S#C": {"models": 200, "folds": 15, "features": 250, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.803},
        "@SM#C": {"models": 150, "folds": 15, "features": 250, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.327},
        "@TY#C": {"models": 200, "folds": 10, "features": 250, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 2.239},
        "@US#C": {"models": 150, "folds": 10, "features": 250, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 1.820},
        "@W#C": {"models": 150, "folds": 15, "features": 250, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.799},
        "BD#C": {"models": 100, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 1.737},
        "BL#C": {"models": 150, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "tiered", "production_sharpe": 2.092},
        "QGC#C": {"models": 100, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 2.661},
        "QHG#C": {"models": 100, "folds": 15, "features": 100, "q_metric": "hit_rate", "xgb_type": "standard", "production_sharpe": 1.947},
        "QPL#C": {"models": 150, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.986},
        "QSI#C": {"models": 150, "folds": 15, "features": 100, "q_metric": "sharpe", "xgb_type": "standard", "production_sharpe": 1.759}
    }

    if symbol not in optimal_configs:
        raise ValueError(f"No optimal configuration found for {symbol}. Available symbols: {list(optimal_configs.keys())}")

    return optimal_configs[symbol]

def build_production_model(symbol: str, test_mode: bool = False):
    """Build production model for a specific symbol using optimal configuration."""

    print(f"\n{'='*60}")
    print(f"Building Production Model: {symbol}")
    print(f"{'='*60}")

    try:
        # Get optimal configuration
        optimal_config = get_optimal_config(symbol)
        print(f"Optimal Config: {optimal_config['models']}M, {optimal_config['folds']}F, {optimal_config['features']}feat, {optimal_config['q_metric']}, {optimal_config['xgb_type']}")
        print(f"Benchmark Production Sharpe: {optimal_config['production_sharpe']}")

        # For test mode, use smaller parameters
        if test_mode:
            print("\nTEST MODE: Using reduced parameters")
            n_models = min(5, optimal_config['models'])
            n_folds = min(2, optimal_config['folds'])
            max_features = min(50, optimal_config['features'])
        else:
            n_models = optimal_config['models']
            n_folds = optimal_config['folds']
            max_features = optimal_config['features']

        print(f"Building with: {n_models} models, {n_folds} folds, {max_features} features")

        # Use xgb_compare to build the models (simpler approach)
        os.chdir(xgb_compare_dir)

        import subprocess
        cmd = [
            sys.executable, "xgb_compare.py",
            "--target_symbol", symbol,
            "--n_models", str(n_models),
            "--n_folds", str(n_folds),
            "--max_features", str(max_features),
            "--q_metric", optimal_config['q_metric'],
            "--xgb_type", optimal_config['xgb_type'],
            "--export-production-models",  # Add export flag
            "--log_label", f"PROD_{'test_' if test_mode else ''}{symbol.replace('#', '').replace('@', '')}"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"SUCCESS: {symbol} production model built successfully")
            return True
        else:
            print(f"FAILED: {symbol} failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"ERROR: {symbol} error: {e}")
        return False

def main():
    """Main execution - build production models for specified symbols."""
    import argparse

    parser = argparse.ArgumentParser(description='Build production models for specific symbols')
    parser.add_argument('--symbols', nargs='+', required=True,
                       help='Symbols to build models for (e.g., @AD#C @ES#C QGC#C)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: use smaller parameters for quick validation')

    args = parser.parse_args()

    print("Production Model Builder")
    print("=" * 60)
    print(f"Symbols to build: {args.symbols}")
    if args.test:
        print("TEST MODE: Using reduced parameters")

    # Build models for each symbol
    results = {}
    for symbol in args.symbols:
        success = build_production_model(symbol, test_mode=args.test)
        results[symbol] = success

    # Summary
    print(f"\n{'='*60}")
    print("PRODUCTION MODEL BUILD SUMMARY")
    print(f"{'='*60}")

    for symbol, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{symbol}: {status}")

    successful = sum(results.values())
    total = len(results)
    print(f"\nOverall: {successful}/{total} models built successfully")

if __name__ == "__main__":
    main()
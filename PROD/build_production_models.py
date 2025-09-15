#!/usr/bin/env python3
"""
Build Production Models for Essential Symbols
Builds top-performing models for production deployment.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "xgb_compare"))

def build_tier1_models():
    """Build models for Tier 1 symbols (>2.0 Sharpe)."""
    tier1_symbols = [
        "QGC#C",    # 2.837 - Gold
        "@NQ#C",    # 2.385 - NASDAQ
        "@ES#C",    # 2.319 - S&P500
        "BL#C",     # 2.290 - EU Bund
        "@RTY#C",   # 2.193 - Russell
        "@TY#C",    # 2.067 - US 10Y
        "BD#C",     # 2.050 - EU Bund
        "QPL#C"     # 2.047 - Platinum
    ]

    print("Building Tier 1 Production Models (>2.0 Sharpe)")
    print("=" * 55)

    for i, symbol in enumerate(tier1_symbols, 1):
        print(f"\n{i}/8: Building {symbol}...")

        # Import here to avoid path issues
        from production_model_builder import ProductionModelBuilder

        builder = ProductionModelBuilder()
        success = builder.build_production_models(symbol)

        if success:
            print(f"✅ {symbol} completed")
        else:
            print(f"❌ {symbol} failed")

def build_tier2_models():
    """Build models for Tier 2 symbols (1.5-2.0 Sharpe)."""
    tier2_symbols = [
        "@AD#C",    # 1.990 - Australian Dollar
        "@S#C",     # 1.985 - Soybeans
        "@KW#C",    # 1.981 - Wheat
        "@W#C",     # 1.837 - Wheat
        "@US#C",    # 1.837 - US 30Y
        "@EU#C",    # 1.769 - Euro
        "@FV#C"     # 1.721 - US 5Y
    ]

    print("\nBuilding Tier 2 Production Models (1.5-2.0 Sharpe)")
    print("=" * 55)

    for i, symbol in enumerate(tier2_symbols, 1):
        print(f"\n{i}/7: Building {symbol}...")

        from production_model_builder import ProductionModelBuilder

        builder = ProductionModelBuilder()
        success = builder.build_production_models(symbol)

        if success:
            print(f"✅ {symbol} completed")
        else:
            print(f"❌ {symbol} failed")

def main():
    """Main execution - build essential production models."""
    import argparse

    parser = argparse.ArgumentParser(description='Build production models')
    parser.add_argument('--tier', choices=['1', '2', 'all'], default='1',
                       help='Which tier to build (1=excellent, 2=good, all=both)')

    args = parser.parse_args()

    if args.tier in ['1', 'all']:
        build_tier1_models()

    if args.tier in ['2', 'all']:
        build_tier2_models()

    print("\nProduction model building completed!")
    print("Models saved to PROD/models/ with top performers only")

if __name__ == "__main__":
    main()
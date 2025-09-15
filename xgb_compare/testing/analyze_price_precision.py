"""
Analyze price precision in JY and TY raw data exports.
"""

import pandas as pd
import os
from decimal import Decimal

def analyze_precision(file_path, symbol_name):
    """Analyze the precision of price data in CSV file."""
    print(f"\n{'='*60}")
    print(f"PRECISION ANALYSIS: {symbol_name}")
    print(f"{'='*60}")

    df = pd.read_csv(file_path)

    # Analyze each price column
    price_cols = ['open', 'high', 'low', 'close']

    for col in price_cols:
        print(f"\n{col.upper()} COLUMN:")
        print("-" * 20)

        # Convert to string to analyze decimal places
        values = df[col].astype(str)

        # Count decimal places
        decimal_places = []
        for val in values.head(100):  # Sample first 100 values
            if '.' in val:
                decimal_places.append(len(val.split('.')[1]))
            else:
                decimal_places.append(0)

        # Statistics
        print(f"Sample size: {len(decimal_places)}")
        print(f"Min decimal places: {min(decimal_places)}")
        print(f"Max decimal places: {max(decimal_places)}")
        print(f"Most common decimal places: {max(set(decimal_places), key=decimal_places.count)}")

        # Show some example values
        print(f"Example raw values:")
        for i in range(min(10, len(values))):
            print(f"  {values.iloc[i]}")

        # Check for precision patterns
        unique_decimals = set(decimal_places)
        print(f"All decimal place counts: {sorted(unique_decimals)}")

        # Sample some exact values to see precision
        print(f"\nExact values (first 5):")
        for i in range(min(5, len(df))):
            exact_val = Decimal(str(df[col].iloc[i]))
            print(f"  {exact_val}")

def main():
    """Main analysis function."""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Find the exported files
    jy_files = [f for f in os.listdir(current_dir) if f.startswith('JY_raw_data_') and f.endswith('.csv')]
    ty_files = [f for f in os.listdir(current_dir) if f.startswith('TY_raw_data_') and f.endswith('.csv')]

    if jy_files:
        jy_file = os.path.join(current_dir, jy_files[-1])  # Latest file
        print(f"Analyzing JY file: {jy_files[-1]}")
        analyze_precision(jy_file, "@JY#C (Japanese Yen)")

    if ty_files:
        ty_file = os.path.join(current_dir, ty_files[-1])  # Latest file
        print(f"Analyzing TY file: {ty_files[-1]}")
        analyze_precision(ty_file, "@TY#C (US 10-Year Treasury)")

    print(f"\n{'='*60}")
    print("SUMMARY COMPARISON")
    print(f"{'='*60}")

    if jy_files and ty_files:
        jy_df = pd.read_csv(os.path.join(current_dir, jy_files[-1]))
        ty_df = pd.read_csv(os.path.join(current_dir, ty_files[-1]))

        print(f"\nJY#C Price Range: {jy_df['close'].min():.8f} to {jy_df['close'].max():.8f}")
        print(f"TY#C Price Range: {ty_df['close'].min():.8f} to {ty_df['close'].max():.8f}")

        # Calculate tick sizes
        jy_diffs = jy_df['close'].diff().dropna()
        ty_diffs = ty_df['close'].diff().dropna()

        jy_nonzero_diffs = jy_diffs[jy_diffs != 0]
        ty_nonzero_diffs = ty_diffs[ty_diffs != 0]

        if len(jy_nonzero_diffs) > 0:
            min_jy_tick = abs(jy_nonzero_diffs).min()
            print(f"\nJY#C Minimum tick size: {min_jy_tick:.8f}")

        if len(ty_nonzero_diffs) > 0:
            min_ty_tick = abs(ty_nonzero_diffs).min()
            print(f"TY#C Minimum tick size: {min_ty_tick:.8f}")

if __name__ == "__main__":
    main()
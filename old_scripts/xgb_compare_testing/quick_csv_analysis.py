#!/usr/bin/env python3
"""Quick analysis of CSV outputs from underperforming symbols"""

import pandas as pd
import numpy as np

def analyze_csv(symbol, csv_path):
    """Analyze a single CSV file"""
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol}")
    print(f"{'='*60}")

    try:
        df = pd.read_csv(csv_path)
        print(f"Data shape: {df.shape}")
        print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

        # Signal analysis
        signal_dist = df['signal_direction'].value_counts()
        print(f"Signal distribution: {signal_dist.to_dict()}")

        # Hit rate analysis
        hit_rate = df['hit'].mean()
        print(f"Overall hit rate: {hit_rate:.3f}")

        # PnL analysis
        total_pnl = df['pnl'].sum()
        mean_pnl = df['pnl'].mean()
        print(f"Total PnL: {total_pnl:.6f}")
        print(f"Mean PnL per trade: {mean_pnl:.6f}")

        # Return analysis
        ret_mean = df['target_return'].mean()
        ret_std = df['target_return'].std()
        print(f"Target return: mean={ret_mean:.6f}, std={ret_std:.6f}")

        # Signal vs return correlation
        signal_corr = df['signal'].corr(df['target_return'])
        print(f"Signal-return correlation: {signal_corr:.6f}")

        # Period analysis
        if 'period' in df.columns:
            period_stats = df.groupby('period').agg({
                'hit': 'mean',
                'pnl': 'sum',
                'target_return': 'mean'
            })
            print(f"Performance by period:")
            for period, stats in period_stats.iterrows():
                print(f"  {period}: hit={stats['hit']:.3f}, pnl={stats['pnl']:.6f}")

        # Check for extreme values
        extreme_returns = (df['target_return'].abs() > df['target_return'].std() * 3).sum()
        extreme_pnl = (df['pnl'].abs() > df['pnl'].std() * 3).sum()
        print(f"Extreme values: {extreme_returns} returns, {extreme_pnl} PnL")

        # Signal consistency
        wrong_direction = ((df['signal_direction'] > 0) & (df['target_return'] < 0)) | \
                         ((df['signal_direction'] < 0) & (df['target_return'] > 0))
        wrong_pct = wrong_direction.mean()
        print(f"Wrong direction signals: {wrong_pct:.3f}")

        return {
            'symbol': symbol,
            'hit_rate': hit_rate,
            'total_pnl': total_pnl,
            'signal_corr': signal_corr,
            'wrong_direction_pct': wrong_pct,
            'extreme_returns': extreme_returns,
            'signal_balance': signal_dist.to_dict()
        }

    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return {'symbol': symbol, 'error': str(e)}

def main():
    """Analyze all CSV files"""
    csv_files = {
        '@BO#C': 'results/logs- rerun/20250917_155254_signal_distribution_rerun_BO_multiprocessing.csv',
        'QCL#C': 'results/logs- rerun/20250917_180458_signal_distribution_rerun_QCL_multiprocessing.csv',
        'QRB#C': 'results/logs- rerun/20250917_180545_signal_distribution_rerun_QRB_multiprocessing.csv'
    }

    results = []
    for symbol, csv_path in csv_files.items():
        result = analyze_csv(symbol, csv_path)
        results.append(result)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")

    for result in results:
        if 'error' not in result:
            print(f"\n{result['symbol']}:")
            print(f"  Hit rate: {result['hit_rate']:.3f}")
            print(f"  Total PnL: {result['total_pnl']:.6f}")
            print(f"  Signal correlation: {result['signal_corr']:.6f}")
            print(f"  Wrong direction: {result['wrong_direction_pct']:.3f}")
            print(f"  Signal balance: {result['signal_balance']}")
        else:
            print(f"\n{result['symbol']}: ERROR - {result['error']}")

    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")

    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        hit_rates = [r['hit_rate'] for r in successful_results]
        correlations = [r['signal_corr'] for r in successful_results]
        wrong_dirs = [r['wrong_direction_pct'] for r in successful_results]

        print(f"Hit rate range: {min(hit_rates):.3f} - {max(hit_rates):.3f}")
        print(f"Signal correlation range: {min(correlations):.6f} - {max(correlations):.6f}")
        print(f"Wrong direction range: {min(wrong_dirs):.3f} - {max(wrong_dirs):.3f}")

        # Identify the worst performer
        worst_hit = min(successful_results, key=lambda x: x['hit_rate'])
        worst_corr = min(successful_results, key=lambda x: x['signal_corr'])

        print(f"\nWorst hit rate: {worst_hit['symbol']} ({worst_hit['hit_rate']:.3f})")
        print(f"Worst correlation: {worst_corr['symbol']} ({worst_corr['signal_corr']:.6f})")

if __name__ == "__main__":
    main()
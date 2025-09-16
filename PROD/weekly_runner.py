#!/usr/bin/env python3
"""
Simplified Weekly Analysis for XGBoost Production
Core reconciliation and performance tracking.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import TRADING_SYMBOLS
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine

def get_live_signals(logs_dir: Path, days: int = 7):
    """Extract signals from recent trade files."""
    signals = []

    for i in range(days):
        date = datetime.now() - timedelta(days=i)
        date_str = date.strftime('%Y%m%d')
        date_dir = logs_dir / date_str

        if not date_dir.exists():
            continue

        for trade_file in date_dir.glob("*_GMS_Trade_File_*.xlsx"):
            try:
                df = pd.read_excel(trade_file)
                for _, row in df.iterrows():
                    qty = float(row['QUANTITY'])
                    signal = 0 if qty == 0 else (1 if row['SIDE'] == 'BUY' else -1)
                    signals.append({
                        'date': date.date(),
                        'symbol': row['SECURITY_ID'],
                        'signal': signal
                    })
            except Exception:
                continue

    return signals

def generate_backtest_signals(symbol: str, days: int = 7):
    """Generate backtest signals using database data."""
    data_engine = DataEngine()
    signal_engine = SignalEngine(Path(__file__).parent / "models", Path(__file__).parent / "config")

    # Database mock
    class DatabaseClient:
        def __init__(self):
            sys.path.append(str(Path(__file__).parent.parent))
            from data.loaders import get_arcticdb_connection
            self.lib = get_arcticdb_connection()

        def get_live_data_multi(self, symbols, days=20, interval_min=60):
            result = {}
            for sym in symbols:
                data = self.lib.read(sym).data
                result[sym] = data.tail(days * 24) if len(data) > days * 24 else data
            return result

    data_engine.iqfeed_client = DatabaseClient()

    signals = []
    for i in range(min(days, 3)):  # Limit to 3 days for speed
        features_df, _ = data_engine.get_prediction_data(symbol, [symbol, "@TY#C", "QGC#C"], 12)
        if features_df is not None:
            result = signal_engine.generate_signal(features_df, symbol)
            if result:
                signal, _ = result
                date = datetime.now() - timedelta(days=i)
                signals.append({'date': date.date(), 'symbol': symbol, 'signal': signal})

    return signals

def create_performance_plot(symbols, weeks: int = 12):
    """Create performance plot using PROD signal generation and simple backtest."""
    # Initialize PROD components
    data_engine = DataEngine()
    signal_engine = SignalEngine(Path(__file__).parent / "models", Path(__file__).parent / "config")

    # Database mock for data engine
    class DatabaseClient:
        def __init__(self):
            sys.path.append(str(Path(__file__).parent.parent))
            from data.loaders import get_arcticdb_connection
            self.lib = get_arcticdb_connection()

        def get_live_data_multi(self, symbols, days=20, interval_min=60):
            result = {}
            for sym in symbols:
                data = self.lib.read(sym).data
                result[sym] = data.tail(days * 24) if len(data) > days * 24 else data
            return result

    data_engine.iqfeed_client = DatabaseClient()

    n_symbols = len(symbols)
    cols = 4
    rows = (n_symbols + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    fig.suptitle(f'XGBoost Production Backtest ({weeks} weeks)', fontsize=16)

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()

    for i, symbol in enumerate(symbols):
        ax = axes[i]

        # Get database data
        lib = data_engine.iqfeed_client.lib
        data = lib.read(symbol).data
        signal_hour_data = data[data.index.hour == 12]

        # Get period data
        end_date = signal_hour_data.index[-1]
        start_date = end_date - timedelta(weeks=weeks)
        period_data = signal_hour_data[signal_hour_data.index >= start_date]

        # Simple backtest: generate signals and calculate PnL
        daily_pnl = []

        for j in range(1, len(period_data)):  # Start from day 1 (need previous day for return)
            current_date = period_data.index[j]

            # Generate signal using PROD components
            features_df, price = data_engine.get_prediction_data(symbol, [symbol] + TRADING_SYMBOLS[:3], 12)

            if features_df is not None:
                result = signal_engine.generate_signal(features_df, symbol)
                if result:
                    signal, raw_score = result

                    # Calculate target return (current day's return)
                    prev_price = period_data.iloc[j-1]['close']
                    curr_price = period_data.iloc[j]['close']
                    target_return = (curr_price - prev_price) / prev_price

                    # Simple backtest: signal × target return
                    pnl = signal * target_return
                    daily_pnl.append(pnl)

        # Plot cumulative PnL
        if daily_pnl:
            cumulative_pnl = pd.Series(daily_pnl).cumsum() * 100
            backtest_dates = period_data.index[1:len(daily_pnl)+1]

            ax.plot(backtest_dates, cumulative_pnl.values, linewidth=1.5)
            ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)
            ax.set_title(f'{symbol}', fontsize=10)
            ax.set_ylabel('PnL (%)', fontsize=8)

            print(f"  {symbol}: {cumulative_pnl.iloc[-1]:.2f}% PnL, {len(daily_pnl)} signals")
        else:
            ax.text(0.5, 0.5, f'{symbol}\nNo Signals', ha='center', va='center', transform=ax.transAxes)
            ax.axhline(y=0, color='red', linestyle='-', alpha=0.5)

        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)

    # Hide empty subplots
    for i in range(n_symbols, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    output_dir = Path(__file__).parent / "weekly_analysis"
    output_dir.mkdir(exist_ok=True)

    plot_file = output_dir / f"production_backtest_{weeks}w_{datetime.now().strftime('%Y%m%d')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_file

def main():
    """Main weekly analysis."""
    print("XGBoost Weekly Analysis")
    print("=" * 30)

    logs_dir = Path(__file__).parent / "logs"

    # Signal reconciliation
    print("1. Signal Reconciliation...")
    live_signals = get_live_signals(logs_dir)
    print(f"   Found {len(live_signals)} live signals")

    backtest_signals = []
    for symbol in TRADING_SYMBOLS:  # Limit for speed
        bt_signals = generate_backtest_signals(symbol)
        backtest_signals.extend(bt_signals)

    print(f"   Generated {len(backtest_signals)} backtest signals")

    # Performance analysis
    print("2. Performance Analysis...")
    for weeks in [12, 52]:
        plot_file = create_performance_plot(TRADING_SYMBOLS[:4], weeks)
        print(f"   {weeks}-week plot: {plot_file}")

    print("Weekly analysis completed")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Weekly XGBoost Backtest and PnL Analysis
Follows same logic as daily_signal_runner.py but for longer timeframes.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))  # Add parent for research imports

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, SIGNAL_HOUR
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine
from common.trades_util import TradeProcessor

def setup_logging():
    """Setup simple logging."""
    weekly_dir = Path(__file__).parent / "weekly_analysis"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    log_file = weekly_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}_xgb_weekly.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    return logging.getLogger(__name__)

def get_live_signals(logs_dir: Path, days: int = 7):
    """Extract signals from recent trade files (keep for future reconciliation)."""
    # Keep this function for future signal reconciliation but don't use it for now
    return []

def run_backtest(symbol: str, days: int = 30) -> pd.DataFrame:
    """Run backtest using research data preparation function for perfect alignment"""

    # Use research function to get features and target returns
    from data.data_utils_simple import prepare_real_data_simple

    signal_engine = SignalEngine(
        Path(__file__).parent / "models",
        Path(__file__).parent / "config"
    )

    # Get backtest date range
    end_date = datetime(2025, 6, 30)  # Use known good data end date

    start_date = end_date - timedelta(days=days + 30)  # 30-day buffer for feature calculation

    # Use research function to prepare data with target returns
    full_df = prepare_real_data_simple(
        symbol,
        symbols=FEATURE_SYMBOLS,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        signal_hour=SIGNAL_HOUR
    )

    if full_df.empty:
        return pd.DataFrame()

    # Get target column and features
    target_col = f"{symbol}_target_return"
    feature_cols = [col for col in full_df.columns if col != target_col]

    # Get last 'days' worth of data
    recent_data = full_df.tail(days)

    results = []
    for idx, row in recent_data.iterrows():
        features_row = row[feature_cols].to_frame().T
        target_return = row[target_col]

        # Apply feature selection to match production models (CRITICAL FIX)
        # Load the production package to get expected features
        try:
            import pickle
            package_file = Path(__file__).parent / "models" / f"{symbol}_production.pkl"
            if package_file.exists():
                with open(package_file, 'rb') as f:
                    package = pickle.load(f)
                expected_features = package['selected_features']

                # Select only the expected features in the exact order
                features_row = features_row[expected_features]
        except Exception as e:
            logger.warning(f"Feature selection failed for {symbol}: {e}")

        # Generate signal using the exact same logic as daily runner
        result = signal_engine.generate_signal(features_row, symbol)
        if result:
            signal, raw_score = result

            results.append({
                'date': idx.date(),
                'symbol': symbol,
                'signal': signal,
                'raw_score': raw_score,
                'target_return': target_return,
                'daily_pnl_pct': signal * target_return * 100  # Convert to percentage
            })

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
        df['cumulative_pnl_pct'] = df['daily_pnl_pct'].cumsum()

    return df

def create_performance_plot(backtest_results: dict, days: int = 30):
    """Create performance plot from actual backtest results."""
    symbols = list(backtest_results.keys())

    # Calculate optimal grid size based on number of symbols
    n_symbols = len(symbols)
    if n_symbols <= 4:
        rows, cols = 2, 2
    elif n_symbols <= 6:
        rows, cols = 2, 3
    elif n_symbols <= 9:
        rows, cols = 3, 3
    elif n_symbols <= 12:
        rows, cols = 3, 4
    elif n_symbols <= 16:
        rows, cols = 4, 4
    else:
        rows, cols = 5, 5  # Max 25 symbols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'XGBoost Production Backtest ({days} days)')

    # Flatten axes for easier indexing
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for i, symbol in enumerate(symbols):
        ax = axes[i]
        df = backtest_results[symbol]

        if not df.empty:
            # Plot cumulative PnL percentage
            ax.plot(pd.to_datetime(df['date']), df['cumulative_pnl_pct'],
                   label=f'{symbol} PnL%', linewidth=2)

            # Add zero line
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            # Stats for title
            total_pnl_pct = df['cumulative_pnl_pct'].iloc[-1] if not df.empty else 0
            n_trades = len(df[df['signal'] != 0]) if not df.empty else 0

            ax.set_title(f'{symbol}: PnL={total_pnl_pct:.2f}%, Trades={n_trades}')
            ax.set_ylabel('Cumulative PnL (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()

            # Format dates
            ax.tick_params(axis='x', rotation=45)
            if days > 30:
                import matplotlib.dates as mdates
                if days > 90:
                    # For 6+ months, show monthly ticks
                    ax.xaxis.set_major_locator(mdates.MonthLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                else:
                    # For 1-3 months, show weekly ticks
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        else:
            ax.text(0.5, 0.5, f'No data for {symbol}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{symbol}: No Data')

    # Hide unused subplots
    for j in range(len(symbols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    output_dir = Path(__file__).parent / "weekly_analysis"
    output_dir.mkdir(exist_ok=True)

    plot_file = output_dir / f"backtest_{days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_file

def main():
    """Main weekly backtest analysis."""
    parser = argparse.ArgumentParser(description='XGBoost Weekly Backtest Runner')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test (default: all trading symbols)')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== XGBoost Weekly Backtest Analysis ===")

    # Use specified symbols or auto-detect all available models
    if args.symbols:
        symbols_to_test = args.symbols
    else:
        # Auto-detect all symbols with production models
        models_dir = Path(__file__).parent / "models"
        available_symbols = []
        for model_file in models_dir.glob("*_production.pkl"):
            symbol = model_file.stem.replace("_production", "")
            available_symbols.append(symbol)
        symbols_to_test = sorted(available_symbols)
        logger.info(f"Auto-detected {len(symbols_to_test)} symbols with models")

    logger.info(f"Running {args.days}-day backtest for {len(symbols_to_test)} symbols...")

    # Run backtests for each symbol
    backtest_results = {}
    for symbol in symbols_to_test:
        logger.info(f"Running backtest for {symbol}...")
        df = run_backtest(symbol, args.days)
        backtest_results[symbol] = df

        if not df.empty:
            total_pnl_pct = df['cumulative_pnl_pct'].iloc[-1]
            n_trades = len(df[df['signal'] != 0])
            logger.info(f"  {symbol}: Total PnL={total_pnl_pct:.2f}%, Trades={n_trades}")
        else:
            logger.info(f"  {symbol}: No data available")

    # Create performance plot
    logger.info("Creating performance plot...")
    plot_file = create_performance_plot(backtest_results, args.days)
    logger.info(f"Plot saved: {plot_file}")

    # Summary statistics
    logger.info("\n=== Summary ===")
    total_pnl = 0
    total_trades = 0
    successful_symbols = 0

    for symbol, df in backtest_results.items():
        if not df.empty:
            pnl_pct = df['cumulative_pnl_pct'].iloc[-1]
            trades = len(df[df['signal'] != 0])
            total_pnl += pnl_pct
            total_trades += trades
            successful_symbols += 1

    logger.info(f"Symbols with data: {successful_symbols}/{len(symbols_to_test)}")
    logger.info(f"Total PnL: {total_pnl:.2f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info("Weekly backtest analysis completed")

    return True

if __name__ == "__main__":
    main()
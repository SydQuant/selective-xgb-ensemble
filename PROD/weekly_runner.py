#!/usr/bin/env python3
"""
Weekly XGBoost Backtest and PnL Analysis with Signal Reconciliation
- Runs backtests using the same models and data as daily production system
- Compares backtest signals against live trade files for reconciliation
- Generates performance plots and detailed reconciliation reports
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

def create_bloomberg_to_standard_mapping():
    """Create mapping from Bloomberg symbols to standardized symbols with contract rollover support."""
    from config import trading_config

    bloomberg_to_standard = {}

    # Add exact mappings from config
    for std_symbol, config in trading_config['instrument_config'].items():
        bloomberg_symbol = config.get('bloomberg', '')
        if bloomberg_symbol:
            bloomberg_to_standard[bloomberg_symbol] = std_symbol

    # Add common contract rollover mappings (U5->Z5, etc.)
    rollover_mapping = {
        # September (U5) to December (Z5) rollover
        'ADZ5 Curncy': '@AD#C',    # was ADU5
        'ESZ5 Index': '@ES#C',     # was ESU5
        'JYZ5 Curncy': '@JY#C',    # was JYU5
        'BPZ5 Curncy': '@BP#C',    # was BPU5
        'NQZ5 Index': '@NQ#C',     # was NQU5
        'RTYZ5 Index': '@RTY#C',   # was RTYU5
        'RXZ5 Comdty': 'BD#C',     # was RXU5
        'ECZ5 Curncy': '@EU#C',    # was ECU5
        'CLZ5 Comdty': 'QCL#C',    # was CLV5
        'NGX25 Comdty': 'QNG#C',   # was NGV25 (contract month changed)

        # Additional contract variations seen in trade files
        'TYU5 Comdty': '@TY#C',    # TY has TYZ5 in config but TYU5 in Sept 16 trades
        'HGU5 Comdty': 'QHG#C',    # HG has HGZ5 in config but HGU5 in Sept 16
        'GCQ5 Comdty': 'QGC#C',    # GC has GCZ5 in config but GCQ5 in Sept 16
        'OZU5 Comdty': 'BL#C',     # was OEU5, now OZU5
        'OEZ5 Comdty': 'BL#C',     # OE rollover to OEZ5
    }

    # Add rollover mappings
    bloomberg_to_standard.update(rollover_mapping)

    return bloomberg_to_standard

def get_live_signals(logs_dir: Path, days: int = 7):
    """Extract signals from recent trade files for reconciliation."""
    live_signals = []

    # Create Bloomberg to standardized symbol mapping
    bloomberg_mapping = create_bloomberg_to_standard_mapping()
    logger.info(f"Created Bloomberg mapping for {len(bloomberg_mapping)} symbols")

    # Get all date directories in logs
    date_dirs = [d for d in logs_dir.glob('*') if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
    date_dirs = sorted(date_dirs, key=lambda x: x.name)[-days:]  # Last N days

    for date_dir in date_dirs:
        date_str = date_dir.name
        date = datetime.strptime(date_str, '%Y%m%d').date()

        # Look for trade files
        excel_files = list(date_dir.glob('*GMS_Trade_File*.xlsx'))
        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file)
                if df.empty:
                    continue

                for _, row in df.iterrows():
                    bloomberg_symbol = str(row['SECURITY_ID']).strip()
                    quantity = float(row.get('QUANTITY', 0))
                    side = str(row.get('SIDE', '')).strip().upper()

                    # Convert Bloomberg symbol to standardized symbol
                    standard_symbol = bloomberg_mapping.get(bloomberg_symbol, bloomberg_symbol)

                    # Skip if we can't map the symbol
                    if standard_symbol == bloomberg_symbol and bloomberg_symbol not in bloomberg_mapping:
                        logger.debug(f"Unknown Bloomberg symbol: {bloomberg_symbol}")
                        continue

                    # Convert to signal: 0=no trade, 1=buy, -1=sell
                    if quantity == 0:
                        signal = 0
                    elif side == 'BUY':
                        signal = 1
                    elif side == 'SELL':
                        signal = -1
                    else:
                        signal = 0

                    live_signals.append({
                        'date': date,
                        'symbol': standard_symbol,  # Use standardized symbol
                        'bloomberg_symbol': bloomberg_symbol,  # Keep original for reference
                        'signal': signal,
                        'quantity': quantity,
                        'side': side,
                        'source': 'live_trade_file'
                    })

            except Exception as e:
                logger.warning(f"Failed to read {excel_file}: {e}")
                continue

    return live_signals

def run_backtest(symbol: str, days: int = 30) -> pd.DataFrame:
    """Run backtest using database only (research mode - no live IQFeed connection)"""

    signal_engine = SignalEngine(
        Path(__file__).parent / "models",
        Path(__file__).parent / "config"
    )

    # Research mode: Use database data only (no IQFeed)
    logger.info(f"Running research backtest for {symbol} using database...")

    try:
        from data.data_utils_simple import prepare_real_data_simple

        # Get backtest date range to match available live trade files
        # Find the latest date from live trade files
        logs_dir = Path(__file__).parent / "logs"
        live_dates = []
        if logs_dir.exists():
            date_dirs = [d for d in logs_dir.glob('*') if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
            live_dates = [datetime.strptime(d.name, '%Y%m%d').date() for d in date_dirs]

        if live_dates:
            end_date = datetime.combine(max(live_dates), datetime.min.time())
            logger.info(f"Using end date {end_date.date()} to match live trade files")
        else:
            end_date = datetime(2025, 9, 19)  # Default to latest known live data
            logger.warning("No live trade dates found, using default end date")

        start_date = end_date - timedelta(days=days + 30)  # 30-day buffer for feature calculation

        # Use research function to prepare data with target returns
        # For reconciliation, keep rows without target returns to get Sept 18-19 data
        full_df = prepare_real_data_simple(
            symbol,
            symbols=FEATURE_SYMBOLS,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            signal_hour=SIGNAL_HOUR,
            keep_rows_without_targets=True  # Keep Sept 18-19 for reconciliation
        )

        if full_df.empty:
            logger.warning(f"No data available for {symbol} in database")
            return pd.DataFrame()

        # Get target column and features
        target_col = f"{symbol}_target_return"
        if target_col not in full_df.columns:
            logger.error(f"No target column {target_col} found for {symbol}")
            return pd.DataFrame()

        # Apply feature selection to match production models
        import pickle
        models_dir = Path(__file__).parent / "models"
        timestamp_files = list(models_dir.glob(f"{symbol}_*.pkl"))
        if not timestamp_files:
            logger.error(f"No model package found for {symbol}")
            return pd.DataFrame()

        package_file = sorted(timestamp_files)[-1]
        with open(package_file, 'rb') as f:
            package = pickle.load(f)
        expected_features = package['selected_features']

        # Build feature matrix in correct order
        features_aligned = full_df[expected_features].tail(days)
        returns_aligned = full_df[target_col].tail(days)
        common_index = features_aligned.index.intersection(returns_aligned.index)

        if common_index.empty:
            logger.warning(f"No common index after alignment for {symbol}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Database backtest failed for {symbol}: {e}")
        return pd.DataFrame()

    # Get recent period
    recent_idx = common_index[-days:] if len(common_index) >= days else common_index

    results = []
    for idx in recent_idx:
        features_row = features_aligned.loc[[idx]]  # Keep as DataFrame
        target_return = returns_aligned.loc[idx] if not pd.isna(returns_aligned.loc[idx]) else 0.0

        # Generate signal using exact production logic
        result = signal_engine.generate_signal(features_row, symbol)
        if result:
            signal, raw_score = result

            # For reconciliation, we don't need accurate PnL, just signal comparison
            daily_pnl_pct = signal * target_return * 100 if not pd.isna(returns_aligned.loc[idx]) else 0.0

            results.append({
                'date': idx.date(),
                'symbol': symbol,
                'signal': signal,
                'raw_score': raw_score,
                'target_return': target_return,
                'daily_pnl_pct': daily_pnl_pct
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
    elif n_symbols <= 25:
        rows, cols = 5, 5
    elif n_symbols <= 30:
        rows, cols = 5, 6
    else:
        rows, cols = 6, 6  # Max 36 symbols

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

def compare_signals(backtest_results: dict, live_signals: list, days: int = 7) -> pd.DataFrame:
    """Compare backtest signals with live trade signals."""
    comparison_data = []

    # Convert live signals to DataFrame for easier processing
    live_df = pd.DataFrame(live_signals)
    if live_df.empty:
        logger.warning("No live signals found for comparison")
        return pd.DataFrame()

    # Group live signals by date and symbol
    live_grouped = live_df.groupby(['date', 'symbol']).agg({
        'signal': 'first',  # Take first signal if multiple
        'quantity': 'first',
        'side': 'first'
    }).reset_index()

    # Compare each symbol's backtest vs live signals
    for symbol, backtest_df in backtest_results.items():
        if backtest_df.empty:
            continue

        # Get recent backtest signals (last N days)
        recent_backtest = backtest_df.tail(days).copy()

        for _, bt_row in recent_backtest.iterrows():
            bt_date = bt_row['date']
            bt_signal = bt_row['signal']
            bt_score = bt_row['raw_score']

            # Find corresponding live signal
            live_match = live_grouped[
                (live_grouped['date'] == bt_date) &
                (live_grouped['symbol'] == symbol)
            ]

            if len(live_match) > 0:
                live_signal = live_match.iloc[0]['signal']
                live_quantity = live_match.iloc[0]['quantity']
                live_side = live_match.iloc[0]['side']
            else:
                live_signal = 0  # No trade = 0 signal
                live_quantity = 0
                live_side = 'NO_TRADE'

            # Check if signals match
            signals_match = (bt_signal == live_signal)

            comparison_data.append({
                'date': bt_date,
                'symbol': symbol,
                'backtest_signal': bt_signal,
                'backtest_score': bt_score,
                'live_signal': live_signal,
                'live_quantity': live_quantity,
                'live_side': live_side,
                'signals_match': signals_match,
                'difference': bt_signal - live_signal
            })

    return pd.DataFrame(comparison_data)

def save_reconciliation_report(comparison_df: pd.DataFrame, output_dir: Path):
    """Save detailed reconciliation report."""
    if comparison_df.empty:
        logger.warning("No comparison data to save")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    report_file = output_dir / f"reconciliation_report_{timestamp}.csv"

    # Calculate summary statistics
    total_comparisons = len(comparison_df)
    matching_signals = len(comparison_df[comparison_df['signals_match']])
    mismatch_rate = (total_comparisons - matching_signals) / total_comparisons if total_comparisons > 0 else 0

    # Add summary to the top of the file
    summary_stats = pd.DataFrame([
        {'Metric': 'Total Comparisons', 'Value': total_comparisons},
        {'Metric': 'Matching Signals', 'Value': matching_signals},
        {'Metric': 'Mismatches', 'Value': total_comparisons - matching_signals},
        {'Metric': 'Match Rate %', 'Value': f"{(1-mismatch_rate)*100:.2f}%"},
        {'Metric': '', 'Value': ''},  # Empty row separator
    ])

    # Save detailed comparison
    comparison_df.to_csv(report_file, index=False)

    # Log summary
    logger.info("=== Signal Reconciliation Summary ===")
    logger.info(f"Total Comparisons: {total_comparisons}")
    logger.info(f"Matching Signals: {matching_signals}")
    logger.info(f"Mismatches: {total_comparisons - matching_signals}")
    logger.info(f"Match Rate: {(1-mismatch_rate)*100:.2f}%")

    if total_comparisons > 0:
        # Show mismatches by symbol
        mismatches = comparison_df[~comparison_df['signals_match']]
        if not mismatches.empty:
            logger.info("\n=== Mismatches by Symbol ===")
            mismatch_summary = mismatches.groupby('symbol').size().sort_values(ascending=False)
            for symbol, count in mismatch_summary.items():
                logger.info(f"{symbol}: {count} mismatches")

    return report_file

def main():
    """Main weekly backtest analysis with reconciliation."""
    parser = argparse.ArgumentParser(description='XGBoost Weekly Backtest Runner with Reconciliation')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--rec-days', type=int, default=7, help='Number of recent days for reconciliation')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test (default: all trading symbols)')
    parser.add_argument('--reconcile', action='store_true', help='Run reconciliation against live trade files')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== XGBoost Weekly Backtest Analysis ===")

    # Use specified symbols or auto-detect all available models
    if args.symbols:
        symbols_to_test = args.symbols
    else:
        # Auto-detect all symbols with production models (timestamped format)
        models_dir = Path(__file__).parent / "models"
        available_symbols = set()
        for model_file in models_dir.glob("*_*.pkl"):  # e.g., @ES#C_20250917_201818.pkl
            # Extract symbol name (everything before the first underscore + timestamp)
            parts = model_file.stem.split('_')
            if len(parts) >= 3:  # symbol_date_time
                symbol = parts[0]
                available_symbols.add(symbol)
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

    # Reconciliation with live trade files
    if args.reconcile:
        logger.info(f"\n=== Running Reconciliation (last {args.rec_days} days) ===")
        logs_dir = Path(__file__).parent / "logs"

        if logs_dir.exists():
            # Get live signals from trade files
            live_signals = get_live_signals(logs_dir, args.rec_days)
            logger.info(f"Found {len(live_signals)} live trade signals")

            if live_signals:
                # Compare signals
                comparison_df = compare_signals(backtest_results, live_signals, args.rec_days)

                # Save reconciliation report
                output_dir = Path(__file__).parent / "weekly_analysis"
                report_file = save_reconciliation_report(comparison_df, output_dir)
                if report_file:
                    logger.info(f"Reconciliation report saved: {report_file}")
            else:
                logger.warning("No live signals found for reconciliation")
        else:
            logger.warning(f"Logs directory not found: {logs_dir}")

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
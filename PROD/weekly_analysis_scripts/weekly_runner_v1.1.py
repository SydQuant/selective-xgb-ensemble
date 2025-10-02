#!/usr/bin/env python3
"""
Weekly XGBoost Backtest and PnL Analysis with Signal Reconciliation v1.1

Key Improvements:
- Regex-based Bloomberg symbol mapping (no hardcoded dates)
- Simplified logic with cleaner structure
- Dynamic date detection from available data
- Robust error handling
"""

import sys
import argparse
import logging
import re
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add PROD to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add parent for research imports

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, SIGNAL_HOUR
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine


def setup_logging():
    """Setup logging to file and console."""
    weekly_dir = Path(__file__).parent / "outputs" / "weekly_backtest"
    weekly_dir.mkdir(parents=True, exist_ok=True)

    log_file = weekly_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}_weekly_backtest.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    return logging.getLogger(__name__)


def create_bloomberg_symbol_mapper():
    """
    Create regex-based Bloomberg to standard symbol mapping.

    Matches patterns like:
    - ADU5 Curncy, ADZ5 Curncy, ADH6 Curncy -> @AD#C
    - ESU5 Index, ESZ5 Index -> @ES#C
    - RXU5 Comdty, RXZ5 Comdty -> BD#C
    - "C Z5 Comdty" -> @C#C (handles space in base symbol)
    """
    from config import trading_config

    # Build regex patterns from config
    bloomberg_patterns = []

    for std_symbol, config in trading_config['instrument_config'].items():
        bloomberg = config.get('bloomberg', '')
        if not bloomberg:
            continue

        # Extract base symbol and suffix from Bloomberg ticker
        # Examples: "ADU5 Curncy" -> base="AD", suffix="Curncy"
        #           "ESU5 Index" -> base="ES", suffix="Index"
        #           "C Z5 Comdty" -> base="C", suffix="Comdty"
        match = re.match(r'([A-Z\s]+?)\s*[A-Z][0-9]+\s+(\w+)', bloomberg)
        if match:
            base = match.group(1).strip()
            suffix = match.group(2)
            # Create pattern that matches any contract month/year with same base and suffix
            # Handle optional space in base (like "C Z5")
            base_pattern = base.replace(' ', r'\s*')
            pattern = re.compile(rf'^{base_pattern}\s*[A-Z][0-9]+\s+{suffix}$')
            bloomberg_patterns.append((pattern, std_symbol))

    return bloomberg_patterns


def map_bloomberg_to_standard(bloomberg_symbol: str, patterns: list) -> str:
    """
    Map Bloomberg symbol to standard symbol using regex patterns.

    Args:
        bloomberg_symbol: Bloomberg ticker (e.g., "ADZ5 Curncy")
        patterns: List of (regex_pattern, standard_symbol) tuples

    Returns:
        Standard symbol or original if no match found
    """
    bloomberg_symbol = bloomberg_symbol.strip()

    for pattern, std_symbol in patterns:
        if pattern.match(bloomberg_symbol):
            return std_symbol

    logger.debug(f"No mapping found for Bloomberg symbol: {bloomberg_symbol}")
    return bloomberg_symbol


def get_available_trade_dates(logs_dir: Path) -> list:
    """Get sorted list of dates with available trade files."""
    date_dirs = [
        d for d in logs_dir.glob('*')
        if d.is_dir() and d.name.isdigit() and len(d.name) == 8
    ]
    dates = [datetime.strptime(d.name, '%Y%m%d').date() for d in date_dirs]
    return sorted(dates)


def get_live_signals(logs_dir: Path, lookback_days: int = 7):
    """
    Extract signals from recent trade files for reconciliation.

    Args:
        logs_dir: Directory containing dated log folders
        lookback_days: Number of recent days to include

    Returns:
        List of signal dictionaries
    """
    live_signals = []

    # Create Bloomberg mapping patterns
    bloomberg_patterns = create_bloomberg_symbol_mapper()
    logger.info(f"Created {len(bloomberg_patterns)} Bloomberg mapping patterns")

    # Get available trade dates
    available_dates = get_available_trade_dates(logs_dir)
    if not available_dates:
        logger.warning("No trade date directories found")
        return live_signals

    # Select recent dates
    recent_dates = available_dates[-lookback_days:] if len(available_dates) >= lookback_days else available_dates
    logger.info(f"Processing trade files from {len(recent_dates)} dates: {recent_dates[0]} to {recent_dates[-1]}")

    for date in recent_dates:
        date_dir = logs_dir / date.strftime('%Y%m%d')

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

                    # Map to standard symbol using regex
                    standard_symbol = map_bloomberg_to_standard(bloomberg_symbol, bloomberg_patterns)

                    # Skip unmapped symbols
                    if standard_symbol == bloomberg_symbol:
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
                        'symbol': standard_symbol,
                        'bloomberg_symbol': bloomberg_symbol,
                        'signal': signal,
                        'quantity': quantity,
                        'side': side
                    })

            except Exception as e:
                logger.warning(f"Failed to read {excel_file.name}: {e}")
                continue

    return live_signals


def run_backtest(symbol: str, end_date: datetime, backtest_days: int = 30) -> pd.DataFrame:
    """
    Run backtest using database data.

    Args:
        symbol: Target symbol to backtest
        end_date: Latest date for backtest
        backtest_days: Number of days to backtest

    Returns:
        DataFrame with backtest results
    """
    signal_engine = SignalEngine(
        Path(__file__).parent.parent / "models",
        Path(__file__).parent.parent / "config"
    )

    logger.info(f"Running backtest for {symbol} ending {end_date.date()}...")

    try:
        from data.data_utils_simple import prepare_real_data_simple

        # Calculate date range with buffer for feature calculation
        start_date = end_date - timedelta(days=backtest_days + 30)

        # Prepare data with targets
        full_df = prepare_real_data_simple(
            symbol,
            symbols=FEATURE_SYMBOLS,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            signal_hour=SIGNAL_HOUR,
            keep_rows_without_targets=True  # Keep recent data for reconciliation
        )

        if full_df.empty:
            logger.warning(f"No data available for {symbol}")
            return pd.DataFrame()

        # Get target column
        target_col = f"{symbol}_target_return"
        if target_col not in full_df.columns:
            logger.error(f"Target column {target_col} not found")
            return pd.DataFrame()

        # Load model package to get expected features
        models_dir = Path(__file__).parent.parent / "models"
        model_files = sorted(models_dir.glob(f"{symbol}_*.pkl"))
        if not model_files:
            logger.error(f"No model found for {symbol}")
            return pd.DataFrame()

        with open(model_files[-1], 'rb') as f:
            package = pickle.load(f)
        expected_features = package['selected_features']

        # Align features and returns
        features_aligned = full_df[expected_features].tail(backtest_days)
        returns_aligned = full_df[target_col].tail(backtest_days)
        common_index = features_aligned.index.intersection(returns_aligned.index)

        if common_index.empty:
            logger.warning(f"No common index for {symbol}")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        return pd.DataFrame()

    # Generate signals for each date
    results = []
    for idx in common_index:
        features_row = features_aligned.loc[[idx]]
        target_return = returns_aligned.loc[idx] if not pd.isna(returns_aligned.loc[idx]) else 0.0

        result = signal_engine.generate_signal(features_row, symbol)
        if result:
            signal, raw_score = result
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


def create_performance_plot(backtest_results: dict, backtest_days: int):
    """Create performance plot from backtest results."""
    symbols = list(backtest_results.keys())
    n_symbols = len(symbols)

    # Calculate grid dimensions
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
    else:
        rows, cols = 6, 6

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'XGBoost Production Backtest ({backtest_days} days)', fontsize=16)

    axes = axes.flatten() if n_symbols > 1 else [axes]

    for i, symbol in enumerate(symbols):
        ax = axes[i]
        df = backtest_results[symbol]

        if not df.empty:
            ax.plot(pd.to_datetime(df['date']), df['cumulative_pnl_pct'],
                   label=f'{symbol} PnL%', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

            total_pnl = df['cumulative_pnl_pct'].iloc[-1]
            n_trades = len(df[df['signal'] != 0])

            ax.set_title(f'{symbol}: PnL={total_pnl:.2f}%, Trades={n_trades}')
            ax.set_ylabel('Cumulative PnL (%)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {symbol}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{symbol}: No Data')

    # Hide unused subplots
    for j in range(len(symbols), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    output_dir = Path(__file__).parent / "outputs" / "weekly_backtest"
    plot_file = output_dir / f"backtest_{backtest_days}d_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_file


def compare_signals(backtest_results: dict, live_signals: list, lookback_days: int) -> pd.DataFrame:
    """Compare backtest signals with live trade signals."""
    if not live_signals:
        logger.warning("No live signals for comparison")
        return pd.DataFrame()

    live_df = pd.DataFrame(live_signals)
    live_grouped = live_df.groupby(['date', 'symbol']).agg({
        'signal': 'first',
        'quantity': 'first',
        'side': 'first'
    }).reset_index()

    comparison_data = []

    for symbol, backtest_df in backtest_results.items():
        if backtest_df.empty:
            continue

        recent_backtest = backtest_df.tail(lookback_days)

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
                live_signal = 0
                live_quantity = 0
                live_side = 'NO_TRADE'

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
    """Save reconciliation report with summary statistics."""
    if comparison_df.empty:
        logger.warning("No comparison data to save")
        return None

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    report_file = output_dir / f"reconciliation_{timestamp}.csv"

    total = len(comparison_df)
    matching = len(comparison_df[comparison_df['signals_match']])
    match_rate = (matching / total * 100) if total > 0 else 0

    comparison_df.to_csv(report_file, index=False)

    logger.info("\n=== Signal Reconciliation Summary ===")
    logger.info(f"Total Comparisons: {total}")
    logger.info(f"Matching Signals: {matching}")
    logger.info(f"Mismatches: {total - matching}")
    logger.info(f"Match Rate: {match_rate:.2f}%")

    if total > 0:
        mismatches = comparison_df[~comparison_df['signals_match']]
        if not mismatches.empty:
            logger.info("\n=== Mismatches by Symbol ===")
            mismatch_summary = mismatches.groupby('symbol').size().sort_values(ascending=False)
            for symbol, count in mismatch_summary.items():
                logger.info(f"{symbol}: {count} mismatches")

    return report_file


def main():
    """Main weekly backtest analysis."""
    parser = argparse.ArgumentParser(description='XGBoost Weekly Backtest with Reconciliation v1.1')
    parser.add_argument('--backtest-days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--reconcile-days', type=int, default=7, help='Number of recent days for reconciliation')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test (default: auto-detect)')
    parser.add_argument('--reconcile', action='store_true', help='Run reconciliation against live trades')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== XGBoost Weekly Backtest Analysis v1.1 ===")

    # Auto-detect symbols with models (filter by max_traded > 0)
    if args.symbols:
        symbols_to_test = args.symbols
    else:
        from config import trading_config
        models_dir = Path(__file__).parent.parent / "models"
        available_symbols = set()
        for model_file in models_dir.glob("*_*.pkl"):
            parts = model_file.stem.split('_')
            if len(parts) >= 3:  # symbol_date_time
                symbol = parts[0]
                # Only include symbols with max_traded > 0
                inst_config = trading_config['instrument_config'].get(symbol, {})
                if inst_config.get('max_traded', 0) > 0:
                    available_symbols.add(symbol)
        symbols_to_test = sorted(available_symbols)
        logger.info(f"Auto-detected {len(symbols_to_test)} symbols with models and max_traded > 0")

    # Determine end date from available trade data
    logs_dir = Path(__file__).parent.parent / "logs"
    available_dates = get_available_trade_dates(logs_dir) if logs_dir.exists() else []

    if available_dates:
        end_date = datetime.combine(max(available_dates), datetime.min.time())
        logger.info(f"Using end date {end_date.date()} from available trade files")
    else:
        end_date = datetime.now()
        logger.warning("No trade files found, using current date")

    logger.info(f"Running {args.backtest_days}-day backtest for {len(symbols_to_test)} symbols...")

    # Run backtests
    backtest_results = {}
    for symbol in symbols_to_test:
        logger.info(f"Processing {symbol}...")
        df = run_backtest(symbol, end_date, args.backtest_days)
        backtest_results[symbol] = df

        if not df.empty:
            total_pnl = df['cumulative_pnl_pct'].iloc[-1]
            n_trades = len(df[df['signal'] != 0])
            logger.info(f"  {symbol}: PnL={total_pnl:.2f}%, Trades={n_trades}")
        else:
            logger.info(f"  {symbol}: No data")

    # Reconciliation
    if args.reconcile and logs_dir.exists():
        logger.info(f"\n=== Running Reconciliation (last {args.reconcile_days} days) ===")
        live_signals = get_live_signals(logs_dir, args.reconcile_days)
        logger.info(f"Found {len(live_signals)} live trade signals")

        if live_signals:
            comparison_df = compare_signals(backtest_results, live_signals, args.reconcile_days)
            output_dir = Path(__file__).parent / "outputs" / "weekly_backtest"
            report_file = save_reconciliation_report(comparison_df, output_dir)
            if report_file:
                logger.info(f"Reconciliation report: {report_file}")
        else:
            logger.warning("No live signals found")

    # Skip performance plot - only generate reconciliation reports

    # Summary
    logger.info("\n=== Summary ===")
    total_pnl = sum(df['cumulative_pnl_pct'].iloc[-1] for df in backtest_results.values() if not df.empty)
    total_trades = sum(len(df[df['signal'] != 0]) for df in backtest_results.values() if not df.empty)
    successful = sum(1 for df in backtest_results.values() if not df.empty)

    logger.info(f"Symbols with data: {successful}/{len(symbols_to_test)}")
    logger.info(f"Total PnL: {total_pnl:.2f}%")
    logger.info(f"Total Trades: {total_trades}")
    logger.info("Weekly backtest completed")

    return True


if __name__ == "__main__":
    main()

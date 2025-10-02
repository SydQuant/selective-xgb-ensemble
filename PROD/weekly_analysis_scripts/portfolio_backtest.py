#!/usr/bin/env python3
"""
Portfolio Backtest with Production Position Sizing

Calculates portfolio-level PnL using:
- Production position sizing logic from trades_util.py
- Weighted portfolio returns based on actual notional allocations
- Dollar-weighted performance metrics
"""

import sys
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add PROD to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Add parent for research imports

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, SIGNAL_HOUR, trading_config, portfolio_config
from common.signal_engine import SignalEngine
from common.trades_util import TradeProcessor
from viz_helper import create_symbol_performance_grid, create_performance_and_metrics


def setup_logging():
    """Setup logging to file and console."""
    portfolio_dir = Path(__file__).parent / "outputs" / "portfolio"
    portfolio_dir.mkdir(parents=True, exist_ok=True)

    log_file = portfolio_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}_portfolio_backtest.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    return logging.getLogger(__name__)


def calculate_position_weight(symbol: str, signal: int, price: float) -> dict:
    """
    Calculate position size and weight using production logic.

    Returns:
        dict with: contracts, notional_usd, weight_pct
    """
    inst_config = trading_config['instrument_config'][symbol]

    # Calculate target position (production logic from trades_util.py)
    basket = inst_config['basket']
    notional = portfolio_config['allocations'][basket]
    multiplier = inst_config['multiplier']
    fraction = inst_config['fraction']

    # Raw position calculation (same as process_trade)
    raw_pos = signal * notional / fraction / price / multiplier
    target_contracts = int(np.floor(abs(raw_pos)) * np.sign(signal))

    # Apply position limits
    max_held = inst_config['max_held']
    if abs(target_contracts) > max_held:
        target_contracts = int(max_held * np.sign(target_contracts))

    # Calculate actual notional in USD
    actual_notional = abs(target_contracts) * price * multiplier

    # Calculate weight as % of total AUM
    total_aum = portfolio_config['total_aum']
    weight_pct = (actual_notional / total_aum) * 100 if total_aum > 0 else 0

    return {
        'symbol': symbol,
        'signal': signal,
        'contracts': target_contracts,
        'price': price,
        'notional_usd': actual_notional,
        'weight_pct': weight_pct,
        'basket': basket
    }


def run_portfolio_backtest(start_date: datetime, end_date: datetime,
                          symbols: list = None) -> pd.DataFrame:
    """
    Run portfolio backtest with production position sizing.

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        symbols: List of symbols (None = all available)

    Returns DataFrame with columns:
    - date
    - symbol
    - signal
    - contracts
    - weight_pct
    - return_pct (single-day return)
    - weighted_return_pct (weight * return)
    """
    signal_engine = SignalEngine(
        Path(__file__).parent.parent / "models",
        Path(__file__).parent.parent / "config"
    )

    if symbols is None:
        # Auto-detect symbols with models
        models_dir = Path(__file__).parent.parent / "models"
        available_symbols = set()
        for model_file in models_dir.glob("*_*.pkl"):
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                available_symbols.add(parts[0])
        symbols = sorted(available_symbols)

    logger.info(f"Running portfolio backtest from {start_date.date()} to {end_date.date()} for {len(symbols)} symbols...")

    # Get backtest data for each symbol
    from data.data_utils_simple import prepare_real_data_simple
    from data.loaders import get_arcticdb_connection

    # Add buffer for feature calculation
    data_start_date = start_date - timedelta(days=60)

    # Load raw price data from database
    futures_lib = get_arcticdb_connection()

    results = []

    for symbol in symbols:
        try:
            # Prepare features and returns
            full_df = prepare_real_data_simple(
                symbol,
                symbols=FEATURE_SYMBOLS,
                start_date=data_start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                signal_hour=SIGNAL_HOUR,
                keep_rows_without_targets=True
            )

            if full_df.empty:
                logger.warning(f"No data for {symbol}")
                continue

            # Get target column
            target_col = f"{symbol}_target_return"
            if target_col not in full_df.columns:
                logger.warning(f"No target column for {symbol}")
                continue

            # Load model features
            models_dir = Path(__file__).parent.parent / "models"
            model_files = sorted(models_dir.glob(f"{symbol}_*.pkl"))
            if not model_files:
                logger.warning(f"No model for {symbol}")
                continue

            with open(model_files[-1], 'rb') as f:
                package = pickle.load(f)
            expected_features = package['selected_features']

            # Get raw price data from database
            try:
                versioned_item = futures_lib.read(symbol)
                raw_price_df = versioned_item.data
                # Filter to signal hour and date range
                raw_price_df = raw_price_df[raw_price_df.index.hour == SIGNAL_HOUR]
                raw_price_df = raw_price_df[
                    (raw_price_df.index >= pd.Timestamp(start_date.strftime('%Y-%m-%d'))) &
                    (raw_price_df.index <= pd.Timestamp(end_date.strftime('%Y-%m-%d')))
                ]
                prices_aligned = raw_price_df['close']
            except Exception as e:
                logger.warning(f"Could not load price data for {symbol}: {e}")
                continue

            # Filter features and returns to backtest period
            backtest_mask = (full_df.index >= pd.Timestamp(start_date)) & (full_df.index <= pd.Timestamp(end_date))
            features_aligned = full_df.loc[backtest_mask, expected_features]
            returns_aligned = full_df.loc[backtest_mask, target_col]

            common_index = features_aligned.index.intersection(returns_aligned.index).intersection(prices_aligned.index)

            if common_index.empty:
                continue

            # Generate signals and calculate positions
            for idx in common_index:
                features_row = features_aligned.loc[[idx]]
                target_return = returns_aligned.loc[idx]
                price = prices_aligned.loc[idx]

                if pd.isna(price) or price <= 0:
                    continue

                # Generate signal
                result = signal_engine.generate_signal(features_row, symbol)
                if not result:
                    continue

                signal, raw_score = result

                # Calculate position size and weight
                position_info = calculate_position_weight(symbol, signal, price)

                # Calculate weighted return
                if not pd.isna(target_return):
                    # Single-day return (signal * actual return)
                    single_return = signal * target_return * 100  # in %
                    # Weighted by position size
                    weighted_return = (position_info['weight_pct'] / 100) * single_return
                else:
                    single_return = 0.0
                    weighted_return = 0.0

                results.append({
                    'date': idx.date(),
                    'symbol': symbol,
                    'signal': signal,
                    'raw_score': raw_score,
                    'price': price,
                    'contracts': position_info['contracts'],
                    'notional_usd': position_info['notional_usd'],
                    'weight_pct': position_info['weight_pct'],
                    'basket': position_info['basket'],
                    'return_pct': single_return,
                    'weighted_return_pct': weighted_return
                })

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue

    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

    return df


def calculate_portfolio_metrics(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily portfolio-level metrics.

    Returns DataFrame with:
    - date
    - portfolio_return_pct (sum of weighted returns)
    - cumulative_return_pct
    - num_positions
    - total_notional
    """
    if portfolio_df.empty:
        return pd.DataFrame()

    # Group by date
    daily_metrics = portfolio_df.groupby('date').agg({
        'weighted_return_pct': 'sum',  # Portfolio return = sum of weighted returns
        'symbol': 'count',              # Number of positions
        'notional_usd': 'sum'           # Total notional
    }).reset_index()

    daily_metrics.columns = ['date', 'portfolio_return_pct', 'num_positions', 'total_notional']

    # Calculate cumulative return
    daily_metrics['cumulative_return_pct'] = daily_metrics['portfolio_return_pct'].cumsum()

    # Calculate total AUM utilization
    total_aum = portfolio_config['total_aum']
    daily_metrics['utilization_pct'] = (daily_metrics['total_notional'] / total_aum) * 100

    return daily_metrics


def print_portfolio_summary(portfolio_df: pd.DataFrame, daily_metrics: pd.DataFrame):
    """Print comprehensive portfolio statistics."""
    if daily_metrics.empty:
        logger.warning("No portfolio data to summarize")
        return

    total_return = daily_metrics['cumulative_return_pct'].iloc[-1]
    mean_daily = daily_metrics['portfolio_return_pct'].mean()
    std_daily = daily_metrics['portfolio_return_pct'].std()
    sharpe = (mean_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0

    max_return = daily_metrics['portfolio_return_pct'].max()
    min_return = daily_metrics['portfolio_return_pct'].min()

    avg_positions = daily_metrics['num_positions'].mean()
    avg_utilization = daily_metrics['utilization_pct'].mean()

    # Calculate max drawdown
    cumulative = daily_metrics['cumulative_return_pct']
    running_max = cumulative.expanding().max()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    logger.info("\n" + "="*60)
    logger.info("PORTFOLIO BACKTEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Backtest Period: {daily_metrics['date'].iloc[0]} to {daily_metrics['date'].iloc[-1]}")
    logger.info(f"Number of Days: {len(daily_metrics)}")
    logger.info("")
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"  Total Return: {total_return:.2f}%")
    logger.info(f"  Daily Mean: {mean_daily:.3f}%")
    logger.info(f"  Daily Std Dev: {std_daily:.3f}%")
    logger.info(f"  Sharpe Ratio (annualized): {sharpe:.3f}")
    logger.info(f"  Max Daily Gain: {max_return:.2f}%")
    logger.info(f"  Max Daily Loss: {min_return:.2f}%")
    logger.info(f"  Max Drawdown: {max_drawdown:.2f}%")
    logger.info("")
    logger.info("PORTFOLIO COMPOSITION:")
    logger.info(f"  Avg Positions: {avg_positions:.1f}")
    logger.info(f"  Avg Utilization: {avg_utilization:.1f}%")
    logger.info(f"  Total Symbols: {portfolio_df['symbol'].nunique()}")
    logger.info("")

    # Basket breakdown
    logger.info("BASKET ALLOCATION:")
    basket_summary = portfolio_df.groupby('basket').agg({
        'notional_usd': 'mean',
        'weighted_return_pct': 'sum'
    })
    total_aum = portfolio_config['total_aum']
    for basket, row in basket_summary.iterrows():
        avg_notional = row['notional_usd']
        total_return = row['weighted_return_pct']
        pct_of_portfolio = (avg_notional / total_aum) * 100
        logger.info(f"  {basket:12s}: {pct_of_portfolio:5.1f}% of AUM, Return: {total_return:6.2f}%")
    logger.info("="*60)


def main():
    """Main portfolio backtest runner."""
    parser = argparse.ArgumentParser(description='Portfolio Backtest with Production Sizing')
    parser.add_argument('--start-date', type=str, default='2025-01-01',
                       help='Start date for backtest (YYYY-MM-DD, default: 2025-01-01)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD, default: latest available)')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test (default: all)')
    parser.add_argument('--save-details', action='store_true', help='Save detailed position-level data')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== Portfolio Backtest Analysis ===")

    # Parse start date
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    # Determine end date from available data or argument
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
        logger.info(f"Using specified end date: {end_date.date()}")
    else:
        logs_dir = Path(__file__).parent.parent / "logs"
        if logs_dir.exists():
            date_dirs = [d for d in logs_dir.glob('*') if d.is_dir() and d.name.isdigit() and len(d.name) == 8]
            if date_dirs:
                dates = [datetime.strptime(d.name, '%Y%m%d').date() for d in date_dirs]
                end_date = datetime.combine(max(dates), datetime.min.time())
                logger.info(f"Using end date {end_date.date()} from available data")
            else:
                end_date = datetime.now()
                logger.warning("No log directories found, using current date")
        else:
            end_date = datetime.now()

    # Run portfolio backtest
    portfolio_df = run_portfolio_backtest(start_date, end_date, args.symbols)

    if portfolio_df.empty:
        logger.error("No portfolio data generated")
        return False

    # Calculate portfolio metrics
    daily_metrics = calculate_portfolio_metrics(portfolio_df)

    # Create visualizations
    output_dir = Path(__file__).parent / "outputs" / "portfolio"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Creating portfolio visualizations...")

    # Combined performance and metrics (portfolio, baskets, drawdown, Sharpe, risk-return)
    perf_plot = create_performance_and_metrics(portfolio_df, daily_metrics, output_dir, start_date, end_date)
    logger.info(f"Performance & metrics saved: {perf_plot}")

    # Symbol performance grid
    symbol_plot = create_symbol_performance_grid(portfolio_df, daily_metrics, start_date, end_date, output_dir)
    logger.info(f"Symbol grid saved: {symbol_plot}")

    # Save detailed data if requested
    if args.save_details:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        positions_file = output_dir / f"positions_{timestamp}.csv"
        portfolio_df.to_csv(positions_file, index=False)
        logger.info(f"Position details saved: {positions_file}")

        metrics_file = output_dir / f"daily_metrics_{timestamp}.csv"
        daily_metrics.to_csv(metrics_file, index=False)
        logger.info(f"Daily metrics saved: {metrics_file}")

    # Print summary statistics
    print_portfolio_summary(portfolio_df, daily_metrics)

    return True


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simplified XGBoost Daily Signal Runner
Streamlined production script with core functionality only.
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, SIGNAL_HOUR
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine
from common.trades_util import TradeProcessor

def setup_logging():
    """Setup simple logging."""
    daily_dir = Path(__file__).parent / "logs" / datetime.now().strftime('%Y%m%d')
    daily_dir.mkdir(parents=True, exist_ok=True)

    log_file = daily_dir / f"{datetime.now().strftime('%Y%m%d_%H%M')}_xgb_daily.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )

    return logging.getLogger(__name__)

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='XGBoost Daily Signal Runner')
    parser.add_argument('--signal-hour', type=int, default=SIGNAL_HOUR)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=== XGBoost Daily Signal Generation ===")

    try:
        # Initialize components
        data_engine = DataEngine()
        signal_engine = SignalEngine(
            Path(__file__).parent / "models",
            Path(__file__).parent / "config"
        )
        trade_processor = TradeProcessor(Path(__file__).parent / "config" / "trading_config.yaml")

        logger.info(f"Processing {len(TRADING_SYMBOLS)} symbols...")

        # Generate signals
        all_signals = {}
        all_prices = {}

        for symbol in TRADING_SYMBOLS:
            features_df, price = data_engine.get_prediction_data(symbol, FEATURE_SYMBOLS, args.signal_hour)

            if features_df is not None and price is not None:
                result = signal_engine.generate_signal(features_df, symbol)
                if result:
                    signal, raw_score = result
                    all_signals[symbol] = {'signal': signal, 'raw_score': raw_score}
                    all_prices[symbol] = price
                    logger.info(f"{symbol}: Signal={signal}, Price={price:.2f}")

        # Process trades
        trades = []
        for symbol, signal_data in all_signals.items():
            trade = trade_processor.process_trade(
                symbol, signal_data['signal'], all_prices[symbol]
            )
            trades.append(trade)

        # Filter active trades
        active_trades = [t for t in trades if t['trade_size'] != 0]
        logger.info(f"Generated {len(active_trades)} active trades")

        for trade in active_trades:
            logger.info(f"  {trade['symbol']}: {trade['trade_size']:+.0f} @ {trade['price']:.2f}")

        # Save results
        if args.dry_run:
            logger.info("DRY RUN - No files saved")
        else:
            trade_file = trade_processor.save_gms_file(trades, args.signal_hour)
            trade_processor.upload_to_s3(trade_file)
            logger.info(f"Saved to {trade_file}")

        logger.info("Daily signal generation completed")
        return True

    except Exception as e:
        logger.error(f"Daily run failed: {e}")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""XGBoost Daily Signal Runner - Production Version"""

import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
sys.path.insert(0, str(Path(__file__).parent))

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, global_config
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine
from common.trades_util import TradeProcessor, send_sl_tp_email

EXCLUDED_SYMBOLS = {"@RTY#C"}  # Symbols to skip in signal generation

def filter_tradeable_symbols(symbols, signal_engine, logger):
    """Filter symbols based on exclusion list and model availability."""
    tradeable_symbols = []

    for symbol in symbols:
        # Filter 1: Check exclusion list
        if symbol in EXCLUDED_SYMBOLS:
            logger.info(f"Skipping {symbol}: in exclusion list")
            continue

        # Filter 2: Check model availability
        package = signal_engine.load_symbol_package(symbol)
        if not package or not package.get('models'):
            logger.info(f"Skipping {symbol}: no models available")
            continue

        tradeable_symbols.append(symbol)

    return tradeable_symbols

def main():
    """Main execution."""
    signal_hour = global_config.get('signal_config', {}).get('signal_hour', 12)

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger(__name__)
    logger.info("=== XGBoost Daily Signal Generation ===")

    try:
        # Initialize components
        base_path = Path(__file__).parent
        data_engine = DataEngine()
        signal_engine = SignalEngine(base_path / "models", base_path / "config")
        trade_processor = TradeProcessor(base_path / "config" / "trading_config.yaml")

        # Filter tradeable symbols
        tradeable_symbols = filter_tradeable_symbols(TRADING_SYMBOLS, signal_engine, logger)
        if not tradeable_symbols:
            logger.error("No tradeable symbols found")
            return False

        logger.info(f"Processing {len(tradeable_symbols)} tradeable symbols")

        # Generate signals and process trades
        batch_data = data_engine.get_prediction_data_batch(tradeable_symbols, FEATURE_SYMBOLS, signal_hour)
        if not batch_data:
            logger.error("No data retrieved")
            return False

        trades, active_trades = [], []
        for symbol, (features_df, price) in batch_data.items():
            result = signal_engine.generate_signal(features_df, symbol)
            if not result:
                continue

            signal, raw_score = result
            if signal != 0 or raw_score != 0.0:
                logger.info(f"{symbol}: Signal={signal}, Score={raw_score:.6f}, Price={price:.2f}")
                trade = trade_processor.process_trade(symbol, signal, price)
                trades.append(trade)
                if trade['trade_size'] != 0:
                    active_trades.append(trade)

        # Save and notify
        logger.info(f"Active trades: {len(active_trades)}")
        for trade in sorted(active_trades, key=lambda x: x['symbol']):
            logger.info(f"  {trade['symbol']}: {trade['trade_size']:+.0f} @ {trade['price']:.2f}")

        trade_file = trade_processor.save_gms_file(trades, signal_hour)
        trade_processor.upload_to_s3(trade_file)
        logger.info(f"Saved to {trade_file}")

        # Send email notification
        email_config = global_config.get('email_config', {})
        if active_trades and email_config.get('enabled', False):
            try:
                send_sl_tp_email(None, None, trade_file, email_config.get('recipients', []))
                logger.info("Email sent")
            except Exception as e:
                logger.error(f"Email failed: {e}")

        logger.info(f"Completed: {len(batch_data)} signals, {len(active_trades)} trades")
        return True

    except Exception as e:
        logger.error(f"Run failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
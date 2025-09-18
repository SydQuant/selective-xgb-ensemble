#!/usr/bin/env python3
"""XGBoost Daily Signal Runner - Production Version"""

import sys
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
sys.path.insert(0, str(Path(__file__).parent))

from config import TRADING_SYMBOLS, FEATURE_SYMBOLS, FX_CONFIG, global_config, instrument_config, config
from common.data_engine import DataEngine
from common.signal_engine import SignalEngine
from common.trades_util import TradeProcessor, send_email, convert_to_usd

EXCLUDED_SYMBOLS = {} #{"@RTY#C"}  # Symbols to skip in signal generation

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
        trade_processor = TradeProcessor(
            base_path / "config" / "trading_config.yaml",
            base_path / "config" / "trading_config_q.yaml"
        )

        # Filter tradeable symbols & get FX symbols
        tradeable_symbols = filter_tradeable_symbols(TRADING_SYMBOLS, signal_engine, logger)
        fx_symbols = config.get_required_fx_symbols(tradeable_symbols)
        logger.info(f"Processing {len(tradeable_symbols)} tradeable symbols & {len(fx_symbols)} FX symbols")

        # Get FX data
        fx_data = {}
        if fx_symbols:
            fx_data = data_engine.iqfeed_client.get_live_data_multi(fx_symbols, days=1, interval_min=60)

        # Get batch data
        batch_data = data_engine.get_prediction_data_batch(tradeable_symbols, FEATURE_SYMBOLS, signal_hour)

        # Process signals into P and Q trades
        trades_p, trades_q = [], []
        active_trades_p, active_trades_q = [], []

        for symbol, (features_df, price) in batch_data.items():
            result = signal_engine.generate_signal(features_df, symbol)

            signal, raw_score = result
            if signal != 0 or raw_score != 0.0:
                # Convert price to USD if needed
                usd_price = convert_to_usd(symbol, price, instrument_config, fx_data, FX_CONFIG, logger)

                logger.info(f"{symbol}: Signal={signal}, Avg Score={raw_score*1000:.2f} bps, Price={usd_price:.4f}")

                # Process P trade (regular allocation)
                trade_p = trade_processor.process_trade(symbol, signal, usd_price)
                trades_p.append(trade_p)
                if trade_p['trade_size'] != 0:
                    active_trades_p.append(trade_p)

                # Process Q trade (Q allocation)
                trade_q = trade_processor.process_trade_q(symbol, signal, usd_price)
                trades_q.append(trade_q)
                if trade_q['trade_size'] != 0:
                    active_trades_q.append(trade_q)

        # Log active trades
        logger.info(f"Active P trades: {len(active_trades_p)}")
        for trade in sorted(active_trades_p, key=lambda x: x['symbol']):
            logger.info(f"  P {trade['symbol']}: {trade['trade_size']:+.0f} @ {trade['price']:.4f}")

        logger.info(f"Active Q trades: {len(active_trades_q)}")
        for trade in sorted(active_trades_q, key=lambda x: x['symbol']):
            logger.info(f"  Q {trade['symbol']}: {trade['trade_size']:+.0f} @ {trade['price']:.4f}")

        # Save trade files
        trade_file_p = trade_processor.save_gms_file(trades_p, signal_hour)
        trade_file_q = trade_processor.save_q_trade_file(trades_q)

        # Upload P file to S3 (only P file)
        trade_processor.upload_to_s3(trade_file_p)
        logger.info(f"Saved P trades to {trade_file_p}")
        logger.info(f"Saved Q trades to {trade_file_q}")

        # Email both files together
        email_config = global_config.get('email_config', {})
        if (active_trades_p or active_trades_q) and email_config.get('enabled', False):
            try:
                send_email([trade_file_p, trade_file_q], email_config.get('recipients', []))
                logger.info("Email sent with both trade files")
            except Exception as e:
                logger.error(f"Email failed: {e}")

        logger.info(f"Completed: {len(batch_data)} signals, {len(active_trades_p)} P trades, {len(active_trades_q)} Q trades")
        return True

    except Exception as e:
        logger.error(f"Run failed: {e}")
        return False

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
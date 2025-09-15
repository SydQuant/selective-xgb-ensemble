"""
Simplified Trade Processing for XGBoost Production
Core trade generation functionality without complexity.
"""

import math
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

class TradeProcessor:
    """Simplified trade processing."""

    def __init__(self, config_file: Path):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.instrument_config = config['instrument_config']
        self.portfolio_config = config['portfolio_config']
        self.current_positions = config['current_positions']
        self.s3_config = config.get('s3', {})

    def process_trade(self, symbol: str, signal: int, price: float) -> Dict:
        """Process single trade - exact v2.1 logic."""
        inst_config = self.instrument_config[symbol]
        current_pos = self.current_positions.get(symbol, 0)

        # Calculate position
        notional = self.portfolio_config['allocations'][inst_config['basket']]
        raw_pos = signal * notional / inst_config['fraction'] / price / inst_config['multiplier']
        target_pos = math.floor(abs(raw_pos)) * np.sign(signal)
        trade_size = target_pos - current_pos

        # Apply limits
        max_held = inst_config['max_held']
        if abs(target_pos) > max_held:
            target_pos = max_held * np.sign(target_pos)
            trade_size = target_pos - current_pos

        max_traded = inst_config['max_traded']
        if abs(trade_size) > max_traded:
            trade_size = max_traded * np.sign(trade_size)

        return {
            'symbol': symbol,
            'signal': signal,
            'current_pos': current_pos,
            'target_pos': target_pos,
            'trade_size': trade_size,
            'price': price
        }

    def save_gms_file(self, trades: List[Dict], signal_hour: int = 12) -> Path:
        """Save GMS trade file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        daily_dir = Path(__file__).parent.parent / "logs" / timestamp
        daily_dir.mkdir(parents=True, exist_ok=True)

        trade_file = daily_dir / f"{timestamp}_GMS_Trade_File_xgb_prod_{signal_hour}hr.xlsx"

        gms_trades = []
        for trade in trades:
            if trade['trade_size'] == 0:
                continue

            symbol = trade['symbol']
            inst_config = self.instrument_config[symbol]

            gms_trades.append({
                'TRADER_ID': 'christian-beulen',
                'SECURITY_ID_TYPE': 'BLOOMBERG_TICKER',
                'SECURITY_ID': inst_config['bloomberg'],
                'QUANTITY': abs(trade['trade_size']),
                'URGENCY': 5,
                'SIDE': "BUY" if trade['trade_size'] > 0 else "SELL",
                'STRATEGY': 'XGBOOST',
                'ORDER_TYPE': 'MARKET',
                'LIMIT_PRICE': '',
                'STOP_PRICE': '',
                'POSITION_GROUP': f"{inst_config['basket']}|BEULEN_SQP",
                'HELD': 'Y',
                'TIF': 'DAY',
                'Current Position': trade['current_pos'],
                'Target': trade['target_pos'],
                'Description': inst_config['description']
            })

        if gms_trades:
            df = pd.DataFrame(gms_trades)
            df.to_excel(trade_file, index=False)
            logger.info(f"Saved {len(gms_trades)} trades to {trade_file}")

        return trade_file

    def upload_to_s3(self, file_path: Path) -> bool:
        """Upload file to S3."""
        if not self.s3_config.get('trades_bucket'):
            return True

        try:
            import boto3
            bucket, key = self.s3_config['trades_bucket'].split('/', 1)
            boto3.client('s3').upload_file(str(file_path), bucket, f"{key}/{file_path.name}")
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False
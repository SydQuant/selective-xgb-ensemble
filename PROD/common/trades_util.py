"""XGBoost Production Trade Processing - Streamlined and Simplified"""

import math
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)

class TradeProcessor:
    """Trade processing and file generation."""

    def __init__(self, config_file: Path):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.instrument_config = config['instrument_config']
        self.portfolio_config = config['portfolio_config']
        self.current_positions = config['current_positions']
        self.s3_config = config.get('s3', {})

    def process_trade(self, symbol: str, signal: int, price: float) -> Dict:
        """Process single trade using portfolio allocation logic."""
        inst_config = self.instrument_config[symbol]
        current_pos = self.current_positions.get(symbol, 0)

        # Calculate target position
        notional = self.portfolio_config['allocations'][inst_config['basket']]
        raw_pos = signal * notional / inst_config['fraction'] / price / inst_config['multiplier']
        target_pos = math.floor(abs(raw_pos)) * np.sign(signal)
        trade_size = target_pos - current_pos

        # Apply position limits
        if abs(target_pos) > inst_config['max_held']:
            target_pos = inst_config['max_held'] * np.sign(target_pos)
            trade_size = target_pos - current_pos

        # Apply trade size limits
        if abs(trade_size) > inst_config['max_traded']:
            trade_size = inst_config['max_traded'] * np.sign(trade_size)

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
                'STRATEGY': 'LIQSEEK',
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


def send_sl_tp_email(sl_trades, tp_trades, trade_file: Path, recipients: List[str]) -> bool:

    try:
        try:
            from dotenv import load_dotenv  # type: ignore
            load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
        except Exception:
            pass

        import smtplib
        from email.message import EmailMessage

        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        sender_email = os.getenv("EMAIL_USER")
        sender_password = os.getenv("EMAIL_PASS")

        if not recipients:
            logger.warning("No email recipients configured; skipping email send.")
            return False

        if not sender_email or not sender_password:
            logger.error("EMAIL_USER or EMAIL_PASS not set; cannot send email.")
            return False

        # Gmail app passwords are displayed with spaces; remove them if present
        sender_password = sender_password.replace(" ", "")

        # Compose message
        msg = EmailMessage()
        msg['From'] = sender_email
        msg['To'] = ", ".join(recipients)
        msg['Subject'] = f"Trading Report (XGB): {trade_file.name.replace('.xlsx', '')}"
        msg.set_content("Please find attached trade file, SL/TP report, and plots.\n\n")

        # Attach files (only if provided and exist)
        for file_path in [trade_file, sl_trades, tp_trades]:
            try:
                if file_path is None:
                    continue
                fp = Path(file_path)
                if not fp.exists():
                    continue
                maintype = 'application'
                subtype = fp.suffix.lstrip('.') or 'octet-stream'
                with open(fp, 'rb') as f:
                    data = f.read()
                msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=fp.name)
            except Exception as attach_err:
                logger.warning(f"Failed to attach {file_path}: {attach_err}")

        # Connect and send
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        logger.info(f"Email sent to {recipients}")
        return True

    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
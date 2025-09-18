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

    def __init__(self, config_file: Path, config_q_file: Path = None):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        self.instrument_config = config['instrument_config']
        self.portfolio_config = config['portfolio_config']
        self.current_positions = config['current_positions']
        self.s3_config = config.get('s3', {})

        # Load Q config if provided
        if config_q_file and config_q_file.exists():
            with open(config_q_file, 'r') as f:
                config_q = yaml.safe_load(f)
            self.portfolio_config_q = config_q['portfolio_config']
            self.instrument_config_q = config_q['instrument_config']
        else:
            self.portfolio_config_q = None
            self.instrument_config_q = None

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

    def process_trade_q(self, symbol: str, signal: int, price: float) -> Dict:
        """Process single trade using Q portfolio allocation logic."""
        if not self.portfolio_config_q or not self.instrument_config_q:
            raise ValueError("Q config not loaded")

        inst_config = self.instrument_config_q[symbol]
        current_pos = self.current_positions.get(symbol, 0)

        # Calculate target position using Q allocation
        notional = self.portfolio_config_q['allocations'][inst_config['basket']]
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

    def save_q_trade_file(self, trades: List[Dict], signal_hour: int = 12) -> Path:
        """Save Q trades in LPXD external advisors format to CSV file."""
        if not self.portfolio_config_q or not self.instrument_config_q:
            raise ValueError("Q config not loaded")
        from datetime import timezone

        # Generate timestamp in GMT timezone with YYYYMMDD-HHMM format
        gmt_now = datetime.now(timezone.utc)
        timestamp_file = gmt_now.strftime("%Y%m%d-%H%M")
        timestamp_value = gmt_now.strftime("%Y-%m-%d %H:%M")

        # Create daily directory and file path
        timestamp = datetime.now().strftime("%Y%m%d")
        daily_dir = Path(__file__).parent.parent / "logs" / timestamp
        daily_dir.mkdir(parents=True, exist_ok=True)
        trade_file = daily_dir / f"lpxd_external_advisors_CB_{timestamp_file}.csv"

        q_trades = []
        for trade in trades:
            if trade['trade_size'] == 0:
                continue

            symbol = trade['symbol']
            inst_config = self.instrument_config_q[symbol]
            bloomberg_ticker = inst_config.get('bloomberg_generic', inst_config.get('bloomberg', ''))
            internal_code = inst_config.get('q_internal_code', '')
            if internal_code == '' or pd.isna(internal_code):
                internal_code = 'nan'
            extra_key = f"S1_{internal_code}"
            currency = inst_config.get('currency', 'USD')
            basket = inst_config['basket']

            trade_size = trade['trade_size']
            ref_price = trade['price']
            target_notional = trade_size * ref_price * inst_config.get('multiplier', 1)

            q_trades.append({
                'id_specific': 'CB',
                'extra_key': extra_key,
                'value_ts': timestamp_value,
                'strategy': 'S1',
                'internal_code': internal_code,
                'ric': bloomberg_ticker,
                'ticker': bloomberg_ticker,
                'target_notional': round(target_notional, 2),
                'currency': currency,
                'target_contracts': trade_size,
                'ref_price': ref_price,
                'advisor_name': 'Christian BEULEN',
                'basket': basket
            })

        if not q_trades:
            logger.info("No Q trades to save")
            return trade_file

        # Convert to DataFrame and save as CSV
        q_df = pd.DataFrame(q_trades)
        q_df = q_df.sort_values(by=['basket', 'ticker']).drop('basket', axis=1)
        q_df = q_df[q_df['target_contracts'] != 0]
        logger.info(f"Saved {len(q_df)} Q trades to {trade_file}")
        q_df.to_csv(trade_file, index=False)

        return trade_file


def convert_to_usd(symbol: str, price: float, instrument_config: dict, fx_data: dict, fx_config: dict, logger=None) -> float:
    """Convert price to USD using FX data."""
    currency = instrument_config[symbol].get('currency', 'USD')

    # Return early for USD or missing data
    if currency == 'USD' or not fx_data or currency not in fx_config:
        return price

    # Get FX rate and convert
    fx_ticker = fx_config[currency]['ticker']
    if fx_ticker not in fx_data:
        return price

    fx_rate = fx_data[fx_ticker]['close'].iloc[-1]
    usd_price = price / fx_rate if fx_config[currency]['inverted'] else price * fx_rate

    if logger:
        logger.info(f"{symbol}: {currency} {price:.2f} -> USD {usd_price:.2f}")

    return usd_price


def save_q_trade_file(trades: pd.DataFrame) -> Path:
        """Save trades in LPXD external advisors format to CSV file."""
        # Generate timestamp in GMT timezone with YYYYMMDD-HHMM format
        gmt_now = datetime.now(timezone.utc)
        timestamp_file = gmt_now.strftime("%Y%m%d-%H%M")
        timestamp_value = gmt_now.strftime("%Y-%m-%d %H:%M")
        
        # Create daily directory and file path
        daily_dir = get_daily_dir()
        trade_file = daily_dir / f"lpxd_external_advisors_CB_{timestamp_file}.csv"
        
        q_trades = []
        for _, row in trades.iterrows():
            
            symbol = row['symbol']
            inst_config = instrument_config[symbol]
            bloomberg_ticker = inst_config['bloomberg']
            internal_code = inst_config['internal_code']
            if internal_code == '':  
                internal_code = 'nan'
            extra_key = f"S1_{internal_code}"
            currency = inst_config['currency']
            basket = inst_config['basket']

            trade_size = row['trade_size']
            ref_price = row['ref_price'] 
            target_notional = trade_size * ref_price * inst_config.get('multiplier', 1)
            
            q_trades.append({
                'id_specific': 'CB',
                'extra_key': extra_key,
                'value_ts': timestamp_value,
                'strategy': 'S1',
                'internal_code': internal_code,
                'ric': bloomberg_ticker,
                'ticker': bloomberg_ticker,
                'target_notional': round(target_notional, 2),
                'currency': currency,
                'target_contracts': trade_size,
                'ref_price': ref_price,
                'advisor_name': 'Christian BEULEN',
                'basket': basket
            })
        
        if not q_trades:
            print("No trades to save")
            return Path()
        
        # Convert to DataFrame and save as CSV
        q_df = pd.DataFrame(q_trades)
        q_df = q_df.sort_values(by=['basket', 'ticker']).drop('basket', axis=1)
        q_df = q_df[q_df['target_contracts'] != 0]
        print(f"*** Total trades: {len(q_df)} ***")
        q_df.to_csv(trade_file, index=False)
        print(f"Saved Q trade file: {trade_file}")
        
        return trade_file

def send_email(trade_files: List[Path], recipients: List[str]) -> bool:

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
        msg['Subject'] = f"Trading Report (XGB): {datetime.now().strftime('%Y%m%d')}"
        msg.set_content("Please find attached trade files.\n\n")

        # Attach files (only if provided and exist)
        for file_path in trade_files:
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
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Dict, List

from rich import print
import pandas as pd
import pytz

logger = logging.getLogger(__name__)

class IQFeedClient:
    """IQFeed client wrapper for fetching live market data."""

    def __init__(self, host: str = 'localhost', port: int = 9100):
        self.host = host
        self.port = port
        self.timezone = pytz.timezone('America/New_York')

    def get_live_data(self, symbol: str, days: int = 5, interval_min: int = 60) -> pd.DataFrame:
        """Fetch live interval data for a single symbol."""
        # Calculate time range
        end_time = datetime.now(self.timezone)
        # Align end_time to interval
        total_min = end_time.hour * 60 + end_time.minute
        delta = (interval_min - (total_min % interval_min)) % interval_min
        end_time = (end_time + timedelta(minutes=delta)).replace(second=0, microsecond=0)
        start_time = end_time - timedelta(days=days)

        # Build request
        req = (
            f"HIT,{symbol},{interval_min*60},"
            f"{start_time.strftime('%Y%m%d %H%M%S')},"
            f"{end_time.strftime('%Y%m%d %H%M%S')},,,,1\r\n"
        )

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.host, self.port))
        try:
            sock.sendall(req.encode())
            time.sleep(0.1)
            buffer = []
            while True:
                chunk = sock.recv(4096).decode()
                if not chunk:
                    break
                if '!ENDMSG!' in chunk:
                    buffer.append(chunk.split('!ENDMSG!')[0])
                    break
                buffer.append(chunk)
            data = ''.join(buffer).strip()
            if not data:
                return pd.DataFrame()
            records = []
            for line in data.split('\n'):
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                try:
                    ts = datetime.strptime(parts[0].strip(), '%Y-%m-%d %H:%M:%S')
                    ts = self.timezone.localize(ts)
                    records.append({
                        'timestamp': ts,
                        'high': float(parts[1]),
                        'low': float(parts[2]),
                        'open': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]),
                    })
                except Exception as e:
                    logger.error(f"Error parsing line for {symbol}: {e}")
                    continue
            df = pd.DataFrame(records)
            if not df.empty:
                df.set_index('timestamp', inplace=True)
            return df
        finally:
            sock.close()

    def get_live_data_multi(
        self,
        symbols: List[str],
        days: int = 5,
        interval_min: int = 60,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch live data for multiple symbols in parallel."""
        out: Dict[str, pd.DataFrame] = {}

        print("====Getting live prices====")

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.get_live_data, sym, days, interval_min): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    out[sym] = fut.result()
                except Exception as e:
                    logger.error(f"Error fetching live data for {sym}: {e}")
        return out
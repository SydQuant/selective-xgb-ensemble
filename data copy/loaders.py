"""
Data loaders for futures and equity market data.
Updated to load real futures data from ArcticDB with proper data types and formatting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Legacy data directory for fallback
DATA_DIR = os.environ.get("AGENTS_DATA_DIR", "/mnt/data")

def get_arcticdb_connection():
    """Get ArcticDB connection and futures library."""
    import arcticdb as adb
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "data", "arctic_data")
    # Fallback to nested data directory if main one doesn't exist
    if not os.path.exists(db_path):
        db_path = os.path.join(current_dir, "data", "data", "arctic_data")
    arctic = adb.Arctic(f"lmdb://{db_path}")
    return arctic['futures_data']

def load_equity_panel():
    """Load equity panel data - returns empty DataFrame for futures-only operation."""
    # For futures-only operation, return empty DataFrame
    # This avoids dependency on synthetic equity data files
    return pd.DataFrame(columns=['date', 'ticker', 'close'])

def _ensure_signal_hour_coverage(df, signal_hour=12):
    """Ensure every weekday has a signal_hour row, filling gaps if needed."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Filter to signal hour only
    signal_hour_data = df[df.index.hour == signal_hour].copy()
    
    # Forward fill missing data by symbol, only for business days
    signal_hour_data = signal_hour_data.groupby('symbol').apply(lambda x: x.resample('B').ffill()).reset_index(level=0, drop=True)
    
    return signal_hour_data.reset_index()

def _process_daily_aggregation(df):
    """Process daily aggregation from signal hour to signal hour."""
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])
    
    # Use shift to get T+1 close for return calculation
    df['price'] = df.groupby('symbol')['close'].shift(-1)
    
    # Set date to be the start of the 24h window
    df['date'] = df['date'].dt.normalize()
    
    # Drop the last row for each group as it will have a NaN price
    df = df.dropna(subset=['price'])
    return df

def _load_symbols_preset():
    """Load symbol preset from config file."""
    import yaml
    symbols_file = os.path.join(os.path.dirname(__file__), '..', 'common', 'symbols.yaml')
    
    # Use hardcoded default symbols to ensure consistency
    default_symbols = ['@AD#C', '@ES#C', 'QCL#C', '@TY#C', '@S#C']
    
    with open(symbols_file, 'r') as f:
        symbols_config = yaml.safe_load(f)
        symbols = symbols_config.get('default', default_symbols)
        return symbols, symbols_config

def load_futures_panel(symbols=None, start_date=None, end_date=None, resample_freq='D', signal_hour=12):
    """Load futures data from ArcticDB with daily aggregation."""
    if symbols is None:
        symbols, symbols_config = _load_symbols_preset()
    
    futures_lib = get_arcticdb_connection()
    available_symbols = futures_lib.list_symbols()
    symbols_to_load = [s for s in symbols if s in available_symbols]
    
    # Load data with date range optimization
    start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=30) if start_date else None
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=2) if end_date else None
    date_range = (start_dt, end_dt)

    all_data = []
    for symbol_key in symbols_to_load:
        data = futures_lib.read(symbol_key, date_range=date_range).data
        data.rename(columns={'price': 'close'}, inplace=True)
        data.index.name = 'date'
        data['symbol'] = symbol_key
        all_data.append(data)

    data = pd.concat(all_data).reset_index()

    # Calculate returns only (vol20 handled by individual agents)
    data['returns'] = data.groupby('symbol')['close'].pct_change(fill_method=None)

    # Ensure signal hour coverage and daily aggregation
    data_with_coverage = _ensure_signal_hour_coverage(data, signal_hour)
    df = _process_daily_aggregation(data_with_coverage)

    # Set final column structure (removed vol20 as agents handle it internally)
    df = df[['date', 'symbol', 'price', 'open', 'high', 'low', 'volume']]
    return df.sort_values(['date', 'symbol']).reset_index(drop=True)

def get_available_symbols() -> List[str]:
    """Get list of available symbols in ArcticDB."""
    futures_lib = get_arcticdb_connection()
    return sorted(futures_lib.list_symbols())

def get_symbol_info(symbol):
    """Get symbol info."""
    futures_lib = get_arcticdb_connection()
    data = futures_lib.read(symbol).data
    return {
        "symbol": symbol, "records": len(data),
        "start_date": data.index.min(), "end_date": data.index.max(),
        "price_range": (data['close'].min(), data['close'].max())
    }

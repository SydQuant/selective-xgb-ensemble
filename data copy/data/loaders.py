
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
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(current_dir), "data_files")

def get_arcticdb_connection():
    """Get ArcticDB connection and futures library."""
    try:
        import arcticdb as adb
        
        # Database configuration - ArcticDB data is now in data/arctic_data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "arctic_data")
        db_uri = f"lmdb://{db_path}"
        
        # Connect to ArcticDB
        arctic = adb.Arctic(db_uri)
        
        futures_lib = arctic['futures_data']
        return futures_lib
        
    except ImportError:
        raise ImportError("ArcticDB not installed. Please install with: pip install arcticdb")
    except Exception as e:
        logger.error(f"Failed to connect to ArcticDB: {e}")
        raise

def load_equity_panel():
    """Load synthetic equity panel data for gym environment compatibility."""
    df = pd.read_csv(os.path.join(DATA_DIR, "synth_equity_panel.csv"), parse_dates=["date"])
    df = df.rename(columns={'ticker': 'symbol', 'close': 'price'})
    return df

def _ensure_signal_hour_coverage(data, signal_hour=12):
    """Ensure every weekday has a signal_hour row, filling gaps if needed."""
    import logging
    logger = logging.getLogger(__name__)
    
    # Get all unique dates and ensure signal_hour exists for each weekday
    all_dates = pd.date_range(data.index.min().date(), data.index.max().date(), freq='D')
    weekdays = all_dates[all_dates.weekday < 5]  # Mon=0 to Fri=4
    
    filled_data = data.copy()
    for date in weekdays:
        signal_time = pd.Timestamp(date) + pd.Timedelta(hours=signal_hour)
        
        # Check if signal_hour exists for this date
        if signal_time not in filled_data.index:
            # Find the closest available data for this date
            date_data = filled_data[filled_data.index.date == date.date()]
            if not date_data.empty:
                # Use the last available data point of the day
                last_row = date_data.iloc[-1]
                filled_data.loc[signal_time] = last_row
                # logger.info(f"Filled missing signal_hour {signal_hour}:00 for {date.date()}")
    
    return filled_data.sort_index()

def load_futures_panel(symbols=None, start_date=None, end_date=None, resample_freq='D', signal_hour=12):
    """Load futures data from ArcticDB with daily aggregation from signal_hour close to next signal_hour close."""
    futures_lib = get_arcticdb_connection()
    symbols = symbols or futures_lib.list_symbols()
    symbols = [s for s in symbols if s in futures_lib.list_symbols()]
    
    panel_data = []
    for symbol in symbols:
        data = futures_lib.read(symbol).data
        if start_date: 
            start_dt = pd.Timestamp(start_date) if not isinstance(start_date, pd.Timestamp) else start_date
            data = data[data.index >= start_dt]
        if end_date: 
            end_dt = pd.Timestamp(end_date) if not isinstance(end_date, pd.Timestamp) else end_date
            data = data[data.index <= end_dt]
        if data.empty: continue
        
        if resample_freq == 'D' and signal_hour is not None:
            # Ensure signal_hour coverage for all weekdays
            data = _ensure_signal_hour_coverage(data, signal_hour)
            
            # Get signal_hour data points
            signal_data = data[data.index.hour == signal_hour].copy()
            if len(signal_data) < 2: continue
            
            daily_agg = []
            for i in range(len(signal_data) - 1):
                start_time = signal_data.index[i]
                end_time = signal_data.index[i + 1]
                
                # Get data window from start_time to end_time (exclusive)
                window_data = data.loc[start_time:end_time].iloc[:-1]
                if len(window_data) == 0: continue
                
                daily_row = {
                    'open': window_data['open'].iloc[0],
                    'high': window_data['high'].max(),
                    'low': window_data['low'].min(),
                    'close': window_data['close'].iloc[-1],
                    'volume': window_data['volume'].sum()
                }
                daily_agg.append((start_time.normalize(), daily_row))
            
            # Convert to DataFrame
            if daily_agg:
                dates, rows = zip(*daily_agg)
                data = pd.DataFrame(rows, index=dates)
        elif resample_freq != 'H':
            data = data.resample(resample_freq).agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
        
        for date, row in data.iterrows():
            panel_data.append({
                'date': date, 'symbol': symbol, 'price': round(row['close'], 4),
                'open': round(row['open'], 4), 'high': round(row['high'], 4),
                'low': round(row['low'], 4), 'volume': int(row['volume'])
            })
    
    df = pd.DataFrame(panel_data)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['date', 'symbol']).reset_index(drop=True)

def get_available_symbols() -> List[str]:
    """Get list of available symbols in ArcticDB."""
    try:
        futures_lib = get_arcticdb_connection()
        return sorted(futures_lib.list_symbols())
    except Exception as e:
        logger.error(f"Failed to get available symbols: {e}")
        return []

def get_symbol_info(symbol):
    """Get symbol info."""
    futures_lib = get_arcticdb_connection()
    data = futures_lib.read(symbol).data
    return {
        "symbol": symbol, "records": len(data),
        "start_date": data.index.min(), "end_date": data.index.max(),
        "price_range": (data['close'].min(), data['close'].max())
    }

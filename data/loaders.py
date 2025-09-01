"""
Simplified Arctic DB loader for futures market data.
"""

import os
import pandas as pd
from typing import List
import logging

logger = logging.getLogger(__name__)

def get_arcticdb_connection():
    """Get ArcticDB connection and futures library."""
    import arcticdb as adb
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "arctic_data")
    
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Arctic DB path not found: {db_path}")
    
    arctic = adb.Arctic(f"lmdb://{db_path}")
    
    # Check if futures_data library exists, create if needed
    if 'futures_data' not in arctic.list_libraries():
        logger.warning("futures_data library not found in Arctic DB - create it first to use real data")
        raise Exception("futures_data library not found - run data ingestion first")
    
    return arctic['futures_data']

def get_available_symbols() -> List[str]:
    """Get list of available symbols in ArcticDB."""
    try:
        futures_lib = get_arcticdb_connection()
        return sorted(futures_lib.list_symbols())
    except Exception as e:
        logger.error(f"Failed to get available symbols: {e}")
        return []

def get_symbol_info(symbol: str):
    """Get basic symbol information."""
    try:
        futures_lib = get_arcticdb_connection()
        data = futures_lib.read(symbol).data
        return {
            "symbol": symbol, 
            "records": len(data),
            "start_date": data.index.min(), 
            "end_date": data.index.max(),
            "columns": list(data.columns)
        }
    except Exception as e:
        logger.error(f"Failed to get info for {symbol}: {e}")
        return None
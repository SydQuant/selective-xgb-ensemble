#!/usr/bin/env python3
"""
Create sample data in Arctic DB to test the real data pipeline.
This demonstrates how to populate the database with futures data.
"""

import pandas as pd
import numpy as np
import arcticdb as adb
import os
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_futures_data():
    """Create sample futures data for testing."""
    symbols = ['@ES#C', '@AD#C', '@BP#C', '@EU#C', '@TY#C']
    
    # Create 6 months of hourly data (smaller for testing)
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    date_range = pd.date_range(start_date, end_date, freq='h')
    
    # Connect to Arctic DB
    db_path = os.path.join('.', 'data', 'arctic_data')
    arctic = adb.Arctic(f"lmdb://{db_path}")
    
    # Create futures_data library if it doesn't exist
    if 'futures_data' in arctic.list_libraries():
        arctic.delete_library('futures_data')
        logger.info("Deleted existing futures_data library")
    
    futures_lib = arctic.create_library('futures_data')
    logger.info("Created futures_data library")
    
    for symbol in symbols:
        logger.info(f"Creating sample data for {symbol}")
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 2**32)  # Consistent data per symbol
        
        # Starting price based on symbol
        if symbol == '@ES#C':
            base_price = 4200
        elif symbol.startswith('@'):  # FX
            base_price = 1.0 + np.random.uniform(-0.5, 0.5)
        else:
            base_price = 100 + np.random.uniform(-50, 50)
        
        # Generate price series with realistic properties
        returns = np.random.normal(0, 0.001, len(date_range))  # 0.1% hourly vol
        
        # Add autocorrelation manually
        for i in range(1, len(returns)):
            returns[i] = returns[i] + 0.3 * returns[i-1]
        
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        
        prices = np.array(prices)
        
        # Create OHLCV data
        high = prices * (1 + np.abs(np.random.normal(0, 0.0005, len(prices))))
        low = prices * (1 - np.abs(np.random.normal(0, 0.0005, len(prices))))
        volume = np.random.lognormal(10, 0.5, len(prices))  # Realistic volume
        
        # Create DataFrame
        data = pd.DataFrame({
            'open': prices,
            'high': high,
            'low': low,
            'close': prices,
            'price': prices,  # Keep both for compatibility
            'volume': volume.astype(int)
        }, index=date_range)
        
        # Write to Arctic
        futures_lib.write(symbol, data)
        logger.info(f"Wrote {len(data)} records for {symbol}")
    
    logger.info(f"Sample data creation complete! Available symbols: {futures_lib.list_symbols()}")
    
    # Show sample info
    for symbol in symbols[:2]:
        info = futures_lib.read(symbol).data
        logger.info(f"{symbol}: {len(info)} records from {info.index.min()} to {info.index.max()}")

if __name__ == "__main__":
    create_sample_futures_data()
    print("\n🎉 Sample data created! Now you can run:")
    print("python main.py --target_symbol '@ES#C' --folds 2 --n_models 4 --max_features 20")
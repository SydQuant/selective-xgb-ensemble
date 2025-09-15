"""
Export JY (@JY#C) raw data from ArcticDB database.

This script connects to the ArcticDB database and exports raw OHLCV data
for the Japanese Yen futures (@JY#C) symbol to CSV format for analysis.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add data directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
data_dir = os.path.join(project_root, 'data')
sys.path.insert(0, data_dir)

from loaders import get_arcticdb_connection, get_symbol_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def export_jy_raw_data(output_dir=None, start_date=None, end_date=None, format='csv'):
    """
    Export JY (@JY#C) raw data from ArcticDB.

    Args:
        output_dir: Directory to save exported data (default: testing directory)
        start_date: Start date for data export (format: 'YYYY-MM-DD')
        end_date: End date for data export (format: 'YYYY-MM-DD')
        format: Export format ('csv', 'parquet', 'excel')

    Returns:
        str: Path to exported file
    """
    symbol = "@JY#C"

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Connect to database
        logger.info("Connecting to ArcticDB...")
        futures_lib = get_arcticdb_connection()

        # Get symbol info first
        logger.info(f"Getting info for symbol {symbol}...")
        symbol_info = get_symbol_info(symbol)
        if symbol_info:
            logger.info(f"Symbol: {symbol_info['symbol']}")
            logger.info(f"Records: {symbol_info['records']:,}")
            logger.info(f"Date range: {symbol_info['start_date']} to {symbol_info['end_date']}")
            logger.info(f"Columns: {symbol_info['columns']}")

        # Read raw data
        logger.info(f"Loading raw data for {symbol}...")
        versioned_item = futures_lib.read(symbol)
        df = versioned_item.data

        # Apply date filters if specified
        original_records = len(df)
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
            logger.info(f"Applied start_date filter: {start_date}")

        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]
            logger.info(f"Applied end_date filter: {end_date}")

        filtered_records = len(df)
        if original_records != filtered_records:
            logger.info(f"Filtered from {original_records:,} to {filtered_records:,} records")

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_suffix = ""
        if start_date or end_date:
            start_str = start_date.replace('-', '') if start_date else "start"
            end_str = end_date.replace('-', '') if end_date else "end"
            date_suffix = f"_{start_str}_to_{end_str}"

        filename = f"JY_raw_data_{timestamp}{date_suffix}.{format}"
        output_path = os.path.join(output_dir, filename)

        # Export data
        logger.info(f"Exporting {len(df):,} records to {output_path}...")

        if format.lower() == 'csv':
            df.to_csv(output_path)
        elif format.lower() == 'parquet':
            df.to_parquet(output_path)
        elif format.lower() == 'excel':
            df.to_excel(output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Show sample of exported data
        logger.info("\nSample of exported data:")
        logger.info(f"\nFirst 5 rows:\n{df.head()}")
        logger.info(f"\nLast 5 rows:\n{df.tail()}")
        logger.info(f"\nData info:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Index: {df.index.name} ({df.index.dtype})")
        logger.info(f"Columns: {list(df.columns)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

        # Basic statistics
        logger.info(f"\nBasic statistics:")
        logger.info(df.describe())

        logger.info(f"\nExport completed successfully!")
        logger.info(f"File saved: {output_path}")
        logger.info(f"File size: {os.path.getsize(output_path):,} bytes")

        return output_path

    except Exception as e:
        logger.error(f"Error exporting JY data: {e}")
        raise

def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Export JY (@JY#C) raw data from ArcticDB')
    parser.add_argument('--output-dir', type=str, help='Output directory (default: current directory)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD format)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD format)')
    parser.add_argument('--format', type=str, choices=['csv', 'parquet', 'excel'],
                       default='csv', help='Export format (default: csv)')

    args = parser.parse_args()

    try:
        output_path = export_jy_raw_data(
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            format=args.format
        )
        print(f"\nSuccess! Data exported to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
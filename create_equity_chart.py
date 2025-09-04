#!/usr/bin/env python3
"""
Simple equity chart generator for XGBoost trading results.
Creates and saves equity curve visualization from oos_timeseries.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os

def create_equity_chart(input_file="artifacts/oos_timeseries.csv", 
                       output_dir="artifacts/charts",
                       symbol="@ES#C"):
    """Create and save equity chart from out-of-sample results."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return None
    
    df = pd.read_csv(input_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
    
    # Plot equity curve
    ax1.plot(df['datetime'], df['equity'], 'b-', linewidth=1.5, label='Equity Curve')
    ax1.set_title(f'{symbol} - 10-fold CV Extended History (2015-2025)\nEquity Curve', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Format x-axis for top chart
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
    
    # Plot signals
    ax2.plot(df['datetime'], df['signal'], 'r-', alpha=0.7, linewidth=0.5, label='Signal')
    ax2.set_ylabel('Signal', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(-1.1, 1.1)
    
    # Format x-axis for bottom chart
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_minor_locator(mdates.MonthLocator([1, 4, 7, 10]))
    
    # Rotate x-axis labels
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"equity_chart_{symbol.replace('#', '').replace('@', '')}_{timestamp}.png")
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Equity chart saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    # Create chart for test 1d results
    chart_file = create_equity_chart()
    if chart_file:
        print("Chart generation completed successfully!")
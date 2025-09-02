#!/usr/bin/env python3
"""
Comprehensive results analysis for multi-symbol backtest
"""

import os
import pandas as pd
import numpy as np
from glob import glob
import json

def analyze_symbol_results():
    """Analyze all completed symbol results"""
    
    # Look for CSV files in artifacts/results
    result_files = glob("artifacts/results/*_performance.csv")
    
    if not result_files:
        print("No result files found in artifacts/results/")
        return
        
    results = []
    
    for file_path in result_files:
        try:
            # Extract symbol from filename
            filename = os.path.basename(file_path)
            symbol = filename.split('_')[0]
            
            # Read CSV
            df = pd.read_csv(file_path)
            
            if len(df) > 0:
                result = df.iloc[-1].to_dict()  # Last row has final results
                result['symbol'] = symbol
                result['file_path'] = file_path
                results.append(result)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
    if not results:
        print("No valid results found")
        return
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by Information Ratio descending
    if 'IR_Test' in results_df.columns:
        results_df = results_df.sort_values('IR_Test', ascending=False)
    
    print("=" * 80)
    print("📊 MULTI-SYMBOL BACKTEST RESULTS (2020-07-01 to 2024-08-01)")
    print("=" * 80)
    
    # Summary table
    columns_to_show = ['symbol', 'IR_Test', 'Hit_Rate_Test', 'Return_Test', 'Volatility_Test', 'Max_DD_Test']
    available_cols = [col for col in columns_to_show if col in results_df.columns]
    
    if available_cols:
        display_df = results_df[available_cols].copy()
        
        # Format percentages
        pct_cols = [col for col in display_df.columns if 'Return' in col or 'Hit_Rate' in col or 'DD' in col or 'Volatility' in col]
        for col in pct_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
                
        # Format IR
        if 'IR_Test' in display_df.columns:
            display_df['IR_Test'] = display_df['IR_Test'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
            
        print(display_df.to_string(index=False))
        
    print("\n" + "=" * 80)
    
    # Asset class analysis
    asset_classes = {
        'Equities': ['@ES#C', '@NQ#C', '@YM#C', '@RTY#C'],
        'Bonds': ['@TY#C', '@US#C', '@FV#C', '@TU#C'], 
        'FX': ['@EU#C', '@BP#C', '@JY#C', '@AD#C'],
        'Commodities': ['@C#C', '@S#C', '@W#C', '@LE#C', '@GF#C', '@HE#C', '@BO#C', '@SM#C'],
        'Volatility': ['@VX#C'],
        'Softs': ['@KC#C', '@SB#C', '@CC#C', '@CT#C']
    }
    
    print("📈 ASSET CLASS PERFORMANCE")
    print("-" * 40)
    
    for asset_class, symbols in asset_classes.items():
        class_results = results_df[results_df['symbol'].isin(symbols)]
        
        if len(class_results) > 0:
            avg_ir = class_results['IR_Test'].mean() if 'IR_Test' in class_results.columns else None
            avg_hit_rate = class_results['Hit_Rate_Test'].mean() if 'Hit_Rate_Test' in class_results.columns else None
            
            print(f"{asset_class:12} ({len(class_results):2d} symbols): ", end="")
            
            if avg_ir is not None:
                print(f"Avg IR: {avg_ir:.3f}", end="")
            if avg_hit_rate is not None:
                print(f", Avg Hit Rate: {avg_hit_rate:.1%}", end="")
                
            print()
            
    # Top performers
    if 'IR_Test' in results_df.columns:
        print("\n🏆 TOP 5 PERFORMERS (by Information Ratio)")
        print("-" * 50)
        top_5 = results_df.head(5)
        for _, row in top_5.iterrows():
            symbol = row['symbol']
            ir = row.get('IR_Test', 0)
            hit_rate = row.get('Hit_Rate_Test', 0)
            returns = row.get('Return_Test', 0)
            print(f"{symbol:8} IR: {ir:6.3f}  Hit: {hit_rate:5.1%}  Return: {returns:6.1%}")
            
    print("\n" + "=" * 80)
    
    return results_df

if __name__ == "__main__":
    analyze_symbol_results()
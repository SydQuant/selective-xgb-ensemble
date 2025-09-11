#!/usr/bin/env python3
"""
XGBoost Log Parser - Comprehensive Analysis Tool
Parses all XGBoost comparison logs and extracts configuration parameters and results
"""

import os
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class XGBoostLogParser:
    def __init__(self):
        self.log_directories = [
            "xgb_compare/results/logs",
            "xgb_compare/in_progress/logs"
        ]
        self.results = []
        
    def find_all_logs(self) -> List[str]:
        """Find all log files in the specified directories"""
        log_files = []
        for log_dir in self.log_directories:
            if os.path.exists(log_dir):
                for root, dirs, files in os.walk(log_dir):
                    for file in files:
                        if file.endswith('.log'):
                            log_files.append(os.path.join(root, file))
        return sorted(log_files)
    
    def parse_single_log(self, log_path: str) -> Optional[Dict]:
        """Parse a single log file and extract all relevant information"""
        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = {}
            
            # Extract timestamp from filename only
            filename = os.path.basename(log_path)
            timestamp_match = re.search(r'(\d{8}_\d{6})', filename)
            if timestamp_match:
                result['timestamp'] = timestamp_match.group(1)
            else:
                result['timestamp'] = 'Unknown'
            
            # Configuration parsing
            result.update(self._parse_configuration(content))
            
            # Results parsing
            result.update(self._parse_results(content))
            
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing {log_path}: {e}")
            return None
    
    def _parse_configuration(self, content: str) -> Dict:
        """Extract configuration parameters from log content"""
        config = {}
        
        # Target Symbol
        symbol_match = re.search(r'Target Symbol: (.+)', content)
        config['symbol'] = symbol_match.group(1).strip() if symbol_match else 'Unknown'
        
        # Date Range
        date_match = re.search(r'Date Range: (.+) to (.+)', content)
        if date_match:
            config['start_date'] = date_match.group(1).strip()
            config['end_date'] = date_match.group(2).strip()
        else:
            config['start_date'] = config['end_date'] = 'Unknown'
        
        # Models and Type
        models_match = re.search(r'Models: (\d+), Type: (\w+)', content)
        if models_match:
            config['n_models'] = int(models_match.group(1))
            config['xgb_type'] = models_match.group(2)
        else:
            config['n_models'] = config['xgb_type'] = 'Unknown'
        
        # Folds
        folds_match = re.search(r'Folds: (\d+)', content)
        config['n_folds'] = int(folds_match.group(1)) if folds_match else 'Unknown'
        
        # Features
        features_match = re.search(r'Max Features: (.+)', content)
        if features_match:
            feat_str = features_match.group(1).strip()
            if 'Limited' in feat_str:
                config['max_features'] = int(re.search(r'(\d+)', feat_str).group(1))
                config['feature_limitation'] = 'Limited'
            elif 'All after' in feat_str:
                config['max_features'] = 'All'
                config['feature_limitation'] = 'All after cluster reduction'
            else:
                config['max_features'] = feat_str
                config['feature_limitation'] = 'Unknown'
        else:
            config['max_features'] = config['feature_limitation'] = 'Unknown'
        
        # Actual selected features
        selected_match = re.search(r'Selected (\d+) features', content)
        config['selected_features'] = int(selected_match.group(1)) if selected_match else 'Unknown'
        
        # Signal Type
        signal_match = re.search(r'Signal Type: (.+)', content)
        config['signal_type'] = signal_match.group(1).strip() if signal_match else 'Unknown'
        
        # Cross-validation Type
        cv_match = re.search(r'Cross-validation: (.+)', content)
        if cv_match:
            cv_str = cv_match.group(1).strip()
            if 'Rolling' in cv_str:
                config['cv_type'] = 'Rolling'
                rolling_days_match = re.search(r'Rolling (\d+) days', cv_str)
                config['rolling_days'] = int(rolling_days_match.group(1)) if rolling_days_match else 'Unknown'
            elif 'Expanding' in cv_str:
                config['cv_type'] = 'Expanding'
                config['rolling_days'] = None
            else:
                config['cv_type'] = cv_str
                config['rolling_days'] = 'Unknown'
        else:
            config['cv_type'] = config['rolling_days'] = 'Unknown'
        
        # Production Configuration
        prod_match = re.search(r'Production: (.+)', content)
        if prod_match:
            prod_str = prod_match.group(1).strip()
            
            # Cutoff
            cutoff_match = re.search(r'Cutoff=([0-9.]+)', prod_str)
            config['cutoff_fraction'] = float(cutoff_match.group(1)) if cutoff_match else 'Unknown'
            
            # Top N
            topn_match = re.search(r'Top N=(\d+)', prod_str)
            config['top_n'] = int(topn_match.group(1)) if topn_match else 'Unknown'
            
            # Q-Metric
            qmetric_match = re.search(r'Q-Metric=(\w+)', prod_str)
            config['q_metric'] = qmetric_match.group(1) if qmetric_match else 'Unknown'
        else:
            config['cutoff_fraction'] = config['top_n'] = config['q_metric'] = 'Unknown'
        
        # Combined Q-Score configuration
        combined_match = re.search(r'Combined Q-Score: Sharpe=([0-9.]+), Hit=([0-9.]+)', content)
        if combined_match:
            config['sharpe_weight'] = float(combined_match.group(1))
            config['hit_weight'] = float(combined_match.group(2))
        else:
            config['sharpe_weight'] = config['hit_weight'] = 'Unknown'
        
        # EWMA and Quality parameters
        ewma_match = re.search(r'EWMA Alpha: ([0-9.]+), Quality Halflife: (\d+) days', content)
        if ewma_match:
            config['ewma_alpha'] = float(ewma_match.group(1))
            config['quality_halflife'] = int(ewma_match.group(2))
        else:
            config['ewma_alpha'] = config['quality_halflife'] = 'Unknown'
        
        
        return config
    
    def _parse_results(self, content: str) -> Dict:
        """Extract results from log content"""
        results = {}
        
        # Final Results (most important)
        final_match = re.search(r'Final Results: Sharpe=([0-9.-]+) \| Hit=([0-9.%]+) \| Return=([0-9.%-]+)', content)
        if final_match:
            results['prod_sharpe'] = float(final_match.group(1))
            results['prod_hit'] = float(final_match.group(2).rstrip('%')) / 100
            results['prod_return'] = final_match.group(3)
        else:
            results['prod_sharpe'] = results['prod_hit'] = results['prod_return'] = 'Unknown'
        
        # Training Period Results (extract from fold summaries)
        fold_summaries = re.findall(r'Mean OOS Sharpe: ([0-9.-]+), Mean Hit: ([0-9.]+)', content)
        if fold_summaries:
            sharpe_values = [float(s[0]) for s in fold_summaries]
            hit_values = [float(s[1]) for s in fold_summaries]
            results['train_sharpe_mean'] = sum(sharpe_values) / len(sharpe_values)
            results['train_hit_mean'] = sum(hit_values) / len(hit_values)
            results['train_sharpe_max'] = max(sharpe_values)
            results['train_hit_max'] = max(hit_values)
        else:
            results['train_sharpe_mean'] = results['train_hit_mean'] = 'Unknown'
            results['train_sharpe_max'] = results['train_hit_max'] = 'Unknown'
        
        # Full Timeline Backtest Results (if available)
        backtest_matches = re.findall(r'Fold \d+ \(Production\): Sharpe=([0-9.-]+), Hit=([0-9.%]+)', content)
        if backtest_matches:
            backtest_sharpe = [float(m[0]) for m in backtest_matches]
            backtest_hit = [float(m[1].rstrip('%')) / 100 for m in backtest_matches]
            results['full_timeline_sharpe_mean'] = sum(backtest_sharpe) / len(backtest_sharpe)
            results['full_timeline_hit_mean'] = sum(backtest_hit) / len(backtest_hit)
        else:
            results['full_timeline_sharpe_mean'] = results['full_timeline_hit_mean'] = 'Unknown'
        
        # Status
        if 'ANALYSIS COMPLETED SUCCESSFULLY!' in content:
            results['status'] = 'Completed'
        elif 'Final Results:' in content:
            results['status'] = 'Completed'
        else:
            results['status'] = 'In Progress'
        
        # GPU Usage
        results['gpu_used'] = 'GPU' if 'GPU available: True' in content else 'CPU'
        
        return results
    
    
    def parse_all_logs(self) -> pd.DataFrame:
        """Parse all logs and return results as DataFrame"""
        log_files = self.find_all_logs()
        logger.info(f"Found {len(log_files)} log files to parse")
        
        results = []
        for i, log_path in enumerate(log_files, 1):
            logger.info(f"Parsing {i}/{len(log_files)}: {os.path.basename(log_path)}")
            result = self.parse_single_log(log_path)
            if result:
                results.append(result)
        
        logger.info(f"Successfully parsed {len(results)} log files")
        
        if results:
            df = pd.DataFrame(results)
            
            # Convert prod_sharpe to numeric for sorting
            df['prod_sharpe_numeric'] = pd.to_numeric(df['prod_sharpe'], errors='coerce')
            
            # Sort by symbol, then by production Sharpe (descending)
            df = df.sort_values([
                'symbol', 
                'prod_sharpe_numeric'
            ], ascending=[True, False])
            
            # Drop the temporary column
            df = df.drop('prod_sharpe_numeric', axis=1)
            
            # Reorder columns as requested
            desired_columns = [
                'timestamp', 'symbol', 'prod_sharpe', 'prod_hit', 'prod_return',
                'n_models', 'top_n', 'n_folds', 'max_features', 'xgb_type', 
                'signal_type', 'q_metric', 'cv_type', 'rolling_days', 
                'cutoff_fraction', 'ewma_alpha', 'sharpe_weight', 'hit_weight',
                'train_sharpe_mean', 'train_hit_mean', 'full_timeline_sharpe_mean', 
                'full_timeline_hit_mean', 'status'
            ]
            
            # Only include columns that exist in the dataframe
            available_columns = [col for col in desired_columns if col in df.columns]
            df = df[available_columns]
            
            return df
        else:
            return pd.DataFrame()
    
    def save_results(self, df: pd.DataFrame, csv_path: str = "xgb_log_analysis.csv", 
                    md_path: str = "XGBoost_Log_Analysis_Summary.md"):
        """Save results to CSV and Markdown files"""
        # Save to CSV
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV results to: {csv_path}")
        
        # Generate Markdown summary
        self._generate_markdown_summary(df, md_path)
        logger.info(f"Saved Markdown summary to: {md_path}")
    
    def _generate_markdown_summary(self, df: pd.DataFrame, md_path: str):
        """Generate comprehensive Markdown summary"""
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# XGBoost Log Analysis Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Total Logs Analyzed:** {len(df)}\n\n")
            
            # Summary statistics by symbol
            f.write("## Summary by Symbol\n\n")
            for symbol in sorted(df['symbol'].unique()):
                if symbol == 'Unknown':
                    continue
                symbol_data = df[df['symbol'] == symbol]
                completed = symbol_data[symbol_data['status'] == 'Completed']
                
                f.write(f"### {symbol}\n")
                f.write(f"- **Total Runs:** {len(symbol_data)}\n")
                f.write(f"- **Completed:** {len(completed)}\n")
                f.write(f"- **In Progress:** {len(symbol_data) - len(completed)}\n")
                
                if len(completed) > 0:
                    valid_sharpe = completed[pd.to_numeric(completed['prod_sharpe'], errors='coerce').notnull()]
                    if len(valid_sharpe) > 0:
                        best_sharpe = valid_sharpe.loc[pd.to_numeric(valid_sharpe['prod_sharpe']).idxmax()]
                        f.write(f"- **Best Production Sharpe:** {best_sharpe['prod_sharpe']} ({best_sharpe['timestamp']})\n")
                
                f.write("\n")
            
            # Top performing configurations
            f.write("## Top 20 Performing Configurations (by Production Sharpe)\n\n")
            valid_results = df[
                (df['status'] == 'Completed') & 
                (pd.to_numeric(df['prod_sharpe'], errors='coerce').notnull())
            ]
            
            if len(valid_results) > 0:
                # Create numeric column for sorting
                valid_results = valid_results.copy()
                valid_results['prod_sharpe_numeric'] = pd.to_numeric(valid_results['prod_sharpe'], errors='coerce')
                top_20 = valid_results.nlargest(20, 'prod_sharpe_numeric')
                
                f.write("| Rank | Symbol | Prod Sharpe | Prod Hit | XGB Type | Models | Folds | CV Type | Signal | Q-Metric | Timestamp |\n")
                f.write("|------|--------|-------------|----------|----------|---------|--------|---------|---------|-----------|----------|\n")
                
                for i, (_, row) in enumerate(top_20.iterrows(), 1):
                    cv_info = f"{row['cv_type']}"
                    if row.get('rolling_days') and pd.notna(row['rolling_days']):
                        cv_info += f" ({row['rolling_days']}d)"
                    
                    f.write(f"| {i} | {row['symbol']} | {row['prod_sharpe']:.3f} | {row['prod_hit']:.1%} | "
                           f"{row['xgb_type']} | {row['n_models']} | {row['n_folds']} | {cv_info} | "
                           f"{row['signal_type']} | {row['q_metric']} | {row['timestamp']} |\n")
            
            # Configuration analysis
            f.write("\n\n## Configuration Analysis\n\n")
            
            # XGB Type performance
            f.write("### XGB Type Performance\n\n")
            valid_results = df[
                (df['status'] == 'Completed') & 
                (pd.to_numeric(df['prod_sharpe'], errors='coerce').notnull())
            ]
            
            if len(valid_results) > 0:
                xgb_perf = valid_results.groupby('xgb_type').agg({
                    'prod_sharpe': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                    'prod_hit': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                    'timestamp': 'count'
                }).round(3)
                xgb_perf.columns = ['Avg Sharpe', 'Avg Hit Rate', 'Count']
                f.write(xgb_perf.to_markdown())
                f.write("\n\n")
            
            # Signal Type performance
            f.write("### Signal Type Performance\n\n")
            if len(valid_results) > 0:
                signal_perf = valid_results.groupby('signal_type').agg({
                    'prod_sharpe': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                    'prod_hit': lambda x: pd.to_numeric(x, errors='coerce').mean(),
                    'timestamp': 'count'
                }).round(3)
                signal_perf.columns = ['Avg Sharpe', 'Avg Hit Rate', 'Count']
                f.write(signal_perf.to_markdown())
                f.write("\n\n")
            
            # Detailed results table
            f.write("## Detailed Results Table\n\n")
            f.write("### Key Configuration Parameters and Results\n\n")
            
            # Select key columns for the summary table
            summary_cols = [
                'symbol', 'timestamp', 'status', 'prod_sharpe', 'prod_hit', 'prod_return',
                'n_models', 'n_folds', 'xgb_type', 'cv_type', 'rolling_days', 'signal_type', 
                'q_metric', 'selected_features'
            ]
            
            available_cols = [col for col in summary_cols if col in df.columns]
            summary_df = df[available_cols].copy()
            
            # Format the data for better readability
            if 'prod_hit' in summary_df.columns:
                summary_df['prod_hit'] = summary_df['prod_hit'].apply(
                    lambda x: f"{float(x):.1%}" if pd.notna(x) and x != 'Unknown' else str(x)
                )
            
            f.write(summary_df.to_markdown(index=False))
            f.write("\n\n")


def main():
    """Main execution function"""
    parser = XGBoostLogParser()
    
    # Parse all logs
    df = parser.parse_all_logs()
    
    if len(df) > 0:
        # Save results
        parser.save_results(df)
        
        # Print summary
        print(f"\n[SUCCESS] Successfully parsed {len(df)} log files")
        print(f"[CSV] Results saved to: xgb_log_analysis.csv")
        print(f"[MD] Summary saved to: XGBoost_Log_Analysis_Summary.md")
        
        # Show statistics
        completed = df[df['status'] == 'Completed']
        in_progress = df[df['status'] == 'In Progress']
        print(f"\n[STATUS] Completed: {len(completed)}, In Progress: {len(in_progress)}")
        
        # Show top 5 results
        if len(completed) > 0:
            valid_results = completed[pd.to_numeric(completed['prod_sharpe'], errors='coerce').notnull()].copy()
            if len(valid_results) > 0:
                print(f"\n[TOP PERFORMERS] Top 5 Production Sharpe Results:")
                valid_results['prod_sharpe_numeric'] = pd.to_numeric(valid_results['prod_sharpe'], errors='coerce')
                top_5 = valid_results.nlargest(5, 'prod_sharpe_numeric')
                
                print(f"{'Rank':<4} {'Symbol':<8} {'Sharpe':<7} {'Hit Rate':<8} {'Models':<7} {'Folds':<6} {'CV Type':<12} {'Timestamp'}")
                print("-" * 80)
                
                for i, (_, row) in enumerate(top_5.iterrows(), 1):
                    cv_type = f"{row['cv_type']}"
                    if row.get('rolling_days') and pd.notna(row['rolling_days']):
                        cv_type += f"({row['rolling_days']}d)"
                    
                    print(f"{i:<4} {row['symbol']:<8} {float(row['prod_sharpe']):<7.3f} {float(row['prod_hit']):<7.1%} "
                          f"{row['n_models']:<7} {row['n_folds']:<6} {cv_type:<12} {row['timestamp']}")
        
        # Show symbol breakdown
        print(f"\n[SYMBOL BREAKDOWN]")
        symbol_stats = df.groupby('symbol').agg({
            'status': ['count', lambda x: sum(x == 'Completed')],
            'prod_sharpe': lambda x: pd.to_numeric(x, errors='coerce').max()
        }).round(3)
        
        symbol_stats.columns = ['Total', 'Completed', 'Best_Sharpe']
        print(symbol_stats.to_string())
        
    else:
        print("[ERROR] No log files found or parsed successfully")


if __name__ == "__main__":
    main()
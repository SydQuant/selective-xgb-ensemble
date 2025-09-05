#!/usr/bin/env python3
"""
Horse Race Individual Quality Testing Script
Tests @ES#C over 10-year period using the baseline configuration.

This script demonstrates how to use the horse_race_individual_quality.py framework
to evaluate different driver selection methods based on stability and quality momentum.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import List
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import horse race framework
from ensemble.horse_race_individual_quality import (
    rolling_horse_race_individual_quality, Driver, MetricConfig,
    metric_sharpe, metric_adj_sharpe, metric_hit_rate
)

# Import data preparation and XGBoost utilities
from data.data_utils_simple import prepare_real_data_simple
from model.xgb_drivers import generate_xgb_specs
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_xgb_drivers(specs: List[dict]) -> List[Driver]:
    """Create XGBoost drivers from specifications."""
    drivers = []
    for spec in specs:
        xgb_model = XGBRegressor(**spec)
        drivers.append(Driver(xgb_model))
    return drivers

def main():
    """Run horse race test on @ES#C with optimal parameters."""
    
    # Load baseline configuration
    config_path = "configs/test_objectives_corrected.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Override target symbol and period
    target_symbol = "@ES#C"
    start_date = "2015-01-01"
    end_date = "2025-08-01"
    
    logger.info(f"Testing {target_symbol} from {start_date} to {end_date}")
    
    # Prepare data using the existing pipeline
    logger.info("Loading and preparing data...")
    df = prepare_real_data_simple(
        target_symbol=target_symbol,
        symbols=config['symbols'].split(','),
        start_date=start_date,
        end_date=end_date,
        signal_hour=config.get('signal_hour', 12),
        n_hours=config.get('n_hours', 24)
    )
    
    # Extract features and target
    target_col = f"{target_symbol}_target_return"
    y = df[target_col]
    X = df.drop(columns=[col for col in df.columns if col.endswith('_target_return')])
    
    logger.info(f"Data prepared: X.shape={X.shape}, y.shape={y.shape}")
    logger.info(f"Date range: {X.index.min()} to {X.index.max()}")
    
    # Generate diverse XGBoost specifications (reduced for testing)
    n_models = 25  # Reduced from 75 for faster testing
    specs = generate_xgb_specs(n_models=n_models, seed=42)
    drivers = create_xgb_drivers(specs)
    logger.info(f"Created {len(drivers)} XGBoost drivers")
    
    # Define metric configurations for horse race
    metrics = [
        # Standard Sharpe ratio with stability weighting
        MetricConfig(
            name="Sharpe",
            fn=metric_sharpe,
            kwargs={"costs_per_turn": 0.0001},  # Small transaction cost
            alpha=1.0,           # Full weight on validation performance
            lam_gap=0.3,         # Moderate penalty for train-val gap
            relative_gap=False,  # Absolute gap penalty
            eta_quality=0.0      # No quality momentum initially
        ),
        
        # Adjusted Sharpe with turnover penalty  
        MetricConfig(
            name="AdjSharpe",
            fn=metric_adj_sharpe,
            kwargs={"lambda_to": 0.05},  # Turnover penalty from config
            alpha=1.0,
            lam_gap=0.4,         # Higher gap penalty (more stability focus)
            relative_gap=False,
            eta_quality=0.0
        ),
        
        # Hit rate with quality momentum
        MetricConfig(
            name="HitRate",
            fn=metric_hit_rate,
            kwargs={},
            alpha=1.0,
            lam_gap=0.2,         # Lower gap penalty (hit rate is more stable)
            relative_gap=False,
            eta_quality=0.1      # Include quality momentum
        ),
        
        # Sharpe with strong quality momentum (EWMA of past performance)
        MetricConfig(
            name="Sharpe_QualMom",
            fn=metric_sharpe,
            kwargs={"costs_per_turn": 0.0001},
            alpha=0.8,           # Slightly reduce validation weight
            lam_gap=0.3,
            relative_gap=False,
            eta_quality=0.3      # Strong quality momentum
        ),
    ]
    
    # Optimal parameters based on financial data characteristics
    # 10-year period (2015-2025) ≈ 2600 trading days
    start_train = 750    # ~3 years initial training (minimum for XGBoost stability)  
    step = 21           # ~1 month steps (21 trading days)
    horizon = 21        # ~1 month forward testing horizon
    inner_val_frac = 0.2 # 20% of training data for inner validation
    costs_per_turn_backtest = 0.0001  # Realistic transaction costs
    quality_halflife = 63  # ~3 months quality memory (63 trading days)
    
    logger.info("Horse race parameters:")
    logger.info(f"  start_train: {start_train} days (~{start_train/252:.1f} years)")
    logger.info(f"  step: {step} days (~{step} trading days per step)")  
    logger.info(f"  horizon: {horizon} days (forward testing period)")
    logger.info(f"  inner_val_frac: {inner_val_frac} ({inner_val_frac*100}% for validation)")
    logger.info(f"  quality_halflife: {quality_halflife} days (~{quality_halflife/21:.1f} months)")
    logger.info(f"  Expected windows: ~{(len(X) - start_train) // step}")
    
    # Run horse race
    logger.info("Starting horse race evaluation...")
    summary_df, window_df, details = rolling_horse_race_individual_quality(
        X=X,
        y=y,
        drivers=drivers,
        metrics=metrics,
        start_train=start_train,
        step=step,
        horizon=horizon,
        inner_val_frac=inner_val_frac,
        costs_per_turn_backtest=costs_per_turn_backtest,
        quality_halflife=quality_halflife
    )
    
    # Display results
    logger.info("Horse Race Results Summary:")
    print("\n" + "="*80)
    print("HORSE RACE INDIVIDUAL QUALITY - SUMMARY RESULTS")
    print("="*80)
    print(summary_df.round(4))
    
    print("\n" + "="*60)
    print("DETAILED ANALYSIS BY METRIC")
    print("="*60)
    
    for metric_name in summary_df.index:
        print(f"\n{metric_name}:")
        print(f"  Mean Sharpe: {summary_df.loc[metric_name, 'Sharpe_mean']:.3f}")
        print(f"  Median Sharpe: {summary_df.loc[metric_name, 'Sharpe_median']:.3f}")
        print(f"  Mean Hit Rate: {summary_df.loc[metric_name, 'HitRate_mean']:.3f}")
        print(f"  Predictive Correlation: {summary_df.loc[metric_name, 'PredCorr_mean']:.3f}")
        
        # Show driver selection frequency
        picks = details[metric_name]['picks']
        pick_counts = pd.Series(picks).value_counts().head(5)
        print(f"  Top 5 selected drivers: {dict(pick_counts)}")
    
    # Save results
    os.makedirs("artifacts/horse_race", exist_ok=True)
    
    summary_path = "artifacts/horse_race/individual_quality_summary.csv"
    window_path = "artifacts/horse_race/individual_quality_windows.csv"
    
    summary_df.to_csv(summary_path)
    window_df.to_csv(window_path)
    
    logger.info(f"Results saved:")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Windows: {window_path}")
    
    # Performance ranking
    print("\n" + "="*60)
    print("PERFORMANCE RANKING (by Mean Sharpe)")
    print("="*60)
    ranking = summary_df.sort_values('Sharpe_mean', ascending=False)
    for i, (metric, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {metric}: {row['Sharpe_mean']:.3f} Sharpe, "
              f"{row['HitRate_mean']:.3f} HitRate, "
              f"{row['PredCorr_mean']:.3f} PredCorr")

if __name__ == "__main__":
    main()
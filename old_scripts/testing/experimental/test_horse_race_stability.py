#!/usr/bin/env python3
"""
Horse Race Stability Testing Script
Tests ensemble stability approach using @ES#C over 10-year period.

This script demonstrates the horse_race_stability.py framework which:
- Selects top-k drivers per metric (ensemble approach)
- Uses equal-weight combination of selected drivers
- Focuses on ensemble stability rather than individual driver quality
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

# Import horse race stability framework
from ensemble.horse_race_stability import (
    rolling_horse_race, Driver, MetricConfig,
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
    """Run horse race stability test on @ES#C."""
    
    # Load baseline configuration
    config_path = "configs/production_full_system.yaml"
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Test parameters (balanced for speed vs data sufficiency)
    target_symbol = "@ES#C"
    start_date = "2020-01-01"  # 4.5-year period for sufficient data
    end_date = "2024-08-01"
    
    logger.info(f"Testing {target_symbol} from {start_date} to {end_date}")
    
    # Prepare data
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
    
    # Generate XGBoost drivers (smaller for speed)
    n_models = 20  # Reduced for faster testing
    specs = generate_xgb_specs(n_models=n_models, seed=42)
    drivers = create_xgb_drivers(specs)
    logger.info(f"Created {len(drivers)} XGBoost drivers")
    
    # Define metric configurations for ensemble stability testing
    metrics = [
        # Sharpe-based ensemble (top-5 drivers)
        MetricConfig(
            name="Sharpe_Ensemble",
            fn=metric_sharpe,
            kwargs={"costs_per_turn": 0.0001},
            alpha=1.0,           # Full weight on validation performance
            lam_gap=0.3,         # Moderate stability penalty
            relative_gap=False,
            top_k=5,             # Select top 5 drivers for ensemble
            eta_quality=0.0      # No quality momentum initially
        ),
        
        # Conservative Sharpe ensemble (top-3 drivers, higher stability focus)
        MetricConfig(
            name="Sharpe_Conservative",
            fn=metric_sharpe,
            kwargs={"costs_per_turn": 0.0001},
            alpha=0.8,           # Slightly lower validation weight
            lam_gap=0.5,         # Higher stability penalty
            relative_gap=False,
            top_k=3,             # More conservative selection
            eta_quality=0.0
        ),
        
        # Adjusted Sharpe ensemble with turnover penalty
        MetricConfig(
            name="adj_sharpe_Ensemble",
            fn=metric_adj_sharpe,
            kwargs={"costs_per_turn": 0.0001, "lambda_to": 0.05},
            alpha=1.0,
            lam_gap=0.4,
            relative_gap=False,
            top_k=4,             # Balanced ensemble size
            eta_quality=0.0
        ),
        
        # Hit Rate ensemble (testing directional accuracy)
        MetricConfig(
            name="hit_Ensemble", 
            fn=metric_hit_rate,
            kwargs={},
            alpha=1.0,
            lam_gap=0.2,         # Lower penalty (hit rate more stable)
            relative_gap=False,
            top_k=6,             # Larger ensemble for stability
            eta_quality=0.0
        ),
        
        # Sharpe with quality momentum ensemble
        MetricConfig(
            name="Sharpe_QualMom_Ensemble",
            fn=metric_sharpe,
            kwargs={"costs_per_turn": 0.0001},
            alpha=0.9,           # Balance validation and quality memory
            lam_gap=0.3,
            relative_gap=False,
            top_k=5,
            eta_quality=0.2      # Moderate quality momentum
        ),
    ]
    
    # Adjusted parameters for testing dataset size
    start_train = 500    # ~2 years initial training (reduced for smaller dataset)
    step = 21           # ~1 month rebalancing
    horizon = 21        # ~1 month forward testing
    inner_val_frac = 0.2 # 20% inner validation
    costs_per_turn_backtest = 0.0001
    quality_halflife = 63  # ~3 months quality memory
    
    logger.info("Horse race stability parameters:")
    logger.info(f"  start_train: {start_train} days")
    logger.info(f"  step: {step} days")
    logger.info(f"  horizon: {horizon} days")
    logger.info(f"  inner_val_frac: {inner_val_frac}")
    logger.info(f"  quality_halflife: {quality_halflife} days")
    logger.info(f"  Expected windows: ~{(len(X) - start_train) // step}")
    
    for i, metric in enumerate(metrics):
        logger.info(f"  Metric {i+1}: {metric.name} (top_k={metric.top_k}, lam_gap={metric.lam_gap})")
    
    # Run stability horse race
    logger.info("Starting horse race stability evaluation...")
    summary_df, window_df, details = rolling_horse_race(
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
    logger.info("Horse Race Stability Results Summary:")
    print("\n" + "="*80)
    print("HORSE RACE STABILITY - ENSEMBLE PERFORMANCE RESULTS")
    print("="*80)
    print(summary_df.round(4))
    
    print("\n" + "="*70)
    print("DETAILED ANALYSIS BY ENSEMBLE METHOD")
    print("="*70)
    
    for metric_name in summary_df.index:
        print(f"\n{metric_name}:")
        print(f"  Mean Sharpe: {summary_df.loc[metric_name, 'Sharpe_mean']:.3f}")
        print(f"  Median Sharpe: {summary_df.loc[metric_name, 'Sharpe_median']:.3f}")
        print(f"  Mean Hit Rate: {summary_df.loc[metric_name, 'HitRate_mean']:.3f}")
        print(f"  Predictive Correlation: {summary_df.loc[metric_name, 'PredCorr_mean']:.3f}")
        print(f"  Annual Return: {summary_df.loc[metric_name, 'AnnRet_mean']:.2f}%")
        print(f"  Annual Volatility: {summary_df.loc[metric_name, 'AnnVol_mean']:.2f}%")
        
        # Show ensemble composition analysis
        picks_history = details[metric_name]['picks']
        all_picks = []
        for window_picks in picks_history:
            all_picks.extend(window_picks)
        pick_counts = pd.Series(all_picks).value_counts().head(10)
        print(f"  Top 10 selected drivers (frequency): {dict(pick_counts)}")
    
    # Save results
    os.makedirs("artifacts/horse_race", exist_ok=True)
    
    summary_path = "artifacts/horse_race/stability_summary.csv"
    window_path = "artifacts/horse_race/stability_windows.csv"
    
    summary_df.to_csv(summary_path)
    window_df.to_csv(window_path)
    
    logger.info(f"Results saved:")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Windows: {window_path}")
    
    # Performance ranking
    print("\n" + "="*70)
    print("ENSEMBLE PERFORMANCE RANKING (by Mean Sharpe)")
    print("="*70)
    ranking = summary_df.sort_values('Sharpe_mean', ascending=False)
    for i, (metric, row) in enumerate(ranking.iterrows(), 1):
        print(f"{i}. {metric}:")
        print(f"   Sharpe: {row['Sharpe_mean']:.3f}, Hit Rate: {row['HitRate_mean']:.3f}")
        print(f"   Return: {row['AnnRet_mean']:.2f}%, Vol: {row['AnnVol_mean']:.2f}%")
        print(f"   Predictive Power: {row['PredCorr_mean']:.3f}")
    
    # Compare with individual quality results (if available)
    indiv_path = "artifacts/horse_race/individual_quality_summary.csv"
    if os.path.exists(indiv_path):
        print("\n" + "="*70)
        print("COMPARISON: STABILITY vs INDIVIDUAL QUALITY")
        print("="*70)
        
        indiv_df = pd.read_csv(indiv_path, index_col=0)
        
        print("Best Stability Ensemble vs Best Individual Selection:")
        best_stability = ranking.iloc[0]
        best_individual = indiv_df.sort_values('Sharpe_mean', ascending=False).iloc[0]
        
        print(f"Stability ({best_stability.name}): {best_stability['Sharpe_mean']:.3f} Sharpe")
        print(f"Individual ({best_individual.name}): {best_individual['Sharpe_mean']:.3f} Sharpe")
        
        improvement = best_stability['Sharpe_mean'] - best_individual['Sharpe_mean']
        print(f"Ensemble Advantage: {improvement:+.3f} Sharpe ({improvement/best_individual['Sharpe_mean']*100:+.1f}%)")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Optimized Horse Race Testing with Parallelization and GPU Support
Demonstrates parallelization within rolling windows while maintaining temporal integrity.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
import yaml
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ensemble.horse_race_individual_quality import Driver, MetricConfig, metric_sharpe
from data.data_utils_simple import prepare_real_data_simple
from model.xgb_drivers import generate_xgb_specs
from xgboost import XGBRegressor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedDriver(Driver):
    """Enhanced driver with GPU-optimized prediction caching."""
    
    def __init__(self, model, driver_id: int):
        super().__init__(model)
        self.driver_id = driver_id
        self._prediction_cache = {}
    
    def predict_with_cache(self, X: pd.DataFrame, cache_key: str) -> np.ndarray:
        """Cache predictions to avoid redundant GPU transfers."""
        if cache_key not in self._prediction_cache:
            self._prediction_cache[cache_key] = self.predict(X)
        return self._prediction_cache[cache_key]

def parallel_driver_evaluation(args) -> Dict:
    """Evaluate single driver on train/val/test splits - parallelizable."""
    driver_idx, driver, X_train, y_train, X_val, y_val, X_test, y_test = args
    
    try:
        # Fit model
        fitted_driver = driver.fit(X_train, y_train)
        
        # Generate predictions
        pred_val = fitted_driver.predict(X_val)
        pred_test = fitted_driver.predict(X_test)
        
        # Calculate metrics (simplified for speed)
        # Validation Sharpe
        val_positions = np.tanh((pred_val - np.mean(pred_val)) / (np.std(pred_val) + 1e-9))
        val_pnl = val_positions[:-1] * y_val.values[1:]  # 1-day lag
        val_sharpe = np.mean(val_pnl) / (np.std(val_pnl) + 1e-9) * np.sqrt(252)
        
        # Test Sharpe (for predictive correlation)
        test_positions = np.tanh((pred_test - np.mean(pred_test)) / (np.std(pred_test) + 1e-9))
        test_pnl = test_positions[:-1] * y_test.values[1:]
        test_sharpe = np.mean(test_pnl) / (np.std(test_pnl) + 1e-9) * np.sqrt(252)
        
        return {
            'driver_idx': driver_idx,
            'val_sharpe': float(val_sharpe) if np.isfinite(val_sharpe) else 0.0,
            'test_sharpe': float(test_sharpe) if np.isfinite(test_sharpe) else 0.0,
            'pred_test': pred_test,
            'test_positions': test_positions
        }
    except Exception as e:
        logger.warning(f"Driver {driver_idx} evaluation failed: {e}")
        return {
            'driver_idx': driver_idx,
            'val_sharpe': 0.0,
            'test_sharpe': 0.0,
            'pred_test': np.zeros(len(X_test)),
            'test_positions': np.zeros(len(X_test))
        }

def optimized_rolling_horse_race(
    X: pd.DataFrame,
    y: pd.Series,
    drivers: List[OptimizedDriver],
    start_train: int = 500,
    step: int = 21,
    horizon: int = 21,
    inner_val_frac: float = 0.2,
    max_workers: int = None
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Optimized rolling horse race with parallelized driver evaluation.
    Maintains temporal integrity while parallelizing within each window.
    """
    
    n = len(X)
    max_workers = max_workers or min(mp.cpu_count(), len(drivers))
    
    results = []
    window_details = []
    
    t0 = start_train
    window_count = 0
    
    logger.info(f"Starting optimized rolling evaluation with {max_workers} workers")
    
    while t0 + horizon <= n:
        window_count += 1
        logger.info(f"Processing window {window_count}: t0={t0}, horizon={horizon}")
        
        # Define temporal splits (NO LOOK-AHEAD)
        train_idx = np.arange(0, t0)
        test_idx = np.arange(t0, t0 + horizon)
        
        # Inner train/validation split
        val_cutoff = int(len(train_idx) * (1.0 - inner_val_frac))
        inner_train_idx = train_idx[:val_cutoff]
        inner_val_idx = train_idx[val_cutoff:]
        
        # Extract data splits
        X_train, y_train = X.iloc[inner_train_idx], y.iloc[inner_train_idx]
        X_val, y_val = X.iloc[inner_val_idx], y.iloc[inner_val_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        
        # Parallel driver evaluation within this window
        driver_args = [
            (i, driver, X_train, y_train, X_val, y_val, X_test, y_test)
            for i, driver in enumerate(drivers)
        ]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            driver_results = list(executor.map(parallel_driver_evaluation, driver_args))
        
        # Aggregate results for this window
        val_sharpes = np.array([r['val_sharpe'] for r in driver_results])
        test_sharpes = np.array([r['test_sharpe'] for r in driver_results])
        
        # Select top performers based on validation Sharpe
        top_k = 5
        selected_indices = np.argsort(-val_sharpes)[:top_k]
        
        # Build ensemble from selected drivers
        selected_positions = np.column_stack([
            driver_results[i]['test_positions'] for i in selected_indices
        ])
        ensemble_position = np.mean(selected_positions, axis=1)
        
        # Calculate ensemble performance
        ensemble_pnl = ensemble_position[:-1] * y_test.values[1:]  # 1-day lag
        ensemble_sharpe = np.mean(ensemble_pnl) / (np.std(ensemble_pnl) + 1e-9) * np.sqrt(252)
        ensemble_hit_rate = np.mean(np.sign(ensemble_position[:-1]) == np.sign(y_test.values[1:]))
        
        # Predictive correlation (validation vs realized performance)
        pred_corr = np.corrcoef(val_sharpes, test_sharpes)[0, 1] if len(np.unique(val_sharpes)) > 1 else 0.0
        
        # Record window results
        window_result = {
            't0': int(t0),
            't1': int(t0 + horizon),
            'ensemble_sharpe': float(ensemble_sharpe) if np.isfinite(ensemble_sharpe) else 0.0,
            'ensemble_hit_rate': float(ensemble_hit_rate) if np.isfinite(ensemble_hit_rate) else 0.5,
            'predictive_corr': float(pred_corr) if np.isfinite(pred_corr) else 0.0,
            'selected_drivers': selected_indices.tolist(),
            'n_drivers_evaluated': len(drivers)
        }
        
        results.append(window_result)
        window_details.append({
            'window': window_count,
            'val_sharpes': val_sharpes.tolist(),
            'test_sharpes': test_sharpes.tolist(),
            'selected_indices': selected_indices.tolist()
        })
        
        # Advance to next window
        t0 += step
    
    # Aggregate summary
    df_results = pd.DataFrame(results)
    summary = {
        'total_windows': len(results),
        'mean_sharpe': df_results['ensemble_sharpe'].mean(),
        'median_sharpe': df_results['ensemble_sharpe'].median(),
        'mean_hit_rate': df_results['ensemble_hit_rate'].mean(),
        'mean_pred_corr': df_results['predictive_corr'].mean(),
        'sharpe_std': df_results['ensemble_sharpe'].std()
    }
    
    logger.info(f"Completed {len(results)} windows")
    logger.info(f"Mean Ensemble Sharpe: {summary['mean_sharpe']:.3f}")
    logger.info(f"Mean Hit Rate: {summary['mean_hit_rate']:.3f}")
    logger.info(f"Mean Predictive Correlation: {summary['mean_pred_corr']:.3f}")
    
    return df_results, window_details

def main():
    """Main execution with optimized parallel processing."""
    
    # Load configuration
    config_path = "configs/quick_test_development.yaml"  # Use fast config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test parameters
    target_symbol = "@ES#C"
    start_date = "2020-01-01"
    end_date = "2024-01-01"  # Reduced for speed
    
    logger.info(f"Optimized testing: {target_symbol} ({start_date} to {end_date})")
    
    # Load data
    df = prepare_real_data_simple(
        target_symbol=target_symbol,
        symbols=["@ES#C", "@NQ#C", "@YM#C", "@RTY#C"],  # Reduced symbol set
        start_date=start_date,
        end_date=end_date,
        signal_hour=12,
        n_hours=24
    )
    
    target_col = f"{target_symbol}_target_return"
    y = df[target_col]
    X = df.drop(columns=[col for col in df.columns if col.endswith('_target_return')])
    
    logger.info(f"Data: X.shape={X.shape}, y.shape={y.shape}")
    
    # Create optimized drivers
    n_models = 15  # Moderate size for testing
    specs = generate_xgb_specs(n_models=n_models, seed=42)
    drivers = [OptimizedDriver(XGBRegressor(**spec), i) for i, spec in enumerate(specs)]
    
    logger.info(f"Created {len(drivers)} optimized drivers")
    
    # Run optimized horse race
    results_df, details = optimized_rolling_horse_race(
        X=X,
        y=y,
        drivers=drivers,
        start_train=400,  # Reduced for testing
        step=21,
        horizon=21,
        inner_val_frac=0.2,
        max_workers=4  # Controlled parallelization
    )
    
    # Save results
    os.makedirs("artifacts/horse_race", exist_ok=True)
    results_df.to_csv("artifacts/horse_race/optimized_results.csv")
    
    print("\n" + "="*60)
    print("OPTIMIZED HORSE RACE RESULTS")
    print("="*60)
    print(f"Total Windows: {len(results_df)}")
    print(f"Mean Sharpe: {results_df['ensemble_sharpe'].mean():.3f}")
    print(f"Median Sharpe: {results_df['ensemble_sharpe'].median():.3f}")
    print(f"Mean Hit Rate: {results_df['ensemble_hit_rate'].mean():.3f}")
    print(f"Mean Predictive Correlation: {results_df['predictive_corr'].mean():.3f}")
    print(f"Sharpe Std Dev: {results_df['ensemble_sharpe'].std():.3f}")

if __name__ == "__main__":
    main()
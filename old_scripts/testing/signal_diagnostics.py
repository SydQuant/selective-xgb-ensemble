#!/usr/bin/env python3
"""
Signal Diagnostics Script
=========================
Analyze XGB model signals to detect subtle bugs:
- Individual model signal patterns
- Combined ensemble signal behavior
- Signal transformations (tanh vs binary)
- Signal distribution and outliers
- Signal-target correlations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime

# Import framework components
from data.symbol_loader import load_symbol_data
from model.xgb_drivers import create_standard_xgb_bank, create_tiered_xgb_bank
from model.feature_selection import select_features
from cv.wfo import expanding_window_split
from xgb_compare.metrics_utils import calculate_pnl, calculate_sharpe_ratio, calculate_hit_rate

def analyze_individual_signals(models_data, X_test, y_test, symbol="TEST"):
    """Analyze individual model signals before combination"""
    print(f"\n{'='*60}")
    print(f"INDIVIDUAL MODEL SIGNAL ANALYSIS - {symbol}")
    print(f"{'='*60}")

    individual_signals = []
    signal_stats = []

    for i, model_data in enumerate(models_data[:10]):  # Analyze first 10 models
        model = model_data['model']

        # Get raw predictions
        raw_preds = model.predict(X_test)

        # Apply tanh transformation
        tanh_signal = np.tanh(raw_preds)

        # Apply binary transformation
        binary_signal = np.where(raw_preds > 0, 1, -1)

        individual_signals.append({
            'model_id': i,
            'raw_preds': raw_preds,
            'tanh_signal': tanh_signal,
            'binary_signal': binary_signal
        })

        # Calculate stats
        stats = {
            'model_id': i,
            'raw_mean': np.mean(raw_preds),
            'raw_std': np.std(raw_preds),
            'raw_min': np.min(raw_preds),
            'raw_max': np.max(raw_preds),
            'tanh_mean': np.mean(tanh_signal),
            'tanh_std': np.std(tanh_signal),
            'binary_pos_pct': np.mean(binary_signal > 0) * 100,
            'corr_with_target': np.corrcoef(tanh_signal, y_test)[0,1] if len(y_test) > 1 else 0
        }
        signal_stats.append(stats)

    # Convert to DataFrame for analysis
    stats_df = pd.DataFrame(signal_stats)

    print(f"\nRaw Prediction Statistics:")
    print(f"Mean range: [{stats_df['raw_mean'].min():.4f}, {stats_df['raw_mean'].max():.4f}]")
    print(f"Std range: [{stats_df['raw_std'].min():.4f}, {stats_df['raw_std'].max():.4f}]")
    print(f"Min range: [{stats_df['raw_min'].min():.4f}, {stats_df['raw_min'].max():.4f}]")
    print(f"Max range: [{stats_df['raw_max'].min():.4f}, {stats_df['raw_max'].max():.4f}]")

    print(f"\nTanh Signal Statistics:")
    print(f"Mean range: [{stats_df['tanh_mean'].min():.4f}, {stats_df['tanh_mean'].max():.4f}]")
    print(f"Std range: [{stats_df['tanh_std'].min():.4f}, {stats_df['tanh_std'].max():.4f}]")

    print(f"\nBinary Signal Statistics:")
    print(f"Positive signal %: [{stats_df['binary_pos_pct'].min():.1f}%, {stats_df['binary_pos_pct'].max():.1f}%]")

    print(f"\nSignal-Target Correlations:")
    print(f"Correlation range: [{stats_df['corr_with_target'].min():.4f}, {stats_df['corr_with_target'].max():.4f}]")
    print(f"Average correlation: {stats_df['corr_with_target'].mean():.4f}")

    # Flag suspicious patterns
    print(f"\n🚨 POTENTIAL ISSUES:")
    if stats_df['raw_std'].min() < 0.01:
        print(f"⚠️  Very low prediction variance detected (min std: {stats_df['raw_std'].min():.6f})")

    if abs(stats_df['binary_pos_pct'].mean() - 50) > 10:
        print(f"⚠️  Biased binary signals (avg positive %: {stats_df['binary_pos_pct'].mean():.1f}%)")

    if stats_df['corr_with_target'].mean() < 0.01:
        print(f"⚠️  Very low signal-target correlation (avg: {stats_df['corr_with_target'].mean():.4f})")

    return individual_signals, stats_df

def analyze_ensemble_combination(individual_signals, y_test, symbol="TEST"):
    """Analyze how individual signals combine into ensemble signal"""
    print(f"\n{'='*60}")
    print(f"ENSEMBLE SIGNAL COMBINATION ANALYSIS - {symbol}")
    print(f"{'='*60}")

    n_models = len(individual_signals)
    n_samples = len(individual_signals[0]['tanh_signal'])

    # Create signal matrices
    tanh_matrix = np.array([sig['tanh_signal'] for sig in individual_signals])
    binary_matrix = np.array([sig['binary_signal'] for sig in individual_signals])

    # Calculate ensemble signals
    ensemble_tanh = np.mean(tanh_matrix, axis=0)
    ensemble_binary_votes = np.sum(binary_matrix, axis=0)  # Sum of +1/-1 votes
    ensemble_binary_normalized = ensemble_binary_votes / n_models

    print(f"Ensemble Signal Statistics:")
    print(f"Tanh ensemble - Mean: {np.mean(ensemble_tanh):.4f}, Std: {np.std(ensemble_tanh):.4f}")
    print(f"Binary votes - Mean: {np.mean(ensemble_binary_votes):.2f}, Std: {np.std(ensemble_binary_votes):.2f}")
    print(f"Binary normalized - Mean: {np.mean(ensemble_binary_normalized):.4f}, Std: {np.std(ensemble_binary_normalized):.4f}")

    # Analyze vote distribution
    vote_counts = np.bincount(ensemble_binary_votes + n_models, minlength=2*n_models+1)
    vote_values = np.arange(-n_models, n_models+1)

    print(f"\nBinary Vote Distribution:")
    for vote_val, count in zip(vote_values, vote_counts):
        if count > 0:
            pct = count / n_samples * 100
            print(f"  {vote_val:+3d} votes: {count:4d} samples ({pct:5.1f}%)")

    # Check for ties and extreme votes
    tie_count = vote_counts[n_models]  # Zero votes
    extreme_pos = np.sum(vote_counts[-2:])  # Very positive votes
    extreme_neg = np.sum(vote_counts[:2])   # Very negative votes

    print(f"\nEnsemble Signal Patterns:")
    print(f"Tie votes (0): {tie_count} ({tie_count/n_samples*100:.1f}%)")
    print(f"Extreme positive: {extreme_pos} ({extreme_pos/n_samples*100:.1f}%)")
    print(f"Extreme negative: {extreme_neg} ({extreme_neg/n_samples*100:.1f}%)")

    # Signal-target correlation
    ensemble_target_corr = np.corrcoef(ensemble_tanh, y_test)[0,1] if len(y_test) > 1 else 0
    print(f"Ensemble-target correlation: {ensemble_target_corr:.4f}")

    # 🚨 Flag issues
    print(f"\n🚨 POTENTIAL ISSUES:")
    if tie_count / n_samples > 0.1:
        print(f"⚠️  High tie rate ({tie_count/n_samples*100:.1f}%) - ensemble indecision")

    if abs(np.mean(ensemble_binary_votes)) > 0.5:
        print(f"⚠️  Biased ensemble voting (mean votes: {np.mean(ensemble_binary_votes):.2f})")

    if abs(ensemble_target_corr) < 0.02:
        print(f"⚠️  Very weak ensemble-target correlation ({ensemble_target_corr:.4f})")

    return {
        'ensemble_tanh': ensemble_tanh,
        'ensemble_binary_votes': ensemble_binary_votes,
        'ensemble_binary_normalized': ensemble_binary_normalized,
        'ensemble_target_corr': ensemble_target_corr
    }

def analyze_target_returns(y_data, dates, symbol="TEST"):
    """Analyze target return patterns for anomalies"""
    print(f"\n{'='*60}")
    print(f"TARGET RETURN ANALYSIS - {symbol}")
    print(f"{'='*60}")

    # Basic statistics
    print(f"Target Return Statistics:")
    print(f"  Count: {len(y_data)}")
    print(f"  Mean: {np.mean(y_data):.6f}")
    print(f"  Std: {np.std(y_data):.6f}")
    print(f"  Min: {np.min(y_data):.6f}")
    print(f"  Max: {np.max(y_data):.6f}")
    print(f"  Skewness: {pd.Series(y_data).skew():.4f}")
    print(f"  Kurtosis: {pd.Series(y_data).kurtosis():.4f}")

    # Check for outliers
    q1, q3 = np.percentile(y_data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = np.sum((y_data < lower_bound) | (y_data > upper_bound))

    print(f"\nOutlier Analysis:")
    print(f"  Q1: {q1:.6f}, Q3: {q3:.6f}, IQR: {iqr:.6f}")
    print(f"  Outliers: {outliers} ({outliers/len(y_data)*100:.1f}%)")

    # Check for zero returns
    zero_returns = np.sum(np.abs(y_data) < 1e-8)
    print(f"  Zero returns: {zero_returns} ({zero_returns/len(y_data)*100:.1f}%)")

    # Check for NaN/inf
    nan_count = np.sum(np.isnan(y_data))
    inf_count = np.sum(np.isinf(y_data))
    print(f"  NaN values: {nan_count}")
    print(f"  Inf values: {inf_count}")

    # Time series patterns
    if dates is not None and len(dates) == len(y_data):
        df = pd.DataFrame({'date': dates, 'return': y_data})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()

        # Check for time gaps
        date_diffs = df.index.to_series().diff().dt.days
        median_gap = date_diffs.median()
        unusual_gaps = np.sum(date_diffs > median_gap * 3)

        print(f"\nTime Series Analysis:")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Median gap: {median_gap} days")
        print(f"  Unusual gaps: {unusual_gaps}")

    # 🚨 Flag issues
    print(f"\n🚨 POTENTIAL ISSUES:")
    if np.std(y_data) < 1e-6:
        print(f"⚠️  Very low return variance (std: {np.std(y_data):.8f})")

    if outliers / len(y_data) > 0.05:
        print(f"⚠️  High outlier rate ({outliers/len(y_data)*100:.1f}%)")

    if zero_returns / len(y_data) > 0.1:
        print(f"⚠️  High zero return rate ({zero_returns/len(y_data)*100:.1f}%)")

    if nan_count > 0 or inf_count > 0:
        print(f"⚠️  Invalid values detected (NaN: {nan_count}, Inf: {inf_count})")

def test_signal_diagnostics(symbol="@ES#C", n_models=5, max_features=50):
    """Run comprehensive signal diagnostics on a symbol"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE SIGNAL DIAGNOSTICS - {symbol}")
    print(f"{'='*80}")
    print(f"Models: {n_models}, Features: {max_features}")
    print(f"Test run at: {datetime.now()}")

    try:
        # Load data
        print(f"\n1. Loading data for {symbol}...")
        symbol_data = load_symbol_data([symbol])
        if symbol_data.empty:
            print(f"❌ No data loaded for {symbol}")
            return

        print(f"   Loaded {len(symbol_data)} rows")

        # Prepare features and target
        target_col = f"{symbol}_target_return"
        if target_col not in symbol_data.columns:
            print(f"❌ Target column {target_col} not found")
            return

        # Drop NaN targets
        initial_len = len(symbol_data)
        symbol_data = symbol_data.dropna(subset=[target_col])
        dropped = initial_len - len(symbol_data)
        if dropped > 0:
            print(f"   Dropped {dropped} rows with NaN targets")

        y = symbol_data[target_col].values
        feature_cols = [col for col in symbol_data.columns if not col.endswith('_target_return')]
        X = symbol_data[feature_cols].values
        dates = symbol_data.index if hasattr(symbol_data, 'index') else None

        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")

        # Feature selection
        print(f"\n2. Selecting features...")
        selected_features = select_features(X, y, max_features=max_features, threshold=0.7)
        X_selected = X[:, selected_features]
        print(f"   Selected {len(selected_features)} features")

        # Create train/test split
        print(f"\n3. Creating train/test split...")
        splits = list(expanding_window_split(len(y), n_folds=3))
        train_idx, test_idx = splits[-1]  # Use last fold for testing

        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

        # Analyze target returns
        print(f"\n4. Analyzing target returns...")
        test_dates = dates[test_idx] if dates is not None else None
        analyze_target_returns(y_test, test_dates, symbol)

        # Train models
        print(f"\n5. Training {n_models} XGB models...")
        models_bank = create_standard_xgb_bank(n_models)
        models_data = []

        for i, model_config in enumerate(models_bank[:n_models]):
            model = model_config['model']
            model.fit(X_train, y_train)
            models_data.append({
                'model': model,
                'config': model_config
            })

        print(f"   Trained {len(models_data)} models")

        # Analyze individual signals
        print(f"\n6. Analyzing individual model signals...")
        individual_signals, signal_stats = analyze_individual_signals(
            models_data, X_test, y_test, symbol
        )

        # Analyze ensemble combination
        print(f"\n7. Analyzing ensemble signal combination...")
        ensemble_analysis = analyze_ensemble_combination(
            individual_signals, y_test, symbol
        )

        # Save diagnostic results
        results_dir = Path("testing/diagnostic_results")
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"signal_diagnostics_{symbol.replace('@', '').replace('#C', '')}_{timestamp}.pkl"

        diagnostic_results = {
            'symbol': symbol,
            'timestamp': timestamp,
            'individual_signals': individual_signals,
            'signal_stats': signal_stats,
            'ensemble_analysis': ensemble_analysis,
            'target_stats': {
                'mean': np.mean(y_test),
                'std': np.std(y_test),
                'min': np.min(y_test),
                'max': np.max(y_test)
            }
        }

        with open(results_file, 'wb') as f:
            pickle.dump(diagnostic_results, f)

        print(f"\n✅ Diagnostics completed successfully!")
        print(f"📁 Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Error during diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test on multiple symbols for comparison
    test_symbols = ["@ES#C", "@TY#C", "QCL#C"]  # Good, good, problematic

    for symbol in test_symbols:
        test_signal_diagnostics(symbol=symbol, n_models=5, max_features=50)
        print(f"\n{'='*80}\n")
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# XGBoost Ensemble Trading System - Production Framework

## System Overview

This is a comprehensive machine learning trading system built around XGBoost ensembles for financial market prediction. The system uses multiple XGBoost models with different hyperparameters, sophisticated feature selection, signal transformation pipelines, and advanced ensemble optimization techniques.

**Core Architecture**:
1. **Data Pipeline**: Multi-symbol financial data loading with feature engineering (1000+ features)
2. **Feature Selection**: Smart block-wise correlation filtering and selection algorithms
3. **Model Training**: Parallel XGBoost ensembles with GPU/CPU detection and walk-forward validation
4. **Signal Processing**: Raw predictions → z-score normalization → tanh squashing → trading signals
5. **Ensemble Optimization**: GROPE (RBF-based global optimization) or Stability Selection for combining models
6. **Backtesting**: Rolling Q-score based model selection with comprehensive performance analysis

## Essential Commands

**Development Environment**: Windows with Anaconda Python

```bash
# Core system testing
~/anaconda3/python.exe main.py --config configs/production_full_system.yaml --target_symbol "@ES#C"

# XGBoost architecture comparison (primary development tool)
~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "test_run"

# Run all 6 XGBoost architecture tests (Sharpe Q-metric)
run_all_tests.bat

# Run all 6 XGBoost architecture tests (Hit Rate Q-metric)  
run_all_tests_hitrate.bat

# Rolling window training (fixed window instead of expanding)
~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "rolling_test" --rolling 504

# Monitor running processes
~/anaconda3/python.exe -c "import psutil; [print(f'PID {p.pid}: {\" \".join(p.cmdline())}') for p in psutil.process_iter() if 'xgb_compare_clean.py' in ' '.join(p.cmdline())]"

# Check results
ls -la xgb_compare/results/*.png
tail -f xgb_compare/results/logs/*.log
```

**XGBoost Architecture Types**:
- `standard`: Default XGBoost hyperparameter ranges
- `deep`: Deeper trees with more complex interactions  
- `tiered`: Multi-level ensemble approach

**Signal Processing Options**:
- Add `--binary_signal` for +1/-1 signals instead of tanh normalization
- Binary signals typically achieve higher hit rates due to decisive directional bets

**Cross-Validation Options**:
- Default: Expanding window (training data grows with each fold)
- `--rolling N`: Fixed rolling window of N days for training (e.g., `--rolling 504` for 2-year window)

**Q-Metric Options**:
- Default: `sharpe` (risk-adjusted returns)
- `--q_metric hit_rate`: Use directional accuracy for model selection
- `--q_metric cb_ratio`: Calmar-Burke ratio (return/max drawdown)
- `--q_metric adj_sharpe`: Adjusted Sharpe with turnover penalty

## System Architecture

### Data Flow Pipeline
1. **Data Loading** (`data/data_utils_simple.py`): Load multi-symbol financial data with feature engineering
2. **Feature Selection** (`model/feature_selection.py`): Block-wise correlation filtering (1000+ → ~400 features)
3. **XGBoost Training** (`model/xgb_drivers.py`): Parallel ensemble training with GPU/CPU detection
4. **Signal Processing** (`ensemble/combiner.py`): Raw predictions → z-score → tanh → trading signals [-1,1]
5. **Ensemble Optimization** (`opt/grope.py`): RBF-based global optimization for model weights
6. **Walk-forward Validation** (`cv/wfo.py`): Out-of-sample testing with rolling windows

### Key Components

**XGBoost Comparison Framework** (Primary Development Tool):
- `xgb_compare/xgb_compare_clean.py` - Multi-architecture XGBoost comparison with Q-score tracking
- `xgb_compare/full_timeline_backtest.py` - Rolling model selection based on quality scores
- `xgb_compare/production_detailed.py` - Comprehensive backtest visualizations
- `xgb_compare/visualization_clean.py` - Fold-by-fold performance analysis
- `xgb_compare/metrics_utils.py` - Performance metrics and quality tracking

**Core System**:
- `main.py` - Main orchestration script with stability ensemble support
- `model/xgb_drivers.py` - XGBoost model creation with automatic GPU detection
- `ensemble/combiner.py` - Signal transformation (zscore → tanh squashing)
- `ensemble/stability_selection.py` - Alternative to GROPE for production
- `opt/grope.py` - Global RBF optimization for ensemble weights

## Development Patterns

### Configuration Management
- **Main configs**: `configs/` directory with YAML files for different testing scenarios
- **Production config**: `configs/production_full_system.yaml` - optimized 10-year full system test
- **Quick testing**: `configs/quick_test_development.yaml` for rapid iteration
- **Specialty configs**: Various extreme parameter tests (hitrate, sharpe, etc.)

### Testing and Validation
- **XGBoost Comparison**: Primary tool for architecture evaluation with rolling backtest
- **Walk-forward Validation**: Out-of-sample testing with configurable fold counts
- **Statistical Significance**: P-value gating with Monte Carlo simulation (optional for speed)
- **Feature Selection**: Correlation-based block-wise filtering (configurable thresholds)

### Performance Optimization
- **GPU Detection**: Automatic CUDA detection with graceful CPU fallback
- **Multiprocessing**: Parallel XGBoost training (enabled by default)
- **Large Model Handling**: Optimized for 100+ model ensembles
- **Memory Management**: Efficient handling of 1000+ feature datasets

### Key Implementation Notes
- **Windows Environment**: Use `~/anaconda3/python.exe` for all Python execution
- **Fold 2 Exclusion**: Skip Fold 2 in backtesting (Q=0 scores select M00-M09 consistently)
- **GPU Warnings**: XGBoost CUDA warnings are normal - system falls back appropriately
- **Quality Tracking**: EWMA with 63-day halflife for model performance momentum
- **Results Storage**: All outputs saved to `xgb_compare/results/` and `artifacts/` directories

### Recent Fixes (September 2025)
All critical bugs in the XGBoost Comparison Framework have been completely resolved:

**Performance & Architecture Fixes:**
- **MAJOR PERFORMANCE FIX**: Eliminated redundant model retraining in backtesting phase
  - Fold 1 Q-score initialization now reuses pre-computed metrics (100x speedup)
  - Backtesting uses stored OOS predictions instead of retraining models
  - Typical backtesting time reduced from hours to minutes

**Q-Score & Model Selection Fixes:**
- Fixed Q-score stale selection bug - now uses ALL model metrics for updates (not just selected models)
- Corrected fold indexing and Q-score timing for proper rolling model selection
- Q-scores now evolve dynamically across folds with proper EWMA calculation
- Fixed "avg Q-score: 0.000" bug by using actual historical performance data

**Logging & Visualization Improvements:**
- Fixed inconsistent fold count logging (now shows actual vs configured folds)
- Production fold breakdown table now shows actual model numbers (M04, M03) instead of "2 models"
- Cleaned up excessive logging while maintaining essential configuration information
- Corrected confusing log messages with proper 1-based fold indexing

**System Robustness:**
- Visualization spacing fixed for large model counts (100+ models × 15 folds)
- QualityTracker halflife attribute error resolved
- Model usage matrix correctly excludes problematic fold data
- Proper separation between training and backtesting phases

The framework now runs reliably with proper rolling selection, dynamic Q-score evolution, clean visualizations, and optimal performance. All indexing inconsistencies have been resolved and the system provides clear, accurate logging throughout execution.

## Testing and Debugging

### Quick Robustness Testing
Before running full test suites, validate system robustness with minimal parameters:

```bash
# Run small test suite (5 models, 3 folds, 1 year data)
test_small.bat

# Manual quick test example
~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2020-01-01" --end_date "2021-01-01" --n_models 5 --n_folds 3 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "quick_test" --max_features 50
```

### Process Monitoring and Debugging
Long-running processes can get stuck during backtesting phase:

```bash
# Monitor running processes and log activity
~/anaconda3/python.exe debug_stuck_processes.py

# Kill stuck processes (use with caution)
~/anaconda3/python.exe debug_stuck_processes.py --kill-stuck

# Check recent log activity
tail -f xgb_compare/results/logs/*.log

# Monitor system resources
~/anaconda3/python.exe -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

### Common Issues and Solutions
- **Stuck at "Fold 1: Skipped"**: Process is retraining all models for Q-score initialization - this is computationally expensive but necessary
- **High memory usage**: Large feature sets (1000+) with many models (100+) require significant RAM
- **GPU warnings**: Normal XGBoost CUDA warnings - system automatically falls back to CPU
- **Process hanging**: Check if system has sufficient resources; use smaller parameters for testing

### Performance Parameters
For different testing scenarios:
- **Quick testing**: 5 models, 3 folds, 1 year data, 50 features
- **Development**: 15 models, 5 folds, 3 years data, 100 features  
- **Production**: 100 models, 15 folds, 10 years data, 400 features

## New Features (September 2024)

### Rolling Window Training (`--rolling N`)
- **Feature**: Fixed training window instead of expanding window
- **Implementation**: `wfo_splits_rolling()` function in `cv/wfo.py`
- **Usage**: `--rolling 504` for 2-year rolling window
- **Benefit**: Focuses on more recent data patterns, may improve model adaptability
- **Testing**: Successfully tested with small configurations

### Hit Rate Q-Metric (`--q_metric hit_rate`)
- **Feature**: Model selection based on directional accuracy instead of Sharpe ratio
- **Implementation**: Uses existing hit_rate calculation in `metrics_utils.py`
- **Usage**: `--q_metric hit_rate` in any xgb_compare command
- **Test Suite**: `run_all_tests_hitrate.bat` runs all 6 architectures with hit_rate
- **Benefit**: Prioritizes models with higher directional accuracy

### Why Binary Signals Have Higher Hit Rates
Binary signals (`--binary_signal`) convert continuous predictions to pure +1/-1 values based on z-score sign, creating decisive directional bets. Tanh normalization preserves continuous values that may include weak signals near zero, which can hurt hit rate despite potentially being profitable with proper position sizing.
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# XGBoost Ensemble Trading System - Production Framework

## Key Features

- **Smart Block-wise Feature Selection**: Handles 1000+ features efficiently
- **GROPE Optimization**: Global optimization for ensemble weights
- **Walk-forward Cross-validation**: Out-of-sample testing
- **Statistical Significance Testing**: P-value gating with Monte Carlo simulation
- **Full Universe Support**: Multi-asset feature engineering
- **Optimized XGBoost Hyperparameters**: Specialized parameter ranges for financial data

## Major Changes from Original

1. **Added --bypass_pvalue_gating parameter**: Optional bypass for statistical significance testing while preserving original functionality
2. **Optimized XGBoost Hyperparameters** (`model/xgb_drivers.py`):
   - Specialized parameter ranges for small financial returns
   - Prevents constant prediction issues with conservative regularization  
   - Broader diversity while maintaining stability (depth 2-6, lr 0.03-0.3)
   - Eliminates need for target scaling through proper hyperparameter selection
3. **Enhanced Data Engineering** (`data/data_utils_simple.py`):
   - Improved feature calculation logic (fixed breakout_1h, removed artificial clipping)
   - Time-series respecting data cleaning (forward-fill only, no look-ahead bias)
   - Reduced feature quality issues from 371 NaN to 0
4. **Enhanced Logging**: Added detailed prediction diagnostics for troubleshooting
5. **Smart Block-wise Feature Selection**:
   - **Step 1**: Split 1316 features into blocks of 100
   - **Step 2**: For each block, rank features by |target-correlation|
   - **Step 3**: Local clustering - select best features while removing highly correlated ones (threshold 0.7)
   - **Step 4**: Global deduplication across blocks to remove cross-block correlations
   - **Result**: Intelligently reduces 1316 → 50 features in ~1 second
   - **Performance**: Approximates full clustering but 10x faster on large feature sets

## Dependencies and Setup

### Required Python Packages
```bash
# Install via Anaconda pip (recommended on Windows)
~/anaconda3/Scripts/pip.exe install numpy pandas scikit-learn xgboost scipy pyyaml

# Alternative installation methods
python -m pip install numpy pandas scikit-learn xgboost scipy pyyaml
conda install numpy pandas scikit-learn xgboost scipy pyyaml
```

**Note**: ArcticDB may need separate installation: `~/anaconda3/Scripts/pip.exe install arcticdb`

### Core Dependencies
- **XGBoost**: Main ML algorithm for driver predictions
- **NumPy/Pandas**: Data manipulation and numerical computing
- **SciPy**: Hierarchical clustering for feature selection
- **scikit-learn**: Utilities for feature selection and metrics
- **PyYAML**: Configuration file parsing

No requirements.txt or setup.py files exist - dependencies must be installed manually.

## Usage Examples

**Important**: On Windows, use the Anaconda Python executable to ensure all dependencies are available:

```bash
# Basic single symbol testing
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C"

# Train-production method (faster, single train-test split)
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C" --train_production

# Bypass p-value gating (recommended for 4+ year periods)
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C" --bypass_pvalue_gating

# Quick test with fewer models
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C" --n_models 10

# Custom date range testing
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@VX#C" --start_date "2022-01-01" --end_date "2024-01-01"

# Long-term testing with all optimizations
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@TY#C" --start_date "2020-07-01" --end_date "2024-08-01" --bypass_pvalue_gating --train_production

# Full universe testing (all 25 symbols)
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml
```

### Alternative Python Execution Methods
If `~/anaconda3/python.exe` doesn't work, try:
- `python main.py` (if Python is in PATH)
- `py main.py` (Windows Python launcher)
- `conda run python main.py` (if using conda environment)

## Configuration Files

### `configs/individual_target_test.yaml`

Key parameters:

- `n_models: 50` - XGB ensemble size
- `n_select: 12` - Selected models after GROPE optimization
- `folds: 6` - Cross-validation folds
- `corr_threshold: 0.7` - Feature clustering threshold
- `pmax: 0.80` - P-value significance threshold
- `symbols: "..."` - Full universe of instruments for feature engineering

### Key CLI Parameters

- `--bypass_pvalue_gating` - Skip statistical significance tests (recommended for long periods, identical performance with 3x speed)
- `--train_production` - Use train-test split instead of cross-validation (NOT recommended - consistently underperforms CV by -0.17 to -1.23 Sharpe)
- `--target_symbol` - Override config to test single symbol
- `--n_models` - Override number of XGBoost models (optimal: 75, default: 50)
- `--max_features` - Limit number of features after selection (optimal: 50, max useful: ~78)
- `--start_date` / `--end_date` - Custom date range (format: "YYYY-MM-DD")
- `--equal_weights` - Use equal weighting instead of GROPE optimization (asset-dependent effectiveness)
- `--tiered_xgb` - Use tiered XGBoost configuration (breakthrough for underperformers: +0.24 to +0.37 Sharpe)
- `--deep_xgb` - Use deep XGBoost configuration (advanced architecture option)
- `--dapy_style` - DAPY calculation method: "hits" (default), "eri_long", "eri_short", "eri_both" (asset-specific optimization)

## Focus Areas for Inspection

### 1. Signal Generation Quality

```bash
# Monitor signal strength and consistency
grep "OOS signal magnitude:" logs/

# Monitor prediction variance across models
grep "test prediction stats" logs/
```

### 2. Feature Selection Effectiveness

```bash
# Check feature correlation levels and block processing
grep "Block.*Selected.*features.*best:" logs/

# Verify feature reduction efficiency
grep "Smart block-wise selection complete" logs/
```

### 3. Performance Validation

```bash
# Check backtest results
grep "OUT-OF-SAMPLE PERFORMANCE" logs/

# Verify statistical significance
grep "OOS Shuffling p-value" logs/

# Monitor ensemble optimization
grep "GROPE optimization" logs/
```

## Dataset Recommendations

- **Optimal timeframe**: 3-4 years for reliable XGB training
- **Target variance requirement**: std > 0.003 for meaningful patterns
- **Observation ratio**: 20:1 observations-to-features minimum
- **P-value bypass**: Recommended for periods > 4 years

## Recent Test Results and Optimization Guidelines

### Systematic Experimental Validation (4.1 Year Period: 2020-07-01 to 2024-08-01)

**Framework Status**: Production-ready with comprehensive optimization guidelines established across 8+ symbols and 12+ experimental configurations.

#### Validated Optimizations (Step-by-Step Results)

1. **P-Value Bypass (Step 1)**: ✅ **Recommended** - Identical performance with 3x speed improvement
2. **Feature Count (Steps 2a/2b)**: ✅ **50 features confirmed optimal** - expansion to 70+ consistently degrades performance
3. **Model Count (Step 5b)**: ✅ **75 models optimal** - breakthrough configuration with +0.02 to +0.50 Sharpe improvements
4. **Cross-Validation Method**: ✅ **6-fold CV required** - train-test split underperforms by -0.17 to -1.23 Sharpe
5. **Architecture Variants**:
   - **Tiered XGBoost**: Major breakthrough for underperformers (+0.24 to +0.37 Sharpe on @TY#C/@RTY#C)
   - **Equal Weights**: Asset-dependent alternative (competitive for bonds/commodities, inferior for equities)
   - **ERI_both Objective**: Asset-class specific (+0.59 for volatiles, -0.21 for stable assets)

#### Asset-Specific Performance Baselines

**@TY#C (10-Year Treasury) - Conservative Fixed Income:**
- Baseline: Sharpe 0.40, Return 5.22%, Max DD -6.36%, Win Rate 47.97%
- **Optimized (Tiered XGB)**: Sharpe 0.77 (+0.37), Return 10.13%, Max DD -5.89%
- Signal Magnitude: 441.92, P-value: 0.0049 ✅

**@ES#C (S&P 500 E-mini) - Equity Benchmark:**
- Baseline: Sharpe 0.41, Return 13.75%, Max DD -16.61%, Win Rate 47.87%  
- **Optimized (75 models)**: Sharpe 0.51 (+0.10), consistent signal strength
- Signal Magnitude: 454.45, P-value: 0.0049 ✅

**@EU#C (EUR/USD) - Currency Pair:**
- Baseline: Sharpe 0.15, Return 2.52%, Max DD -8.28%, Win Rate 45.98%
- Optimization responsive to equal weights and ERI objectives
- Signal Magnitude: 456.30, P-value: 0.0049 ✅

**QGC#C (Gold) - Commodities:**
- Baseline: Sharpe -0.42, Return -11.78%, Max DD -22.15%, Win Rate 47.78%
- **Strong signal generation despite negative returns** (magnitude: 466.12)
- Responds well to alternative architectures and objectives

#### Processing Performance Metrics
- **Feature Selection**: 1316 → 50 features in ~1 second (10x faster than full clustering)
- **Training Speed**: 6-fold CV in ~10-15 minutes per symbol
- **Signal Quality**: Consistent 400-550 magnitude across all tested instruments
- **Statistical Significance**: All configurations achieve p-value < 0.01 with Monte Carlo validation

## Next Steps

1. **Multi-symbol testing**: Run full universe tests across all 25 symbols
2. **Parameter optimization**: Fine-tune GROPE parameters for different asset classes
3. **Production deployment**: Implement real-time signal generation
4. **Performance analysis**: Compare cross-validation vs train-production methods
5. **Risk management**: Add position sizing and portfolio-level risk controls

## Development Commands

### Testing and Validation
```bash
# Check signal strength and consistency
grep "OOS signal magnitude:" logs/

# Monitor prediction statistics during training
grep "test prediction stats" logs/

# Check feature selection effectiveness
grep "Smart block-wise selection complete" logs/

# Validate backtest performance
grep "OUT-OF-SAMPLE PERFORMANCE" logs/

# Check statistical significance results
grep "OOS Shuffling p-value" logs/
```

### Debugging Commands
```bash
# Monitor signal quality trends
grep "OOS signal magnitude:" logs/ | tail -10

# Check feature correlation and selection quality
grep "Block.*Selected.*features.*best:" logs/

# Monitor GROPE optimization convergence
grep "GROPE optimization" logs/

# Check fold-by-fold model performance
grep "Fold.*XGBoost predictions:" logs/

# Monitor hyperparameter diversity
grep "test prediction stats.*model_0:" logs/
```

### Common Development Tasks
- **Modify XGBoost parameters**: Edit `model/xgb_drivers.py:generate_xgb_specs()` for standard or `generate_deep_xgb_specs()` for advanced
- **Adjust feature selection**: Modify parameters in `model/feature_selection.py` (optimal: 50 features, correlation threshold 0.7)
- **Change optimization settings**: Update GROPE parameters in `opt/grope.py` (asset-specific tuning recommended)
- **Add new metrics**: Extend functions in `metrics/` directory (supports DAPY variants and adjusted Sharpe)
- **Debug signals**: Check `ensemble/combiner.py:build_driver_signals()` for z-score/tanh transformation pipeline
- **Analyze alternative metrics**: Run `analyze_task8_metrics.py` for multi-metric validation (Adjusted Sharpe, CB_ratio)

### Performance Artifacts
Generated in `artifacts/` directory:
- `oos_timeseries.csv` - Out-of-sample equity curve with signal/PnL/equity columns
- `production_timeseries.csv` - Production method results (not recommended for primary use)
- `performance_summary.csv` - Key metrics summary with DAPY, IR, Sharpe, drawdown
- `fold_summaries.json` - Detailed fold-by-fold results for cross-validation analysis
- `diagnostics/` - Timestamped diagnostic summaries for troubleshooting

### Analysis Scripts
- `analyze_task8_metrics.py` - Alternative metrics validation (Adjusted Sharpe, CB_ratio, DAPY comparison)
- View `EXPERIMENT_RESULTS.md` - Comprehensive systematic validation results across 12 experimental configurations
- Check `logs/` directory for detailed execution diagnostics and signal quality monitoring

## Architecture

```
Data Loading → Feature Engineering → Smart Block-wise Selection
          ↓
Cross-validation Splits (6 folds) 
          ↓
[XGB Driver Predictions] (50 models per fold)
  - Optimized hyperparameters for small financial returns
  - Diverse parameter ranges with stability constraints
          ↓
[Transform to Signals]
  - Rolling z-score (win=100)
  - Tanh squashing to [-1,1]
          ↓
[Driver Selection] (per fold)
  - Metric = w_dapy*DAPY + w_ir*IR
  - P-value gating (shuffle test, block=10)
  - Greedy selection with diversity penalty
          ↓
[Weight Optimization (GROPE)]
  - Optimize {w_i} + temperature τ
  - Objective: DAPY + IR - λ*turnover
  - Global RBF optimization
          ↓
[Combine Selected Signals]
  - Softmax(w/τ) weighting
  - Weighted sum, clip [-1,1]
          ↓
[Walk-forward Stitching]
  - Train on each fold's past data
  - Generate signals on fold's future
  - Stitch out-of-sample signals
          ↓
[Backtest & Metrics]
  - Shift signal by 1 day (avoid look-ahead)
  - PnL = signal * return
  - Equity = cumsum(PnL)
  - Report DAPY, IR, Sharpe, drawdown
```

## Codebase Structure

### Core Module Architecture

**Entry Point:**
- `main.py` - Main orchestration script with CLI interface and logging

**Model Components:**
- `model/xgb_drivers.py` - **CRITICAL**: XGBoost model creation with 1000x target scaling fix
- `model/feature_selection.py` - Smart block-wise feature selection (1316→50 features)
- `model/block_ensemble.py` - Ensemble model coordination

**Data Pipeline:**
- `data/data_utils_simple.py` - **CRITICAL**: Enhanced feature engineering with time-series respecting data cleaning
- `data/loaders.py` - ArcticDB connection and data loading
- `data/symbol_loader.py` - Trading symbol management and defaults  
- `data/symbols.yaml` - Symbol configuration

**Signal Processing:**
- `ensemble/combiner.py` - Signal transformation (z-score → tanh → clipping)
- `ensemble/selection.py` - Driver selection with diversity penalty and p-value gating
- `utils/transforms.py` - Mathematical transformations for signals

**Optimization:**
- `opt/grope.py` - Global RBF optimization engine (GROPE algorithm)
- `opt/weight_objective.py` - Objective function factory for weight optimization

**Evaluation:**
- `cv/wfo.py` - Walk-forward cross-validation splits
- `eval/target_shuffling.py` - Statistical significance testing (Monte Carlo)
- `metrics/` - Performance calculation (DAPY, IR, Sharpe, drawdown)

**Configuration:**
- `configs/individual_target_test.yaml` - Single-symbol testing config
- `configs/full_26_symbol_test.yaml` - Multi-symbol universe testing
- `configs/production.yaml` - Production deployment config

### Data Flow Architecture
1. **Data Loading** → Real market data via ArcticDB connection
2. **Feature Engineering** → Multi-timeframe technical indicators (1316 features)
3. **Smart Feature Selection** → Block-wise clustering reduces to ~50 features
4. **Cross-validation Splits** → 6-fold walk-forward splits
5. **XGB Ensemble Training** → 50 diverse XGBoost models per fold
6. **Signal Generation** → Rolling z-score → tanh squashing → [-1,1] clipping
7. **Driver Selection** → P-value gating + diversity penalty + greedy selection
8. **Weight Optimization** → GROPE optimizes softmax weights + temperature
9. **Signal Combination** → Weighted ensemble with turnover penalty
10. **Out-of-Sample Stitching** → Walk-forward combination for clean backtest
11. **Performance Evaluation** → DAPY, IR, Sharpe, drawdown, statistical significance

### Key Implementation Details

**Optimized XGBoost Hyperparameters (`model/xgb_drivers.py`)**:
- Specialized parameter ranges for small financial returns (std ~0.01)
- Conservative regularization to prevent constant predictions (gamma=0, light reg_alpha/reg_lambda)
- Balanced depth (2-6), learning rate (0.03-0.3), and estimators (30-300) for diversity
- Eliminates constant prediction issues through proper hyperparameter selection

**Smart Feature Selection (`model/feature_selection.py`)**:
- Block-wise processing: 1316 features → blocks of 100
- Local clustering within blocks (correlation threshold 0.7)
- Global deduplication across blocks
- 10x faster than full clustering, approximates results

**GROPE Optimization (`opt/grope.py`)**:
- Latin Hypercube Sampling + RBF surrogate model
- Adaptive sampling with acquisition function
- Optimizes both ensemble weights and temperature parameter

**Signal Pipeline (`ensemble/combiner.py`)**:
- Raw predictions → rolling z-score (window=100)
- Z-scores → tanh squashing with beta parameter  
- Final clipping to [-1,1] range for risk management

Framework optimized for financial data. Validated across bonds, commodities, equities, volatility.

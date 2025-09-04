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
- **GPU Detection**: Automatic CUDA detection with CPU fallback

## Core Architecture

The system follows a pipeline architecture for financial signal generation:

```
Data Loading → Feature Engineering → Smart Block-wise Selection
          ↓
Cross-validation Splits (6 folds) 
          ↓
[XGB Driver Predictions] (50 models per fold)
  - Optimized hyperparameters for small financial returns
  - Diverse parameter ranges with stability constraints
  - Automatic GPU detection via detect_gpu()
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
- **XGBoost**: Main ML algorithm for driver predictions (with GPU support)
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

# Bypass p-value gating (recommended for 4+ year periods)
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C" --bypass_pvalue_gating

# Quick test with fewer models
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@ES#C" --n_models 10

# Custom date range testing
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@VX#C" --start_date "2022-01-01" --end_date "2024-01-01"

# Use train-test split instead of cross-validation (faster but less reliable)
~/anaconda3/python.exe main.py --config configs/individual_target_test.yaml --target_symbol "@TY#C" --train_test_split

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
- `pmax: 0.05` - P-value significance threshold (5% level)
- `symbols: "..."` - Full universe of instruments for feature engineering

### Key CLI Parameters

- `--bypass_pvalue_gating` - Skip statistical significance tests (recommended for long periods, identical performance with 3x speed)
- `--train_test_split` - Use single train-test split instead of cross-validation (faster but less reliable)
- `--target_symbol` - Override config to test single symbol
- `--n_models` - Override number of XGBoost models (optimal: 75, default: 50)
- `--max_features` - Limit number of features after selection (optimal: 50, max useful: ~78)
- `--start_date` / `--end_date` - Custom date range (format: "YYYY-MM-DD")
- `--equal_weights` - Use equal weighting instead of GROPE optimization (asset-dependent effectiveness)
- `--tiered_xgb` - Use tiered XGBoost configuration (breakthrough for underperformers: +0.24 to +0.37 Sharpe)
- `--deep_xgb` - Use deep XGBoost configuration (advanced architecture option)
- `--dapy_style` - DAPY calculation method: "hits" (default), "eri_long", "eri_short", "eri_both" (asset-specific optimization)

## Development Commands

### Testing and Validation
```bash
# Check final performance metrics
grep "OOS DAPY" logs/

# Check statistical significance results
grep "OOS Shuffling p-value" logs/

# Monitor data loading and processing
grep "Data loaded:" logs/

# Check signal quality (essential for validation)
grep "Processing complete:" logs/
```

### Common Development Tasks

**Modify XGBoost parameters**: 
- Edit `model/xgb_drivers.py:generate_xgb_specs()` for standard configuration
- Use `generate_deep_xgb_specs()` or `generate_tiered_xgb_specs()` for advanced architectures
- GPU detection handled automatically by `detect_gpu()` function

**Adjust feature selection**: 
- Modify parameters in `model/feature_selection.py` (optimal: 50 features, correlation threshold 0.7)
- Block-wise selection processes features in blocks of 100 for efficiency

**Change optimization settings**: 
- Update GROPE parameters in `opt/grope.py` (asset-specific tuning recommended)
- Weight bounds typically [-2.0, 2.0], temperature [0.2, 3.0]

**Debug signals**: 
- Check `ensemble/combiner.py:build_driver_signals()` for z-score/tanh transformation pipeline
- Signal flow: raw predictions → z-score → tanh squashing → clipping [-1,1]

**Add new metrics**: 
- Extend functions in `metrics/` directory (supports DAPY variants and adjusted Sharpe)
- Run `analyze_task8_metrics.py` for multi-metric validation

## Critical Codebase Components

### Entry Point
- `main.py` - Main orchestration script with CLI interface and streamlined logging

### Model Components
- `model/xgb_drivers.py` - **CRITICAL**: XGBoost model creation with optimized hyperparameters and GPU detection
- `model/feature_selection.py` - Smart block-wise feature selection (1316→50 features)
- `model/block_ensemble.py` - Ensemble model coordination

### Data Pipeline
- `data/data_utils_simple.py` - **CRITICAL**: Enhanced feature engineering with time-series respecting data cleaning
- `data/loaders.py` - ArcticDB connection and data loading
- `data/symbol_loader.py` - Trading symbol management and defaults  
- `data/symbols.yaml` - Symbol configuration

### Signal Processing
- `ensemble/combiner.py` - Signal transformation (z-score → tanh → clipping)
- `ensemble/selection.py` - Driver selection with diversity penalty and p-value gating
- `utils/transforms.py` - Mathematical transformations for signals

### Optimization
- `opt/grope.py` - Global RBF optimization engine (GROPE algorithm)
- `opt/weight_objective.py` - Objective function factory for weight optimization

### Evaluation
- `cv/wfo.py` - Walk-forward cross-validation splits
- `eval/target_shuffling.py` - Statistical significance testing (Monte Carlo)
- `metrics/` - Performance calculation (DAPY, IR, Sharpe, drawdown)

## Performance Artifacts

Generated in `artifacts/` directory:
- `oos_timeseries.csv` - Out-of-sample equity curve with signal/PnL/equity columns
- `performance_summary.csv` - Key metrics summary with DAPY, IR, Sharpe, drawdown
- `fold_summaries.json` - Detailed fold-by-fold results for cross-validation analysis
- `diagnostics/` - Timestamped diagnostic summaries for troubleshooting

## Key Implementation Details

**Optimized XGBoost Hyperparameters (`model/xgb_drivers.py`)**:
- Specialized parameter ranges for small financial returns (std ~0.01)
- Conservative regularization to prevent constant predictions (gamma=0, light reg_alpha/reg_lambda)
- Balanced depth (2-6), learning rate (0.03-0.3), and estimators (30-300) for diversity
- Automatic GPU detection with `detect_gpu()` function - tests CUDA availability, falls back to CPU

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

## Validated Optimizations

Based on systematic testing across 8+ symbols and 12+ experimental configurations:

1. **P-Value Bypass**: ✅ **Recommended** - Identical performance with 3x speed improvement
2. **Feature Count**: ✅ **50 features confirmed optimal** - expansion to 70+ consistently degrades performance
3. **Model Count**: ✅ **75 models optimal** - breakthrough configuration with +0.02 to +0.50 Sharpe improvements
4. **Cross-Validation Method**: ✅ **6-fold CV required** - train-test split underperforms by -0.17 to -1.23 Sharpe
5. **Architecture Variants**:
   - **Tiered XGBoost**: Major breakthrough for underperformers (+0.24 to +0.37 Sharpe on @TY#C/@RTY#C)
   - **Equal Weights**: Asset-dependent alternative (competitive for bonds/commodities, inferior for equities)
   - **ERI_both Objective**: Asset-class specific (+0.59 for volatiles, -0.21 for stable assets)

## Dataset Recommendations

- **Optimal timeframe**: 3-4 years for reliable XGB training
- **Target variance requirement**: std > 0.003 for meaningful patterns
- **Observation ratio**: 20:1 observations-to-features minimum
- **P-value bypass**: Recommended for periods > 4 years

## Complete Pipeline Architecture (Post-XGBoost)

The system follows a sophisticated 7-phase pipeline after XGBoost model execution:

### Phase Flow: XGBoost → Driver Selection → GROPE → Validation → Backtesting

1. **Signal Generation**: 50 XGBoost predictions → z-score + tanh → normalized signals [-1,1]
2. **Driver Selection**: Greedy diverse selection (12 from 50) using configurable objectives + diversity penalty  
3. **GROPE Optimization**: Global RBF optimization of ensemble weights + temperature (13 parameters)
4. **Walk-Forward Stitching**: 6-fold CV creates complete out-of-sample signal
5. **Validation**: P-value testing via block shuffling (600 shuffles, preserves autocorrelation)
6. **Backtesting**: Signal lagged 1-day → PnL calculation → risk metrics (Sharpe, drawdown, hit rate)
7. **Reporting**: Comprehensive artifacts (equity curves, fold summaries, diagnostics)

### Key Methodological Innovations

**Configurable Objectives System**:
- `predictive_icir_logscore`: ICIR + calibrated probability scoring (excellent scale compatibility)
- `adjusted_sharpe`: Multiple testing corrected Sharpe ratio
- Traditional: DAPY + Information Ratio combinations
- Scale compatibility analysis reveals optimal combinations (see OBJECTIVE_SCALE_ANALYSIS.md)

**Statistical Rigor**:  
- True out-of-sample: Walk-forward CV with 1-day signal lag (no look-ahead bias)
- Block shuffling preserves return autocorrelation structure in significance testing
- Multiple testing correction via Monte Carlo (600 shuffles → p-values)

**Optimization Architecture**:
- **Driver Selection**: Individual signal evaluation with diversity penalty (correlation-based)
- **Weight Optimization**: Global GROPE (Latin Hypercube + RBF surrogate) on combined signals
- **Regularization**: Turnover penalty prevents excessive trading
- **Temperature Parameter**: Softmax scaling for ensemble combination

### Critical Implementation Details

**Signal Processing Pipeline**:
```python
XGBoost_predictions → zscore(win=100) → tanh_squash(beta=1.0) → clip([-1,1])
Raw: 0.00234 → Z-score: 1.2 → Tanh: 0.83 → Final signal
```

**Backtesting Methodology**:
```python  
PnL = signal.shift(1) * forward_returns  # 1-day lag prevents look-ahead
Equity = PnL.cumsum()  # Walk-forward out-of-sample equity curve
Sharpe = (PnL.mean() * 252) / (PnL.std() * sqrt(252))  # Annualized
```

**Validation Framework**:
- **Quality Gates**: Non-zero folds, meaningful signal variance
- **Statistical Significance**: Block shuffle p-value testing (typically p < 0.05)
- **Performance Consistency**: Cross-fold stability analysis
- **Business Metrics**: 52%+ hit rate, positive Sharpe, controlled drawdown

### Artifacts Generated

**Core Outputs**:
- `artifacts/oos_timeseries.csv`: Complete signal, returns, PnL, equity time series
- `artifacts/performance_summary.csv`: Key metrics (Sharpe, drawdown, hit rate, etc.)
- `artifacts/fold_summaries.json`: Selected drivers + weights per fold
- `artifacts/diagnostics/`: Model statistics and feature information

**Typical Performance Profile**:
- **Sharpe Ratio**: 0.6-1.2 (good risk-adjusted returns)
- **Hit Rate**: 51-55% (better than random)
- **Max Drawdown**: 5-15% (controlled risk)
- **Statistical Significance**: p < 0.05 (95%+ confidence)

## Known Critical Issues

### Test 2 Analysis: Driver Selection Objective Bug (September 2024)

**CRITICAL BUG DISCOVERED**: All driver selection objectives produce identical results despite using different algorithms.

**Issue Details**:
- **Test 2a-2e**: All 5 different driver selection objectives (`hits`, `eri_both`, `adjusted_sharpe`, `cb_ratio`, `predictive_icir_logscore`) produce identical performance metrics
- **Identical Results**: Same Sharpe ratio (0.38), hit rate (24.41%), total return (8.36%) across all objectives
- **Configuration**: @ES#C, 3 models, 2 folds, bypassed p-value gating
- **Bug Persistence**: Issue exists even after removing all fallback logic (strict mode implementation)

**Root Cause Investigation**:
1. **Initial Discovery**: `driver_selection_obj` was None when using CLI arguments, causing fallback to same function
2. **Fallback Removal**: Implemented strict mode to remove all fallbacks in 3 locations:
   - P-value gating objective functions
   - Driver selection objective functions  
   - GROPE weight optimization objective functions
3. **Bug Persistence**: Even with strict mode (no fallbacks), different objectives still produce identical results

**Code Locations Affected**:
- `main.py:get_objective_functions()` - Fixed CLI argument mapping but bug persists
- `ensemble/selection.py:pick_top_n_greedy_diverse()` - Driver selection algorithm
- `opt/grope.py` - Weight optimization uses different objectives but same driver selection results

**Test Results Summary**:
```
Test 2a (hits):              Sharpe: 0.38, Hit Rate: 24.41%, Return: 8.36%
Test 2b (eri_both):          Sharpe: 0.38, Hit Rate: 24.41%, Return: 8.36%  
Test 2c (adjusted_sharpe):   Sharpe: 0.38, Hit Rate: 24.41%, Return: 8.36%
Test 2d (cb_ratio):          Sharpe: 0.38, Hit Rate: 24.41%, Return: 8.36%
Test 2e (predictive_icir):   Sharpe: 0.38, Hit Rate: 24.41%, Return: 8.36%
```

**Status**: ❌ **UNRESOLVED** - Deeper architectural issue requires investigation
- All objective functions are correctly defined in `metrics/objective_registry.py`
- Strict mode correctly raises errors for invalid objectives
- Issue appears to be in driver selection execution, not objective function mapping

**Impact**: High - Undermines driver selection optimization and algorithm comparison validity

**Next Steps Required**:
1. Debug driver selection algorithm to verify objective function is actually being used
2. Add diagnostic logging to trace objective function calls during driver selection
3. Verify that different objectives produce different scores during greedy selection process

Framework optimized for financial data. Validated across bonds, commodities, equities, volatility.
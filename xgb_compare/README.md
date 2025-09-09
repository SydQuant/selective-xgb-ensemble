# XGBoost Comparison Framework - Optimized

**Clean, efficient XGBoost model comparison system** with comprehensive analysis, Q-score tracking, and production backtesting.

## Key Features

- **Cross-Validation Analysis**: Walk-forward CV with 100 XGBoost models per fold (configurable)
- **Q-Score Tracking**: EWMA quality momentum with proper rolling updates (fixed stale selection bug)
- **Consolidated Visualizations**: Clean timestamped files with actual model numbers displayed
- **Production Backtesting**: Rolling model selection with stored OOS predictions (100x performance improvement)
- **Dynamic Model Selection**: Q-scores evolve across folds for proper model rotation
- **Multiple XGBoost Types**: Standard, Deep, and Tiered architectures with Tanh/Binary signals

## Quick Start

### Recommended Usage
```bash
# Current optimized version (recommended)
~/anaconda3/python.exe xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "production"

# Quick test (8 models, 5 folds)
~/anaconda3/python.exe xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2020-01-01" --end_date "2023-01-01" --n_models 8 --n_folds 5 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "quick_test" --max_features 50

# XGBoost Architecture Comparison (all 6 variants)
# See test_commands.txt for complete test suite
```

### Alternative Versions
```bash
# Original full version (more verbose)
~/anaconda3/python.exe xgb_compare_simple.py

# Legacy version (comprehensive)
~/anaconda3/python.exe xgb_compare_main.py
```

### Production Configuration
```bash
# Production backtesting with custom parameters
~/anaconda3/python.exe xgb_compare_main.py \
    --target_symbol "@ES#C" \
    --cutoff_fraction 0.7 \
    --top_n_models 5 \
    --q_metric "sharpe" \
    --log_label "production_test"
```

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target_symbol` | @ES#C | Trading symbol to analyze |
| `--start_date` | 2015-01-01 | Analysis start date (YYYY-MM-DD) |
| `--end_date` | 2025-08-01 | Analysis end date (YYYY-MM-DD) |
| `--n_folds` | 10 | Number of cross-validation folds |
| `--n_models` | 50 | Number of XGBoost models per fold |
| `--inner_val_frac` | 0.2 | Inner validation fraction (20%) |
| `--cutoff_fraction` | 0.7 | Production model selection cutoff (70%) |
| `--top_n_models` | 5 | Number of models to select for production |
| `--q_metric` | sharpe | Q-score metric for selection (sharpe/hit_rate/cb_ratio/adj_sharpe) |
| `--log_label` | comparison | Label for log files |

## Output Structure

```
xgb_compare/results/
├── logs/
│   └── xgb_compare_[label]_YYYYMMDD_HHMMSS.log
├── performance_analysis_@ES#C_YYYYMMDD_HHMMSS.png    # 📊 Complete OOS + Q-Score + P-Value analysis
├── q_evolution_@ES#C_YYYYMMDD_HHMMSS.png             # 📈 All Q-metrics evolution (no fold 0)
└── backtest_analysis_@ES#C_YYYYMMDD_HHMMSS.png       # 💰 Production backtest summary
```

**🎯 Streamlined Output**: Only 3 essential files per run, all timestamped to prevent overwrites.

## Core Features

### 🎯 **Optimized Analysis Pipeline**
- **Walk-Forward CV**: Proper fold scheduling with expanding windows
- **50 XGBoost Models**: Diverse hyperparameters with GPU acceleration
- **Q-Score Tracking**: EWMA quality momentum starting from Fold 1 (no fold 0 baseline)
- **Top 10% Highlighting**: Dynamic scaling (5/50 for production, 1/5 for testing)

### 📊 **3 Consolidated Visualizations**
1. **Performance Analysis**: All models shown, top 10% highlighted
   - OOS Performance (Sharpe, Hit Rate, etc.)
   - Q-Score Evolution (historical quality momentum)
   - Statistical Significance (p-values with green=significant)

2. **Q-Evolution Analysis**: All 4 Q-metrics in single 2x2 grid
   - Starts from Fold 1 (removes unnecessary fold 0)
   - Shows top 10% models per metric
   - Clear EWMA evolution tracking

3. **Backtest Analysis**: Complete production tracking
   - Model selection timeline with Q-scores
   - Fold performance breakdown
   - Model usage frequency
   - Production summary metrics

### 💰 **Production Backtesting**
**Proper CV Schedule** (Example: 10 folds, 70% cutoff):
- **Folds 1-7**: Model training and Q-score accumulation
- **End Fold 7**: Select top 5 models based on Q-scores
- **Folds 8-10**: Production backtesting with selected models
- **Rolling Reselection**: Update model selection each fold

### 🔧 **Key Metrics**
- **IS/IV/OOS**: In-sample, Inner-validation, Out-of-sample performance
- **Q-Scores**: EWMA quality momentum for Sharpe, Hit Rate, CB Ratio, Adj Sharpe  
- **P-Values**: Bootstrap significance testing (green=significant, red=random)
- **Production Metrics**: Overall Sharpe, Hit Rate, Return, Volatility

## Technical Implementation

### Data Flow
```
Raw Data → Feature Selection (cluster reduction) → Cross-Validation Splits
    ↓
Model Training (50 XGB models per fold) → IS/IV/OOS Predictions
    ↓
Metrics Calculation → Q-Score Update → Visualization Generation
    ↓
Production Model Selection (at 70% cutoff) → Backtesting → Final Results
```

### Key Features
- **Feature Selection**: Uses cluster reduction with correlation threshold 0.7
- **Signal Processing**: Raw predictions → normalize → shift(1) for OOS → calculate returns
- **Quality Tracking**: EWMA momentum preserves historical performance memory
- **Statistical Rigor**: Bootstrap p-values test significance vs. random performance

### Performance Expectations
- **Runtime**: ~15-30 minutes for 50 models × 10 folds (depending on data size)
- **Memory Usage**: ~2-4 GB for typical financial time series
- **Output Size**: ~10-20 MB (logs + visualizations)

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're in the correct directory and using the right Python path
2. **Memory Issues**: Reduce `n_models` or `n_folds` for large datasets
3. **Visualization Errors**: Check matplotlib/seaborn installation
4. **Data Loading Issues**: Verify symbol names and date ranges

### Dependencies
Required packages (install via anaconda):
```bash
~/anaconda3/Scripts/pip.exe install numpy pandas scikit-learn xgboost scipy matplotlib seaborn
```

### Debugging
- Check log files in `results/logs/` for detailed execution information
- Enable verbose logging by modifying the logging level in `xgb_compare_main.py`
- Validate data loading separately using `prepare_real_data_simple()`

## Example Output Interpretation

### Performance Table
```
Model | OOS_Sharpe | OOS_Hit | Q_Sharpe | Q_Hit
M01   | 0.234      | 52.1%   | 0.156    | 0.521
M02   | 0.456      | 54.2%   | 0.389    | 0.537  <- Top performer
```

### Production Backtest Summary
```
Configuration:
  - Cutoff Fraction: 70%
  - Top N Models: 5
  - Q-Metric: sharpe

Model Selection History:
  Fold 7: M02, M15, M33, M41, M47 (Avg Q-sharpe: 0.342)
  Fold 8: M02, M15, M28, M33, M41 (Avg Q-sharpe: 0.367)
```

This framework provides comprehensive model comparison with full transparency into performance evolution and production model selection decisions.
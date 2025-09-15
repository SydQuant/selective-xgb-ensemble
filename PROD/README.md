# XGBoost Production Framework

Production-ready framework for deploying XGBoost ensemble models for financial signal generation.

## Overview

This framework bridges the research XGBoost framework (`xgb_compare/`) with production deployment, maintaining the same feature engineering and signal logic while providing robust daily execution capabilities.

## Architecture

```
PROD/
├── config/
│   ├── models/           # Per-symbol model configurations
│   ├── trading_config.yaml  # Portfolio, instruments, positions
│   └── global_config.yaml   # Data sources, timing, paths
├── models/
│   ├── @ES#C/           # Saved XGBoost models per symbol
│   └── ...
├── common/              # Core production modules
├── daily_signal_runner.py  # Main production script
└── logs/               # Daily execution logs
```

## Key Differences from v2.1 Production

| Component | v2.1 Production | XGBoost Production |
|-----------|----------------|-------------------|
| Signal Generation | L1 feature weights + majority voting | XGBoost ensemble + democratic voting |
| Model Selection | Manual feature selection | Automated optimal configs per symbol |
| Configuration | Feature weights CSV files | YAML model configurations |
| Data Access | IQFeed (same) | IQFeed (same) |
| Trade Processing | Same GMS format | Same GMS format |

## Setup and Deployment

### 1. Build Production Models

First, extract final production models from optimal research configurations:

```bash
# Build models for all symbols with optimal configs
cd xgb_compare
python production_model_builder.py --symbol all

# Or build for specific symbol
python production_model_builder.py --symbol "@ES#C"
```

This creates:
- `PROD/models/{symbol}/model_*.pkl` - Individual XGBoost models
- `PROD/config/models/{symbol}.yaml` - Symbol configuration
- Model metadata and feature specifications

### 2. Configure Trading Parameters

Update `PROD/config/trading_config.yaml` with:
- Current positions
- Bloomberg ticker mappings (already configured from v2.1)
- Portfolio allocations
- Position limits

### 3. Run Daily Signals

```bash
cd PROD
python daily_signal_runner.py --signal-hour 12

# Dry run (no file output)
python daily_signal_runner.py --signal-hour 12 --dry-run
```

## Production Model Configurations

Based on multi-symbol testing results (`MULTI_SYMBOL_TESTING_MATRIX.md`):

| Symbol | Config | Sharpe | Models | Folds | Features | Architecture |
|--------|--------|--------|--------|-------|----------|-------------|
| @ES#C | HIT_Q + 15F + std + 100feat + 150M | 2.319 | 150 | 15 | 100 | standard |
| @TY#C | SHARPE_Q + 10F + tiered + 250feat + 200M | 2.067 | 200 | 10 | 250 | tiered |
| @EU#C | SHARPE_Q + 20F + tiered + 100feat + 200M | 1.769 | 200 | 20 | 100 | tiered |
| @S#C | SHARPE_Q + 15F + std + 250feat + 200M | 1.985 | 200 | 15 | 250 | standard |

## Critical Production Considerations

### 1. Model Selection Issue ⚠️

**IMPORTANT**: The current research framework uses cross-validation folds, but production needs the **final trained models** that would be deployed.

- **Current**: CV folds with model selection per fold
- **Needed**: Final models trained on ALL data for deployment

The `production_model_builder.py` script addresses this by:
- Training models on ALL available data (no holdout)
- Using exact feature selection from research
- Saving deployment-ready model artifacts

### 2. Feature Engineering

Production feature engineering **must match exactly** what was used in research:
- Same feature calculation logic (`data_utils_simple.py`)
- Same feature selection methodology
- **Critical**: NO target column in production features
- Forward-fill only (no future leakage)

### 3. Signal Generation Process

```python
# 1. Fetch live data (IQFeed)
features_data = data_engine.get_prediction_features(symbol, feature_symbols)

# 2. Generate ensemble predictions
predictions = []
for model in loaded_models:
    pred = model.predict(features_data.iloc[-1:])  # Latest row only
    predictions.append(pred)

# 3. Democratic voting
normalized_preds = normalize_predictions(predictions, binary_signal=True)
signal = sign(sum(normalized_preds))  # Majority wins

# 4. Process to trades
trade = process_trade(symbol, signal, current_pos, price)
```

### 4. Data Pipeline

- **Input**: Live IQFeed data (same as v2.1)
- **Processing**: Feature engineering → XGBoost prediction → Signal transformation
- **Output**: GMS Excel file (same format as v2.1)

### 5. Deployment Checklist

- [ ] Production models built and validated
- [ ] Configuration files updated
- [ ] IQFeed connection available
- [ ] S3 credentials configured (if using)
- [ ] Email notification setup
- [ ] Dry run successful
- [ ] Logging directory writable

## Monitoring and Diagnostics

### Log Files
- Daily logs: `logs/YYYYMMDD/YYYYMMDD_HHMM_xgb_production.log`
- Model metadata: `models/{symbol}/model_metadata.yaml`
- Signal summaries: `logs/YYYYMMDD/YYYYMMDD_signals_summary_12hr.xlsx`

### Key Metrics to Monitor
- Model consensus strength
- Feature availability
- Price data freshness
- Trade generation counts
- S3 upload success

## Troubleshooting

### Common Issues

1. **No features generated**
   - Check IQFeed connection
   - Verify symbol data availability
   - Review feature engineering logs

2. **Model loading failures**
   - Ensure all models built correctly
   - Check file permissions
   - Verify YAML configurations

3. **No trades generated**
   - All signals may be zero/neutral
   - Check position limits
   - Review price data

### Recovery Procedures

1. **Missing models**: Re-run `production_model_builder.py`
2. **Config errors**: Validate YAML syntax
3. **Data issues**: Check IQFeed status and symbol mappings

## Weekly Analysis and Monitoring

### Weekly Runner (`weekly_runner.py`)

Consolidated weekly analysis tool that provides:

1. **Signal Reconciliation**: Compare backtest vs live signals
2. **Performance Analysis**: Generate equity curves and metrics
3. **Plotting**: Comprehensive performance visualization

**Usage:**
```bash
cd PROD
python weekly_runner.py
```

**Output:**
- `weekly_analysis/signal_comparison_h12_YYYYMMDD.csv` - Signal accuracy analysis
- `weekly_analysis/performance_analysis_h12_Xw_YYYYMMDD.png` - Performance plots
- Console summary of portfolio metrics

### Key Features:

- **Signal Validation**: Ensures backtest signals match live production
- **Performance Tracking**: Rolling Sharpe, drawdown, equity curves
- **Multi-timeframe Analysis**: 12-week and 52-week lookbacks
- **Portfolio Visualization**: Combined instrument and portfolio views

## Future Enhancements

1. **Real-time monitoring dashboard**
2. **Automated model retraining**
3. **Risk management integration**
4. **Performance tracking vs benchmarks**
5. **Automated weekly reporting via email**

---

*Generated by XGBoost Production Framework v6*
*Last updated: 2025-09-15*
# Claude Project Documentation

## Overview
XGBoost ensemble framework for financial time series prediction with comprehensive systematic testing across multiple model selection strategies.

## Current Status (2025-09-12)

### ✅ Completed Testing Phases

**SHARPE_Q Testing Matrix** (Sharpe-based model selection):
- **Phase 1-5**: All 27 tests completed ✅
- **Key Results**: Phase 5 optimal configs show strong performance improvements
- **Best Performers**: TY 150M (1.642 Sharpe), ES 100M (1.327), EU 100M (1.279)

**HIT_Q Testing Matrix** (Hit rate-based model selection):
- **Phase 1**: 6/6 tests completed ✅  
- **Phase 2-4**: 16+ tests currently running 🔄
- **Early Findings**: Hit_Q shows mixed results vs Sharpe_Q (beneficial for ES, neutral for TY, varies for EU)

### 🔧 Critical Bug Fixes Applied
1. **Artificial Signal Lag**: Removed unnecessary shift(1) for proper temporal alignment
2. **Binary Signal Logic**: Fixed democratic voting implementation (sum of ±1 votes)
3. **Hit Rate Calculation**: Exclude zero signals (ties) from hit rate computation  
4. **Tiered XGB Bug**: Fixed to use stratified_xgb_bank instead of standard XGB specs

### 📊 Key Architecture Files
- `xgb_compare/xgb_compare.py`: Main framework entry point
- `xgb_compare/full_timeline_backtest.py`: Core backtesting logic (critical fixes applied)
- `xgb_compare/metrics_utils.py`: PnL calculation and hit rate computation (fixed)
- `xgb_compare/results/SHARPE_Q_TESTING_MATRIX.md`: Complete results documentation
- `xgb_compare/results/HIT_Q_TESTING_MATRIX.md`: Ongoing comparative testing

### 🚀 Current Running Tests (16+ parallel)
- ES/TY/EU architecture tests (standard/tiered/deep)
- ES/TY/EU feature count tests (100/250 features)  
- ES/TY fold count tests (15/20 folds)

### 📈 Key Insights
1. **Model Selection Strategy Impact**: Hit_Q vs Sharpe_Q produces different model selections and performance
2. **Architecture Benefits**: Tiered XGB shows genuine improvements when properly implemented
3. **Binary Signals**: Work as "volatility capture systems" rather than directional predictors
4. **Scalability Success**: 100M/150M models generally improve over 50M baseline
5. **Market Differences**: ES most predictable, TY moderate, EU varies by configuration

### 🎯 Optimal Configurations Identified
- **ES**: 20 folds, binary signals, standard XGB, 100 features, 100M models
- **TY**: 8 folds, binary signals, tiered XGB, 100 features, 150M models  
- **EU**: 10 folds, tanh signals, tiered XGB, 250 features, 100M models

### ⚡ Running Commands
Use these commands to run tests:
```bash
# Standard architecture test with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_standard"

# Feature count test
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 250 --q_metric hit_rate --log_label "hitQ_TY_250feat"

# Tiered architecture test  
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_EU_tiered"
```

### 📝 Documentation Status
- All documentation files up to date and pushed to git
- Systematic testing matrices track progress across all test dimensions
- Results include performance metrics, log timestamps, and status indicators

---
*Last Updated: 2025-09-12 01:52*  
*Current Focus: Hit_Q vs Sharpe_Q comparative analysis*
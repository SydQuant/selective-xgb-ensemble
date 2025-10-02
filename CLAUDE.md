# XGBoost Ensemble Trading System

## Overview
Production-ready XGBoost ensemble framework for systematic futures trading with comprehensive backtesting, model selection, and portfolio-level analysis.

**Status:** Live production deployment since Sept 2025
**Performance (YTD 2025):** +12.46% return, 4.86 Sharpe ratio, -1.52% max drawdown

---

## System Architecture

### 🎯 Production System (`PROD/`)
**Live signal generation and trade execution**

**Core Components:**
- `daily_signal_runner.py` - Daily production signals with IQFeed integration
- `config.py` - Centralized configuration (symbols, baskets, allocations)
- `common/signal_engine.py` - Model loading and signal generation
- `common/data_engine.py` - Real-time and database data fetching
- `common/iqfeed.py` - IQFeed DTN connection manager
- `common/trades_util.py` - Position sizing and trade file generation

**Production Models:**
- 27 active symbols across 7 baskets (EQUITY, FX, RATESUS, RATESEU, AGS, Energy, METALS)
- $50M AUM with basket-level allocation
- Position sizing: `contracts = basket_allocation / fraction / price / multiplier`

### 📊 Weekly Analysis (`PROD/weekly_analysis_scripts/`)
**Backtesting and reconciliation tools**

**Scripts:**
- `portfolio_backtest.py` - Portfolio-level backtest with production sizing
  - Default: YTD from 2025-01-01
  - Outputs: Performance charts, symbol grid, position/metrics CSVs

- `weekly_runner_v1.1.py` - Signal reconciliation vs live trades
  - Regex-based Bloomberg symbol mapping (handles contract rollovers)
  - 100% reconciliation match rate achieved
  - Outputs: CSV reconciliation reports only

- `viz_helper.py` - Visualization utilities
  - Combined performance & metrics panel (portfolio, baskets, drawdown, Sharpe, risk-return)
  - Symbol grid with uniform scale, variable line thickness, end labels

**Output Structure:**
```
outputs/
├── portfolio/
│   ├── performance_*.png      # Portfolio + baskets + metrics (4 panels)
│   ├── symbol_grid_*.png      # 27 symbols (raw returns, uniform scale)
│   ├── positions_*.csv        # Position details with notional
│   └── daily_metrics_*.csv    # Daily portfolio metrics
└── weekly_backtest/
    └── reconciliation_*.csv   # Signal reconciliation reports
```

### 🔬 Research Framework (`xgb_compare/`)
**Systematic model testing and optimization**

**Core Files:**
- `xgb_compare.py` - Main comparative testing framework
- `full_timeline_backtest.py` - CV-aware backtesting engine
- `metrics_utils.py` - Performance metrics (Sharpe, hit rate, PnL)
- `extract_production_models.py` - Model selection for deployment
- `config.py` - XGBoost architectures (standard/tiered/deep)

**Testing Capabilities:**
- Q-score model selection (Sharpe or Hit Rate based)
- Walk-forward optimization with purge/embargo
- Multiple XGBoost architectures (standard, tiered, deep)
- Feature selection via Boruta
- Comprehensive visualization suite

### 📚 Core Libraries
**Shared modules**

- `data/data_utils_simple.py` - Feature engineering (RSI, ATR, momentum, cross-correlations)
- `data/loaders.py` - ArcticDB connection and data access
- `model/feature_selection.py` - Boruta feature selection
- `model/xgb_drivers.py` - XGBoost training (multiple architectures)
- `cv/wfo.py` - Walk-forward optimization with purge/embargo

---

## Key Results & Findings

### Production Performance (YTD 2025)
- **Total Return:** +12.46%
- **Sharpe Ratio:** 4.86 (annualized)
- **Max Drawdown:** -1.52%
- **Avg Utilization:** 87% of $50M AUM
- **Active Positions:** 26.7 symbols average

### Basket Attribution (YTD 2025)
- **EQUITY** (5.9% AUM): +5.76% return
- **METALS** (2.3% AUM): +4.28% return
- **FX** (2.4% AUM): +0.77% return
- **RATESUS** (7.8% AUM): +0.73% return
- **AGS** (0.6% AUM): +0.65% return
- **RATESEU** (7.8% AUM): +0.17% return
- **Energy** (1.9% AUM): +0.09% return

### Model Selection Research (Q-Score Testing)

**Sharpe_Q Results:**
- TY 150M: 1.642 Sharpe
- ES 100M: 1.327 Sharpe
- EU 100M: 1.279 Sharpe

**Hit_Q vs Sharpe_Q:**
- Different model selection leads to performance variance
- Hit_Q beneficial for ES, neutral for TY, mixed for EU

**Optimal Configurations:**
- **ES:** 20 folds, binary signals, standard XGB, 100 features, 100M models
- **TY:** 8 folds, binary signals, tiered XGB, 100 features, 150M models
- **EU:** 10 folds, tanh signals, tiered XGB, 250 features, 100M models

---

## Quick Start Commands

### Run Portfolio Backtest
```bash
# YTD backtest with all symbols (default)
python3 PROD/weekly_analysis_scripts/portfolio_backtest.py --save-details

# Custom date range
python3 PROD/weekly_analysis_scripts/portfolio_backtest.py --start-date 2025-06-01 --end-date 2025-09-30

# Specific symbols only
python3 PROD/weekly_analysis_scripts/portfolio_backtest.py --symbols "@ES#C" "@TY#C" "@EU#C"
```

### Run Signal Reconciliation
```bash
# 10-day backtest with 5-day reconciliation (all symbols)
python3 PROD/weekly_analysis_scripts/weekly_runner_v1.1.py --backtest-days 10 --reconcile-days 5 --reconcile

# Specific symbols
python3 PROD/weekly_analysis_scripts/weekly_runner_v1.1.py --backtest-days 7 --reconcile-days 3 --reconcile --symbols "@ES#C" "@TY#C"
```

### Run XGBoost Comparative Testing
```bash
# Standard architecture
python3 xgb_compare/xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 20 --max_features 100

# Tiered architecture with Hit_Q
python3 xgb_compare/xgb_compare.py --target_symbol "@TY#C" --n_models 150 --n_folds 8 --xgb_type tiered --q_metric hit_rate
```

---

## Critical Bug Fixes Applied

1. **Artificial Signal Lag** - Removed unnecessary shift(1) for proper temporal alignment
2. **Binary Signal Logic** - Fixed democratic voting (sum of ±1 votes)
3. **Hit Rate Calculation** - Exclude zero signals from hit rate computation
4. **Tiered XGB** - Fixed to use stratified_xgb_bank instead of standard specs
5. **Future Leakage** - Forward-fill only (no bfill/median) in feature engineering

---

## Archived Code (`old_scripts/`)

Legacy modules preserved for reference:
- `main.py` - Original entry point
- `ensemble/` - Older ensemble selection (horse race, stability)
- `metrics/` - Legacy performance metrics
- `opt/` - Weight optimization and grope algorithm
- `testing/` - 60+ diagnostic/debugging scripts
- `xgb_compare_utils/` - Standalone utilities

---

*Last Updated: 2025-10-02*
*System Status: Production deployment with 100% signal reconciliation*

# Corrected Logic Testing Results

## Overview

Testing corrected backtesting logic with multiple bug fixes:

- **✅ Fixed**: Removed artificial signal lag (direct signal-return alignment)
- **✅ Fixed**: Binary signals now use democratic voting (not averaging)
- **✅ Fixed**: Tiered XGB now uses stratified_xgb_bank (not standard XGB)

**Testing Matrix:**

- **Symbols**: ES, TY, EU
- **Signal Types**: tanh vs binary (with proper voting)
- **Folds**: 8, 10, 15, 20
- **Features**: 100, 250, -1 (all)
- **Architectures**: standard vs tiered XGB

## Baseline Results (50 models, 8 folds, 100 features, tanh, standard XGB)

| Symbol       | Training Sharpe | **Production Sharpe (Baseline)** | Hit Rate | Annual Return | Log Timestamp |
| ------------ | --------------- | -------------------------------------- | -------- | ------------- | ------------- |
| **ES** | 1.571           | **0.996**                        | 50.6%    | 12.77%        | 201417        |
| **TY** | 0.855           | **1.609**                        | 52.6%    | 3.43%         | 202106        |
| **EU** | 1.563           | **0.740**                        | 52.6%    | 4.88%         | 202110        |

**Bold production Sharpe = Better than baseline**

---

## Testing Matrix

### Phase 1: Signal Type Comparison (Binary vs Tanh) - WITH FIXED VOTING LOGIC

**Config**: 50 models, 8 folds, 100 features - **NEW: Binary signals use vote-based combination**

*(SD: It's the same voting logic as AWS PROD. Sum of all the +1/-1.  Because it selects top 2 models, it's very likely to have a lot of 0 signal (+1 and -1). I forgot to exclude 0-signal from the hit rate calculation - fixed now. But Sharpe should be corect.)*

| Test | Symbol | Signal Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| 1.1  | ES     | binary      | ✅     | 0.468           | **1.754**   | 33.6%    | 11.06%        | 224628        |
| 1.2  | TY     | binary      | ✅     | 0.249           | **2.076**   | 32.6%    | 4.15%         | 213626        |
| 1.3  | EU     | binary      | ✅     | 1.237           | 0.133             | 40.7%    | 5.27%         | 214235        |

### Phase 2: Fold Count Analysis

**Config**: 50 models, tanh signals, 100 features

| Test | Symbol | Folds | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| 2.1  | ES     | 10    | ✅     | 1.077           | 0.840             | 51.0%    | 8.52%         | 214420        |
| 2.2  | ES     | 15    | ✅     | 0.806           | **1.015**   | 50.8%    | 7.70%         | 220935        |
| 2.3  | ES     | 20    | ✅     | 0.941           | **1.162**   | 51.9%    | 9.15%         | 224014        |
| 2.4  | TY     | 10    | ✅     | 1.487           | 1.586             | 52.2%    | 4.42%         | 220621        |
| 2.5  | TY     | 15    | ✅     | 1.282           | 0.925             | 51.4%    | 3.03%         | 223502        |
| 2.6  | TY     | 20    | ✅     | 1.398           | 0.915             | 51.7%    | 3.23%         | 235635        |
| 2.7  | EU     | 10    | ✅     | 1.581           | **1.280**   | 50.9%    | 5.33%         | 220852        |
| 2.8  | EU     | 15    | ✅     | 1.553           | **1.122**   | 51.4%    | 5.28%         | 235644        |
| 2.9  | EU     | 20    | ✅     | 1.455           | **1.183**   | 52.0%    | 4.58%         | 000604        |

*(SD:  TY, it doesn't make sense to go backwards. So i'd say we just use 8 folds.)*

### Phase 3: XGB Architecture Comparison

**Config**: 50 models, 8 folds, 100 features, tanh signals

| Test | Symbol | XGB Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| 3.1  | ES     | tiered   | ✅     | 1.160           | 0.553             | 51.4%    | 9.07%         | 235625        |
| 3.2  | TY     | tiered   | ✅     | 1.150           | **1.609**   | 52.6%    | 3.86%         | 224019        |
| 3.3  | EU     | tiered   | ✅     | 1.365           | **0.789**   | 51.6%    | 4.58%         | 223316        |

### Phase 4: Feature Count Analysis

**Config**: 50 models, 8 folds, tanh signals

| Test | Symbol | Features  | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | --------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| 4.1  | ES     | 250       | ✅     | 0.781           | 0.939             | 48.9%    | 7.94%         | 215056        |
| 4.2  | ES     | -1 (all)  | ✅     | 0.357           | 0.588             | 50.9%    | 3.65%         | 221043        |
| 4.3  | TY     | 250       | ✅     | 1.002           | 1.033             | 51.8%    | 3.07%         | 215107        |
| 4.4  | TY     | -1 (all)  | ✅     | 1.067           | 0.237             | 51.3%    | 1.98%         | 002123        |
| 4.5  | EU     | 250       | ✅     | 0.764           | **1.042**   | 51.9%    | 3.26%         | 001506        |
| 4.6  | EU     | 741 (all) | ✅     | -0.349          | -0.095            | 48.6%    | -0.96%        | 002445        |

### Phase 5: Optimal Configuration Testing (PENDING - Wait for Phase 1-4 completion)

**Strategy**: Use best config from Phases 1-4 for each symbol with increased model counts

**Phase 5 Grid Testing:**

| Test | Symbol | Signal | Folds | Arch     | Features | Models | Status | Production Sharpe | Hit Rate | Log Timestamp |
| ---- | ------ | ------ | ----- | -------- | -------- | ------ | ------ | ----------------- | -------- | ------------- |
| 5.1a | ES     | binary | 20    | standard | 100      | 100    | 🔄     | -                 | -        | -             |
| 5.1b | ES     | binary | 20    | standard | 100      | 150    | 🔄     | -                 | -        | -             |
| 5.2a | TY     | binary | 8     | tiered   | 100      | 100    | 🔄     | -                 | -        | -             |
| 5.2b | TY     | binary | 8     | tiered   | 100      | 150    | 🔄     | -                 | -        | -             |
| 5.3a | EU     | tanh   | 10    | tiered   | 250      | 100    | ⏸️   | -                 | -        | -             |
| 5.3b | EU     | tanh   | 10    | tiered   | 250      | 150    | ⏸️   | -                 | -        | -             |
|      |        |        |       |          |          |        |        |                   |          |               |
|      |        |        |       |          |          |        |        |                   | -        | -             |

---

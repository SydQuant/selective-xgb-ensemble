# TY XGBoost Systematic Testing Plan

## Overview

Comprehensive testing matrix to optimize XGBoost configuration for @TY#C (10-Year Treasury) financial time series prediction. Based on ES framework but optimized for Treasury bond characteristics.

## Current Status: Phase 1 & 2 COMPLETED ✅

- **Completed**: 2025-09-11 16:44
- **Results**: TY significantly more challenging than ES - all configs show poor performance
- **Best TY**: 75M/10F expanding (Sharpe=0.112, Production=-0.103)
- **Recommendation**: Focus optimization efforts on ES, TY needs fundamental strategy revision

---

## Phase 1: Window Strategy Results for @TY#C

**Baseline**: 50 models, 8 folds, standard XGB, 100 features

| Test | Window    | Period       | Full Sharpe      | Train Sharpe | Prod Sharpe         | Train Return | Prod Return      | Overfitting         | Winner  | Log    |
| ---- | --------- | ------------ | ---------------- | ------------ | ------------------- | ------------ | ---------------- | ------------------- | ------- | ------ |
| 1A   | Expanding | All data     | **-0.437** | -0.219       | **-0.767** ❌ | -0.55%       | **-2.61%** | **Medium** ❌ | Poor    | 160309 |
| 1B   | Rolling   | 1yr (252d)   | **-0.514** | -0.819       | **-0.194** ❌ | -1.68%       | **-0.67%** | **Medium** ❌ | Poor    | 160323 |
| 1C   | Rolling   | 1.5yr (378d) | ❌               | ❌           | ❌                  | ❌           | ❌               | ❌                  | Not run | -      |
| 1D   | Rolling   | 2yr (504d)   | ❌               | ❌           | ❌                  | ❌           | ❌               | ❌                  | Not run | -      |

### Key TY Findings:

- **❌ TY very challenging asset**: ALL tested configurations show negative or barely positive Sharpe
- **📈 Scale helps significantly**: 75M/10F (0.112) > 50M/8F (-0.437) = **0.55 Sharpe improvement**
- **⚠️ More models hurt TY**: 100M/10F (0.088) < 75M/10F (0.112) - TY needs fewer models than ES
- **❌ TY window strategy unclear**: Both expanding (-0.437) and rolling (-0.514) poor
- **✅ CB ratios working**: Showing realistic values (0.185, -0.093, 0.023)
- **🎯 Best TY config**: 75M/10F expanding (Sharpe=0.112) but still challenging

### TY vs ES Comparison:

- **ES optimal**: 100M/15F (Production Sharpe: **0.209** ✅)
- **TY optimal**: 75M/10F (Production Sharpe: **-0.103** ❌)
- **ES >> TY**: ES significantly more predictable than Treasury bonds

---

## Phase 2: Scale Optimization for @TY#C

**Baseline**: Expanding window, standard XGB, 100 features

| Test | Models | Folds | Full Sharpe     | Train Sharpe | Prod Sharpe      | Train Return | Prod Return | Log     |
| ---- | ------ | ----- | --------------- | ------------ | ---------------- | ------------ | ----------- | ------- |
| 2A   | 75     | 10    | **0.112** | 0.329        | **-0.103** | 0.66%        | -0.34%      | 160326  |
| 2B   | 100    | 10    | **0.088** | 0.433        | **-0.276** | 0.86%        | -0.85%      | 160332  |
| 2C   | 100    | 15    | ❌              | ❌           | ❌               | ❌           | ❌          | Not run |
| 2D   | 150    | 15    | ❌              | ❌           | ❌               | ❌           | ❌          | Not run |

---

*Last Updated: 2025-09-11 16:50*
*Status: TY optimization complete - recommend focusing on ES*

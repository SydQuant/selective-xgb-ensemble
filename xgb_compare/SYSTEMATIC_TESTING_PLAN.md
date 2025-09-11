# XGBoost Systematic Testing Plan

## Overview
Comprehensive testing matrix to optimize XGBoost configuration for financial time series prediction. Tests are designed to respect parameter dependencies and executed in strategic phases.

## Current Status: Phase 1 In Progress
- **Started**: 2025-09-11 14:55
- **Phase 1 Running**: Expanding vs Rolling window comparison
- **Next**: Scale optimization, then architecture, then features

---

## Phase 1: Window Strategy (CRITICAL FOUNDATION)
**Objective**: Determine optimal training window strategy - impacts all subsequent decisions
**Fixed baseline**: 50 models, 8 folds, standard XGB, 100 features

| Test | Status | Window | Rolling Period | Log Label | Est. Time |
|------|--------|--------|---------------|-----------|-----------|
| 1A | 🔄 Running | Expanding | N/A | `50M_8F_expand_standard_100feat_baseline` | 15-20 min |
| 1B | 🔄 Running | Rolling | 1yr (252 days) | `50M_8F_roll1yr_standard_100feat` | 15-20 min |
| 1C | ⏳ Queued | Rolling | 1.5yr (378 days) | `50M_8F_roll1p5yr_standard_100feat` | 15-20 min |
| 1D | ⏳ Queued | Rolling | 2yr (504 days) | `50M_8F_roll2yr_standard_100feat` | 15-20 min |

**Why First**: Window strategy affects data availability, model stability, and overfitting patterns - fundamental to all other choices.

**Expected Outcome**: Expanding typically wins for financial data, but need to validate optimal rolling period if rolling performs better.

---

## Phase 2: Scale Optimization
**Objective**: Find optimal model count vs fold count balance
**Use**: Best window strategy from Phase 1

| Test | Models | Folds | Log Label Template |
|------|--------|-------|--------------------|
| 2A | 100 | 10 | `100M_10F_[bestwindow]_standard_100feat` |
| 2B | 150 | 15 | `150M_15F_[bestwindow]_standard_100feat` |
| 2C | 50 | 15 | `50M_15F_[bestwindow]_standard_100feat` |

**Why After Window**: Model count and folds interact differently with expanding vs rolling windows. More folds help with rolling windows, more models help with expanding.

**Decision Matrix**:
- If expanding wins: Focus on higher model counts (100-150)
- If rolling wins: Focus on higher fold counts (15+)

---

## Phase 3: Architecture Optimization
**Objective**: Validate XGBoost architecture choice
**Use**: Best window + scale combination from Phases 1-2

| Test | XGB Type | Log Label Template |
|------|----------|--------------------|
| 3A | Tiered | `[bestscale]_[bestwindow]_tiered_100feat` |
| 3B | Deep | `[bestscale]_[bestwindow]_deep_100feat` |

**Why After Scale**: Architecture choice depends on model count:
- Deep requires more models (100+) to be effective
- Tiered works well with medium model counts (50-100)
- Standard is baseline for comparison

---

## Phase 4: Feature Optimization
**Objective**: Final feature count tuning
**Use**: Best window + scale + architecture from Phases 1-3

| Test | Features | Log Label Template |
|------|----------|--------------------|
| 4A | 200 | `[best]_200feat_final` |
| 4B | -1 (all) | `[best]_allfeat_final` |

**Why Last**: Feature count should be optimized after all other architectural decisions are made. More complex architectures can handle more features.

---

## Parameter Space
### Models: 50, 100, 150
- **50**: Fast baseline, good for initial testing
- **100**: Sweet spot for most configurations
- **150**: High diversity, slower but potentially better

### Folds: 8, 10, 15
- **8**: Minimum for reliable CV
- **10**: Good balance
- **15**: Maximum stability, slower

### Window Types:
- **Expanding**: Uses all historical data (typical winner)
- **Rolling 1yr**: Recent data focus, adapts quickly
- **Rolling 1.5yr**: Balance of adaptation and stability  
- **Rolling 2yr**: More stable, less adaptive

### XGB Types:
- **Standard**: Balanced trees, good baseline
- **Tiered**: Progressive complexity layers
- **Deep**: More complex trees, needs more data

### Features: 100, 200, -1 (all ~400)
- **100**: Focused feature set, less noise
- **200**: Extended features, more signal potential
- **All**: Maximum information, risk of overfitting

---

## Execution Strategy

### Resource Management
- **Parallel execution**: 2 tests simultaneously when possible
- **Time estimation**: ~1 hour per test (50M x 8F), ~2 hours (100M x 15F)
- **Total time**: 8-12 hours over 2-3 days

### Decision Points
1. **After Phase 1**: If expanding dominates, skip 1.5yr and 2yr rolling tests
2. **After Phase 2**: If clear winner emerges, skip remaining scale tests
3. **After Phase 3**: If standard XGB performs best, skip deep/tiered tests
4. **Early termination**: Stop if patterns are conclusive

### Success Metrics
- **Primary**: Sharpe ratio (target: >1.5 for production viability)
- **Secondary**: Hit rate, drawdown, stability across folds
- **Tertiary**: Training time, computational efficiency

---

## Historical Context
Based on analysis of successful configurations from logs:
- **Best performers**: 100 models, 15 folds, expanding window, 100-400 features
- **Typical winners**: Expanding > Rolling 2yr > Rolling 1.5yr > Rolling 1yr
- **Architecture**: Standard often wins, Deep requires more models
- **Features**: 100-200 optimal, diminishing returns beyond 400

---

## Current Issues Resolved
✅ **Length mismatch bug**: Fixed double-counting in backtest aggregation
✅ **Index alignment**: Fixed pandas Series comparison issues  
✅ **GPU detection**: Properly handles CUDA unavailable scenarios
✅ **Model selection**: Fixed fold indexing bug in Q-score selection
✅ **Feature selection**: Optimized for efficiency and diversity

---

## Next Actions
1. **Monitor Phase 1**: Check expanding vs rolling 1yr results
2. **Queue Phase 1C,1D**: Start rolling 1.5yr and 2yr tests
3. **Analyze results**: Determine winning window strategy
4. **Design Phase 2**: Customize scale tests based on Phase 1 winner
5. **Execute systematically**: Continue through all phases

---

*Last Updated: 2025-09-11 14:55*
*Status: Phase 1 executing (2/4 tests running)*
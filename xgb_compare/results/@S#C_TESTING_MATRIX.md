# S Symbol (@S#C) Consolidated Testing Matrix

## Overview

**Optimized testing strategy for @S#C** based on comprehensive analysis of ES/TY/EU v2 results. This consolidated approach eliminates redundant tests and focuses on essential configurations to identify optimal performance with minimal time investment.

### Strategy Foundation:
- **✅ Learning-Based**: Each phase uses optimal results from previous phases
- **✅ Consolidated Testing**: Multiple variables tested simultaneously in early phases  
- **✅ Efficiency Focus**: 12 essential tests vs 30+ traditional approach (~60% time savings)
- **✅ Production Sharpe Benchmark**: All decisions based on production performance metrics

### Key Insights from ES/TY/EU v2 Analysis:
- **Signal Type**: tanh consistently outperforms binary across all symbols
- **Model Selection**: SHARPE_Q vs HIT_Q varies by symbol, requires direct testing
- **Architecture**: tiered/deep consistently outperform standard XGB
- **Fold Optimization**: Varies by symbol (ES=15F, TY=10-20F, EU=15-20F)  
- **Feature Efficiency**: 100 features proven effective baseline

---

## Phase 1: Signal Type + Model Selection Consolidation

**Goal**: Determine optimal signal type AND model selection method simultaneously

**Config**: 100 models, 15 folds, 100 features, standard XGB

| Test | Signal | Q-Metric | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S1.1 | tanh   | sharpe   | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S1.2 | tanh   | hit_rate | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S1.3 | binary | sharpe   | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S1.4 | binary | hit_rate | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

### Phase 1 Commands:
```bash
# Consolidated Signal + Method Testing
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@S#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric sharpe --log_label "S1.1_tanh_sharpe"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@S#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "S1.2_tanh_hit"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@S#C" --n_models 100 --n_folds 15 --max_features 100 --binary_signal --q_metric sharpe --log_label "S1.3_binary_sharpe"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@S#C" --n_models 100 --n_folds 15 --max_features 100 --binary_signal --q_metric hit_rate --log_label "S1.4_binary_hit"
```

---

## Phase 2: Architecture Optimization

**Goal**: Test advanced architectures using optimal signal+method from Phase 1

**Config**: 100 models, 15 folds, 100 features, [BEST from Phase 1]

| Test | Architecture | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------------ | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S2.1 | tiered       | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S2.2 | deep         | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

**Note**: Standard architecture baseline established in Phase 1. Only test tiered/deep for efficiency.

---

## Phase 3: Fold Count Refinement

**Goal**: Optimize fold count using best signal+method+architecture from Phase 2

**Config**: 100 models, [BEST architecture], 100 features, [BEST signal+method]

| Test | Folds | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ----- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S3.1 | 10    | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S3.2 | 20    | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

**Note**: 15 folds baseline from Phase 1. Test 10F (efficiency) vs 20F (precision) for refinement.

---

## Phase 4: Feature Count Validation

**Goal**: Test if additional features improve optimal configuration

**Config**: 100 models, [BEST folds], [BEST architecture], [BEST signal+method]

| Test | Features | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | -------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S4.1 | 250      | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S4.2 | -1 (all) | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

**Note**: 100 features baseline established. Only test higher counts if justified by performance.

---

## Phase 5: Final Model Scaling

**Goal**: Scale model count with ultimate optimal configuration

**Config**: [ULTIMATE optimal config from Phases 1-4]

| Test | Models | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S5.1 | 150    | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| S5.2 | 200    | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

---

## Execution Strategy

### **Phase Sequence Rationale:**

1. **Phase 1 (4 tests)**: Foundation - establishes signal type + selection method
2. **Phase 2 (2 tests)**: Architecture - leverages Phase 1 winner for architecture testing  
3. **Phase 3 (2 tests)**: Fold optimization - uses Phase 1+2 optimal config
4. **Phase 4 (2 tests)**: Feature validation - tests if more features help optimal config
5. **Phase 5 (2 tests)**: Final scaling - ultimate performance with enhanced model count

### **Testing Batches:**
- **Batch 1**: Phase 1 (4 tests) - Critical foundation  
- **Batch 2**: Phase 2 + Phase 3 (4 tests) - Architecture + fold optimization
- **Batch 3**: Phase 4 + Phase 5 (4 tests) - Feature + model scaling

### **Key Optimizations:**

**✅ Consolidated Variables**: Test signal type + Q-metric together in Phase 1
**✅ Sequential Optimization**: Each phase builds on previous optimal results  
**✅ Skip Redundant Tests**: No standard architecture testing after Phase 1
**✅ Strategic Baselines**: 15F/100F proven effective starting points
**✅ Early Elimination**: Stop testing variants that underperform

### **Expected Outcomes:**

**Total Tests**: 12 (vs 30+ traditional approach)
**Time Savings**: ~60% reduction  
**Execution Time**: ~8-10 hours total
**Quality**: Focused on essential performance drivers

---

## Resource Management

- **Model Count**: 100 (baseline) → 150/200 (Phase 5 scaling)
- **Concurrent Tests**: Maximum 4 per batch for resource efficiency
- **GPU Utilization**: Sequential processing for stability
- **Results Tracking**: Real-time matrix updates with comprehensive metrics

---

*Created: 2025-09-13*  
*Purpose: Optimized @S#C testing based on ES/TY/EU v2 learnings*  
*Strategy: Consolidated, sequential, efficiency-focused approach*
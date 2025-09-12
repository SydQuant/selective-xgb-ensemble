# HIT_Q Testing Matrix v2

## Overview

Complete retesting of HIT_Q model selection with latest code changes and enhanced configuration:

### Key Changes from v1:
- **✅ Latest Bug Fixes**: Applied all recent code improvements
- **✅ Enhanced Model Count**: 100 models (vs 50 in v1) for better statistical significance  
- **✅ Consistent Features**: 100 features baseline (vs variable in v1)
- **✅ Systematic Testing**: Clean slate approach with current codebase

### Testing Matrix:
- **Symbols**: ES, TY, EU
- **Signal Types**: tanh vs binary 
- **Folds**: 8, 10, 15, 20
- **Features**: 100 (baseline), 250, -1 (all)
- **Architectures**: standard vs tiered vs deep XGB
- **Models**: 100 (enhanced from v1's 50)

---

## Phase 1: Signal Type Comparison (HIT_Q)

**Config**: 100 models, 8 folds, 100 features, HIT_Q selection

| Test | Symbol | Signal Type | Status | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----------- | ------ | ----------------- | -------- | ------------- | ------------- |
| H1.1 | ES     | tanh        | ⏸️     | -                 | -        | -             | -             |
| H1.2 | ES     | binary      | ⏸️     | -                 | -        | -             | -             |
| H1.3 | TY     | tanh        | ⏸️     | -                 | -        | -             | -             |
| H1.4 | TY     | binary      | ⏸️     | -                 | -        | -             | -             |
| H1.5 | EU     | tanh        | ⏸️     | -                 | -        | -             | -             |
| H1.6 | EU     | binary      | ⏸️     | -                 | -        | -             | -             |

### Phase 1 Commands:
```bash
# ES Tests
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "v2_H1.1_ES_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "v2_H1.2_ES_binary"

# TY Tests  
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "v2_H1.3_TY_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "v2_H1.4_TY_binary"

# EU Tests
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "v2_H1.5_EU_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "v2_H1.6_EU_binary"
```

---

## Phase 2: Fold Count Analysis (HIT_Q)

**Config**: 100 models, tanh signals, 100 features, HIT_Q selection

| Test | Symbol | Folds | Status | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----- | ------ | ----------------- | -------- | ------------- | ------------- |
| H2.1 | ES     | 10    | ⏸️     | -                 | -        | -             | -             |
| H2.2 | ES     | 15    | ⏸️     | -                 | -        | -             | -             |
| H2.3 | ES     | 20    | ⏸️     | -                 | -        | -             | -             |
| H2.4 | TY     | 10    | ⏸️     | -                 | -        | -             | -             |
| H2.5 | TY     | 15    | ⏸️     | -                 | -        | -             | -             |
| H2.6 | TY     | 20    | ⏸️     | -                 | -        | -             | -             |
| H2.7 | EU     | 10    | ⏸️     | -                 | -        | -             | -             |
| H2.8 | EU     | 15    | ⏸️     | -                 | -        | -             | -             |
| H2.9 | EU     | 20    | ⏸️     | -                 | -        | -             | -             |

---

## Phase 3: Architecture Analysis (HIT_Q)

**Config**: 100 models, 8 folds, 100 features, tanh signals, HIT_Q selection

| Test | Symbol | XGB Type | Status | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | ----------------- | -------- | ------------- | ------------- |
| H3.1 | ES     | tiered   | ⏸️     | -                 | -        | -             | -             |
| H3.2 | ES     | deep     | ⏸️     | -                 | -        | -             | -             |
| H3.3 | TY     | tiered   | ⏸️     | -                 | -        | -             | -             |
| H3.4 | TY     | deep     | ⏸️     | -                 | -        | -             | -             |
| H3.5 | EU     | tiered   | ⏸️     | -                 | -        | -             | -             |
| H3.6 | EU     | deep     | ⏸️     | -                 | -        | -             | -             |

---

## Phase 4: Feature Count Analysis (HIT_Q)

**Config**: 100 models, 8 folds, tanh signals, HIT_Q selection

| Test | Symbol | Features | Status | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | ----------------- | -------- | ------------- | ------------- |
| H4.1 | ES     | 250      | ⏸️     | -                 | -        | -             | -             |
| H4.2 | ES     | -1 (all) | ⏸️     | -                 | -        | -             | -             |
| H4.3 | TY     | 250      | ⏸️     | -                 | -        | -             | -             |
| H4.4 | TY     | -1 (all) | ⏸️     | -                 | -        | -             | -             |
| H4.5 | EU     | 250      | ⏸️     | -                 | -        | -             | -             |
| H4.6 | EU     | -1 (all) | ⏸️     | -                 | -        | -             | -             |

---

## Phase 5: Optimal Configuration Testing (HIT_Q)

**Strategy**: Use best config from Phases 1-4 for each symbol with 150/200 models

| Test  | Symbol | Config | Models | Status | Production Sharpe | Hit Rate | Log Timestamp |
| ----- | ------ | ------ | ------ | ------ | ----------------- | -------- | ------------- |
| H5.1a | ES     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| H5.1b | ES     | TBD    | 200    | ⏸️     | -                 | -        | -             | 
| H5.2a | TY     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| H5.2b | TY     | TBD    | 200    | ⏸️     | -                 | -        | -             |
| H5.3a | EU     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| H5.3b | EU     | TBD    | 200    | ⏸️     | -                 | -        | -             |

---

## Execution Strategy

### Testing Batches (10 tests per batch):
1. **Batch 1**: Phase 1 (6 tests) + Phase 2 start (4 tests)
2. **Batch 2**: Phase 2 remaining (5 tests) + Phase 3 start (5 tests)
3. **Batch 3**: Phase 3 remaining (1 test) + Phase 4 (6 tests) + Phase 5 start (3 tests)
4. **Batch 4**: Phase 5 remaining (3 tests) + analysis

### Resource Management:
- **Enhanced Models**: 100 models (2x improvement from v1)
- **Execution Time**: ~20-25 hours total for all phases
- **Batch Size**: Maximum 10 concurrent tests
- **Monitor Progress**: Update tables as batches complete

---

*Created: 2025-09-12*  
*Version: 2.0 - Enhanced model count with latest code changes*  
*Purpose: Comprehensive HIT_Q testing with current codebase*
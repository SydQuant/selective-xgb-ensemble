# SHARPE_Q Testing Matrix v2

## Overview

Complete retesting of SHARPE_Q model selection with latest code changes and enhanced configuration:

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

## Phase 1: Signal Type Comparison (SHARPE_Q)

**Config**: 100 models, 8 folds, 100 features, SHARPE_Q selection

| Test | Symbol | Signal Type | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ----------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S1.1 | ES     | tanh        | ✅     | 1.436           | 51.9%        | 1.675             | 52.5%          | 1.490                | 52.1%             | 20250912_185256 |
| S1.2 | ES     | binary      | ✅     | 1.026           | 51.4%        | 1.136             | 52.1%          | 1.058                | 51.6%             | 20250912_185336 |
| S1.3 | TY     | tanh        | ✅     | 1.305           | 51.8%        | 2.042             | 52.4%          | 1.588                | 52.0%             | 20250912_185336 |
| S1.4 | TY     | binary      | ✅     | 0.808           | 51.0%        | 1.792             | 54.3%          | 1.190                | 52.1%             | 20250912_185339 |
| S1.5 | EU     | tanh        | ✅     | 1.446           | 52.6%        | 1.429             | 50.4%          | 1.440                | 51.8%             | 20250912_185420 |
| S1.6 | EU     | binary      | ✅     | 0.943           | 51.7%        | 1.083             | 52.2%          | 0.992                | 51.9%             | 20250912_185420 |

### Phase 1 Commands:
```bash
# ES Tests
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric sharpe --log_label "v2_S1.1_ES_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric sharpe --log_label "v2_S1.2_ES_binary"

# TY Tests  
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric sharpe --log_label "v2_S1.3_TY_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric sharpe --log_label "v2_S1.4_TY_binary"

# EU Tests
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds 8 --max_features 100 --q_metric sharpe --log_label "v2_S1.5_EU_tanh"
cd xgb_compare && ~/anaconda3/python.exe xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds 8 --max_features 100 --binary_signal --q_metric sharpe --log_label "v2_S1.6_EU_binary"
```

---

## Phase 2: Fold Count Analysis (SHARPE_Q)

**Config**: 100 models, tanh signals, 100 features, SHARPE_Q selection

| Test | Symbol | Folds | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ----- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S2.1 | ES     | 10    | ✅     | 1.281           | 52.0%        | 1.529             | 53.6%          | 1.364                | 52.6%             | 20250912_190458 |
| S2.2 | ES     | 15    | ✅     | 0.935           | 51.1%        | 1.984             | 53.2%          | 1.378                | 52.0%             | 20250912_190458 |
| S2.3 | ES     | 20    | ✅     | 0.957           | 52.0%        | 1.564             | 51.8%          | 1.212                | 51.9%             | 20250912_190458 |
| S2.4 | TY     | 10    | ✅     | 1.541           | 53.1%        | 1.499             | 53.8%          | 1.471                | 53.4%             | 20250912_190458 |
| S2.5 | TY     | 15    | ✅     | 1.577           | 55.3%        | 1.329             | 52.0%          | 1.366                | 53.9%             | 20250912_190458 |
| S2.6 | TY     | 20    | ✅     | 1.925           | 55.6%        | 1.450             | 51.6%          | 1.591                | 53.9%             | 20250912_190458 |
| S2.7 | EU     | 10    | ✅     | 1.449           | 51.9%        | 1.545             | 51.8%          | 1.480                | 51.9%             | 20250912_190458 |
| S2.8 | EU     | 15    | ✅     | 1.618           | 52.5%        | 1.752             | 54.6%          | 1.654                | 53.4%             | 20250912_190458 |
| S2.9 | EU     | 20    | ✅     | 1.318           | 52.0%        | 1.877             | 54.6%          | 1.570                | 53.1%             | 20250912_190458 |

---

## Phase 3: Architecture Analysis (SHARPE_Q)

**Config**: 100 models, 8 folds, 100 features, tanh signals, SHARPE_Q selection

| Test | Symbol | XGB Type | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| S3.1 | ES     | tiered   | ✅     | 1.033           | 51.3%        | 2.020             | 54.0%          | 1.450                | 52.4%             | 20250912_213001 |
| S3.2 | ES     | deep     | ✅     | 0.814           | 51.0%        | 2.048             | 53.7%          | 1.305                | 52.1%             | 20250912_213039 |
| S3.3 | TY     | tiered   | ✅     | 1.703           | 53.6%        | 1.423             | 52.3%          | 1.508                | 53.1%             | 20250912_213006 |
| S3.4 | TY     | deep     | ✅     | 1.806           | 54.0%        | 1.408             | 52.9%          | 1.560                | 53.6%             | 20250912_213008 |
| S3.5 | EU     | tiered   | ✅     | 1.650           | 52.8%        | 1.541             | 53.0%          | 1.581                | 52.9%             | 20250912_213113 |
| S3.6 | EU     | deep     | ✅     | 1.547           | 52.9%        | 1.501             | 53.8%          | 1.516                | 53.3%             | 20250912_213008 |

---

## Phase 4: Feature Count Analysis (SHARPE_Q)

**Config**: 100 models, 8 folds, tanh signals, SHARPE_Q selection

| Test | Symbol | Features | Status | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | ----------------- | -------- | ------------- | ------------- |
| S4.1 | ES     | 250      | ⏸️     | -                 | -        | -             | -             |
| S4.2 | ES     | -1 (all) | ⏸️     | -                 | -        | -             | -             |
| S4.3 | TY     | 250      | ⏸️     | -                 | -        | -             | -             |
| S4.4 | TY     | -1 (all) | ⏸️     | -                 | -        | -             | -             |
| S4.5 | EU     | 250      | ⏸️     | -                 | -        | -             | -             |
| S4.6 | EU     | -1 (all) | ⏸️     | -                 | -        | -             | -             |

---

## Phase 5: Optimal Configuration Testing (SHARPE_Q)

**Strategy**: Use best config from Phases 1-4 for each symbol with 150/200 models

| Test  | Symbol | Config | Models | Status | Production Sharpe | Hit Rate | Log Timestamp |
| ----- | ------ | ------ | ------ | ------ | ----------------- | -------- | ------------- |
| S5.1a | ES     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| S5.1b | ES     | TBD    | 200    | ⏸️     | -                 | -        | -             |
| S5.2a | TY     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| S5.2b | TY     | TBD    | 200    | ⏸️     | -                 | -        | -             |
| S5.3a | EU     | TBD    | 150    | ⏸️     | -                 | -        | -             |
| S5.3b | EU     | TBD    | 200    | ⏸️     | -                 | -        | -             |

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
*Purpose: Comprehensive SHARPE_Q testing with current codebase*
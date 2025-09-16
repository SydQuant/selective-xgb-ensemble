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

| Test | Symbol | Signal Type | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ----------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| H1.1 | ES     | tanh        | ✅     | 1.278           | 51.3%        | 1.670             | 52.5%          | 1.376                | 51.7%             | 20250912_185433 |
| H1.2 | ES     | binary      | ✅     | 1.331           | 51.8%        | 1.411             | 53.8%          | 1.353                | 52.5%             | 20250912_185432 |
| H1.3 | TY     | tanh        | ✅     | 1.124           | 51.5%        | 2.172             | 54.0%          | 1.537                | 52.4%             | 20250912_185431 |
| H1.4 | TY     | binary      | ✅     | 0.435           | 50.2%        | 1.741             | 54.0%          | 0.949                | 51.5%             | 20250912_185435 |
| H1.5 | EU     | tanh        | ✅     | 1.584           | 52.6%        | 1.439             | 51.7%          | 1.534                | 52.3%             | 20250912_185435 |
| H1.6 | EU     | binary      | ✅     | 1.062           | 52.4%        | 1.281             | 52.0%          | 1.139                | 52.3%             | 20250912_185435 |

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

| Test | Symbol | Folds | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ----- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| H2.1 | ES     | 10    | ✅     | 1.038           | 51.6%        | 1.406             | 52.6%          | 1.155                | 52.0%             | 20250912_212729 |
| H2.2 | ES     | 15    | ✅     | 0.862           | 50.4%        | 2.169             | 53.2%          | 1.402                | 51.6%             | 20250912_212735 |
| H2.3 | ES     | 20    | ✅     | 0.794           | 50.6%        | 1.654             | 53.7%          | 1.124                | 51.9%             | 20250912_212740 |
| H2.4 | TY     | 10    | ✅     | 1.723           | 54.4%        | 1.297             | 53.0%          | 1.462                | 53.9%             | 20250912_212745 |
| H2.5 | TY     | 15    | ✅     | 1.605           | 53.1%        | 0.936             | 51.9%          | 1.181                | 52.6%             | 20250912_212750 |
| H2.6 | TY     | 20    | ✅     | 1.890           | 54.5%        | 1.374             | 52.1%          | 1.532                | 53.5%             | 20250912_212755 |
| H2.7 | EU     | 10    | ✅     | 1.539           | 51.3%        | 1.163             | 53.3%          | 1.375                | 52.1%             | 20250912_212759 |
| H2.8 | EU     | 15    | ✅     | 1.716           | 52.9%        | 1.364             | 52.6%          | 1.524                | 52.8%             | 20250912_212807 |
| H2.9 | EU     | 20    | ✅     | 1.557           | 51.6%        | 1.242             | 52.4%          | 1.375                | 51.9%             | 20250912_212812 |

---

## Phase 3: Architecture Analysis (HIT_Q)

**Config**: 100 models, 8 folds, 100 features, tanh signals, HIT_Q selection

| Test | Symbol | XGB Type | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| H3.1 | ES     | tiered   | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| H3.2 | ES     | deep     | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| H3.3 | TY     | tiered   | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| H3.4 | TY     | deep     | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| H3.5 | EU     | tiered   | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |
| H3.6 | EU     | deep     | ⏸️     | -               | -            | -                 | -              | -                    | -                 | -             |

---

## Phase 4: Feature Count Analysis (HIT_Q)

**Config**: 100 models, optimal folds, tanh signals, HIT_Q selection

| Test | Symbol | Features | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| H4.1 | ES     | 250      | ✅     | 0.739           | 50.6%        | 1.498             | 50.7%          | 1.053                | 50.6%             | 20250913_081821 |
| H4.2 | ES     | -1 (all) | ✅     | 0.227           | 50.2%        | 0.812             | 49.0%          | 0.475                | 49.7%             | 20250913_081827 |
| H4.3 | TY     | 250      | ✅     | 0.964           | 52.2%        | 1.653             | 53.0%          | 1.294                | 52.5%             | 20250913_081858 |
| H4.4 | TY     | -1 (all) | ✅     | 0.549           | 50.6%        | 0.788             | 50.9%          | 0.656                | 50.7%             | 20250913_082002 |
| H4.5 | EU     | 250      | ✅     | 1.317           | 50.5%        | 1.304             | 53.1%          | 1.292                | 51.6%             | 20250913_082148 |
| H4.6 | EU     | -1 (all) | ✅     | 0.394           | 49.5%        | -0.108            | 50.9%          | 0.133                | 50.1%             | 20250913_082148 |

---

## Phase 5: Optimal Configuration Testing (HIT_Q)

**Strategy**: Use best config from Phases 1-4 for each symbol with 150/200 models

| Test | Symbol | Config | Models | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ---- | ------ | ------ | ------ | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| H5.1 | ES     | 15F+std+100feat | 150 | ✅ | 0.736 | 52.0% | 2.319 | 53.7% | 1.391 | 52.7% | 20250913_161436 |
| H5.2 | ES     | 15F+std+100feat | 200 | ✅ | 0.766 | 52.1% | 2.093 | 55.2% | 1.304 | 53.4% | 20250913_161442 |
| H5.3 | TY     | 20F+std+250feat | 150 | ✅ | 1.262 | 53.7% | 1.672 | 53.8% | 1.429 | 53.7% | 20250913_161449 |
| H5.4 | TY     | 20F+std+250feat | 200 | ✅ | 1.311 | 53.7% | 1.830 | 52.9% | 1.534 | 53.3% | 20250913_161456 |
| H5.5 | EU     | 15F+std+250feat | 150 | ✅ | 1.272 | 51.3% | 1.506 | 54.3% | 1.367 | 52.6% | 20250913_161503 |
| H5.6 | EU     | 15F+std+250feat | 200 | ✅ | 1.272 | 50.6% | 1.574 | 54.7% | 1.403 | 52.3% | 20250913_161510 |

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
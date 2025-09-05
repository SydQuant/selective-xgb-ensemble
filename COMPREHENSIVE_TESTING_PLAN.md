# Comprehensive XGBoost Performance Testing Plan

**Constant Configuration**: 10-year analysis, @ES#C symbol, 2014-01-01 to 2024-01-01

*SD: Backtest period was accidently set to 2024-01-01. I'll leave the last 1.5 years as OOOOOS.*

2014 start is actually 2015.

## Analysis Methodology

**Optimization Scoring System**: Each configuration is evaluated using a weighted multi-criteria scoring approach:

```python
# Weighted Performance Score (0.0 to 1.0)
score = 0.25 × (normalized_sharpe_ratio) +        # 25% Sharpe performance weight
        0.15 × (normalized_hit_rate) +            # 15% Hit rate performance weight  
        0.30 × (1 - normalized_consistency) +      # 30% Consistency weight (lower std dev = better)
        0.30 × (statistical_significance_pct)      # 30% Reliability weight
```

**Key Metrics:**

- **Sharpe Performance**: Average OOS Sharpe ratio across all folds (risk-adjusted returns)
- **Hit Rate Performance**: Average OOS Hit rate across all folds (directional accuracy)
- **Consistency**: Standard deviation of fold-by-fold Sharpe ratios (lower = better)
- **Statistical Significance**: Percentage of folds with positive Sharpe ratios
- **Overall Score**: Weighted combination balancing returns, accuracy, consistency, and reliability

Testing Strategy Overview

This comprehensive testing plan follows a systematic approach to identify optimal configurations across multiple dimensions:

1. **Phase 1**: Fold Optimization (Find optimal number of cross-validation folds) ✅ **COMPLETE - Winner: 10 Folds**
2. **Phase 2**: Model Count Optimization (Find optimal number of XGBoost models) ✅ **COMPLETE - Winner: 25 Models**
3. **Phase 3**: Feature Count Optimization (Compare all features vs feature selection) ⚙️ **RUNNING**
4. **Phase 4**: Architecture Comparison (Standard vs Tiered vs Deep XGBoost)

Each phase builds on the optimal configuration from the previous phase.

---

## Phase 1: Cross-Validation Fold Optimization

**Objective**: Determine optimal number of folds for robust cross-validation

**Configuration**:

- Symbol: @ES#C (2014-01-01 to 2024-01-01)
- Models: 25 (moderate count for initial testing)
- Features: 100 (reduced for faster execution)
- XGBoost: standard architecture

### Test Matrix - Phase 1

| Test ID | Folds | Log File                                                                                              | Notes         |
| ------- | ----- | ----------------------------------------------------------------------------------------------------- | ------------- |
| P1-T1   | 5     | `xgb_analysis_p1_05folds_25models_100feat_standard_ES_standard_100feat_5folds_20250905_135341.log`  | Minimum folds |
| P1-T2   | 10    | `xgb_analysis_p1_10folds_25models_100feat_standard_ES_standard_100feat_10folds_20250905_135346.log` | Standard CV   |
| P1-T3   | 15    | `xgb_analysis_p1_15folds_25models_100feat_standard_ES_standard_100feat_15folds_20250905_135352.log` | High folds    |

### Phase 1 Results Summary

| Metric                                | 5 Folds | 10 Folds        | 15 Folds | Winner             |
| ------------------------------------- | ------- | --------------- | -------- | ------------------ |
| Avg OOS Sharpe                        | -0.075  | +0.050          | +0.025   | 10_folds           |
| Avg OOS Hit Rate                      | 50.0%   | 50.6%           | 50.4%    | 10_folds           |
| Sharpe Consistency (StdDev)           | 0.700   | 0.852           | 1.181    | 5_folds            |
| Statistical Significance (% positive) | 55.0%   | 52.1%           | 52.2%    | 5_folds            |
| **Overall Score**              | 0.195   | **0.768** | 0.560    | **10_folds** |

**Phase 1 Winner**: **10 Folds** (Best balance of performance, consistency, and reliability)

---

## Phase 2: Model Count Optimization

**Objective**: Determine optimal number of XGBoost models for ensemble

*SD: This step is unnecessary. We are running individual eval so it doens't matter if we run 500 models at once.*

I messed up this step

*We should do another test of [5,10,15,20] for top_k (stability) or n_select (GROPE)*

**Configuration** (using optimal folds from Phase 1):

- Symbol: @ES#C (2014-01-01 to 2024-01-01)
- Folds: **10** (optimal from Phase 1)
- Features: 100
- XGBoost: standard architecture

### Test Matrix - Phase 2

| Test ID | Models | Log File                                                                   | Notes             |
| ------- | ------ | -------------------------------------------------------------------------- | ----------------- |
| P2-T1   | 25     | `xgb_analysis_p2_25models_standard_100feat_10folds_20250905_140627.log`  | Fast ensemble     |
| P2-T2   | 50     | `xgb_analysis_p2_50models_standard_100feat_10folds_20250905_140632.log`  | Standard ensemble |
| P2-T3   | 75     | `xgb_analysis_p2_75models_standard_100feat_10folds_20250905_140636.log`  | Large ensemble    |
| P2-T4   | 100    | `xgb_analysis_p2_100models_standard_100feat_10folds_20250905_140641.log` | Maximum ensemble  |

### Phase 2 Results Summary

| Metric                                | 25 Models | 50 Models       | 75 Models | 100 Models | Winner              |
| ------------------------------------- | --------- | --------------- | --------- | ---------- | ------------------- |
| Avg OOS Sharpe                        | +0.050    | +0.128          | +0.115    | +0.115     | 50_models           |
| Avg OOS Hit Rate                      | 50.6%     | 50.6%           | 50.5%     | 50.5%      | 25/50_models        |
| Sharpe Consistency (StdDev)           | 0.852     | 0.873           | 0.885     | 0.895      | 25_models           |
| Statistical Significance (% positive) | 52.1%     | 56.3%           | 57.1%     | 57.4%      | 100_models          |
| **Overall Score**               | 0.409     | **0.604** | 0.573     | 0.571      | **50_models** |

**Phase 2 Winner**: **50 Models** (Best balance of performance and consistency)

---

## Phase 3: Feature Count Optimization

**Objective**: Compare different feature selection strategies

*SD: max_features=-1 means "no limit"  but block_wise_feature_selection() still applies with correlation threshold 0.7, It's both a bug and a 'feature' I'd say.     -- no-feature-selection is literally using all raw features. ~1k in total.*

**Configuration** (using optimal folds & models from Phases 1-2):

- Symbol: @ES#C (2014-01-01 to 2024-01-01)
- Folds: **10** (optimal from Phase 1)
- Models: **50** (corrected from Phase 2)
- XGBoost: standard architecture

### Test Matrix - Phase 3 CORRECTED

| Test ID | Features | Selection Method          | Log File                                                                                            | Notes                  | Status                |
| ------- | -------- | ------------------------- | --------------------------------------------------------------------------------------------------- | ---------------------- | --------------------- |
| P3-T1   | 50       | Block-wise double-filter  | `xgb_performance_corrected_p3_50feat_standard_50feat_10folds_20250905_151711.log`                 | Conservative selection | ✅**COMPLETE**  |
| P3-T2   | 100      | Block-wise double-filter  | `xgb_performance_corrected_p3_100feat_standard_100feat_10folds_20250905_151716.log`               | Standard selection     | ✅**COMPLETE**  |
| P3-T3   | 150      | Block-wise double-filter | `xgb_performance_corrected_p3_150feat_standard_150feat_10folds_20250905_151722.log`               | Generous selection     | ✅**COMPLETE**  |
| P3-T4   | 389      | Block-wise filtered       | `xgb_performance_corrected_p3_ALLfeat_standard_-1feat_10folds_20250905_151728.log` (389 features) | All filtered features  | ✅**COMPLETE**  |
| P3-T5   | ALL      | No selection              | `xgb_performance_p3_T5_ALLraw_feat_standard_50feat_10folds_20250905_HHMMSS.log` (1054 features)   | All raw features       | ⚙️**RUNNING** |

### Phase 3 Results Summary ✅ **COMPLETE**

| Metric                                | 50 Features | 100 Features | 150 Features | All Filtered Features | All Raw Features | Winner                 |
| ------------------------------------- | ----------- | ------------ | ------------ | --------------------- | ---------------- | ---------------------- |
| Features Used                         | 50          | 100          | 150          | 389                   | 1054             | -                      |
| Avg OOS Sharpe                        | +0.499      | +0.298       | +0.356       | +0.528                | TBD              | **ALL_filtered** |
| Avg OOS Hit Rate                      | 50.0%       | 50.0%        | 50.0%        | 50.0%                 | TBD              | **ALL_tied**     |
| Sharpe Consistency (StdDev)           | 1.190       | 1.109        | 1.049        | 1.187                 | TBD              | **150_features** |
| Statistical Significance (% positive) | 63.4%       | 56.8%        | 59.3%        | 64.4%                 | TBD              | **ALL_filtered** |
| **Overall Score**               | 0.684       | 0.598        | 0.646        | **0.701**       | TBD              | **ALL_filtered** |

**Phase 3 Current Winner**: **All Filtered Features (389)** (Score: 0.701, OOS Sharpe: +0.528) - *Pending P3-T5 completion*

---

## Phase 4: Architecture Comparison ✅ **READY TO LAUNCH**

**Objective**: Compare XGBoost architectures using optimal configuration from Phases 1-3

**Configuration** (using confirmed optimal settings from Phases 1-3):

- Symbol: @ES#C (2014-01-01 to 2024-01-01)
- Folds: **10** (optimal from Phase 1)
- Models: **50** (optimal from Phase 2)
- Features: **ALL (389)** (confirmed optimal from Phase 3)

### Test Matrix - Phase 4 READY

| Test ID | Architecture | Description         | Optimal Configuration Command                                     | Notes                    | Status              |
| ------- | ------------ | ------------------- | ----------------------------------------------------------------- | ------------------------ | ------------------- |
| P4-T1   | Standard     | Balanced parameters | `--target_symbol "@ES#C" --n_models 50 --folds 10`              | Baseline architecture    | ⚙️**READY** |
| P4-T2   | Tiered       | Multi-tier ensemble | `--target_symbol "@ES#C" --n_models 50 --folds 10 --tiered_xgb` | Alternative architecture | ⚙️**READY** |
| P4-T3   | Deep         | Deeper trees        | `--target_symbol "@ES#C" --n_models 50 --folds 10 --deep_xgb`   | Complex architecture     | ⚙️**READY** |

**Configuration**: Using ALL features (389) - optimal from Phase 3 confirmed analysis.

### Phase 4 Results Summary

| Metric                                | Standard | Tiered | Deep            | Winner                      |
| ------------------------------------- | -------- | ------ | --------------- | --------------------------- |
| Avg OOS Sharpe                        | 4.178    | 3.886  | 4.519           | deep_architecture           |
| Avg OOS Hit Rate                      | 62.1%    | 62.1%  | 61.3%           | standard/tiered             |
| Sharpe Consistency (StdDev)           | 0.864    | 2.327  | 0.937           | standard_architecture       |
| Statistical Significance (% positive) | 100.0%   | 100.0% | 100.0%          | ALL_TIED                    |
| **Overall Score**               | 0.870    | 0.665  | **0.877** | **deep_architecture** |

**Phase 4 Winner**: **Deep Architecture** (Best balance of performance, consistency, and reliability)

### P4-T99 Validation Test Results

**Validation Test Confirmation**: P4-T99 using optimal parameters (Deep XGBoost, 10 folds, 25 models, ALL features) achieved **identical results** to P4-T3:

- **Avg OOS Sharpe**: 4.519 (matching P4-T3)
- **Consistency**: 0.937 StdDev (matching P4-T3)
- **Hit Rate**: 61.3% (matching P4-T3)
- **Statistical Significance**: 100% (all 6 folds positive)

---

## Final Optimal Configuration

**CORRECTED Final Optimal Configuration** (Based on Proper Multi-Year Cross-Validation):

- **Optimal Folds**: **10** (Winner from Phase 1 with score: 0.768, OOS Sharpe: +0.050)
- **Optimal Models**: **50** (Winner from Phase 2 with score: 0.604, OOS Sharpe: +0.128)
- **Optimal Features**: **ALL Filtered (389)** (Current winner from Phase 3 with score: 0.701, OOS Sharpe: +0.528) ⚙️ **PENDING P3-T5**
- **Optimal Architecture**: **TBD** (Status: Ready for Phase 4 analysis)

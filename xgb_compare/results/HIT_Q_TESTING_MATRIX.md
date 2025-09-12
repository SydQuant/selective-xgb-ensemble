# Hit_Q Testing Matrix

## Overview

Complete replication of the corrected logic testing matrix using **hit_Q** instead of sharpe_Q for model selection.
This will test if hit rate-based model selection produces different/better results than Sharpe-based selection.

## Baseline Results (sharpe_Q) - For Comparison

| Symbol       | Training Sharpe | **Production Sharpe (Baseline)** | Hit Rate | Annual Return | Log Timestamp |
| ------------ | --------------- | -------------------------------------- | -------- | ------------- | ------------- |
| **ES** | 1.571           | **0.996**                        | 50.6%    | 12.77%        | 201417        |
| **TY** | 0.855           | **1.609**                        | 52.6%    | 3.43%         | 202106        |
| **EU** | 1.563           | **0.740**                        | 52.6%    | 4.88%         | 202110        |

**Goal**: Test if hit_Q selection beats these baselines

---

## Testing Matrix - Hit_Q Model Selection

### Phase 1: Signal Type Comparison (hit_Q)

**Config**: 50 models, 8 folds, 100 features, **--q_metric hit_rate**

| Test | Symbol | Signal Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| H1.1 | ES     | tanh        | ✅     | 0.653           | **1.737**   | 52.1%    | 8.47%         | 233856        |
| H1.2 | ES     | binary      | ✅     | 0.468           | **1.754**   | 33.6%    | 11.06%        | 235055        |
| H1.3 | TY     | tanh        | ✅     | 0.962           | 1.609             | 52.9%    | 3.65%         | 233904        |
| H1.4 | TY     | binary      | ✅     | 1.088           | 1.670             | 37.3%    | 5.92%         | 235054        |
| H1.5 | EU     | tanh        | ✅     | 1.375           | 0.134             | 49.7%    | 3.51%         | 235028        |
| H1.6 | EU     | binary      | ✅     | 1.280           | 0.035             | 34.5%    | 4.73%         | 235057        |

### Phase 2: Fold Count Analysis (hit_Q, tanh signals)

**Config**: 50 models, tanh signals, 100 features, **--q_metric hit_rate**

| Test | Symbol | Folds | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | ----- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| H2.1 | ES     | 10    | ✅     | 1.005           | **1.369**   | 50.7%    | 12.13%        | 012354        |
| H2.2 | ES     | 15    | ✅     | 0.867           | **1.242**   | 50.2%    | 11.39%        | 012406        |
| H2.3 | ES     | 20    | ✅     | 0.808           | **1.277**   | 52.8%    | 10.46%        | 013337        |
| H2.4 | TY     | 10    | ✅     | 1.374           | **1.477**   | 52.5%    | 5.68%         | 012454        |
| H2.5 | TY     | 15    | ✅     | 0.841           | 0.732             | 54.4%    | 2.83%         | 013345        |
| H2.6 | TY     | 20    | ✅     | 1.323           | **1.315**   | 50.7%    | 4.93%         | 091216        |
| H2.7 | EU     | 10    | ✅     | 1.532           | **1.485**   | 51.5%    | 6.21%         | 023134        |
| H2.8 | EU     | 15    | ✅     | 1.717           | **1.019**   | 52.3%    | 4.67%         | 091216        |
| H2.9 | EU     | 20    | ✅     | 1.156           | **1.192**   | 53.2%    | 5.17%         | 091216        |

### Phase 3: Architecture Analysis (hit_Q, tanh signals)

**Config**: 50 models, 8 folds, 100 features, tanh signals, **--q_metric hit_rate**

| Test | Symbol | XGB Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| H3.1 | ES     | standard | ✅     | 0.653           | **1.737**   | 53.1%    | 12.46%        | 014648        |
| H3.2 | ES     | tiered   | ✅     | 1.105           | **1.938**   | 50.5%    | 15.00%        | 012425        |
| H3.3 | ES     | deep     | ✅     | 1.045           | **2.072**   | 52.5%    | 16.94%        | 014650        |
| H3.4 | TY     | standard | ✅     | 1.003           | **1.609**   | 56.3%    | 5.64%         | 014747        |
| H3.5 | TY     | tiered   | ✅     | 1.072           | **1.754**   | 53.7%    | 6.44%         | 014650        |
| H3.6 | TY     | deep     | ✅     | 0.708           | **1.456**   | 52.7%    | 5.39%         | 014746        |
| H3.7 | EU     | standard | ✅     | 1.375           | 0.134             | 48.5%    | 0.53%         | 023133        |
| H3.8 | EU     | tiered   | ✅     | 1.296           | **1.075**   | 52.2%    | 4.40%         | 023133        |
| H3.9 | EU     | deep     | ✅     | 0.970           | 0.594             | 51.4%    | 2.20%         | 012521        |

### Phase 4: Feature Count Analysis (hit_Q, tanh signals)

**Config**: 50 models, 8 folds, tanh signals, **--q_metric hit_rate**

| Test | Symbol | Features | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
| ---- | ------ | -------- | ------ | --------------- | ----------------- | -------- | ------------- | ------------- |
| H4.1 | ES     | 100      | ✅     | 0.653           | **1.737**   | 53.1%    | 12.46%        | 023133        |
| H4.2 | ES     | 250      | ✅     | 1.231           | **1.063**   | 51.8%    | 8.22%         | 014650        |
| H4.3 | ES     | -1 (all) | ✅     | 0.418           | 0.289             | 49.5%    | 1.98%         | 023133        |
| H4.4 | TY     | 100      | ✅     | 1.003           | **1.609**   | 56.3%    | 5.64%         | 023133        |
| H4.5 | TY     | 250      | ✅     | 0.902           | 0.950             | 50.9%    | 3.14%         | 014650        |
| H4.6 | TY     | -1 (all) | ✅     | 1.074           | 0.092             | 50.8%    | 0.30%         | 023133        |
| H4.7 | EU     | 100      | ✅     | 1.375           | 0.134             | 48.5%    | 0.53%         | 023133        |
| H4.8 | EU     | 250      | ✅     | 0.975           | 0.754             | 51.4%    | 2.94%         | 014747        |
| H4.9 | EU     | -1 (all) | ✅     | -0.233          | -0.077            | 50.5%    | -0.31%        | 023134        |

### Phase 5: Optimal Hit_Q Configuration Testing

**Strategy**: Use BEST config from each phase, scale to 100/150 models

#### **Optimal Hit_Q Configurations (Best from Each Phase):**

- **ES**: Binary + 10 folds + Deep + 100 features (combines all phase winners)
- **TY**: Binary + 10 folds + Tiered + 100 features (combines all phase winners)
- **EU**: Tanh + 10 folds + Tiered + 250 features (combines all phase winners)

#### **Phase 5 Testing Matrix:**

| Test  | Symbol | Signal | Folds | Arch   | Features | Models | Status | Production Sharpe | Hit Rate | Log Timestamp |
| ----- | ------ | ------ | ----- | ------ | -------- | ------ | ------ | ----------------- | -------- | ------------- |
| H5.1a | ES     | binary | 10    | deep   | 100      | 100    | ⏸️   | -                 | -        | -             |
| H5.1b | ES     | binary | 10    | deep   | 100      | 150    | ⏸️   | -                 | -        | -             |
| H5.2a | TY     | binary | 10    | tiered | 100      | 100    | ⏸️   | -                 | -        | -             |
| H5.2b | TY     | binary | 10    | tiered | 100      | 150    | ⏸️   | -                 | -        | -             |
| H5.3a | EU     | tanh   | 10    | tiered | 250      | 100    | ⏸️   | -                 | -        | -             |
| H5.3b | EU     | tanh   | 10    | tiered | 250      | 150    | ⏸️   | -                 | -        | -             |

#### **Phase 5 Commands:**

```bash
# ES Optimal (Binary + 10F + Deep + 100feat)
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds 10 --max_features 100 --xgb_type deep --binary_signal --q_metric hit_rate --log_label "hitQ_ES_optimal_100M"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 150 --n_folds 10 --max_features 100 --xgb_type deep --binary_signal --q_metric hit_rate --log_label "hitQ_ES_optimal_150M"

# TY Optimal (Binary + 10F + Tiered + 100feat)
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds 10 --max_features 100 --xgb_type tiered --binary_signal --q_metric hit_rate --log_label "hitQ_TY_optimal_100M"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 150 --n_folds 10 --max_features 100 --xgb_type tiered --binary_signal --q_metric hit_rate --log_label "hitQ_TY_optimal_150M"

# EU Optimal (Tanh + 10F + Tiered + 250feat)  
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds 10 --max_features 250 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_EU_optimal_100M"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 150 --n_folds 10 --max_features 250 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_EU_optimal_150M"
```

---

## Command Matrix - Hit_Q Testing

### Phase 5: Optimal Config + Model Count (hit_Q)

**Strategy**: Use best hit_Q config from phases 1-4, scale to 100/150 models

```bash
# TO BE DETERMINED based on Phase 1-4 hit_Q results
# Example commands (update configs based on hit_Q winners):

# ES Optimal hit_Q Config:
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 100 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_ES_100models_optimal"
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 150 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_ES_150models_optimal"

# TY Optimal hit_Q Config:
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 100 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_TY_100models_optimal" 
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 150 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_TY_150models_optimal"

# EU Optimal hit_Q Config:
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 100 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_EU_100models_optimal"
# cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 150 --n_folds [BEST] --max_features [BEST] --xgb_type [BEST] --q_metric hit_rate --log_label "hitQ_EU_150models_optimal"
```

---

## Execution Strategy

### Batch Approach

1. **Phase 1**: Run 6 signal type tests (3 symbols × tanh/binary)
2. **Phase 2**: Run 9 fold count tests (3 symbols × 10/15/20 folds)
3. **Phase 3**: Run 9 architecture tests (3 symbols × standard/tiered/deep)
4. **Phase 4**: Run 9 feature tests (3 symbols × 100/250/all features)
5. **Phase 5**: Run 6 optimal config tests (3 symbols × 100/150 models)

### Expected Insights

1. **Hit_Q vs Sharpe_Q**: Does hit rate-based model selection improve consistency?
2. **Model Selection Differences**: Which models get selected with hit_Q vs sharpe_Q?
3. **Performance Comparison**: Better hit rates with hit_Q selection?
4. **Optimal Configurations**: Best config per symbol for each metric type

### Resource Management

- **Total tests**: 33 hit_Q tests
- **Execution time**: ~15-20 hours total
- **Run in batches**: 3-4 tests simultaneously
- **Monitor progress**: Update tables as each completes

---

## Phase 1 Results Analysis - Hit_Q vs Sharpe_Q

### Hit_Q vs Sharpe_Q Comparison (Tanh Signals)

| Symbol       | Metric   | Training Sharpe | Production Sharpe | Hit Rate | Improvement                 |
| ------------ | -------- | --------------- | ----------------- | -------- | --------------------------- |
| **ES** | sharpe_Q | 1.571           | **0.996**   | 50.6%    | (baseline)                  |
| **ES** | hit_Q    | 0.653           | **1.737**   | 52.1%    | **+74% Sharpe** ✅    |
| **TY** | sharpe_Q | 0.855           | **1.609**   | 52.6%    | (baseline)                  |
| **TY** | hit_Q    | 0.962           | **1.609**   | 52.9%    | **+0.3% hit rate** ✅ |
| **EU** | sharpe_Q | 1.563           | **0.740**   | 52.6%    | (baseline)                  |
| **EU** | hit_Q    | 1.375           | **0.134**   | 49.7%    | **-81% Sharpe** ❌    |

### Binary Signal Results (Still Problematic)

| Symbol       | Metric | Hit Rate | vs Tanh Hit Rate | Issue Persists         |
| ------------ | ------ | -------- | ---------------- | ---------------------- |
| **ES** | hit_Q  | 33.6%    | 52.1%            | **-18.5 pts** ❌ |
| **TY** | hit_Q  | 37.3%    | 52.9%            | **-15.6 pts** ❌ |
| **EU** | hit_Q  | 34.5%    | 49.7%            | **-15.2 pts** ❌ |

### Key Findings

✅ **Hit_Q Benefits:**

- **ES**: Dramatically better production Sharpe (0.996 → 1.737)
- **TY**: Slightly better hit rate (52.6% → 52.9%)
- **Different model selection**: Hit_Q chooses models optimized for directional accuracy

❌ **Hit_Q Drawbacks:**

- **EU**: Much worse production performance (0.740 → 0.134)
- **Market dependent**: Hit_Q benefits vary significantly by symbol

🚨 **Binary Signal Issue Persists:**

- Even with hit_Q selection, binary signals still show ~33-37% hit rates
- Problem is deeper than model selection - likely in signal generation/combination logic

---

*Phase 1 completed: Hit_Q shows mixed results - beneficial for ES, neutral for TY, harmful for EU*

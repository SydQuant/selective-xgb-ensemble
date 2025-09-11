# Hit_Q Testing Matrix

## Overview
Complete replication of the corrected logic testing matrix using **hit_Q** instead of sharpe_Q for model selection.
This will test if hit rate-based model selection produces different/better results than Sharpe-based selection.

## Baseline Results (sharpe_Q) - For Comparison
| Symbol | Training Sharpe | **Production Sharpe (Baseline)** | Hit Rate | Annual Return | Log Timestamp |
|--------|----------------|-----------------------------------|----------|---------------|---------------|
| **ES** | 1.571 | **0.996** | 50.6% | 12.77% | 201417 |
| **TY** | 0.855 | **1.609** | 52.6% | 3.43% | 202106 |
| **EU** | 1.563 | **0.740** | 52.6% | 4.88% | 202110 |

**Goal**: Test if hit_Q selection beats these baselines

---

## Testing Matrix - Hit_Q Model Selection

### Phase 1: Signal Type Comparison (hit_Q)
**Config**: 50 models, 8 folds, 100 features, **--q_metric hit_rate**

| Test | Symbol | Signal Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
|------|--------|-------------|--------|----------------|-------------------|----------|---------------|---------------|
| H1.1 | ES     | tanh        | ✅     | 0.653          | **1.737**         | 52.1%    | 8.47%         | 233856 |
| H1.2 | ES     | binary      | ✅     | 0.468          | **1.754**         | 33.6%    | 11.06%        | 235055 |
| H1.3 | TY     | tanh        | ✅     | 0.962          | 1.609             | 52.9%    | 3.65%         | 233904 |
| H1.4 | TY     | binary      | ✅     | 1.088          | 1.670             | 37.3%    | 5.92%         | 235054 |
| H1.5 | EU     | tanh        | ✅     | 1.375          | 0.134             | 49.7%    | 3.51%         | 235028 |
| H1.6 | EU     | binary      | ✅     | 1.280          | 0.035             | 34.5%    | 4.73%         | 235057 |

### Phase 2: Fold Count Analysis (hit_Q, tanh signals)
**Config**: 50 models, tanh signals, 100 features, **--q_metric hit_rate**

| Test | Symbol | Folds | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
|------|--------|-------|--------|----------------|-------------------|----------|---------------|---------------|
| H2.1 | ES     | 10    | ✅     | 1.005          | **1.369**         | 50.7%    | 12.13%        | 012354 |
| H2.2 | ES     | 15    | ✅     | 0.867          | **1.242**         | 50.2%    | 11.39%        | 012406 |
| H2.3 | ES     | 20    | ✅     | 0.808          | **1.277**         | 52.8%    | 10.46%        | 013337 |
| H2.4 | TY     | 10    | ✅     | 1.374          | **1.477**         | 52.5%    | 5.68%         | 012454 |
| H2.5 | TY     | 15    | ✅     | 0.841          | 0.732             | 54.4%    | 2.83%         | 013345 |
| H2.6 | TY     | 20    | ⏸️     | -              | -                 | -        | -             | - |
| H2.7 | EU     | 10    | ✅     | 1.532          | **1.485**         | 51.5%    | 6.21%         | 023134 |
| H2.8 | EU     | 15    | ⏸️     | -              | -                 | -        | -             | - |
| H2.9 | EU     | 20    | ⏸️     | -              | -                 | -        | -             | - |

### Phase 3: Architecture Analysis (hit_Q, tanh signals)
**Config**: 50 models, 8 folds, 100 features, tanh signals, **--q_metric hit_rate**

| Test | Symbol | XGB Type | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
|------|--------|----------|--------|----------------|-------------------|----------|---------------|---------------|
| H3.1 | ES     | standard | ✅     | 0.653          | **1.737**         | 53.1%    | 12.46%        | 014648 |
| H3.2 | ES     | tiered   | ✅     | 1.105          | **1.938**         | 50.5%    | 15.00%        | 012425 |
| H3.3 | ES     | deep     | 🔄     | -              | -                 | -        | -             | - |
| H3.4 | TY     | standard | 🔄     | -              | -                 | -        | -             | - |
| H3.5 | TY     | tiered   | 🔄     | -              | -                 | -        | -             | - |
| H3.6 | TY     | deep     | 🔄     | -              | -                 | -        | -             | - |
| H3.7 | EU     | standard | ⏸️     | -              | -                 | -        | -             | - |
| H3.8 | EU     | tiered   | ✅     | 1.296          | **1.075**         | 52.2%    | 4.40%         | 023133 |
| H3.9 | EU     | deep     | ✅     | 0.970          | 0.594             | 51.4%    | 2.20%         | 012521 |

### Phase 4: Feature Count Analysis (hit_Q, tanh signals)
**Config**: 50 models, 8 folds, tanh signals, **--q_metric hit_rate**

| Test | Symbol | Features | Status | Training Sharpe | Production Sharpe | Hit Rate | Annual Return | Log Timestamp |
|------|--------|----------|--------|----------------|-------------------|----------|---------------|---------------|
| H4.1 | ES     | 100      | ⏸️     | -              | -                 | -        | -             | - |
| H4.2 | ES     | 250      | 🔄     | -              | -                 | -        | -             | - |
| H4.3 | ES     | -1 (all) | ✅     | 0.418          | 0.289             | 49.5%    | 1.98%         | 023133 |
| H4.4 | TY     | 100      | ⏸️     | -              | -                 | -        | -             | - |
| H4.5 | TY     | 250      | 🔄     | -              | -                 | -        | -             | - |
| H4.6 | TY     | -1 (all) | ✅     | 1.074          | 0.092             | 50.8%    | 0.30%         | 023133 |
| H4.7 | EU     | 100      | ⏸️     | -              | -                 | -        | -             | - |
| H4.8 | EU     | 250      | 🔄     | -              | -                 | -        | -             | - |
| H4.9 | EU     | -1 (all) | ✅     | -0.233         | -0.077            | 50.5%    | -0.31%        | 023134 |

---

## Command Matrix - Hit_Q Testing

### Phase 1: Signal Type (hit_Q)
```bash
# Test H1.1: ES Tanh with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_tanh_baseline"

# Test H1.2: ES Binary with hit_Q  
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "hitQ_ES_binary_voting"

# Test H1.3: TY Tanh with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_tanh_baseline"

# Test H1.4: TY Binary with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "hitQ_TY_binary_voting"

# Test H1.5: EU Tanh with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_tanh_baseline"

# Test H1.6: EU Binary with hit_Q
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --binary_signal --q_metric hit_rate --log_label "hitQ_EU_binary_voting"
```

### Phase 2: Fold Count (hit_Q, tanh)
```bash
# ES Fold Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 10 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_10folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_15folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 20 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_20folds"

# TY Fold Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 10 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_10folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_15folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 20 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_20folds"

# EU Fold Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 10 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_10folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_15folds"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 20 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_20folds"
```

### Phase 3: Architecture (hit_Q, tanh)
```bash
# ES Architecture Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_standard"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_ES_tiered"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type deep --q_metric hit_rate --log_label "hitQ_ES_deep"

# TY Architecture Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_standard"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_TY_tiered"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type deep --q_metric hit_rate --log_label "hitQ_TY_deep"

# EU Architecture Variations  
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_standard"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type tiered --q_metric hit_rate --log_label "hitQ_EU_tiered"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --xgb_type deep --q_metric hit_rate --log_label "hitQ_EU_deep"
```

### Phase 4: Feature Count (hit_Q, tanh)  
```bash
# ES Feature Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_ES_100feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features 250 --q_metric hit_rate --log_label "hitQ_ES_250feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@ES#C" --n_models 50 --n_folds 8 --max_features -1 --q_metric hit_rate --log_label "hitQ_ES_allfeat"

# TY Feature Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_TY_100feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features 250 --q_metric hit_rate --log_label "hitQ_TY_250feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@TY#C" --n_models 50 --n_folds 8 --max_features -1 --q_metric hit_rate --log_label "hitQ_TY_allfeat"

# EU Feature Variations
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 100 --q_metric hit_rate --log_label "hitQ_EU_100feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features 250 --q_metric hit_rate --log_label "hitQ_EU_250feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@EU#C" --n_models 50 --n_folds 8 --max_features -1 --q_metric hit_rate --log_label "hitQ_EU_allfeat"
```

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

| Symbol | Metric | Training Sharpe | Production Sharpe | Hit Rate | Improvement |
|--------|--------|----------------|-------------------|----------|-------------|
| **ES** | sharpe_Q | 1.571 | **0.996** | 50.6% | (baseline) |
| **ES** | hit_Q | 0.653 | **1.737** | 52.1% | **+74% Sharpe** ✅ |
| **TY** | sharpe_Q | 0.855 | **1.609** | 52.6% | (baseline) |
| **TY** | hit_Q | 0.962 | **1.609** | 52.9% | **+0.3% hit rate** ✅ |
| **EU** | sharpe_Q | 1.563 | **0.740** | 52.6% | (baseline) |
| **EU** | hit_Q | 1.375 | **0.134** | 49.7% | **-81% Sharpe** ❌ |

### Binary Signal Results (Still Problematic)

| Symbol | Metric | Hit Rate | vs Tanh Hit Rate | Issue Persists |
|--------|--------|----------|------------------|----------------|
| **ES** | hit_Q | 33.6% | 52.1% | **-18.5 pts** ❌ |
| **TY** | hit_Q | 37.3% | 52.9% | **-15.6 pts** ❌ |
| **EU** | hit_Q | 34.5% | 49.7% | **-15.2 pts** ❌ |

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
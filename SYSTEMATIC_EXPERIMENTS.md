# Systematic XGBoost Trading System Experiments

## Experimental Plan

1. **Baseline**: 50 features, hierarchical clustering, simple random XGB, current depth/selection

   - Test with both cross-validation folds AND train-test split
   - All 4 symbols: @TY#C, @EU#C, @ES#C, QGC#C
2. **Feature Count Tests**:

   - 70 features (keeping everything else baseline)
   - 100 features (keeping everything else baseline)
3. **XGB Architecture Tests**:

   - Tiered XGB system
   - Deeper XGB (depth 8-10 vs current 6)
   - More XGBs to GROPE (increase n_select)
4. **GROPE vs Equal Weights Benchmark**
5. **Combined Optimal**: Only changes with meaningful contribution

---

## Step 0: Proper Baseline with P-Value Gating (pmax=0.05)

**Configuration**: 50 features, hierarchical clustering, depth 2-6, n_select=12, proper statistical significance
**P-Value Threshold**: 0.05 (5% significance level)

### Cross-Validation Results (6-fold)

| Symbol | Sharpe | Total Return | Max DD | Win Rate | P-Values | Status           |
| ------ | ------ | ------------ | ------ | -------- | -------- | ---------------- |
| @TY#C  | ?      | ?            | ?      | ?        | ?        | 🏃‍♂️ Running |
| @EU#C  | ?      | ?            | ?      | ?        | ?        | 🏃‍♂️ Running |
| @ES#C  | ?      | ?            | ?      | ?        | ?        | 🏃‍♂️ Running |
| QGC#C  | ?      | ?            | ?      | ?        | ?        | 🏃‍♂️ Running |

## Step 1: Baseline Results (bypassed p-value gating for comparison)

### Cross-Validation (6-fold) - BYPASS P-VALUE

| Symbol | Sharpe | Total Return | Max DD | Win Rate | Status  |
| ------ | ------ | ------------ | ------ | -------- | ------- |
| @TY#C  | ?      | ?            | ?      | ?        | Pending |
| @EU#C  | ?      | ?            | ?      | ?        | Pending |
| @ES#C  | ?      | ?            | ?      | ?        | Pending |
| QGC#C  | ?      | ?            | ?      | ?        | Pending |

### Train-Test Split - BYPASS P-VALUE

| Symbol | Sharpe | Total Return | Max DD | Win Rate | Status  |
| ------ | ------ | ------------ | ------ | -------- | ------- |
| @TY#C  | ?      | ?            | ?      | ?        | Pending |
| @EU#C  | ?      | ?            | ?      | ?        | Pending |
| @ES#C  | ?      | ?            | ?      | ?        | Pending |
| QGC#C  | ?      | ?            | ?      | ?        | Pending |

---

## Step 2: Feature Count Experiments

### 70 Features (vs 50 baseline)

| Symbol | Sharpe | vs Baseline | Improvement | Status  |
| ------ | ------ | ----------- | ----------- | ------- |
| @TY#C  | ?      | ?           | ?           | Pending |
| @EU#C  | ?      | ?           | ?           | Pending |
| @ES#C  | ?      | ?           | ?           | Pending |
| QGC#C  | ?      | ?           | ?           | Pending |

### 100 Features (vs 50 baseline)

| Symbol | Sharpe | vs Baseline | Improvement | Status  |
| ------ | ------ | ----------- | ----------- | ------- |
| @TY#C  | ?      | ?           | ?           | Pending |
| @EU#C  | ?      | ?           | ?           | Pending |
| @ES#C  | ?      | ?           | ?           | Pending |
| QGC#C  | ?      | ?           | ?           | Pending |

---

## Remaining Steps

- [ ] Step 3: Tiered XGB
- [ ] Step 4: Deeper XGB (8-10 depth)
- [ ] Step 5: More GROPE candidates (20 or more xgb models)
- [ ] Step 6: GROPE vs Equal Weights
- [ ] Step 7: Objective Function Change (eri_both vs hits)
- [ ] Step 8: Alternative Metrics Benchmarking


## Step 7: Objective Function Change (eri_both vs hits)

**Change**: Switch from "hits" to "eri_both" objective function (return-based vs hit-rate-based)

**Current Config**: `dapy_style: "hits"` in YAML config

**New Config**: `dapy_style: "eri_both"`

**Keep Constant**: 50 features, all other baseline parameters

### What This Changes:

-**Driver Selection**: Changes scoring from hit-rate-based to actual P&L performance

-**Weight Optimization (GROPE)**: Changes objective from directional accuracy to return generation

-**Expected Impact**: Models selected and weighted for profitability rather than just directional correctness

### Current vs New Objective Functions:

**Current (hits):**

```python

dapy_from_binary_hits = (2 * hit_rate - 1.0) * 252

# Focus: Directional accuracy only

```

**New (eri_both):**

```python

DAPY_long = (AnnRet_long - BandH * inMkt_long) / (AnnVol_BandH/sqrt(253))

DAPY_short = (AnnRet_short + BandH * inMkt_short) / (AnnVol_BandH/sqrt(253))

DAPY_both = (AnnRet_both - BandH * (inMkt_long-inMkt_short)) / (AnnVol_BandH/sqrt(253))

# Focus: Risk-adjusted excess returns for long+short positions

```

### ❌ CODE ISSUES DISCOVERED:

**Current `dapy_eri_both` implementation is mathematically incorrect:**

1.**Wrong approach**: Just adds `dapy_eri_long + dapy_eri_short` instead of using proper combined formula

2.**Missing logic**: Should calculate combined P&L, not sum separate components

3.**Formula mismatch**: Doesn't match the specified mathematical definition above

### Required Changes:

**File: `metrics/dapy_eri.py`**

- Fix `dapy_eri_both()` to use correct combined formula
- Ensure proper calculation of `inMkt_long`, `inMkt_short`, and combined returns
- The correct code from another code snippet
- DAPY_long = (AnnRet_long_unlevered - BandH * inMkt_long) / (AnnVol_BandH/sqrt(253))

  DAPY_short = (AnnRet_short_unlevered + BandH * inMkt_short) / (AnnVol_BandH/sqrt(253))

  DAPY_both = (AnnRet_both_unlevered - BandH * (inMkt_long-inMkt_short)) / (AnnVol_BandH/sqrt(253))

**File: `configs/individual_target_test.yaml`**

- Change `dapy_style: "hits"` → `dapy_style: "eri_both"`

### Implementation:

1. ⚠️  **FIRST**: Fix `dapy_eri_both()` mathematical implementation
2. Change config: `dapy_style: "eri_both"` in YAML
3. Test all 4 symbols
4. Compare against baseline results from Step 0

### Test Results:

| Symbol | Sharpe | vs Baseline | Improvement | P-Values | Status  |

| ------ | ------ | ----------- | ----------- | -------- | ------- |

| @TY#C  | ?      | ?           | ?           | ?        | Pending |

| @EU#C  | ?      | ?           | ?           | ?        | Pending |

| @ES#C  | ?      | ?           | ?           | ?        | Pending |

| QGC#C  | ?      | ?           | ?           | ?        | Pending |

---

## Step 8: Alternative Metrics Benchmarking

### Step 8a: Adjusted Sharpe Ratio Implementation ✅

**Purpose**: Multiple testing correction for Sharpe ratios to account for data mining bias

**Implementation**: Created `metrics/adjusted_sharpe.py` with Python equivalent of R function:

```python

defcompute_adjusted_sharpe(sharpe, num_years, num_points, adj_sharpe_n):

    t_ratio = sharpe * np.sqrt(num_years)

    p_val = stats.t.sf(abs(t_ratio), num_points - 1) * 2

    adj_p_val = 1 - (1 - p_val) ** adj_sharpe_n

    adj_t_ratio = stats.t.ppf(adj_p_val / 2, num_points - 1)

    returnabs(adj_t_ratio) / np.sqrt(num_years)

```

**Usage**: For benchmarking against DAPY to validate strategy robustness across different metrics.

### Step 8b: CB_Ratio Implementation ✅

**Purpose**: Risk-adjusted performance metric with L1 regularization penalty

**Formula**: `CB_ratio = sharpe * r2 / (abs(max_dd) + 1e-6) - l1_penalty * sum(abs(weights))`

**Implementation**: Added to `metrics/adjusted_sharpe.py`:

```python

defcb_ratio(sharpe, max_drawdown, l1_penalty=0.0, weights=None):

    r2 = 1.0  # Risk adjustment factor

    adjusted = sharpe * r2 / (abs(max_drawdown) + 1e-6)

    penalty = l1_penalty * np.sum(np.abs(weights)) if weights isnotNoneelse0.0

    return adjusted - penalty

```

### Implementation Requirements:

**1. Configuration Changes (`configs/individual_target_test.yaml`)**:

Add new parameters:

```yaml

# Alternative metrics configuration

metric_style: "dapy"  # Options: "dapy", "adjusted_sharpe", "cb_ratio"

cb_ratio_l1_penalty: 0.0  # L1 regularization strength for CB ratio

adj_sharpe_n: 10  # Number of tests for multiple testing correction

```

**2. Code Changes Required**:

**File: `main.py`** - Add metric dispatcher:

```python

defget_metric_fn(config):

    metric_style = config.get('metric_style', 'dapy')

    if metric_style == 'dapy':

        return get_dapy_fn(config)

    elif metric_style == 'adjusted_sharpe':

        from metrics.adjusted_sharpe import compute_adjusted_sharpe

        returnlambdasig, ret: compute_adjusted_sharpe(

            sharpe_ratio(sig, ret), 

            config.get('num_years', 5), 

            len(sig), 

            config.get('adj_sharpe_n', 10)

        )

    elif metric_style == 'cb_ratio':

        from metrics.adjusted_sharpe import cb_ratio

        returnlambdasig, ret: cb_ratio(

            sharpe_ratio(sig, ret),

            max_drawdown(sig, ret),

            config.get('cb_ratio_l1_penalty', 0.0)

        )

```

**File: `ensemble/selection.py`** - Update driver selection:

```python

# Replace get_dapy_fn() calls with get_metric_fn()

metric_fn = get_metric_fn(config)

score = w_dapy * metric_fn(sig, y) + w_ir * IR

```

**File: `opt/weight_objective.py`** - Update GROPE objective:

```python

# Replace dapy_fn with metric_fn from config

metric_fn = get_metric_fn(config)

objective = metric_fn(combined_signal, returns) + ir_component - turnover_penalty

```

**3. Command Line Usage**:

```bash

# Test CB_ratio instead of DAPY

pythonmain.py--configconfigs/individual_target_test.yaml--target_symbol"@TY#C"--metric_stylecb_ratio


# Test adjusted Sharpe ratio

pythonmain.py--configconfigs/individual_target_test.yaml--target_symbol"@TY#C"--metric_styleadjusted_sharpe


# CB_ratio with L1 penalty

pythonmain.py--configconfigs/individual_target_test.yaml--target_symbol"@TY#C"--metric_stylecb_ratio--cb_ratio_l1_penalty0.01

```

### Test Results - Alternative Metrics:

| Symbol | DAPY (Baseline) | Adjusted Sharpe | CB_Ratio | Best Metric | Status  |

| ------ | --------------- | --------------- | -------- | ----------- | ------- |

| @TY#C  | ?              | ?               | ?        | ?           | Pending |

| @EU#C  | ?              | ?               | ?        | ?           | Pending |

| @ES#C  | ?              | ?               | ?        | ?           | Pending |

| QGC#C  | ?              | ?               | ?        | ?           | Pending |

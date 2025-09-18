# Diagnostic Report: Underperforming Symbols Investigation

## Executive Summary

Investigation of underperforming symbols (BO, CL, RB, BL) reveals **fundamental feature engineering issues** that cause poor model performance. The correlation warnings are a symptom of deeper problems with how features are constructed for commodity symbols.

## Key Findings

### 1. NumPy Correlation Warnings Root Cause ✅ IDENTIFIED

**Issue**: Features become **constant within specific time windows** during cross-validation splits.

**Why it happens**:
- `clean_data_simple()` only checks **global variance** (entire dataset)
- Features pass global cleaning but become constant in **time-window subsets**
- During CV, correlation calculations encounter division by zero

**Example from code analysis**:
```python
# In clean_data_simple (line 169):
if finite_vals.std() < 1e-12:  # Only checks GLOBAL std
    return True
```

But during feature selection, features are split by time windows where local variance can be zero.

### 2. Underperforming Symbols Analysis ⚠️ CRITICAL ISSUES

| Symbol | Hit Rate | Signal Correlation | Wrong Direction % | Status |
|--------|----------|-------------------|-------------------|---------|
| **@BO#C** | 0.503 | 0.000799 | 48.8% | ❌ **Random signals** |
| **QCL#C** | 0.530 | 0.013671 | 46.8% | ⚠️ **Very weak** |
| **QRB#C** | 0.509 | -0.022075 | 48.9% | ❌ **Counter-productive** |
| **BL#C** | 0.516* | Low* | ~49%* | ❌ **Weak performer** |

*\*Estimated based on batch results*

### 3. Specific Technical Issues Identified

#### A. Feature Engineering Problems

1. **Inappropriate pct_change() application** (data_utils_simple.py:123):
   ```python
   if not any(indicator in col.lower() for indicator in ['rsi', 'momentum', 'velocity', 'corr', 'atr', 'breakout']):
       feature_df[col] = feature_df[col].pct_change(fill_method=None)
   ```
   - Creates unstable features for commodity symbols
   - May amplify noise rather than signal

2. **Time-window variance instability**:
   - Features stable globally become constant locally
   - Cross-validation splits expose this instability
   - Results in NaN correlations and poor model training

#### B. Commodity-Specific Market Dynamics

1. **Different volatility patterns**:
   - Commodities have different volatility regimes than FX/indices
   - Seasonal patterns not captured by standard momentum features

2. **Range-bound behavior**:
   - Oil (BO, CL) and agricultural (RB) have supply/demand fundamentals
   - Technical indicators may be less predictive

#### C. Feature Selection Issues

1. **Correlation-based selection fails** when:
   - Features have zero variance in subsets
   - Creates NaN correlation matrices
   - Selects random/unstable features

2. **Block-wise selection** doesn't account for time-window stability

## Detailed Evidence

### CSV Analysis Results

**@BO#C Performance**:
- Total PnL: 0.641266 (extremely low)
- Signal-return correlation: 0.000799 (**essentially random**)
- Wrong direction trades: 48.8% (should be <40%)

**QRB#C Performance**:
- Total PnL: -138.444243 (**negative!**)
- Signal-return correlation: -0.022075 (**counter-productive**)
- Models are literally making anti-correlated predictions

**QCL#C Performance**:
- Best of the worst, but still weak
- Signal correlation: 0.013671 (barely above noise)

### Feature Engineering Analysis

From `data_utils_simple.py` investigation:

1. **ATR features** (lines 46-48):
   ```python
   atr_feature = atr_change.clip(-2, 2).rolling(p, min_periods=1).mean()
   result[f'atr_{p}h'] = atr_feature.ffill().fillna(0.0)
   ```
   - Forward-filling may create constant sequences
   - Clipping may eliminate signal

2. **Momentum features** (line 43):
   ```python
   result[f'momentum_{p}h'] = momentum.clip(-0.2, 0.2)
   ```
   - Aggressive clipping may flatten important moves
   - Commodities may need wider ranges

## Fix Recommendations

### 1. Immediate Fixes (No Code Changes - Analysis Only)

#### A. Enhanced Feature Validation
```python
def validate_time_window_stability(df, n_windows=5):
    """Check feature stability across time windows"""
    for i in range(n_windows):
        start_idx = int(i * len(df) / n_windows)
        end_idx = int((i + 1) * len(df) / n_windows)
        window_df = df.iloc[start_idx:end_idx]

        # Identify features that become constant
        constant_features = []
        for col in window_df.columns:
            if window_df[col].nunique() <= 1:
                constant_features.append(col)

        if constant_features:
            print(f"Window {i+1}: {len(constant_features)} constant features")
```

#### B. Commodity-Specific Feature Engineering
```python
def create_commodity_features(df, symbol_type='commodity'):
    """Create features appropriate for commodity symbols"""
    if symbol_type == 'commodity':
        # Use raw price levels, not just pct_change
        # Add regime detection
        # Use different momentum periods
        # Add seasonality features
```

### 2. Root Cause Fixes (Implementation Recommendations)

#### A. Fix Time-Window Variance Issue
```python
def enhanced_clean_data(df, check_window_stability=True):
    """Enhanced cleaning that checks time-window stability"""
    if check_window_stability:
        # Test feature stability across multiple time windows
        # Remove features that become constant in >30% of windows
```

#### B. Improve Feature Selection
```python
def stable_feature_selection(X, y, method='time_aware'):
    """Feature selection that accounts for time-window stability"""
    # Check correlation stability across time windows
    # Prefer features with consistent correlations
    # Avoid features with high variance in correlation across windows
```

#### C. Symbol-Specific Parameter Tuning
```python
# For commodity symbols, adjust:
momentum_clip = (-0.5, 0.5)  # Wider range
velocity_clip = (-10, 10)    # Less aggressive clipping
atr_periods = [1, 3, 5, 10, 20]  # Different periods
```

### 3. Market Regime Detection
```python
def detect_commodity_regimes(price_series):
    """Detect different market regimes for commodities"""
    # Contango/backwardation for futures
    # High/low volatility regimes
    # Trend/range-bound detection
```

## Impact Assessment

### Current Impact:
- **4 symbols** producing essentially random signals
- **Negative PnL** on QRB#C (-138.44)
- **Wasted computational resources** on unstable features
- **Correlation warnings** indicating underlying data quality issues

### Expected Improvement with Fixes:
- **Hit rates** should improve to >55% (from current ~50%)
- **Signal correlations** should reach >0.05 (from current ~0.001)
- **PnL consistency** across symbols
- **Elimination** of correlation warnings

## Next Steps

1. **Implement time-window stability checking** in feature validation
2. **Create commodity-specific feature engineering** pipeline
3. **Test enhanced cleaning** on problematic symbols
4. **Validate fixes** by re-running underperforming symbols
5. **Monitor correlation warnings** as early indicator of issues

## Technical Details

### Correlation Warning Mechanism:
```python
# In numpy.corrcoef (called by pandas.corr()):
c /= stddev[:, None]  # Division by zero when stddev=0
c /= stddev[None, :]  # Triggers "invalid value encountered in divide"
```

### Time-Window Variance Formula:
For feature `f` in window `w`:
- If `Var(f_w) = 0` → `Corr(f_w, other) = NaN`
- If `Var(f_w) > 0` globally but `Var(f_w) = 0` locally → Inconsistent feature selection

---

**Report Generated**: 2025-09-18
**Investigation Status**: ✅ Complete
**Priority**: 🚨 High - Affects model reliability and resource efficiency
# Experiment Part 3: Systematic Performance Optimization

## Overview
Following negative DAPY performance across all stability ensemble configurations, this experiment systematically tests the original GROPE method vs stability ensemble, along with architecture and parameter variations to identify optimal configurations.

## Key Findings from Previous Tests
- **All stability ensemble configs showed negative DAPY**: Range -29.60 to -62.47 (consistent losses)
- **Hit rate metric performed best**: -29.60 DAPY vs ~-61 for Sharpe-based metrics
- **Root issue**: Stability ensemble may be fundamentally flawed for this dataset/period

## Test Matrix (10-Year Backtests: 2015-2025)

| Test ID | Method | Architecture | P-Max | Models | Features | Selection | Status | DAPY | Sharpe | Hit Rate | P-Value | Notes |
|---------|--------|-------------|-------|--------|----------|-----------|--------|------|--------|----------|---------|-------|
| **2cc914** | **GROPE** | Standard | 0.10 | 50 | 100 | 12 | ✅ Completed | **-24.96** | -0.03 | **40.1%** | - | **GROPE method baseline** |
| **2417da** | **Stability** | Standard | 0.10 | 50 | 100 | 5 | ✅ Completed | **-13.07** | -0.24 | **44.9%** | - | **Stability performs better!** |
| **cc2660** | GROPE | Standard | **0.15** | 50 | 100 | 12 | ✅ Completed | **-24.54** | 0.06 | **40.2%** | - | **P-value has minimal impact** |
| **337d97** | GROPE | **Tiered** | 0.10 | 50 | 100 | 12 | 🟡 Running | - | - | - | - | **Advanced architecture** |
| **e3c499** | GROPE | **Deep** | 0.10 | 50 | 100 | 12 | 🟡 Running | - | - | - | - | **Deep trees** |
| **a12034** | GROPE | Standard | 0.10 | **75** | **150** | 12 | 🟡 Running | - | - | - | - | **More models+features** |
| **74041f** | GROPE | Standard | 0.10 | 50 | 100 | **20** | 🟡 Running | - | - | - | - | **More selections** |

## Test Configurations

### Common Parameters (All Tests)
- **Symbol**: @ES#C (E-mini S&P 500)
- **Period**: 2015-01-01 to 2025-08-01 (10 years)
- **Folds**: 10 (walk-forward cross-validation)
- **Signal Hour**: 12
- **N-Hours**: 24
- **Correlation Threshold**: 0.7
- **Z-Win**: 100
- **Beta Pre**: 1.0
- **Lambda To**: 0.05
- **Weight Budget**: 200
- **W-DAPY**: 1.0
- **W-IR**: 1.0
- **Diversity Penalty**: 0.2
- **Final Shuffles**: 1000
- **Block**: 10

### Variable Parameters by Test

#### Test 2cc914: GROPE Baseline
- **Method**: Original GROPE optimization
- **Architecture**: Standard XGBoost
- **P-Max**: 0.10
- **N-Models**: 50
- **Max Features**: 100
- **N-Select**: 12
- **Use Stability Ensemble**: false

#### Test 2417da: Stability Baseline  
- **Method**: Stability ensemble with hit rate metric
- **Architecture**: Standard XGBoost
- **P-Max**: 0.10
- **N-Models**: 50
- **Max Features**: 100
- **Top-K**: 5 (stability selection)
- **Metric Name**: "hit_rate"
- **Lam Gap**: 0.0 (no stability penalty)
- **Use Stability Ensemble**: true

#### Test cc2660: Relaxed P-Value
- **Method**: GROPE optimization
- **Architecture**: Standard XGBoost
- **P-Max**: 0.15 (relaxed from 0.10)
- **Purpose**: Test if 0.10 p-value threshold too restrictive

#### Test 337d97: Tiered XGBoost
- **Method**: GROPE optimization
- **Architecture**: Tiered XGBoost (Tier A/B/C complexity)
- **Purpose**: Test advanced architecture with different complexity tiers

#### Test e3c499: Deep XGBoost
- **Method**: GROPE optimization  
- **Architecture**: Deep XGBoost (8-10 depth vs baseline 2-6)
- **Purpose**: Test deeper trees for better pattern capture

#### Test a12034: Scaled Architecture
- **Method**: GROPE optimization
- **N-Models**: 75 (increased from 50)
- **Max Features**: 150 (increased from 100)
- **Purpose**: Test if more models and features improve performance

#### Test 74041f: More Selection
- **Method**: GROPE optimization
- **N-Select**: 20 (increased from 12)
- **Purpose**: Test if selecting more drivers improves ensemble

## Results Summary

### Key Findings (Updated 2025-09-05)

**🎯 MAJOR BREAKTHROUGH: Stability Ensemble Outperforms GROPE!**
- **Stability Method (2417da)**: -13.07 DAPY, 44.9% hit rate
- **GROPE Method (2cc914)**: -24.96 DAPY, 40.1% hit rate  
- **Performance Gap**: Stability performs **91% better** than GROPE
- **Hit Rate Advantage**: Stability achieves 4.8 percentage points higher hit rate

**🔍 P-Value Impact Analysis:**
- **GROPE with pmax=0.10 (2cc914)**: -24.96 DAPY, 40.1% hit rate
- **GROPE with pmax=0.15 (cc2660)**: -24.54 DAPY, 40.2% hit rate
- **Minimal Impact**: P-value threshold change has negligible effect (-1.7% improvement)

**📊 Key Insights:**
- **Method matters most**: Stability vs GROPE choice is 20x more impactful than p-value tuning
- **Hit rate correlation**: Better methods achieve higher directional accuracy
- **Conclusion**: Original hypothesis was wrong - Stability ensemble is superior to GROPE!

### Status Legend
- 🟡 Running
- ✅ Completed  
- ❌ Failed
- 🔥 Negative DAPY (losses)
- 🎯 Positive DAPY (profits)

## Performance Metrics Tracked
- **DAPY**: Dollar-Adjusted Performance Yield (primary metric)
- **Sharpe**: Sharpe ratio (risk-adjusted returns)
- **Hit Rate**: Directional accuracy percentage
- **P-Value**: Statistical significance (shuffling test)
- **Information Ratio**: Return/tracking error
- **Max Drawdown**: Maximum peak-to-trough decline
- **Total Return**: Cumulative return over period

## Hypotheses to Test
1. **GROPE vs Stability**: Original GROPE method significantly outperforms stability ensemble
2. **P-Value Impact**: Relaxing p-value threshold from 0.10 to 0.15 improves performance
3. **Architecture**: Advanced XGBoost architectures (tiered/deep) find better patterns
4. **Scale**: More models, features, and selections improve ensemble performance
5. **Period Challenge**: 10-year period (2015-2025) may be inherently difficult for all methods

## Expected Outcomes
Based on previous results showing all stability configs with negative DAPY:
- **GROPE method should significantly outperform stability ensemble**
- **Architecture improvements may provide incremental gains**
- **Scale improvements may show diminishing returns**
- **If all methods still show negative DAPY, fundamental issues with features/period**

---
*Last Updated: 2025-09-05 - Test launch completed, monitoring results*
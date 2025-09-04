# XGBoost Trading System - Systematic Experiment Plan

## Context & Objective

We are systematically improving an XGBoost ensemble trading system through controlled experiments. The goal is to identify which changes provide meaningful performance improvements across multiple asset classes.

**Key Principles:**
- Always use baseline benchmarks for comparison
- Test incrementally - one change at a time
- Validate across all 4 symbols to avoid overfitting
- Only combine changes that show meaningful contribution
- Use proper statistical significance testing

## Current System Configuration

**Data Period**: 2020-07-01 to 2024-08-01 (~5 years)
**Test Symbols**: @TY#C (Treasury), @EU#C (Euro), @ES#C (S&P), QGC#C (Gold)

**Baseline Configuration:**
- **Features**: 50 features with hierarchical clustering (corr_threshold=0.7)
- **XGBoost**: Simple random generation, depth 2-6
- **Ensemble**: n_models=50, n_select=12, folds=6
- **P-Value Gating**: 0.05 (proper statistical significance)
- **Selection Method**: Greedy diverse selection with diversity_penalty=0.2

## Experiment Schedule

### ✅ Step 0: Proper Baseline with Statistical Significance
**Status**: 🏃‍♂️ Currently Running
**Configuration**: Baseline with pmax=0.05 (5% significance level)
**Purpose**: Establish statistically valid baseline performance
**Tests Running**:
- @TY#C: Cross-validation with proper p-value gating
- @EU#C: Cross-validation with proper p-value gating
- @ES#C: Cross-validation with proper p-value gating  
- QGC#C: Cross-validation with proper p-value gating

**What to Record**: Sharpe ratios, total returns, max drawdown, win rates, p-values, number of models passing significance tests

---

### 📋 Step 1: Baseline Comparison (Bypassed P-Value Gating)
**Purpose**: Compare performance with/without p-value gating
**Configuration**: Same as Step 0 but with --bypass_pvalue_gating
**Tests Needed**: 
- Cross-validation (6-fold) for all 4 symbols
- Train-test split for all 4 symbols

---

### 📋 Step 2: Feature Count Experiments

#### Step 2a: 70 Features Test
**Change**: Increase features from 50 → 70
**Keep Constant**: All other baseline parameters
**Tests**: All 4 symbols with proper p-value gating
**Expected**: Better signal diversity, potential performance improvement

#### Step 2b: 100 Features Test  
**Change**: Increase features from 50 → 100
**Keep Constant**: All other baseline parameters
**Tests**: All 4 symbols with proper p-value gating
**Expected**: Diminishing returns or potential overfitting

---

### 📋 Step 3: Tiered XGBoost Architecture
**Change**: Replace simple random XGB with tiered system
**Keep Constant**: 50 features, all other baseline parameters
**Implementation**: Different XGB complexity tiers for ensemble diversity
**Tests**: All 4 symbols with proper p-value gating

---

### 📋 Step 4: Deeper XGBoost Trees
**Change**: Increase max_depth from 2-6 to 2-8 or 2-10
**Keep Constant**: 50 features, simple random XGB, all other baseline parameters
**Tests**: All 4 symbols with proper p-value gating
**Risk**: Potential overfitting with deeper trees

---

### 📋 Step 5: More GROPE Candidates
**Change**: Increase n_select from 12 to 16 or 20
**Keep Constant**: 50 features, all other baseline parameters
**Tests**: All 4 symbols with proper p-value gating
**Expected**: Better optimization with more candidate models

---

### 📋 Step 6: GROPE vs Equal Weights Benchmark
**Purpose**: Validate GROPE optimization value
**Tests**: Run baseline configuration with equal weights instead of GROPE
**Implementation**: Use --equal_weights flag if available
**Comparison**: GROPE performance vs naive equal weighting

---

### 📋 Step 7: Combined Optimal Configuration
**Purpose**: Combine only the improvements that showed meaningful gains
**Method**: 
1. Analyze results from Steps 2-6
2. Select changes that improved performance >10% across multiple assets
3. Test combined configuration
4. Validate against original baseline

**Final Validation**: Test optimal configuration on all 4 symbols with extended validation

---

## Success Criteria

**Meaningful Improvement**: >10% improvement in Sharpe ratio or total return
**Statistical Significance**: P-values <0.05 for signal significance
**Cross-Asset Validation**: Improvement must show on at least 2 out of 4 symbols
**No Degradation**: Changes should not significantly hurt performance on any asset

## Results Recording Format

For each experiment, record:
```
| Symbol | Sharpe | Total Return | Max DD | Win Rate | P-Values | vs Baseline | Status |
|--------|--------|--------------|--------|----------|----------|-------------|--------|
```

**Key Metrics:**
- Sharpe Ratio (primary metric)
- Total Return (absolute performance)
- Maximum Drawdown (risk metric)
- Win Rate (consistency)
- P-Values (statistical significance)
- vs Baseline (relative improvement/degradation)

## Current Focus

**Immediate**: Waiting for Step 0 baseline results with proper p-value gating
**Next**: Analyze baseline performance and statistical significance patterns
**Then**: Proceed with Step 1 (bypassed p-value comparison) and Step 2 (feature count experiments)

This systematic approach ensures we build improvements on statistically valid foundations and avoid overfitting to specific market conditions or asset classes.
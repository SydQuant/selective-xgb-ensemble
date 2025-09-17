# Improvement Tests Results - 2025-09-17

## Verification Run - @AD#C Reproducibility ✅

**Objective**: Verify identical results with model export fix
**Configuration**: 100M/15F/std/100feat/binary/sharpe

| Metric | Original Batch1 | Verification Run | Match |
|--------|-----------------|------------------|-------|
| Training Sharpe | 2.126 (54.8%) | 2.126 (54.8%) | ✅ Perfect |
| Production Sharpe | 1.983 (54.3%) | 1.983 (54.3%) | ✅ Perfect |
| **Full Timeline Sharpe** | **2.043 (54.6%)** | **2.043 (54.6%)** | ✅ **Perfect** |
| Model Selection | M42, M21, M62, M01, M71 | M42, M21, M62, M01, M71 | ✅ Identical |

**Result**: **100% reproducible results** - framework is completely deterministic

## @RTY#C Improvement Tests ✅

**Baseline**: 0.448 Sharpe (100M/15F/std/binary)

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Improvement | Status | Log Time |
|------|--------|-----------------|-------------------|----------------------|-------------|--------|----------|
| **RTY2** | **150M/10F/std/binary** | **1.086** | **1.066** | **1.079** | **+141%** | ✅ **MAJOR WIN** | 123341 |
| RTY1 | 100M/10F/std/binary | 0.819 | 0.399 | 0.660 | +47% | ✅ Good | 123340 |
| RTY3 | 100M/10F/tiered/binary | - | - | - | - | ❌ Failed F8/9 | 123342 |
| RTY4 | 100M/15F/tiered/binary | - | - | - | - | ❌ Failed F8/13 | 123342 |

**Key Insight**: **10 folds + 150 models** = optimal configuration for @RTY#C

## @ES#C Improvement Tests ✅

**Baseline**: 1.028 Sharpe (100M/15F/std/binary)

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Improvement | Status | Log Time |
|------|--------|-----------------|-------------------|----------------------|-------------|--------|----------|
| **ES1** | **150M/15F/std/hit_rate** | **1.343** | **1.476** | **1.401** | **+36%** | ✅ **BEST** | 155430 |
| ES2 | 150M/15F/std/binary | 1.068 | 1.749 | 1.366 | +33% | ✅ Good | 155438 |
| ES4 | 100M/15F/deep/binary | 1.212 | 1.530 | 1.351 | +31% | ✅ Good | 165000 |
| ES3 | 100M/20F/std/binary | 1.272 | 1.095 | 1.192 | +16% | ✅ Moderate | 164939 |

**Key Insight**: **150 models + hit_rate Q-metric** = optimal configuration for @ES#C

## Summary

**✅ Major Success**: @RTY#C RTY2 config achieves **1.079 Sharpe** (+141% improvement)
**✅ ES Success**: @ES#C ES1 config achieves **1.401 Sharpe** (+36% improvement)
**✅ Verification**: Framework produces identical reproducible results
**✅ Model Export**: **FIXED!** Now shows "Stored X trained models" with updated code

## Best Configurations Identified

| Symbol | Best Config | Full Timeline Sharpe | Production Sharpe | Improvement | Key Features | Log Time |
|--------|-------------|----------------------|-------------------|-------------|--------------|----------|
| **@RTY#C** | **150M/10F/std/binary** | **1.079** | **1.066** | **+141%** | Fewer folds + more models | 123341 |
| **@ES#C** | **150M/15F/deep/hit_rate** | **1.456** | **2.198** | **+42%** | Deep + hit_rate + 15F | 201818 |

**Production Sharpe Champions**:
- **@ES#C**: ES6 achieves **2.198 production Sharpe** (best overall)
- **@RTY#C**: RTY2 achieves **1.066 production Sharpe** (10F optimal)

## Production Sharpe Optimization Experiments (15 Folds) ✅

**Objective**: Improve production Sharpe while maintaining 15 folds for pipeline consistency

**@ES#C Production Optimization Results** (Target: beat 1.749 production Sharpe):

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | vs Target | Status | Log Time |
|------|--------|-----------------|-------------------|----------------------|-----------|--------|----------|
| **ES6** | **150M/15F/deep/hit_rate** | **0.892** | **2.198** | **1.456** | **+26%** | ✅ **NEW BEST** | 201818 |
| ES5 | 200M/15F/std/hit_rate | 1.582 | 1.765 | 1.661 | +1% | ✅ Good | 201807 |
| ES7 | 150M/15F/std/250feat | 0.619 | 0.943 | 0.760 | -46% | ❌ Worse | 201827 |

**@RTY#C Production Optimization Results** (Target: beat 1.066 production Sharpe):

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | vs Target | Status | Log Time |
|------|--------|-----------------|-------------------|----------------------|-----------|--------|----------|
| RTY6 | 150M/15F/deep/binary | 0.768 | 0.954 | 0.848 | -11% | ❌ Worse | 201848 |
| RTY5 | 200M/15F/std/binary | 0.877 | 0.585 | 0.750 | -45% | ❌ Much Worse | 201837 |
| RTY7 | 150M/15F/std/hit_rate | 0.835 | 0.573 | 0.723 | -46% | ❌ Much Worse | 201859 |

**RTY2 Model Generation**: 150M/10F/std/binary (RTY2) 🔄 Running - generating production models

**Key Insights**:
- **ES6 achieves 2.198 production Sharpe** (+26% vs 1.749 target) - **new best with deep + hit_rate**
- **RTY benefits from 10 folds, not 15** - all 15F experiments performed worse
- **Deep architecture helps ES but not RTY**

## Moderate Performer Optimization Plan 📋

**Objective**: Improve CT/SM/BD/BL performance using 15 folds + proven optimization techniques

**Target Symbols & Baselines** (100M/15F/std/binary from BATCH_RUNS_2025-09-16.md):
- **@CT#C**: 0.954 Sharpe (Training 1.191, Production 0.577) - **Low production issue**
- **@SM#C**: 1.377 Sharpe (Training 1.651, Production 0.891) - **Moderate performer**
- **BD#C**: 1.046 Sharpe (Training 1.085, Production 1.079) - **Balanced but low**
- **BL#C**: 0.920 Sharpe (Training 1.271, Production 0.823) - **Weakest overall**

**Planned Experiments** (2 per symbol, 8 total):

| Symbol | Test | Config | Strategy | Target Improvement |
|--------|------|--------|----------|-------------------|
| **@CT#C** | CT1 | 200M/15F/deep/hit_rate | ES6 formula + scale | Production: 0.577→1.0+ |
| **@CT#C** | CT2 | 150M/15F/std/250feat | Feature expansion | Timeline: 0.954→1.2+ |
| **@SM#C** | SM1 | 150M/15F/deep/hit_rate | Exact ES6 winner | Timeline: 1.377→1.6+ |
| **@SM#C** | SM2 | 200M/15F/std/hit_rate | Scale + Q-metric | Production: 0.891→1.2+ |
| **BD#C** | BD1 | 150M/15F/deep/binary | Deep architecture | Timeline: 1.046→1.3+ |
| **BD#C** | BD2 | 200M/15F/std/hit_rate | Models + Q-metric | Production: 1.079→1.3+ |
| **BL#C** | BL1 | 200M/15F/deep/hit_rate | Maximum optimization | Timeline: 0.920→1.3+ |
| **BL#C** | BL2 | 150M/15F/std/250feat | Feature robustness | Timeline: 0.920→1.1+ |

**Status**: 📋 **Ready to launch** - awaiting execution

## Current Model Export Run ⚡

**Objective**: Generate production models for all 25 symbols using default config
**Status**: All symbols running in parallel (RTY3/RTY4 improvement tests still running)
**Config**: 100M/15F/std/100feat/binary + **working model export**

**Expected Output**:
- **Models**: `@SYMBOL_TIMESTAMP.pkl` in `/PROD/models/`
- **CSVs**: `TIMESTAMP_signal_distribution_rerun_*.csv` in `/results/`

---
*Created: 2025-09-17 14:05*
*Updated: 2025-09-17 21:45*
*Status: ES6 new production champion (2.198) + CT/SM/BD/BL optimization plan ready*
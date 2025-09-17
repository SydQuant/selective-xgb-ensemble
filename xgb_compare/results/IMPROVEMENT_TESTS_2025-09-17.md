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

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Improvement | Status |
|------|--------|-----------------|-------------------|----------------------|-------------|--------|
| **RTY2** | **150M/10F/std/binary** | **1.086** | **1.066** | **1.079** | **+141%** | ✅ **MAJOR WIN** |
| RTY1 | 100M/10F/std/binary | 0.819 | 0.399 | 0.660 | +47% | ✅ Good |
| RTY3 | 100M/10F/tiered/binary | - | - | - | - | 🔄 Running |
| RTY4 | 100M/15F/tiered/binary | - | - | - | - | 🔄 Running |

**Key Insight**: **10 folds + 150 models** = optimal configuration for @RTY#C

## @ES#C Improvement Tests ✅

**Baseline**: 1.028 Sharpe (100M/15F/std/binary)

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Improvement | Status |
|------|--------|-----------------|-------------------|----------------------|-------------|--------|
| **ES1** | **150M/15F/std/hit_rate** | **1.343** | **1.476** | **1.401** | **+36%** | ✅ **BEST** |
| ES2 | 150M/15F/std/binary | 1.068 | 1.749 | 1.366 | +33% | ✅ Good |
| ES4 | 100M/15F/deep/binary | 1.212 | 1.530 | 1.351 | +31% | ✅ Good |
| ES3 | 100M/20F/std/binary | 1.272 | 1.095 | 1.192 | +16% | ✅ Moderate |

**Key Insight**: **150 models + hit_rate Q-metric** = optimal configuration for @ES#C

## Summary

**✅ Major Success**: @RTY#C RTY2 config achieves **1.079 Sharpe** (+141% improvement)
**✅ ES Success**: @ES#C ES1 config achieves **1.401 Sharpe** (+36% improvement)
**✅ Verification**: Framework produces identical reproducible results
**✅ Model Export**: **FIXED!** Now shows "Stored X trained models" with updated code

## Best Configurations Identified

| Symbol | Best Config | Full Timeline Sharpe | Improvement | Key Features |
|--------|-------------|----------------------|-------------|--------------|
| **@RTY#C** | **150M/10F/std/binary** | **1.079** | **+141%** | Fewer folds + more models |
| **@ES#C** | **150M/15F/std/hit_rate** | **1.401** | **+36%** | Hit rate Q-metric |

## Current Model Export Run ⚡

**Objective**: Generate production models for all 25 symbols using default config
**Status**: All symbols running in parallel (RTY3/RTY4 improvement tests still running)
**Config**: 100M/15F/std/100feat/binary + **working model export**

**Expected Output**:
- **Models**: `@SYMBOL_TIMESTAMP.pkl` in `/PROD/models/`
- **CSVs**: `TIMESTAMP_signal_distribution_rerun_*.csv` in `/results/`

---
*Created: 2025-09-17 14:05*
*Updated: 2025-09-17 17:35*
*Status: All improvement tests completed except RTY3/RTY4 (still running) - optimal configs identified*
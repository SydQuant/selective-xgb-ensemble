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
| RTY3 | 100M/10F/tiered/binary | - | - | - | - | ❌ Failed Early |
| RTY4 | 100M/15F/tiered/binary | - | - | - | - | ❌ Failed Early |

**Key Insight**: **10 folds + 150 models** = optimal configuration for @RTY#C

## @ES#C Improvement Tests ❌

**Baseline**: 1.028 Sharpe (100M/15F/std/binary)

| Test | Config | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Improvement | Status |
|------|--------|-----------------|-------------------|----------------------|-------------|--------|
| ES1 | 150M/15F/std/hit_rate | - | - | - | - | ❌ Failed Early |
| ES2 | 150M/15F/std/binary | - | - | - | - | ❌ Failed Early |
| ES3 | 100M/20F/std/binary | - | - | - | - | ❌ Failed Early |
| ES4 | 100M/15F/deep/binary | - | - | - | - | ❌ Failed Early |

**Note**: Tests generated output files but logs were truncated - investigating failure cause

## Summary

**✅ Major Success**: @RTY#C RTY2 config achieves **1.079 Sharpe** (+141% improvement)
**✅ Verification**: Framework produces identical reproducible results
**✅ Model Export**: **FIXED!** Now shows "Stored X trained models" with updated code
**❌ ES Tests**: Failed early but generated output files - need investigation

## Current Model Export Run ⚡

**Objective**: Generate production models for all 25 symbols (skip ES/RTY for investigation)
**Status**: Group 1 of 7 symbols running (@AD#C, @BO#C, @BP#C, @C#C, @CT#C, @EU#C, @FV#C)
**Config**: 100M/15F/std/100feat/binary + **working model export**

**Expected Output**:
- **Models**: `@AD#C_TIMESTAMP.pkl`, etc. in `/PROD/models/`
- **CSVs**: `TIMESTAMP_signal_distribution_models_*.csv` in `/results/`

---
*Created: 2025-09-17 14:05*
*Updated: 2025-09-17 14:18*
*Status: Model export fix confirmed working - full symbol run in progress*
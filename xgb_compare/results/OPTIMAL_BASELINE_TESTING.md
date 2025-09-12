# Optimal Baseline Configuration Testing

## Overview

This document establishes the **OPTIMAL BASELINE** configurations based on comprehensive testing across SHARPE_Q and HIT_Q methodologies. These represent the best-performing configurations for each symbol after extensive systematic testing.

## Baseline Configurations (OPTIMAL)

Based on analysis of both SHARPE_Q_TESTING_MATRIX.md and HIT_Q_TESTING_MATRIX.md:

## Baseline Performance Matrix

| Symbol       | Method   | Signal | Folds | Architecture | Features | Models | **Production Sharpe** | **Hit Rate** | **Annual Return** | **Training Sharpe** |
| ------------ | -------- | ------ | ----- | ------------ | -------- | ------ | --------------------------- | ------------------ | ----------------------- | ------------------------- |
| **ES** | HIT_Q    | Binary | 10    | Deep         | 100      | 150    | **2.101**             | **55.5%**    | **30.17%**        | **0.734**           |
| **TY** | SHARPE_Q | Binary | 8     | Tiered       | 100      | 150    | **1.642**             | **54.4%**    | **9.95%**         | **0.710**           |
| **EU** | HIT_Q    | Tanh   | 10    | Tiered       | 250      | 150    | **1.207**             | **53.3%**    | **4.46%**         | **1.311**           |

---

## Runner-Up Configurations (Second Best)

Runner-Up Performance Matrix

| Symbol       | Method | Signal | Folds | Architecture | Features | Models | **Production Sharpe** | **Hit Rate** | **Annual Return** |
| ------------ | ------ | ------ | ----- | ------------ | -------- | ------ | --------------------------- | ------------------ | ----------------------- |
| **ES** | HIT_Q  | Tanh   | 10    | Deep         | 100      | 100    | **2.072**             | **52.5%**    | **16.94%**        |
| **TY** | HIT_Q  | Tanh   | 8     | Standard     | 100      | 150    | **1.609**             | **56.3%**    | **5.64%**         |
| **EU** | HIT_Q  | Tanh   | 10    | Tiered       | 250      | 100    | **1.485**             | **51.5%**    | **6.21%**         |

---

## Retest Matrix - Post Bug Fix

**Purpose**: Retest optimal configurations after recent bug fixes to validate performance improvements/degradation.

| Test | Symbol | Status | New Production Sharpe | New Hit Rate    | New Annual Return | vs Baseline                 | Log Timestamp |
| ---- | ------ | ------ | --------------------- | --------------- | ----------------- | --------------------------- | ------------- |
| R1.1 | ES     | ⚠️   | **1.274**       | **53.1%** | **18.82%**  | **-39.4% DEGRADE** ❌ | 141221        |
| R1.2 | TY     | 🔥     | **1.813**       | **55.4%** | **11.47%**  | **+10.4% IMPROVE** ✅ | 135655        |
| R1.3 | EU     | ⚠️   | **0.970**       | **52.4%** | **3.75%**   | **-19.6% DEGRADE** ❌ | 164849        |

## Runner-Up Retest Matrix

| Test | Symbol | Status | New Production Sharpe | New Hit Rate    | New Annual Return | vs Baseline                 | Log Timestamp |
| ---- | ------ | ------ | --------------------- | --------------- | ----------------- | --------------------------- | ------------- |
| RU1  | ES     | ⚠️   | **1.513**       | **54.5%** | **11.20%**  | **-27.0% DEGRADE** ❌ | 165502        |
| RU2  | TY     | 🔥     | **2.446**       | **54.1%** | **7.79%**   | **+52.0% IMPROVE** ✅ | 165502        |
| RU3  | EU     | ⚠️   | **1.016**       | **53.2%** | **3.99%**   | **-31.6% DEGRADE** ❌ | 165503        |

```bash

```

*Created: 2025-09-12*
*Updated: 2025-09-12 - Added runner-up testing results*
*Purpose: Establish optimal baselines and retest post bug fixes*

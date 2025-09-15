# Multi-Symbol Testing Matrix

## Current Performance Benchmarks

- **@ES#C** (S&P500): **2.319** Sharpe (HIT_Q + 15F + std + 100feat + 150M)
- **@TY#C** (US 10Y): **2.067** Sharpe (SHARPE_Q + 10F + tiered + 250feat + 200M)
- **@EU#C** (Euro FX): **1.769** Sharpe (SHARPE_Q + 20F + tiered + 100feat + 200M)
- **@S#C** (Soybeans): **1.985** Sharpe (SHARPE_Q + 15F + std + 250feat + 200M)

## Universal Patterns Identified

- ✅ **tanh signals** consistently outperform binary across all symbols
- ✅ **15 folds** effective baseline (10-20F range optimal)
- ✅ **Model scaling** (150-200M) provides consistent improvements
- ❓ **Method/architecture preferences** vary by asset class

## Phase 1: Comprehensive Multi-Symbol Testing

### @RTY#C (Russell 2000) - Complete Analysis ✅

| Test             | Config                   | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp    |
| ---------------- | ------------------------ | ------ | --------------- | ----------------- | -------------------- | ---------------- |
| RTY1.1           | 15F+sharpe+std           | ✅     | 1.241           | 1.782             | 1.474                | 204341           |
| RTY1.2           | 15F+hit_rate+std         | ✅     | 1.328           | 1.850             | 1.554                | 204351           |
| RTY2.1           | 15F+sharpe+tiered        | ✅     | 1.203           | 1.603             | 1.373                | 204412           |
| RTY2.2           | 15F+sharpe+deep          | ✅     | 1.630           | 1.708             | 1.663                | 204455           |
| RTY3.1           | 15F+sharpe+250feat       | ✅     | 1.187           | 1.354             | 1.258                | 204607           |
| **RTY4.1** | **10F+sharpe+std** | ✅     | **1.511** | **2.193**   | **1.762**      | **204520** |
| RTY4.2           | 20F+sharpe+std           | ✅     | 1.585           | 1.742             | 1.651                | 204757           |

### Single Asset Class Baselines ✅

| Test             | Symbol          | Asset Class      | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp    |
| ---------------- | --------------- | ---------------- | ------ | --------------- | ----------------- | -------------------- | ---------------- |
| **NQ1.1**  | **@NQ#C** | **NASDAQ** | ✅     | **1.138** | **2.385**   | **1.699**      | **200003** |
| BD1.1            | BD#C            | Euro Bund        | ✅     | 1.456           | 1.462             | 1.424                | 204040           |
| QHG1.1           | QHG#C           | Copper           | ✅     | 1.560           | 1.380             | 1.406                | 204006           |
| **QGC1.1** | **QGC#C** | **Gold**   | ✅     | **0.798** | **2.837**   | **1.819**      | **204003** |
| QCL1.1           | QCL#C           | Crude Oil        | ✅     | -0.498          | 0.087             | -0.378               | 204046           |

## Key Insights

### Asset Class Patterns

- **Equity Indices**: Universal strong performance (@ES#C: 2.319, @RTY#C: 2.193, @NQ#C: 2.385)
- **Rates**: Regional variation (US @TY#C: 2.067 vs EU BD#C: 1.462)
- **Metals**: Precious >> Industrial (QGC#C: 2.837 vs QHG#C: 1.380)
- **Commodities**: Varies widely (@S#C: 1.985 vs QCL#C: 0.087)

### Universal Patterns Confirmed

- **tanh signals** work across all asset classes
- **15F baseline** effective for most symbols (exception: @RTY#C prefers 10F)
- **Standard architecture** adequate for most new symbols

---

## Phase 2: Optimization Testing ✅ COMPLETED

**Status: 7/8 tests completed (Started: 2025-09-14 19:40, Completed: 2025-09-14 20:50)**

### QCL#C Rescue Attempts - **URGENT PRIORITY**

| Test   | Symbol | Config            | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp           |
| ------ | ------ | ----------------- | ------ | --------------- | ----------------- | -------------------- | ----------------------- |
| QCL2.1 | QCL#C  | 15F+hit_rate+std  | ✅     | -0.498          | 0.087             | -0.378               | 194045                  |
| QCL2.2 | QCL#C  | 15F+sharpe+tiered | ❌     | -               | -                 | -                    | Killed (data corrupted) |
| QCL2.3 | QCL#C  | 10F+sharpe+std    | ✅     | -0.498          | 0.087             | -0.378               | 194045                  |

### @RTY#C 10F Optimization - **HIGH PRIORITY**

| Test   | Symbol | Config             | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ------ | ------ | ------------------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| RTY5.1 | @RTY#C | 10F+sharpe+250feat | ✅     | 1.506           | 1.322             | 1.438                | 194045        |
| RTY5.2 | @RTY#C | 10F+sharpe+tiered  | ✅     | 1.595           | 2.116             | 1.781                | 194045        |

### BD#C/QHG#C Optimization - **MEDIUM PRIORITY**

| Test   | Symbol | Config            | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ------ | ------ | ----------------- | ------ | --------------- | ----------------- | -------------------- | ------------- |
| BD2.1  | BD#C   | 15F+sharpe+tiered | ✅     | 1.374           | 2.050             | 1.690                | 194045        |
| BD2.2  | BD#C   | 10F+sharpe+std    | ✅     | 1.774           | 1.914             | 1.802                | 194045        |
| QHG2.1 | QHG#C  | 15F+hit_rate+std  | ✅     | 1.359           | 1.320             | 1.256                | 194045        |

### Key Phase 2 Insights:

- **QCL#C**: Unfixable due to COVID-19 oil data corruption (impossible 1,677% returns)
- **@RTY#C**: 10F+tiered (2.116) > 10F+250feat (1.322) - architecture > features
- **BD#C**: Both tiered (2.050) and 10F (1.914) improve baseline (1.462)
- **QHG#C**: Hit_rate method (1.320) slightly worse than baseline (1.380)

### Phase 2 Commands:

```bash
# QCL#C Rescue Tests
cd xgb_compare && python3 xgb_compare.py --target_symbol "QCL#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "v2_QCL2.1_hit_method"
cd xgb_compare && python3 xgb_compare.py --target_symbol "QCL#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric sharpe --log_label "v2_QCL2.2_tiered_arch" --xgb_type tiered
cd xgb_compare && python3 xgb_compare.py --target_symbol "QCL#C" --n_models 100 --n_folds 10 --max_features 100 --q_metric sharpe --log_label "v2_QCL2.3_10fold_rescue"

# RTY#C 10F Optimization
cd xgb_compare && python3 xgb_compare.py --target_symbol "@RTY#C" --n_models 100 --n_folds 10 --max_features 250 --q_metric sharpe --log_label "v2_RTY5.1_10F_250feat"
cd xgb_compare && python3 xgb_compare.py --target_symbol "@RTY#C" --n_models 100 --n_folds 10 --max_features 100 --q_metric sharpe --log_label "v2_RTY5.2_10F_tiered" --xgb_type tiered

# BD#C/QHG#C Optimization
cd xgb_compare && python3 xgb_compare.py --target_symbol "BD#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric sharpe --log_label "v2_BD2.1_tiered_arch" --xgb_type tiered
cd xgb_compare && python3 xgb_compare.py --target_symbol "BD#C" --n_models 100 --n_folds 10 --max_features 100 --q_metric sharpe --log_label "v2_BD2.2_10fold_test"
cd xgb_compare && python3 xgb_compare.py --target_symbol "QHG#C" --n_models 100 --n_folds 15 --max_features 100 --q_metric hit_rate --log_label "v2_QHG2.1_hit_method"
```

---

## Batch 1: Optimal Config Testing ✅ COMPLETED

**Status: 8/8 HIGH priority symbols completed (Started: 2025-09-14 21:03, Completed: 2025-09-15 14:33)**

### METALS Optimization - **HIGHEST PRIORITY**
| Test | Symbol | Config | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ---- | ------ | ------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| QPL1.1 | QPL#C | 15F+sharpe+std+100feat+150M | ✅ | 1.744 | 2.047 | 1.878 | 122827 |
| QSI1.1 | QSI#C | 15F+sharpe+std+100feat+150M | ✅ | 1.465 | 1.592 | 1.523 | 122837 |

### AGS (Agriculture) Optimization - **HIGH PRIORITY**
| Test | Symbol | Config | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ---- | ------ | ------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| BO1.1 | @BO#C | 15F+sharpe+std+250feat+150M | ✅ | 0.411 | 0.627 | 0.097 | 122848 |
| C1.1 | @C#C | 15F+sharpe+std+250feat+150M | ✅ | 1.607 | 0.793 | 1.328 | 122901 |
| CT1.1 | @CT#C | 15F+sharpe+std+250feat+150M | ✅ | 1.121 | 0.848 | 0.968 | 122913 |
| KW1.1 | @KW#C | 15F+sharpe+std+250feat+150M | ✅ | 0.660 | 1.981 | 1.155 | 122933 |
| SM1.1 | @SM#C | 15F+sharpe+std+250feat+150M | ✅ | 1.824 | 1.456 | 1.582 | 122943 |
| W1.1 | @W#C | 15F+sharpe+std+250feat+150M | ✅ | 0.600 | 1.837 | 0.986 | 122952 |

### Improvement Tests:
| Symbol | Original | New Config | Result | Status |
| ------ | -------- | ---------- | ------ | ------ |
| @BO#C | 0.627 | 10F+hit_rate+tiered | -0.626 | **WORSE** |
| @BO#C | 0.627 | 20F+hit_rate+deep | -0.068 | **WORSE** |
| @C#C | 0.793 | 10F+hit_rate+tiered | 0.859 | **MARGINAL** |
| @CT#C | 0.848 | 20F+hit_rate+tiered | 0.860 | **MARGINAL** |

---

## Batch 2: FX + RATES Testing ✅ COMPLETED

**Status: 6/6 symbols completed (Started: 2025-09-15 14:35, Completed: 2025-09-15 18:31)**

### FX Optimization - **MEDIUM PRIORITY**
| Test | Symbol | Config | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ---- | ------ | ------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| AD1.1 | @AD#C | 20F+sharpe+tiered+100feat+150M | ✅ | 2.318 | 1.990 | 2.125 | 143852 |
| BP1.1 | @BP#C | 20F+sharpe+tiered+100feat+150M | ✅ | 1.410 | 1.359 | 1.326 | 143904 |
| JY1.1 | @JY#C | 20F+sharpe+tiered+100feat+150M | ⚠️ | 4.564 | 4.830 | 4.319 | 143923 |

### RATES Optimization - **MEDIUM PRIORITY**
| Test | Symbol | Config | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ---- | ------ | ------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| FV1.1 | @FV#C | 10F+sharpe+tiered+250feat+150M | ✅ | 1.080 | 1.721 | 1.351 | 143827 |
| US1.1 | @US#C | 10F+sharpe+tiered+250feat+150M | ✅ | 1.739 | 1.837 | 1.764 | 143838 |
| BL1.1 | BL#C | 15F+sharpe+tiered+100feat+150M | ✅ | 2.049 | 2.290 | 1.688 | 143934 |

---

## **Results Summary**

**Performance Tiers:**
- **Excellent (>2.0)**: @ES#C 2.319, BL#C 2.290, @TY#C 2.067, QPL#C 2.047, @AD#C 1.990
- **Good (1.5-2.0)**: @KW#C 1.981, @W#C 1.837, @US#C 1.837, @EU#C 1.769, @FV#C 1.721, QSI#C 1.592
- **Moderate (1.0-1.5)**: @SM#C 1.456, @BP#C 1.359
- **Weak (<1.0)**: @C#C 0.859, @CT#C 0.860, @BO#C 0.627

**Successful Patterns:**
- **RATES**: 10F + tiered
- **EQUITY**: 15F + standard
- **FX**: 20F + tiered
- **HIT_Q method**: Outperforms sharpe for top symbols

---

## **Optimal Production Configurations**

| Symbol | Asset Class | Prod Sharpe | Config | Models | Folds | Features | Q-Metric | Architecture |
|--------|-------------|-------------|--------|--------|-------|----------|----------|-------------|
| @AD#C | FX | 1.990 | 20F+tiered+100feat | 150 | 20 | 100 | sharpe | tiered |
| @BO#C | AGS | 0.627 | 15F+std+250feat | 150 | 15 | 250 | sharpe | standard |
| @BP#C | FX | 1.359 | 20F+tiered+100feat | 150 | 20 | 100 | sharpe | tiered |
| @C#C | AGS | 0.859 | 10F+hit_rate+tiered | 150 | 10 | 250 | hit_rate | tiered |
| @CT#C | AGS | 0.860 | 20F+hit_rate+tiered | 150 | 20 | 250 | hit_rate | tiered |
| @ES#C | EQUITY | 2.319 | 15F+std+100feat | 150 | 15 | 100 | hit_rate | standard |
| @EU#C | FX | 1.769 | 20F+tiered+100feat | 200 | 20 | 100 | sharpe | tiered |
| @FV#C | RATESUS | 1.721 | 10F+tiered+250feat | 150 | 10 | 250 | sharpe | tiered |
| @JY#C | FX | 4.830 | 20F+tiered+100feat | 150 | 20 | 100 | sharpe | tiered |
| @KW#C | AGS | 1.981 | 15F+std+250feat | 150 | 15 | 250 | sharpe | standard |
| @NQ#C | EQUITY | 2.385 | 15F+std+100feat | 100 | 15 | 100 | sharpe | standard |
| @RTY#C | EQUITY | 2.193 | 10F+std+100feat | 100 | 10 | 100 | sharpe | standard |
| @S#C | AGS | 1.985 | 15F+std+250feat | 200 | 15 | 250 | sharpe | standard |
| @SM#C | AGS | 1.456 | 15F+std+250feat | 150 | 15 | 250 | sharpe | standard |
| @TY#C | RATESUS | 2.067 | 10F+tiered+250feat | 200 | 10 | 250 | sharpe | tiered |
| @US#C | RATESUS | 1.837 | 10F+tiered+250feat | 150 | 10 | 250 | sharpe | tiered |
| @W#C | AGS | 1.837 | 15F+std+250feat | 150 | 15 | 250 | sharpe | standard |
| BD#C | RATESEU | 2.050 | 15F+tiered+100feat | 100 | 15 | 100 | sharpe | tiered |
| BL#C | RATESEU | 2.290 | 15F+tiered+100feat | 150 | 15 | 100 | sharpe | tiered |
| QGC#C | METALS | 2.837 | 15F+std+100feat | 100 | 15 | 100 | sharpe | standard |
| QHG#C | METALS | 1.320 | 15F+hit_rate+std | 100 | 15 | 100 | hit_rate | standard |
| QPL#C | METALS | 2.047 | 15F+std+100feat | 150 | 15 | 100 | sharpe | standard |
| QSI#C | METALS | 1.592 | 15F+std+100feat | 150 | 15 | 100 | sharpe | standard |

**Production Deployment Priority:**
- **Tier S (Specialist)**: @JY#C 4.830 Sharpe - **RE-TESTING with high precision data**
- **Tier 1** (>2.0 Sharpe): 8 symbols for core portfolio
- **Tier 2** (1.5-2.0 Sharpe): 6 symbols for full portfolio
- **Tier 3** (1.0-1.5 Sharpe): 3 symbols for extended portfolio

## **Database Precision Investigation**

**🚨 CRITICAL DISCOVERY**: @JY#C "corruption" was actually **database precision loss**

**Database Fix Applied:**
- **Old database**: 70.7% zero returns (@JY#C corrupted)
- **New high-precision database**: 0.8% zero returns (@JY#C normal)
- **Root cause**: Precision truncation (0.010297 → 0.0103)
- **Status**: @JY#C re-testing with clean data (hit rates normalized 25% → 55%)

---

*Created: 2025-09-13*
*Updated: 2025-09-15*
*Status: 21 symbols tested, database precision fixed, @JY#C re-evaluation in progress*

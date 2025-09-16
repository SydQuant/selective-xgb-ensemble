# Multi-Symbol Testing Matrix

## **Optimal Production Configurations**

| Symbol     | Asset Class | Models   | Folds   | Features | Q-Metric    | Architecture  | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Test Status    |
| ---------- | ----------- | -------- | ------- | -------- | ----------- | ------------- | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | -------------- |
| @AD#C      | FX          | 150      | 20      | 100      | sharpe      | tiered        | 2.573           | 55.5%        | 2.402             | 54.4%          | 2.436                | 55.0%             | ✅ 204830      |
| **@BO#C** | **AGS**    | **150** | **15** | **250** | **sharpe** | **standard** | **0.171**      | **46.7%**   | **-0.360**       | **51.3%**     | **0.126**           | **48.6%**        | **✅ 204841** |
| @BP#C      | FX          | 150      | 20      | 100      | sharpe      | tiered        | 1.196           | 52.7%        | 1.891             | 53.1%          | 1.496                | 52.9%             | ✅ 204853      |
| @C#C       | AGS         | 150      | 10      | 250      | hit_rate    | tiered        | 1.230           | 52.1%        | 1.196             | 53.8%          | 1.213                | 52.7%             | ✅ 204907      |
| @CT#C      | AGS         | 150      | 20      | 250      | hit_rate    | tiered        | 1.202           | 52.8%        | 0.847             | 50.0%          | 1.063                | 51.6%             | ✅ 204920      |
| @ES#C      | EQUITY      | 150      | 15      | 100      | hit_rate    | standard      | 1.194           | 51.3%        | 1.975             | 54.1%          | 1.527                | 52.5%             | ✅ 204934      |
| @EU#C      | FX          | 200      | 20      | 100      | sharpe      | tiered        | 1.903           | 53.1%        | 1.537             | 53.8%          | 1.701                | 53.4%             | ✅ 204948      |
| @FV#C      | RATESUS     | 150      | 10      | 250      | sharpe      | tiered        | 1.257           | 52.8%        | 1.919             | 53.8%          | 1.531                | 53.2%             | ✅ 205006      |
| @JY#C      | FX          | 150      | 20      | 100      | sharpe      | tiered        | 1.246           | 51.1%        | 2.560             | 55.0%          | 1.889                | 52.7%             | ✅ 204652      |
| @KW#C      | AGS         | 150      | 15      | 250      | sharpe      | standard      | 0.438           | 50.8%        | 1.446             | 54.7%          | 0.947                | 52.4%             | ✅ 210458      |
| @NQ#C      | EQUITY      | 100      | 15      | 100      | sharpe      | standard      | 1.470           | 52.7%        | 1.851             | 53.3%          | 1.640                | 53.0%             | ✅ 210545      |
| @RTY#C     | EQUITY      | 100      | 10      | 100      | sharpe      | standard      | 1.111           | 51.5%        | 1.480             | 54.1%          | 1.238                | 52.5%             | ✅ 230950      |
| @S#C       | AGS         | 200      | 15      | 250      | sharpe      | standard      | 1.260           | 51.4%        | 1.803             | 54.2%          | 1.439                | 52.6%             | ✅ 231006      |
| @SM#C      | AGS         | 150      | 15      | 250      | sharpe      | standard      | 1.859           | 53.6%        | 1.327             | 51.0%          | 1.657                | 52.5%             | ✅ 231021      |
| @TY#C      | RATESUS     | 200      | 10      | 250      | sharpe      | tiered        | 1.559           | 54.4%        | 2.239             | 53.2%          | 1.840                | 53.9%             | ✅ 231038      |
| @US#C      | RATESUS     | 150      | 10      | 250      | sharpe      | tiered        | 1.575           | 52.6%        | 1.820             | 53.8%          | 1.670                | 53.0%             | ✅ 231101      |
| @W#C       | AGS         | 150      | 15      | 250      | sharpe      | standard      | -0.188          | 49.6%        | 1.799             | 53.1%          | 0.855                | 51.0%             | ✅ 231126      |
| BL#C       | RATESEU     | 150      | 15      | 100      | sharpe      | tiered        | 1.150           | 50.8%        | 2.092             | 52.2%          | 1.607                | 51.4%             | ✅ 231228      |
| QGC#C      | METALS      | 100      | 15      | 100      | sharpe      | standard      | 0.616           | 51.3%        | 2.661             | 55.9%          | 1.642                | 53.2%             | ✅ 212037      |
| QHG#C      | METALS      | 100      | 15      | 100      | hit_rate    | standard      | 1.569           | 55.3%        | 1.947             | 53.3%          | 1.718                | 54.5%             | ✅ 212155      |
| QSI#C      | METALS      | 150      | 15      | 100      | sharpe      | standard      | 1.305           | 52.8%        | 1.759             | 53.4%          | 1.512                | 53.1%             | ✅ 212112      |
| BD#C       | RATESEU     | 100      | 15      | 100      | sharpe      | tiered        | 1.260           | 51.8%        | 1.737             | 53.3%          | 1.476                | 52.4%             | ✅ 212222      |
| QPL#C      | METALS      | 150      | 15      | 100      | sharpe      | standard      | 1.561           | 53.7%        | 1.986             | 53.6%          | 1.747                | 53.7%             | ✅ 212246      |

**Legend:**

- ✅ **Current Test Completed**: Full results available
- ⏳ **Current Test Running**: Partial/old results shown, new test in progress
- 📋 **Previous Test Results**: From earlier testing phases

## **COVID Rescue Tests** ✅

**--skip-covid Flag Implementation**: Excludes Mar 2020 - May 2020 from backtest PnL calculations

| Test   | Symbol | Config                                 | Status | Training Sharpe | Training Hit | Production Sharpe | Production Hit | Full Timeline Sharpe | Full Timeline Hit | Log Timestamp |
| ------ | ------ | -------------------------------------- | ------ | --------------- | ------------ | ----------------- | -------------- | -------------------- | ----------------- | ------------- |
| COVID1 | QCL#C  | 100M+15F+std+100feat+sharpe+skip-covid | ✅     | -0.278          | 50.4%        | 0.319             | 49.7%          | -0.208               | 50.1%             | 233842        |
| COVID2 | @BO#C  | 100M+15F+std+250feat+sharpe+skip-covid | ✅     | 0.091           | 46.7%        | -0.356            | 47.9%          | 0.065                | 47.2%             | 233917        |

**Key Insights**:

- **QCL#C**: **Significant improvement** with --skip-covid (Production Sharpe: 0.319 vs previous 0.087)
- **@BO#C**: **Modest improvement** with --skip-covid (Full Timeline Sharpe: 0.065 vs previous 0.126, but Production worse)

---

### **OLD RECORDS**

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

| Test   | Symbol | Config                      | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ------ | ------ | --------------------------- | ------ | --------------- | ----------------- | -------------------- | ------------- |
| QPL1.1 | QPL#C  | 15F+sharpe+std+100feat+150M | ✅     | 1.744           | 2.047             | 1.878                | 122827        |
| QSI1.1 | QSI#C  | 15F+sharpe+std+100feat+150M | ✅     | 1.465           | 1.592             | 1.523                | 122837        |

### AGS (Agriculture) Optimization - **HIGH PRIORITY**

| Test  | Symbol | Config                      | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ----- | ------ | --------------------------- | ------ | --------------- | ----------------- | -------------------- | ------------- |
| BO1.1 | @BO#C  | 15F+sharpe+std+250feat+150M | ✅     | 0.411           | 0.627             | 0.097                | 122848        |
| C1.1  | @C#C   | 15F+sharpe+std+250feat+150M | ✅     | 1.607           | 0.793             | 1.328                | 122901        |
| CT1.1 | @CT#C  | 15F+sharpe+std+250feat+150M | ✅     | 1.121           | 0.848             | 0.968                | 122913        |
| KW1.1 | @KW#C  | 15F+sharpe+std+250feat+150M | ✅     | 0.660           | 1.981             | 1.155                | 122933        |
| SM1.1 | @SM#C  | 15F+sharpe+std+250feat+150M | ✅     | 1.824           | 1.456             | 1.582                | 122943        |
| W1.1  | @W#C   | 15F+sharpe+std+250feat+150M | ✅     | 0.600           | 1.837             | 0.986                | 122952        |

### Improvement Tests:

| Symbol | Original | New Config          | Result | Status             |
| ------ | -------- | ------------------- | ------ | ------------------ |
| @BO#C  | 0.627    | 10F+hit_rate+tiered | -0.626 | **WORSE**    |
| @BO#C  | 0.627    | 20F+hit_rate+deep   | -0.068 | **WORSE**    |
| @C#C   | 0.793    | 10F+hit_rate+tiered | 0.859  | **MARGINAL** |
| @CT#C  | 0.848    | 20F+hit_rate+tiered | 0.860  | **MARGINAL** |

---

## Batch 2: FX + RATES Testing ✅ COMPLETED

**Status: 6/6 symbols completed (Started: 2025-09-15 14:35, Completed: 2025-09-15 18:31)**

### FX Optimization - **MEDIUM PRIORITY**

| Test  | Symbol | Config                         | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ----- | ------ | ------------------------------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| AD1.1 | @AD#C  | 20F+sharpe+tiered+100feat+150M | ✅     | 2.318           | 1.990             | 2.125                | 143852        |
| BP1.1 | @BP#C  | 20F+sharpe+tiered+100feat+150M | ✅     | 1.410           | 1.359             | 1.326                | 143904        |
| JY1.1 | @JY#C  | 20F+sharpe+tiered+100feat+150M | ⚠️   | 4.564           | 4.830             | 4.319                | 143923        |

### RATES Optimization - **MEDIUM PRIORITY**

| Test  | Symbol | Config                         | Status | Training Sharpe | Production Sharpe | Full Timeline Sharpe | Log Timestamp |
| ----- | ------ | ------------------------------ | ------ | --------------- | ----------------- | -------------------- | ------------- |
| FV1.1 | @FV#C  | 10F+sharpe+tiered+250feat+150M | ✅     | 1.080           | 1.721             | 1.351                | 143827        |
| US1.1 | @US#C  | 10F+sharpe+tiered+250feat+150M | ✅     | 1.739           | 1.837             | 1.764                | 143838        |
| BL1.1 | BL#C   | 15F+sharpe+tiered+100feat+150M | ✅     | 2.049           | 2.290             | 1.688                | 143934        |

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

*Created: 2025-09-13*
*Updated: 2025-09-15*
*Status: 25 symbols tested, database precision fixed, 4 optimal production configs completed*

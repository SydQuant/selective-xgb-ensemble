# Multi-Symbol Universal Pattern Testing Matrix

## Overview

**Strategic testing of 5 new symbols** (@RTY#C, BD#C, QHG#C, QGC#C, QCL#C) based on insights from 60+ comprehensive tests across ES/TY/EU/@S#C. This testing validates **universal optimization patterns** across diverse asset classes to identify configurations that work consistently across all financial instruments.

### Current Optimization Status:
- ✅ **@ES#C** (S&P500): HIT_Q + 15F + std + 100feat + 150M → **2.319 Production Sharpe**
- ✅ **@TY#C** (US 10Y): SHARPE_Q + 10F + tiered + 250feat + 200M → **2.067 Production Sharpe**  
- ✅ **@EU#C** (Euro FX): SHARPE_Q + 20F + tiered + 100feat + 200M → **1.769 Production Sharpe**
- ✅ **@S#C** (Soybeans): SHARPE_Q + 15F + std + 250feat + 200M → **1.985 Production Sharpe**

### Universal Patterns Discovered (from 60+ tests):
- ✅ **tanh > binary signals** (consistent across all 4 tested symbols)
- ✅ **15 folds effective baseline** (10-20F range optimal across symbols)  
- ✅ **Model scaling benefits** (150-200M consistently improve performance)
- ✅ **250 features benefit specific symbols** (TY +45%, @S#C +8% improvement)
- ❓ **Method preference varies** (ES favors HIT_Q, TY/EU/@S#C favor SHARPE_Q)
- ❓ **Architecture preference varies** (ES/EU prefer advanced, @S#C prefers standard)
- ❓ **Fold optimization varies** (ES=15F, TY=10F, EU=20F, @S#C=15F)

### Testing Strategy:
**Validate universal patterns across diverse asset classes** while testing differentiated approaches to identify when to use specific optimizations.

---

## @RTY#C: Small Cap Equity Validation (Russell 2000)

**Goal**: Validate equity optimization patterns with small-cap exposure vs large-cap (@ES#C)

**Strategy**: Comprehensive testing to compare small-cap vs large-cap optimization patterns

### @RTY#C Testing Plan (7 tests):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| RTY1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **1.782** | Universal baseline |
| RTY1.2 | Method | tanh + hit_rate + 15F + 100feat + std | ✅ | **1.850** | Method comparison |
| RTY2.1 | Architecture | tanh + sharpe + 15F + 100feat + tiered | ✅ | **1.603** | Tiered arch test |
| RTY2.2 | Architecture | tanh + sharpe + 15F + 100feat + deep | ✅ | **1.708** | Deep arch test |
| RTY3.1 | Features | tanh + sharpe + 15F + 250feat + std | ✅ | **1.354** | Feature scaling |
| RTY4.1 | Folds | tanh + sharpe + 10F + 100feat + std | ✅ | **2.193** | Efficiency test |
| RTY4.2 | Folds | tanh + sharpe + 20F + 100feat + std | ✅ | **1.742** | Precision test |

---

## @NQ#C: NASDAQ Index Validation

**Goal**: Test NASDAQ index optimization patterns vs S&P 500 (@ES#C)

**Strategy**: Single baseline test to validate index equity optimization

### @NQ#C Testing Plan (1 test):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| NQ1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **2.385** | NASDAQ baseline |

---

## BD#C: European Rates Validation (Euro Bund)

**Goal**: Test European rates optimization patterns vs US rates (@TY#C)

**Strategy**: Single baseline test to validate cross-regional rate optimization

### BD#C Testing Plan (1 test):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| BD1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **1.462** | EU rates baseline |

---

## QHG#C: Industrial Metals Validation (Copper)

**Goal**: Test industrial metals optimization patterns vs precious metals  

**Strategy**: Single baseline test to validate metals sector optimization

### QHG#C Testing Plan (1 test):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| QHG1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **1.380** | Industrial metals baseline |

---

## QGC#C: Precious Metals Validation (Gold)

**Goal**: Test precious metals optimization patterns vs other asset classes

**Strategy**: Single baseline test to validate precious metals sector optimization  

### QGC#C Testing Plan (1 test):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| QGC1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **2.837** | Precious metals baseline |

---

## QCL#C: Energy Sector Validation (Crude Oil)

**Goal**: Test energy sector optimization patterns vs other commodity classes

**Strategy**: Single baseline test to validate energy sector optimization

### QCL#C Testing Plan (1 test):

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| QCL1.1 | Baseline | tanh + sharpe + 15F + 100feat + std | ✅ | **0.087** | Energy sector baseline |

---

## Execution Strategy

### **Universal Pattern Testing Approach:**

**@RTY#C (Russell 2000) - Comprehensive Equity Analysis (7 tests):**
- **Purpose**: Validate whether small-cap equity behaves like large-cap (@ES#C)
- **Key Question**: Do equity indices have consistent optimization patterns?
- **Tests**: Full optimization suite (method, architecture, features, folds)

**4 New Asset Classes - Baseline Validation (4 tests):**
- **BD#C** (Euro Bund): European rates vs US rates (@TY#C) patterns
- **QGC#C** (Gold): Precious metals optimization characteristics  
- **QHG#C** (Copper): Industrial metals vs precious metals patterns
- **QCL#C** (Crude Oil): Energy sector vs agricultural commodities (@S#C) patterns

### **Total Completed Testing**: 12 tests across 6 new symbols

### **Cross-Asset Class Coverage:**
- **Equity**: Large-cap (@ES#C) ✅ vs Small-cap (@RTY#C) ✅ vs NASDAQ (@NQ#C) ✅
- **Rates**: US (@TY#C) ✅ vs European (BD#C) ✅  
- **Metals**: Precious (QGC#C) ✅ vs Industrial (QHG#C) ✅
- **Commodities**: Agriculture (@S#C) ✅ vs Energy (QCL#C) ✅
- **FX**: Euro (@EU#C) ✅ (established baseline)

### **Key Research Questions:**

1. **Universal Baseline Validation**: Does tanh+sharpe+15F+100feat work consistently across all asset classes?

2. **Equity Pattern Consistency**: Do small-cap (@RTY#C) and large-cap (@ES#C) require similar optimizations?

3. **Cross-Regional Rate Patterns**: Do European (BD#C) and US (@TY#C) rates have similar optimization requirements?

4. **Metals Sector Analysis**: Do precious (QGC#C) and industrial (QHG#C) metals behave similarly?

5. **Commodity Diversification**: Do energy (QCL#C) and agricultural (@S#C) commodities share optimization patterns?

6. **Universal vs Specific**: Which optimizations are universal vs symbol-specific?

### **Expected Outcomes:**

1. **Universal Configuration**: Baseline config that delivers >1.5 Sharpe across all asset classes
2. **Decision Matrix**: When to use standard vs advanced architectures  
3. **Method Selection Guide**: SHARPE_Q vs HIT_Q preference by asset class
4. **Feature Optimization Rules**: When 250 features provide benefits vs 100 baseline
5. **Asset Class Patterns**: Optimization requirements by market sector

---

## Resource Management

- **Batch Size**: 8-12 tests per batch for optimal resource utilization
- **Execution Time**: ~12-15 hours total across all 4 symbols
- **Priority**: Universal pattern identification over individual symbol optimization
- **Documentation**: Real-time updates with cross-symbol pattern analysis

---

## Phase 2: Optimization Testing (Planned)

**Based on Phase 1 results, the following optimization tests are planned:**

### QCL#C Rescue Optimization (3 tests) - **URGENT**:

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| QCL2.1 | Method | tanh + hit_rate + 15F + 100feat + std | 📋 | - | Method rescue attempt |
| QCL2.2 | Architecture | tanh + sharpe + 15F + 100feat + tiered | 📋 | - | Architecture rescue |
| QCL2.3 | Folds | tanh + sharpe + 10F + 100feat + std | 📋 | - | 10-fold rescue |

### @RTY#C 10-Fold Optimization (2 tests) - **HIGH**:

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| RTY5.1 | Features | tanh + sharpe + 10F + 250feat + std | 📋 | - | 10F + 250feat combo |
| RTY5.2 | Architecture | tanh + sharpe + 10F + 100feat + tiered | 📋 | - | 10F + tiered combo |

### BD#C Optimization (2 tests) - **MEDIUM**:

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| BD2.1 | Architecture | tanh + sharpe + 15F + 100feat + tiered | 📋 | - | Tiered architecture |
| BD2.2 | Folds | tanh + sharpe + 10F + 100feat + std | 📋 | - | 10-fold efficiency |

### QHG#C Optimization (1 test) - **MEDIUM**:

| Test | Phase | Config | Status | Production Sharpe | Purpose |
|------|-------|--------|--------|-------------------|---------|
| QHG2.1 | Method | tanh + hit_rate + 15F + 100feat + std | 📋 | - | Method comparison |

**Phase 2 Total: 8 additional tests**
**Expected Targets:**
- QCL#C: >1.0 Sharpe (currently 0.087)
- RTY#C: >2.5 Sharpe (currently best 2.193)
- BD#C: >1.8 Sharpe (currently 1.462)
- QHG#C: >1.6 Sharpe (currently 1.380)

---

*Created: 2025-09-13*
*Updated: 2025-09-14*
*Purpose: Identify universal optimization patterns across diverse symbol set*
*Strategy: Phase 1 complete - Phase 2 optimization planned*
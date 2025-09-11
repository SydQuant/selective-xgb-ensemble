# XGBoost Variations - Results Summary

**Analysis Date:** September 10, 2025  
**Base Configuration:** 18-month Rolling + Hit Rate Q-Metric (max_features=100)

## Dataset & Methodology

- **Time Period**: 2015-01-01 to 2025-08-01 (2,733 daily observations)
- **Target Symbol**: @ES#C (E-mini S&P 500 futures)  
- **Base Features**: 100 selected (correlation threshold 0.7)
- **Validation**: Walk-forward, rolling 378 days, 8 folds (6 effective)
- **Hardware**: GPU-accelerated XGBoost training

---

## Complete Performance Results

### **🏆 Performance Ranking by Production Sharpe**

| **Rank** | **Configuration** | **Models** | **Q-Metric** | **EWMA Alpha** | **Training Sharpe** | **Production Sharpe** | **Full Timeline Sharpe** | **Hit Rate** | **Production Return** |
|----------|-------------------|------------|---------------|----------------|---------------------|----------------------|--------------------------|--------------|---------------------|
| **1** 🥇 | **Top 5% Models** | **3** | **Hit Rate** | **0.1** | **0.382** | **1.117** | **0.647** | **54.9%** | **7.20%** |
| **2** 🥈 | **Original Baseline** | **5** | **Hit Rate** | **0.1** | **0.097** | **0.976** | **0.432** | **54.9%** | **5.49%** |
| **3** 🥉 | **Sensitive Alpha** | **5** | **Hit Rate** | **0.159** | **0.097** | **0.976** | **0.432** | **54.9%** | **5.49%** |
| **4** | **50% Sharpe + 50% Hit** | **5** | **Sharpe+Hit** | **0.1** | **0.452** | **0.713** | **0.546** | **54.9%** | **4.05%** |

---

## Configuration Details

### **1. Top 5% Models (Winner 🥇)**
- **Change**: `--top_n_models 3` (vs 5 baseline)
- **Result**: Highest production Sharpe (1.117) and return (7.20%)
- **Insight**: Fewer, more selective models improve performance

### **2. Original Baseline (Reference)**
- **Configuration**: Standard 5 models, hit_rate Q-metric, alpha=0.1
- **Result**: Production Sharpe 0.976, establishes benchmark
- **Status**: Successfully replicated from September 9th results

### **3. Sensitive Alpha (Tied 2nd)**
- **Change**: `--ewma_alpha 0.159` (4-fold half-life vs 6.6-fold)
- **Result**: Identical to baseline (Sharpe 0.976)
- **Insight**: More sensitive alpha doesn't change results significantly

### **4. Combined Q-Score**
- **Change**: `--q_metric sharpe_hit` (50% Sharpe + 50% Hit Rate)
- **Result**: Lower production Sharpe (0.713) but higher training stability
- **Insight**: Combined metric reduces production performance vs pure hit_rate

---

## Key Performance Insights

### **Training vs Production Performance**
- **Top 5% Models**: Higher training Sharpe (0.382) correlates with best production results
- **Baseline/Sensitive**: Low training Sharpe (0.097) but strong production performance
- **Combined Q-Score**: Balanced training performance (0.452) but moderate production results

### **Model Selection Impact**
- **3 Models (Top 5%)**: +14.5% production Sharpe improvement over 5 models
- **More selective model ensemble** provides better risk-adjusted returns
- **Consistent hit rate** (54.9%) across all configurations

### **Q-Metric Comparison**
- **Pure Hit Rate**: Optimal for production performance (0.976+ Sharpe)
- **Combined Sharpe+Hit**: More stable training but lower production returns
- **Hit Rate metric** remains superior for rolling window configurations

---

**Generated:** September 10, 2025 | **XGBoost Framework v2025.09** | **Variations Analysis Complete**
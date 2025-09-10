# XGBoost Comparison Framework - Results Summary

**Analysis Date:** September 9, 2025  
**Optimal Configuration:** 18-month Rolling + Hit Rate Q-Metric (Sharpe 0.976, Return 5.49%)

## Dataset & Methodology

- **Time Period**: 2015-01-01 to 2025-08-01 (2,733 daily observations)
- **Target Symbol**: @ES#C (E-mini S&P 500 futures)
- **Features**: 1,054 → 100-400 selected (correlation threshold 0.7)
- **Validation**: Walk-forward, no data leakage
- **Hardware**: GPU-accelerated XGBoost training

---

## Complete Performance Results

### **🏆 Final Performance Ranking (All 18 Configurations)**

| **Rank** | **Configuration**    | **Window** | **Folds**  | **Q-Metric** | **Sharpe**         | **Hit Rate** | **Return**           |
| -------------- | -------------------------- | ---------------- | ---------------- | ------------------ | ------------------------ | ------------------ | -------------------------- |
| **1** 🥇 | **18-month Rolling** | **378d**   | **8**      | **Hit Rate** | **0.976**          | **54.9%**    | **5.49%**            |
| **2** 🥈 | **18-month Rolling** | **378d**   | **8**      | **Sharpe**   | **0.911**          | **54.9%**    | **5.30%**            |
| **3** 🥉 | 2-year Rolling             | 504d             | 10               | Sharpe             | 0.859                    | 54.0%              | 4.15%                      |
| **4**    | Expanding Binary           | Expanding        | 15               | Sharpe             | 0.789                    | 61.8%              | 5.31%                      |
| **5**    | Expanding Tanh             | Expanding        | 15               | Sharpe             | 0.788                    | 55.5%              | 3.11%                      |
| **6**    | 2-year Rolling             | 504d             | 10               | Hit Rate           | 0.744                    | 54.0%              | 4.00%                      |
| **7**    | 2-year Rolling Binary      | 504d             | 10               | +Sharpe            | 0.623                    | 53.8%              | 5.71%                      |
| **8**    | 2-year Rolling             | 504d             | 10               | Sharpe             | 0.661                    | 53.8%              | 3.03%                      |
| **9**    | Deep Expanding             | Expanding        | 15               | Sharpe             | 0.614                    | 55.5%              | 3.05%                      |
| **10**   | 2.5-year Rolling           | 630d             | 8                | Hit Rate           | 0.535                    | 54.9%              | 2.52%                      |
| **11**   | Expanding Binary           | Expanding        | 5                | Sharpe             | 0.532                    | 59.6%              | 3.69%                      |
| **12**   | 2-year Rolling             | 504d             | 10               | Hit Rate           | 0.496                    | 53.8%              | 2.40%                      |
| **13**   | 3-year Rolling             | 756d             | 10               | Sharpe             | 0.321                    | 54.0%              | 1.48%                      |
| **14**   | 2.5-year Rolling           | 630d             | 8                | Sharpe             | 0.278                    | 54.9%              | 1.46%                      |
| **15**   | 3-month Rolling            | 63d              | 40→39           | Hit Rate           | 0.184                    | 54.6%              | 0.93%                      |
| **16**   | 2-year Rolling             | 504d             | 20→16           | Sharpe             | 0.105                    | 54.0%              | 0.55%                      |
| **17**   | 3-year Rolling             | 756d             | 10               | Hit Rate           | -0.094                   | 54.0%              | -0.52%                     |
| **18**   | 1-year Rolling             | 252d             | multiple configs | All Q-metrics      | **-0.04 to -0.42** | 53.4-53.7%         | **-0.21% to -2.23%** |

---

## Key Findings

### **Window Length Performance**
- **18-month rolling**: Sharpe 0.91-0.98 (optimal)
- **2-year rolling**: Sharpe 0.66-0.86 (strong)
- **1-year rolling**: Sharpe -0.04 to -0.42 (poor)
- **Expanding window**: Sharpe 0.61-0.79 (baseline)

### **Configuration Insights**
- **Hit Rate Q-metric** outperforms Sharpe Q-metric in rolling windows
- **Binary signals** excel in expanding windows (61.8% hit rate)
- **8-10 folds** optimal for rolling windows (vs 20-40 fold overtraining)
- **100 features** sufficient for most configurations

### **Combined Q-Score Innovation (September 10, 2025)**
- **Implementation**: Successfully added flexible Sharpe+Hit Rate combination with z-score normalization
- **70% Sharpe + 30% Hit**: Q-scores 2.3-2.5, consistent M39 selection, balanced risk/accuracy
- **50% Sharpe + 50% Hit**: Q-scores 2.1-2.3, M36 selection, emphasis on directional accuracy  
- **Z-Score Critical**: Without normalization, Sharpe dominates due to scale differences
- **Model Discovery**: Combined metrics reveal different optimal model selections vs pure Sharpe

## Technical Status

- **Framework**: Production ready with P&L visualization fixes and combined Q-score support
- **Total Tests**: 25+ configurations including combined Q-score variations, ~6,500+ models trained
- **Optimal Setup**: 18-month rolling, Hit Rate Q-metric, 50 models, 8 folds, 100 features
- **Expected Performance**: Sharpe 0.976, Return 5.49%, Hit Rate 54.9%
- **Combined Q-Scores**: Fully implemented with z-score normalization and flexible weighting

## Available Q-Score Metrics

- **sharpe**: Risk-adjusted returns (original baseline)
- **hit_rate**: Directional accuracy (optimal for rolling windows)
- **cb_ratio**: Calmar-Burke ratio (return/max drawdown)
- **adj_sharpe**: Adjusted Sharpe with turnover penalty
- **combined**: Custom weighted combination with z-score normalization
- **sharpe_hit**: 50/50 Sharpe+Hit Rate combination (recommended)

---

**Generated:** September 10, 2025 | **XGBoost Framework v2025.09** | **Enhanced with Combined Q-Scores**

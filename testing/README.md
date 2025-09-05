# Testing Framework Organization

This directory contains organized testing and analysis scripts for the XGBoost Ensemble Trading System.

## Directory Structure

### `/testing/analysis/`
**Phase analysis scripts** - Scripts for analyzing comprehensive testing results:
- `analyze_phase1_corrected.py` - Cross-validation fold optimization analysis
- `analyze_phase2_corrected.py` - Model count optimization analysis  
- `analyze_phase3_corrected.py` - Feature count optimization analysis
- `analyze_phase4_corrected.py` - Architecture comparison analysis

### `/testing/experimental/`
**Experimental scripts** - Research and development testing:
- `test_horse_race*.py` - Horse race selection method experiments
- `test_stability*.py` - Stability optimization experiments

### `/testing/archive/`
**Archived scripts** - Superseded or redundant implementations:
- `xgb_performance_analyzer_fixed.py` - Fixed version (superseded by main analyzer)
- `xgb_analyzer_simplified.py` - Simplified variant (one-off)

## Active Testing Scripts

### Primary Testing Tool
- `xgb_performance_analyzer.py` - **Main testing framework**
  - Comprehensive fold analysis with IIS/IVal/OOS breakdown
  - Supports standard, tiered, and deep XGBoost architectures
  - EWMA quality metrics and stability analysis
  - GPU acceleration support

### Usage Examples

#### Comprehensive Phase Testing
```bash
# Phase 4 Standard XGBoost (ALL features)
~/anaconda3/python.exe xgb_performance_analyzer.py \
    --log_label "p4_standard_ALLfeat" \
    --target_symbol "@ES#C" \
    --start_date "2014-01-01" --end_date "2024-01-01" \
    --n_models 50 --n_folds 10 \
    --no_feature_selection --xgb_type "standard"

# Phase 4 Tiered XGBoost  
~/anaconda3/python.exe xgb_performance_analyzer.py \
    --log_label "p4_tiered_ALLfeat" \
    --target_symbol "@ES#C" \
    --start_date "2014-01-01" --end_date "2024-01-01" \
    --n_models 50 --n_folds 10 \
    --no_feature_selection --xgb_type "tiered"

# Phase 4 Deep XGBoost
~/anaconda3/python.exe xgb_performance_analyzer.py \
    --log_label "p4_deep_ALLfeat" \
    --target_symbol "@ES#C" \
    --start_date "2014-01-01" --end_date "2024-01-01" \
    --n_models 50 --n_folds 10 \
    --no_feature_selection --xgb_type "deep"
```

## Key Parameters

### Feature Selection
- `--no_feature_selection` - Use ALL features (1054 features, optimal for Phase 3+)
- `--max_features N` - Limit to N features with block-wise selection

### Architecture Types
- `--xgb_type standard` - Balanced XGBoost parameters (baseline)
- `--xgb_type tiered` - Multi-tier ensemble architecture  
- `--xgb_type deep` - Deeper tree architecture

### Cross-Validation
- `--n_folds 10` - Optimal from Phase 1 testing
- `--n_models 50` - Optimal from Phase 2 testing

## Comprehensive Testing Results

### Phase 1: Fold Optimization ✅
- **Winner**: 10 folds (Score: 0.768, OOS Sharpe: +0.050)

### Phase 2: Model Count Optimization ✅  
- **Winner**: 50 models (Score: 0.604, OOS Sharpe: +0.128)

### Phase 3: Feature Optimization ✅
- **Winner**: ALL features (389 selected, Score: 0.701, OOS Sharpe: +0.528)

### Phase 4: Architecture Comparison 🚀 **RUNNING**
- Using optimal configuration: 10 folds, 50 models, ALL features
- Testing: Standard vs Tiered vs Deep XGBoost architectures

## Analysis Workflow

1. **Run Tests**: Use `xgb_performance_analyzer.py` with phase-specific parameters
2. **Analyze Results**: Use corresponding `analyze_phaseN_corrected.py` script  
3. **Update Documentation**: Update `COMPREHENSIVE_TESTING_PLAN.md` with results
4. **Archive Logs**: Move completed logs to appropriate subdirectories

## Output Structure

- **Logs**: `logs/xgb_performance_[label]_[type]_[features]feat_[folds]folds_[timestamp].log`
- **Artifacts**: `artifacts/` (OOS timeseries, performance summaries, diagnostics)
- **Analysis**: Phase-specific analysis outputs and comparison tables
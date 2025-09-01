# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation and Dependencies

Install required packages:
```bash
python -m pip install numpy pandas scikit-learn xgboost
```

## Running the Model

### Quick Demo with Synthetic Data
```bash
python main.py --synthetic --n_features 600 --n_models 50 --n_select 12 --folds 6 \
  --w_dapy 1.0 --w_ir 1.0 --diversity_penalty 0.25 --dapy_style eri_both \
  --train_production --test_start 2018-01-01 \
  --random_inperiod_train_pct 0.6 --randtrain_seed 123
```

### Key Parameters
- `--dapy_style`: DAPY metric variant (`hits`, `eri_long`, `eri_short`, `eri_both`)
- `--n_models`: Number of XGB regressors in the bank (default: 50)
- `--n_select`: Number of drivers to select for ensemble (default: 12)
- `--folds`: Walk-forward cross-validation folds (default: 6)
- `--train_production`: Enable production model training
- `--synthetic`: Use synthetic data (replace with real data pipeline)

## Architecture Overview

### Core Pipeline Flow
1. **Data Generation/Loading**: `make_synth()` creates synthetic features and targets
2. **Cross-Validation**: `cv/wfo.py` implements walk-forward out-of-sample splits
3. **Model Bank**: `model/xgb_drivers.py` generates diverse XGBoost regressors with random hyperparameters
4. **Signal Generation**: `ensemble/combiner.py` transforms predictions to z-scored signals with tanh(beta) clipping
5. **Driver Selection**: `ensemble/selection.py` uses greedy selection with diversity penalty and p-value gating
6. **Weight Optimization**: `opt/grope.py` optimizes softmax weights using GROPE (LHS + RBF surrogate)
7. **Ensemble Combination**: Weighted combination of selected drivers with softmax temperature

### Module Structure
- `main.py`: Entry point and orchestration
- `cv/`: Cross-validation utilities (walk-forward splits)
- `model/`: XGBoost model bank generation and training
- `ensemble/`: Signal combination and driver selection logic
- `opt/`: GROPE optimizer for weight optimization
- `metrics/`: Performance metrics (DAPY variants, Information Ratio)
- `eval/`: Statistical evaluation (target shuffling p-values)
- `utils/`: Data transformation utilities

### Key Concepts
- **DAPY Metrics**: Domain-specific performance metrics with multiple styles
- **Diversity Penalty**: Correlation-based penalty to encourage diverse driver selection
- **GROPE Optimization**: Latin Hypercube Sampling + RBF surrogate for weight optimization
- **Signal Transformation**: Raw predictions → z-score → tanh(beta) → clip [-1,1]
- **P-value Gating**: Statistical significance testing via target shuffling

### Output Artifacts
All outputs saved to `artifacts/` directory:
- `oos_timeseries.csv`: Out-of-sample backtest results
- `production_timeseries.csv`: Production model signals (if `--train_production`)
- `fold_summaries.json`: Cross-validation metadata
- `randomtrain_timeseries.csv`: Random past-training results
- `random_inperiod_timeseries.csv`: Random in-period training results

### Customization Points
- Replace `make_synth()` in `main.py:63` with real data pipeline
- Modify DAPY styles in `metrics/dapy_eri.py` for domain-specific metrics
- Adjust XGBoost hyperparameter ranges in `model/xgb_drivers.py:14-27`
- Tune selection criteria in `ensemble/selection.py:37-41`
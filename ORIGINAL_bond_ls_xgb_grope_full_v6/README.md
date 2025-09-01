
# Bond L/S — XGB Driver Bank + GROPE (v6, ERI DAPY integrated)

Features
- Bank of ~50 XGB regressors as "drivers" (random hyperparams for diversity)
- Signals: rolling z-score → tanh(beta) → clipped [-1,1]
- Selection: greedy, p-value gated, **diversity penalty**; objective mixes **DAPY + IR**
- **DAPY styles**: `hits`, `eri_long`, `eri_short`, `eri_both` (switch via `--dapy_style`)
- Optimizer: GROPE-style (LHS + RBF surrogate + adaptive sampling) for **softmax weights + temperature**
- Walk-Forward OOS backtest (stitched series) + artifact CSVs
- Production model training (unrealistic backtest) + CSV
- Random **past**-train → test from date; Random **in-period** train → full in-period backtest

Install
```
python -m pip install numpy pandas scikit-learn xgboost
```

Quick demo
```
python main.py --synthetic --n_features 600 --n_models 50 --n_select 12 --folds 6   --w_dapy 1.0 --w_ir 1.0 --diversity_penalty 0.25 --dapy_style eri_both   --train_production   --test_start 2018-01-01   --random_inperiod_train_pct 0.6 --randtrain_seed 123
```

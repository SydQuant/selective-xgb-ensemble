@echo off
echo Starting XGBoost Comparison Framework - Hit Rate Q-Metric Tests
echo Running 6 parallel tests: standard/deep/tiered x tanh/binary with hit_rate Q-metric
echo.

REM Create unique window titles for each process
start "Standard Tanh Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'standard' --log_label 'standard_tanh_hitrate' --q_metric 'hit_rate'"

start "Standard Binary Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'standard' --log_label 'standard_binary_hitrate' --binary_signal --q_metric 'hit_rate'"

start "Deep Tanh Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'deep' --log_label 'deep_tanh_hitrate' --q_metric 'hit_rate'"

start "Deep Binary Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'deep' --log_label 'deep_binary_hitrate' --binary_signal --q_metric 'hit_rate'"

start "Tiered Tanh Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'tiered' --log_label 'tiered_tanh_hitrate' --q_metric 'hit_rate'"

start "Tiered Binary Hit Rate" cmd /k "~/anaconda3/python.exe xgb_compare/xgb_compare_clean.py --target_symbol '@ES#C' --start_date '2015-01-01' --end_date '2025-08-01' --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type 'tiered' --log_label 'tiered_binary_hitrate' --binary_signal --q_metric 'hit_rate'"

echo.
echo All 6 tests launched in parallel with hit_rate Q-metric!
echo Check individual windows for progress.
echo Results will be saved to xgb_compare/results/
echo Logs will be saved to xgb_compare/results/logs/
echo.
pause
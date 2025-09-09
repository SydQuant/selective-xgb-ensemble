#!/bin/bash

# Sequential execution of all test commands from test_commands.txt
# Updated for xgb_compare_clean.py only (adapted for Mac)

echo "Starting sequential execution of all XGBoost comparison tests..."
echo "=================================================="

# Command 2: Standard Binary
echo "Running Command 2: Standard Binary..."
python xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "standard" --log_label "standard_binary" --binary_signal

echo "=================================================="

# Command 3: Deep Tanh
echo "Running Command 3: Deep Tanh..."
python xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "deep" --log_label "deep_tanh"

echo "=================================================="

# Command 4: Deep Binary
echo "Running Command 4: Deep Binary..."
python xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "deep" --log_label "deep_binary" --binary_signal

echo "=================================================="

# Command 5: Tiered Tanh
echo "Running Command 5: Tiered Tanh..."
python xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "tiered" --log_label "tiered_tanh"

echo "=================================================="

# Command 6: Tiered Binary
echo "Running Command 6: Tiered Binary..."
python xgb_compare/xgb_compare_clean.py --target_symbol "@ES#C" --start_date "2015-01-01" --end_date "2025-08-01" --n_models 100 --n_folds 15 --cutoff_fraction 0.6 --xgb_type "tiered" --log_label "tiered_binary" --binary_signal

echo "=================================================="
echo "All tests completed!"
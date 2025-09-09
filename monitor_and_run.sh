#!/bin/bash

# Monitor background task and automatically run remaining tests
TASK_ID="de8273"

echo "Monitoring background task $TASK_ID..."
echo "Will automatically start remaining tests when it completes."

# Check task status every 2 minutes
while true; do
    # Check if background task is still running
    if ps aux | grep -v grep | grep "python.*xgb_compare.*standard_tanh" > /dev/null; then
        echo "$(date): Background task still running..."
        sleep 120  # Wait 2 minutes
    else
        echo "$(date): Background task completed! Starting remaining tests..."
        break
    fi
done

# Run the remaining test scripts
echo "=================================================="
echo "Starting sequential execution of remaining tests..."
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
echo "Results saved to xgb_compare/results/"
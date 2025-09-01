
import numpy as np
from typing import List, Tuple

def wfo_splits(n: int, k_folds: int = 6, min_train: int = 252) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward out-of-sample splits with expanding training window.
    
    FIXED: Ensure indices don't exceed data length and handle small datasets properly.
    """
    if k_folds < 2:
        k_folds = 2
    
    # Adjust min_train for small datasets
    min_train = min(min_train, n // 2)
    
    fold_size = max(1, n // k_folds)
    splits = []
    
    for f in range(1, k_folds+1):
        test_start = (f-1) * fold_size
        test_end = min(f * fold_size, n) if f < k_folds else n
        
        # FIXED: For small datasets, use expanding window but avoid overlap
        if test_start == 0:
            # First fold: use early data for train, later for test
            effective_min_train = min(min_train, test_end // 2)
            train_end = max(10, effective_min_train)
            test_start = max(train_end, fold_size)  # Ensure no overlap
            test_end = min(test_start + fold_size, n)
        else:
            # Later folds: standard walk-forward
            train_end = test_start  # No overlap allowed
            
        # Skip if we don't have enough data for this fold
        if train_end <= 10 or test_start >= n or test_end <= test_start or test_start < train_end:
            continue
            
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        # Final safety check
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
            
        # Ensure indices don't exceed bounds
        train_idx = train_idx[train_idx < n]
        test_idx = test_idx[test_idx < n]
        
        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, test_idx))
    
    return splits

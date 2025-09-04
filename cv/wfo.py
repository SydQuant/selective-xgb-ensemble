import numpy as np
from typing import Generator, Tuple

def wfo_splits(n: int, k_folds: int = 6, min_train: int = 252) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Walk-forward cross-validation splits with expanding window and no data leakage.
    
    CRITICAL FIX: Previous implementation had data leakage where training data 
    overlapped with test data in early folds. This version ensures NO OVERLAP.
    
    Args:
        n: Total number of samples in dataset
        k_folds: Number of folds (default 6)
        min_train: Minimum training samples required (default 252 = ~1 year)
    
    Yields:
        (train_indices, test_indices) tuples with NO OVERLAP
    """
    if k_folds < 2:
        k_folds = 2
    fold_size = max(1, n // k_folds)
    
    for f in range(1, k_folds + 1):
        t0 = (f - 1) * fold_size  # test start
        t1 = f * fold_size if f < k_folds else n  # test end
        
        # Require enough history; otherwise skip this fold
        if t0 < min_train:
            continue
            
        # CRITICAL FIX: Ensure training data NEVER overlaps with test data
        # OLD BUG: train_end = max(t0, min_train)  # Could extend past t0!
        # NEW FIX: enforce train_end <= t0 to eliminate data leakage
        train_end = t0
        
        if train_end <= 0 or t1 <= t0:
            continue
            
        # EXPANDING WINDOW: Train from start (0) to test start (t0)
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(t0, t1)
        yield train_idx, test_idx
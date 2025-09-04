import numpy as np
from typing import List, Tuple

def wfo_splits(n: int, k_folds: int = 6, min_train: int = 252) -> List[Tuple[np.ndarray, np.ndarray]]:
    if k_folds < 2:
        k_folds = 2
    fold_size = max(1, n // k_folds)
    splits = []
    for f in range(1, k_folds+1):
        test_start = (f-1) * fold_size
        test_end = f * fold_size if f < k_folds else n
        train_end = max(test_start, min_train)
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        if len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits
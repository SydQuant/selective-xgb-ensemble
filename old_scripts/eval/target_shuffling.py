
import numpy as np
import pandas as pd
from typing import Callable, Tuple

def block_permutation(series: pd.Series, block: int = 10, rng: np.random.Generator = None) -> pd.Series:
    if rng is None:
        rng = np.random.default_rng()
    x = series.values
    n = len(x)
    if block < 1:
        block = 1
    blocks = [x[i:i+block] for i in range(0, n, block)]
    order = rng.permutation(len(blocks))
    x_perm = np.concatenate([blocks[i] for i in order])[:n]
    return pd.Series(x_perm, index=series.index)

def shuffle_pvalue(
    signal: pd.Series,
    target_ret: pd.Series,
    metric_fn: Callable[[pd.Series, pd.Series], float],
    n_shuffles: int = 200,
    block: int = 10,
    tail: str = "right",
    rng: np.random.Generator = None,
) -> Tuple[float, float, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(123)
    obs = metric_fn(signal, target_ret)
    null_vals = []
    for _ in range(n_shuffles):
        y_perm = block_permutation(target_ret, block=block, rng=rng)
        null_vals.append(metric_fn(signal, y_perm))
    null = np.array(null_vals)
    if tail == "right":
        # CHANGED: Use Monte Carlo p-value formula (add 1 to numerator and denominator)
        count_extreme = (null >= obs).sum()
        p = float((count_extreme + 1) / (n_shuffles + 1))
    elif tail == "left":
        count_extreme = (null <= obs).sum()
        p = float((count_extreme + 1) / (n_shuffles + 1))
    else:
        count_extreme = (np.abs(null) >= abs(obs)).sum()
        p = float((count_extreme + 1) / (n_shuffles + 1))
    return p, obs, null

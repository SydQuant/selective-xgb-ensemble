
import numpy as np
import pandas as pd
from typing import List, Tuple, Callable
from metrics.perf import information_ratio

def combo_metric(signal: pd.Series, y: pd.Series, w_dapy: float = 1.0, w_ir: float = 1.0, dapy_fn=None) -> float:
    d = dapy_fn(signal, y) if dapy_fn is not None else 0.0
    ir = information_ratio(signal, y)
    return float(w_dapy * d + w_ir * ir)

def pick_top_n_greedy_diverse(
    train_signals: List[pd.Series],
    y_tr: pd.Series,
    n: int,
    pval_gate: Callable[[pd.Series, pd.Series], bool],
    w_dapy: float = 1.0,
    w_ir: float = 1.0,
    diversity_penalty: float = 0.2,
    dapy_fn=None,
) -> List[int]:
    S: List[int] = []
    remaining = list(range(len(train_signals)))
    M = len(train_signals)
    corr = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            ci = np.corrcoef(train_signals[i].values, train_signals[j].values)[0,1]
            if not np.isfinite(ci): ci = 0.0
            corr[i, j] = corr[j, i] = ci
    while len(S) < n and remaining:
        best_score, best_idx = -1e18, None
        for i in remaining:
            s = train_signals[i]
            if not pval_gate(s, y_tr):
                continue
            base = combo_metric(s, y_tr, w_dapy=w_dapy, w_ir=w_ir, dapy_fn=dapy_fn)
            pen = 0.0 if len(S) == 0 else max(abs(corr[i, j]) for j in S)
            score = base - diversity_penalty * pen
            if score > best_score:
                best_score, best_idx = score, i
        if best_idx is None: break
        S.append(best_idx); remaining.remove(best_idx)
    return S

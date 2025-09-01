
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable
from metrics.perf import information_ratio, turnover as sig_turnover
from ensemble.combiner import softmax, combine_signals
from eval.target_shuffling import shuffle_pvalue

def weight_objective_factory(
    train_signals: List[pd.Series],
    y_tr: pd.Series,
    turnover_penalty: float = 0.05,
    pmax: float = 0.20,
    w_dapy: float = 1.0,
    w_ir: float = 1.0,
    metric_fn_dapy: Callable[[pd.Series, pd.Series], float] = None,
    metric_fn_ir: Callable[[pd.Series, pd.Series], float] = information_ratio,
):
    k = len(train_signals)
    def f(theta: Dict[str, float]) -> float:
        w = np.array([theta[f"w{i}"] for i in range(k)], dtype=float)
        tau = float(theta["tau"]); tau = min(max(tau, 0.2), 3.0)
        ww = softmax(w, temperature=tau)
        s = combine_signals(train_signals, ww)
        d = metric_fn_dapy(s, y_tr) if metric_fn_dapy is not None else 0.0
        ir = metric_fn_ir(s, y_tr)
        val = w_dapy * d + w_ir * ir
        to = sig_turnover(s); val -= turnover_penalty * to
        # p-value gate on the chosen DAPY metric
        if metric_fn_dapy is not None:
            pval, _, _ = shuffle_pvalue(s, y_tr, metric_fn_dapy, n_shuffles=200, block=10)
            if pval > pmax: return -1e9
        return float(val)
    return f

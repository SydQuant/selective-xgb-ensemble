
import numpy as np
import pandas as pd
from typing import List
from utils.transforms import zscore, tanh_squash

def build_driver_signals(train_preds, test_preds, y_tr, z_win=100, beta=1.0):
    s_tr, s_te = [], []
    for p_tr, p_te in zip(train_preds, test_preds):
        z_tr = zscore(p_tr, win=z_win)
        z_te = zscore(p_te, win=z_win)
        s_tr.append(tanh_squash(z_tr, beta=beta))
        s_te.append(tanh_squash(z_te, beta=beta))
    return s_tr, s_te

def softmax(w: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    a = (w / max(1e-8, temperature))
    a = a - a.max()
    e = np.exp(a)
    return e / e.sum()

def combine_signals(signals: List[pd.Series], weights: np.ndarray) -> pd.Series:
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)
    out = None
    for s, w in zip(signals, weights):
        out = s.mul(w) if out is None else out.add(s.mul(w), fill_value=0.0)
    return out.clip(-1, 1).fillna(0.0)

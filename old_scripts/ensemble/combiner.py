
import numpy as np
import pandas as pd
from typing import List

# Signal transformation utilities (consolidated from utils.transforms)
def zscore(series: pd.Series, win: int = 100) -> pd.Series:
    """Rolling z-score normalization with configurable window."""
    x = series.astype(float)
    mu = x.rolling(win, min_periods=10).mean()
    sd = x.rolling(win, min_periods=10).std(ddof=0).replace(0, np.nan)
    z = (x - mu) / sd
    return z.fillna(0.0)

def tanh_squash(x: pd.Series, beta: float = 1.0) -> pd.Series:
    """Tanh squashing to [-1,1] range with beta parameter control."""
    return np.tanh(beta * x).astype(float)

def build_driver_signals(train_preds, test_preds, y_tr, z_win=100, beta=1.0):
    """Transform raw XGBoost predictions into normalized trading signals [-1,1]."""
    s_tr, s_te = [], []
    for i, (p_tr, p_te) in enumerate(zip(train_preds, test_preds)):
        if p_tr is None or p_te is None:
            continue
        
        # Step 1: Rolling z-score normalization (standardize relative to recent predictions)
        z_tr = zscore(p_tr, win=z_win)
        z_te = zscore(p_te, win=z_win)
        # Step 2: Tanh squashing to bounded range [-1,1] with beta parameter control
        s_tr.append(tanh_squash(z_tr, beta=beta))
        s_te.append(tanh_squash(z_te, beta=beta))
    
    return s_tr, s_te

def softmax(w: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert raw weights to normalized probabilities with temperature control."""
    # Scale by temperature (higher temp = more uniform, lower temp = more concentrated)
    a = (w / max(1e-8, temperature))
    a = a - a.max()  # Numerical stability
    e = np.exp(a)
    return e / e.sum()  # Normalize to probabilities

def combine_signals(signals: List[pd.Series], weights: np.ndarray) -> pd.Series:
    """Combine multiple signals using weighted sum, bounded to [-1,1]."""
    weights = np.asarray(weights, dtype=float)
    weights = weights / (weights.sum() + 1e-12)  # Ensure normalization
    out = None
    # Weighted combination of all signals
    for s, w in zip(signals, weights):
        out = s.mul(w) if out is None else out.add(s.mul(w), fill_value=0.0)
    return out.clip(-1, 1).fillna(0.0)  # Final signal bounded and cleaned

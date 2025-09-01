
import numpy as np
import pandas as pd

def zscore(series: pd.Series, win: int = 100) -> pd.Series:
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    
    x = series.astype(float)
    
    # For small datasets, use the entire series length as window
    effective_win = min(win or 100, len(x))
    min_periods = min(10, max(2, effective_win // 2))  # At least 2, at most half the window
    
    if effective_win <= 0:
        return pd.Series(0.0, index=series.index)
    
    mu = x.rolling(effective_win, min_periods=min_periods).mean()
    sd = x.rolling(effective_win, min_periods=min_periods).std(ddof=0)

    z = (x - mu) / sd
    return z.fillna(0.0)

def tanh_squash(x: pd.Series, beta: float = 1.0) -> pd.Series:
    if x is None:
        return pd.Series(dtype=float)
    return np.tanh(beta * x).astype(float)

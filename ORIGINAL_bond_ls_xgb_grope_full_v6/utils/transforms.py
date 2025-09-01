
import numpy as np
import pandas as pd

def zscore(series: pd.Series, win: int = 100) -> pd.Series:
    x = series.astype(float)
    mu = x.rolling(win, min_periods=10).mean()
    sd = x.rolling(win, min_periods=10).std(ddof=0).replace(0, np.nan)
    z = (x - mu) / sd
    return z.fillna(0.0)

def tanh_squash(x: pd.Series, beta: float = 1.0) -> pd.Series:
    return np.tanh(beta * x).astype(float)

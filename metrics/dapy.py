
import numpy as np
import pandas as pd

def dapy_from_binary_hits(signal: pd.Series, target_ret: pd.Series, annual_trading_days: int = 252) -> float:
    s = np.sign(signal.values)
    y = np.sign(target_ret.reindex_like(signal).values)
    mask = ~np.isnan(s) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    hits = (s[mask] == y[mask]).mean()
    return float((2 * hits - 1.0) * annual_trading_days)

def hit_rate(signal: pd.Series, target_ret: pd.Series) -> float:
    s = np.sign(signal.values)
    y = np.sign(target_ret.reindex_like(signal).values)
    mask = ~np.isnan(s) & ~np.isnan(y)
    if mask.sum() == 0:
        return 0.0
    return float((s[mask] == y[mask]).mean())


import numpy as np
import pandas as pd

def information_ratio(signal: pd.Series, target_ret: pd.Series, annual_trading_days: int = 252) -> float:
    pnl = (signal.shift(1).fillna(0.0) * target_ret.reindex_like(signal)).astype(float)
    mu = pnl.mean()
    sd = pnl.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float((mu / sd) * np.sqrt(annual_trading_days))

def turnover(signal: pd.Series) -> float:
    return float(np.abs(signal.diff().fillna(0.0)).mean())

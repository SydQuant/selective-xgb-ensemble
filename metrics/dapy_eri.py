
import numpy as np
import pandas as pd

def _ann(mu_daily: float, days: int) -> float:
    return float(mu_daily * days)

def _ann_vol(daily_std: float, days: int) -> float:
    return float(daily_std * np.sqrt(days))

def _bh_stats(returns: pd.Series, days: int = 253):
    r = returns.astype(float)
    mu_d = float(r.mean())
    sd_d = float(r.std(ddof=0))
    ann_ret = _ann(mu_d, days)
    ann_vol = _ann_vol(sd_d, days)
    return ann_ret, ann_vol, mu_d, sd_d

def dapy_eri_long(signal: pd.Series, returns: pd.Series, days: int = 253) -> float:
    pos = (signal > 0).astype(float)
    pnl = (pos.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    ann_ret_long = _ann(float(pnl.mean()), days)
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    in_mkt_long = float(pos.mean())
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float((ann_ret_long - bh_ann_ret * in_mkt_long) / denom)

def dapy_eri_short(signal: pd.Series, returns: pd.Series, days: int = 253) -> float:
    neg = (signal < 0).astype(float)
    pos_short = -neg
    pnl = (pos_short.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    ann_ret_short = _ann(float(pnl.mean()), days)
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    in_mkt_short = float(neg.mean())
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom):
        return 0.0
    return float((ann_ret_short - (-bh_ann_ret) * in_mkt_short) / denom)

def dapy_eri_both(signal: pd.Series, returns: pd.Series, days: int = 253) -> float:
    """
    Correct implementation of combined long+short DAPY_both formula
    DAPY_both = (AnnRet_both - BandH * (inMkt_long-inMkt_short)) / (AnnVol_BandH/sqrt(253))
    """
    # Calculate positions
    pos_long = (signal > 0).astype(float)
    pos_short = (signal < 0).astype(float)
    
    # Combined position (long +1, short -1)
    combined_pos = pos_long - pos_short
    
    # Combined P&L
    pnl_both = (combined_pos.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    ann_ret_both = _ann(float(pnl_both.mean()), days)
    
    # Buy and hold stats
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    
    # Market exposure
    in_mkt_long = float(pos_long.mean())
    in_mkt_short = float(pos_short.mean())
    
    # Denominator
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom):
        return 0.0
    
    # Combined DAPY formula
    return float((ann_ret_both - bh_ann_ret * (in_mkt_long - in_mkt_short)) / denom)

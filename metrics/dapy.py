
import numpy as np
import pandas as pd

# ---- DAPY ERI Functions (merged from dapy_eri.py) ----

def _ann(mu_daily: float, days: int) -> float:
    return float(mu_daily * days)

def _ann_vol(daily_std: float, days: int) -> float:
    return float(daily_std * np.sqrt(days))

def _bh_stats(returns: pd.Series, days: int = 253):
    """Calculate buy-and-hold statistics with robust error handling"""
    r = returns.astype(float)
    
    if len(r) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    r_clean = r.dropna()
    if len(r_clean) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    mu_d = float(r_clean.mean())
    sd_d = float(r_clean.std(ddof=0))
    
    if np.isnan(mu_d):
        mu_d = 0.0
    if np.isnan(sd_d):
        sd_d = 0.0
    
    ann_ret = _ann(mu_d, days)
    ann_vol = _ann_vol(sd_d, days)
    return ann_ret, ann_vol, mu_d, sd_d

def dapy_from_binary_hits(signal: pd.Series, target_ret: pd.Series, annual_trading_days: int = 252, **kwargs) -> float:
    """DAPY from binary hits with proper temporal alignment - simplified and GROPE compatible"""
    # Handle edge cases
    if len(signal) == 0 or len(target_ret) == 0:
        return 0.0
    
    # Apply temporal lag: signal T-1 predicts return T
    lagged_signal = signal.shift(1).fillna(0.0)
    aligned_returns = target_ret.reindex_like(signal)
    
    # Calculate directional accuracy
    signal_dir = np.sign(lagged_signal.values)
    return_dir = np.sign(aligned_returns.values)
    
    # Valid observations (non-NaN)
    valid_mask = ~(np.isnan(signal_dir) | np.isnan(return_dir))
    if not np.any(valid_mask):
        return 0.0
    
    # Hit rate and DAPY calculation
    hit_rate = np.mean(signal_dir[valid_mask] == return_dir[valid_mask])
    return float((2 * hit_rate - 1.0) * annual_trading_days)

def hit_rate(signal: pd.Series, target_ret: pd.Series, **kwargs) -> float:
    """Hit rate with proper temporal alignment - simplified and GROPE compatible"""
    # Handle edge cases
    if len(signal) == 0 or len(target_ret) == 0:
        return 0.0
    
    # Apply temporal lag: signal T-1 predicts return T
    lagged_signal = signal.shift(1).fillna(0.0)
    aligned_returns = target_ret.reindex_like(signal)
    
    # Calculate directional accuracy
    signal_dir = np.sign(lagged_signal.values)
    return_dir = np.sign(aligned_returns.values)
    
    # Valid observations (non-NaN)
    valid_mask = ~(np.isnan(signal_dir) | np.isnan(return_dir))
    if not np.any(valid_mask):
        return 0.0
    
    return float(np.mean(signal_dir[valid_mask] == return_dir[valid_mask]))

# ---- DAPY ERI Variants ----

def dapy_eri_long(signal: pd.Series, returns: pd.Series, days: int = 253, **kwargs) -> float:
    """DAPY ERI Long with robust error handling and proper temporal alignment"""
    if len(signal) == 0 or len(returns) == 0:
        return 0.0
    
    pos = (signal > 0).astype(float)
    pnl = (pos.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    
    pnl_clean = pnl.dropna()
    if len(pnl_clean) == 0:
        return 0.0
    
    ann_ret_long = _ann(float(pnl_clean.mean()), days)
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    
    pos_clean = pos.dropna()
    in_mkt_long = float(pos_clean.mean()) if len(pos_clean) > 0 else 0.0
    
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom) or np.isnan(ann_ret_long) or np.isnan(bh_ann_ret):
        return 0.0
    return float((ann_ret_long - bh_ann_ret * in_mkt_long) / denom)

def dapy_eri_short(signal: pd.Series, returns: pd.Series, days: int = 253, **kwargs) -> float:
    """DAPY ERI Short with robust error handling and proper temporal alignment"""
    if len(signal) == 0 or len(returns) == 0:
        return 0.0
    
    neg = (signal < 0).astype(float)
    pos_short = -neg
    pnl = (pos_short.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    
    pnl_clean = pnl.dropna()
    if len(pnl_clean) == 0:
        return 0.0
    
    ann_ret_short = _ann(float(pnl_clean.mean()), days)
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    
    neg_clean = neg.dropna()
    in_mkt_short = float(neg_clean.mean()) if len(neg_clean) > 0 else 0.0
    
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom) or np.isnan(ann_ret_short) or np.isnan(bh_ann_ret):
        return 0.0
    return float((ann_ret_short - (-bh_ann_ret) * in_mkt_short) / denom)

def dapy_eri_both(signal: pd.Series, returns: pd.Series, days: int = 253, **kwargs) -> float:
    """
    Combined long+short DAPY_both formula with robust error handling
    DAPY_both = (AnnRet_both - BandH * (inMkt_long-inMkt_short)) / (AnnVol_BandH/sqrt(253))
    """
    if len(signal) == 0 or len(returns) == 0:
        return 0.0
    
    pos_long = (signal > 0).astype(float)
    pos_short = (signal < 0).astype(float)
    
    # Combined position (long +1, short -1)
    combined_pos = pos_long - pos_short
    
    # Combined P&L
    pnl_both = (combined_pos.shift(1).fillna(0.0) * returns.reindex_like(signal)).astype(float)
    
    pnl_clean = pnl_both.dropna()
    if len(pnl_clean) == 0:
        return 0.0
    
    ann_ret_both = _ann(float(pnl_clean.mean()), days)
    
    # Buy and hold stats
    bh_ann_ret, bh_ann_vol, _, _ = _bh_stats(returns.reindex_like(signal), days)
    
    # Market exposure
    pos_long_clean = pos_long.dropna()
    pos_short_clean = pos_short.dropna()
    in_mkt_long = float(pos_long_clean.mean()) if len(pos_long_clean) > 0 else 0.0
    in_mkt_short = float(pos_short_clean.mean()) if len(pos_short_clean) > 0 else 0.0
    
    # Denominator
    denom = bh_ann_vol / np.sqrt(days)
    if denom == 0 or np.isnan(denom) or np.isnan(ann_ret_both) or np.isnan(bh_ann_ret):
        return 0.0
    
    # Combined DAPY formula
    return float((ann_ret_both - bh_ann_ret * (in_mkt_long - in_mkt_short)) / denom)

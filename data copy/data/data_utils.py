import pandas as pd
import numpy as np
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)

"""
Data utilities for cleaning and preparing market data from ArcticDB.
Simplified for ArcticDB-based data ingestion.
"""

# Data cleaning constants
CLIP_BOUNDS = (-1e6, 1e6)

# === Private Helpers ===
def fill_intraday_gaps(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Forward-fill hourly gaps within each trading day.
    """
    counts = df.groupby(df.index.date).size()
    avg_bars = math.ceil(counts.mean())
    logger.info(f"fill_intraday_gaps: avg bars/day for {symbol or '<unknown>'}: {avg_bars}")
    
    filled = []
    for day, grp in df.groupby(df.index.date):
        ts_day = pd.Timestamp(day)
        weekday = ts_day.weekday()  # Mon=0 ... Sun=6
        if weekday == 5:  # skip Saturday
            continue
        day_start = grp.index.min().floor('h')
        day_end = grp.index.max().floor('h')
        if weekday == 6:  # Sunday: start at 19:00
            day_start = max(day_start, ts_day + pd.Timedelta(hours=19))
        if weekday == 4:  # Friday: end at 17:00
            day_end = min(day_end, ts_day + pd.Timedelta(hours=17))
        
        # Create hourly range and reindex
        hourly_range = pd.date_range(day_start, day_end, freq='h')
        day_filled = grp.reindex(hourly_range, method='ffill')
        filled.append(day_filled)
    
    return pd.concat(filled).sort_index() if filled else df


def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic technical indicators for market data.
    """
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR calculation
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    
    # Basic features
    features = {
        'rsi': rsi,
        'atr': atr,
        'sma_20': df['close'].rolling(20).mean(),
        'price_change': df['close'].pct_change(),
        'volume_change': df['volume'].pct_change()
    }
    
    # Add features to original DataFrame
    feature_df = pd.DataFrame(features, index=df.index)
    return pd.concat([df, feature_df], axis=1)


def calculate_cross_asset_correlations(
    target_symbol: str, raw_data: Dict[str, pd.DataFrame], signal_hour_data_index=None
) -> pd.DataFrame:
    """
    Compute rolling cross-asset correlations.
    """
    # for each other symbol and period, compute rolling corr with target
    periods = list(range(2,13,2)) + list(range(13,25,4))
    # periods = [2]
    target_close = raw_data[target_symbol]['close']
    cols, out = [], []
    for sym, df in raw_data.items():
        if sym == target_symbol: continue
        idx = target_close.index.intersection(df.index)
        series = df.loc[idx,'close']
        for p in periods:
            if len(idx) <= p: continue
            corr = target_close.loc[idx].rolling(p).corr(series)
            cols.append(f"{target_symbol}_corr_{sym}_{p}h")
            out.append(corr)
    # if no other symbols, return empty DataFrame to avoid concat error
    if not out:
        return pd.DataFrame(index=signal_hour_data_index)
    corr_df = pd.concat(out, axis=1)
    corr_df.columns = cols
    # align correlations to desired index, allowing missing timestamps
    return corr_df.reindex(signal_hour_data_index)

def extract_signal_features(
    df: pd.DataFrame, signal_hour: int = 16, as_features: bool = False, symbol: str = None
) -> pd.DataFrame:
    """
    Filter to signal hour, compute indicators, and prefix if needed.
    """
    # Apply technical indicators
    df = calculate_features(df)
    df = df[df.index.hour == signal_hour]
    if as_features and symbol:
        df = df.add_prefix(f"{symbol}_")
    return df





def drop_sparse_features(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Drop features where fraction of zero values exceeds threshold.
    """
    to_drop = [c for c in df.columns if (df[c] == 0).mean() > threshold]
    if to_drop:
        logger.info(f"drop_sparse_features: Dropped {len(to_drop)} sparse features")
    return df.drop(columns=to_drop)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by:
     - replacing infinite values with NaN
     - dropping sparse features (>80% zeros)
     - forward-filling missing values
     - dropping rows with >50% missing
     - clipping values to CLIP_BOUNDS
    """
    logger.info(f"clean_df: start with shape {df.shape}")
    
    # Replace inf values and drop sparse features
    df_clean = (df.replace([np.inf, -np.inf], np.nan)
                 .pipe(drop_sparse_features))
    
    # Forward fill missing values
    df_clean = df_clean.ffill()
    
    # Drop rows with >50% missing values
    mask_drop = df_clean.isna().mean(axis=1) > 0.5
    if mask_drop.any():
        cnt = mask_drop.sum()
        logger.info(f"clean_df: dropping {cnt} rows with >50% missing")
        df_clean = df_clean.loc[~mask_drop]
    
    # Clip extreme values
    df_clean = df_clean.clip(*CLIP_BOUNDS)
    logger.info(f"clean_df: end with shape {df_clean.shape}")
    return df_clean

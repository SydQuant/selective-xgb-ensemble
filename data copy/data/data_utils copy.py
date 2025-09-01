from rich import print
import pandas as pd
import numpy as np
import math
from typing import Dict, Optional
from pathlib import Path

# === Config and Globals ===
from config.config_manager import log, FEATURE_SYMBOLS, DATA_DIR, SIGNAL_HOUR, instrument_config, cfg as config, FEATURE_CONFIG_DIR, DYNAMIC_SIGNAL_HOUR

"""
Data utilities for cleaning, scaling, and preparing both training and live market data.
"""

# === Data Cleaning Constants ===
CLIP_BOUNDS = (-1e6, 1e6)

# === Training Data Preparation ===

def prepare_training_data(target_symbol: str, live_data: Optional[Dict[str, pd.DataFrame]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, dropna: bool = True, n_hours_override: Optional[int] = None) -> pd.DataFrame:
    """
    Build full training dataframe using txt files.
    """
    # determine lookahead hours and load data for all symbols
    n_hours = n_hours_override or instrument_config.get(target_symbol, {}).get('n_hours', 3)
    print(f'target_symbol, {target_symbol}.  n_hours, {n_hours}, signal_hour, {SIGNAL_HOUR}')
    symbols = [target_symbol] + [s for s in FEATURE_SYMBOLS if s != target_symbol]
    if live_data:
        for df in live_data.values():
            df.index = df.index.tz_localize(None)
        raw_data = live_data
    else:
        raw_data = {s: load_historical_data(s, start_date, end_date) for s in symbols}
    
    # compute target return series
    target_df = get_target_df(raw_data, target_symbol, n_hours, dropna)
    # keep only target return to avoid duplicate feature columns
    target_col = f"{target_symbol}_target_return"
    target_df = target_df[[target_col]]
    # build feature matrix from correlations and signal features
    feature_df = get_feature_df(raw_data, target_symbol)
    # Merge on timestamp and drop missing targets
    df = target_df.join(feature_df, how='left')
    if dropna:
        df = df.dropna(subset=[f"{target_symbol}_target_return"])
    df = clean_df(df)
    return df



# === Historical Data Loading ===
def load_historical_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load OHLCV data for a symbol from txt files.
    """
    # read raw OHLCV file
    path = Path(DATA_DIR) / f"{symbol}_60.txt"
    df = pd.read_csv(path, names=["date", "time", "open", "high", "low", "close", "volume"])
    # ensure date/time are strings and padded
    df["date"] = df["date"].astype(str).str.zfill(6)
    df["time"] = df["time"].astype(str).str.zfill(5)
    # create timestamp and set as index
    df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%y%m%d%H:%M")
    df = df.set_index("timestamp")[['open','high','low','close','volume']]

    # filter to only include data from one business day before start_date
    if start_date:
        from pandas.tseries.offsets import BDay
        sd = pd.to_datetime(start_date)
        prev_bd = sd - BDay(1)
        df = df[df.index >= prev_bd]

    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df

# === Private Helpers ===
def _fill_intraday_gaps(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Forward-fill hourly gaps within each trading day.
    Logs only when gaps exceed the average daily bar count.
    """
    # compute typical bars per day for threshold
    counts = df.groupby(df.index.date).size()
    avg_bars = math.ceil(counts.mean())
    log(f"_fill_intraday_gaps: avg bars/day for {symbol or '<unknown>'}: {avg_bars}")
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
        hrs = pd.date_range(day_start, day_end, freq='h')
        missing = len(hrs) - len(grp)
        # only log large gaps
        if missing > avg_bars:
            log(f"_fill_intraday_gaps({symbol or '<unknown>'}): filled {missing} gaps on {day}, avg expected {avg_bars}")
        filled.append(grp.reindex(hrs).ffill())
    return pd.concat(filled).sort_index()


def _calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators (RSI and ATR) and period-based features.
    Optimized to avoid DataFrame fragmentation by using pd.concat.
    """
    # Create a copy to avoid modifying the input DataFrame
    result_df = df.copy()
    
    # # Compute RSI
    delta = result_df['close'].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean().replace(0, 1e-6)
    result_df['rsi'] = 100 - 100 / (1 + gain / loss)
    result_df['rsi'] = result_df['rsi'].replace(0, 1e-6)
    
    # Compute ATR as rolling average of the high-low range
    range_series = result_df[['high', 'low', 'close']].max(axis=1) - result_df[['high', 'low', 'close']].min(axis=1)
    result_df['atr'] = range_series.rolling(6).mean()

    periods = [1, 2, 3] + [p for p in range(4, 41) if p % 4 == 0]
    # periods = [1, 4]
    change = result_df['close'].pct_change(fill_method=None)
    
    new_columns = {}
    for p in periods:
        momentum = change.rolling(p).mean()
        new_columns[f'level_momentum_{p}h'] = momentum
        new_columns[f'momentum_{p}h'] = momentum
        new_columns[f'velocity_{p}h'] = momentum.pct_change(fill_method=None)

        rsi_pct = result_df['rsi'].pct_change(fill_method=None)
        atr_pct = result_df['atr'].pct_change(fill_method=None)
        new_columns[f'rsi_{p}h'] = rsi_pct.rolling(p).mean()
        new_columns[f'atr_{p}h'] = atr_pct.rolling(p).mean()
        new_columns[f'level_rsi_{p}h'] = rsi_pct.rolling(p).mean()
        new_columns[f'level_atr_{p}h'] = atr_pct.rolling(p).mean()
        new_columns[f'breakout_{p}h'] = result_df['close'] - result_df['close'].rolling(p).min()
    
    if new_columns:
        result_df = pd.concat([result_df, pd.DataFrame(new_columns, index=result_df.index)], axis=1)
    
    momentum_diffs = {
        "short_vs_long_momentum_1_4h": result_df["momentum_1h"] - result_df["momentum_4h"],
        "short_vs_long_momentum_3_12h": result_df["momentum_3h"] - result_df["momentum_12h"],
        "short_vs_long_momentum_8_24h": result_df["momentum_8h"] - result_df["momentum_24h"]
    }
    result_df = pd.concat([result_df, pd.DataFrame(momentum_diffs, index=result_df.index)], axis=1)

    return result_df

def _calculate_cross_asset_correlations(
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
    df: pd.DataFrame, as_features: bool = False, symbol: str = None, signal_hour: int = None
) -> pd.DataFrame:
    """
    Filter to signal hour, compute indicators, and prefix if needed.
    """
    # apply technical indicators
    df = _calculate_features(df)
    df = df[df.index.hour == signal_hour]
    if as_features and symbol:
        df = df.add_prefix(f"{symbol}_")
    return df

def get_target_df(raw_data: Dict[str, pd.DataFrame], target_symbol: str, n_hours: int, dropna:Optional[bool] = True) -> pd.DataFrame:
    """
    Generate target DataFrame with returns.
    """
    # ensure continuous hourly series
    df_hourly = _fill_intraday_gaps(raw_data[target_symbol], target_symbol)
    # compute forward returns over n_hours
    future = df_hourly['close'].shift(-n_hours)
    ret = (future - df_hourly['close']) / df_hourly['close']
    # extract signal-hour features with prefix
    df_signal = extract_signal_features(
        df_hourly,
        as_features=True,
        symbol=target_symbol,
        signal_hour=SIGNAL_HOUR
    )
    # attach target return aligned to signal hour index
    df_signal[f"{target_symbol}_target_return"] = ret.reindex(df_signal.index)
    if dropna:
        df_signal.dropna(subset=[f"{target_symbol}_target_return"], inplace=True)
    return df_signal

def get_feature_df(raw_data: Dict[str, pd.DataFrame], target_symbol: str) -> pd.DataFrame:
    """
    Combine cross-asset correlations and signal-hour features. (used in both research and live runs)
    """
    # Extract features WITHOUT percentage changes first
    main_feat = extract_signal_features(raw_data[target_symbol], as_features=True, symbol=target_symbol, signal_hour=SIGNAL_HOUR)
    corr_df = _calculate_cross_asset_correlations(target_symbol, raw_data, signal_hour_data_index=main_feat.index)
    other_feats = [extract_signal_features(df_temp, as_features=True, symbol=sym, signal_hour=SIGNAL_HOUR) for sym, df_temp in raw_data.items() if sym != target_symbol]

    # Concatenate ALL features first
    feat_df = main_feat
    feat_df = pd.concat([main_feat] + other_feats + [corr_df], axis=1)
    feat_df.replace(0, 1e-8, inplace=True)
    for col in feat_df.columns:
        if not col.startswith(f'{target_symbol}_level_'):
            feat_df[col] = feat_df[col].pct_change(fill_method=None)


    # remove raw price columns
    exclude = {'open','high','low','close','volume'}
    feat_df = feat_df[[c for c in feat_df.columns if c.split('_')[-1] not in exclude]]
    return feat_df

def drop_sparse_features(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """
    Drop features where fraction of zero values exceeds threshold.
    """
    to_drop = [c for c in df.columns if (df[c] == 0).mean() > threshold]
    if to_drop:
        log(f"drop_sparse_features: Dropped {len(to_drop)} sparse features")
    return df.drop(columns=to_drop)

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame by:
     - replacing infinite values with NaN
     - dropping sparse features (>80% zeros)
     - forward-filling missing values (warn if >50% filled per column)
     - dropping rows with >50% missing
     - clipping values to CLIP_BOUNDS
    """
    log(f"clean_df: start with shape {df.shape}")
    # pipeline: replace inf → drop sparse → ffill → warn fills → drop bad rows → clip
    df_clean = (df.replace([np.inf, -np.inf], np.nan)
                 .pipe(drop_sparse_features))
    na_before = df_clean.isna().sum()
    df_clean = df_clean.ffill()
    na_after = df_clean.isna().sum()
    pct_filled = (na_before - na_after) / len(df_clean)
    for col, pct in pct_filled.items():
        if pct > 0.5:
            log(f"clean_df: {col} ffill filled {pct*100:.1f}% values")
    # drop rows >50% missing
    mask_drop = df_clean.isna().mean(axis=1) > 0.5
    if mask_drop.any():
        cnt = mask_drop.sum()
        log(f"clean_df: dropping {cnt} rows with >50% missing")
        df_clean = df_clean.loc[~mask_drop]
    df_clean = df_clean.clip(*CLIP_BOUNDS)
    log(f"clean_df: end with shape {df_clean.shape}")
    return df_clean

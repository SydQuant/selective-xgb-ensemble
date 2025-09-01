"""
Data utilities for real market data loading and feature engineering.
Integrates Arctic DB loading with technical indicator computation.
"""

import pandas as pd
import numpy as np
import math
import os
from typing import Dict, Optional, List
import logging
from .loaders import get_arcticdb_connection
from .symbol_loader import get_default_symbols

logger = logging.getLogger(__name__)

# Data cleaning constants
CLIP_BOUNDS = (-1e6, 1e6)

def load_real_data(symbols: List[str], start_date: str = None, end_date: str = None, signal_hour: int = 12) -> Dict[str, pd.DataFrame]:
    """
    Load real market data from Arctic DB and format for XGB pipeline.
    Returns dict of {symbol: DataFrame} with OHLCV data.
    """
    try:
        futures_lib = get_arcticdb_connection()
        available_symbols = futures_lib.list_symbols()
        symbols_to_load = [s for s in symbols if s in available_symbols]
        
        if not symbols_to_load:
            logger.warning(f"No symbols found in Arctic DB from: {symbols}")
            return {}
            
        logger.info(f"Loading data for symbols: {symbols_to_load}")
        
        # Load data with date range optimization
        start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=30) if start_date else None
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=2) if end_date else None
        date_range = (start_dt, end_dt) if start_dt and end_dt else None

        raw_data = {}
        for symbol in symbols_to_load:
            try:
                data = futures_lib.read(symbol, date_range=date_range).data
                # Ensure we have required OHLCV columns
                if 'price' in data.columns:
                    data['close'] = data['price']
                
                # Fill missing OHLCV if needed
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in data.columns:
                        if col == 'volume':
                            data[col] = 1000.0  # Default volume
                        else:
                            data[col] = data['close']  # Use close for missing OHLC
                
                # Ensure hourly frequency and fill gaps
                data = fill_intraday_gaps(data, symbol)
                
                # Filter to signal hour if specified
                if signal_hour is not None:
                    data = data[data.index.hour == signal_hour]
                
                raw_data[symbol] = data[required_cols]
                logger.info(f"Loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                continue
                
        return raw_data
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        return {}

def fill_intraday_gaps(df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Forward-fill hourly gaps within each trading day.
    """
    if df.empty:
        return df
        
    counts = df.groupby(df.index.date).size()
    avg_bars = math.ceil(counts.mean()) if len(counts) > 0 else 24
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

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators for market data.
    Based on the original data_utils feature engineering.
    """
    result_df = df.copy()
    
    # RSI calculation (3-period for responsiveness)
    delta = result_df['close'].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean().replace(0, 1e-6)
    result_df['rsi'] = 100 - 100 / (1 + gain / loss)
    result_df['rsi'] = result_df['rsi'].replace(0, 1e-6)
    
    # ATR as rolling average of high-low range
    range_series = result_df[['high', 'low', 'close']].max(axis=1) - result_df[['high', 'low', 'close']].min(axis=1)
    result_df['atr'] = range_series.rolling(6).mean()

    # Multi-period momentum features
    periods = [1, 2, 3] + [p for p in range(4, 41) if p % 4 == 0]
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
    
    # Momentum differentials
    momentum_diffs = {
        "short_vs_long_momentum_1_4h": result_df["momentum_1h"] - result_df["momentum_4h"],
        "short_vs_long_momentum_3_12h": result_df["momentum_3h"] - result_df["momentum_12h"],
        "short_vs_long_momentum_8_24h": result_df["momentum_8h"] - result_df["momentum_24h"]
    }
    result_df = pd.concat([result_df, pd.DataFrame(momentum_diffs, index=result_df.index)], axis=1)

    return result_df

def calculate_cross_asset_correlations(
    target_symbol: str, raw_data: Dict[str, pd.DataFrame], signal_hour_data_index=None
) -> pd.DataFrame:
    """
    Compute rolling cross-asset correlations between target and other symbols.
    """
    periods = list(range(2,13,2)) + list(range(13,25,4))
    
    if target_symbol not in raw_data:
        logger.warning(f"Target symbol {target_symbol} not found in raw_data")
        return pd.DataFrame(index=signal_hour_data_index)
        
    target_close = raw_data[target_symbol]['close']
    cols, out = [], []
    
    for sym, df in raw_data.items():
        if sym == target_symbol: 
            continue
            
        idx = target_close.index.intersection(df.index)
        if len(idx) == 0:
            continue
            
        series = df.loc[idx,'close']
        for p in periods:
            if len(idx) <= p: 
                continue
            corr = target_close.loc[idx].rolling(p).corr(series)
            cols.append(f"{target_symbol}_corr_{sym}_{p}h")
            out.append(corr)
    
    # If no other symbols, return empty DataFrame
    if not out:
        return pd.DataFrame(index=signal_hour_data_index)
        
    corr_df = pd.concat(out, axis=1)
    corr_df.columns = cols
    
    # Align correlations to desired index
    if signal_hour_data_index is not None:
        corr_df = corr_df.reindex(signal_hour_data_index)
        
    return corr_df

def extract_signal_features(
    df: pd.DataFrame, signal_hour: int = 12, as_features: bool = False, symbol: str = None
) -> pd.DataFrame:
    """
    Filter to signal hour, compute indicators, and prefix if needed.
    """
    # Apply technical indicators
    df = calculate_technical_features(df)
    df = df[df.index.hour == signal_hour]
    
    if as_features and symbol:
        df = df.add_prefix(f"{symbol}_")
    return df

def get_target_df(raw_data: Dict[str, pd.DataFrame], target_symbol: str, n_hours: int, signal_hour: int = 12, dropna: bool = True) -> pd.DataFrame:
    """
    Generate target DataFrame with forward returns.
    """
    if target_symbol not in raw_data:
        logger.error(f"Target symbol {target_symbol} not found in raw_data")
        return pd.DataFrame()
        
    # Ensure continuous hourly series
    df_hourly = fill_intraday_gaps(raw_data[target_symbol], target_symbol)
    
    # Compute forward returns over n_hours
    future = df_hourly['close'].shift(-n_hours)
    ret = (future - df_hourly['close']) / df_hourly['close']
    
    # Extract signal-hour features with prefix
    df_signal = extract_signal_features(
        df_hourly,
        signal_hour=signal_hour,
        as_features=True,
        symbol=target_symbol
    )
    
    # Attach target return aligned to signal hour index
    df_signal[f"{target_symbol}_target_return"] = ret.reindex(df_signal.index)
    
    if dropna:
        df_signal.dropna(subset=[f"{target_symbol}_target_return"], inplace=True)
        
    return df_signal

def get_feature_df(raw_data: Dict[str, pd.DataFrame], target_symbol: str, signal_hour: int = 12) -> pd.DataFrame:
    """
    Combine cross-asset correlations and signal-hour features.
    """
    if target_symbol not in raw_data:
        logger.error(f"Target symbol {target_symbol} not found in raw_data")
        return pd.DataFrame()
        
    # Extract features for target symbol
    main_feat = extract_signal_features(
        raw_data[target_symbol], 
        signal_hour=signal_hour, 
        as_features=True, 
        symbol=target_symbol
    )
    
    # Cross-asset correlations
    corr_df = calculate_cross_asset_correlations(target_symbol, raw_data, signal_hour_data_index=main_feat.index)
    
    # Features from other symbols
    other_feats = []
    for sym, df_temp in raw_data.items():
        if sym != target_symbol:
            feat = extract_signal_features(df_temp, signal_hour=signal_hour, as_features=True, symbol=sym)
            other_feats.append(feat)

    # Concatenate all features
    feat_df = pd.concat([main_feat] + other_feats + [corr_df], axis=1)
    
    # Replace zeros to avoid division issues
    feat_df.replace(0, 1e-8, inplace=True)
    
    # Convert to percentage changes (except level features)
    for col in feat_df.columns:
        if not col.startswith(f'{target_symbol}_level_'):
            feat_df[col] = feat_df[col].pct_change(fill_method=None)

    # Remove raw price columns
    exclude = {'open','high','low','close','volume'}
    feat_df = feat_df[[c for c in feat_df.columns if c.split('_')[-1] not in exclude]]
    
    return feat_df

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
    Clean DataFrame by replacing inf, dropping sparse features, filling NaN, and clipping.
    """
    logger.info(f"clean_df: start with shape {df.shape}")
    
    # Replace inf values and drop sparse features
    df_clean = (df.replace([np.inf, -np.inf], np.nan)
                 .pipe(drop_sparse_features))
    
    # Forward fill missing values
    na_before = df_clean.isna().sum()
    df_clean = df_clean.ffill()
    na_after = df_clean.isna().sum()
    pct_filled = (na_before - na_after) / len(df_clean)
    
    for col, pct in pct_filled.items():
        if pct > 0.5:
            logger.info(f"clean_df: {col} ffill filled {pct*100:.1f}% values")
    
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

def prepare_real_data(target_symbol: str, symbols: List[str] = None, start_date: str = None, end_date: str = None, 
                     n_hours: int = 3, signal_hour: int = 12, max_features: int = None) -> pd.DataFrame:
    """
    Main function to prepare real market data for XGB training.
    
    Args:
        target_symbol: Symbol to predict
        symbols: List of symbols to use as features (defaults to symbols.yaml)
        start_date: Start date for data
        end_date: End date for data  
        n_hours: Lookahead hours for target return
        signal_hour: Hour for signal generation (default 12 = 1PM close)
        max_features: Limit features for testing (e.g., 50)
    """
    if symbols is None:
        symbols = get_default_symbols()
    
    # Ensure target symbol is included
    if target_symbol not in symbols:
        symbols = [target_symbol] + symbols
    
    logger.info(f"Preparing data for target={target_symbol}, symbols={len(symbols)}, signal_hour={signal_hour}")
    
    # Load raw market data
    raw_data = load_real_data(symbols, start_date, end_date, signal_hour=None)  # Load all hours first
    
    if not raw_data:
        logger.error("No data loaded from Arctic DB")
        return pd.DataFrame()
    
    # Generate target returns
    target_df = get_target_df(raw_data, target_symbol, n_hours, signal_hour)
    if target_df.empty:
        logger.error("Failed to generate target data")
        return pd.DataFrame()
    
    # Keep only target return column
    target_col = f"{target_symbol}_target_return"
    target_df = target_df[[target_col]]
    
    # Build feature matrix
    feature_df = get_feature_df(raw_data, target_symbol, signal_hour)
    if feature_df.empty:
        logger.error("Failed to generate features")
        return pd.DataFrame()
    
    # Merge target and features
    df = target_df.join(feature_df, how='left')
    
    # Drop missing targets
    df = df.dropna(subset=[target_col])
    
    # Clean data
    df = clean_df(df)
    
    # Limit features for testing if specified
    if max_features and len(df.columns) > max_features + 1:  # +1 for target
        feature_cols = [c for c in df.columns if c != target_col]
        selected_features = feature_cols[:max_features]
        df = df[[target_col] + selected_features]
        logger.info(f"Limited to {max_features} features for testing")
    
    logger.info(f"Final dataset shape: {df.shape}")
    return df
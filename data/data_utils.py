"""
Simplified data utilities for real market data loading and feature engineering.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Optional, List
import logging
from .loaders import get_arcticdb_connection
from .symbol_loader import get_default_symbols

logger = logging.getLogger(__name__)

def load_real_data(symbols: List[str], start_date: str = None, end_date: str = None, signal_hour: int = 12) -> Dict[str, pd.DataFrame]:
    """Load real market data from Arctic DB with OHLCV format."""
    try:
        futures_lib = get_arcticdb_connection()
        available_symbols = futures_lib.list_symbols()
        symbols_to_load = [s for s in symbols if s in available_symbols]
        
        if not symbols_to_load:
            logger.warning(f"No symbols found in Arctic DB from: {symbols}")
            return {}
            
        logger.info(f"Loading {len(symbols_to_load)} symbols from Arctic DB")
        
        # Date range optimization
        date_range = None
        if start_date and end_date:
            start_dt = pd.to_datetime(start_date) - pd.Timedelta(days=30)
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=2)
            date_range = (start_dt, end_dt)

        raw_data = {}
        for symbol in symbols_to_load:
            try:
                data = futures_lib.read(symbol, date_range=date_range).data
                
                # Ensure OHLCV columns exist
                if 'price' in data.columns:
                    data['close'] = data['price']
                
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in data.columns:
                        data[col] = data['close'] if col != 'volume' else 1000.0
                
                # Fill gaps and filter to signal hour
                data = fill_intraday_gaps(data, symbol)
                if signal_hour is not None:
                    data = data[data.index.hour == signal_hour]
                
                raw_data[symbol] = data[required_cols]
                logger.info(f"Loaded {len(data)} records for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load {symbol}: {e}")
                
        return raw_data
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        return {}

def fill_intraday_gaps(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    """Forward-fill hourly gaps within trading days."""
    if df.empty:
        return df
        
    filled = []
    for day, grp in df.groupby(df.index.date):
        ts_day = pd.Timestamp(day)
        weekday = ts_day.weekday()
        
        if weekday == 5:  # Skip Saturday
            continue
            
        # Trading hours: Sunday 19:00 - Friday 17:00
        day_start = grp.index.min().floor('h')
        day_end = grp.index.max().floor('h')
        
        if weekday == 6:  # Sunday start at 19:00
            day_start = max(day_start, ts_day + pd.Timedelta(hours=19))
        if weekday == 4:  # Friday end at 17:00
            day_end = min(day_end, ts_day + pd.Timedelta(hours=17))
        
        hourly_range = pd.date_range(day_start, day_end, freq='h')
        filled.append(grp.reindex(hourly_range, method='ffill'))
    
    return pd.concat(filled).sort_index() if filled else df

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators: RSI, ATR, momentum, and breakouts."""
    result = df.copy()
    
    # RSI (3-period for responsiveness)
    delta = result['close'].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean().replace(0, 1e-6)
    result['rsi'] = 100 - 100 / (1 + gain / loss)
    result['rsi'] = result['rsi'].replace(0, 1e-6)
    
    # ATR (Average True Range)
    range_series = result[['high', 'low', 'close']].max(axis=1) - result[['high', 'low', 'close']].min(axis=1)
    result['atr'] = range_series.rolling(6).mean()

    # Multi-period momentum features (reduced for less NaN)
    periods = [1, 2, 3, 4, 8, 12, 16, 20]  # Reduced periods
    change = result['close'].pct_change(fill_method=None)
    
    # Batch create features with min_periods to handle NaN
    features = {}
    for p in periods:
        momentum = change.rolling(p, min_periods=1).mean()  # Allow partial windows
        features[f'momentum_{p}h'] = momentum
        features[f'velocity_{p}h'] = momentum.pct_change(fill_method=None)
        features[f'rsi_{p}h'] = result['rsi'].pct_change(fill_method=None).rolling(p, min_periods=1).mean()
        features[f'atr_{p}h'] = result['atr'].pct_change(fill_method=None).rolling(p, min_periods=1).mean()
        features[f'breakout_{p}h'] = result['close'] - result['close'].rolling(p, min_periods=1).min()
    
    # Momentum differentials (only if both exist)
    if "momentum_1h" in features and "momentum_4h" in features:
        features.update({
            "momentum_diff_1_4h": features["momentum_1h"] - features["momentum_4h"],
            "momentum_diff_3_12h": features["momentum_3h"] - features["momentum_12h"],
            "momentum_diff_8_16h": features["momentum_8h"] - features["momentum_16h"]
        })
    
    return pd.concat([result, pd.DataFrame(features, index=result.index)], axis=1)

def calculate_cross_correlations(target_symbol: str, raw_data: Dict[str, pd.DataFrame], target_index=None) -> pd.DataFrame:
    """Compute rolling cross-asset correlations."""
    if target_symbol not in raw_data:
        return pd.DataFrame(index=target_index)
        
    periods = [2, 4, 6, 8, 10, 12, 16, 20, 24]
    target_close = raw_data[target_symbol]['close']
    
    corr_features = {}
    for sym, df in raw_data.items():
        if sym == target_symbol:
            continue
            
        idx = target_close.index.intersection(df.index)
        if len(idx) == 0:
            continue
            
        series = df.loc[idx, 'close']
        for p in periods:
            if len(idx) > p:
                corr = target_close.loc[idx].rolling(p).corr(series)
                corr_features[f"{target_symbol}_corr_{sym}_{p}h"] = corr
    
    if not corr_features:
        return pd.DataFrame(index=target_index)
        
    corr_df = pd.DataFrame(corr_features)
    return corr_df.reindex(target_index) if target_index is not None else corr_df

def build_feature_matrix(raw_data: Dict[str, pd.DataFrame], target_symbol: str, signal_hour: int = 12) -> pd.DataFrame:
    """Build complete feature matrix with technical indicators and cross-correlations."""
    if target_symbol not in raw_data:
        return pd.DataFrame()
    
    # Process target symbol features
    target_data = calculate_features(raw_data[target_symbol])
    target_features = target_data[target_data.index.hour == signal_hour].add_prefix(f"{target_symbol}_")
    
    # Process other symbol features
    other_features = []
    for sym, data in raw_data.items():
        if sym != target_symbol:
            sym_features = calculate_features(data)
            sym_features = sym_features[sym_features.index.hour == signal_hour].add_prefix(f"{sym}_")
            other_features.append(sym_features)
    
    # Cross-asset correlations
    corr_features = calculate_cross_correlations(target_symbol, raw_data, target_features.index)
    
    # Combine all features
    all_features = [target_features] + other_features + [corr_features]
    feature_df = pd.concat(all_features, axis=1)
    
    # Convert to percentage changes (except level features)
    feature_df = feature_df.replace(0, 1e-8)
    for col in feature_df.columns:
        if not col.startswith(f'{target_symbol}_level_'):
            feature_df[col] = feature_df[col].pct_change(fill_method=None)
    
    # Remove raw price columns
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_df = feature_df[[c for c in feature_df.columns if not any(c.endswith(pc) for pc in price_cols)]]
    
    return feature_df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: handle inf/nan, drop sparse features, clip extremes."""
    logger.info(f"Cleaning data: {df.shape}")
    
    # Replace inf and drop sparse features (>80% zeros)
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    sparse_cols = [c for c in df_clean.columns if (df_clean[c] == 0).mean() > 0.8]
    if sparse_cols:
        logger.info(f"Dropping {len(sparse_cols)} sparse features")
        df_clean = df_clean.drop(columns=sparse_cols)
    
    # Check NaN percentage before cleaning
    nan_pct_before = df_clean.isna().mean().mean()
    logger.info(f"NaN percentage before filling: {nan_pct_before:.1%}")
    
    # Forward fill and drop rows with >80% missing (more lenient)
    df_clean = df_clean.ffill()
    bad_rows = df_clean.isna().mean(axis=1) > 0.8  # More lenient threshold
    if bad_rows.any():
        logger.info(f"Dropping {bad_rows.sum()} rows with >80% missing")
        df_clean = df_clean.loc[~bad_rows]
    
    # Final backfill for remaining NaNs
    df_clean = df_clean.bfill()
    
    # Drop any remaining rows with NaN in target column
    target_cols = [c for c in df_clean.columns if c.endswith('_target_return')]
    if target_cols:
        df_clean = df_clean.dropna(subset=target_cols)
    
    # Clip extreme values
    df_clean = df_clean.clip(-1e6, 1e6)
    logger.info(f"Cleaned data: {df_clean.shape}")
    return df_clean

def prepare_real_data(target_symbol: str, symbols: List[str] = None, start_date: str = None, 
                     end_date: str = None, n_hours: int = 3, signal_hour: int = 12, 
                     max_features: int = None) -> pd.DataFrame:
    """
    Main function: prepare real market data for XGB training.
    
    Returns DataFrame with target column + features, ready for ML.
    """
    if symbols is None:
        symbols = get_default_symbols()
    
    if target_symbol not in symbols:
        symbols = [target_symbol] + symbols
    
    logger.info(f"Preparing data: target={target_symbol}, {len(symbols)} symbols, signal_hour={signal_hour}")
    
    # Load raw data
    raw_data = load_real_data(symbols, start_date, end_date, signal_hour=None)
    if not raw_data:
        logger.error("No data loaded from Arctic DB")
        return pd.DataFrame()
    
    # Generate target returns (forward n_hours)
    target_data = fill_intraday_gaps(raw_data[target_symbol], target_symbol)
    future_close = target_data['close'].shift(-n_hours)
    target_returns = (future_close - target_data['close']) / target_data['close']
    
    # Filter to signal hour and create target DataFrame
    target_signal_hour = target_data[target_data.index.hour == signal_hour]
    target_col = f"{target_symbol}_target_return"
    target_df = pd.DataFrame({target_col: target_returns.reindex(target_signal_hour.index)})
    
    # Build feature matrix
    feature_df = build_feature_matrix(raw_data, target_symbol, signal_hour)
    if feature_df.empty:
        logger.error("Failed to generate features")
        return pd.DataFrame()
    
    # Combine target + features
    df = target_df.join(feature_df, how='left')
    df = df.dropna(subset=[target_col])  # Remove missing targets
    df = clean_data(df)
    
    # Limit features for testing
    if max_features and len(df.columns) > max_features + 1:
        feature_cols = [c for c in df.columns if c != target_col][:max_features]
        df = df[[target_col] + feature_cols]
        logger.info(f"Limited to {max_features} features for testing")
    
    logger.info(f"Final dataset: {df.shape}")
    return df
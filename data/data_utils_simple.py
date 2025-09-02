"""
Simplified data utilities following original data_utils.py logic.
Uses rate of change (pct_change) for feature engineering as per original.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List
from .loaders import get_arcticdb_connection
from .symbol_loader import get_default_symbols

logger = logging.getLogger(__name__)

def calculate_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic technical features following original logic."""
    result = df.copy()
    
    # Basic RSI (3-period) - keep raw values, no artificial clipping
    delta = result['close'].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean()
    rs = gain / loss.replace(0, np.nan)
    result['rsi'] = (100 - 100 / (1 + rs)).fillna(50)  # Remove .clip(0, 100)
    
    # Basic ATR as percentage of close price (avoids extreme absolute values)
    high_low = result['high'] - result['low']
    atr_absolute = high_low.rolling(6).mean()
    result['atr'] = atr_absolute / result['close']  # ATR as % of price
    
    # Momentum features for various periods (following original pattern)
    periods = [1, 2, 3, 4, 8, 12, 16, 20]
    change = result['close'].pct_change(fill_method=None)
    
    for p in periods:
        momentum = change.rolling(p, min_periods=1).mean()
        result[f'momentum_{p}h'] = momentum.clip(-0.2, 0.2)
        
        velocity = momentum.pct_change(fill_method=None)
        result[f'velocity_{p}h'] = velocity.clip(-5, 5)
        
        rsi_rolling = result['rsi'].rolling(p, min_periods=1).mean()
        result[f'rsi_{p}h'] = rsi_rolling  # Remove artificial clipping
        
        # ATR change (already as % of price, so pct_change is appropriate)
        atr_change = result['atr'].pct_change(fill_method=None)
        result[f'atr_{p}h'] = atr_change.clip(-2, 2).rolling(p, min_periods=1).mean()
        
        # Breakout as percentage of recent low (fix 1h issue by using min period of 2)
        if p == 1:
            # For 1h, use 2-period min to avoid constant zero
            recent_low = result['close'].rolling(2, min_periods=2).min()
        else:
            recent_low = result['close'].rolling(p, min_periods=2).min()
        
        # Only calculate where we have valid recent_low
        result[f'breakout_{p}h'] = (result['close'] - recent_low) / recent_low
    
    # Momentum differentials
    if 'momentum_1h' in result.columns and 'momentum_4h' in result.columns:
        result['momentum_diff_1_4h'] = result['momentum_1h'] - result['momentum_4h']
        result['momentum_diff_3_12h'] = result['momentum_3h'] - result['momentum_12h']
        result['momentum_diff_8_16h'] = result['momentum_8h'] - result['momentum_16h']
    
    return result

def calculate_cross_correlations_simple(target_symbol: str, raw_data: Dict[str, pd.DataFrame], target_index) -> pd.DataFrame:
    """Simple cross-correlations following original pattern."""
    if target_symbol not in raw_data:
        return pd.DataFrame(index=target_index)
    
    periods = [2, 4, 6, 8, 10, 12, 16, 20]
    target_close = raw_data[target_symbol]['close']
    
    corr_features = {}
    for sym, df in raw_data.items():
        if sym == target_symbol:
            continue
            
        idx = target_close.index.intersection(df.index)
        if len(idx) <= 2:
            continue
            
        series = df.loc[idx, 'close']
        for p in periods:
            if len(idx) > p:
                corr = target_close.loc[idx].rolling(p).corr(series)
                corr = corr.fillna(0.0).clip(-1.0, 1.0)
                corr_features[f"{target_symbol}_corr_{sym}_{p}h"] = corr
    
    if not corr_features:
        return pd.DataFrame(index=target_index)
    
    corr_df = pd.DataFrame(corr_features)
    return corr_df.reindex(target_index)

def build_features_simple(raw_data: Dict[str, pd.DataFrame], target_symbol: str, signal_hour: int = 12) -> pd.DataFrame:
    """Build feature matrix following original data_utils.py logic."""
    if target_symbol not in raw_data:
        return pd.DataFrame()
    
    # Process target symbol
    target_data = calculate_simple_features(raw_data[target_symbol])
    target_features = target_data[target_data.index.hour == signal_hour].add_prefix(f"{target_symbol}_")
    
    # Process other symbols
    other_features = []
    for sym, data in raw_data.items():
        if sym != target_symbol:
            sym_features = calculate_simple_features(data)
            sym_features = sym_features[sym_features.index.hour == signal_hour].add_prefix(f"{sym}_")
            other_features.append(sym_features)
    
    # Cross-correlations
    corr_features = calculate_cross_correlations_simple(target_symbol, raw_data, target_features.index)
    
    # Combine features
    all_features = [target_features] + other_features + [corr_features]
    feature_df = pd.concat(all_features, axis=1)
    
    # CRITICAL: Apply pct_change as per original logic
    feature_df = feature_df.replace(0, 1e-8)
    for col in feature_df.columns:
        # Don't apply pct_change to already normalized indicators (RSI, correlations, breakout)
        # and momentum/velocity/atr which are already rates of change
        if not any(indicator in col.lower() for indicator in ['rsi', 'momentum', 'velocity', 'corr', 'atr', 'breakout']):
            feature_df[col] = feature_df[col].pct_change(fill_method=None)
    
    # Remove raw price columns
    price_cols = ['open', 'high', 'low', 'close', 'volume']
    feature_df = feature_df[[c for c in feature_df.columns if not any(c.endswith(pc) for pc in price_cols)]]
    
    return feature_df

def prepare_target_returns(raw_data: Dict[str, pd.DataFrame], target_symbol: str, n_hours: int = 24, signal_hour: int = 12) -> pd.Series:
    """Calculate target returns for a given symbol."""
    if target_symbol not in raw_data:
        return pd.Series(dtype=float)
    
    df = raw_data[target_symbol].copy()
    
    # Calculate future returns
    future_close = df['close'].shift(-n_hours)
    returns = (future_close - df['close']) / df['close']
    
    # Filter to signal hour
    signal_returns = returns[returns.index.hour == signal_hour]
    
    return signal_returns

def clean_data_simple(df: pd.DataFrame) -> pd.DataFrame:
    """Intelligent data cleaning that preserves high-quality features."""
    logger.info(f"Cleaning data: {df.shape}")
    
    target_cols = [c for c in df.columns if c.endswith('_target_return')]
    feature_cols = [c for c in df.columns if c not in target_cols]
    
    df_clean = df.copy()
    
    # 1. Handle inf values (replace with NaN for proper handling)
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # 2. Identify and remove truly problematic features
    features_to_drop = []
    
    for col in feature_cols:
        series = df_clean[col]
        n_total = len(series)
        n_nan = series.isna().sum()
        n_finite = series.replace([np.inf, -np.inf], np.nan).notna().sum()
        
        # Drop if >90% NaN (very low quality)
        if n_nan / n_total > 0.9:
            features_to_drop.append(col)
            continue
            
        # Check if constant (std < 1e-12 on finite values)
        finite_vals = series.dropna()
        if len(finite_vals) > 0 and finite_vals.std() < 1e-12:
            features_to_drop.append(col)
            logger.info(f"Dropping constant feature: {col} (value={finite_vals.iloc[0]:.2e})")
            continue
    
    if features_to_drop:
        logger.info(f"Dropping {len(features_to_drop)} low-quality features")
        df_clean = df_clean.drop(columns=features_to_drop)
    
    # 3. Time-series respecting missing value handling
    # Forward fill for time series continuity (no future leakage)
    df_clean = df_clean.ffill()
    
    # For remaining NaNs (typically at start due to rolling windows), 
    # check if they're a small number that can be handled, or if the feature should be dropped
    features_to_drop_nan = []
    
    for col in feature_cols:
        series = df_clean[col]
        nan_count = series.isna().sum()
        
        if nan_count > 0:
            nan_pct = nan_count / len(series)
            
            if nan_pct > 0.1:  # >10% NaN - drop the feature
                features_to_drop_nan.append(col)
                logger.info(f"Dropping feature with excessive NaNs: {col} ({nan_pct:.1%})")
            else:
                # Small number of NaNs (likely at start) - fill with 0 (no forward-looking bias)
                df_clean[col] = series.fillna(0.0)
                logger.info(f"Filled {nan_count} NaNs with 0 in {col}")
    
    if features_to_drop_nan:
        df_clean = df_clean.drop(columns=features_to_drop_nan)
        logger.info(f"Dropped {len(features_to_drop_nan)} features with excessive NaNs")
    
    # 4. Remove rows where target is NaN (essential for ML)
    if target_cols:
        target_col = target_cols[0]
        before_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=[target_col])
        after_rows = len(df_clean)
        if before_rows != after_rows:
            logger.info(f"Dropped {before_rows - after_rows} rows with missing target")
    
    # 5. Final quality check - no NaNs should remain in features
    feature_nan_count = df_clean[df_clean.columns.difference(target_cols)].isna().sum().sum()
    if feature_nan_count > 0:
        logger.warning(f"Still have {feature_nan_count} NaN values in features after cleaning")
    
    logger.info(f"Cleaned data: {df_clean.shape} (dropped {len(features_to_drop)} features)")
    return df_clean

def prepare_real_data_simple(target_symbol: str, symbols: List[str] = None, start_date: str = None, 
                           end_date: str = None, n_hours: int = 24, signal_hour: int = 12) -> pd.DataFrame:
    """
    Simplified data preparation following original data_utils.py logic.
    
    Returns DataFrame with target returns + features, ready for XGB training.
    No scaling applied - uses raw pct_change features as per original.
    """
    logger.info(f"Preparing data for {target_symbol}, period {start_date} to {end_date}")
    
    if symbols is None:
        symbols = get_default_symbols()
    
    if target_symbol not in symbols:
        symbols = [target_symbol] + symbols
    
    # Load data
    try:
        futures_lib = get_arcticdb_connection()
        raw_data = {}
        for symbol in symbols[:10]:  # Limit to first 10 symbols for testing
            try:
                versioned_item = futures_lib.read(symbol)
                df = versioned_item.data  # Extract DataFrame from VersionedItem
                if start_date:
                    df = df[df.index >= pd.Timestamp(start_date)]
                if end_date:
                    df = df[df.index <= pd.Timestamp(end_date)]
                if len(df) > 100:  # Only include symbols with sufficient data
                    raw_data[symbol] = df
            except Exception as e:
                logger.warning(f"Could not load {symbol}: {e}")
        
        logger.info(f"Loaded data for {len(raw_data)} symbols")
        
        if not raw_data or target_symbol not in raw_data:
            raise ValueError(f"Could not load data for target symbol {target_symbol}")
        
        # Build features
        feature_df = build_features_simple(raw_data, target_symbol, signal_hour)
        
        # Build target
        target_returns = prepare_target_returns(raw_data, target_symbol, n_hours, signal_hour)
        target_col = f"{target_symbol}_target_return"
        
        # Combine
        df = feature_df.copy()
        df[target_col] = target_returns.reindex(df.index)
        
        # Clean
        df = clean_data_simple(df)
        df = df.dropna(subset=[target_col])
        
        logger.info(f"Final dataset: {df.shape}, target mean={df[target_col].mean():.6f}, target std={df[target_col].std():.6f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        raise
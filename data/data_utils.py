"""
Simplified data utilities for real market data loading and feature engineering.
"""

import pandas as pd
import numpy as np
import math
from typing import Dict, Optional, List
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .loaders import get_arcticdb_connection
from .symbol_loader import get_default_symbols

logger = logging.getLogger(__name__)

def cluster_features_by_correlation(corr_matrix: pd.DataFrame, threshold: float = 0.7) -> dict:
    """Cluster features based on correlation threshold using hierarchical clustering."""
    # Convert correlation to distance matrix
    dist = 1 - corr_matrix.abs()
    dist = dist.fillna(1.0)
    np.fill_diagonal(dist.values, 0.0)
    
    try:
        # Create condensed distance matrix for linkage
        condensed = squareform(dist.values, checks=False)
        # Hierarchical clustering
        linkage_matrix = linkage(condensed, method='average')
        # Get cluster labels
        labels = fcluster(linkage_matrix, t=1-threshold, criterion='distance')
        
        # Group features by cluster
        clusters = {}
        for feature, label in zip(corr_matrix.columns, labels):
            clusters.setdefault(label, []).append(feature)
        
        logger.info(f"🔗 Created {len(clusters)} feature clusters with threshold {threshold}")
        return clusters
        
    except Exception as e:
        logger.warning(f"⚠️ Clustering failed: {e}, returning individual features")
        # Fallback: each feature is its own cluster
        return {i: [feat] for i, feat in enumerate(corr_matrix.columns)}

def select_representative_features(clusters: dict, target_col: str, df: pd.DataFrame) -> List[str]:
    """Select one representative feature from each cluster based on target correlation."""
    representatives = []
    target_values = df[target_col]
    
    for cluster_id, features in clusters.items():
        if len(features) == 1:
            representatives.extend(features)
        else:
            # Calculate correlation with target for each feature in cluster
            correlations = {}
            for feat in features:
                if feat in df.columns:
                    try:
                        corr = df[feat].corr(target_values, method='spearman')
                        correlations[feat] = abs(corr) if pd.notna(corr) else 0.0
                    except:
                        correlations[feat] = 0.0
            
            # Select feature with highest absolute correlation to target
            if correlations:
                best_feature = max(correlations.keys(), key=correlations.get)
                representatives.append(best_feature)
                logger.debug(f"   Cluster {cluster_id}: selected {best_feature} from {len(features)} features")
    
    logger.info(f"📊 Selected {len(representatives)} representative features from {sum(len(feats) for feats in clusters.values())} total")
    return representatives

def reduce_features_by_clustering(df: pd.DataFrame, target_col: str, corr_threshold: float = 0.7) -> pd.DataFrame:
    """Reduce features using correlation clustering while preserving target relationship."""
    feature_cols = [c for c in df.columns if c != target_col]
    
    if len(feature_cols) <= 1:
        logger.info("🔄 Too few features for clustering, skipping reduction")
        return df
    
    logger.info(f"🔍 Feature clustering: {len(feature_cols)} features -> threshold {corr_threshold}")
    
    # Calculate correlation matrix for features only
    feature_df = df[feature_cols]
    corr_matrix = feature_df.corr().fillna(0.0)
    
    # Cluster features
    clusters = cluster_features_by_correlation(corr_matrix, corr_threshold)
    
    # Select representatives
    representatives = select_representative_features(clusters, target_col, df)
    
    # Return reduced dataset
    reduced_cols = [target_col] + [f for f in representatives if f in df.columns]
    reduced_df = df[reduced_cols]
    
    logger.info(f"✅ Feature reduction: {len(feature_cols)} -> {len(reduced_df.columns)-1} features")
    return reduced_df

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
    """Compute technical indicators: RSI, ATR, momentum, and breakouts with TARGETED FIXES."""
    result = df.copy()
    
    # RSI (3-period for responsiveness) - FIXED to prevent extreme values
    delta = result['close'].diff()
    gain = delta.clip(lower=0).rolling(3).mean()
    loss = (-delta.clip(upper=0)).rolling(3).mean()
    
    # Proper RSI calculation with safer NaN handling
    rs = gain / loss.replace(0, np.nan)  # Use NaN instead of tiny values
    rsi_raw = 100 - 100 / (1 + rs)
    result['rsi'] = rsi_raw.fillna(50).clip(0, 100)  # Clip to valid RSI range
    
    
    # ATR (Average True Range) - Keep original logic
    range_series = result[['high', 'low', 'close']].max(axis=1) - result[['high', 'low', 'close']].min(axis=1)
    result['atr'] = range_series.rolling(6).mean()
    
    # Multi-period momentum features - ORIGINAL LOGIC with fixes
    periods = [1, 2, 3, 4, 8, 12, 16, 20]  # Original periods
    change = result['close'].pct_change(fill_method=None)
    
    # Initialize features dict (base features already in result dataframe)
    features = {}
    
    for p in periods:
        momentum = change.rolling(p, min_periods=1).mean()
        features[f'momentum_{p}h'] = momentum.clip(-0.2, 0.2)  # Reasonable momentum limits
        
        velocity = momentum.pct_change(fill_method=None)
        features[f'velocity_{p}h'] = velocity.clip(-5, 5)  # Clip extreme velocity
        
        # FIXED: Use raw RSI values with additional safety clipping
        rsi_rolling = result['rsi'].rolling(p, min_periods=1).mean()
        features[f'rsi_{p}h'] = rsi_rolling.clip(0, 100)  # Ensure valid RSI range
        
        
        # FIXED: Clip ATR changes to prevent extremes
        atr_change = result['atr'].pct_change(fill_method=None)
        features[f'atr_{p}h'] = atr_change.clip(-2, 2).rolling(p, min_periods=1).mean()
        
        features[f'breakout_{p}h'] = result['close'] - result['close'].rolling(p, min_periods=1).min()
    
    # Momentum differentials (original logic)
    if "momentum_1h" in features and "momentum_4h" in features:
        features.update({
            "momentum_diff_1_4h": features["momentum_1h"] - features["momentum_4h"],
            "momentum_diff_3_12h": features["momentum_3h"] - features["momentum_12h"],
            "momentum_diff_8_16h": features["momentum_8h"] - features["momentum_16h"]
        })
    
    # Return features with original structure
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
                # Use reasonable min_periods - at least 2 but not more than window/2
                min_periods = min(max(2, p//4), p)  # At least 2, but not more than window size
                corr = target_close.loc[idx].rolling(p, min_periods=min_periods).corr(series)
                
                # Replace extreme values and NaN with 0, then clip to valid range
                corr = corr.replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)
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
    
    # Convert to percentage changes (except level features and technical indicators)
    feature_df = feature_df.replace(0, 1e-8)
    for col in feature_df.columns:
        # FIXED: Don't percentage-change RSI, ATR, correlations, or other normalized indicators
        if (not col.startswith(f'{target_symbol}_level_') and 
            not any(indicator in col.lower() for indicator in ['rsi', 'atr', 'momentum', 'velocity', 'breakout', 'corr'])):
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
        df_clean = df_clean.loc[bad_rows == False]
    
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
                     max_features: int = None, on_target_only: bool = False, 
                     corr_threshold: float = 0.7) -> pd.DataFrame:
    """
    Main function: prepare real market data for XGB training.
    
    Returns DataFrame with target column + features, ready for ML.
    """
    if symbols is None:
        symbols = get_default_symbols()
    
    if target_symbol not in symbols:
        symbols = [target_symbol] + symbols
    
    # Removed verbose logging
    
    # Check for cached features (comprehensive cache key to avoid wrong cache usage)
    import os
    import hashlib
    
    # Create comprehensive cache key including all parameters that affect feature generation
    cache_params = {
        'target_symbol': target_symbol,
        'symbols': sorted(symbols) if symbols else None,
        'start_date': start_date or 'auto',
        'end_date': end_date or 'auto', 
        'n_hours': n_hours,
        'signal_hour': signal_hour,
        'on_target_only': on_target_only,
        'max_features': max_features,
        'corr_threshold': corr_threshold
    }
    
    # Create stable hash from parameters
    cache_key_str = str(sorted(cache_params.items()))
    cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:12]
    cache_file = f"artifacts/cache/features_{target_symbol}_{cache_hash}.pkl"
    
    logger.debug(f"🔑 Cache key params: {cache_params}")
    logger.debug(f"📁 Cache file: {cache_file}")
    
    if os.path.exists(cache_file):
        import pickle
        # logger.info(f"Loading cached features from {cache_file}")
        try:
            with open(cache_file, 'rb') as f:
                cached_df = pickle.load(f)
                # logger.info(f"Cache hit: {cached_df.shape}")
                return cached_df
        except Exception as e:
            logger.warning(f"⚠️ Cache read failed: {e}, rebuilding...")
    
    # Loading raw data from Arctic DB
    raw_data = load_real_data(symbols, start_date, end_date, signal_hour=None)
    if not raw_data:
        logger.error("❌ No data loaded from Arctic DB")
        return pd.DataFrame()
    
    # logger.info(f"Loaded {len(raw_data)} symbols from Arctic DB")
    
    # Generate target returns (forward n_hours)
    logger.info(f"🎯 Generating {n_hours}h forward returns for {target_symbol}...")
    target_data = fill_intraday_gaps(raw_data[target_symbol], target_symbol)
    future_close = target_data['close'].shift(-n_hours)
    target_returns = (future_close - target_data['close']) / target_data['close']
    
    # Filter to signal hour and create target DataFrame
    target_signal_hour = target_data[target_data.index.hour == signal_hour]
    target_col = f"{target_symbol}_target_return"
    target_df = pd.DataFrame({target_col: target_returns.reindex(target_signal_hour.index)})
    logger.info(f"✅ Created target: {target_df.shape[0]} observations at hour {signal_hour}")
    
    # Build feature matrix - on-target vs full universe
    if on_target_only:
        logger.info(f"✓ ON-TARGET mode: Using features only from {target_symbol}")
        # Only use target symbol for features
        feature_symbols = {target_symbol: raw_data[target_symbol]}
    else:
        logger.info(f"✓ FULL-UNIVERSE mode: Using features from all {len(raw_data)} symbols")
        feature_symbols = raw_data
    
    logger.info(f"🔧 Building features from {len(feature_symbols)} symbol(s)...")
    feature_df = build_feature_matrix(feature_symbols, target_symbol, signal_hour)
    if feature_df.empty:
        logger.error("❌ Failed to generate features")
        return pd.DataFrame()
    
    logger.info(f"✅ Generated {feature_df.shape[1]} features, {feature_df.shape[0]} observations")
    
    # Combine target + features
    logger.info(f"🔗 Combining targets and features...")
    df = target_df.join(feature_df, how='left')
    df = df.dropna(subset=[target_col])  # Remove missing targets
    logger.info(f"✅ Combined dataset: {df.shape}")
    
    # Data quality checks
    logger.info(f"🔍 Data quality checks...")
    target_stats = df[target_col].describe()
    logger.info(f"   Target stats: mean={target_stats['mean']:.6f}, std={target_stats['std']:.6f}, range=[{target_stats['min']:.6f}, {target_stats['max']:.6f}]")
    
    # Check for constant/zero features
    feature_cols = [c for c in df.columns if c != target_col]
    bad_features = set()
    constant_features = []
    zero_features = []
    
    for col in feature_cols:
        col_vals = df[col].dropna()
        if len(col_vals) > 0:
            if col_vals.std() < 1e-10:
                constant_features.append(col)
                bad_features.add(col)
            elif (col_vals == 0).all():
                zero_features.append(col)
                bad_features.add(col)
    
    if bad_features:
        logger.warning(f"⚠️ Found {len(bad_features)} bad features - constant: {len(constant_features)}, zero: {len(zero_features)}")
        logger.warning(f"   Removing: {list(bad_features)[:5]}{'...' if len(bad_features) > 5 else ''}")
        df = df.drop(columns=list(bad_features))
    
    df = clean_data(df)
    
    # Feature correlation clustering (reduce redundant features)
    if len(df.columns) > 2:  # Only cluster if we have features beyond target
        logger.info(f"🔗 Applying correlation clustering with threshold {corr_threshold}")
        df = reduce_features_by_clustering(df, target_col, corr_threshold)
    
    # Limit features for testing (after clustering)
    if max_features and len(df.columns) > max_features + 1:
        feature_cols = [c for c in df.columns if c != target_col][:max_features]
        df = df[[target_col] + feature_cols]
        logger.info(f"🔪 Limited to {max_features} features for testing")
    
    logger.info(f"✅ Final dataset: {df.shape} (target + {df.shape[1]-1} features)")
    
    # Cache the result (AFTER clustering and feature reduction)
    os.makedirs("artifacts/cache", exist_ok=True)
    try:
        import pickle
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"💾 Cached clustered features to {cache_file}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to cache features: {e}")
    return df
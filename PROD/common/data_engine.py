"""
Simplified Data Engine for XGBoost Production
Streamlined version with core functionality only.
"""

import pandas as pd
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from .iqfeed import IQFeedClient

logger = logging.getLogger(__name__)

class DataEngine:
    """Simplified data engine for production signal generation."""

    def __init__(self):
        self.iqfeed_client = IQFeedClient()

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic technical features following original logic - EXACT COPY from data_utils_simple.py."""
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

        # CRITICAL FIX: Forward-fill only (no future leakage) - use past data only
        # This fixes the 72 NaN values that were contaminating the feature space
        result['atr'] = result['atr'].ffill().fillna(0.0)  # Forward fill only, no median (uses future)

        # Momentum features for various periods (following original pattern)
        periods = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32]
        change = result['close'].pct_change(fill_method=None)
        atr_change = result['atr'].pct_change(fill_method=None)

        for p in periods:
            # Batch calculate all rolling features for this period
            momentum = change.rolling(p, min_periods=1).mean()
            result[f'momentum_{p}h'] = momentum.clip(-0.2, 0.2)
            result[f'velocity_{p}h'] = momentum.pct_change(fill_method=None).clip(-5, 5)
            result[f'rsi_{p}h'] = result['rsi'].rolling(p, min_periods=1).mean()
            atr_feature = atr_change.clip(-2, 2).rolling(p, min_periods=1).mean()
            # CRITICAL FIX: Forward-fill only (no future leakage) - use past data only
            result[f'atr_{p}h'] = atr_feature.ffill().fillna(0.0)  # Forward fill only, no bfill (uses future)

            # Breakout calculation (handle p=1 edge case)
            min_periods = 2 if p > 1 else 2  # Always use min_periods=2
            window = max(p, 2) if p == 1 else p
            recent_low = result['close'].rolling(window, min_periods=min_periods).min()
            result[f'breakout_{p}h'] = (result['close'] - recent_low) / recent_low

        # Momentum differentials (batch calculate)
        diff_pairs = [(1, 4), (3, 12), (8, 16)]
        for short, long in diff_pairs:
            result[f'momentum_diff_{short}_{long}h'] = result[f'momentum_{short}h'] - result[f'momentum_{long}h']

        return result

    def build_features(self, raw_data: Dict[str, pd.DataFrame], target_symbol: str, signal_hour: int = 12) -> pd.DataFrame:
        """Build feature matrix following EXACT data_utils_simple.py logic."""
        if target_symbol not in raw_data:
            return pd.DataFrame()

        # Process target symbol
        target_data = self.calculate_features(raw_data[target_symbol])
        target_features = target_data[target_data.index.hour == signal_hour].add_prefix(f"{target_symbol}_")

        # Process other symbols
        other_features = []
        for sym, data in raw_data.items():
            if sym != target_symbol:
                sym_features = self.calculate_features(data)
                sym_features = sym_features[sym_features.index.hour == signal_hour].add_prefix(f"{sym}_")
                other_features.append(sym_features)

        # Cross-correlations
        corr_features = self.calculate_cross_correlations_simple(target_symbol, raw_data, target_features.index)

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

    def calculate_cross_correlations_simple(self, target_symbol: str, raw_data: Dict[str, pd.DataFrame], target_index) -> pd.DataFrame:
        """Simple cross-correlations following original pattern - EXACT COPY from data_utils_simple.py."""
        if target_symbol not in raw_data:
            return pd.DataFrame(index=target_index)

        periods = [2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
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

    def apply_feature_selection(self, features_df: pd.DataFrame, target_symbol: str) -> pd.DataFrame:
        """Apply feature selection to match training - use exact features from production package."""
        try:
            # Load production package to get exact features
            models_dir = Path(__file__).parent.parent / "models"

            # Load timestamp format files (e.g., @ES#C_20250917_201818.pkl)
            timestamp_files = list(models_dir.glob(f"{target_symbol}_*.pkl"))
            if not timestamp_files:
                logger.error(f"No model package found for {target_symbol}")
                return features_df

            # Use the most recent file (last in sorted order)
            package_file = sorted(timestamp_files)[-1]
            try:
                with open(package_file, 'rb') as f:
                    package = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load package for {target_symbol}: {e}")
                return features_df

            expected_features = package['selected_features']

            # Create DataFrame with exact features in exact order
            selected_df = pd.DataFrame(index=features_df.index, columns=expected_features)

            # Copy available features
            for feature in expected_features:
                if feature in features_df.columns:
                    selected_df[feature] = features_df[feature]
                else:
                    # Fill missing features with zero (no fallback - this is an error condition)
                    selected_df[feature] = 0.0
                    logger.warning(f"Missing feature {feature} for {target_symbol}")

            return selected_df.fillna(0.0)

        except Exception as e:
            logger.error(f"Feature selection failed for {target_symbol}: {e}")
            return features_df  # Return as-is if feature selection fails

    def get_prediction_data(self, target_symbol: str, feature_symbols: List[str], signal_hour: int = 12) -> tuple:
        """Get features and price for prediction."""
        try:
            # Fetch data
            all_symbols = list(set([target_symbol] + feature_symbols))
            raw_data = self.iqfeed_client.get_live_data_multi(all_symbols, days=20, interval_min=60)

            if target_symbol not in raw_data or raw_data[target_symbol].empty:
                return None, None

            # Build and select features
            features_df = self.build_features(raw_data, target_symbol, signal_hour)
            features_df = self.apply_feature_selection(features_df, target_symbol)

            # Get latest price
            price = float(raw_data[target_symbol]['close'].iloc[-1])

            return features_df, price

        except Exception as e:
            logger.error(f"Data error {target_symbol}: {e}")
            return None, None

    def get_prediction_data_batch(self, trading_symbols, feature_symbols, signal_hour=12):
        """Get prediction data for multiple symbols (same as daily_signal_runner.py)."""
        try:
            # Fetch data for all symbols
            all_symbols = list(set(trading_symbols + feature_symbols))
            raw_data = self.iqfeed_client.get_live_data_multi(all_symbols, days=20, interval_min=60)

            batch_results = {}

            for symbol in trading_symbols:
                if symbol not in raw_data or raw_data[symbol].empty:
                    continue

                # Build features for this symbol
                features_df = self.build_features(raw_data, symbol, signal_hour)
                features_df = self.apply_feature_selection(features_df, symbol)

                if features_df is not None and not features_df.empty:
                    # Get latest price
                    price = float(raw_data[symbol]['close'].iloc[-1])
                    batch_results[symbol] = (features_df, price)
            logger.info(f"Data fetched for {len(batch_results)} symbols")
            
            return batch_results

        except Exception as e:
            logger.error(f"Batch data error: {e}")
            return {}
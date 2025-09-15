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
        """Calculate features exactly matching research stack logic."""
        result = df.copy()

        # Basic RSI (3-period) - exact research stack logic
        delta = result['close'].diff()
        gain = delta.clip(lower=0).rolling(3).mean()
        loss = (-delta.clip(upper=0)).rolling(3).mean()
        rs = gain / loss.replace(0, np.nan)
        result['rsi'] = (100 - 100 / (1 + rs)).fillna(50)

        # Basic ATR as percentage - exact research stack logic
        high_low = result['high'] - result['low']
        atr_absolute = high_low.rolling(6).mean()
        result['atr'] = atr_absolute / result['close']
        result['atr'] = result['atr'].ffill().fillna(0.0)

        # Momentum features - exact research stack periods and logic
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
            result[f'atr_{p}h'] = atr_feature.ffill().fillna(0.0)

            # Breakout calculation - exact research stack logic
            min_periods = 2 if p > 1 else 2
            window = max(p, 2) if p == 1 else p
            recent_low = result['close'].rolling(window, min_periods=min_periods).min()
            result[f'breakout_{p}h'] = (result['close'] - recent_low) / recent_low

        # Momentum differentials - exact research stack pairs
        diff_pairs = [(1, 4), (3, 12), (8, 16)]
        for short, long in diff_pairs:
            result[f'momentum_diff_{short}_{long}h'] = result[f'momentum_{short}h'] - result[f'momentum_{long}h']

        return result

    def build_features(self, raw_data: Dict[str, pd.DataFrame], target_symbol: str, signal_hour: int = 12) -> pd.DataFrame:
        """Build feature matrix."""
        if target_symbol not in raw_data:
            return pd.DataFrame()

        # Process target symbol
        target_data = self.calculate_features(raw_data[target_symbol])
        target_features = target_data[target_data.index.hour == signal_hour].add_prefix(f"{target_symbol}_")

        # Cross-correlations - exact research stack logic
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

        # Add correlations to target features
        if corr_features:
            corr_df = pd.DataFrame(corr_features)
            corr_aligned = corr_df.reindex(target_features.index, fill_value=0.0)
            target_features = pd.concat([target_features, corr_aligned], axis=1)

        # Clean and return
        return target_features.ffill().fillna(0.0).replace([np.inf, -np.inf], 0.0)

    def apply_feature_selection(self, features_df: pd.DataFrame, target_symbol: str) -> pd.DataFrame:
        """Apply feature selection to match training."""
        models_dir = Path(__file__).parent.parent / "models" / target_symbol
        metadata_file = models_dir / "model_metadata.yaml"

        if not metadata_file.exists():
            # Fallback: select top variance features
            if features_df.shape[1] > 50:
                variances = features_df.var().sort_values(ascending=False)
                features_df = features_df[variances.head(50).index]
            return features_df

        # Extract feature names from metadata
        try:
            with open(metadata_file, 'r') as f:
                content = f.read()

            feature_names = []
            in_features = False
            for line in content.split('\n'):
                if 'feature_names:' in line:
                    in_features = True
                elif in_features and line.startswith('- '):
                    feature_names.append(line.replace('- ', '').strip())
                elif in_features and not line.startswith(' ') and not line.startswith('-'):
                    break

            if feature_names:
                # Create exact feature set
                selected_df = pd.DataFrame(index=features_df.index, columns=feature_names)
                for feature in feature_names:
                    if feature in features_df.columns:
                        selected_df[feature] = features_df[feature]
                return selected_df.fillna(0.0)

        except Exception as e:
            logger.warning(f"Feature selection fallback for {target_symbol}: {e}")

        # Fallback
        if features_df.shape[1] > 50:
            variances = features_df.var().sort_values(ascending=False)
            features_df = features_df[variances.head(50).index]
        return features_df

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
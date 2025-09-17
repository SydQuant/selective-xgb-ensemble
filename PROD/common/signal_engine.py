"""XGBoost Production Signal Engine - Streamlined and Optimized"""

import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class SignalEngine:
    """XGBoost ensemble signal generation for production."""

    def __init__(self, models_dir: Path, config_dir: Path):
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.loaded_models = {}
        self.symbol_configs = {}

    def load_symbol_package(self, symbol: str) -> Optional[Dict]:
        """Load consolidated symbol package (models + config + features)."""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        # Load timestamp format files (e.g., @ES#C_20250917_201818.pkl)
        timestamp_files = list(self.models_dir.glob(f"{symbol}_*.pkl"))
        if not timestamp_files:
            logger.error(f"No model package found for {symbol}")
            return None

        # Use the most recent file (last in sorted order)
        consolidated_file = sorted(timestamp_files)[-1]
        try:
            with open(consolidated_file, 'rb') as f:
                package = pickle.load(f)

            self.loaded_models[symbol] = package
            self.symbol_configs[symbol] = package.get('metadata', {})
            return package
        except Exception as e:
            logger.error(f"Failed to load package for {symbol}: {e}")
            return None


    def generate_signal(self, features_df: pd.DataFrame, symbol: str) -> Optional[Tuple[int, float]]:
        """Generate signal using XGBoost ensemble democratic voting."""
        package = self.load_symbol_package(symbol)
        if not package:
            return None

        try:
            models = package.get('models', {})
            if not models:
                return None

            latest_features = features_df.iloc[-1:].copy()
            expected_features = package['selected_features']

            # Validate feature alignment
            if len(expected_features) != latest_features.shape[1]:
                logger.error(f"Feature count mismatch for {symbol}: expected {len(expected_features)}, got {latest_features.shape[1]}")
                return None

            if not all(feature in latest_features.columns for feature in expected_features):
                logger.error(f"Feature name mismatch for {symbol}")
                return None

            # Generate predictions from ensemble
            predictions = []
            for model_key, model in models.items():
                try:
                    model_data = latest_features[expected_features]
                    pred = model.predict(model_data.values)[0]
                    predictions.append(pred)
                except Exception as e:
                    logger.error(f"Model {model_key} prediction failed: {e}")
                    continue

            if not predictions:
                return None

            # Binary democratic voting
            binary_votes = [1 if pred > 0 else -1 if pred < 0 else 0 for pred in predictions]
            vote_sum = sum(binary_votes)
            signal = int(np.sign(vote_sum)) if vote_sum != 0 else 0
            raw_score = float(np.mean(predictions))

            return signal, raw_score

        except Exception as e:
            logger.error(f"Signal generation failed for {symbol}: {e}")
            return None

    def generate_signals_batch(self, features_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Generate signals for multiple symbols."""
        results = {}
        for symbol, features_df in features_data.items():
            if features_df.empty:
                continue

            result = self.generate_signal(features_df, symbol)
            if result:
                signal, raw_score = result
                results[symbol] = {'signal': signal, 'raw_score': raw_score}
                logger.info(f"{symbol}: Signal={signal}, Score={raw_score:.4f}")

        return results
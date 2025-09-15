"""
Simplified Signal Engine for XGBoost Production
Core XGBoost prediction functionality without redundant code.
"""

import pickle
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)

class SignalEngine:
    """Simplified XGBoost signal generation."""

    def __init__(self, models_dir: Path, config_dir: Path):
        self.models_dir = Path(models_dir)
        self.config_dir = Path(config_dir)
        self.loaded_models = {}
        self.symbol_configs = {}

    def load_symbol_config(self, symbol: str) -> Optional[Dict]:
        """Load symbol configuration."""
        if symbol in self.symbol_configs:
            return self.symbol_configs[symbol]

        config_file = self.config_dir / "models" / f"{symbol}.yaml"
        if not config_file.exists():
            return None

        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            self.symbol_configs[symbol] = config
            return config
        except Exception:
            return None

    def load_models(self, symbol: str) -> Optional[List]:
        """Load XGBoost models for symbol."""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        symbol_models_dir = self.models_dir / symbol
        if not symbol_models_dir.exists():
            return None

        models = []
        for model_file in sorted(symbol_models_dir.glob("model_*.pkl")):
            try:
                with open(model_file, 'rb') as f:
                    models.append(pickle.load(f))
            except Exception:
                continue

        if models:
            self.loaded_models[symbol] = models
            return models
        return None

    def generate_signal(self, features_df: pd.DataFrame, symbol: str) -> Optional[Tuple[int, float]]:
        """Generate signal using XGBoost ensemble."""
        models = self.load_models(symbol)
        config = self.load_symbol_config(symbol)

        if not models or not config:
            return None

        try:
            latest_features = features_df.iloc[-1:].copy()
            predictions = []

            for model in models:
                try:
                    pred = model.predict(latest_features.values)[0]
                    predictions.append(pred)
                except Exception:
                    continue

            if not predictions:
                return None

            # Normalize predictions exactly like research stack: z-score then tanh/binary
            pred_series = pd.Series(predictions)

            # Z-score normalization first
            if pred_series.std() == 0:
                z_scores = pd.Series(np.zeros_like(predictions))
            else:
                z_scores = (pred_series - pred_series.mean()) / pred_series.std()

            binary_signal = config.get('binary_signal', True)

            if binary_signal:
                # Binary: +1 for positive, -1 for negative z-scores
                normalized_preds = np.where(z_scores > 0, 1.0, np.where(z_scores < 0, -1.0, 0.0))
                # Binary combination: simple sum of votes (research stack logic)
                combined_signal = normalized_preds.sum()
            else:
                # Continuous: tanh of z-scores
                normalized_preds = np.tanh(z_scores)
                # Tanh combination: equal-weighted averaging (research stack logic)
                combined_signal = normalized_preds.mean()

            signal = int(np.sign(combined_signal))
            if signal == 0:
                signal = 1

            raw_score = float(pred_series.mean())
            return signal, raw_score

        except Exception as e:
            logger.error(f"Signal error {symbol}: {e}")
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
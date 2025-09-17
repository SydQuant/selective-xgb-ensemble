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

    def load_symbol_package(self, symbol: str) -> Optional[Dict]:
        """Load consolidated symbol package (models + config + features)."""
        if symbol in self.loaded_models:
            return self.loaded_models[symbol]

        # Try new timestamp format first (e.g., @ES#C_20250917_201818.pkl)
        timestamp_files = list(self.models_dir.glob(f"{symbol}_*.pkl"))
        if timestamp_files:
            # Use the most recent file (last in sorted order)
            consolidated_file = sorted(timestamp_files)[-1]
            try:
                with open(consolidated_file, 'rb') as f:
                    package = pickle.load(f)

                self.loaded_models[symbol] = package
                self.symbol_configs[symbol] = package.get('metadata', {})
                logger.info(f"Loaded timestamp package for {symbol}: {consolidated_file.name}, {len(package.get('models', {}))} models")
                return package
            except Exception as e:
                logger.error(f"Failed to load timestamp package for {symbol}: {e}")

        # Try legacy production format
        consolidated_file = self.models_dir / f"{symbol}_production.pkl"
        if consolidated_file.exists():
            try:
                with open(consolidated_file, 'rb') as f:
                    package = pickle.load(f)

                self.loaded_models[symbol] = package
                self.symbol_configs[symbol] = package.get('metadata', {})
                logger.info(f"Loaded legacy production package for {symbol}: {len(package.get('models', {}))} models")
                return package
            except Exception as e:
                logger.error(f"Failed to load legacy production package for {symbol}: {e}")

        # Fallback to old format (separate files)
        config_file = self.config_dir / "models" / f"{symbol}.yaml"
        symbol_models_dir = self.models_dir / symbol

        if config_file.exists() and symbol_models_dir.exists():
            try:
                # Load config
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)

                # Load individual model files
                models = {}
                for model_file in sorted(symbol_models_dir.glob("model_*.pkl")):
                    model_name = model_file.stem
                    with open(model_file, 'rb') as f:
                        models[model_name] = pickle.load(f)

                # Create package format
                package = {
                    'symbol': symbol,
                    'models': models,
                    'selected_features': config.get('selected_features', []),
                    'binary_signal': config.get('binary_signal', False),
                    'metadata': config
                }

                self.loaded_models[symbol] = package
                self.symbol_configs[symbol] = config
                logger.info(f"Loaded legacy package for {symbol}: {len(models)} models")
                return package

            except Exception as e:
                logger.error(f"Failed to load legacy package for {symbol}: {e}")

        return None

    def load_symbol_config(self, symbol: str) -> Optional[Dict]:
        """Load symbol configuration (legacy compatibility)."""
        package = self.load_symbol_package(symbol)
        return package.get('metadata', {}) if package else None

    def load_models(self, symbol: str) -> Optional[List]:
        """Load XGBoost models for symbol (legacy compatibility)."""
        package = self.load_symbol_package(symbol)
        if package and 'models' in package:
            return list(package['models'].values())
        return None

    def generate_signal(self, features_df: pd.DataFrame, symbol: str) -> Optional[Tuple[int, float]]:
        """Generate signal using XGBoost ensemble with model-specific feature slices."""
        package = self.load_symbol_package(symbol)

        if not package:
            return None

        try:
            models = package.get('models', {})

            if not models:
                return None

            latest_features = features_df.iloc[-1:].copy()
            predictions = []

            for model_key, model in models.items():
                try:
                    # Use exact features from production package (latest format only)
                    expected_features = package['selected_features']

                    # Ensure exact feature match
                    if len(expected_features) != latest_features.shape[1]:
                        logger.error(f"Feature count mismatch for {symbol}: expected {len(expected_features)}, got {latest_features.shape[1]}")
                        continue

                    if not all(feature in latest_features.columns for feature in expected_features):
                        logger.error(f"Feature name mismatch for {symbol}")
                        continue

                    # Use exact feature order
                    model_data = latest_features[expected_features]
                    pred = model.predict(model_data.values)[0]
                    predictions.append(pred)

                except Exception as e:
                    logger.error(f"Model {model_key} prediction failed: {e}")
                    continue

            if not predictions:
                return None

            # SIMPLIFIED BINARY VOTING: Always use binary regardless of config
            # Convert each prediction to +1/-1 based on sign, then sum and take sign
            binary_votes = []
            for pred in predictions:
                if pred > 0:
                    binary_votes.append(1)
                elif pred < 0:
                    binary_votes.append(-1)
                else:
                    binary_votes.append(0)

            # Sum all votes and take sign for final signal
            vote_sum = sum(binary_votes)
            signal = int(np.sign(vote_sum)) if vote_sum != 0 else 0

            raw_score = float(np.mean(predictions))
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
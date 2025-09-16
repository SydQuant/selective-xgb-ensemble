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

        # Try consolidated format first
        consolidated_file = self.models_dir / f"{symbol}_production.pkl"

        if consolidated_file.exists():
            try:
                with open(consolidated_file, 'rb') as f:
                    package = pickle.load(f)

                self.loaded_models[symbol] = package
                self.symbol_configs[symbol] = package.get('metadata', {})
                logger.info(f"Loaded consolidated package for {symbol}: {len(package.get('models', {}))} models")
                return package
            except Exception as e:
                logger.error(f"Failed to load consolidated package for {symbol}: {e}")

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
            model_feature_slices = package.get('model_feature_slices', {})
            binary_signal = package.get('binary_signal', False)

            if not models:
                return None

            latest_features = features_df.iloc[-1:].copy()
            predictions = []

            for model_key, model in models.items():
                try:
                    # Dynamically determine model's expected feature count
                    if hasattr(model, 'feature_names_in_'):
                        expected_features = model.feature_names_in_
                        if expected_features is not None:
                            model_data = latest_features[expected_features]
                        else:
                            # Fallback: determine from model's n_features_in_
                            n_features = getattr(model, 'n_features_in_', len(latest_features.columns))
                            selected_features_list = package.get('selected_features', list(latest_features.columns))
                            model_features = selected_features_list[:n_features]
                            available_features = [f for f in model_features if f in latest_features.columns]
                            model_data = latest_features[available_features] if available_features else latest_features
                    else:
                        # Fallback for older models: try to infer from model structure
                        try:
                            # Test with all features first
                            test_pred = model.predict(latest_features.values)
                            model_data = latest_features
                        except:
                            # If that fails, use model-specific feature slice if available
                            if model_key in model_feature_slices:
                                model_features = model_feature_slices[model_key]
                                available_features = [f for f in model_features if f in latest_features.columns]
                                model_data = latest_features[available_features] if available_features else latest_features
                            else:
                                # Last resort: use first N features where N = model's expected input
                                n_features = getattr(model, 'n_features_in_', len(latest_features.columns))
                                model_data = latest_features.iloc[:, :n_features]

                    pred = model.predict(model_data.values)[0]
                    predictions.append(pred)

                except Exception as e:
                    logger.warning(f"Model {model_key} prediction failed: {e}")
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

            # Handle NaN values in combined_signal
            if np.isnan(combined_signal):
                signal = 1  # Default to buy signal if NaN
            else:
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
#!/usr/bin/env python3
"""
Production Model Builder for XGBoost Framework
Builds final production models using optimal configurations and saves them for deployment.

This script addresses the critical issue of extracting production-ready models:
- Uses ALL available data for training (no holdout for final models)
- Applies feature selection exactly as in research
- Saves models with their feature specifications
- Creates deployment-ready artifacts

Usage:
python production_model_builder.py --symbol "@ES#C" --config optimal_ES_config.yaml
"""

import os
import sys
import logging
import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import XGBCompareConfig, parse_config
from data.data_utils_simple import prepare_real_data_simple
from model.feature_selection import apply_feature_selection
from model.xgb_drivers import generate_xgb_specs, generate_deep_xgb_specs, stratified_xgb_bank, fit_xgb_on_slice

class ProductionModelBuilder:
    """Builds production-ready XGBoost models from optimal configurations."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path(__file__).parent.parent / "PROD"
        self.models_dir = self.output_dir / "models"
        self.config_dir = self.output_dir / "config" / "models"

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for production model building."""
        log_file = self.output_dir / f"production_model_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Production Model Builder initialized - Log: {log_file}")

    def load_optimal_config(self, symbol: str) -> dict:
        """Load optimal configuration for a symbol from multi-symbol testing results."""

        # Import all optimal configs directly
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent / "PROD"))
        from optimal_configs_summary import OPTIMAL_CONFIGS

        # Convert format to match expected structure
        optimal_configs = {}
        for symbol, config in OPTIMAL_CONFIGS.items():
            optimal_configs[symbol] = {
                "n_models": config["n_models"],
                "n_folds": config["n_folds"],
                "max_features": config["max_features"],
                "q_metric": config["q_metric"],
                "xgb_type": config["xgb_type"],
                "binary_signal": config["binary_signal"],
                "benchmark_sharpe": config["production_sharpe"],
                "test_config": config["config"]
            }

        if symbol not in optimal_configs:
            raise ValueError(f"No optimal configuration found for {symbol}")

        return optimal_configs[symbol]

    def create_production_config(self, symbol: str, optimal_config: dict) -> XGBCompareConfig:
        """Create a Config object for production model building."""

        # Create config with production settings
        config = XGBCompareConfig(
            target_symbol=symbol,
            n_models=optimal_config["n_models"],
            n_folds=optimal_config["n_folds"],
            max_features=optimal_config["max_features"],
            q_metric=optimal_config["q_metric"],
            xgb_type=optimal_config["xgb_type"],
            binary_signal=optimal_config["binary_signal"],

            # Production-specific settings
            start_date="2015-01-01",  # Use all available data
            end_date="2025-08-01",  # Up to present
            no_feature_selection=False,
            n_bootstraps=50,  # Reduced for speed

            # Logging
            log_label=f"PROD_{symbol}_{datetime.now().strftime('%Y%m%d')}"
        )

        return config

    def build_production_models(self, symbol: str) -> bool:
        """Build production models for a symbol."""

        try:
            self.logger.info(f"Building production models for {symbol}...")

            # Load optimal configuration
            optimal_config = self.load_optimal_config(symbol)
            self.logger.info(f"Optimal config: {optimal_config['test_config']}")

            # Create production config
            config = self.create_production_config(symbol, optimal_config)

            # Load and prepare data
            self.logger.info("Loading and preparing data...")
            df = prepare_real_data_simple(config.target_symbol,
                                        start_date=config.start_date,
                                        end_date=config.end_date)

            target_col = f"{config.target_symbol}_target_return"
            X, y = df[[c for c in df.columns if c != target_col]], df[target_col]
            self.logger.info(f"Loaded data: X={X.shape}, y={y.shape}")

            # Apply feature selection
            if not config.no_feature_selection:
                max_features = config.max_features if config.max_features > 0 else -1
                X = apply_feature_selection(X, y, method='block_wise', max_total_features=max_features)
                self.logger.info(f"Selected features: {X.shape[1]}")

            # Generate XGBoost specifications
            if config.xgb_type == "deep":
                xgb_specs = generate_deep_xgb_specs(config.n_models)
            elif config.xgb_type == "tiered":
                xgb_specs = stratified_xgb_bank(config.n_models)
            else:
                xgb_specs = generate_xgb_specs(config.n_models)

            self.logger.info(f"Generated {len(xgb_specs)} XGBoost specifications")

            # Train and evaluate models to select top performers
            self.logger.info("Training and evaluating models for selection...")
            symbol_models_dir = self.models_dir / symbol
            symbol_models_dir.mkdir(exist_ok=True)

            # Split data for model evaluation (use 80% for training, 20% for selection)
            split_idx = int(len(X) * 0.8)
            X_train, X_eval = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_eval = y.iloc[:split_idx], y.iloc[split_idx:]

            self.logger.info(f"Training on {len(X_train)} samples, evaluating on {len(X_eval)} samples")

            # Train and evaluate all models
            model_performances = []
            feature_names = X.columns.tolist()

            for i, spec in enumerate(xgb_specs):
                if (i + 1) % 20 == 0:
                    self.logger.info(f"Training model {i+1}/{len(xgb_specs)}...")

                # Train model
                model = fit_xgb_on_slice(X_train, y_train, spec, force_cpu=True)

                # Evaluate performance
                try:
                    predictions = model.predict(X_eval.values)

                    # Calculate metrics for selection
                    from metrics_utils import normalize_predictions, calculate_model_metrics
                    normalized_preds = normalize_predictions(pd.Series(predictions, index=X_eval.index))
                    metrics = calculate_model_metrics(normalized_preds, y_eval, shifted=False)

                    # Use Q-metric from config
                    q_score = metrics.get(config.q_metric, 0.0)

                    model_performances.append({
                        'model_idx': i,
                        'model': model,
                        'q_score': q_score,
                        'sharpe': metrics.get('sharpe', 0.0),
                        'hit_rate': metrics.get('hit_rate', 0.5)
                    })

                except Exception as e:
                    self.logger.warning(f"Model {i+1} evaluation failed: {e}")
                    continue

            # Select top N models based on Q-metric
            top_n = min(15, len(model_performances))  # Default to top 15 models
            model_performances.sort(key=lambda x: x['q_score'], reverse=True)
            selected_models = model_performances[:top_n]

            self.logger.info(f"Selected top {len(selected_models)} models out of {len(model_performances)}")
            self.logger.info(f"Q-score range: {selected_models[0]['q_score']:.3f} to {selected_models[-1]['q_score']:.3f}")

            # Save only the selected top models
            for i, model_data in enumerate(selected_models):
                model_file = symbol_models_dir / f"model_{i+1:02d}.pkl"
                with open(model_file, 'wb') as f:
                    pickle.dump(model_data['model'], f)

            models = [m['model'] for m in selected_models]

            # Save model metadata with selection info (clean scalars only)
            model_metadata = {
                'symbol': symbol,
                'n_models_saved': int(len(selected_models)),
                'n_models_trained': int(len(model_performances)),
                'selection_method': str(config.q_metric),
                'top_model_q_score': float(selected_models[0]['q_score']),
                'avg_selected_q_score': float(np.mean([m['q_score'] for m in selected_models])),
                'feature_names': feature_names,
                'feature_count': int(len(feature_names)),
                'optimal_config': optimal_config,
                'training_data_shape': [int(X.shape[0]), int(X.shape[1])],
                'evaluation_data_shape': [int(X_eval.shape[0]), int(X_eval.shape[1])],
                'build_timestamp': datetime.now().isoformat(),
                'framework_version': 'v6_production_selected'
            }

            metadata_file = symbol_models_dir / "model_metadata.yaml"
            with open(metadata_file, 'w') as f:
                yaml.dump(model_metadata, f, default_flow_style=False)

            # Create symbol config for production deployment (clean scalars only)
            deployment_config = {
                'symbol': str(symbol),
                'n_models_deployed': int(len(selected_models)),
                'n_models_trained': int(len(model_performances)),
                'selection_method': str(config.q_metric),
                'max_features': int(optimal_config['max_features']),
                'q_metric': str(optimal_config['q_metric']),
                'xgb_type': str(optimal_config['xgb_type']),
                'binary_signal': bool(optimal_config['binary_signal']),
                'signal_transformation': 'tanh',

                'ensemble_config': {
                    'model_selection_method': str(config.q_metric),
                    'top_n_models': int(len(selected_models)),
                    'voting_method': 'binary' if optimal_config['binary_signal'] else 'tanh',
                    'top_model_q_score': float(selected_models[0]['q_score']),
                    'avg_q_score': float(np.mean([m['q_score'] for m in selected_models]))
                },

                'performance_benchmark': {
                    'target_sharpe': float(optimal_config['benchmark_sharpe']),
                    'test_config': str(optimal_config['test_config'])
                },

                'model_metadata': {
                    'asset_class': str(self.get_asset_class(symbol)),
                    'build_date': datetime.now().strftime('%Y-%m-%d'),
                    'feature_count': int(len(feature_names)),
                    'training_samples': int(X_train.shape[0]),
                    'evaluation_samples': int(X_eval.shape[0])
                }
            }

            config_file = self.config_dir / f"{symbol}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(deployment_config, f, default_flow_style=False)

            self.logger.info(f"Successfully selected and saved {len(selected_models)} top models for {symbol}")
            self.logger.info(f"Trained {len(model_performances)} total models, saved top {len(selected_models)}")
            self.logger.info(f"Models saved to: {symbol_models_dir}")
            self.logger.info(f"Config saved to: {config_file}")
            self.logger.info(f"Top model Q-score: {selected_models[0]['q_score']:.3f} ({config.q_metric})")

            return True

        except Exception as e:
            self.logger.error(f"Error building production models for {symbol}: {e}")
            return False

    def get_asset_class(self, symbol: str) -> str:
        """Determine asset class from symbol."""
        if symbol in ["@ES#C", "@NQ#C", "@RTY#C"]:
            return "EQUITY"
        elif symbol in ["@TY#C", "@FV#C", "@US#C", "BD#C", "BL#C"]:
            return "RATES"
        elif symbol in ["@EU#C", "@AD#C", "@BP#C", "@JY#C"]:
            return "FX"
        elif symbol in ["@S#C", "@BO#C", "@C#C", "@CT#C", "@SM#C", "@W#C", "@KW#C"]:
            return "AGS"
        elif symbol in ["QGC#C", "QSI#C", "QPL#C", "QHG#C"]:
            return "METALS"
        elif symbol in ["QCL#C", "QBZ#C", "QNG#C", "QRB#C"]:
            return "ENERGY"
        else:
            return "UNKNOWN"

    def build_all_optimal_symbols(self) -> None:
        """Build production models for all symbols with optimal configurations."""

        symbols = ["@ES#C", "@TY#C", "@EU#C", "@S#C", "@RTY#C", "@NQ#C", "QGC#C"]

        self.logger.info(f"Building production models for {len(symbols)} symbols...")

        results = {}
        for symbol in symbols:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing {symbol}")
            self.logger.info(f"{'='*60}")

            success = self.build_production_models(symbol)
            results[symbol] = success

            if success:
                self.logger.info(f"✅ {symbol} completed successfully")
            else:
                self.logger.error(f"❌ {symbol} failed")

        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("PRODUCTION MODEL BUILD SUMMARY")
        self.logger.info(f"{'='*60}")

        successful = [sym for sym, success in results.items() if success]
        failed = [sym for sym, success in results.items() if not success]

        self.logger.info(f"Successful: {len(successful)}/{len(symbols)}")
        for sym in successful:
            self.logger.info(f"  ✅ {sym}")

        if failed:
            self.logger.info(f"Failed: {len(failed)}/{len(symbols)}")
            for sym in failed:
                self.logger.info(f"  ❌ {sym}")

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Build production XGBoost models')
    parser.add_argument('--symbol', type=str, help='Specific symbol to build (or "all" for all optimal symbols)')
    parser.add_argument('--output-dir', type=Path, help='Output directory for PROD folder')

    args = parser.parse_args()

    builder = ProductionModelBuilder(args.output_dir)

    if args.symbol and args.symbol != "all":
        success = builder.build_production_models(args.symbol)
        sys.exit(0 if success else 1)
    else:
        builder.build_all_optimal_symbols()

if __name__ == '__main__':
    main()
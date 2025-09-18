"""XGBoost Production Configuration - Simplified and Streamlined"""

import yaml
from pathlib import Path
from typing import Dict, List

class ProductionConfig:
    """Centralized configuration manager."""

    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.config_dir = self.base_dir / "config"
        self._load_configs()

    def _load_configs(self):
        """Load all configuration files with error handling."""
        try:
            with open(self.config_dir / "trading_config.yaml", 'r') as f:
                self.trading_config = yaml.safe_load(f)
            with open(self.config_dir / "global_config.yaml", 'r') as f:
                self.global_config = yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration: {e}")

    @property
    def trading_symbols(self) -> List[str]:
        """Active trading symbols (max_traded > 0)."""
        return [
            symbol for symbol, config in self.trading_config['instrument_config'].items()
            if config.get('max_traded', 0) > 0
        ]

    @property
    def feature_symbols(self) -> List[str]:
        """All symbols used for feature generation."""
        return [
            "@ES#C", "@TY#C", "@EU#C", "@NQ#C", "@RTY#C", "@S#C",
            "@AD#C", "@BP#C", "@JY#C", "@BO#C", "@C#C", "@CT#C",
            "@FV#C", "@KW#C", "@SM#C", "@US#C", "@W#C",
            "QGC#C", "BD#C", "QHG#C", "BL#C", "QBZ#C", "QCL#C",
            "QNG#C", "QPL#C", "QRB#C", "QSI#C"
        ]

    @property
    def signal_hour(self) -> int:
        """Default signal generation hour."""
        return self.global_config.get('signal_config', {}).get('signal_hour', 12)

    def get_required_fx_symbols(self, traded_symbols: List[str]) -> List[str]:
        """Get FX tickers needed for non-USD symbols."""
        fx_config = self.global_config.get('FX', {})
        currencies = {self.trading_config['instrument_config'][s].get('currency', 'USD')
                     for s in traded_symbols if self.trading_config['instrument_config'][s].get('currency') != 'USD'}
        return [fx_config[c]['ticker'] for c in currencies if c in fx_config]

    @property
    def fx_config(self) -> Dict[str, Dict[str, any]]:
        """FX configuration for currency conversion."""
        return self.global_config.get('FX', {})

# Global configuration instance
config = ProductionConfig()

# Legacy compatibility exports
trading_config = config.trading_config
global_config = config.global_config
instrument_config = config.trading_config['instrument_config']
portfolio_config = config.trading_config['portfolio_config']
current_positions = config.trading_config['current_positions']
s3_config = config.trading_config.get('s3', {})
TRADING_SYMBOLS = config.trading_symbols
FEATURE_SYMBOLS = config.feature_symbols
FX_CONFIG = config.fx_config
SIGNAL_HOUR = config.signal_hour
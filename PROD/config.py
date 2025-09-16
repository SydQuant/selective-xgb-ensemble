"""
Simplified Configuration Manager for XGBoost Production
All configuration in one streamlined file.
"""

import yaml
from pathlib import Path

# Load configurations
BASE_DIR = Path(__file__).parent
CONFIG_DIR = BASE_DIR / "config"

def load_config():
    """Load all configurations."""
    with open(CONFIG_DIR / "trading_config.yaml", 'r') as f:
        trading_config = yaml.safe_load(f)

    with open(CONFIG_DIR / "global_config.yaml", 'r') as f:
        global_config = yaml.safe_load(f)

    return trading_config, global_config

# Global configurations
trading_config, global_config = load_config()

# Extract key variables
instrument_config = trading_config['instrument_config']
portfolio_config = trading_config['portfolio_config']
current_positions = trading_config['current_positions']
s3_config = trading_config.get('s3', {})

# Trading symbols (only those with max_traded > 0)
TRADING_SYMBOLS = [
    symbol for symbol, config in instrument_config.items()
    if config.get('max_traded', 0) > 0
]

FEATURE_SYMBOLS = [
    "@ES#C", "@TY#C", "@EU#C", "@NQ#C", "@RTY#C", "@S#C",
    "@AD#C", "@BP#C", "@JY#C", "@BO#C", "@C#C", "@CT#C",
    "@FV#C", "@KW#C", "@SM#C", "@US#C", "@W#C",
    "QGC#C", "BD#C", "QHG#C", "BL#C", "QBZ#C", "QCL#C",
    "QNG#C", "QPL#C", "QRB#C", "QSI#C"
]

# Default signal hour
SIGNAL_HOUR = global_config.get('signal_config', {}).get('signal_hour', 12)
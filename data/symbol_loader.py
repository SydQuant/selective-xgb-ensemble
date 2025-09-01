import yaml
import os
from typing import List, Dict

def load_symbols_config() -> Dict:
    """Load symbols configuration from symbols.yaml"""
    config_path = os.path.join(os.path.dirname(__file__), 'symbols.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_symbols_by_group(group_name: str) -> List[str]:
    """Get symbols for a specific group"""
    config = load_symbols_config()
    groups = config.get('groups', {})
    return groups.get(group_name, [])

def get_default_symbols() -> List[str]:
    """Get default symbol set"""
    config = load_symbols_config()
    return config.get('default', [])

def get_all_symbols() -> List[str]:
    """Get all available symbols"""
    config = load_symbols_config()
    return config.get('all_symbols', [])

def get_symbols_for_universe(universe: str) -> List[str]:
    """Get symbols for a universe specification"""
    if universe == 'default':
        return get_default_symbols()
    elif universe == 'all':
        return get_all_symbols()
    elif universe in ['fx', 'equity', 'ags', 'ratesus', 'rateseu', 'energy', 'metals']:
        return get_symbols_by_group(universe.upper())
    else:
        # Return default if universe not recognized
        return get_default_symbols()

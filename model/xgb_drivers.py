import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def detect_gpu() -> str:
    """Detect if GPU is available for XGBoost."""
    import platform
    import logging
    import warnings
    logger = logging.getLogger(__name__)
    
    # Check for CUDA support with proper warning detection
    try:
        import xgboost as xgb
        
        # Capture warnings to detect if XGBoost falls back to CPU
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1)
            model.fit([[1]], [1])
            
            # Check if any warnings about CUDA not being compiled were raised
            cuda_warnings = [warning for warning in w if "CUDA support" in str(warning.message)]
            
            if cuda_warnings:
                logger.info("GPU detection: XGBoost not compiled with CUDA support - using CPU")
                return "cpu"
            else:
                logger.info("GPU detection: CUDA GPU available and working")
                return "cuda"
                
    except Exception as e:
        logger.info(f"GPU detection: CUDA test failed ({str(e)}) - using CPU")
    
    # Check if we're on Apple Silicon (future-proofing for MPS support)
    if platform.system() == "Darwin":
        if platform.processor() == "arm":
            logger.info("GPU detection: Apple Silicon detected, but XGBoost doesn't support MPS yet - using CPU")
        else:
            logger.info("GPU detection: Intel Mac detected, no CUDA support - using CPU")
        # Future: could try device="mps" when XGBoost adds support
    else:
        logger.info("GPU detection: No CUDA GPU found - using CPU")
    
    return "cpu"

def _create_base_xgb_spec(rng, device: str, **param_ranges) -> Dict[str, Any]:
    """Create base XGBoost specification with configurable parameter ranges."""
    return {
        "max_depth": int(rng.integers(*param_ranges.get('max_depth', (2, 7)))),
        "learning_rate": float(rng.uniform(*param_ranges.get('learning_rate', (0.03, 0.3)))),
        "n_estimators": int(rng.integers(*param_ranges.get('n_estimators', (30, 300)))),
        "subsample": float(rng.uniform(*param_ranges.get('subsample', (0.6, 1.0)))),
        "colsample_bytree": float(rng.uniform(*param_ranges.get('colsample_bytree', (0.6, 1.0)))),
        "reg_alpha": float(10**rng.uniform(*param_ranges.get('reg_alpha_log', (-5, -1)))),
        "reg_lambda": float(10**rng.uniform(*param_ranges.get('reg_lambda_log', (-4, -0.5)))),
        "min_child_weight": float(rng.uniform(*param_ranges.get('min_child_weight', (0.1, 5.0)))),
        "gamma": 0.0,
        "tree_method": "hist",
        "device": device,
        "random_state": int(rng.integers(0, 2**31-1))
    }

def generate_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """Generate diverse XGBoost specs optimized for financial data."""
    rng = np.random.default_rng(seed)
    device = detect_gpu()
    return [_create_base_xgb_spec(rng, device) for _ in range(n_models)]

def generate_deep_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """Generate deep XGBoost specs with 8-10 depth for alternative architectures."""
    rng = np.random.default_rng(seed)
    device = detect_gpu()
    
    deep_params = {
        'max_depth': (8, 11),
        'learning_rate': (0.01, 0.15),
        'n_estimators': (50, 200),
        'subsample': (0.7, 0.9),
        'colsample_bytree': (0.7, 0.9),
        'reg_alpha_log': (-4, -1),
        'reg_lambda_log': (-3, 0),
        'min_child_weight': (2.0, 8.0)
    }
    
    return [_create_base_xgb_spec(rng, device, **deep_params) for _ in range(n_models)]

def stratified_xgb_bank(all_cols, n_models=50, seed=13):
    """Generate tiered XGBoost models: Conservative (30%), Balanced (50%), Aggressive (20%)."""
    rng = np.random.default_rng(seed)
    nA, nC = int(0.30 * n_models), int(0.20 * n_models)
    nB = n_models - nA - nC
    device = detect_gpu()
    
    def logu(lo, hi):
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

    def get_tier_spec(tier):
        n_features = len(all_cols)
        
        # Tier configurations: feature_range_multiplier, params
        tier_configs = {
            "A": {  # Conservative: 20-36 features
                'feature_mult': (1/3, 20, 36),
                'max_depth': (3, 5), 'learning_rate': (0.03, 0.08), 'n_estimators': (250, 501),
                'subsample': (0.7, 0.95), 'colsample_bytree': (0.5, 0.8),
                'reg_alpha_range': (1e-5, 1e-2), 'reg_lambda_range': (1e-4, 0.3),
                'min_child_weight': (2.0, 6.0)
            },
            "B": {  # Balanced: 40-71 features  
                'feature_mult': (1/2, 40, 71),
                'max_depth': (3, 6), 'learning_rate': (0.05, 0.12), 'n_estimators': (180, 401),
                'subsample': (0.6, 0.95), 'colsample_bytree': (0.5, 0.8),
                'reg_alpha_range': (1e-5, 1e-1), 'reg_lambda_range': (1e-4, 1.0),
                'min_child_weight': (1.0, 6.0)
            },
            "C": {  # Aggressive: 60-91 features
                'feature_mult': (0.8, 60, 91),
                'max_depth': (4, 7), 'learning_rate': (0.08, 0.18), 'n_estimators': (120, 301),
                'subsample': (0.6, 0.9), 'colsample_bytree': (0.3, 0.6),
                'reg_alpha_range': (1e-5, 1e-1), 'reg_lambda_range': (1e-4, 1.0),
                'min_child_weight': (1.0, 4.0)
            }
        }
        
        config = tier_configs[tier]
        
        # Calculate feature count
        mult, min_feat, max_feat = config['feature_mult']
        low = min(min_feat, max(1, int(n_features * mult)))
        high = min(max_feat, n_features) + 1
        K = int(rng.integers(low, high)) if high > low else n_features
        
        # Generate spec using configuration
        spec = {
            "max_depth": int(rng.integers(*config['max_depth'])),
            "learning_rate": float(rng.uniform(*config['learning_rate'])),
            "n_estimators": int(rng.integers(*config['n_estimators'])),
            "subsample": float(rng.uniform(*config['subsample'])),
            "colsample_bytree": float(rng.uniform(*config['colsample_bytree'])),
            "reg_alpha": logu(*config['reg_alpha_range']),
            "reg_lambda": logu(*config['reg_lambda_range']),
            "min_child_weight": float(rng.uniform(*config['min_child_weight'])),
            "gamma": 0.0, "tree_method": "hist", "device": device,
            "random_state": int(rng.integers(0, 2**31-1))
        }
        
        return spec, min(K, n_features)

    tiers = (["A"] * nA) + (["B"] * nB) + (["C"] * nC)
    rng.shuffle(tiers)

    specs, col_slices = [], []
    for tier in tiers:
        spec, K = get_tier_spec(tier)
        cols = list(rng.choice(all_cols, size=K, replace=False))
        specs.append(spec)
        col_slices.append(cols)

    return specs, col_slices

def fit_xgb_on_slice(X_tr: pd.DataFrame, y_tr: pd.Series, spec: Dict[str, Any], force_cpu: bool = False):
    """
    Train XGBoost model with GPU support.
    
    Args:
        X_tr: Training features
        y_tr: Training targets
        spec: XGBoost specification with device settings
        force_cpu: If True, force CPU mode (for multiprocessing compatibility)
    """
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
    
    # Use original spec or force CPU if requested
    final_spec = spec.copy()
    if force_cpu:
        final_spec["device"] = "cpu"
        final_spec["tree_method"] = "hist"
    
    try:
        model = XGBRegressor(**final_spec)
        model.fit(X_tr.values, y_tr.values)
        return model
    except Exception as e:
        # Fallback to CPU if GPU fails
        if not force_cpu and final_spec.get("device") == "cuda":
            import logging
            logging.getLogger(__name__).warning(f"GPU training failed, falling back to CPU: {e}")
            final_spec["device"] = "cpu"
            model = XGBRegressor(**final_spec)
            model.fit(X_tr.values, y_tr.values)
            return model
        else:
            raise

def fold_train_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, specs: List[Dict[str, Any]], use_multiprocessing: bool = False):
    """Train multiple models and predict with optional multiprocessing for CPU."""
    train_preds, test_preds = [], []
    
    if use_multiprocessing and len(specs) > 1:
        # Use multiprocessing for CPU training
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        
        def train_predict_single(args):
            spec, X_tr_vals, y_tr_vals, X_te_vals, X_tr_idx, X_te_idx = args
            # Force CPU for multiprocessing compatibility
            model = fit_xgb_on_slice(
                pd.DataFrame(X_tr_vals, index=X_tr_idx), 
                pd.Series(y_tr_vals, index=X_tr_idx), 
                spec, force_cpu=True
            )
            p_tr = model.predict(X_tr_vals)
            p_te = model.predict(X_te_vals)
            return p_tr, p_te
        
        # Prepare arguments for multiprocessing
        args_list = [
            (spec, X_tr.values, y_tr.values, X_te.values, X_tr.index, X_te.index)
            for spec in specs
        ]
        
        # Use up to 8 workers, but don't exceed CPU count
        max_workers = min(8, mp.cpu_count(), len(specs))
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(train_predict_single, args_list))
        
        # Convert results back to pandas Series
        for i, (p_tr, p_te) in enumerate(results):
            train_preds.append(pd.Series(p_tr, index=X_tr.index, name="m"))
            test_preds.append(pd.Series(p_te, index=X_te.index, name="m"))
    else:
        # Sequential processing (original logic)
        for spec in specs:
            m = fit_xgb_on_slice(X_tr, y_tr, spec)
            p_tr = pd.Series(m.predict(X_tr.values), index=X_tr.index, name="m")
            p_te = pd.Series(m.predict(X_te.values), index=X_te.index, name="m")
            train_preds.append(p_tr)
            test_preds.append(p_te)
    
    return train_preds, test_preds

def fold_train_predict_tiered(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, specs: List[Dict[str, Any]], col_slices: List[List[str]]):
    """
    Tiered training where each model uses a specific subset of features.
    """
    train_preds, test_preds = [], []
    for spec, cols in zip(specs, col_slices):
        # Select only the specified columns for this model
        X_tr_subset = X_tr[cols]
        X_te_subset = X_te[cols]
        
        m = fit_xgb_on_slice(X_tr_subset, y_tr, spec)
        p_tr = pd.Series(m.predict(X_tr_subset.values), index=X_tr.index, name="m")
        p_te = pd.Series(m.predict(X_te_subset.values), index=X_te.index, name="m")
        train_preds.append(p_tr); test_preds.append(p_te)
    return train_preds, test_preds
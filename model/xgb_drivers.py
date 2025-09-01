
import numpy as np
import pandas as pd
from typing import List, Dict, Any
# CHANGED: Added multiprocessing support for parallel XGB training
from multiprocessing import Pool, cpu_count
import logging

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

logger = logging.getLogger(__name__)

def generate_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    specs = []
    for i in range(n_models):
        depth = int(rng.integers(2, 10))
        lr = float(10**rng.uniform(-2.3, -0.7))
        est = int(rng.integers(200, 1200))
        subsample = float(rng.uniform(0.6, 1.0))
        colsample = float(rng.uniform(0.4, 1.0))
        reg_alpha = float(10**rng.uniform(-4, -1))
        reg_lambda = float(10**rng.uniform(-4, 1))
        specs.append({
            "max_depth": depth, "learning_rate": lr, "n_estimators": est,
            "subsample": subsample, "colsample_bytree": colsample,
            "reg_alpha": reg_alpha, "reg_lambda": reg_lambda,
            "min_child_weight": float(rng.uniform(1.0, 6.0)),
            "gamma": float(rng.uniform(0.0, 3.0)),
            "tree_method": "hist", "random_state": int(rng.integers(0, 2**31-1))
        })
    return specs

def fit_xgb_on_slice(X_tr: pd.DataFrame, y_tr: pd.Series, spec: Dict[str, Any]):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
    model = XGBRegressor(**spec)
    model.fit(X_tr.values, y_tr.values)
    return model

# CHANGED: Helper function for multiprocessing XGB training
def _train_single_xgb(args):
    """Train single XGB model - used for multiprocessing"""
    X_tr_values, y_tr_values, X_te_values, spec, tr_index, te_index = args
    try:
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
        
        model = XGBRegressor(**spec)
        model.fit(X_tr_values, y_tr_values)
        
        p_tr = model.predict(X_tr_values)
        p_te = model.predict(X_te_values)
        
        return p_tr, p_te, tr_index, te_index
    except Exception as e:
        logger.error(f"XGB training failed: {e}")
        # Return zeros if training fails
        return np.zeros(len(y_tr_values)), np.zeros(len(X_te_values)), tr_index, te_index

def fold_train_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, specs: List[Dict[str, Any]], use_multiprocessing: bool = True):
    """
    CHANGED: Added multiprocessing support for XGB training.
    Original code trained models sequentially, now trains in parallel for significant speedup.
    """
    if not use_multiprocessing or len(specs) < 4:
        # Fall back to sequential training for small numbers of models
        train_preds, test_preds = [], []
        for spec in specs:
            m = fit_xgb_on_slice(X_tr, y_tr, spec)
            p_tr = pd.Series(m.predict(X_tr.values), index=X_tr.index, name="m")
            p_te = pd.Series(m.predict(X_te.values), index=X_te.index, name="m")
            train_preds.append(p_tr); test_preds.append(p_te)
        return train_preds, test_preds
    
    # CHANGED: Parallel training with multiprocessing
    logger.info(f"Training {len(specs)} XGB models in parallel using {min(cpu_count(), len(specs))} processes")
    
    # Prepare arguments for parallel processing
    args_list = []
    for spec in specs:
        args_list.append((
            X_tr.values, y_tr.values, X_te.values, 
            spec, X_tr.index, X_te.index
        ))
    
    # Train models in parallel
    n_processes = min(cpu_count(), len(specs))
    train_preds, test_preds = [], []
    
    with Pool(n_processes) as pool:
        results = pool.map(_train_single_xgb, args_list)
    
    # Convert results back to pandas Series
    for p_tr_values, p_te_values, tr_index, te_index in results:
        p_tr = pd.Series(p_tr_values, index=tr_index, name="m")
        p_te = pd.Series(p_te_values, index=te_index, name="m")
        train_preds.append(p_tr)
        test_preds.append(p_te)
    
    logger.info(f"Completed parallel XGB training for {len(results)} models")
    return train_preds, test_preds

import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def generate_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """
    Generate diverse XGBoost specs with stability constraints for financial data.
    Balances diversity with reliability for small target values.
    """
    rng = np.random.default_rng(seed)
    specs = []
    
    for i in range(n_models):
        # Broader but stable parameter ranges
        depth = int(rng.integers(2, 7))  # 2-6 depth
        lr = float(rng.uniform(0.03, 0.3))  # Wider learning rate range
        est = int(rng.integers(30, 300))  # More estimator variety
        
        # Regularization: light to moderate (avoid heavy reg that causes constants)
        reg_alpha = float(10**rng.uniform(-5, -1))  # 1e-5 to 0.1
        reg_lambda = float(10**rng.uniform(-4, -0.5))  # 1e-4 to ~0.3
        
        # Sample diversity parameters more broadly
        subsample = float(rng.uniform(0.6, 1.0))
        colsample = float(rng.uniform(0.6, 1.0))
        min_child = float(rng.uniform(0.1, 5.0))  # Wider range
        
        spec = {
            "max_depth": depth,
            "learning_rate": lr,
            "n_estimators": est,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_child_weight": min_child,
            "gamma": 0.0,  # Keep gamma=0 for stability with small targets
            "tree_method": "hist",
            "random_state": int(rng.integers(0, 2**31-1))
        }
        specs.append(spec)
    
    return specs

def fit_xgb_on_slice(X_tr: pd.DataFrame, y_tr: pd.Series, spec: Dict[str, Any]):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
    model = XGBRegressor(**spec)
    model.fit(X_tr.values, y_tr.values)
    return model

def fold_train_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, specs: List[Dict[str, Any]]):
    train_preds, test_preds = [], []
    for spec in specs:
        m = fit_xgb_on_slice(X_tr, y_tr, spec)
        p_tr = pd.Series(m.predict(X_tr.values), index=X_tr.index, name="m")
        p_te = pd.Series(m.predict(X_te.values), index=X_te.index, name="m")
        train_preds.append(p_tr); test_preds.append(p_te)
    return train_preds, test_preds
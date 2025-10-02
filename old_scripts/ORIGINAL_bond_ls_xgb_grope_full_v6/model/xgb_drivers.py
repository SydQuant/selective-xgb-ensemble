
import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

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

def fold_train_predict(X_tr: pd.DataFrame, y_tr: pd.Series, X_te: pd.DataFrame, specs: List[Dict[str, Any]]):
    train_preds, test_preds = [], []
    for spec in specs:
        m = fit_xgb_on_slice(X_tr, y_tr, spec)
        p_tr = pd.Series(m.predict(X_tr.values), index=X_tr.index, name="m")
        p_te = pd.Series(m.predict(X_te.values), index=X_te.index, name="m")
        train_preds.append(p_tr); test_preds.append(p_te)
    return train_preds, test_preds

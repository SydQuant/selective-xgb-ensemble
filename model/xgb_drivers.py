import numpy as np
import pandas as pd
from typing import List, Dict, Any
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

def detect_gpu() -> str:
    """Detect if GPU is available for XGBoost."""
    try:
        import xgboost as xgb
        xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1).fit([[1]], [1])
        return "cuda"
    except:
        return "cpu"

def generate_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """Generate diverse XGBoost specs optimized for financial data."""
    rng = np.random.default_rng(seed)
    specs = []
    device = detect_gpu()
    
    for i in range(n_models):
        spec = {
            "max_depth": int(rng.integers(2, 7)),
            "learning_rate": float(rng.uniform(0.03, 0.3)),
            "n_estimators": int(rng.integers(30, 300)),
            "subsample": float(rng.uniform(0.6, 1.0)),
            "colsample_bytree": float(rng.uniform(0.6, 1.0)),
            "reg_alpha": float(10**rng.uniform(-5, -1)),
            "reg_lambda": float(10**rng.uniform(-4, -0.5)),
            "min_child_weight": float(rng.uniform(0.1, 5.0)),
            "gamma": 0.0,
            "tree_method": "hist",
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1))
        }
        specs.append(spec)
    
    return specs

def generate_deep_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """Generate deep XGBoost specs with 8-10 depth for alternative architectures."""
    rng = np.random.default_rng(seed)
    specs = []
    device = detect_gpu()
    
    for i in range(n_models):
        spec = {
            "max_depth": int(rng.integers(8, 11)),
            "learning_rate": float(rng.uniform(0.01, 0.15)),
            "n_estimators": int(rng.integers(50, 200)),
            "subsample": float(rng.uniform(0.7, 0.9)),
            "colsample_bytree": float(rng.uniform(0.7, 0.9)),
            "reg_alpha": float(10**rng.uniform(-4, -1)),
            "reg_lambda": float(10**rng.uniform(-3, 0)),
            "min_child_weight": float(rng.uniform(2.0, 8.0)),
            "gamma": 0.0,
            "tree_method": "hist",
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1))
        }
        specs.append(spec)
    
    return specs

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
        
        if tier == "A":  # Conservative
            # Target range: 20-36 features, but adapt to available features
            low = min(20, max(1, n_features // 3))
            high = min(36, n_features) + 1  # +1 for rng.integers upper bound
            K = int(rng.integers(low, high)) if high > low else n_features
            return {
                "max_depth": int(rng.integers(3, 5)),
                "learning_rate": float(rng.uniform(0.03, 0.08)),
                "n_estimators": int(rng.integers(250, 501)),
                "subsample": float(rng.uniform(0.7, 0.95)),
                "colsample_bytree": float(rng.uniform(0.5, 0.8)),
                "reg_alpha": logu(1e-5, 1e-2),
                "reg_lambda": logu(1e-4, 0.3),
                "min_child_weight": float(rng.uniform(2.0, 6.0)),
                "gamma": 0.0, "tree_method": "hist", "device": device,
                "random_state": int(rng.integers(0, 2**31-1))
            }, min(K, n_features)
        elif tier == "B":  # Balanced
            # Target range: 40-71 features, but adapt to available features
            low = min(40, max(1, n_features // 2))
            high = min(71, n_features) + 1  # +1 for rng.integers upper bound
            K = int(rng.integers(low, high)) if high > low else n_features
            return {
                "max_depth": int(rng.integers(3, 6)),
                "learning_rate": float(rng.uniform(0.05, 0.12)),
                "n_estimators": int(rng.integers(180, 401)),
                "subsample": float(rng.uniform(0.6, 0.95)),
                "colsample_bytree": float(rng.uniform(0.5, 0.8)),
                "reg_alpha": logu(1e-5, 1e-1),
                "reg_lambda": logu(1e-4, 1.0),
                "min_child_weight": float(rng.uniform(1.0, 6.0)),
                "gamma": 0.0, "tree_method": "hist", "device": device,
                "random_state": int(rng.integers(0, 2**31-1))
            }, min(K, n_features)
        else:  # Aggressive
            # Target range: 60-91 features, but adapt to available features
            low = min(60, max(1, int(n_features * 0.8)))
            high = min(91, n_features) + 1  # +1 for rng.integers upper bound
            K = int(rng.integers(low, high)) if high > low else n_features
            return {
                "max_depth": int(rng.integers(4, 7)),
                "learning_rate": float(rng.uniform(0.08, 0.18)),
                "n_estimators": int(rng.integers(120, 301)),
                "subsample": float(rng.uniform(0.6, 0.9)),
                "colsample_bytree": float(rng.uniform(0.3, 0.6)),
                "reg_alpha": logu(1e-5, 1e-1),
                "reg_lambda": logu(1e-4, 1.0),
                "min_child_weight": float(rng.uniform(1.0, 4.0)),
                "gamma": 0.0, "tree_method": "hist", "device": device,
                "random_state": int(rng.integers(0, 2**31-1))
            }, min(K, n_features)

    tiers = (["A"] * nA) + (["B"] * nB) + (["C"] * nC)
    rng.shuffle(tiers)

    specs, col_slices = [], []
    for tier in tiers:
        spec, K = get_tier_spec(tier)
        cols = list(rng.choice(all_cols, size=K, replace=False))
        specs.append(spec)
        col_slices.append(cols)

    return specs, col_slices

def fit_xgb_on_slice(X_tr: pd.DataFrame, y_tr: pd.Series, spec: Dict[str, Any]):
    if XGBRegressor is None:
        raise ImportError("xgboost is not installed. Please `pip install xgboost`.")
    
    # Force CPU mode to avoid GPU/CPU data mismatch warnings
    spec_cpu = spec.copy()
    spec_cpu["device"] = "cpu"
    spec_cpu["tree_method"] = "hist"
    
    model = XGBRegressor(**spec_cpu)
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
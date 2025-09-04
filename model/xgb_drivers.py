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
    """
    Generate diverse XGBoost specs with stability constraints for financial data.
    Balances diversity with reliability for small target values.
    """
    rng = np.random.default_rng(seed)
    specs = []
    
    tree_method = "hist"
    device = detect_gpu()
    
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
            "tree_method": tree_method,
            "device": device,  # Modern XGBoost device parameter
            "random_state": int(rng.integers(0, 2**31-1))
        }
        specs.append(spec)
    
    return specs

def generate_deep_xgb_specs(n_models: int = 50, seed: int = 13) -> List[Dict[str, Any]]:
    """
    Generate deep XGBoost specs with 8-10 depth for Step 4 testing.
    Tests deeper trees vs baseline 2-6 depth.
    """
    rng = np.random.default_rng(seed)
    specs = []
    
    # Determine optimal tree method (GPU if available, CPU fallback)
    tree_method = "hist"  # Default CPU method
    device = "cpu"  # Default device
    try:
        import xgboost as xgb
        # Test if GPU is available with modern syntax
        xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1).fit([[1]], [1])
        device = "cuda"
        print("GPU acceleration enabled for Deep XGBoost")
    except:
        print("Using CPU-based Deep XGBoost (GPU not available or not configured)")
    
    for i in range(n_models):
        # Deep architecture: 8-10 depth (vs baseline 2-6)
        depth = int(rng.integers(8, 11))  # 8-10 depth
        lr = float(rng.uniform(0.01, 0.15))  # Lower LR for deeper trees
        est = int(rng.integers(50, 200))  # Fewer estimators for deeper trees
        
        # Stronger regularization to handle depth complexity
        reg_alpha = float(10**rng.uniform(-4, -1))  # 1e-4 to 0.1
        reg_lambda = float(10**rng.uniform(-3, 0))   # 1e-3 to 1.0
        
        # More conservative sampling for deep trees
        subsample = float(rng.uniform(0.7, 0.9))
        colsample = float(rng.uniform(0.7, 0.9))
        min_child = float(rng.uniform(2.0, 8.0))  # Higher for deeper trees
        
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
            "tree_method": tree_method,
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1))
        }
        specs.append(spec)
    
    return specs

def stratified_xgb_bank(all_cols, n_models=50, seed=13):
    """
    Generate stratified XGBoost models with tiered architecture.
    
    Tier A (30%): Conservative models - shallow trees, low LR, fewer features
    Tier B (50%): Balanced models - medium complexity 
    Tier C (20%): Aggressive models - deeper trees, higher LR, more features
    
    Returns:
      specs: list of dicts (XGB params)
      col_slices: list of column name lists, one per model
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    nA = int(round(0.30 * n_models))  # Tier A count
    nC = int(round(0.20 * n_models))  # Tier C count
    nB = n_models - nA - nC           # Tier B count

    def logu(lo, hi):
        return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

    # Determine optimal tree method (GPU if available, CPU fallback)
    tree_method = "hist"
    device = detect_gpu()
    try:
        import xgboost as xgb
        xgb.XGBRegressor(tree_method="hist", device="cuda", n_estimators=1).fit([[1]], [1])
        device = "cuda"
    except:
        pass

    def tier_A_spec():
        spec = {
            "max_depth": int(rng.integers(3, 5)),                 # 3-4
            "learning_rate": float(rng.uniform(0.03, 0.08)),
            "n_estimators": int(rng.integers(250, 501)),
            "subsample": float(rng.uniform(0.7, 0.95)),
            "colsample_bytree": float(rng.uniform(0.5, 0.8)),
            "colsample_bylevel": float(rng.uniform(0.7, 1.0)),
            "colsample_bynode": float(rng.uniform(0.7, 1.0)),
            "reg_alpha": logu(1e-5, 1e-2),
            "reg_lambda": logu(1e-4, 0.3),
            "gamma": 0.0,  # Keep gamma=0 for stability with small targets
            "min_child_weight": float(rng.uniform(2.0, 6.0)),
            "tree_method": tree_method,
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1)),
        }
        # Tier A: 20-35 features (capped by available)
        min_k = min(20, len(all_cols))
        max_k = min(36, len(all_cols))
        if min_k >= max_k:
            K = len(all_cols)  # Use all available features when not enough for range
        else:
            K = int(rng.integers(min_k, max_k))
        return spec, K

    def tier_B_spec():
        spec = {
            "max_depth": int(rng.integers(3, 6)),                 # 3-5
            "learning_rate": float(rng.uniform(0.05, 0.12)),
            "n_estimators": int(rng.integers(180, 401)),
            "subsample": float(rng.uniform(0.6, 0.95)),
            "colsample_bytree": float(rng.uniform(0.5, 0.8)),
            "colsample_bylevel": float(rng.uniform(0.7, 1.0)),
            "colsample_bynode": float(rng.uniform(0.7, 1.0)),
            "reg_alpha": logu(1e-5, 1e-1),
            "reg_lambda": logu(1e-4, 1.0),
            "gamma": 0.0,  # Keep gamma=0 for stability with small targets
            "min_child_weight": float(rng.uniform(1.0, 6.0)),
            "tree_method": tree_method,
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1)),
        }
        # Tier B: 40-70 features (capped by available)  
        min_k = min(40, len(all_cols))
        max_k = min(71, len(all_cols))
        if min_k >= max_k:
            K = len(all_cols)  # Use all available features when not enough for range
        else:
            K = int(rng.integers(min_k, max_k))
        return spec, K

    def tier_C_spec():
        spec = {
            "max_depth": int(rng.integers(4, 7)),                 # 4-6
            "learning_rate": float(rng.uniform(0.08, 0.18)),
            "n_estimators": int(rng.integers(120, 301)),
            "subsample": float(rng.uniform(0.6, 0.9)),
            "colsample_bytree": float(rng.uniform(0.3, 0.6)),     # lower when K is big
            "colsample_bylevel": float(rng.uniform(0.6, 1.0)),
            "colsample_bynode": float(rng.uniform(0.6, 1.0)),
            "reg_alpha": logu(1e-5, 1e-1),
            "reg_lambda": logu(1e-4, 1.0),
            "gamma": 0.0,  # Keep gamma=0 for stability with small targets
            "min_child_weight": float(rng.uniform(1.0, 4.0)),
            "tree_method": tree_method,
            "device": device,
            "random_state": int(rng.integers(0, 2**31-1)),
        }
        # Tier C: 60-90 features (capped by available)
        min_k = min(60, len(all_cols))
        max_k = min(91, len(all_cols))
        if min_k >= max_k:
            K = len(all_cols)  # Use all available features when not enough for range
        else:
            K = int(rng.integers(min_k, max_k + 1))
        return spec, K

    tiers = (["A"] * nA) + (["B"] * nB) + (["C"] * nC)
    rng.shuffle(tiers)

    specs, col_slices = [], []
    for _t in tiers:
        spec, K = (tier_A_spec() if _t == "A" else tier_B_spec() if _t == "B" else tier_C_spec())
        # Handle case where we have fewer columns than requested
        K = min(K, len(all_cols))
        cols = list(rng.choice(all_cols, size=K, replace=False))
        specs.append(spec)
        col_slices.append(cols)

    return specs, col_slices

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
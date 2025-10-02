
import numpy as np
from typing import Dict, Any, Callable, Tuple, List

def latin_hypercube(n_pts: int, bounds: Dict[str, Tuple[float, float]], rng) -> List[Dict[str, Any]]:
    keys = list(bounds.keys())
    dims = len(keys)
    samples = np.zeros((n_pts, dims))
    for d, k in enumerate(keys):
        low, high = bounds[k]
        perm = rng.permutation(n_pts) + rng.random(n_pts)
        samples[:, d] = low + (high - low) * perm / n_pts
    thetas = []
    for i in range(n_pts):
        thetas.append({k: float(samples[i, d]) for d, k in enumerate(keys)})
    return thetas

def rbf_fit(X: np.ndarray, y: np.ndarray, epsilon: float = 1.0):
    n, d = X.shape
    D = np.sqrt(((X[:, None, :] - X[None, :, :])**2).sum(axis=2))
    Phi = np.sqrt((epsilon*D)**2 + 1.0)
    lam = 1e-6
    w = np.linalg.solve(Phi + lam*np.eye(n), y)
    return w, X.copy(), epsilon

def rbf_predict(model, x: np.ndarray):
    w, X, epsilon = model
    d = np.sqrt(((X - x[None, :])**2).sum(axis=1))
    phi = np.sqrt((epsilon*d)**2 + 1.0)
    return float((w * phi).sum())

def suggest_candidates(bounds, n_cand, rng):
    xs = []
    for _ in range(n_cand):
        cand = {}
        for k, (low, high) in bounds.items():
            cand[k] = float(low + (high - low) * rng.random())
        xs.append(cand)
    return xs

def clamp_theta(theta, bounds):
    out = {}
    for k, (low, high) in bounds.items():
        v = float(theta[k])
        if v < low: v = low
        if v > high: v = high
        out[k] = v
    return out

def grope_optimize(bounds: Dict[str, Tuple[float, float]], f: Callable[[Dict[str, Any]], float], budget: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    history = []
    n_init = max(8, min(16, budget//2))
    for theta in latin_hypercube(n_init, bounds, rng):
        y = f(theta)
        history.append((theta, y))
    while len(history) < budget:
        X = np.array([[t[k] for k in bounds.keys()] for t, _ in history])
        y = np.array([s for _, s in history])
        model = rbf_fit(X, y)
        cands = suggest_candidates(bounds, n_cand=24, rng=rng)
        best_idx = int(np.argmax(y))
        x_best = X[best_idx]
        for rad in [0.05, 0.1, 0.2]:
            eps = rng.normal(scale=rad, size=len(bounds))
            loc = {k: float(x_best[i] + eps[i]*(list(bounds.values())[i][1]-list(bounds.values())[i][0])) for i, k in enumerate(bounds.keys())}
            cands.append(clamp_theta(loc, bounds))
        preds = []
        for c in cands:
            x = np.array([c[k] for k in bounds.keys()])
            preds.append(rbf_predict(model, x))
        order = np.argsort(preds)[::-1]
        picked = False
        for idx in order:
            theta = cands[idx]
            if any(all(abs(theta[k]-t[k]) < 1e-6 for k in bounds.keys()) for t,_ in history):
                continue
            yv = f(theta)
            history.append((theta, yv))
            picked = True
            break
        if not picked:
            theta = cands[order[0]]
            yv = f(theta)
            history.append((theta, yv))
    best_theta, best_y = max(history, key=lambda ty: ty[1])
    return best_theta, best_y, history

# sharpe_stability_selector.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Dict

# ----------------------------
# Sharpe + helpers (no leakage)
# ----------------------------
def _position_from_scores(scores: pd.Series) -> pd.Series:
    """
    Turn raw model scores into a tradeable position in [-1,1].
    You can swap this for your zscore→tanh transform if you prefer.
    """
    s = pd.Series(scores, copy=True)
    s = (s - s.mean()) / (s.std(ddof=0) + 1e-9)
    return np.tanh(s)  # [-1, 1]

def sharpe_annualized(signal: pd.Series,
                      rets: pd.Series,
                      costs_per_turn: float = 0.0,
                      ann: int = 252) -> float:
    """
    One-day delay. Optional linear turnover cost.
    """
    sig = signal.reindex_like(rets)
    pos = sig.shift(1).fillna(0.0)

    cost = 0.0
    if costs_per_turn:
        turnover = pos.diff().abs().fillna(0.0)
        cost = turnover * costs_per_turn

    pnl = (pos * rets).astype(float) - (cost if isinstance(cost, pd.Series) else 0.0)
    mu, sd = pnl.mean(), pnl.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return 0.0
    return float((mu / sd) * np.sqrt(ann))

# ---------------------------------------
# Per-model stability score (Sharpe-based)
# ---------------------------------------
def model_sharpe_stability_score(
    raw_scores: pd.Series,         # model's continuous scores on the WHOLE train slice (we'll index into it)
    y: pd.Series,                  # future returns aligned to raw_scores
    tr_idx: Iterable,              # inner train indices (subset of the outer train)
    va_idx: Iterable,              # inner validation indices (subset of the outer train)
    costs_per_turn: float = 0.0,
    alpha: float = 1.0,            # weight on validation Sharpe
    lam: float = 0.5,              # penalty on train→val drop (gap)
    hr_min: float | None = None,   # optional hit-rate gate on validation
) -> Tuple[float, Dict[str, float]]:
    """
    Stability-first score:
        stab = alpha * SR_val  - lam * max(0, SR_train - SR_val)
    with optional hit-rate gate on the validation slice.
    """
    # Slice to inner train/val
    sc_tr = raw_scores.iloc[tr_idx]
    sc_va = raw_scores.iloc[va_idx]
    y_tr  = y.iloc[tr_idx]
    y_va  = y.iloc[va_idx]

    # Map to positions
    pos_tr = _position_from_scores(sc_tr)
    pos_va = _position_from_scores(sc_va)

    # Metrics
    sr_tr = sharpe_annualized(pos_tr, y_tr, costs_per_turn=costs_per_turn)
    sr_va = sharpe_annualized(pos_va, y_va, costs_per_turn=costs_per_turn)

    # Optional hit-rate gate on validation
    if hr_min is not None:
        hr = float((np.sign(pos_va) == np.sign(y_va)).mean())
        if hr < hr_min:
            return -1e9, dict(sr_tr=sr_tr, sr_va=sr_va, gap=max(0.0, sr_tr - sr_va), hit_rate=hr)

    gap = max(0.0, sr_tr - sr_va)
    stab = alpha * sr_va - lam * gap
    return float(stab), dict(sr_tr=sr_tr, sr_va=sr_va, gap=gap)

# -------------------------------------------------
# Rolling selection + OOS prediction (Sharpe-based)
# -------------------------------------------------
def rolling_select_and_predict_sharpe(
    X: pd.DataFrame,
    y: pd.Series,
    driver_bank,                   # iterable of drivers with .fit(X,y) -> self and .predict(X) -> np.array
    start_train: int = 750,        # how many rows before first OOS
    step: int = 21,                # advance per refit (e.g., ~1 month)
    horizon: int = 21,             # OOS pane length per refit
    inner_val_frac: float = 0.2,   # % of the training slice used as inner validation
    top_k: int = 20,               # how many drivers to keep each refit
    costs_per_turn: float = 0.0,
    alpha: float = 1.0,
    lam: float = 0.5,
    hr_min: float | None = None,   # optional hit-rate gate on inner validation
    enforce_diversity: bool = False,
    corr_thresh: float = 0.7,      # if enforcing diversity, max allowed corr on inner val
) -> pd.Series:
    """
    At each refit date:
      1) Fit all drivers on the training slice (0..t0-1)
      2) Inside that slice, split into inner-train / inner-val
      3) Score stability (Sharpe-based) per driver
      4) Pick top_k (optionally diversity-filtered)
      5) Equal-weight (or plug your optimizer here) on the OOS horizon
    Returns a concatenated OOS signal.
    """
    n = len(X)
    oos_slices: List[pd.Series] = []
    t0 = start_train

    while t0 + horizon <= n:
        tr_full = np.arange(0, t0)
        cut = int(len(tr_full) * (1.0 - inner_val_frac))
        inner_tr, inner_va = tr_full[:cut], tr_full[cut:]

        # Fit all drivers on full training slice
        fitted = []
        preds_train = []
        preds_val   = []
        for drv in driver_bank:
            m = drv.fit(X.iloc[tr_full], y.iloc[tr_full])  # driver returns self
            p_tr = pd.Series(m.predict(X.iloc[inner_tr]), index=X.index[inner_tr])
            p_va = pd.Series(m.predict(X.iloc[inner_va]), index=X.index[inner_va])
            fitted.append(m)
            preds_train.append(p_tr)
            preds_val.append(p_va)

        # Score stability for each driver (Sharpe-based)
        scores = []
        for j, (p_tr, p_va) in enumerate(zip(preds_train, preds_val)):
            # Build a combined series over inner train + val for indexing; metrics use indices directly
            sig = pd.concat([p_tr, p_va]).sort_index()
            stab, comps = model_sharpe_stability_score(
                sig, y, inner_tr, inner_va,
                costs_per_turn=costs_per_turn,
                alpha=alpha, lam=lam, hr_min=hr_min
            )
            scores.append((stab, j, comps))
        scores.sort(key=lambda x: x[0], reverse=True)

        # Optional diversity filter based on correlation on the inner validation window
        picked: List[int] = []
        if enforce_diversity and len(scores) > 0:
            # collect validation predictions matrix in ranked order
            cand_ids = [j for _, j, _ in scores]
            P_va = np.column_stack([preds_val[j].values for j in cand_ids])
            # zscore→tanh to make scales comparable
            P_va = np.tanh((P_va - P_va.mean(axis=0, keepdims=True)) /
                           (P_va.std(axis=0, keepdims=True) + 1e-9))
            corr = np.corrcoef(P_va.T)
            for rank_pos, j in enumerate(cand_ids):
                if len(picked) == 0:
                    picked.append(j)
                else:
                    ok = True
                    for k in picked:
                        c = corr[rank_pos, cand_ids.index(k)]
                        if np.isfinite(c) and abs(c) > corr_thresh:
                            ok = False
                            break
                    if ok:
                        picked.append(j)
                if len(picked) >= top_k:
                    break
        else:
            picked = [j for _, j, _ in scores[:top_k]]

        # Build OOS ensemble for this horizon from picked drivers
        test_idx = np.arange(t0, t0 + horizon)
        P = np.column_stack([fitted[j].predict(X.iloc[test_idx]) for j in picked])
        # map to positions driver-wise
        P = np.tanh((P - P.mean(axis=0, keepdims=True)) / (P.std(axis=0, keepdims=True) + 1e-9))
        w = np.ones(P.shape[1], dtype=float) / max(1, P.shape[1])  # equal weights (swap in optimizer if desired)
        s_oos = pd.Series(P @ w, index=X.index[test_idx]).clip(-1, 1)
        oos_slices.append(s_oos)

        t0 += step

    return (pd.concat(oos_slices).sort_index() if oos_slices else
            pd.Series([], dtype=float, index=X.index))
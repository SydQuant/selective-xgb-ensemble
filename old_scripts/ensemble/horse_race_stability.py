# horse_race_stability.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Utilities: positions, PnL metrics
# ---------------------------------------------------------------------------
def z_tanh(scores: np.ndarray) -> np.ndarray:
    eps = 1e-9
    z = (scores - scores.mean()) / (scores.std(ddof=0) + eps)
    return np.tanh(z)

def positions_from_scores(scores: pd.Series) -> pd.Series:
    return pd.Series(z_tanh(scores.values), index=scores.index)

def pnl_stats_from_signal(signal: pd.Series, rets: pd.Series,
                          costs_per_turn: float = 0.0, ann: int = 252) -> Dict[str, float | pd.Series]:
    sig = signal.reindex_like(rets)
    pos = sig.shift(1).fillna(0.0)

    costs = 0.0
    if costs_per_turn:
        turnover = pos.diff().abs().fillna(0.0)
        costs = turnover * costs_per_turn

    pnl = (pos * rets).astype(float) - (costs if isinstance(costs, pd.Series) else 0.0)
    mu, sd = pnl.mean(), pnl.std(ddof=0)
    ann_ret = mu * ann
    ann_vol = sd * np.sqrt(ann) if sd > 0 else 0.0
    ir = (mu / sd) * np.sqrt(ann) if sd > 0 else 0.0
    hit = float((np.sign(pos) == np.sign(rets)).mean())

    return dict(AnnRet=float(ann_ret), AnnVol=float(ann_vol), Sharpe=float(ir),
                HitRate=float(hit), PnL=pnl, Equity=pnl.cumsum())

# ---------------------------------------------------------------------------
# Metric functions evaluated on a slice
# ---------------------------------------------------------------------------
def metric_sharpe(scores: pd.Series, rets: pd.Series,
                  costs_per_turn: float = 0.0) -> float:
    pos = positions_from_scores(scores)
    stats = pnl_stats_from_signal(pos, rets, costs_per_turn=costs_per_turn)
    return float(stats["Sharpe"])

def metric_adj_sharpe(scores: pd.Series, rets: pd.Series,
                      costs_per_turn: float = 0.0, lambda_to: float = 0.0) -> float:
    # Sharpe - lambda_to * avg daily turnover
    pos = positions_from_scores(scores)
    stats = pnl_stats_from_signal(pos, rets, costs_per_turn=0.0)  # costs handled via turnover term here
    turnover = pos.diff().abs().fillna(0.0).mean()
    return float(stats["Sharpe"] - lambda_to * float(turnover))

def metric_hit_rate(scores: pd.Series, rets: pd.Series) -> float:
    s = np.sign(scores.reindex_like(rets).values)
    r = np.sign(rets.values)
    m = (~np.isnan(s)) & (~np.isnan(r))
    return 0.0 if not np.any(m) else float(np.mean(s[m] == r[m]))

# ---------------------------------------------------------------------------
# Stability score from train/val metrics
# ---------------------------------------------------------------------------
def stability_score(m_tr: float, m_va: float, alpha: float = 1.0,
                    lam_gap: float = 0.5, relative: bool = False) -> float:
    """
    Larger is better. Encourages high validation metric with small train→val drop.
    """
    if relative:
        # penalize drop as a fraction of |train|
        denom = max(1e-9, abs(m_tr))
        gap = max(0.0, (m_tr - m_va) / denom)
    else:
        gap = max(0.0, m_tr - m_va)
    return float(alpha * m_va - lam_gap * gap)

# ---------------------------------------------------------------------------
# Driver bank interface
# ---------------------------------------------------------------------------
class Driver:
    """
    Minimal interface example:
      - fit(self, X, y) -> self
      - predict(self, X) -> np.ndarray shape (n,)
    Replace with your actual driver class.
    """
    def __init__(self, model):
        self.model = model
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.values, y.values)
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

# ---------------------------------------------------------------------------
# Configs for the horse race
# ---------------------------------------------------------------------------
@dataclass
class MetricConfig:
    name: str
    fn: Callable[..., float]                 # metric function(scores, rets, **kwargs)
    kwargs: Dict[str, float]                 # e.g., {"costs_per_turn": 0.0} or {"lambda_to": 0.1}
    alpha: float = 1.0                       # weight on validation metric
    lam_gap: float = 0.5                     # penalty on train→val drop
    relative_gap: bool = False               # use relative gap?
    top_k: int = 3                           # number of drivers to pick per refit
    eta_quality: float = 0.0                 # weight for "previous OOS quality" memory (0 = off)

# ---------------------------------------------------------------------------
# Rolling horse race
# ---------------------------------------------------------------------------
def rolling_horse_race(
    X: pd.DataFrame,
    y: pd.Series,
    drivers: List[Driver],
    metrics: List[MetricConfig],
    start_train: int = 750,
    step: int = 21,
    horizon: int = 21,
    inner_val_frac: float = 0.2,
    costs_per_turn_backtest: float = 0.0,
    quality_halflife: int = 63,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    For each metric config:
      - per refit window, compute stability scores per driver
      - optionally blend with a trailing OOS-quality score (EWMA) per driver
      - select top_k drivers, build equal-weight OOS ensemble for next horizon
      - record realized OOS stats (AnnRet, AnnVol, Sharpe, HitRate)
      - also record predictive power: Spearman corr between validation stability score and OOS Sharpe across drivers
    Returns:
      summary_df: aggregated metrics per metric.name
      window_df:  per-window metrics (wide format, columns MultiIndex (metric, field))
      details:    dict with extra diagnostics per metric
    """
    n = len(X)
    assert len(y) == n

    # Trailing OOS quality memory per driver (for "previous OOS" enhancement)
    # We'll store EWMA of realized OOS Sharpe per driver (updated after each window).
    driver_quality: np.ndarray = np.zeros(len(drivers), dtype=float)
    q_decay = np.exp(np.log(0.5) / max(1, quality_halflife))  # daily-ish decay if step~1

    per_metric_oos: Dict[str, List[Dict[str, float]]] = {m.name: [] for m in metrics}
    per_metric_corrs: Dict[str, List[float]] = {m.name: [] for m in metrics}
    per_metric_picks: Dict[str, List[List[int]]] = {m.name: [] for m in metrics}

    # Also keep realized OOS Sharpe per driver per window for the predictive-corr calc
    oos_sharpe_driver_windows: List[np.ndarray] = []

    t0 = start_train
    window_rows = []

    while t0 + horizon <= n:
        tr_full = np.arange(0, t0)
        cut = int(len(tr_full) * (1.0 - inner_val_frac))
        inner_tr, inner_va = tr_full[:cut], tr_full[cut:]
        test_idx = np.arange(t0, t0 + horizon)

        # Fit all drivers on full training slice; cache preds
        preds_inner_tr = []
        preds_inner_va = []
        preds_test = []
        for drv in drivers:
            m = drv.fit(X.iloc[tr_full], y.iloc[tr_full])
            preds_inner_tr.append(pd.Series(m.predict(X.iloc[inner_tr]), index=X.index[inner_tr]))
            preds_inner_va.append(pd.Series(m.predict(X.iloc[inner_va]), index=X.index[inner_va]))
            preds_test.append(pd.Series(m.predict(X.iloc[test_idx]), index=X.index[test_idx]))

        # Compute realized OOS Sharpe per driver (for predictive corr and quality memory)
        # (We use plain Sharpe on each driver's OOS scores, equal preprocessing.)
        driver_oos_sharpes = np.zeros(len(drivers), dtype=float)
        for j in range(len(drivers)):
            pos_test = positions_from_scores(preds_test[j])
            stats = pnl_stats_from_signal(pos_test, y.iloc[test_idx], costs_per_turn=costs_per_turn_backtest)
            driver_oos_sharpes[j] = float(stats["Sharpe"])
        oos_sharpe_driver_windows.append(driver_oos_sharpes.copy())

        # For each metric, compute stability scores & pick top_k (optionally blending prior OOS quality)
        for M in metrics:
            scores = np.zeros(len(drivers), dtype=float)
            base_scores = np.zeros(len(drivers), dtype=float)
            for j in range(len(drivers)):
                sc_tr = preds_inner_tr[j]; sc_va = preds_inner_va[j]
                # evaluate metric on inner train/val
                if M.name.lower().startswith("adj_sharpe"):
                    m_tr = metric_adj_sharpe(sc_tr, y.iloc[inner_tr], **M.kwargs)
                    m_va = metric_adj_sharpe(sc_va, y.iloc[inner_va], **M.kwargs)
                elif M.name.lower().startswith("sharpe"):
                    m_tr = metric_sharpe(sc_tr, y.iloc[inner_tr], **M.kwargs)
                    m_va = metric_sharpe(sc_va, y.iloc[inner_va], **M.kwargs)
                elif M.name.lower().startswith("hit"):
                    m_tr = metric_hit_rate(sc_tr, y.iloc[inner_tr])
                    m_va = metric_hit_rate(sc_va, y.iloc[inner_va])
                else:
                    # default to Sharpe if unrecognized
                    m_tr = metric_sharpe(sc_tr, y.iloc[inner_tr], **M.kwargs)
                    m_va = metric_sharpe(sc_va, y.iloc[inner_va], **M.kwargs)

                base = stability_score(m_tr, m_va, alpha=M.alpha, lam_gap=M.lam_gap, relative=M.relative_gap)
                base_scores[j] = base

            # blend with trailing OOS quality (if eta_quality>0)
            if M.eta_quality > 0.0:
                # z-score normalize quality across drivers to avoid scale issues
                q = driver_quality.copy()
                if np.isfinite(q).any():
                    qz = (q - q.mean()) / (q.std(ddof=0) + 1e-9)
                else:
                    qz = np.zeros_like(q)
                scores = base_scores + M.eta_quality * qz
            else:
                scores = base_scores

            # Rank and pick
            order = np.argsort(-scores)  # descending
            picked = order[:max(1, M.top_k)].tolist()
            per_metric_picks[M.name].append(picked)

            # Build OOS ensemble for this metric
            P = np.column_stack([preds_test[j].values for j in picked])
            P = z_tanh(P)  # normalize/clip per driver
            w = np.ones(P.shape[1]) / max(1, P.shape[1])
            s_oos = pd.Series(P @ w, index=X.index[test_idx]).clip(-1, 1)
            stats = pnl_stats_from_signal(s_oos, y.iloc[test_idx], costs_per_turn=costs_per_turn_backtest)

            per_metric_oos[M.name].append({k: float(stats[k]) for k in ["AnnRet","AnnVol","Sharpe","HitRate"]})

            # Predictive power: how well did the metric's validation scores predict driver OOS Sharpe?
            # (Spearman across drivers in this window)
            if np.unique(base_scores).size > 1 and np.unique(driver_oos_sharpes).size > 1:
                rho, _ = spearmanr(base_scores, driver_oos_sharpes, nan_policy="omit")
                per_metric_corrs[M.name].append(float(0.0 if rho is None or not np.isfinite(rho) else rho))

        # Update quality memory with realized driver OOS sharpes
        driver_quality = driver_quality * q_decay + driver_oos_sharpes * (1.0 - q_decay)

        # Log window row (wide format later)
        row = {"t0": int(t0), "t1": int(t0 + horizon)}
        for M in metrics:
            last = per_metric_oos[M.name][-1]
            row[f"{M.name}|AnnRet"] = last["AnnRet"]
            row[f"{M.name}|AnnVol"] = last["AnnVol"]
            row[f"{M.name}|Sharpe"] = last["Sharpe"]
            row[f"{M.name}|HitRate"] = last["HitRate"]
            # include predictive corr if present
            if len(per_metric_corrs[M.name]) > 0:
                row[f"{M.name}|PredCorr"] = per_metric_corrs[M.name][-1]
        window_rows.append(row)

        t0 += step

    # Build window-level DataFrame
    window_df = pd.DataFrame(window_rows)
    window_df.set_index(["t0","t1"], inplace=True)

    # Summaries per metric
    summary_rows = []
    details = {}
    for M in metrics:
        oos_list = per_metric_oos[M.name]
        df = pd.DataFrame(oos_list)
        summary = {
            "Metric": M.name,
            "Windows": len(df),
            "AnnRet_mean": float(df["AnnRet"].mean()),
            "AnnVol_mean": float(df["AnnVol"].mean()),
            "Sharpe_mean": float(df["Sharpe"].mean()),
            "Sharpe_median": float(df["Sharpe"].median()),
            "HitRate_mean": float(df["HitRate"].mean()),
            "PredCorr_mean": float(np.mean(per_metric_corrs[M.name])) if len(per_metric_corrs[M.name]) else np.nan,
            "PredCorr_median": float(np.median(per_metric_corrs[M.name])) if len(per_metric_corrs[M.name]) else np.nan,
        }
        summary_rows.append(summary)
        details[M.name] = dict(
            per_window_oos=df,
            predictive_corrs=np.array(per_metric_corrs[M.name], dtype=float),
            picks=per_metric_picks[M.name],
        )
    summary_df = pd.DataFrame(summary_rows).set_index("Metric")
    return summary_df, window_df, details
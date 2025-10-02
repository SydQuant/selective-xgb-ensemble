# horse_race_individual_quality.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from scipy.stats import spearmanr

# ---------- utilities ----------
def _z_tanh(x: np.ndarray) -> np.ndarray:
    eps = 1e-9
    z = (x - x.mean()) / (x.std(ddof=0) + eps)
    return np.tanh(z)

def positions_from_scores(scores: pd.Series) -> pd.Series:
    return pd.Series(_z_tanh(scores.values), index=scores.index)

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
    ann_vol = (sd * np.sqrt(ann)) if sd > 0 else 0.0
    sharpe = ((mu / sd) * np.sqrt(ann)) if sd > 0 else 0.0
    hit = float((np.sign(pos) == np.sign(rets)).mean())

    return dict(AnnRet=float(ann_ret), AnnVol=float(ann_vol), Sharpe=float(sharpe),
                HitRate=float(hit), PnL=pnl, Equity=pnl.cumsum())

# ---------- per-slice metrics ----------
def metric_sharpe(scores: pd.Series, rets: pd.Series, costs_per_turn: float = 0.0) -> float:
    pos = positions_from_scores(scores)
    return float(pnl_stats_from_signal(pos, rets, costs_per_turn=costs_per_turn)["Sharpe"])

def metric_adj_sharpe(scores: pd.Series, rets: pd.Series, lambda_to: float = 0.0) -> float:
    pos = positions_from_scores(scores)
    stats = pnl_stats_from_signal(pos, rets, costs_per_turn=0.0)
    to = pos.diff().abs().fillna(0.0).mean()
    return float(stats["Sharpe"] - lambda_to * float(to))

def metric_hit_rate(scores: pd.Series, rets: pd.Series) -> float:
    s = np.sign(scores.reindex_like(rets).values)
    r = np.sign(rets.values)
    m = (~np.isnan(s)) & (~np.isnan(r))
    return 0.0 if not np.any(m) else float(np.mean(s[m] == r[m]))

# ---------- stability score (train→val drop-off) ----------
def stability_score(m_tr: float, m_va: float,
                    alpha: float = 1.0,
                    lam_gap: float = 0.5,
                    relative_gap: bool = False) -> float:
    if relative_gap:
        denom = max(1e-9, abs(m_tr))
        gap = max(0.0, (m_tr - m_va) / denom)
    else:
        gap = max(0.0, m_tr - m_va)
    return float(alpha * m_va - lam_gap * gap)

# ---------- minimal driver wrapper ----------
class Driver:
    """
    Wrap your XGBRegressor (or similar):
      .fit(X,y) -> self
      .predict(X) -> np.ndarray (n,)
    """
    def __init__(self, model):
        self.model = model
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X.values, y.values)
        return self
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

# ---------- metric config (with quality momentum) ----------
@dataclass
class MetricConfig:
    name: str
    fn: Callable[..., float]                 # metric(scores, rets, **kwargs)
    kwargs: Dict[str, float] | None = None  # e.g., {"costs_per_turn": 0.0} or {"lambda_to": 0.1}
    alpha: float = 1.0                      # weight on validation metric
    lam_gap: float = 0.5                    # penalty on train→val drop
    relative_gap: bool = False              # use relative gap
    eta_quality: float = 0.0                # weight of prior OOS quality (EWMA of driver OOS Sharpe)

# ---------- main: rolling horse race, per-metric picks (no combining) ----------
def rolling_horse_race_individual_quality(
    X: pd.DataFrame,
    y: pd.Series,
    drivers: List[Driver],
    metrics: List[MetricConfig],
    start_train: int = 750,
    step: int = 21,
    horizon: int = 21,
    inner_val_frac: float = 0.2,
    costs_per_turn_backtest: float = 0.0,
    quality_halflife: int = 63,   # ~3 months if step≈1d
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict]]:
    """
    For each metric (Sharpe / AdjSharpe / HitRate …):
      - Fit all drivers on the training slice
      - Compute per-driver stability score on inner train/val
      - Optionally add prior OOS "quality momentum" (EWMA of realized OOS Sharpe)
      - Pick the single best driver for that metric
      - Record that driver's realized OOS stats (AnnRet, AnnVol, Sharpe, HitRate)
      - Record predictive power: Spearman corr between (validation stability + quality) scores
        and the drivers' realized OOS Sharpe (same window)
    Returns:
      summary_df (per metric), window_df (per window × metric), details dict
    """
    n = len(X); assert n == len(y)

    # Quality momentum memory per driver (EWMA of realized OOS Sharpe)
    driver_quality = np.zeros(len(drivers), dtype=float)
    q_decay = np.exp(np.log(0.5) / max(1, quality_halflife))

    per_metric_oos: Dict[str, List[Dict[str, float]]] = {m.name: [] for m in metrics}
    per_metric_predcorrs: Dict[str, List[float]] = {m.name: [] for m in metrics}
    per_metric_picks: Dict[str, List[int]] = {m.name: [] for m in metrics}
    window_rows = []

    t0 = start_train
    while t0 + horizon <= n:
        tr_full = np.arange(0, t0)
        cut = int(len(tr_full) * (1.0 - inner_val_frac))
        inner_tr, inner_va = tr_full[:cut], tr_full[cut:]
        test_idx = np.arange(t0, t0 + horizon)

        # Fit all drivers; cache predictions
        preds_tr, preds_va, preds_te = [], [], []
        for drv in drivers:
            m = drv.fit(X.iloc[tr_full], y.iloc[tr_full])
            preds_tr.append(pd.Series(m.predict(X.iloc[inner_tr]), index=X.index[inner_tr]))
            preds_va.append(pd.Series(m.predict(X.iloc[inner_va]), index=X.index[inner_va]))
            preds_te.append(pd.Series(m.predict(X.iloc[test_idx]), index=X.index[test_idx]))

        # Realized OOS Sharpe per driver (for predictive-corr and quality update)
        driver_oos_sharpe = np.zeros(len(drivers), dtype=float)
        for j in range(len(drivers)):
            pos_te = positions_from_scores(preds_te[j])
            stats = pnl_stats_from_signal(pos_te, y.iloc[test_idx], costs_per_turn=costs_per_turn_backtest)
            driver_oos_sharpe[j] = float(stats["Sharpe"])

        row = {"t0": int(t0), "t1": int(t0 + horizon)}

        for M in metrics:
            # base stability scores from current inner train/val
            base_scores = np.zeros(len(drivers), dtype=float)
            for j in range(len(drivers)):
                kwargs = (M.kwargs or {})
                m_tr = M.fn(preds_tr[j], y.iloc[inner_tr], **kwargs)
                m_va = M.fn(preds_va[j], y.iloc[inner_va], **kwargs)
                base_scores[j] = stability_score(m_tr, m_va, alpha=M.alpha,
                                                 lam_gap=M.lam_gap, relative_gap=M.relative_gap)

            # blend with quality momentum if requested
            if M.eta_quality > 0.0:
                q = driver_quality.copy()
                # z-normalize quality across drivers to keep it scale-free
                qz = (q - q.mean()) / (q.std(ddof=0) + 1e-9) if np.isfinite(q).any() else np.zeros_like(q)
                scores = base_scores + M.eta_quality * qz
            else:
                scores = base_scores

            # pick single best driver for this metric
            j_best = int(np.argmax(scores))
            per_metric_picks[M.name].append(j_best)

            # realized OOS stats for that pick
            pos_best = positions_from_scores(preds_te[j_best])
            stats_best = pnl_stats_from_signal(pos_best, y.iloc[test_idx], costs_per_turn=costs_per_turn_backtest)
            per_metric_oos[M.name].append({
                "AnnRet": float(stats_best["AnnRet"]),
                "AnnVol": float(stats_best["AnnVol"]),
                "Sharpe": float(stats_best["Sharpe"]),
                "HitRate": float(stats_best["HitRate"]),
            })

            # predictive power: do selection scores align with realized OOS Sharpe across drivers?
            predcorr = np.nan
            if np.unique(scores).size > 1 and np.unique(driver_oos_sharpe).size > 1:
                rho, _ = spearmanr(scores, driver_oos_sharpe, nan_policy="omit")
                predcorr = float(0.0 if rho is None or not np.isfinite(rho) else rho)
            per_metric_predcorrs[M.name].append(predcorr)

            # add to window row
            for k in ("AnnRet","AnnVol","Sharpe","HitRate"):
                row[f"{M.name}|{k}"] = per_metric_oos[M.name][-1][k]
            row[f"{M.name}|PredCorr"] = predcorr

        # update quality memory from realized OOS Sharpe this window
        driver_quality = driver_quality * q_decay + driver_oos_sharpe * (1.0 - q_decay)

        window_rows.append(row)
        t0 += step

    # per-window dataframe
    window_df = pd.DataFrame(window_rows).set_index(["t0","t1"])

    # summaries
    summary_rows, details = [], {}
    for M in metrics:
        df = pd.DataFrame(per_metric_oos[M.name])
        predcorrs = np.array(per_metric_predcorrs[M.name], dtype=float)
        summary_rows.append({
            "Metric": M.name,
            "Windows": int(len(df)),
            "AnnRet_mean": float(df["AnnRet"].mean()) if len(df) else np.nan,
            "AnnVol_mean": float(df["AnnVol"].mean()) if len(df) else np.nan,
            "Sharpe_mean": float(df["Sharpe"].mean()) if len(df) else np.nan,
            "Sharpe_median": float(df["Sharpe"].median()) if len(df) else np.nan,
            "HitRate_mean": float(df["HitRate"].mean()) if len(df) else np.nan,
            "PredCorr_mean": float(np.nanmean(predcorrs)) if predcorrs.size else np.nan,
            "PredCorr_median": float(np.nanmedian(predcorrs)) if predcorrs.size else np.nan,
        })
        details[M.name] = dict(
            picks=per_metric_picks[M.name],
            per_window_oos=pd.DataFrame(per_metric_oos[M.name]),
            predictive_corrs=predcorrs,
        )
    summary_df = pd.DataFrame(summary_rows).set_index("Metric")
    return summary_df, window_df, details
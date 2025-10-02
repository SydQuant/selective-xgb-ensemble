
import argparse, numpy as np, pandas as pd, os, json
from cv.wfo import wfo_splits
from model.xgb_drivers import generate_xgb_specs, fold_train_predict
from ensemble.combiner import build_driver_signals, combine_signals, softmax
from ensemble.selection import pick_top_n_greedy_diverse
from opt.grope import grope_optimize
from opt.weight_objective import weight_objective_factory
from metrics.dapy import hit_rate
from metrics.perf import information_ratio
from eval.target_shuffling import shuffle_pvalue

def get_dapy_fn(style: str):
    style = (style or "hits").lower()
    if style == "hits":
        from metrics.dapy import dapy_from_binary_hits as fn
        return fn
    elif style == "eri_long":
        from metrics.dapy_eri import dapy_eri_long as fn
        return fn
    elif style == "eri_short":
        from metrics.dapy_eri import dapy_eri_short as fn
        return fn
    elif style == "eri_both":
        from metrics.dapy_eri import dapy_eri_both as fn
        return fn
    else:
        from metrics.dapy import dapy_from_binary_hits as fn
        return fn

def save_timeseries(path: str, signal: pd.Series, y: pd.Series):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pnl = (signal.shift(1).fillna(0.0) * y.reindex_like(signal)).astype(float)
    eq = pnl.cumsum()
    out = pd.DataFrame({'signal': signal, 'target_ret': y.reindex_like(signal), 'pnl': pnl, 'equity': eq})
    out.to_csv(path)
    return out

def save_oos_artifacts(signal: pd.Series, y: pd.Series, block: int, final_shuffles: int, dapy_fn):
    os.makedirs("artifacts", exist_ok=True)
    out = save_timeseries("artifacts/oos_timeseries.csv", signal, y)
    pval, obs, _ = shuffle_pvalue(signal, y, dapy_fn, n_shuffles=final_shuffles, block=block)
    print(f"OOS Shuffling p-value (DAPY): {pval:.3f} (obs={obs:.2f})")
    return out

def make_synth(n=2400, n_features=600, seed=0, signal_frac=0.04):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-01", periods=n, freq="B")
    X = pd.DataFrame(rng.normal(size=(n, n_features)), index=idx, columns=[f"f{i}" for i in range(n_features)])
    k = max(4, int(n_features*signal_frac))
    cols = rng.choice(n_features, size=k, replace=False)
    w = rng.normal(scale=0.25, size=k)
    y = (X.iloc[:, cols] @ w)
    y = y.shift(-1) + 0.6*rng.normal(size=n)
    y = (y - y.mean()) / (y.std() + 1e-9)
    return X, y.rename("target")

def main(args):
    dapy_fn = get_dapy_fn(args.dapy_style)
    if args.synthetic:
        X, y = make_synth(n=args.n_obs, n_features=args.n_features, seed=42)
    else:
        raise SystemExit("Replace make_synth with your real pipeline.")

    splits = wfo_splits(len(X), k_folds=args.folds, min_train=max(252, len(X)//(args.folds*2)))
    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    os.makedirs("artifacts", exist_ok=True)

    oos_signal = pd.Series(0.0, index=X.index)
    fold_summaries = []

    for f, (tr, te) in enumerate(splits):
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        X_te, y_te = X.iloc[te], y.iloc[te]

        train_preds, test_preds = fold_train_predict(X_tr, y_tr, X_te, specs)
        s_tr, s_te = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

        def gate(sig, y_local):
            pval, _, _ = shuffle_pvalue(sig, y_local, dapy_fn, n_shuffles=200, block=args.block)
            return pval <= args.pmax

        chosen_idx = pick_top_n_greedy_diverse(
            s_tr, y_tr, n=args.n_select, pval_gate=gate,
            w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn
        )
        if len(chosen_idx) == 0:
            print(f"[Fold {f}] No drivers passed the p-value gate; ensemble is zero.")
            continue
        train_sel = [s_tr[i] for i in chosen_idx]
        test_sel  = [s_te[i] for i in chosen_idx]

        bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
        fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
        theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=1234+f)

        w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
        tau = float(theta_star["tau"])
        ww = softmax(w, temperature=tau)
        s_fold = combine_signals(test_sel, ww)
        oos_signal.iloc[te] = s_fold

        fold_summaries.append({"fold": f, "chosen_idx": chosen_idx, "weights": ww.tolist(), "tau": tau, "J_train": J_star})

    # Final OOS metrics
    dapy_val = dapy_fn(oos_signal, y)
    ir = information_ratio(oos_signal, y)
    hr = hit_rate(oos_signal, y)
    print(f"OOS DAPY({args.dapy_style}): {dapy_val:.2f} | OOS IR: {ir:.2f} | OOS hit-rate: {hr:.3f}")
    save_oos_artifacts(oos_signal, y, block=args.block, final_shuffles=args.final_shuffles, dapy_fn=dapy_fn)

    with open("artifacts/fold_summaries.json", "w", encoding="utf-8") as fsum:
        json.dump(fold_summaries, fsum, indent=2)

    # Production model (unrealistic backtest)
    if args.train_production:
        print("\n[Production] Training production model on all data (for deployment only, not for backtest)...")
        train_production_model(X, y, args)

    # Random past-train then test from test_start
    if args.random_train_pct > 0.0:
        print(f"\n[RandTrain] Training on {args.random_train_pct*100:.1f}% of pre-{args.test_start} data, then backtesting from {args.test_start}...")
        random_past_train_then_backtest(X, y, args)

    # Random IN-PERIOD train then full in-period backtest
    if args.random_inperiod_train_pct > 0.0:
        print(f"\n[RandInPeriod] Training on {args.random_inperiod_train_pct*100:.1f}% of data since {args.test_start}, then backtesting on the full period...")
        random_inperiod_train_then_backtest(X, y, args)

def train_production_model(X: pd.DataFrame, y: pd.Series, args):
    import numpy as np
    dapy_fn = get_dapy_fn(args.dapy_style)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    train_preds, test_preds = fold_train_predict(X, y, X, specs)  # predict on full
    s_tr, s_full = build_driver_signals(train_preds, test_preds, y, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        pval, _, _ = shuffle_pvalue(sig, yy, dapy_fn, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx = pick_top_n_greedy_diverse(s_tr, y, n=args.n_select, pval_gate=gate, w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn)
    if len(chosen_idx) == 0:
        print("[Production] No drivers passed p-value gate.")
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    full_sel  = [s_full[i] for i in chosen_idx]

    bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
    fobj = weight_objective_factory(train_sel, y, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
    theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=5678)

    w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
    tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)
    s_full_ens = combine_signals(full_sel, ww)

    save_timeseries("artifacts/production_timeseries.csv", s_full_ens, y)

    prod = {
        "chosen_driver_indices": [int(i) for i in chosen_idx],
        "weights_softmax": ww.tolist(), "temperature": float(tau),
        "n_models_total": args.n_models, "n_select": args.n_select,
        "z_win": args.z_win, "beta_pre": args.beta_pre, "lambda_to": args.lambda_to,
        "w_dapy": args.w_dapy, "w_ir": args.w_ir, "pmax": args.pmax, "weight_budget": args.weight_budget,
        "diversity_penalty": args.diversity_penalty, "dapy_style": args.dapy_style
    }
    with open("artifacts/production_model.json", "w", encoding="utf-8") as f:
        json.dump(prod, f, indent=2)
    print("[Production] Saved artifacts/production_model.json")

def random_past_train_then_backtest(X: pd.DataFrame, y: pd.Series, args):
    dapy_fn = get_dapy_fn(args.dapy_style)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    import numpy as np, pandas as pd
    os.makedirs("artifacts", exist_ok=True)

    ts = pd.Timestamp(args.test_start)
    mask_train_time = X.index < ts
    if mask_train_time.sum() < 50:
        print("[RandTrain] Not enough pre-test data.")
        return
    rng = np.random.default_rng(args.randtrain_seed)
    idx_pre = np.where(mask_train_time)[0]
    n_sample = max(10, int(len(idx_pre) * args.random_train_pct))
    idx_train = np.sort(rng.choice(idx_pre, size=n_sample, replace=False))

    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    train_preds, test_preds = fold_train_predict(X_tr, y_tr, X, specs)
    s_tr, s_full = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        pval, _, _ = shuffle_pvalue(sig, yy, dapy_fn, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx = pick_top_n_greedy_diverse(s_tr, y_tr, n=args.n_select, pval_gate=gate, w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn)
    if len(chosen_idx) == 0:
        print("[RandTrain] No drivers passed p-value gate.")
        return
    train_sel = [s_tr[i] for i in chosen_idx]
    full_sel  = [s_full[i] for i in chosen_idx]

    bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
    fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
    theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=2468)

    w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
    tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)

    s_full_ens = combine_signals(full_sel, ww)
    s_test = s_full_ens[s_full_ens.index >= ts]; y_test = y[y.index >= ts]
    save_timeseries("artifacts/randomtrain_timeseries.csv", s_test, y_test)
    print("[RandTrain] Saved artifacts/randomtrain_timeseries.csv and metadata.")

def random_inperiod_train_then_backtest(X: pd.DataFrame, y: pd.Series, args):
    dapy_fn = get_dapy_fn(args.dapy_style)
    from model.xgb_drivers import generate_xgb_specs, fold_train_predict
    from ensemble.combiner import build_driver_signals, softmax, combine_signals
    from ensemble.selection import pick_top_n_greedy_diverse
    from opt.grope import grope_optimize
    from opt.weight_objective import weight_objective_factory
    from eval.target_shuffling import shuffle_pvalue

    import numpy as np, pandas as pd
    os.makedirs("artifacts", exist_ok=True)

    ts = pd.Timestamp(args.test_start)
    mask_inperiod = X.index >= ts
    if mask_inperiod.sum() < 100:
        print("[RandInPeriod] Not enough in-period data.")
        return

    rng = np.random.default_rng(args.randtrain_seed)
    idx_all = np.where(mask_inperiod)[0]
    n_sample = max(50, int(len(idx_all) * args.random_inperiod_train_pct))
    idx_train = np.sort(rng.choice(idx_all, size=n_sample, replace=False))

    X_tr, y_tr = X.iloc[idx_train], y.iloc[idx_train]
    specs = generate_xgb_specs(n_models=args.n_models, seed=7)
    train_preds, test_preds = fold_train_predict(X_tr, y_tr, X[mask_inperiod], specs)
    s_tr, s_full = build_driver_signals(train_preds, test_preds, y_tr, z_win=args.z_win, beta=args.beta_pre)

    def gate(sig, yy):
        pval, _, _ = shuffle_pvalue(sig, yy, dapy_fn, n_shuffles=200, block=args.block)
        return pval <= args.pmax

    chosen_idx = pick_top_n_greedy_diverse(s_tr, y_tr, n=args.n_select, pval_gate=gate, w_dapy=args.w_dapy, w_ir=args.w_ir, diversity_penalty=args.diversity_penalty, dapy_fn=dapy_fn)
    if len(chosen_idx) == 0:
        print("[RandInPeriod] No drivers passed p-value gate.")
        return

    train_sel = [s_tr[i] for i in chosen_idx]
    full_sel  = [s_full[i] for i in chosen_idx]
    bounds = {**{f"w{i}": (-2.0, 2.0) for i in range(len(train_sel))}, "tau": (0.2, 3.0)}
    fobj = weight_objective_factory(train_sel, y_tr, turnover_penalty=args.lambda_to, pmax=args.pmax, w_dapy=args.w_dapy, w_ir=args.w_ir, metric_fn_dapy=dapy_fn)
    theta_star, J_star, _ = grope_optimize(bounds, fobj, budget=args.weight_budget, seed=97531)

    w = np.array([theta_star[f"w{i}"] for i in range(len(train_sel))], dtype=float)
    tau = float(theta_star["tau"]); ww = softmax(w, temperature=tau)

    s_full_ens = combine_signals(full_sel, ww)
    y_full = y[mask_inperiod]
    save_timeseries("artifacts/random_inperiod_timeseries.csv", s_full_ens, y_full)
    print("[RandInPeriod] Saved artifacts/random_inperiod_timeseries.csv and metadata.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", default=True)
    ap.add_argument("--n_obs", type=int, default=2400)
    ap.add_argument("--n_features", type=int, default=600)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--n_models", type=int, default=50)
    ap.add_argument("--n_select", type=int, default=12)
    ap.add_argument("--z_win", type=int, default=100)
    ap.add_argument("--beta_pre", type=float, default=1.0)
    ap.add_argument("--lambda_to", type=float, default=0.05)
    ap.add_argument("--weight_budget", type=int, default=80)
    ap.add_argument("--w_dapy", type=float, default=1.0)
    ap.add_argument("--w_ir", type=float, default=1.0)
    ap.add_argument("--diversity_penalty", type=float, default=0.2)
    ap.add_argument("--pmax", type=float, default=0.20)
    ap.add_argument("--final_shuffles", type=int, default=600)
    ap.add_argument("--block", type=int, default=10)
    ap.add_argument("--train_production", action="store_true", default=False)
    ap.add_argument("--dapy_style", type=str, default="hits", help="hits | eri_long | eri_short | eri_both")
    ap.add_argument("--test_start", type=str, default="2018-01-01")
    ap.add_argument("--random_train_pct", type=float, default=0.0)
    ap.add_argument("--randtrain_seed", type=int, default=123)
    ap.add_argument("--random_inperiod_train_pct", type=float, default=0.0)
    args = ap.parse_args()
    main(args)

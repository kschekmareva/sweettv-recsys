"""Learned reranker over the union of top-N candidates of several scorers (LightGBM).

Features (per user, show) come from the TRAIN window only (plus the target-period schedule for what airs when).
"""
import time

import numpy as np
import pandas as pd

import hybrid
import metrics
import run_step4 as s4

ORDER = ["H2", "EW", "PPR", "MF", "AR", "P-air"]
TOPN = 50


def candidates(store, pop_norm, n_u, n_p, names=ORDER, topn=TOPN):
    cand = np.zeros((n_u, n_p), dtype=bool)
    rows = np.arange(n_u)[:, None]
    for name in names:
        if name == "P-air":
            cand[:, np.argsort(-pop_norm, kind="stable")[:topn]] = True
        else:
            cand[rows, hybrid.topk(s4.fuse_eps(pop_norm, [store[name]], [1.0]), topn)] = True
    return cand


def make_features(ctx, store, pop_norm, cand, order=ORDER):
    t0 = time.time()
    n_u, n_p = ctx["n_u"], ctx["n_p"]
    ri, ci = np.nonzero(cand)
    N = len(ri)
    F = {}
    # --- component scores and ranks
    bounds = lambda a, b: (np.searchsorted(ri, a), np.searchsorted(ri, b))
    for name in order:
        if name == "P-air":
            rk = np.empty(n_p, dtype=np.int32)
            rk[np.argsort(-pop_norm, kind="stable")] = np.arange(n_p)
            F["s_P-air"], F["r_P-air"] = pop_norm[ci].astype(np.float32), rk[ci]
            continue
        Sn, valid = store[name]
        F["s_" + name], F["v_" + name] = Sn[ri, ci], valid[ri].astype(np.float32)
        rk = np.empty(N, dtype=np.int32)
        for a in range(0, n_u, 1000):
            sl = slice(a, min(a + 1000, n_u))
            S = s4.fuse_eps(pop_norm, [(Sn[sl], valid[sl])], [1.0])
            ranks = np.empty(S.shape, dtype=np.int32)
            np.put_along_axis(ranks, np.argsort(-S, axis=1, kind="stable"), np.broadcast_to(np.arange(n_p, dtype=np.int32), S.shape), axis=1)
            lo, hi = bounds(a, a + 1000)
            rk[lo:hi] = ranks[ri[lo:hi] - a, ci[lo:hi]]
        F["r_" + name] = rk
    # --- user-level and user-show-level features from the TRAIN airing table
    air, a_ri, a_ci = ctx["air"], ctx["air_ri"], ctx["air_ci"]
    ok = a_ri >= 0
    air, a_ri, a_ci, a_cat = air[ok], a_ri[ok], a_ci[ok], ctx["air_cat"][ok]
    done = air["view_pct"].values >= 0.8
    ov, start = air["ov"].values.astype(np.float64), air["start"].values.astype(np.float64)
    df = pd.DataFrame({"ri": a_ri, "done": done, "ov": ov, "start": start})
    u = df.groupby("ri").agg(n=("done", "size"), n_done=("done", "sum"), ov=("ov", "sum"), last=("start", "max")).reindex(range(n_u))
    F["u_completed"] = u["n_done"].fillna(0).values[ri].astype(np.float32)
    F["u_completion_rate"] = (u["n_done"] / u["n"]).fillna(0).values[ri].astype(np.float32)
    F["u_ov_sec"] = u["ov"].fillna(0).values[ri].astype(np.float32)
    F["u_days_since_last"] = ((ctx["train_end"] - u["last"]).fillna(999 * 86400) / 86400.0).values[ri].astype(np.float32)
    u_tot = u["ov"].fillna(0).values
    pair_key = ri.astype(np.int64) * n_p + ci
    sel = a_ci >= 0
    us = pd.DataFrame({"k": a_ri[sel].astype(np.int64) * n_p + a_ci[sel], "done": done[sel], "ov": ov[sel], "start": start[sel]}) \
        .groupby("k").agg(n_done=("done", "sum"), ov=("ov", "sum"), last=("start", "max"))
    g = us.reindex(pair_key)
    F["us_completed"] = g["n_done"].fillna(0).values.astype(np.float32)
    F["us_ov_sec"] = g["ov"].fillna(0).values.astype(np.float32)
    F["us_days_since"] = ((ctx["train_end"] - g["last"]).fillna(999 * 86400) / 86400.0).values.astype(np.float32)

    def share(keys, vals, pk, tot_by_row):
        s = pd.Series(vals).groupby(keys).sum()
        return (s.reindex(pk).fillna(0).values / np.maximum(tot_by_row, 1.0)).astype(np.float32)
    tot = np.maximum(u_tot[ri], 1.0)
    mc = ctx["main_ch"][ci]
    F["u_share_show_channel"] = share(a_ri.astype(np.int64) * 2048 + air["channel_id"].values, ov, np.where(mc >= 0, ri * 2048 + mc, -1), tot)
    sc = ctx["cat_code"][ci]
    cm = a_cat >= 0
    F["u_share_show_category"] = share((a_ri[cm].astype(np.int64) * 32 + a_cat[cm]), ov[cm], np.where(sc >= 0, ri * 32 + sc, -1), tot)
    hour = (air["start"].values % 86400) // 3600
    hs = pd.Series(ov).groupby(a_ri.astype(np.int64) * 24 + hour).sum().rename("ov").reset_index()
    hs["ri"], hs["h"] = hs["index"] // 24, hs["index"] % 24
    hs["rk"] = hs.groupby("ri")["ov"].rank(ascending=False, method="first")
    top3 = set((hs.loc[hs["rk"] <= 3, "index"]).tolist())
    th = ctx["typ_hour"][ci]
    pk = np.where(th >= 0, ri * 24 + th, -1)
    F["hour_in_user_top3"] = np.array([k in top3 for k in pk], dtype=np.float32)
    F["u_share_show_hour"] = share(a_ri.astype(np.int64) * 24 + hour, ov, pk, tot)
    # --- show-level features
    F["show_pop"], F["show_n_train_air"], F["show_n_target_air"] = ctx["P"][ci], ctx["n_air_tr"][ci], ctx["n_target"][ci]
    F["show_duration"] = np.nan_to_num(ctx["med_dur"][ci], nan=-1.0).astype(np.float32)
    F["show_category"] = sc.astype(np.int32)
    X = pd.DataFrame(F)
    return ri, ci, X, time.time() - t0


def top5(model, X, ri, ci, rows):
    """Top-5 pool columns per user in `rows` (sorted ascending), ranked by the model score."""
    m = np.isin(ri, rows)
    sc = model.predict(X[m])
    r, c = ri[m], ci[m]
    o = np.lexsort((-sc, r))
    r, c = r[o], c[o]
    first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    assert len(first) == len(rows) and (np.diff(np.r_[first, len(r)]) >= 5).all()
    return np.stack([c[first + j] for j in range(5)], axis=1)


def fit(X, y, ri, ci, tune_rows_with_truth, T5, seed=42, rounds=800):
    """Fit on TUNE users only. 20% of them (random, seed) are held out for early stopping; the objective
    (lambdarank or binary) is chosen by MAP@5 (VTOP5) on that held-out 20%. Returns (best, all results, seconds)."""
    import lightgbm as lgb
    t0 = time.time()
    perm = np.random.default_rng(seed).permutation(tune_rows_with_truth)
    n_va = int(round(0.2 * len(perm)))
    va_rows, tr_rows = np.sort(perm[:n_va]), np.sort(perm[n_va:])
    m_tr, m_va = np.isin(ri, tr_rows), np.isin(ri, va_rows)
    grp = lambda m: np.unique(ri[m], return_counts=True)[1]
    results = []
    for obj in ("lambdarank", "binary"):
        p = dict(objective=obj, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.8, bagging_fraction=0.8,
                 bagging_freq=1, lambda_l2=1.0, seed=seed, verbose=-1, num_threads=4, deterministic=True, force_row_wise=True)
        p.update(dict(metric="ndcg", eval_at=[5]) if obj == "lambdarank" else dict(metric="binary_logloss"))
        rank = obj == "lambdarank"
        d_tr = lgb.Dataset(X[m_tr], y[m_tr], group=grp(m_tr) if rank else None, categorical_feature=["show_category"])
        d_va = lgb.Dataset(X[m_va], y[m_va], group=grp(m_va) if rank else None, categorical_feature=["show_category"], reference=d_tr)
        model = lgb.train(p, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=[lgb.early_stopping(30, verbose=False)])
        map_va = float(metrics.ap5_matrix(top5(model, X, ri, ci, va_rows), T5[va_rows]).mean())
        imp = pd.Series(model.feature_importance("gain"), index=X.columns)
        results.append({"objective": obj, "best_iteration": int(model.best_iteration), "holdout_map5_vtop5": map_va, "model": model,
                        "top_importance": (imp / imp.sum()).sort_values(ascending=False).head(10).round(4).to_dict(), "params": p})
    best = max(results, key=lambda r: r["holdout_map5_vtop5"])
    return best, results, {"train_users": len(tr_rows), "holdout_users": len(va_rows), "train_pairs": int(m_tr.sum())}, time.time() - t0

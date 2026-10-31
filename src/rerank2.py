"""Step-8 additions to the reranker: CH-SLOT candidates, extra features, per-week scaling, LightGBM helpers.

Everything is computed from the TRAIN (feature) window; the target-period schedule is used only to know what airs when.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import data
import hybrid
import metrics
import rerank
import run_step4 as s4

COUNT_COLS = ["u_completed", "u_ov_sec", "us_completed", "us_ov_sec", "show_pop", "show_n_train_air"]   # per feature-window week
TARGET_COLS = ["show_n_target_air"]                                                                   # per target-window week
BASE_PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, seed=42, verbose=-1, num_threads=4,
                   deterministic=True, force_row_wise=True)


def add_target_rows(ctx, target_key, tw):
    s = data.load_schedule(target_key)
    m = (s["tv_show_id"] != 0) & s["iso_week"].between(*tw) & (s["iso_year"] == 2020)
    t = s.loc[m].copy()
    t["pc"] = ctx["pix"].get_indexer(t["tv_show_id"].values)
    t = t[t["pc"] >= 0]
    t["hour"], t["dow"] = (t["start"] % 86400) // 3600, ((t["start"] // 86400) + 2) % 7     # 2020-01-01 is a Wednesday
    ctx["target_rows"] = t[["channel_id", "pc", "hour", "dow"]]


def slot_counts(ctx):
    """CH-SLOT: for every (user, show) the number of target-period airings on the user's top-3 channels (by TRAIN seconds)
    that start in the user's top-3 viewing hours (hours weighted by TRAIN seconds)."""
    t0 = time.time()
    n_u, n_p = ctx["n_u"], ctx["n_p"]
    air, ok = ctx["air"], ctx["air_ri"] >= 0
    a_ri = ctx["air_ri"][ok]
    ch = air["channel_id"].values[ok].astype(np.int64)
    ov = air["ov"].values[ok].astype(np.float64)
    hr = ((air["start"].values[ok] % 86400) // 3600).astype(np.int64)

    def top3(keys):
        g = pd.DataFrame({"ri": a_ri, "k": keys, "ov": ov}).groupby(["ri", "k"], as_index=False)["ov"].sum()
        g["rk"] = g.groupby("ri")["ov"].rank(ascending=False, method="first")
        return g.loc[g["rk"] <= 3, ["ri", "k"]]
    cells = top3(ch).merge(top3(hr), on="ri", suffixes=("_c", "_h"))
    cells["cell"] = cells["k_c"] * 24 + cells["k_h"]
    t = ctx["target_rows"]
    tcell = t["channel_id"].values.astype(np.int64) * 24 + t["hour"].values
    uniq = pd.Index(np.unique(tcell))
    CH = csr_matrix((np.ones(len(t), dtype=np.float32), (uniq.get_indexer(tcell), t["pc"].values)), shape=(len(uniq), n_p))
    cc = uniq.get_indexer(cells["cell"].values)
    okc = cc >= 0
    UCell = csr_matrix((np.ones(okc.sum(), dtype=np.float32), (cells["ri"].values[okc], cc[okc])), shape=(n_u, len(uniq)))
    return (UCell @ CH).toarray().astype(np.float32), time.time() - t0


def slot_candidates(counts, pop_norm, topn=30):
    top = hybrid.topk(counts + np.float32(1e-3) * pop_norm[None, :], topn)       # ties: popularity
    rows = np.arange(len(counts))[:, None]
    cand = np.zeros(counts.shape, dtype=bool)
    cand[rows, top] = counts[rows, top] > 0
    return cand


def extras(ctx, ri, ci, counts, pop_norm):
    """Extra columns for the pairs (ri, ci): CH-SLOT score/rank, watched-in-TRAIN flag and the step-8 features."""
    t0 = time.time()
    n_u, n_p, N = ctx["n_u"], ctx["n_p"], len(ri)
    F = {"slot_score": counts[ri, ci]}
    sc = counts + np.float32(1e-3) * pop_norm[None, :]
    rk = np.empty(N, dtype=np.int32)
    for a in range(0, n_u, 1000):
        S = sc[a:a + 1000]
        ranks = np.empty(S.shape, dtype=np.int32)
        np.put_along_axis(ranks, np.argsort(-S, axis=1, kind="stable"), np.arange(n_p, dtype=np.int32), axis=1)
        lo, hi = np.searchsorted(ri, a), np.searchsorted(ri, a + 1000)
        rk[lo:hi] = ranks[ri[lo:hi] - a, ci[lo:hi]]
    F["slot_rank"] = rk
    F["watched_in_train"] = s4.bool_matrix(ctx["train_all"], ctx["uix"], ctx["pix"], n_u, n_p)[ri, ci].astype(np.float32)
    air, a_ri, a_ci = ctx["air"], ctx["air_ri"], ctx["air_ci"]
    ok = a_ri >= 0
    air, a_ri, a_ci = air[ok], a_ri[ok], a_ci[ok]
    done = air["view_pct"].values >= 0.8
    start = air["start"].values.astype(np.float64)
    ch = air["channel_id"].values.astype(np.int64)
    ov = air["ov"].values.astype(np.float64)
    # user's completion rate on the show's (main target) channel
    g = pd.DataFrame({"k": a_ri.astype(np.int64) * 2048 + ch, "d": done}).groupby("k")["d"].mean()
    mc = ctx["main_ch"][ci]
    F["u_comp_rate_show_channel"] = g.reindex(np.where(mc >= 0, ri.astype(np.int64) * 2048 + mc, -1)).fillna(0).values.astype(np.float32)
    # share of the user's completed airings in the last 4 weeks of the feature window
    rec = done & (start >= ctx["train_end"] - 28 * 86400)
    u = pd.DataFrame({"ri": a_ri, "d": done, "r": rec}).groupby("ri").agg(d=("d", "sum"), r=("r", "sum")).reindex(range(n_u))
    F["u_recent_completed_share"] = (u["r"] / u["d"].replace(0, np.nan)).fillna(0).values[ri].astype(np.float32)
    # show's completion rate across all users
    sel = a_ci >= 0
    sg = pd.Series(done[sel]).groupby(a_ci[sel]).mean().reindex(range(n_p)).fillna(0).values
    F["show_completion_rate"] = sg[ci].astype(np.float32)
    # weekdays the show airs vs the user's active weekdays (histogram intersection)
    t = ctx["target_rows"]
    sd = np.bincount(t["pc"].values * 7 + t["dow"].values, minlength=n_p * 7).reshape(n_p, 7).astype(np.float64)
    sd /= np.maximum(sd.sum(axis=1, keepdims=True), 1.0)
    dow = ((start // 86400).astype(np.int64) + 2) % 7
    ud = np.bincount(a_ri.astype(np.int64) * 7 + dow, weights=ov, minlength=n_u * 7).reshape(n_u, 7)
    ud /= np.maximum(ud.sum(axis=1, keepdims=True), 1.0)
    F["dow_overlap"] = np.minimum(sd[ci], ud[ri]).sum(axis=1).astype(np.float32)
    return pd.DataFrame(F), time.time() - t0


def per_week(X, weeks_feat, weeks_target):
    X = X.copy()
    for c in COUNT_COLS:
        X[c] = X[c] / weeks_feat
    for c in TARGET_COLS:
        X[c] = X[c] / weeks_target
    return X


def assemble(Xb, E, level, weeks_feat, weeks_target):
    """level: raw | pw | pw_slot | pw_slot_extra."""
    if level == "raw":
        return Xb
    X = per_week(Xb, weeks_feat, weeks_target)
    if level == "pw":
        return X
    X["slot_score"] = E["slot_score"] / weeks_target
    X["slot_rank"], X["watched_in_train"] = E["slot_rank"], E["watched_in_train"]
    if level == "pw_slot":
        return X
    for c in ("u_comp_rate_show_channel", "u_recent_completed_share", "show_completion_rate", "dow_overlap"):
        X[c] = E[c]
    return X


def train(X, y, ri, tr_rows, va_rows, params, rounds=800, es=30):
    """LightGBM lambdarank. Early stopping on va_rows (a held-out part of the training users); va_rows=None -> fixed rounds."""
    import lightgbm as lgb
    m_tr = np.isin(ri, tr_rows)
    grp = lambda m: np.unique(ri[m], return_counts=True)[1]
    d_tr = lgb.Dataset(X[m_tr], y[m_tr], group=grp(m_tr), categorical_feature=["show_category"])
    if va_rows is None:
        return lgb.train(params, d_tr, num_boost_round=rounds)
    m_va = np.isin(ri, va_rows)
    d_va = lgb.Dataset(X[m_va], y[m_va], group=grp(m_va), categorical_feature=["show_category"], reference=d_tr)
    return lgb.train(params, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=[lgb.early_stopping(es, verbose=False)])


def importance(model, cols, top=15):
    imp = pd.Series(model.feature_importance("gain"), index=cols)
    return (imp / imp.sum())

"""Step 18: temporal augmentation rows for the reranker.

Extra training period: features and candidates from weeks 11-15 only (all components fitted on weeks 11-15), labels = VTOP5 of weeks 16-20,
candidate pool = programmes airing in weeks 16-20, features per week (5 feature weeks, 5 target weeks). Only weeks <= 20 are used.
"""
import numpy as np
import pandas as pd

import data
import h2
import rerank
import rerank2
import run_step4 as s4
import run_step7 as s7
import window


def extra_rows(cfg, ew_cfg, uc_pairs, inter, cat_series):
    """Returns (X, y, ri, info) of the extra period; X has the same columns as the standard per-week step-7 features."""
    air = data.airings(["11-20"])
    lo, hi = window.week_start_sec(16), window.week_start_sec(21)
    truth, _ = h2.vtop5_truth(air[(air["start"] >= lo) & (air["start"] < hi)])
    users_x = sorted(int(u) for u in truth["user_id"].unique())
    pool_x = data.pool_ids(data.load_schedule("11-20"), 16, 20)
    n_u, n_p = len(users_x), len(pool_x)
    base = s7.make_ctx(["11-20"], "11-20", (16, 20), users_x, pool_x, uc_pairs, inter, cat_series)
    ctx = window.restrict(base, 11, 15, uc_pairs)
    ctx["train_end"] = window.week_start_sec(16)                      # window 11-15 ends at the Monday of week 16
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
    T = s4.bool_matrix(truth, ctx["uix"], ctx["pix"], n_u, n_p)
    y = T[ri, ci].astype(int)
    info = {"users_with_vtop5_truth_16_20": n_u, "pool_16_20": n_p, "pairs": int(len(ri)), "positive_pairs": int(y.sum()), "truth_items_in_pool": int(T.sum()),
            "truth_items_in_candidates": int(y.sum()), "users_without_h2_data_in_11_15": int((~store["H2"][1]).sum())}
    return rerank2.per_week(X, 5, 5), y, ri, info


def train2(Xs, ys, ris, tr_rows, va_rows, Xe, ye, rie, params, rounds=800, es=30):
    """LightGBM lambdarank on the standard rows of the users `tr_rows` plus ALL extra rows (each user-period is its own query group).
    Early stopping on the standard rows of `va_rows` (never evaluation users); va_rows=None -> fixed number of rounds."""
    import lightgbm as lgb
    m = np.isin(ris, tr_rows)
    X = pd.concat([Xs[m], Xe], ignore_index=True)
    y = np.r_[ys[m], ye]
    grp = np.r_[np.unique(ris[m], return_counts=True)[1], np.unique(rie, return_counts=True)[1]]
    d_tr = lgb.Dataset(X, y, group=grp, categorical_feature=["show_category"])
    if va_rows is None:
        return lgb.train(params, d_tr, num_boost_round=rounds)
    mv = np.isin(ris, va_rows)
    d_va = lgb.Dataset(Xs[mv], ys[mv], group=np.unique(ris[mv], return_counts=True)[1], categorical_feature=["show_category"], reference=d_tr)
    return lgb.train(params, d_tr, num_boost_round=rounds, valid_sets=[d_va], callbacks=[lgb.early_stopping(es, verbose=False)])

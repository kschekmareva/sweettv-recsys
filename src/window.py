"""Step 15: restrict a TRAIN-window context (built by run_step7.make_ctx on weeks 11-20) to the ISO weeks w_lo..w_hi.

Only the window-dependent parts of the context are recomputed (positives, airing table, completed airings, schedule of the window,
channel index and user-channel matrix, popularity); target-schedule fields are left untouched. Window end = Monday after week w_hi.
"""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import data
import h2
import hybrid

THR = 300


def week_start_sec(w):
    """Monday 00:00 of ISO week w of 2020 in seconds since 2020-01-01 (week 1 starts Monday 2019-12-30)."""
    return h2.sec_of("2019-12-30") + 7 * 86400 * (w - 1)


def restrict(base, w_lo, w_hi, uc_pairs, key="11-20"):
    """Copy of `base` (window = the whole schedule file `key`) restricted to weeks w_lo..w_hi of that file."""
    c = dict(base)
    n_u, n_p, uix, pix = c["n_u"], c["n_p"], c["uix"], c["pix"]
    lo, hi = week_start_sec(w_lo), week_start_sec(w_hi + 1)
    air = c["air"]
    air = air[(air["start"] >= lo) & (air["start"] < hi)].reset_index(drop=True)
    g = air.groupby(["user_id", "tv_show_id"], as_index=False).agg(overlap_sec=("ov", "sum"))
    c["train_all"] = g.assign(overlap_sec=g["overlap_sec"].astype(np.int64))
    c["train_pos"] = c["train_all"][c["train_all"]["overlap_sec"] >= THR].reset_index(drop=True)
    c["air"], c["air_ri"], c["air_ci"] = air, uix.get_indexer(air["user_id"].values), pix.get_indexer(air["tv_show_id"].values)
    cat_of = pd.Series(base["air_cat"], index=base["air"]["tv_show_id"].values)
    cat_by_show = cat_of[~cat_of.index.duplicated()]
    c["air_cat"] = cat_by_show.reindex(air["tv_show_id"].values).fillna(-1).values.astype(np.int64)
    sel = (c["air_ri"] >= 0) & (c["air_ci"] >= 0) & (air["view_pct"].values >= 0.8)
    c["hi"] = {"ri": c["air_ri"][sel], "ci": c["air_ci"][sel], "start": air["start"].values[sel].astype(np.float64)}
    s = data.load_schedule(key)
    m = (s["tv_show_id"] != 0) & s["iso_week"].between(w_lo, w_hi) & (s["iso_year"] == 2020)
    st = s.loc[m, ["channel_id", "tv_show_id", "start"]].copy()
    st["pc"] = pix.get_indexer(st["tv_show_id"].values)
    st = st[st["pc"] >= 0]
    c["sched_train"], c["n_air_tr"] = st, np.bincount(st["pc"].values, minlength=n_p).astype(np.float32)
    c["ch_index"] = pd.Index(np.sort(st["channel_id"].unique()))
    up = uc_pairs[uc_pairs["iso_week"].between(w_lo, w_hi)][["user_id", "vsetv_id"]].drop_duplicates()
    r, cc = uix.get_indexer(up["user_id"].values), c["ch_index"].get_indexer(up["vsetv_id"].values)
    ok = (r >= 0) & (cc >= 0)
    c["UC"] = csr_matrix((np.ones(ok.sum(), dtype=np.float32), (r[ok], cc[ok])), shape=(n_u, len(c["ch_index"])))
    c["P"] = hybrid.pop_counts(c["train_pos"], c["pool"])
    c["pop_norm"] = hybrid.minmax_vec(c["P"] * np.log1p(c["n_target"]))
    c["periods"] = [f"{w_lo}-{w_hi}"]
    return c

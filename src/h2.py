"""Airing-level tools: VTOP5 truth and the H2 own-history scorer.

Idea source (concept only, no code used): public Kaggle notebook by kinahem, "gornyaki final solution",
https://www.kaggle.com/code/kinahem/gornyaki-final-solution
"""
import time

import numpy as np
import pandas as pd

import data

PCT = 0.8
C_GRID = (1.0, 0.8, 0.5)


def vtop5_truth(air: pd.DataFrame):
    """Per user: top-5 shows by the number of airings with view_pct >= 0.8, ties by the global count of such airings
    (sum over users), then by tv_show_id. Returns (truth pairs, per-user number of distinct shows with >= 1 such airing)."""
    a = air[air["view_pct"] >= PCT]
    cnt = a.groupby(["user_id", "tv_show_id"], as_index=False).size().rename(columns={"size": "cnt"})
    cnt["g"] = cnt["tv_show_id"].map(cnt.groupby("tv_show_id")["cnt"].sum()).values
    cnt = cnt.sort_values(["user_id", "cnt", "g", "tv_show_id"], ascending=[True, False, False, True], kind="stable")
    cnt["rank"] = cnt.groupby("user_id").cumcount()
    return cnt.loc[cnt["rank"] < 5, ["user_id", "tv_show_id"]].reset_index(drop=True), cnt.groupby("user_id").size()


def train_airing_counts(pool, periods=("11-20",)) -> np.ndarray:
    """Number of airings of each pool show in the TRAIN schedule (rows starting in the weeks of `periods`)."""
    parts = []
    for key in periods:
        w0, w1 = data.PERIODS[key]
        s = data.load_schedule(key)
        m = (s["tv_show_id"] != 0) & s["iso_week"].between(w0, w1) & (s["iso_year"] == 2020)
        parts.append(s.loc[m, "tv_show_id"])
    cnt = pd.concat(parts).value_counts()
    return cnt.reindex(np.asarray(pool)).fillna(0).values.astype(np.float32)


def prepare(air: pd.DataFrame):
    """TRAIN airings with view_pct >= 0.8 and their base reward r (5 if fully watched else 2 + 2 * view_pct^2)."""
    t0 = time.time()
    a = air[air["view_pct"] >= PCT]
    vp = a["view_pct"].values.astype(np.float64)
    r = np.where(vp >= 1.0, 5.0, 2.0 + 2.0 * vp ** 2)
    return {"user": a["user_id"].values, "show": a["tv_show_id"].values, "start": a["start"].values.astype(np.float64),
            "r": r, "seconds": time.time() - t0, "airings": int(len(a))}


def scores(prep, n_airings, users, pool, train_end_sec, c, scale=60.0):
    """H2 score(user, show) = log10(1 + airings of the show in TRAIN) * sum over the user's airings with view_pct >= 0.8
    of r * c ** (days_before_train_end / 60). Returns (S float32 (users, pool), valid, seconds)."""
    t0 = time.time()
    days = np.maximum(train_end_sec - prep["start"], 0.0) / 86400.0
    val = prep["r"] * np.power(c, days / scale)          # decay c ** (days / scale); scale = 60 in steps 5-8
    ri = pd.Index(np.asarray(users, dtype=np.uint64)).get_indexer(prep["user"])
    ci = pd.Index(np.asarray(pool)).get_indexer(prep["show"])
    ok = (ri >= 0) & (ci >= 0)
    S = np.zeros((len(users), len(pool)), dtype=np.float32)
    np.add.at(S, (ri[ok], ci[ok]), val[ok].astype(np.float32))
    S *= np.log10(1.0 + n_airings)[None, :].astype(np.float32)
    return S, S.max(axis=1) > 0, time.time() - t0


def sec_of(date_str):
    return float((np.datetime64(date_str) - data.EPOCH) / np.timedelta64(1, "s"))

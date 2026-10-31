"""Baselines. Every method has the signature  f(train_pos, pool, users, k=5) -> {user_id: [k ids]}.

train_pos: DataFrame of TRAIN positive pairs (user_id uint64, tv_show_id, overlap_sec, ...).
pool: sorted array of candidate tv_show_ids. Ties are broken by ascending tv_show_id.
"""
import numpy as np
import pandas as pd


def pop_ranking(train_pos: pd.DataFrame, pool) -> pd.DataFrame:
    """Pool ranked by number of distinct TRAIN users with a positive pair (desc), then id (asc)."""
    cnt = train_pos[train_pos["tv_show_id"].isin(pool)].groupby("tv_show_id")["user_id"].nunique()
    r = pd.DataFrame({"tv_show_id": np.asarray(pool)})
    r["n_users"] = r["tv_show_id"].map(cnt).fillna(0).astype(int)
    return r.sort_values(["n_users", "tv_show_id"], ascending=[False, True]).reset_index(drop=True)


def recommend_popularity(train_pos, pool, users, k=5):
    top = [int(x) for x in pop_ranking(train_pos, pool)["tv_show_id"].head(k)]
    return {u: list(top) for u in users}


def own_history_lists(train_pos, pool):
    """user -> own positive in-pool programmes ranked by total overlap seconds (desc), id (asc)."""
    pp = train_pos[train_pos["tv_show_id"].isin(pool)].sort_values(
        ["user_id", "overlap_sec", "tv_show_id"], ascending=[True, False, True])
    return {int(u): [int(x) for x in g["tv_show_id"]] for u, g in pp.groupby("user_id")}


def recommend_history(train_pos, pool, users, k=5):
    pop = [int(x) for x in pop_ranking(train_pos, pool)["tv_show_id"]]
    hist = own_history_lists(train_pos, pool)
    out = {}
    for u in users:
        lst = hist.get(u, [])[:k]
        have = set(lst)
        for p in pop:
            if len(lst) == k:
                break
            if p not in have:
                lst.append(p)
                have.add(p)
        out[u] = lst
    return out


METHODS = {"P": recommend_popularity, "H": recommend_history}

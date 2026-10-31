"""Score fusion: score = w_pop * pop_norm + sum_i w_i * method_i_norm, w_pop = 1 - sum_i w_i.

Every score is min-max normalised over the pool per user (pop_norm is the same for all users).
A user for whom a component is invalid (no TRAIN history for it) gets that component's weight moved to popularity.
"""
import numpy as np


def pop_counts(train_pos, pool):
    """Distinct TRAIN users with a positive pair, per pool item."""
    cnt = train_pos[train_pos["tv_show_id"].isin(pool)].groupby("tv_show_id")["user_id"].nunique()
    return cnt.reindex(np.asarray(pool)).fillna(0).values.astype(np.float32)


def minmax_rows(S):
    mn = S.min(axis=1, keepdims=True)
    rng = S.max(axis=1, keepdims=True) - mn
    return (S - mn) / np.where(rng > 0, rng, 1.0)


def minmax_vec(v):
    return (v - v.min()) / (v.max() - v.min())


def fuse(pop_norm, parts, weights):
    """parts: list of (S_norm (n, pool), valid (n,)); weights: list of floats summing to <= 1."""
    eff = [w * v.astype(np.float32)[:, None] for (_, v), w in zip(parts, weights)]
    tot = sum(eff) if eff else 0.0
    out = pop_norm[None, :].astype(np.float32) * (1.0 - tot)
    for (S, _), e in zip(parts, eff):
        out += e * S
    return out


def topk(S, k=5):
    """Column indices of the k best scores per row, best first (ties: arbitrary but deterministic)."""
    part = np.argpartition(-S, k - 1, axis=1)[:, :k]
    sc = np.take_along_axis(S, part, axis=1)
    order = np.argsort(-sc, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)

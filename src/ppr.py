"""Method 3: Personalized PageRank on the bipartite user-programme graph.

Graph: TRAIN positive pairs (>= 300 s), edge weight log1p(overlap_sec / 60), row-normalised transitions in both
directions (user -> programme and programme -> user). The walk restarts at the seed user with probability 1 - alpha
(alpha = link-following probability). Power iteration for many seeds at once (columns of a dense matrix) until the
largest per-seed L1 change is < tol or max_iter iterations. Score = stationary mass on a programme.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

ALPHAS = (0.3, 0.5, 0.7, 0.85)


def build_graph(train_pos: pd.DataFrame) -> dict:
    t0 = time.time()
    uc, uu = pd.factorize(train_pos["user_id"].values)
    ic, ii = pd.factorize(train_pos["tv_show_id"].values)
    w = np.log1p(train_pos["overlap_sec"].values / 60.0).astype(np.float32)
    W = csr_matrix((w, (uc, ic)), shape=(len(uu), len(ii)))
    du, di = np.asarray(W.sum(axis=1)).ravel(), np.asarray(W.sum(axis=0)).ravel()
    inv = lambda d: diags(np.where(d > 0, 1.0 / np.maximum(d, 1e-12), 0.0).astype(np.float32))
    return {
        "users": pd.Index(uu), "items": pd.Index(ii), "n_users": len(uu), "n_items": len(ii), "nnz": int(W.nnz),
        "A": (inv(du) @ W).T.tocsr(),   # (items x users): programme mass received from users, user->programme transition
        "B": (W @ inv(di)).tocsr(),     # (users x items): user mass received from programmes, programme->user transition
        "build_s": time.time() - t0,
    }


def _chunk(g, cols, alpha, max_iter, tol, pool_rows):
    s = len(cols)
    E = np.zeros((g["n_users"], s), dtype=np.float32)
    E[cols, np.arange(s)] = 1.0
    pu, pi = E.copy(), np.zeros((g["n_items"], s), dtype=np.float32)
    delta, it = np.inf, 0
    for it in range(1, max_iter + 1):
        pi_n = alpha * (g["A"] @ pu)
        pu_n = (1.0 - alpha) * E + alpha * (g["B"] @ pi)
        delta = float((np.abs(pu_n - pu).sum(axis=0) + np.abs(pi_n - pi).sum(axis=0)).max())
        pu, pi = pu_n, pi_n
        if delta < tol:
            break
    return pi[pool_rows].T, it, delta


def run(g, seed_users, pool, alpha, max_iter=100, tol=1e-6, threads=4, chunk=500):
    """Returns (S (n_seeds, len(pool)) float32, valid (n_seeds,), info). Seeds absent from the graph are invalid."""
    t0 = time.time()
    ui = g["users"].get_indexer(np.asarray(seed_users, dtype=np.uint64))
    valid = ui >= 0
    pc = g["items"].get_indexer(np.asarray(pool))
    present = pc >= 0
    S = np.zeros((len(seed_users), len(pool)), dtype=np.float32)
    seeds = np.flatnonzero(valid)
    chunks = [seeds[a:a + chunk] for a in range(0, len(seeds), chunk)]
    with ThreadPoolExecutor(max_workers=threads) as ex:
        res = list(ex.map(lambda c: _chunk(g, ui[c], alpha, max_iter, tol, pc[present]), chunks))
    for c, (m, _, _) in zip(chunks, res):
        S[np.ix_(c, np.flatnonzero(present))] = m
    its, deltas = [r[1] for r in res], [r[2] for r in res]
    info = {"alpha": alpha, "iterations_max": int(max(its)), "iterations_per_chunk": its,
            "converged_all_chunks": bool(all(d < tol for d in deltas)), "final_max_l1_change": float(max(deltas)),
            "seeds": int(len(seeds)), "seconds": round(time.time() - t0, 1)}
    return S, valid, info

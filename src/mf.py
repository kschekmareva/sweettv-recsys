"""Method 1: implicit-feedback ALS (Hu-Koren-Volinsky) via the `implicit` package (CPU).

Observed pairs = ALL labelled TRAIN pairs, weight w_ui = log1p(total_overlap_sec / 60),
confidence c_ui = 1 + alpha * w_ui, preference p_ui = 1 for observed pairs.
"""
import time

import numpy as np
import pandas as pd
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

GRID = [dict(factors=f, reg=r, alpha=a) for f in (32, 64) for r in (0.01, 0.1) for a in (10, 40)]
ITERATIONS = 15
SEED = 42


def fit_als(train_all: pd.DataFrame, cfg: dict, seed: int = SEED):
    """train_all: user_id (uint64), tv_show_id, overlap_sec for all labelled TRAIN pairs."""
    t0 = time.time()
    ucode, uniq_u = pd.factorize(train_all["user_id"].values)
    icode, uniq_i = pd.factorize(train_all["tv_show_id"].values)
    w = np.log1p(train_all["overlap_sec"].values / 60.0).astype(np.float32)
    X = csr_matrix((w, (ucode, icode)), shape=(len(uniq_u), len(uniq_i)))
    model = AlternatingLeastSquares(factors=cfg["factors"], regularization=cfg["reg"], alpha=cfg["alpha"],
                                    iterations=ITERATIONS, calculate_training_loss=False, random_state=seed)
    model.fit(X, show_progress=False)
    return {"U": np.asarray(model.user_factors), "V": np.asarray(model.item_factors),
            "users": pd.Index(uniq_u), "items": pd.Index(uniq_i), "fit_s": time.time() - t0,
            "n_users": len(uniq_u), "n_items": len(uniq_i), "nnz": int(X.nnz)}


def score(fitted, users, pool):
    """Dense (len(users), len(pool)) float32 scores. Returns (S, user_valid, item_ok).
    Users absent from TRAIN are invalid (rows left at 0); pool items absent from TRAIN get the
    user's minimum score (they have no factors)."""
    t0 = time.time()
    ur = fitted["users"].get_indexer(np.asarray(users, dtype=np.uint64))
    ic = fitted["items"].get_indexer(np.asarray(pool))
    valid, item_ok = ur >= 0, ic >= 0
    S = np.zeros((len(users), len(pool)), dtype=np.float32)
    if valid.any() and item_ok.any():
        S[np.ix_(valid, item_ok)] = fitted["U"][ur[valid]] @ fitted["V"][ic[item_ok]].T
        if (~item_ok).any():
            S[np.ix_(valid, ~item_ok)] = S[np.ix_(valid, item_ok)].min(axis=1, keepdims=True)
    return S, valid, item_ok, time.time() - t0

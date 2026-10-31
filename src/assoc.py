"""Method 4: item-to-item association rules A -> B from per-user TRAIN positive sets.

confidence(A->B) = n(A and B) / n(A); lift = confidence / (n(B) / N). A rule is kept when n(A and B) >= min_support
users (A != B). User score for candidate B = sum (or max) of confidence(A->B) over the user's TRAIN positives A.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

MIN_SUPPORT = 20
MODES = ("sum", "max")


def fit(train_pos: pd.DataFrame, pool, min_support: int = MIN_SUPPORT) -> dict:
    t0 = time.time()
    uc, uu = pd.factorize(train_pos["user_id"].values)
    ic, ii = pd.factorize(train_pos["tv_show_id"].values)
    ii = pd.Index(ii)
    Bm = csr_matrix((np.ones(len(uc), dtype=np.float32), (uc, ic)), shape=(len(uu), len(ii)))
    n_item = np.asarray(Bm.sum(axis=0)).ravel()
    C = (Bm.T @ Bm).tocsr()
    C.setdiag(0)
    C.data[C.data < min_support] = 0
    C.eliminate_zeros()
    pc = ii.get_indexer(np.asarray(pool))
    present = pc >= 0
    Cp = C[:, pc[present]].tocsr()                       # rules with a pool consequent
    conf = diags_rows(Cp, 1.0 / n_item)                  # confidence(A -> B)
    lift = diags_cols(conf, 1.0 / (n_item[pc[present]] / len(uu)))
    return {
        "users": pd.Index(uu), "items": ii, "pool_cols": np.flatnonzero(present), "n_pool": len(pool),
        "conf": conf, "conf_dense": conf.toarray().astype(np.float32), "N": len(uu),
        "rules_all": int(C.nnz), "rules_pool_consequent": int(Cp.nnz), "rules_lift_gt1_pool": int((lift.data > 1).sum()),
        "antecedents_with_rule": int((np.diff(C.indptr) > 0).sum()), "items_in_train": int(len(ii)),
        "min_support": min_support, "build_s": time.time() - t0,
    }


def diags_rows(M, v):
    out = M.tocsr().astype(np.float32)
    out.data = out.data * np.repeat(v.astype(np.float32), np.diff(out.indptr))
    return out


def diags_cols(M, v):
    out = M.tocsr().copy()
    out.data = out.data * v.astype(np.float32)[out.indices]
    return out


def score(m: dict, train_pos: pd.DataFrame, users, mode: str = "sum"):
    """Dense (len(users), pool) scores and validity (row has at least one firing rule). Returns (S, valid, seconds)."""
    t0 = time.time()
    uix = pd.Index(np.asarray(users, dtype=np.uint64))
    r = uix.get_indexer(train_pos["user_id"].values)
    c = m["items"].get_indexer(train_pos["tv_show_id"].values)
    ok = (r >= 0) & (c >= 0)
    Bu = csr_matrix((np.ones(ok.sum(), dtype=np.float32), (r[ok], c[ok])), shape=(len(users), len(m["items"])))
    cols = m["pool_cols"]
    S = np.zeros((len(users), m["n_pool"]), dtype=np.float32)
    if mode == "sum":
        S[:, cols] = (Bu @ m["conf"]).toarray()
    else:
        D = m["conf_dense"]
        for i in range(len(users)):
            idx = Bu.indices[Bu.indptr[i]:Bu.indptr[i + 1]]
            if len(idx):
                S[i, cols] = D[idx].max(axis=0)
    return S, S.max(axis=1) > 0, time.time() - t0

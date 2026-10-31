"""Method 2: content-based filtering from schedule metadata.

Item vector (blocks L2-normalised separately, concatenated with equal weight, then row-normalised):
  category one-hot | genre_1..3 multi-hot | TF-IDF over director+actors names | decade bucket one-hot.
User profile = sum of item vectors of the user's positive TRAIN programmes weighted by
log1p(overlap_sec / 60), L2-normalised. Score = cosine similarity with the pool items.
"""
import re
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

import data

COLS = ["tv_show_id", "tv_show_category", "tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3",
        "year_of_production", "director", "actors"]


def load_metadata() -> pd.DataFrame:
    """One row per labelled tv_show_id over all three schedule files (first non-null value per column)."""
    f = data.CACHE / "meta.pkl"
    if f.exists():
        return pd.read_pickle(f)
    d = pd.concat([pd.read_csv(data.DATA / fn, usecols=COLS, dtype=str) for fn in data.SCHEDULES.values()])
    d["tv_show_id"] = d["tv_show_id"].astype(np.int64)
    m = d[d["tv_show_id"] != 0].groupby("tv_show_id").first().sort_index()
    m.to_pickle(f)
    return m


def _names(*cells):
    out = []
    for s in cells:
        if isinstance(s, str):
            out += [re.sub(r"\s+", " ", t).strip().lower() for t in s.split(",") if t.strip()]
    return out


def _decade(s):
    """'1996-' -> dec1990, '1998-2001' -> dec1990 (first year), missing -> no feature."""
    m = re.search(r"\d{4}", s) if isinstance(s, str) else None
    return [f"dec{int(m.group()) // 10 * 10}"] if m else []


def build_items(meta: pd.DataFrame):
    """Returns ids, X (csr, L2-normalised rows) and a coverage dict."""
    ident = lambda x: x
    cat = [[c] if isinstance(c, str) else [] for c in meta["tv_show_category"]]
    gen = [[g.strip().lower() for g in row if isinstance(g, str)]
           for row in meta[["tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"]].itertuples(index=False)]
    ppl = [_names(d, a) for d, a in zip(meta["director"], meta["actors"])]
    dec = [_decade(y) for y in meta["year_of_production"]]
    blocks = {
        "category": CountVectorizer(analyzer=ident, binary=True).fit_transform(cat),
        "genre": CountVectorizer(analyzer=ident, binary=True).fit_transform(gen),
        "people_tfidf": TfidfVectorizer(analyzer=ident).fit_transform(ppl),
        "decade": CountVectorizer(analyzer=ident, binary=True).fit_transform(dec),
    }
    blocks = {k: normalize(v.astype(np.float32)) for k, v in blocks.items()}
    X = normalize(hstack(list(blocks.values()), format="csr").astype(np.float32))
    non_cat = sum((blocks[k].getnnz(axis=1) > 0).astype(int) for k in ("genre", "people_tfidf", "decade")) > 0
    info = {"n_items": int(len(meta)), "block_dims": {k: int(v.shape[1]) for k, v in blocks.items()},
            "share_with_genre": float((blocks["genre"].getnnz(axis=1) > 0).mean()),
            "share_with_people": float((blocks["people_tfidf"].getnnz(axis=1) > 0).mean()),
            "share_with_decade": float((blocks["decade"].getnnz(axis=1) > 0).mean())}
    return meta.index.values, X, non_cat, info


def score(ids, X, train_pos, pool, users):
    """Dense (len(users), len(pool)) cosine scores. Returns (S, user_valid, seconds)."""
    t0 = time.time()
    imap = pd.Index(ids)
    ui = pd.Index(np.asarray(users, dtype=np.uint64))
    r = ui.get_indexer(train_pos["user_id"].values.astype(np.uint64))
    c = imap.get_indexer(train_pos["tv_show_id"].values)
    ok = (r >= 0) & (c >= 0)
    w = np.log1p(train_pos["overlap_sec"].values[ok] / 60.0).astype(np.float32)
    W = csr_matrix((w, (r[ok], c[ok])), shape=(len(users), len(ids)))
    P = normalize(W @ X)
    valid = P.getnnz(axis=1) > 0
    pc = imap.get_indexer(np.asarray(pool))
    assert (pc >= 0).all(), "pool item without metadata row"
    Xp = X[pc].T.tocsc()
    S = np.zeros((len(users), len(pool)), dtype=np.float32)
    for a in range(0, len(users), 1000):
        S[a:a + 1000] = (P[a:a + 1000] @ Xp).toarray()
    return S, valid, time.time() - t0

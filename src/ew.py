"""Expected-watches scorer EW.

For (user, show): completion_rate = (# user airings of the show with view_pct >= 0.8) /
                                    (# airings of the show in the TRAIN schedule on channels the user watched),
smoothed with m pseudo-airings at the global completion rate; score = completion_rate * n_target_airings
(airings of the show in the target-period schedule, used only to know what airs). With recency c < 1 every airing
(numerator and denominator) is weighted by c ** (days_before_train_end / 60).
Shows the user never had a chance to complete get the global rate, so EW scores every pool show.
"""
import time

import numpy as np
from scipy.sparse import csr_matrix

M_GRID = (1, 3, 10)
C_GRID = (1.0, 0.5)


def scores(ctx, m, c):
    """Returns (S float32 (users, pool), valid (users watched >= 1 scheduled channel in TRAIN), global rate, seconds)."""
    t0 = time.time()
    n_u, n_p, end = ctx["n_u"], ctx["n_p"], ctx["train_end"]
    st = ctx["sched_train"]
    w = np.power(c, np.maximum(end - st["start"].values.astype(np.float64), 0.0) / 86400.0 / 60.0).astype(np.float32)
    AC = csr_matrix((w, (ctx["ch_index"].get_indexer(st["channel_id"].values), st["pc"].values)), shape=(len(ctx["ch_index"]), n_p))
    den = (ctx["UC"] @ AC).toarray()                       # weighted airings of the show on the user's channels
    hi = ctx["hi"]
    wn = np.power(c, np.maximum(end - hi["start"], 0.0) / 86400.0 / 60.0).astype(np.float32)
    num = np.zeros((n_u, n_p), dtype=np.float32)
    np.add.at(num, (hi["ri"], hi["ci"]), wn)               # weighted completed airings
    g = float(num.sum() / den.sum())
    rate = np.minimum((num + m * g) / (den + m), 1.0)
    S = (rate * ctx["n_target"][None, :]).astype(np.float32)
    return S, ctx["UC"].getnnz(axis=1) > 0, g, time.time() - t0

"""Regenerates the final submission (submission_STEP7_all.csv) from the raw data in one command.

    python src/make_final.py [--rebuild] [--step15] [--out PATH]

  --rebuild   rebuild the view/schedule join cache (outputs/cache/interactions_v3.pkl) from ./data instead of reusing it
  --out PATH  output file (default outputs/submissions/repro_STEP7_all.csv, so the committed final file is never overwritten)
  --step15    regenerate submission_STEP15.csv instead: per-week reranker (rerank2.per_week) fitted on weeks 11-20 features, applied to features and
              candidates from weeks 11-30 with n_target_airings per week of the 31-42 schedule (step 15, long-window branch)

Pipeline (all hyperparameters are the frozen ones of steps 5-11, no tuning here):
  1. join view log and schedules (weeks 11-20 and 21-30), airing-level table, VTOP5 truth for weeks 21-30
  2. components on TRAIN weeks 11-20: H2, EW, PPR, MF, AR (+ P-air) -> top-50 candidate union -> step-7 reranker features
  3. LightGBM lambdarank (seed 42, deterministic, 4 threads, 142 trees, no early stopping) fitted on ALL validation users with VTOP5 truth
  4. the same components/features on TRAIN window = weeks 21-30 and the 31-42 schedule -> top-5 per submission user
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import numpy as np
import pandas as pd

import data
import h2
import rerank
import rerank2
import run_step4 as s4
import run_step7 as s7
from run_validation import write_submission

CFG = {"h2_c": 0.5,                                              # H2 decay c ** (days / 60), step 5
       "mf": {"factors": 32, "reg": 0.1, "alpha": 0.3},           # implicit ALS, 15 iterations, seed 42, step 5
       "ppr_alpha": 0.3, "ar_mode": "max"}                        # steps 4-5
EW_CFG = (10, 0.5)                                                # EW prior m, recency c, step 7
N_TREES = 142                                                     # mean best iteration of the two folds (174, 110), step 11
SEED, K = 42, 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(data.ROOT / "outputs" / "submissions" / "repro_STEP7_all.csv"))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--step15", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    inter, _ = data.build_interactions(force=args.rebuild)
    views = data.load_views()
    uc_pairs = views[["user_id", "vsetv_id", "iso_week"]].drop_duplicates()
    del views
    titles = data.titles()
    cat_list = sorted(titles["tv_show_category"].dropna().unique())
    cat_series = pd.Series(pd.Categorical(titles["tv_show_category"], categories=cat_list).codes, index=titles.index)

    # ---- validation: truth, user universe (V300 users + VTOP5 users + submission users), step-7 features
    val = inter[inter["period"] == "21-30"]
    v300 = val[val["overlap_sec"] >= 300]
    vtop5, _ = h2.vtop5_truth(data.airings(["21-30"]))
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    v300_users = set(int(u) for u in v300["user_id"].unique())
    users = sorted(v300_users | set(int(u) for u in vtop5["user_id"].unique()) | set(sub_users))
    if args.step15:                                                      # step 15 universe also holds the V900 / VHALF users (the EW prior depends on it)
        extra = [val[val["overlap_sec"] >= 900], val[val["max_frac"] >= 0.5]]
        users = sorted(set(users) | set(int(u) for e in extra for u in e["user_id"].unique()))
    n_u, n_p = len(users), len(pool)
    ctx = s7.make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    T5 = s4.bool_matrix(vtop5, ctx["uix"], ctx["pix"], n_u, n_p)
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], CFG)
    store["EW"] = s7.build_store(ctx, ["EW"], CFG, EW_CFG)[0]["EW"]
    cand = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
    y = T5[ri, ci].astype(int)
    in_v300 = np.array([u in v300_users for u in users])                # set membership: numpy would cast 64-bit ids to float64
    fit_rows = np.flatnonzero(T5.any(axis=1) & in_v300)             # all validation users with VTOP5 truth
    Xfit = rerank2.per_week(X, 10, 10) if args.step15 else X           # step 15: counts per week (10-week window, 10-week target)
    model = rerank2.train(Xfit, y, ri, fit_rows, None, rerank2.BASE_PARAMS, rounds=N_TREES)

    # ---- final run: TRAIN window = weeks 21-30, target schedule = weeks 31-42, submission users, no re-fitting
    users_f = data.load_submission_users()
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    nf, npf = len(users_f), len(pool_f)
    ctxf = s7.make_ctx(["11-20", "21-30"] if args.step15 else ["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
    store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], CFG)
    store_f["EW"] = s7.build_store(ctxf, ["EW"], CFG, EW_CFG)[0]["EW"]
    cand_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
    rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
    Xf = rerank2.per_week(Xf, 20, 12) if args.step15 else Xf           # step 15: 20 feature weeks (11-30), 12 target weeks (31-42)
    top = rerank.top5(model, Xf[list(X.columns)], rif, cif, np.arange(nf))
    write_submission({u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}, users_f, args.out)
    assert len(fit_rows) == 4643, len(fit_rows)                          # all validation users with VTOP5 truth
    print(f"wrote {args.out} in {time.time() - t0:.1f} s (fit on {len(fit_rows)} users, {N_TREES} trees)")


if __name__ == "__main__":
    main()

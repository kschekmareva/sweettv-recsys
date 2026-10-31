"""Step 11: apply the pre-declared learning-curve rule (step 10, item 3): refit the step-7 reranker on ALL validation users.
Usage: python src/run_step11.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
import pandas as pd

import data
import h2
import metrics
import rerank
import rerank2
import run_step3 as s3
import run_step4 as s4
import run_step5 as s5
import run_step7 as s7
from run_validation import write_submission

SEED, K = 42, 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"


def main():
    T0 = time.time()
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    J10 = json.loads((OUT / "validation_step10.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    assert J10["learning_curve"]["full_beats_half_in_both_folds"] and J10["learning_curve"]["features"] == "step7 features"
    R = {"rule": J10["learning_curve"]["rule"], "learning_curve": J10["learning_curve"]["vtop5_by_fraction"]}

    # ---- validation features and the two fold fits of step 10 (deterministic; only used to recover the best iterations)
    views = data.load_views()
    uc_pairs = views[["user_id", "vsetv_id", "iso_week"]].drop_duplicates()
    del views
    inter, _ = data.build_interactions()
    titles = data.titles()
    cat_list = sorted(titles["tv_show_category"].dropna().unique())
    cat_series = pd.Series(pd.Categorical(titles["tv_show_category"], categories=cat_list).codes, index=titles.index)
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300]}
    vpairs["VTOP5"], _ = h2.vtop5_truth(data.airings(["21-30"]))
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    users = sorted(set(int(u) for v in vpairs.values() for u in v["user_id"].unique()) | set(sub_users))
    n_u, n_p = len(users), len(pool)
    ctx = s7.make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    uix = ctx["uix"]
    T5 = s4.bool_matrix(vpairs["VTOP5"], uix, ctx["pix"], n_u, n_p)
    ne5 = T5.any(axis=1)
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    tune_truth, report_truth = tune_rows[ne5[tune_rows]], report_rows[ne5[report_rows]]
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand6 = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand6)           # step-7 features
    y = T5[ri, ci].astype(int)

    def split(rows):
        p = np.random.default_rng(SEED).permutation(rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])
    best_iter, vtop5 = {}, {}
    for f, (train, ev) in {"A": (tune_truth, report_rows), "B": (report_truth, tune_rows)}.items():
        tr, va = split(train)
        m = rerank2.train(X, y, ri, tr, va, rerank2.BASE_PARAMS)
        best_iter[f] = int(m.best_iteration)
        ap = metrics.ap5_matrix(rerank.top5(m, X, ri, ci, ev), T5[ev])
        vtop5[f] = float(ap[ne5[ev]].mean())
    assert abs(vtop5["A"] - J10["table"]["step7 features"]["A_VTOP5"]) < 1e-9 and abs(vtop5["B"] - J10["table"]["step7 features"]["B_VTOP5"]) < 1e-9
    rounds = int(round(np.mean(list(best_iter.values()))))
    R["fold_best_iterations"], R["fold_vtop5_reproduced"], R["rounds"] = best_iter, vtop5, rounds

    # ---- fit on ALL validation users, fixed number of trees
    all_train = np.sort(np.r_[tune_truth, report_truth])
    model = rerank2.train(X, y, ri, all_train, None, rerank2.BASE_PARAMS, rounds=rounds)
    R["fit_users"] = int(len(all_train))

    # ---- final run exactly as for submission_STEP7.csv
    users_f = data.load_submission_users()
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    nf, npf = len(users_f), len(pool_f)
    ctxf = s7.make_ctx(["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
    store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
    store_f["EW"] = s7.build_store(ctxf, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
    rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
    top = rerank.top5(model, Xf[list(X.columns)], rif, cif, np.arange(nf))
    preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
    path = SUBDIR / "submission_STEP7_all.csv"
    write_submission(preds, users_f, path)
    prev = pd.read_csv(SUBDIR / "submission_STEP7.csv", dtype=str).set_index("user_id")["tv_show_id"]
    shared = [len(set(preds[u]) & set(int(x) for x in prev[str(u)].split())) / K for u in users_f]
    pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
    R["final"] = {**s3.check_file(path, users_f, pool_f),
                  "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in store_f},
                  "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f])),
                  "share_of_ids_shared_with_submission_STEP7": float(np.mean(shared)),
                  "share_of_users_with_identical_list": float(np.mean([s == 1.0 for s in shared]))}
    R["runtime_s"] = round(time.time() - T0, 1)
    (OUT / "validation_step11.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    sm = SUBDIR / "SUBMISSIONS.md"
    txt = sm.read_text(encoding="utf-8")
    if "submission_STEP7_all.csv" not in txt:
        sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP7_all.csv | step 11 (FINAL): step-7 reranker refit on ALL validation users (learning-curve rule), LightGBM lambdarank | step-7 features and settings, seed 42, {rounds} trees "
                      f"(mean best iteration of the two folds: {best_iter['A']} and {best_iter['B']}); features from weeks 21-30 + 31-42 schedule; cannot be scored on validation | | |\n", encoding="utf-8")
    f = R["final"]
    sec = ["", "", "## Step 11 - learning-curve rule applied", "",
           f"Rule (step 10, item 3): {R['rule']} Outcome: 100% > 50% in both folds (A {R['learning_curve']['1.0']['A']:.4f} > {R['learning_curve']['0.5']['A']:.4f}, "
           f"B {R['learning_curve']['1.0']['B']:.4f} > {R['learning_curve']['0.5']['B']:.4f}), best features = step-7 features.", "",
           f"- Model: step-7 reranker (step-7 features, lambdarank, step-7 hyperparameters, seed 42) fitted on all {R['fit_users']:,} validation users with VTOP5 truth (TUNE + REPORT), "
           f"{rounds} trees = mean of the best iterations of the two folds ({best_iter['A']} and {best_iter['B']}); no early stopping, no re-fitting on test. "
           f"The two fold fits were re-run only to recover the iterations and reproduce step 10 exactly (VTOP5 {vtop5['A']:.4f} and {vtop5['B']:.4f}).",
           f"- Final run as for submission_STEP7.csv: features from weeks 21-30 + the 31-42 schedule. File `{f['file']}`: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, "
           f"5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}.",
           f"- Users without component data (popularity fallback): {f['users_without_component_data']}; repeat share {f['repeat_share_train_positive']:.3f}; "
           f"share of ids shared with submission_STEP7.csv {f['share_of_ids_shared_with_submission_STEP7']:.3f} (identical lists for {f['share_of_users_with_identical_list']:.1%} of users).",
           "- submission_STEP7_all.csv is the final submission. It cannot be scored on validation, because no held-out users remain; the decision rests on the learning curve "
           "(VTOP5 rises with training users in both folds), which was observed up to the size of one fold and is extrapolated to twice that size."]
    p10 = OUT / "validation_step10.md"
    t10 = p10.read_text(encoding="utf-8")
    if "## Step 11 - learning-curve rule applied" in t10:
        t10 = t10[:t10.index("## Step 11 - learning-curve rule applied")].rstrip()
    p10.write_text(t10.rstrip("\n") + "\n" + "\n".join(sec) + "\n", encoding="utf-8")
    print(json.dumps({k: R[k] for k in ("fold_best_iterations", "rounds", "fit_users", "final", "runtime_s")}, indent=1, default=str))


if __name__ == "__main__":
    main()

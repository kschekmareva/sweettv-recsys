"""Step 18: temporal data augmentation for the per-week reranker (last attempt).

Extra training period: features/candidates from weeks 11-15, labels VTOP5 of weeks 16-20 (augment.extra_rows; only weeks <= 20 are used).
Standard setting: features 11-20 -> truth 21-30, 2-fold swap. M0 = training-fold rows only, M1 = training-fold rows + all extra rows.
Early stopping on a 20% inner split of the training-fold users. Rule (pre-declared): ship only if M1 beats M0 on VTOP5 in BOTH folds and the pooled CI excludes zero.
Usage: python src/run_step18.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
import pandas as pd

import augment
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

SEED = 42
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
FOLDS = ("A", "B")


def boot(d, rng):
    bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    return {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def main():
    T0 = time.time()
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    J15 = json.loads((OUT / "validation_step15.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    R = {"seed": SEED, "primary": "VTOP5"}

    views = data.load_views()
    uc_pairs = views[["user_id", "vsetv_id", "iso_week"]].drop_duplicates()
    del views
    inter, _ = data.build_interactions()
    titles = data.titles()
    cat_list = sorted(titles["tv_show_category"].dropna().unique())
    cat_series = pd.Series(pd.Categorical(titles["tv_show_category"], categories=cat_list).codes, index=titles.index)
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300], "V900": val[val["overlap_sec"] >= 900], "VHALF": val[val["max_frac"] >= 0.5]}
    vpairs["VTOP5"], _ = h2.vtop5_truth(data.airings(["21-30"]))
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    users = sorted(set(int(u) for v in vpairs.values() for u in v["user_id"].unique()) | set(sub_users))
    n_u, n_p = len(users), len(pool)
    ctx10 = s7.make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    uix, pix = ctx10["uix"], ctx10["pix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    ne = {v: T[v].any(axis=1) for v in VARS}
    T5, ne5 = T["VTOP5"], ne["VTOP5"]
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    folds = {"A": {"train": tune_rows[ne5[tune_rows]], "eval": report_rows}, "B": {"train": report_rows[ne5[report_rows]], "eval": tune_rows}}

    store, _ = s7.build_store(ctx10, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx10, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand = rerank.candidates(store, ctx10["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx10, store, ctx10["pop_norm"], cand)
    Xs, ys = rerank2.per_week(X, 10, 10), T5[ri, ci].astype(int)
    print(f"standard rows: {len(ri):,} ({time.time() - T0:.0f} s)", flush=True)
    Xe, ye, rie, info = augment.extra_rows(cfg, ew_cfg, uc_pairs, inter, cat_series)
    R["extra_period"] = {**info, "share_positive": float(ye.mean()), "standard_rows": int(len(ri)), "standard_positive_share": float(ys.mean())}
    print("extra rows", R["extra_period"], f"({time.time() - T0:.0f} s)", flush=True)

    def split(rows):
        p = np.random.default_rng(SEED).permutation(rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    ap = lambda top, rows, v: metrics.ap5_matrix(top, T[v][rows])[ne[v][rows]]
    res, per_user, best_iter = {}, {}, {"M0": {}, "M1": {}}
    for f in FOLDS:
        ev = folds[f]["eval"]
        tr, va = split(folds[f]["train"])
        m0 = rerank2.train(Xs, ys, ri, tr, va, rerank2.BASE_PARAMS)
        m1 = augment.train2(Xs, ys, ri, tr, va, Xe, ye, rie, rerank2.BASE_PARAMS)
        best_iter["M0"][f], best_iter["M1"][f] = int(m0.best_iteration), int(m1.best_iteration)
        res[f], per_user[f] = {}, {}
        for lab, m in (("M0", m0), ("M1", m1)):
            top = rerank.top5(m, Xs, ri, ci, ev)
            per_user[f][lab] = {v: ap(top, ev, v) for v in VARS}
            res[f][lab] = {v: float(a.mean()) for v, a in per_user[f][lab].items()}
        print(f, res[f], best_iter["M0"][f], best_iter["M1"][f], flush=True)
    for f in FOLDS:
        assert abs(res[f]["M0"]["VTOP5"] - J15["table"][f]["ref_10w_fit_10w_infer"]["VTOP5"]) < 1e-9, (f, res[f]["M0"])
    R["table"], R["best_iterations"] = res, best_iter
    rng = np.random.default_rng(SEED)
    d = {f: per_user[f]["M1"]["VTOP5"] - per_user[f]["M0"]["VTOP5"] for f in FOLDS}
    R["bootstrap_m1_minus_m0_vtop5"] = {"A": boot(d["A"], rng), "B": boot(d["B"], rng), "pooled": boot(np.r_[d["A"], d["B"]], rng)}
    R["bootstrap_robustness_pooled"] = {v: boot(np.r_[per_user["A"]["M1"][v] - per_user["A"]["M0"][v], per_user["B"]["M1"][v] - per_user["B"]["M0"][v]], rng) for v in VARS}
    both = all(res[f]["M1"]["VTOP5"] > res[f]["M0"]["VTOP5"] for f in FOLDS)
    ci_ok = R["bootstrap_m1_minus_m0_vtop5"]["pooled"]["ci95"][0] > 0
    ship = bool(both and ci_ok)
    R["rule"] = "Ship only if M1 (training-fold rows + all extra-period rows) beats M0 (training-fold rows only) on VTOP5 in BOTH folds and the pooled paired-bootstrap 95% CI of M1 - M0 excludes zero."
    R["decision"] = {"m1_beats_m0_in_both_folds": bool(both), "pooled_ci_excludes_zero": bool(ci_ok), "shipped": ship}
    print("decision", R["decision"], flush=True)

    if ship:
        rounds = int(round(np.mean(list(best_iter["M1"].values()))))
        all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
        model = augment.train2(Xs, ys, ri, all_train, None, Xe, ye, rie, rerank2.BASE_PARAMS, rounds=rounds)
        users_f = data.load_submission_users()
        pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
        nf, npf = len(users_f), len(pool_f)
        ctxf = s7.make_ctx(["11-20", "21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
        store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
        store_f["EW"] = s7.build_store(ctxf, ["EW"], cfg, ew_cfg)[0]["EW"]
        cand_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
        rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
        Xfpw = rerank2.per_week(Xf, 20, 12)[list(Xs.columns)]
        top = rerank.top5(model, Xfpw, rif, cif, np.arange(nf))
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / "submission_STEP18.csv"
        write_submission(preds, users_f, path)
        prev = pd.read_csv(SUBDIR / "submission_STEP15.csv", dtype=str).set_index("user_id")["tv_show_id"]
        shared = [len(set(preds[u]) & set(int(x) for x in prev[str(u)].split())) / 5 for u in users_f]
        pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
        R["final"] = {"trees": rounds, "fit_users": int(len(all_train)), "extra_rows": int(len(ye)), **s3.check_file(path, users_f, pool_f),
                      "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in store_f},
                      "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / 5 for u in users_f])),
                      "share_of_ids_shared_with_STEP15": float(np.mean(shared)), "share_of_users_with_identical_list": float(np.mean([s == 1.0 for s in shared]))}
    R["runtime_s"] = round(time.time() - T0, 1)
    (OUT / "validation_step18.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_reports(R)
    print(json.dumps({k: R[k] for k in ("decision", "best_iterations", "runtime_s")}, indent=1))
    if ship:
        print(json.dumps(R["final"], indent=1))


def write_reports(R):
    t, bs, d, e = R["table"], R["bootstrap_m1_minus_m0_vtop5"], R["decision"], R["extra_period"]
    md = ["# Step 18 - temporal data augmentation for the reranker", "",
          "Extra training period: features and candidates from weeks 11-15 (every component fitted on weeks 11-15 only), labels = VTOP5 of weeks 16-20, candidate pool = programmes airing in weeks 16-20, "
          "per-week features (5 feature weeks, 5 target weeks); all users with such truth. Only weeks <= 20 are used, so validation weeks 21-30 cannot leak. "
          "Standard setting: features 11-20 -> truth 21-30, 2-fold swap (A: fit TUNE, score REPORT; B: the reverse). M0 = training-fold rows only; M1 = training-fold rows + all extra rows; "
          "tree count by early stopping (30 rounds) on a 20% inner split of the training-fold users. Seed 42.", "",
          "## Extra-period rows", "", "| item | value |", "|---|---|",
          f"| users with VTOP5 truth in weeks 16-20 | {e['users_with_vtop5_truth_16_20']:,} |", f"| pool (programmes airing in weeks 16-20) | {e['pool_16_20']:,} |",
          f"| candidate pairs / positive pairs (share) | {e['pairs']:,} / {e['positive_pairs']:,} ({e['share_positive']:.4f}) |",
          f"| truth items in the pool / retrieved by the candidate generators | {e['truth_items_in_pool']:,} / {e['truth_items_in_candidates']:,} |",
          f"| users without H2 data in weeks 11-15 | {e['users_without_h2_data_in_11_15']:,} |",
          f"| standard rows (features 11-20, truth 21-30): pairs (positive share) | {e['standard_rows']:,} ({e['standard_positive_share']:.4f}) |", "",
          "## Validation (MAP@5 on the evaluation fold)", "", "| model | fold | " + " | ".join(VARS) + " | best iteration |", "|---|---|" + "---|" * (len(VARS) + 1)]
    for f in FOLDS:
        for m in ("M0", "M1"):
            md.append(f"| {m} | {f} | " + " | ".join(f"{t[f][m][v]:.4f}" for v in VARS) + f" | {R['best_iterations'][m][f]} |")
    md += ["", "M0 reproduces the step-7 / step-15 reference (0.4047 and 0.4065).", "", "## Paired bootstrap 95% CI of M1 - M0, VTOP5 (2,000 resamples, seed 42)", "", "| scope | users | mean diff | 95% CI |", "|---|---|---|---|"]
    for k in ("A", "B", "pooled"):
        b = bs[k]
        md.append(f"| {k} | {b['n_users']} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "Robustness, pooled M1 - M0:", "", "| truth | mean diff | 95% CI |", "|---|---|---|"]
    for v in VARS:
        b = R["bootstrap_robustness_pooled"][v]
        md.append(f"| {v} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "## Pre-declared rule and decision", "", R["rule"], "",
           f"- M1 > M0 in fold A: {t['A']['M1']['VTOP5'] > t['A']['M0']['VTOP5']}; fold B: {t['B']['M1']['VTOP5'] > t['B']['M0']['VTOP5']}; pooled CI excludes zero: {d['pooled_ci_excludes_zero']}.",
           f"- **Shipped: {'yes, submission_STEP18.csv' if d['shipped'] else 'no - submission_STEP15.csv stays the final file; nothing else was changed'}**.", ""]
    if d["shipped"]:
        fi = R["final"]
        md += ["## Shipped file `outputs/submissions/submission_STEP18.csv`", "",
               f"- Model: per-week reranker on the standard rows of all {fi['fit_users']:,} validation users + {fi['extra_rows']:,} extra-period rows, {fi['trees']} trees (mean best iteration of the two M1 folds {R['best_iterations']['M1']}); applied as in step 15 (features/candidates weeks 11-30, n_target_airings per week of the 31-42 schedule).",
               f"- Format: header {fi['header_ok']}, rows {fi['rows']}, user order {fi['user_order_ok']}, 5 distinct ids {fi['five_distinct_ids']}, ids in final pool {fi['all_ids_in_final_pool']}.",
               f"- Users without component data (popularity fallback): {fi['users_without_component_data']}; repeat share {fi['repeat_share_train_positive']:.3f}.",
               f"- Share of ids identical to submission_STEP15.csv: {fi['share_of_ids_shared_with_STEP15']:.3f} (identical lists for {fi['share_of_users_with_identical_list']:.1%} of users).", ""]
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP18.csv" not in txt:
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP18.csv | step 18: step-15 recipe with temporal augmentation (extra rows features 11-15 -> truth 16-20) | {fi['trees']} trees; cannot be scored on validation | not submitted | not submitted |\n", encoding="utf-8")
    b = bs["pooled"]
    md += ["## Summary", "",
           f"M1 scored VTOP5 {t['A']['M1']['VTOP5']:.4f} / {t['B']['M1']['VTOP5']:.4f} against M0 {t['A']['M0']['VTOP5']:.4f} / {t['B']['M0']['VTOP5']:.4f} in folds A / B, "
           f"pooled difference {b['mean_diff']:+.4f} with 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]. "
           f"The extra period added {e['pairs']:,} rows ({e['share_positive']:.2%} positive) to {e['standard_rows']:,} standard rows. "
           f"By the pre-declared rule the result was {'shipped as submission_STEP18.csv' if d['shipped'] else 'not shipped, so submission_STEP15.csv remains the final file'}."]
    (OUT / "validation_step18.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

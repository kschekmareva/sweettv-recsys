"""Step 15: use the full history at inference, validated by an analogue (fit on a SHORT window, infer on a LONGER one).

Per-week reranker = step-7 reranker with every window-dependent count/seconds feature divided by the number of weeks of its window
(rerank2.per_week; step-7 hyperparameters). Validation analogue: fit on features of weeks 16-20 (labels VTOP5 of weeks 21-30, candidates from the same
window), evaluate on the other fold with inference features/candidates from (a) weeks 16-20 and (b) weeks 11-20. Pre-declared rule decides the final run.
Usage: python src/run_step15.py
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
import window
from run_validation import write_submission

SEED, K = 42, 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
W_SHORT, W_LONG = 5, 10          # feature-window lengths in weeks (weeks 16-20 and 11-20); target of the validation = weeks 21-30 (10 weeks)
FOLDS = ("A", "B")


def boot(d, rng):
    bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    return {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def main():
    T0 = time.time()
    laps = {}
    lap = lambda k: laps.__setitem__(k, round(time.time() - T0, 1))
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    R = {"seed": SEED, "primary": "VTOP5", "configs": {**cfg, "ew": ew_cfg}, "windows": {"short": "weeks 16-20", "long": "weeks 11-20"}}

    # ================================================================== setup (as steps 7-11)
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
    R["users"] = {"universe": n_u, "tune": int(len(tune_rows)), "report": int(len(report_rows)), "fit_all_with_vtop5": int(len(folds["A"]["train"]) + len(folds["B"]["train"]))}
    lap("setup")

    def window_features(ctx, label):
        store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
        store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
        cand = rerank.candidates(store, ctx["pop_norm"], ctx["n_u"], ctx["n_p"], names=rerank.ORDER)
        ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
        print(f"features {label}: {len(ri):,} pairs", flush=True)
        return store, ri, ci, X

    store10, ri10, ci10, X10 = window_features(ctx10, "weeks 11-20")
    lap("features_11-20")
    ctx5 = window.restrict(ctx10, 16, 20, uc_pairs)
    store5, ri5, ci5, X5 = window_features(ctx5, "weeks 16-20")
    lap("features_16-20")
    y10, y5 = T5[ri10, ci10].astype(int), T5[ri5, ci5].astype(int)
    X10pw, X5pw = rerank2.per_week(X10, W_LONG, 10), rerank2.per_week(X5, W_SHORT, 10)   # target = weeks 21-30 = 10 weeks in both

    def split(rows):
        p = np.random.default_rng(SEED).permutation(rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    def ap_all(top, ev):
        return {v: metrics.ap5_matrix(top, T[v][ev])[ne[v][ev]] for v in VARS}

    # ================================================================== validation analogue
    res = {f: {} for f in FOLDS}
    best_iter = {"short": {}, "long": {}}
    for f in FOLDS:
        ev = folds[f]["eval"]
        tr, va = split(folds[f]["train"])
        m5 = rerank2.train(X5pw, y5, ri5, tr, va, rerank2.BASE_PARAMS)
        m10 = rerank2.train(X10pw, y10, ri10, tr, va, rerank2.BASE_PARAMS)
        best_iter["short"][f], best_iter["long"][f] = int(m5.best_iteration), int(m10.best_iteration)
        res[f]["a_5w_fit_5w_infer"] = ap_all(rerank.top5(m5, X5pw[list(X5pw.columns)], ri5, ci5, ev), ev)
        res[f]["b_5w_fit_10w_infer"] = ap_all(rerank.top5(m5, X10pw[list(X5pw.columns)], ri10, ci10, ev), ev)
        res[f]["ref_10w_fit_10w_infer"] = ap_all(rerank.top5(m10, X10pw, ri10, ci10, ev), ev)
        nz = ne5[ev]
        res[f]["h2_missing_users"] = {"a_weeks_16_20": int((~store5["H2"][1][ev][nz]).sum()), "b_weeks_11_20": int((~store10["H2"][1][ev][nz]).sum()), "eval_users_with_vtop5": int(nz.sum())}
        print(f, {k: round(float(v["VTOP5"].mean()), 4) for k, v in res[f].items() if k != "h2_missing_users"}, res[f]["h2_missing_users"], flush=True)
    lap("validation_fits")
    R["best_iterations"] = best_iter
    R["table"] = {f: {k: ({v: float(a.mean()) for v, a in d.items()} if k != "h2_missing_users" else d) for k, d in res[f].items()} for f in FOLDS}
    rng = np.random.default_rng(SEED)
    diff = {f: res[f]["b_5w_fit_10w_infer"]["VTOP5"] - res[f]["a_5w_fit_5w_infer"]["VTOP5"] for f in FOLDS}
    R["bootstrap_b_minus_a_vtop5"] = {"A": boot(diff["A"], rng), "B": boot(diff["B"], rng), "pooled": boot(np.r_[diff["A"], diff["B"]], rng)}
    R["bootstrap_b_minus_a_robustness"] = {v: boot(np.r_[res["A"]["b_5w_fit_10w_infer"][v] - res["A"]["a_5w_fit_5w_infer"][v],
                                                         res["B"]["b_5w_fit_10w_infer"][v] - res["B"]["a_5w_fit_5w_infer"][v]], rng) for v in VARS}
    both = all(R["table"][f]["b_5w_fit_10w_infer"]["VTOP5"] > R["table"][f]["a_5w_fit_5w_infer"]["VTOP5"] for f in FOLDS)
    ci_ok = R["bootstrap_b_minus_a_vtop5"]["pooled"]["ci95"][0] > 0
    branch = "long_window" if (both and ci_ok) else "weeks_21_30_per_week"
    R["rule"] = ("(b) beats (a) on VTOP5 in BOTH folds and the pooled paired-bootstrap 95% CI of (b)-(a) excludes zero -> final run = per-week reranker trained on "
                 "weeks 11-20 features, applied to features and candidates from weeks 11-30; otherwise per-week reranker applied to weeks 21-30 as in step 11.")
    R["decision"] = {"b_beats_a_in_both_folds": bool(both), "pooled_ci_excludes_zero": bool(ci_ok), "branch": branch}
    print("decision", R["decision"], flush=True)

    # ================================================================== final model: per-week reranker on ALL validation users, weeks 11-20 features
    rounds = int(round(np.mean(list(best_iter["long"].values()))))
    all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
    model = rerank2.train(X10pw, y10, ri10, all_train, None, rerank2.BASE_PARAMS, rounds=rounds)
    R["final_model"] = {"fit_users": int(len(all_train)), "trees": rounds, "fold_best_iterations_11_20_fit": best_iter["long"]}

    users_f = data.load_submission_users()
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    nf, npf = len(users_f), len(pool_f)
    periods, wf = (["11-20", "21-30"], 20) if branch == "long_window" else (["21-30"], 10)
    ctxf = s7.make_ctx(periods, "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
    store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
    store_f["EW"] = s7.build_store(ctxf, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
    rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
    Xfpw = rerank2.per_week(Xf, wf, 12)[list(X10pw.columns)]                      # target weeks 31-42 = 12
    top = rerank.top5(model, Xfpw, rif, cif, np.arange(nf))
    preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
    path = SUBDIR / "submission_STEP15.csv"
    write_submission(preds, users_f, path)
    prev = pd.read_csv(SUBDIR / "submission_STEP7_all.csv", dtype=str).set_index("user_id")["tv_show_id"]
    shared = [len(set(preds[u]) & set(int(x) for x in prev[str(u)].split())) / K for u in users_f]
    pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
    R["final"] = {"branch": branch, "feature_window_weeks": wf, "target_weeks": 12, **s3.check_file(path, users_f, pool_f),
                  "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in store_f},
                  "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f])),
                  "share_of_ids_shared_with_STEP7_all": float(np.mean(shared)), "share_of_users_with_identical_list": float(np.mean([s == 1.0 for s in shared]))}
    lap("final")
    R["runtime_s"], R["laps_s"] = round(time.time() - T0, 1), laps
    (OUT / "validation_step15.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_reports(R)
    print(json.dumps({k: R[k] for k in ("decision", "final_model", "final", "runtime_s")}, indent=1, default=str))


def write_reports(R):
    t, bs, d = R["table"], R["bootstrap_b_minus_a_vtop5"], R["decision"]
    rows = [("(a) fit 16-20, infer 16-20", "a_5w_fit_5w_infer"), ("(b) fit 16-20, infer 11-20", "b_5w_fit_10w_infer"), ("reference: fit 11-20, infer 11-20", "ref_10w_fit_10w_infer")]
    md = ["# Step 15 - use the full history at inference, validated by an analogue", "",
          "Per-week reranker = step-7 reranker with window-dependent counts divided by the number of weeks of their window (`rerank2.per_week`), step-7 hyperparameters, seed 42, "
          "lambdarank, early stopping on 20% of the training users. Analogue of the final run: fit on features from weeks 16-20 only (labels VTOP5 of weeks 21-30, candidates from the same window), "
          "infer with features/candidates from (a) weeks 16-20 or (b) weeks 11-20 (H2 decay unchanged). 2-fold swap (A: fit TUNE, score REPORT; B: the reverse). Primary criterion VTOP5.", "",
          "## Validation (MAP@5, users with non-empty truth of the respective type)", "",
          "| variant | fold | " + " | ".join(VARS) + " |", "|---|---|---|---|---|---|"]
    for lab, k in rows:
        for f in FOLDS:
            md.append(f"| {lab} | {f} | " + " | ".join(f"{t[f][k][v]:.4f}" for v in VARS) + " |")
    md += ["", "## Paired bootstrap 95% CI of (b) - (a), VTOP5 (2,000 resamples, seed 42)", "", "| scope | users | mean diff | 95% CI |", "|---|---|---|---|"]
    for k in ("A", "B", "pooled"):
        b = bs[k]
        md.append(f"| {k} | {b['n_users']} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "Robustness, pooled (b) - (a):", "", "| truth | mean diff | 95% CI |", "|---|---|---|"]
    for v in VARS:
        b = R["bootstrap_b_minus_a_robustness"][v]
        md.append(f"| {v} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "## Evaluation users without H2 data (no scheduled channel watched in the feature window, popularity fallback)", "", "| fold | evaluation users with VTOP5 truth | (a) weeks 16-20 | (b) weeks 11-20 |", "|---|---|---|---|"]
    for f in FOLDS:
        m = t[f]["h2_missing_users"]
        md.append(f"| {f} | {m['eval_users_with_vtop5']} | {m['a_weeks_16_20']} | {m['b_weeks_11_20']} |")
    fi = R["final"]
    md += ["", "## Pre-declared rule and decision", "", R["rule"], "",
           f"- (b) > (a) in fold A: {t['A']['b_5w_fit_10w_infer']['VTOP5'] > t['A']['a_5w_fit_5w_infer']['VTOP5']}; fold B: {t['B']['b_5w_fit_10w_infer']['VTOP5'] > t['B']['a_5w_fit_5w_infer']['VTOP5']}; "
           f"pooled CI excludes zero: {d['pooled_ci_excludes_zero']}.",
           f"- **Branch taken: {'long window (weeks 11-30 features)' if d['branch'] == 'long_window' else 'weeks 21-30 per-week (as step 11 with per-week features)'}**.", "",
           "## Shipped file `outputs/submissions/submission_STEP15.csv`", "",
           f"- Model: per-week reranker, features from weeks 11-20, fitted on all {R['final_model']['fit_users']:,} validation users with VTOP5 truth, {R['final_model']['trees']} trees "
           f"(mean of the fold best iterations {R['final_model']['fold_best_iterations_11_20_fit']}). Inference features/candidates from a {fi['feature_window_weeks']}-week window, "
           f"n_target_airings divided by {fi['target_weeks']} (weeks 31-42).",
           f"- Format: header {fi['header_ok']}, rows {fi['rows']}, user order {fi['user_order_ok']}, 5 distinct ids {fi['five_distinct_ids']}, ids in final pool {fi['all_ids_in_final_pool']}.",
           f"- Users without component data (popularity fallback): {fi['users_without_component_data']}.",
           f"- Repeat share (ids the user already watched >= 300 s in the feature window): {fi['repeat_share_train_positive']:.3f}.",
           f"- Share of ids identical to submission_STEP7_all.csv: {fi['share_of_ids_shared_with_STEP7_all']:.3f} (identical lists for {fi['share_of_users_with_identical_list']:.1%} of users).",
           f"- Runtime of this script: {R['runtime_s']} s.", "",
           "## Summary", ""]
    a, b = t["A"], t["B"]
    pooled = bs["pooled"]
    md.append(f"Inferring with 10 weeks instead of the 5 weeks used for fitting gave VTOP5 {b['A']['b_5w_fit_10w_infer']['VTOP5'] if False else a['b_5w_fit_10w_infer']['VTOP5']:.4f} vs {a['a_5w_fit_5w_infer']['VTOP5']:.4f} in fold A and "
              f"{b['b_5w_fit_10w_infer']['VTOP5']:.4f} vs {b['a_5w_fit_5w_infer']['VTOP5']:.4f} in fold B; the pooled difference is {pooled['mean_diff']:+.4f} with 95% CI [{pooled['ci95'][0]:+.4f}, {pooled['ci95'][1]:+.4f}]. "
              f"Users without H2 data fell from {a['h2_missing_users']['a_weeks_16_20'] + b['h2_missing_users']['a_weeks_16_20']} to {a['h2_missing_users']['b_weeks_11_20'] + b['h2_missing_users']['b_weeks_11_20']} "
              f"of {a['h2_missing_users']['eval_users_with_vtop5'] + b['h2_missing_users']['eval_users_with_vtop5']} evaluation users. "
              f"The rule gave the branch '{d['branch']}', so submission_STEP15.csv uses a {fi['feature_window_weeks']}-week inference window. "
              f"Its ids coincide with submission_STEP7_all.csv for {fi['share_of_ids_shared_with_STEP7_all']:.1%} of the slots; this file has no validation score and no Kaggle result yet.")
    (OUT / "validation_step15.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    sm = SUBDIR / "SUBMISSIONS.md"
    txt = sm.read_text(encoding="utf-8")
    if "submission_STEP15.csv" not in txt:
        sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP15.csv | step 15: per-week reranker (step-7 features divided by window weeks), fit on weeks 11-20 features; branch '{d['branch']}' | "
                      f"{R['final_model']['trees']} trees, all validation users; inference window {fi['feature_window_weeks']} weeks, target 12 weeks; cannot be scored on validation | not submitted | not submitted |\n", encoding="utf-8")


if __name__ == "__main__":
    main()

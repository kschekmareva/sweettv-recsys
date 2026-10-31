"""Step 8: improve the step-7 reranker (per-week features, CH-SLOT candidates, extra features, graded labels, small LightGBM search),
evaluated with a 2-fold swap (TUNE <-> REPORT). Primary criterion VTOP5.
Usage: python src/run_step8.py
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
import hybrid
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
SEARCH_GRID = [dict(num_leaves=nl, learning_rate=lr, min_data_in_leaf=mc, feature_fraction=ff)
               for nl in (15, 31, 63) for lr in (0.03, 0.1) for mc in (20, 100) for ff in (0.7, 1.0)]
STEPS = [("S7: step-7 reranker", "raw", "c6", "bin"), ("+1 per-week features", "pw", "c6", "bin"),
         ("+2 CH-SLOT candidates and features", "pw_slot", "c7", "bin"), ("+3 extra features", "pw_slot_extra", "c7", "bin"),
         ("+4 graded labels", "pw_slot_extra", "c7", "gr")]
TUNED = "+5 tuned LightGBM (small search)"


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
    R = {"seed": SEED, "primary": "VTOP5", "configs": {**cfg, "ew": ew_cfg}}

    # ================================================================== setup (same data, truth, halves as step 7)
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
    ctx = s7.make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    rerank2.add_target_rows(ctx, "21-30", (21, 30))
    ctx["weeks_feat"], ctx["weeks_target"] = 10, 10
    uix, pix = ctx["uix"], ctx["pix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    T5 = T["VTOP5"]
    Tall = s4.bool_matrix(ctx["train_all"], uix, pix, n_u, n_p)
    tp = vpairs["VTOP5"].copy()
    tp["rel"] = 5 - tp.groupby("user_id").cumcount()                          # graded relevance 5..1 by position in the VTOP5 truth
    Rel = np.zeros((n_u, n_p), dtype=np.int8)
    Rel[uix.get_indexer(tp["user_id"].values), pix.get_indexer(tp["tv_show_id"].values)] = tp["rel"].values
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    folds = {"A": {"desc": "fit TUNE -> score REPORT", "train": tune_rows[nonempty["VTOP5"][tune_rows]], "eval": report_rows},
             "B": {"desc": "fit REPORT -> score TUNE", "train": report_rows[nonempty["VTOP5"][report_rows]], "eval": tune_rows}}
    pop_norm = ctx["pop_norm"]
    lap("setup")

    # ================================================================== components, candidates (step-7 union + CH-SLOT)
    print("components", flush=True)
    store, times = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"], times["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"], (0, 0)
    cand6 = rerank.candidates(store, pop_norm, n_u, n_p, names=rerank.ORDER)
    counts, slot_s = rerank2.slot_counts(ctx)
    cand_slot = rerank2.slot_candidates(counts, pop_norm)
    cand7 = cand6 | cand_slot
    lap("components")

    allrows = np.flatnonzero(nonempty["VTOP5"])
    unseen = T5 & ~Tall
    rec = lambda M, rows: {"all_truth": float((T5[rows] & M[rows]).sum() / T5[rows].sum()), "unseen_truth": float((unseen[rows] & M[rows]).sum() / unseen[rows].sum())}

    def oracle(M, rows, v):
        h = np.minimum(K, (T[v][rows] & M[rows]).sum(axis=1))
        den = np.minimum(K, T[v][rows].sum(axis=1))
        mk = den > 0
        return float((h[mk] / den[mk]).mean())
    rep_truth = report_rows[nonempty["VTOP5"][report_rows]]
    R["candidates"] = {
        "unseen_share_of_truth_items": float(unseen[allrows].sum() / T5[allrows].sum()),
        "mean_candidates_per_user": {"step7_union_6comp": float(cand6[allrows].sum(1).mean()), "ch_slot_only": float(cand_slot[allrows].sum(1).mean()), "union_with_ch_slot": float(cand7[allrows].sum(1).mean())},
        "recall_all_users": {"step7_union_6comp": rec(cand6, allrows), "ch_slot_only": rec(cand_slot, allrows), "union_with_ch_slot": rec(cand7, allrows)},
        "recall_report": {"step7_union_6comp": rec(cand6, rep_truth), "ch_slot_only": rec(cand_slot, rep_truth), "union_with_ch_slot": rec(cand7, rep_truth)},
        "oracle_report": {n: {v: oracle(M, report_rows, v) for v in VARS} for n, M in (("step7_union_6comp", cand6), ("union_with_ch_slot", cand7))},
        "slot_seconds": round(slot_s, 1)}
    print("recall", R["candidates"]["recall_all_users"], flush=True)
    lap("candidates")

    # ================================================================== features
    print("features", flush=True)
    ri, ci, Xb, f_s = rerank.make_features(ctx, store, pop_norm, cand7)
    E, e_s = rerank2.extras(ctx, ri, ci, counts, pop_norm)
    mask6 = cand6[ri, ci]
    y_bin, y_gr = T5[ri, ci].astype(np.int8), Rel[ri, ci].astype(np.int8)
    Xs = {lv: rerank2.assemble(Xb, E, lv, ctx["weeks_feat"], ctx["weeks_target"]) for lv in ("raw", "pw", "pw_slot", "pw_slot_extra")}
    R["features"] = {"pairs_cand7": int(len(ri)), "pairs_cand6": int(mask6.sum()), "base_feature_seconds": round(f_s, 1), "extra_feature_seconds": round(e_s, 1),
                     "columns_final": list(Xs["pw_slot_extra"].columns)}
    cache = {}

    def view(cl, rl, lab):
        key = (cl, rl, lab)
        if key not in cache:
            m = mask6 if rl == "c6" else np.ones(len(ri), dtype=bool)
            cache.clear()
            cache[key] = (Xs[cl][m].reset_index(drop=True), (y_gr if lab == "gr" else y_bin)[m].astype(int), ri[m], ci[m])
        return cache[key]
    lap("features")

    def split(train_rows):
        p = np.random.default_rng(SEED).permutation(train_rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    def evaluate(top, rows):
        return {v: (metrics.ap5_matrix(top, T[v][rows]), nonempty[v][rows]) for v in VARS}

    # ================================================================== ablation ladder (2-fold swap)
    res, times_fit = {}, {}
    for name, cl, rl, lab in STEPS:
        X, y, ri_, ci_ = view(cl, rl, lab)
        res[name], times_fit[name] = {}, 0.0
        for f, fd in folds.items():
            t = time.time()
            tr, va = split(fd["train"])
            model = rerank2.train(X, y, ri_, tr, va, rerank2.BASE_PARAMS)
            top = rerank.top5(model, X, ri_, ci_, fd["eval"])
            res[name][f] = evaluate(top, fd["eval"])
            res[name][f + "_iter"] = int(model.best_iteration)
            times_fit[name] += time.time() - t
        print(name, {f: round(float(res[name][f]["VTOP5"][0][res[name][f]["VTOP5"][1]].mean()), 4) for f in folds}, flush=True)
    chk = float(res[STEPS[0][0]]["A"]["VTOP5"][0][res[STEPS[0][0]]["A"]["VTOP5"][1]].mean())
    R["step7_reproduction"] = {"fold_A_vtop5_now": chk, "step7_report": J7["rows"]["reranker"]["VTOP5"]["report"], "abs_diff": abs(chk - J7["rows"]["reranker"]["VTOP5"]["report"])}
    lap("ablation")

    # ================================================================== +5: small LightGBM search per fold (inner 20% split of the fold's training users)
    print("lightgbm search", flush=True)
    X, y, ri_, ci_ = view("pw_slot_extra", "c7", "gr")
    search, best_models = {}, {}
    for f, fd in folds.items():
        tr, va = split(fd["train"])
        rows_f = []
        for i, g in enumerate(SEARCH_GRID):
            t = time.time()
            model = rerank2.train(X, y, ri_, tr, va, {**rerank2.BASE_PARAMS, **g}, rounds=1000)
            mp = float(metrics.ap5_matrix(rerank.top5(model, X, ri_, ci_, va), T5[va]).mean())
            rows_f.append({**g, "holdout_map5_vtop5": mp, "best_iteration": int(model.best_iteration), "seconds": round(time.time() - t, 1)})
            if f not in best_models or mp > best_models[f][0]:
                best_models[f] = (mp, i, model)
        search[f] = rows_f
        print("  fold", f, "best", max(rows_f, key=lambda r: r["holdout_map5_vtop5"]), flush=True)
    res[TUNED] = {}
    for f, fd in folds.items():
        model = best_models[f][2]
        res[TUNED][f] = evaluate(rerank.top5(model, X, ri_, ci_, fd["eval"]), fd["eval"])
        res[TUNED][f + "_iter"] = int(model.best_iteration)
    mean_hold = [np.mean([search["A"][i]["holdout_map5_vtop5"], search["B"][i]["holdout_map5_vtop5"]]) for i in range(len(SEARCH_GRID))]
    ibest = int(np.argmax(mean_hold))
    final_cfg = SEARCH_GRID[ibest]
    final_rounds = int(round(np.mean([search["A"][ibest]["best_iteration"], search["B"][ibest]["best_iteration"]])))
    imp = sum(rerank2.importance(best_models[f][2], X.columns) for f in folds) / 2
    R["search"] = {"grid": SEARCH_GRID, "per_fold": search, "fold_best_config": {f: SEARCH_GRID[best_models[f][1]] for f in folds},
                   "final_config": final_cfg, "final_rounds": final_rounds, "rule": "config with the best mean inner-split MAP@5 (VTOP5) over the two folds; rounds = mean best iteration of that config"}
    R["importance_top15_gain_share"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(15).items()}
    lap("search")

    # ================================================================== tables, bootstrap, shipping
    def mean_of(name, v, f=None):
        fs = [f] if f else ["A", "B"]
        return float(np.mean([res[name][x][v][0][res[name][x][v][1]].mean() for x in fs]))
    ablation = {}
    names = [s[0] for s in STEPS] + [TUNED]
    for n in names:
        ablation[n] = {"A_VTOP5": mean_of(n, "VTOP5", "A"), "B_VTOP5": mean_of(n, "VTOP5", "B"),
                       **{f"mean_{v}": mean_of(n, v) for v in VARS}, "fit_seconds": round(times_fit.get(n, sum(r["seconds"] for f in search for r in search[f])), 1)}
    R["ablation"] = ablation
    R["fold_tables"] = {n: {f: {v: float(res[n][f][v][0][res[n][f][v][1]].mean()) for v in VARS} for f in folds} for n in (names[0], TUNED)}
    rng = np.random.default_rng(SEED)
    base, new = names[0], TUNED
    sig = {v: {} for v in VARS}
    for v in VARS:
        diffs = {}
        for f in folds:
            (a1, mk), (a0, _) = res[new][f][v], res[base][f][v]
            diffs[f] = (a1 - a0)[mk]
            sig[v][f] = boot(diffs[f], rng)
        sig[v]["pooled"] = boot(np.r_[diffs["A"], diffs["B"]], rng)
    R["bootstrap_new_vs_step7"] = sig
    ship = bool(sig["VTOP5"]["A"]["mean_diff"] > 0 and sig["VTOP5"]["B"]["mean_diff"] > 0 and sig["VTOP5"]["pooled"]["ci95"][0] > 0)
    R["decision"] = {"ship_new": ship, "rule": "ship only if the new reranker beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero"}
    lap("evaluation")

    # ================================================================== final model on ALL validation users, applied to weeks 21-30 features
    if ship:
        print("final", flush=True)
        t = time.time()
        all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
        model_f = rerank2.train(X, y, ri_, all_train, None, {**rerank2.BASE_PARAMS, **final_cfg}, rounds=final_rounds)
        users_f = data.load_submission_users()
        pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
        ctxf = s7.make_ctx(["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
        rerank2.add_target_rows(ctxf, "31-42", (31, 42))
        ctxf["weeks_feat"], ctxf["weeks_target"] = 10, 12
        store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
        store_f["EW"] = s7.build_store(ctxf, ["EW"], cfg, ew_cfg)[0]["EW"]
        nf, npf = len(users_f), len(pool_f)
        cand6_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
        counts_f, _ = rerank2.slot_counts(ctxf)
        cand7_f = cand6_f | rerank2.slot_candidates(counts_f, ctxf["pop_norm"])
        rif, cif, Xbf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand7_f)
        Ef, _ = rerank2.extras(ctxf, rif, cif, counts_f, ctxf["pop_norm"])
        Xf = rerank2.assemble(Xbf, Ef, "pw_slot_extra", ctxf["weeks_feat"], ctxf["weeks_target"])[list(X.columns)]
        top = rerank.top5(model_f, Xf, rif, cif, np.arange(nf))
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / "submission_STEP8.csv"
        write_submission(preds, users_f, path)
        pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
        slot_share = float(np.mean([np.isin(top[i], np.flatnonzero(cand7_f[i] & ~cand6_f[i])).mean() for i in range(nf)]))
        R["final"] = {**s3.check_file(path, users_f, pool_f), "train_window": ["21-30"], "weeks_feat": 10, "weeks_target": 12, "final_config": final_cfg, "rounds": final_rounds,
                      "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in store_f},
                      "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f])),
                      "share_of_recommended_from_ch_slot_only_candidates": slot_share, "seconds": round(time.time() - t, 1),
                      "importance_top15": {k: round(float(v), 4) for k, v in rerank2.importance(model_f, Xf.columns).sort_values(ascending=False).head(15).items()}}
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP8.csv" not in txt:
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP8.csv | step 8: LightGBM lambdarank reranker (graded labels, per-week features, CH-SLOT candidates, extra features) | "
                          f"{final_cfg}, {final_rounds} trees; fitted on all validation users (features weeks 11-20, labels weeks 21-30), applied to features from weeks 21-30 and the 31-42 schedule | | |\n", encoding="utf-8")
    R["laps_seconds"], R["runtime_s"] = laps, round(time.time() - T0, 1)
    (OUT / "validation_step8.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_md(R, names)
    print(f"done in {R['runtime_s']} s; ship {ship}")


def write_md(R, names):
    f4 = lambda x: f"{x:.4f}"
    V = VARS
    c = R["candidates"]
    md = ["# Validation, step 8", "",
          "Primary criterion VTOP5; V300/V900/VHALF are robustness checks. Same pool, repeats allowed, 5 distinct ids, popularity fill, seed 42. "
          "Evaluation is a 2-fold swap: fold A = fit on TUNE users, score REPORT users; fold B = fit on REPORT users, score TUNE users; 'mean' = average of the two folds. "
          "Caveat: step-5/7 component hyperparameters (H2 c, MF, EW) were chosen on TUNE, so fold B's features carry a small optimism. No Kaggle number is used.", "",
          f"Step-7 reproduction (fold A, VTOP5): {R['step7_reproduction']['fold_A_vtop5_now']:.4f} vs {R['step7_reproduction']['step7_report']:.4f} in the step-7 report (abs diff {R['step7_reproduction']['abs_diff']:.2e}).", "",
          "## 1. Candidate sources (VTOP5 truth)", "",
          f"Share of VTOP5 truth items the user never watched in TRAIN: {c['unseen_share_of_truth_items']:.4f}. Mean candidates per user: {c['mean_candidates_per_user']}.", "",
          "| candidate set | recall of all truth items (all users / REPORT) | recall of truth items unseen in TRAIN (all users / REPORT) |", "|---|---|---|"]
    for k in ("step7_union_6comp", "ch_slot_only", "union_with_ch_slot"):
        a, r = c["recall_all_users"][k], c["recall_report"][k]
        md.append(f"| {k} | {f4(a['all_truth'])} / {f4(r['all_truth'])} | {f4(a['unseen_truth'])} / {f4(r['unseen_truth'])} |")
    md += ["", "Oracle ceiling on REPORT (perfect ranking of the candidates): " + "; ".join(f"{k}: " + ", ".join(f"{v} {f4(x)}" for v, x in d.items()) for k, d in c["oracle_report"].items()), "",
           "## 2. Ablation (cumulative changes, 2-fold swap)", "",
           "| step | fold A VTOP5 | fold B VTOP5 | mean VTOP5 | change in mean VTOP5 | mean V300 | mean V900 | mean VHALF | fit s (2 folds) |", "|---|---|---|---|---|---|---|---|---|"]
    prev = None
    for n in names:
        a = R["ablation"][n]
        delta = "" if prev is None else "%+.4f" % (a["mean_VTOP5"] - prev)
        md.append(f"| {n} | {f4(a['A_VTOP5'])} | {f4(a['B_VTOP5'])} | {f4(a['mean_VTOP5'])} | {delta} | {f4(a['mean_V300'])} | {f4(a['mean_V900'])} | {f4(a['mean_VHALF'])} | {a['fit_seconds']} |")
        prev = a["mean_VTOP5"]
    md += ["", "Step +1 divides count/seconds features by the number of weeks of their window; in validation both windows have 10 weeks, so it is a constant rescaling and trees are unchanged - "
               "its effect is on the final run, where the target window has 12 weeks.", "",
           "Per fold, all four variants:", "", "| model | fold | " + " | ".join(V) + " |", "|---|---|" + "---|" * len(V)]
    for n, d in R["fold_tables"].items():
        for f, vals in d.items():
            md.append(f"| {n} | {f} | " + " | ".join(f4(vals[v]) for v in V) + " |")
    md += ["", "## 3. LightGBM search (inner 20% split of each fold's training users; holdout MAP@5 VTOP5)", ""]
    s = R["search"]
    for f in ("A", "B"):
        top3 = sorted(s["per_fold"][f], key=lambda r: -r["holdout_map5_vtop5"])[:3]
        md.append(f"- fold {f} best three: " + "; ".join(f"leaves {r['num_leaves']}, lr {r['learning_rate']}, min_child {r['min_data_in_leaf']}, ff {r['feature_fraction']} -> {f4(r['holdout_map5_vtop5'])} ({r['best_iteration']} trees)" for r in top3))
    md += ["", f"Final config (best mean over the two folds): {s['final_config']}, {s['final_rounds']} trees. Rule: {s['rule']}.", "",
           "## 4. Paired bootstrap, new reranker vs step-7 reranker (2,000 resamples, seed 42): mean AP@5 difference [95% CI]", "",
           "| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |", "|---|---|---|---|"]
    for v in V:
        d = R["bootstrap_new_vs_step7"][v]
        md.append(f"| {v} | " + " | ".join(f"{d[k]['mean_diff']:+.4f} [{d[k]['ci95'][0]:+.4f}, {d[k]['ci95'][1]:+.4f}]" for k in ("A", "B", "pooled")) + " |")
    md += ["", "## 5. Feature importances, top 15 (gain share, average of the two fold models)", "", ", ".join(f"{k} {v}" for k, v in R["importance_top15_gain_share"].items()), "",
           "## 6. Shipping decision", "", f"Rule: {R['decision']['rule']}. Ship: **{R['decision']['ship_new']}**.", ""]
    if "final" in R:
        f = R["final"]
        md += [f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}",
               f"- model fitted on all validation users with {f['final_config']}, {f['rounds']} trees; features from weeks 21-30 (10 weeks) and the 31-42 schedule (12 weeks, per-week scaling); no re-fitting on test",
               f"- users without component data (popularity fallback): {f['users_without_component_data']}; repeat share {f['repeat_share_train_positive']:.3f}; "
               f"share of recommended ids that come only from CH-SLOT candidates: {f['share_of_recommended_from_ch_slot_only_candidates']:.3f}",
               f"- top-15 importances of the final model: {f['importance_top15']}"]
    else:
        md += ["No new file written; `submission_STEP7.csv` stays the latest reranker file."]
    md += ["", f"Runtime: {R['runtime_s']} s (stage clocks: {R['laps_seconds']})"]
    (OUT / "validation_step8.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

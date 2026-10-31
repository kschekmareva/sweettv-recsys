"""Step 8b: seed/fold bagging of the step-7 reranker (step-7 features and settings) and rank fusion with the step-6 blend.
Primary criterion VTOP5. Usage: python src/run_step8b.py
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

SEEDS = [42, 43, 44, 45, 46]
W_GRID = [0.0, 0.1, 0.2, 0.3]
K = 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
FAMILY = "P-air+MF+PPR+H2"


def pct_by_user(r, s):
    """Rank-normalised score within each user's candidates, in (0, 1], higher is better."""
    return pd.Series(s).groupby(r).rank(pct=True, method="average").values


def top5_of(r, c, sc, rows):
    m = np.isin(r, rows)
    r, c, sc = r[m], c[m], sc[m]
    o = np.lexsort((-sc, r))
    r, c = r[o], c[o]
    first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    assert len(first) == len(rows) and (np.diff(np.r_[first, len(r)]) >= K).all()
    return np.stack([c[first + j] for j in range(K)], axis=1)


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
    w6 = dict(J5["rows"][FAMILY]["weights"]["VTOP5"])
    R = {"seeds": SEEDS, "primary": "VTOP5", "w_grid": W_GRID, "step6_blend_weights": w6}

    # ================================================================== setup (as step 7)
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
    uix, pix = ctx["uix"], ctx["pix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    T5 = T["VTOP5"]
    perm = np.random.default_rng(42).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    folds = {"A": {"train": tune_rows[nonempty["VTOP5"][tune_rows]], "eval": report_rows},
             "B": {"train": report_rows[nonempty["VTOP5"][report_rows]], "eval": tune_rows}}
    pop_norm = ctx["pop_norm"]
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand6 = rerank.candidates(store, pop_norm, n_u, n_p, names=rerank.ORDER)
    ri, ci, X, f_s = rerank.make_features(ctx, store, pop_norm, cand6)          # step-7 features, unchanged
    y = T5[ri, ci].astype(int)
    S6 = s4.fuse_eps(pop_norm, [store[n] for n in ("H2", "MF", "PPR")], [w6[n] for n in ("H2", "MF", "PPR")])
    blend_norm = pct_by_user(ri, S6[ri, ci])                                       # step-6 blend, rank-normalised within candidates
    del S6
    lap("setup")

    def split(train_rows, seed):
        p = np.random.default_rng(seed).permutation(train_rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    # ================================================================== 1. seed bagging x 2-fold swap
    print("bagging", flush=True)
    bag, single, oob_sum, oob_n, best_iter = {}, {}, {}, {}, {f: {} for f in folds}
    fit_s = 0.0
    for f, fd in folds.items():
        m_ev = np.isin(ri, fd["eval"])
        bag[f] = np.zeros(m_ev.sum())
        oob_sum[f], oob_n[f] = np.zeros(len(ri)), np.zeros(len(ri))
        for s in SEEDS:
            t = time.time()
            tr, va = split(fd["train"], s)
            model = rerank2.train(X, y, ri, tr, va, {**rerank2.BASE_PARAMS, "seed": s})
            sc = pct_by_user(ri[m_ev], model.predict(X[m_ev]))
            bag[f] += sc / len(SEEDS)
            if s == 42:
                single[f] = model.predict(X[m_ev])
            m_va = np.isin(ri, va)
            oob_sum[f][m_va] += pct_by_user(ri[m_va], model.predict(X[m_va]))
            oob_n[f][m_va] += 1
            best_iter[f][s] = int(model.best_iteration)
            fit_s += time.time() - t
        print("  fold", f, "done", flush=True)
    lap("bagging")

    # ================================================================== 2. rank fusion with the step-6 blend; w chosen out-of-bag inside each fold
    def evaluate(top, rows):
        out = {}
        for v in VARS:
            ap, mk = metrics.ap5_matrix(top, T[v][rows]), nonempty[v][rows]
            out[v] = (ap, mk)
        return out
    res = {"step7 (single seed 42)": {}, "bagged (w=0)": {}, "bagged + step-6 fusion (w chosen out-of-bag)": {}}
    chosen_w, oob_table, eval_by_w = {}, {}, {}
    for f, fd in folds.items():
        m_ev = np.isin(ri, fd["eval"])
        r_ev, c_ev, b_ev = ri[m_ev], ci[m_ev], blend_norm[m_ev]
        res["step7 (single seed 42)"][f] = evaluate(top5_of(r_ev, c_ev, single[f], fd["eval"]), fd["eval"])
        res["bagged (w=0)"][f] = evaluate(top5_of(r_ev, c_ev, bag[f], fd["eval"]), fd["eval"])
        m_o = oob_n[f] > 0
        rows_o = np.unique(ri[m_o])
        oob_mean = oob_sum[f][m_o] / oob_n[f][m_o]
        scores = {}
        for w in W_GRID:
            top = top5_of(ri[m_o], ci[m_o], (1 - w) * oob_mean + w * blend_norm[m_o], rows_o)
            scores[w] = float(metrics.ap5_matrix(top, T5[rows_o]).mean())
        wbest = max(W_GRID, key=lambda w: (scores[w], -w))
        chosen_w[f], oob_table[f] = wbest, {"users": int(len(rows_o)), "oob_map5_vtop5_by_w": scores}
        res["bagged + step-6 fusion (w chosen out-of-bag)"][f] = evaluate(top5_of(r_ev, c_ev, (1 - wbest) * bag[f] + wbest * b_ev, fd["eval"]), fd["eval"])
        eval_by_w[f] = {w: float(metrics.ap5_matrix(top5_of(r_ev, c_ev, (1 - w) * bag[f] + w * b_ev, fd["eval"]), T5[fd["eval"]])[nonempty["VTOP5"][fd["eval"]]].mean()) for w in W_GRID}
    R["fusion"] = {"chosen_w": chosen_w, "oob_selection": oob_table, "eval_half_vtop5_by_w_DESCRIPTIVE_ONLY_not_used_for_choice": eval_by_w,
                   "selection_note": "w is chosen on out-of-bag predictions of the fold's training users (users in a seed's inner 20% holdout, scored by that seed's model only)"}
    lap("fusion")

    # ================================================================== tables, bootstrap, rule
    mean_of = lambda name, v, f=None: float(np.mean([res[name][x][v][0][res[name][x][v][1]].mean() for x in ([f] if f else ["A", "B"])]))
    table = {n: {"A_VTOP5": mean_of(n, "VTOP5", "A"), "B_VTOP5": mean_of(n, "VTOP5", "B"), **{f"mean_{v}": mean_of(n, v) for v in VARS}} for n in res}
    R["table"] = table
    R["step7_reproduction"] = {"fold_A_vtop5": table["step7 (single seed 42)"]["A_VTOP5"], "step7_report": J7["rows"]["reranker"]["VTOP5"]["report"]}
    rng = np.random.default_rng(42)
    sig = {}
    for n in ("bagged (w=0)", "bagged + step-6 fusion (w chosen out-of-bag)"):
        sig[n] = {}
        for v in VARS:
            d = {f: (res[n][f][v][0] - res["step7 (single seed 42)"][f][v][0])[res[n][f][v][1]] for f in folds}
            sig[n][v] = {f: boot(d[f], rng) for f in folds}
            sig[n][v]["pooled"] = boot(np.r_[d["A"], d["B"]], rng)
    R["bootstrap_vs_step7"] = sig
    best = max(("bagged (w=0)", "bagged + step-6 fusion (w chosen out-of-bag)"), key=lambda n: table[n]["A_VTOP5"] + table[n]["B_VTOP5"])
    sb = sig[best]["VTOP5"]
    ship = bool(sb["A"]["mean_diff"] > 0 and sb["B"]["mean_diff"] > 0 and sb["pooled"]["ci95"][0] > 0)
    R["decision"] = {"best_of_1_2": best, "ship": ship,
                     "rule": "ship only if the best of bagging / bagging+fusion beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero"}
    R["fit_seconds_all_models"] = round(fit_s, 1)

    # ================================================================== 5. final run (only if shipped)
    if ship:
        print("final", flush=True)
        users_f = data.load_submission_users()
        pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
        nf, npf = len(users_f), len(pool_f)
        ctxf = s7.make_ctx(["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
        store_f, _ = s7.build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
        store_f["EW"] = s7.build_store(ctxf, ["EW"], cfg, ew_cfg)[0]["EW"]
        cand_f = rerank.candidates(store_f, ctxf["pop_norm"], nf, npf, names=rerank.ORDER)
        rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
        all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
        bag_f = np.zeros(len(rif))
        rounds_used = {}
        for s in SEEDS:
            rounds = int(round(np.mean([best_iter["A"][s], best_iter["B"][s]])))
            rounds_used[s] = rounds
            model = rerank2.train(X, y, ri, all_train, None, {**rerank2.BASE_PARAMS, "seed": s}, rounds=rounds)
            bag_f += pct_by_user(rif, model.predict(Xf[list(X.columns)])) / len(SEEDS)
        w_final = 0.0
        if best.startswith("bagged + step-6"):
            w_final = round(float(np.mean(list(chosen_w.values()))) * 10) / 10
        score = bag_f
        if w_final > 0:
            ctx6 = s7.make_ctx(["11-20", "21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
            st6, _ = s7.build_store(ctx6, ["H2", "PPR", "MF"], cfg)
            S6f = s4.fuse_eps(ctx6["pop_norm"], [st6[n] for n in ("H2", "MF", "PPR")], [w6[n] for n in ("H2", "MF", "PPR")])
            score = (1 - w_final) * bag_f + w_final * pct_by_user(rif, S6f[rif, cif])
        top = top5_of(rif, cif, score, np.arange(nf))
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        # cold-start rule: users with no labelled airing in weeks 21-30 get the step-6 list (history 11-30); not validated
        has_air = np.bincount(ctxf["air_ri"][ctxf["air_ri"] >= 0], minlength=nf) > 0
        fin6 = pd.read_csv(SUBDIR / "submission_FINAL_vtop5.csv", dtype=str).set_index("user_id")["tv_show_id"]
        cold = [u for u, h in zip(users_f, has_air) if not h]
        for u in cold:
            preds[u] = [int(x) for x in fin6[str(u)].split()]
        path = SUBDIR / "submission_STEP8b.csv"
        write_submission(preds, users_f, path)
        pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
        R["final"] = {**s3.check_file(path, users_f, pool_f), "seeds": SEEDS, "rounds_per_seed": rounds_used, "w_step6_fusion": w_final,
                      "cold_start_rule": "users with no labelled airing in weeks 21-30 receive their step-6 blend list (history weeks 11-30); this rule cannot be validated",
                      "cold_start_users": len(cold), "users_without_H2_data": int((~store_f["H2"][1]).sum()),
                      "repeat_share_train_positive_window_21_30": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f]))}
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP8b.csv" not in txt:
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP8b.csv | step 8b: step-7 reranker bagged over 5 seeds" + (f" + rank fusion with the step-6 blend (w {w_final})" if w_final > 0 else "") +
                          f" | 5 LightGBM lambdarank models (seeds {SEEDS}, trees {list(rounds_used.values())}) fitted on all validation users, features from weeks 21-30 + 31-42 schedule; "
                          f"{len(cold)} users without weeks 21-30 data get the step-6 list | | |\n", encoding="utf-8")
    R["laps_seconds"], R["runtime_s"] = laps, round(time.time() - T0, 1)
    (OUT / "validation_step8b.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_md(R)
    print(f"done in {R['runtime_s']} s; ship {ship}; best {best}")


def write_md(R):
    f4 = lambda x: f"{x:.4f}"
    V = VARS
    md = ["# Validation, step 8b", "",
          "Seed/fold bagging of the step-7 reranker (step-7 features and settings, no step-8 changes): 5 seeds x the 2-fold swap (fold A: fit TUNE, score REPORT; fold B: fit REPORT, score TUNE). "
          "Each user's score is the average of the rank-normalised scores of the 5 models that did not see that user. Primary criterion VTOP5. No Kaggle number is used.", "",
          f"Step-7 reproduction (fold A, VTOP5, seed 42): {R['step7_reproduction']['fold_A_vtop5']:.4f} vs {R['step7_reproduction']['step7_report']:.4f} in the step-7 report.", "",
          "## 1. MAP@5 (mean over the two folds; fold A / fold B for VTOP5)", "",
          "| model | VTOP5 fold A | VTOP5 fold B | mean VTOP5 | mean V300 | mean V900 | mean VHALF |", "|---|---|---|---|---|---|---|"]
    for n, t in R["table"].items():
        md.append(f"| {n} | {f4(t['A_VTOP5'])} | {f4(t['B_VTOP5'])} | {f4((t['A_VTOP5'] + t['B_VTOP5']) / 2)} | {f4(t['mean_V300'])} | {f4(t['mean_V900'])} | {f4(t['mean_VHALF'])} |")
    fu = R["fusion"]
    md += ["", f"Fusion weight on the step-6 blend (grid {R['w_grid']}), chosen out-of-bag inside each fold: {fu['chosen_w']}. {fu['selection_note']}. "
               f"Out-of-bag selection MAP@5 (VTOP5) by w: {json.dumps({f: v['oob_map5_vtop5_by_w'] for f, v in fu['oob_selection'].items()})} on {[v['users'] for v in fu['oob_selection'].values()]} users.", "",
           "## 2. Paired bootstrap vs step 7 (2,000 resamples, seed 42): mean AP@5 difference [95% CI]", ""]
    for n, d in R["bootstrap_vs_step7"].items():
        md += [f"**{n}**", "", "| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |", "|---|---|---|---|"]
        for v in V:
            md.append(f"| {v} | " + " | ".join(f"{d[v][k]['mean_diff']:+.4f} [{d[v][k]['ci95'][0]:+.4f}, {d[v][k]['ci95'][1]:+.4f}]" for k in ("A", "B", "pooled")) + " |")
        md.append("")
    dc = R["decision"]
    md += ["## 3. Decision", "", f"Rule: {dc['rule']}. Best of 1-2 by mean VTOP5: **{dc['best_of_1_2']}**. Ship: **{dc['ship']}**.", ""]
    if "final" in R:
        f = R["final"]
        md += [f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}",
               f"- 5 models (seeds {f['seeds']}, trees per seed {f['rounds_per_seed']}) fitted on all validation users; features from weeks 21-30 + the 31-42 schedule; step-6 fusion weight {f['w_step6_fusion']}",
               f"- cold-start rule: {f['cold_start_rule']}. Users affected: {f['cold_start_users']} (users without H2 data in the window: {f['users_without_H2_data']}); repeat share {f['repeat_share_train_positive_window_21_30']:.3f}"]
    else:
        md += ["Not shipped; no submission file was written and nothing else was changed."]
    md += ["", f"Runtime: {R['runtime_s']} s (all model fits {R['fit_seconds_all_models']} s; stage clocks {R['laps_seconds']})"]
    (OUT / "validation_step8b.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

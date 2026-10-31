"""Step 10 (8c): H2 decay grid, multi-scale recency features, learning curve, 2-fold swap vs the step-7 reranker.
Primary criterion VTOP5. Usage: python src/run_step10.py
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
C_GRID, T_GRID = (0.5, 0.3, 0.2, 0.1), (60, 30, 14)
FRACS = (0.25, 0.5, 1.0)


def boot(d, rng):
    bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    return {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def score_rank_at_pairs(Sn, valid, ri, ci, pop_norm):
    """Score and rank (tie-broken by popularity) of the pairs under one scorer."""
    n_u, n_p = Sn.shape
    rk = np.empty(len(ri), dtype=np.int32)
    for a in range(0, n_u, 1000):
        S = s4.fuse_eps(pop_norm, [(Sn[a:a + 1000], valid[a:a + 1000])], [1.0])
        ranks = np.empty(S.shape, dtype=np.int32)
        np.put_along_axis(ranks, np.argsort(-S, axis=1, kind="stable"), np.arange(n_p, dtype=np.int32), axis=1)
        lo, hi = np.searchsorted(ri, a), np.searchsorted(ri, a + 1000)
        rk[lo:hi] = ranks[ri[lo:hi] - a, ci[lo:hi]]
    return Sn[ri, ci], rk


def multiscale(ctx, ri, ci, cfgs, prep):
    """Extra reranker features: H2 score/rank for each (c, T) in cfgs and recent completed-airing counts of the show."""
    t0 = time.time()
    F = {}
    for c, T in cfgs:
        S, v, _ = h2.scores(prep, ctx["n_air_tr"], ctx["users"], ctx["pool"], ctx["train_end"], c, scale=T)[:3]
        s, r = score_rank_at_pairs(hybrid.minmax_rows(S), v, ri, ci, ctx["pop_norm"])
        F[f"s_H2_c{c}_T{T}"], F[f"r_H2_c{c}_T{T}"] = s, r
    n_p = ctx["n_p"]
    ok = (ctx["air_ri"] >= 0) & (ctx["air_ci"] >= 0) & (ctx["air"]["view_pct"].values >= 0.8)
    a_ri, a_ci = ctx["air_ri"][ok].astype(np.int64), ctx["air_ci"][ok].astype(np.int64)
    st = ctx["air"]["start"].values[ok].astype(np.float64)
    end = ctx["train_end"]
    df = pd.DataFrame({"k": a_ri * n_p + a_ci, "l2": st >= end - 14 * 86400, "l4": st >= end - 28 * 86400, "st": st})
    g = df.groupby("k").agg(l2=("l2", "sum"), l4=("l4", "sum"), last=("st", "max")).reindex(ri.astype(np.int64) * n_p + ci)
    F["us_completed_last2w"] = g["l2"].fillna(0).values.astype(np.float32)
    F["us_completed_last4w"] = g["l4"].fillna(0).values.astype(np.float32)
    F["us_days_since_last_completed"] = ((end - g["last"]).fillna(999 * 86400) / 86400.0).values.astype(np.float32)
    return pd.DataFrame(F), time.time() - t0


def main():
    T0 = time.time()
    laps = {}
    lap = lambda k: laps.__setitem__(k, round(time.time() - T0, 1))
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    R = {"seed": SEED, "primary": "VTOP5", "step7_h2": {"c": cfg["h2_c"], "T": 60}}

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
    uix = ctx["uix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, ctx["pix"], n_u, n_p) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    T5 = T["VTOP5"]
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    folds = {"A": {"train": tune_rows[nonempty["VTOP5"][tune_rows]], "eval": report_rows},
             "B": {"train": report_rows[nonempty["VTOP5"][report_rows]], "eval": tune_rows}}
    pop_norm = ctx["pop_norm"]
    lap("setup")

    # ================================================================== 1. H2 decay grid (H2 alone, TUNE)
    print("H2 grid", flush=True)
    prep = h2.prepare(ctx["air"])
    Tt = {v: T[v][tune_rows] for v in VARS}
    mt = {v: nonempty[v][tune_rows] for v in VARS}
    grid, best = [], None
    for c in C_GRID:
        for Tscale in T_GRID:
            S, v_, _ = h2.scores(prep, ctx["n_air_tr"], users, pool, ctx["train_end"], c, scale=Tscale)[:3]
            Sn = hybrid.minmax_rows(S)
            top = hybrid.topk(s4.fuse_eps(pop_norm, [(Sn[tune_rows], v_[tune_rows])], [1.0]), K)
            m4 = [float(metrics.ap5_matrix(top, Tt[v])[mt[v]].mean()) for v in VARS]
            grid.append({"c": c, "T": Tscale, "decay_per_day": round(-np.log(c) / Tscale, 4), **{f"tune_{v}": x for v, x in zip(VARS, m4)}})
            if best is None or m4[3] > best[0]:
                best = (m4[3], c, Tscale)
    c_star, T_star = best[1], best[2]
    R["h2_grid"] = {"grid": grid, "chosen": {"c": c_star, "T": T_star}, "note": "c ** (days / T) = exp(-days * ln(1/c) / T), so (c, T) pairs with equal decay_per_day are identical"}
    print("  chosen", c_star, T_star, flush=True)
    lap("h2_grid")

    # ================================================================== step-7 candidates/features, multi-scale additions
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
    cand6 = rerank.candidates(store, pop_norm, n_u, n_p, names=rerank.ORDER)
    ri, ci, X7, f_s = rerank.make_features(ctx, store, pop_norm, cand6)
    ms_cfgs = [(1.0, 60)] + ([(c_star, T_star)] if (c_star, T_star) != (cfg["h2_c"], 60) else [])
    XE, e_s = multiscale(ctx, ri, ci, ms_cfgs, prep)
    Xn = pd.concat([X7, XE], axis=1)
    y = T5[ri, ci].astype(int)
    R["multiscale"] = {"extra_h2_configs": [{"c": c, "T": t} for c, t in ms_cfgs], "added_columns": list(XE.columns), "seconds": round(e_s, 1),
                       "note": "H2 (c=0.5, T=60) is already in the step-7 features; the added H2 configs are the others in {1.0, 0.5, best}"}
    lap("features")

    def split(train_rows, seed=SEED):
        p = np.random.default_rng(seed).permutation(train_rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    def fit_eval(Xdf, fd, frac=1.0):
        rows = fd["train"]
        if frac < 1.0:
            rows = np.sort(np.random.default_rng(SEED).permutation(rows)[:int(round(frac * len(rows)))])
        tr, va = split(rows)
        model = rerank2.train(Xdf, y, ri, tr, va, rerank2.BASE_PARAMS)
        top = rerank.top5(model, Xdf, ri, ci, fd["eval"])
        return model, {v: (metrics.ap5_matrix(top, T[v][fd["eval"]]), nonempty[v][fd["eval"]]) for v in VARS}

    # ================================================================== 2. 2-fold swap: step-7 features vs multi-scale features
    print("fits", flush=True)
    res, models, tfit = {}, {}, {}
    for name, Xdf in (("step7 features", X7), ("step7 + multi-scale features", Xn)):
        res[name], models[name], t = {}, {}, time.time()
        for f, fd in folds.items():
            models[name][f], res[name][f] = fit_eval(Xdf, fd)
        tfit[name] = round(time.time() - t, 1)
    mean_of = lambda r, v, f=None: float(np.mean([r[x][v][0][r[x][v][1]].mean() for x in ([f] if f else ["A", "B"])]))
    R["table"] = {n: {"A_VTOP5": mean_of(res[n], "VTOP5", "A"), "B_VTOP5": mean_of(res[n], "VTOP5", "B"), **{f"mean_{v}": mean_of(res[n], v) for v in VARS}, "fit_seconds_2_folds": tfit[n]} for n in res}
    R["step7_reproduction"] = {"fold_A_vtop5": R["table"]["step7 features"]["A_VTOP5"], "step7_report": J7["rows"]["reranker"]["VTOP5"]["report"]}
    best_feat = max(res, key=lambda n: R["table"][n]["A_VTOP5"] + R["table"][n]["B_VTOP5"])
    Xbest = X7 if best_feat == "step7 features" else Xn
    lap("fits")

    # ================================================================== 3. learning curve (best features)
    print("learning curve", flush=True)
    lc = {}
    for fr in FRACS:
        lc[fr] = {}
        for f, fd in folds.items():
            if fr == 1.0:
                lc[fr][f] = float(res[best_feat][f]["VTOP5"][0][res[best_feat][f]["VTOP5"][1]].mean())
                lc[fr][f + "_users"] = int(len(fd["train"]))
            else:
                _, r = fit_eval(Xbest, fd, fr)
                lc[fr][f] = float(r["VTOP5"][0][r["VTOP5"][1]].mean())
                lc[fr][f + "_users"] = int(round(fr * len(fd["train"])))
    use_all = bool(lc[1.0]["A"] > lc[0.5]["A"] and lc[1.0]["B"] > lc[0.5]["B"])
    R["learning_curve"] = {"features": best_feat, "vtop5_by_fraction": {str(k): v for k, v in lc.items()}, "full_beats_half_in_both_folds": use_all,
                           "rule": "if 100% > 50% in both folds the final model is fitted on ALL validation users with the same settings and number of trees; otherwise the step-7 final fitting (TUNE-fitted model) is kept"}
    lap("learning_curve")

    # ================================================================== 4. bootstrap vs step 7, rule
    rng = np.random.default_rng(SEED)
    new, base = "step7 + multi-scale features", "step7 features"
    sig = {}
    for v in VARS:
        d = {f: (res[new][f][v][0] - res[base][f][v][0])[res[new][f][v][1]] for f in folds}
        sig[v] = {f: boot(d[f], rng) for f in folds}
        sig[v]["pooled"] = boot(np.r_[d["A"], d["B"]], rng)
    R["bootstrap_vs_step7"] = sig
    sv = sig["VTOP5"]
    ship = bool(sv["A"]["mean_diff"] > 0 and sv["B"]["mean_diff"] > 0 and sv["pooled"]["ci95"][0] > 0)
    imp = sum(rerank2.importance(models[new][f], Xn.columns) for f in folds) / 2
    R["importance_top15_gain_share"] = {k: round(float(v), 4) for k, v in imp.sort_values(ascending=False).head(15).items()}
    R["decision"] = {"ship": ship, "rule": "ship only if the new reranker (step-7 + multi-scale features) beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero"}
    lap("evaluation")

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
        rif, cif, Xf7, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], cand_f)
        XEf, _ = multiscale(ctxf, rif, cif, ms_cfgs, h2.prepare(ctxf["air"]))
        Xf = pd.concat([Xf7, XEf], axis=1)[list(Xn.columns)]
        if use_all:
            all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
            rounds = int(round(np.mean([models[new][f].best_iteration for f in folds])))
            model_f = rerank2.train(Xn, y, ri, all_train, None, rerank2.BASE_PARAMS, rounds=rounds)
            fit_desc = f"fitted on ALL validation users, {rounds} trees"
        else:
            model_f, fit_desc = models[new]["A"], f"step-7 final fitting kept (fitted on TUNE users, {models[new]['A'].best_iteration} trees)"
        top = rerank.top5(model_f, Xf, rif, cif, np.arange(nf))
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / "submission_STEP10.csv"
        write_submission(preds, users_f, path)
        pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
        R["final"] = {**s3.check_file(path, users_f, pool_f), "fit": fit_desc, "train_window": ["21-30"],
                      "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in store_f},
                      "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f]))}
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP10.csv" not in txt:
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP10.csv | step 10: step-7 reranker + multi-scale recency features (H2 decay configs {ms_cfgs}, recent completed-airing counts) | LightGBM lambdarank, step-7 settings; {fit_desc}; features from weeks 21-30 + 31-42 schedule | | |\n", encoding="utf-8")
    R["laps_seconds"], R["runtime_s"] = laps, round(time.time() - T0, 1)
    (OUT / "validation_step10.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_md(R)
    print(f"done in {R['runtime_s']} s; ship {ship}; best features {best_feat}; learning curve full>half both folds: {use_all}")


def write_md(R):
    f4 = lambda x: f"{x:.4f}"
    V = VARS
    md = ["# Validation, step 10 (8c)", "",
          "Primary criterion VTOP5; same pool, repeats allowed, 5 distinct ids, popularity fill, seed 42; 2-fold swap (fold A: fit TUNE, score REPORT; fold B: fit REPORT, score TUNE). "
          "Starts from the step-7 reranker (step-7 features and settings). No Kaggle number is used.", "",
          f"Step-7 reproduction (fold A VTOP5): {R['step7_reproduction']['fold_A_vtop5']:.4f} vs {R['step7_reproduction']['step7_report']:.4f}.", "",
          "## 1. H2 decay grid (H2 alone, TUNE users; decay = c ** (days / T))", "",
          "| c | T (days) | decay per day | TUNE V300 | V900 | VHALF | VTOP5 |", "|---|---|---|---|---|---|---|"]
    for g in R["h2_grid"]["grid"]:
        md.append(f"| {g['c']} | {g['T']} | {g['decay_per_day']} | " + " | ".join(f4(g[f"tune_{v}"]) for v in V) + " |")
    ch = R["h2_grid"]["chosen"]
    md += ["", f"Chosen on TUNE VTOP5: c = {ch['c']}, T = {ch['T']}. Step-5 value: c = {R['step7_h2']['c']}, T = 60. {R['h2_grid']['note']}.", "",
           "## 2. Multi-scale features and 2-fold swap", "",
           f"Added columns: {R['multiscale']['added_columns']}. {R['multiscale']['note']}.", "",
           "| model | VTOP5 fold A | VTOP5 fold B | mean VTOP5 | mean V300 | mean V900 | mean VHALF | fit s (2 folds) |", "|---|---|---|---|---|---|---|---|"]
    for n, t in R["table"].items():
        md.append(f"| {n} | {f4(t['A_VTOP5'])} | {f4(t['B_VTOP5'])} | {f4((t['A_VTOP5'] + t['B_VTOP5']) / 2)} | {f4(t['mean_V300'])} | {f4(t['mean_V900'])} | {f4(t['mean_VHALF'])} | {t['fit_seconds_2_folds']} |")
    lc = R["learning_curve"]
    md += ["", f"## 3. Learning curve ({lc['features']}; VTOP5 on the other fold)", "", "| share of training-fold users | fold A (users) | fold B (users) |", "|---|---|---|"]
    for fr, d in lc["vtop5_by_fraction"].items():
        md.append(f"| {float(fr):.0%} | {f4(d['A'])} ({d['A_users']}) | {f4(d['B'])} ({d['B_users']}) |")
    md += ["", f"Rule: {lc['rule']}. Outcome: 100% > 50% in both folds = **{lc['full_beats_half_in_both_folds']}**.", "",
           "## 4. Paired bootstrap, step7 + multi-scale vs step 7 (2,000 resamples, seed 42): mean AP@5 difference [95% CI]", "",
           "| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |", "|---|---|---|---|"]
    for v in V:
        d = R["bootstrap_vs_step7"][v]
        md.append(f"| {v} | " + " | ".join(f"{d[k]['mean_diff']:+.4f} [{d[k]['ci95'][0]:+.4f}, {d[k]['ci95'][1]:+.4f}]" for k in ("A", "B", "pooled")) + " |")
    md += ["", "Feature importances, top 15 (gain share, mean of the two fold models of the multi-scale model): " + ", ".join(f"{k} {v}" for k, v in R["importance_top15_gain_share"].items()), "",
           "## 5. Decision", "", f"Rule: {R['decision']['rule']}. Ship: **{R['decision']['ship']}**.", ""]
    if "final" in R:
        f = R["final"]
        md += [f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}",
               f"- {f['fit']}; features from weeks 21-30 + the 31-42 schedule, no re-fitting on test; users without component data (popularity fallback): {f['users_without_component_data']}; repeat share {f['repeat_share_train_positive']:.3f}"]
    else:
        md += ["Not shipped; no submission file was written and nothing else was changed."]
    md += ["", f"Runtime: {R['runtime_s']} s (stage clocks {R['laps_seconds']})"]
    (OUT / "validation_step10.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

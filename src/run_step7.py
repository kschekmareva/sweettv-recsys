"""Step 7: ceiling diagnostics, EW scorer, learned reranker. Primary criterion VTOP5 (pre-declared in step 6).
Usage: python src/run_step7.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import assoc
import data
import ew
import h2
import hybrid
import metrics
import ppr
import rerank
import run_step3 as s3
import run_step4 as s4
import run_step5 as s5
from run_validation import write_submission

SEED, THR, K = 42, 300, 5
VARS = s5.VARS
FAMILY = "P-air+MF+PPR+H2"
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
fuse_spec, fuse_eps = s5.fuse_spec, s4.fuse_eps


# ------------------------------------------------------------------ context of a TRAIN window + target schedule
def make_ctx(train_periods, target_key, tw, users, pool, uc_pairs, inter, cat_series):
    n_u, n_p = len(users), len(pool)
    uix, pix = pd.Index(np.asarray(users, dtype=np.uint64)), pd.Index(np.asarray(pool))
    ranges = [data.PERIODS[k] for k in train_periods]
    lo, hi_w = min(r[0] for r in ranges), max(r[1] for r in ranges)
    ctx = {"users": users, "pool": pool, "uix": uix, "pix": pix, "n_u": n_u, "n_p": n_p, "periods": list(train_periods)}
    ctx["train_all"] = data.positives(inter, train_periods, 0)
    ctx["train_pos"] = data.positives(inter, train_periods, THR)
    air = data.airings(train_periods)
    ctx["air"], ctx["air_ri"], ctx["air_ci"] = air, uix.get_indexer(air["user_id"].values), pix.get_indexer(air["tv_show_id"].values)
    ctx["air_cat"] = cat_series.reindex(air["tv_show_id"].values).fillna(-1).values.astype(np.int64)
    ctx["train_end"] = h2.sec_of("2020-07-27" if hi_w == 30 else "2020-05-18")
    sel = (ctx["air_ri"] >= 0) & (ctx["air_ci"] >= 0) & (air["view_pct"].values >= 0.8)
    ctx["hi"] = {"ri": ctx["air_ri"][sel], "ci": ctx["air_ci"][sel], "start": air["start"].values[sel].astype(np.float64)}
    parts = []
    for k, (w0, w1) in zip(train_periods, ranges):
        s = data.load_schedule(k)
        m = (s["tv_show_id"] != 0) & s["iso_week"].between(w0, w1) & (s["iso_year"] == 2020)
        parts.append(s.loc[m, ["channel_id", "tv_show_id", "start"]])
    st = pd.concat(parts)
    st["pc"] = pix.get_indexer(st["tv_show_id"].values)
    st = st[st["pc"] >= 0]
    ctx["sched_train"], ctx["n_air_tr"] = st, np.bincount(st["pc"].values, minlength=n_p).astype(np.float32)
    ctx["ch_index"] = pd.Index(np.sort(st["channel_id"].unique()))
    up = uc_pairs[uc_pairs["iso_week"].between(lo, hi_w)][["user_id", "vsetv_id"]].drop_duplicates()
    r, c = uix.get_indexer(up["user_id"].values), ctx["ch_index"].get_indexer(up["vsetv_id"].values)
    ok = (r >= 0) & (c >= 0)
    ctx["UC"] = csr_matrix((np.ones(ok.sum(), dtype=np.float32), (r[ok], c[ok])), shape=(n_u, len(ctx["ch_index"])))
    s = data.load_schedule(target_key)
    m = (s["tv_show_id"] != 0) & s["iso_week"].between(*tw) & (s["iso_year"] == 2020)
    t = s.loc[m].copy()
    t["pc"] = pix.get_indexer(t["tv_show_id"].values)
    t = t[t["pc"] >= 0]
    t["dur"], t["hour"] = t["end"] - t["start"], (t["start"] % 86400) // 3600
    ctx["n_target"] = np.bincount(t["pc"].values, minlength=n_p).astype(np.float32)
    ctx["med_dur"] = t.groupby("pc")["dur"].median().reindex(range(n_p)).values
    mode = lambda col: (t.groupby(["pc", col]).size().reset_index(name="n").sort_values(["pc", "n", col], ascending=[True, False, True])
                        .drop_duplicates("pc").set_index("pc")[col].reindex(range(n_p)).fillna(-1).values.astype(np.int64))
    ctx["main_ch"], ctx["typ_hour"] = mode("channel_id"), mode("hour")
    ctx["cat_code"] = cat_series.reindex(pool).fillna(-1).values.astype(np.int64)
    ctx["P"] = hybrid.pop_counts(ctx["train_pos"], pool)
    ctx["pop_norm"] = hybrid.minmax_vec(ctx["P"] * np.log1p(ctx["n_target"]))          # P-air
    return ctx


def build_store(ctx, names, cfg, ew_cfg=None):
    store, times = {}, {}
    if "H2" in names:
        prep = h2.prepare(ctx["air"])
        S, v, ts = h2.scores(prep, ctx["n_air_tr"], ctx["users"], ctx["pool"], ctx["train_end"], cfg["h2_c"])
        t = time.time(); store["H2"] = (hybrid.minmax_rows(S), v); times["H2"] = (prep["seconds"], ts + time.time() - t)
    if "EW" in names:
        m, c = ew_cfg
        S, v, g, ts = ew.scores(ctx, m, c)
        t = time.time(); store["EW"] = (hybrid.minmax_rows(S), v); times["EW"] = (0.0, ts + time.time() - t)
    if "PPR" in names:
        g = ppr.build_graph(ctx["train_pos"])
        S, v, info = ppr.run(g, ctx["users"], ctx["pool"], cfg["ppr_alpha"])
        t = time.time(); store["PPR"] = (hybrid.minmax_rows(S), v); times["PPR"] = (g["build_s"], info["seconds"] + time.time() - t)
    if "MF" in names:
        c = s3.mf_component(ctx["train_all"], cfg["mf"], ctx["users"], ctx["pool"])
        store["MF"] = (c["Sn"], c["valid"]); times["MF"] = (c["fit_s"], c["score_s"])
    if "AR" in names:
        mdl = assoc.fit(ctx["train_pos"], ctx["pool"])
        S, v, ts = assoc.score(mdl, ctx["train_pos"], ctx["users"], cfg["ar_mode"])
        t = time.time(); store["AR"] = (hybrid.minmax_rows(S), v); times["AR"] = (mdl["build_s"], ts + time.time() - t)
    return store, times


def main():
    T0 = time.time()
    laps = {}
    lap = lambda name: laps.__setitem__(name, round(time.time() - T0, 1))
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J6 = json.loads((OUT / "validation_step6.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    w6 = dict(J5["rows"][FAMILY]["weights"]["VTOP5"])
    R = {"seed": SEED, "primary": "VTOP5", "configs_from_step5": cfg, "step6_weights": w6}
    np.random.seed(SEED)

    # ================================================================== data, truth, split (identical to steps 5-6)
    views = data.load_views()
    uc_pairs = views[["user_id", "vsetv_id", "iso_week"]].drop_duplicates()
    del views
    inter, _ = data.build_interactions()
    titles = data.titles()
    cat_list = sorted(titles["tv_show_category"].dropna().unique())
    cat_series = pd.Series(pd.Categorical(titles["tv_show_category"], categories=cat_list).codes, index=titles.index)
    air_val = data.airings(["21-30"])
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300], "V900": val[val["overlap_sec"] >= 900], "VHALF": val[val["max_frac"] >= 0.5]}
    vpairs["VTOP5"], _ = h2.vtop5_truth(air_val)
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    users = sorted(set(int(u) for v in vpairs.values() for u in v["user_id"].unique()) | set(sub_users))
    n_u, n_p = len(users), len(pool)
    ctx = make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    uix, pix = ctx["uix"], ctx["pix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    Tpos = s4.bool_matrix(ctx["train_pos"], uix, pix, n_u, n_p)
    Tall = s4.bool_matrix(ctx["train_all"], uix, pix, n_u, n_p)
    Tdone = np.zeros((n_u, n_p), dtype=bool)
    Tdone[ctx["hi"]["ri"], ctx["hi"]["ci"]] = True
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    Tt = {v: T[v][tune_rows] for v in VARS}
    mt = {v: nonempty[v][tune_rows] for v in VARS}
    tune4 = lambda top: np.array([metrics.ap5_matrix(top, Tt[v])[mt[v]].mean() for v in VARS])
    pop_norm = ctx["pop_norm"]
    top_of = s5.make_top_fn(pop_norm)
    tune_eval = lambda spec, store_t: tune4(top_of(spec, store_t, len(tune_rows)))
    lap("setup")

    # ================================================================== components (steps 5-6 configs, TRAIN weeks 11-20)
    print("components", flush=True)
    store, times = build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    lap("components")

    # ================================================================== 2. EW grid (m, c) on TUNE VTOP5
    print("EW grid", flush=True)
    ew_rows, best_ew = [], None
    for c in ew.C_GRID:
        for m in ew.M_GRID:
            S, v, g, ts = ew.scores(ctx, m, c)
            Sn = hybrid.minmax_rows(S)
            m4 = tune_eval(fuse_spec({"EW": 1.0}), {"EW": (Sn[tune_rows], v[tune_rows])})
            ew_rows.append({"m": m, "c": c, "global_rate": g, "score_s": round(ts, 2), **{f"tune_{x}": float(y) for x, y in zip(VARS, m4)}})
            print("  ", ew_rows[-1], flush=True)
            if best_ew is None or m4[3] > best_ew[0]:
                best_ew = (m4[3], m, c, Sn, v, ts)
    _, ew_m, ew_c = best_ew[:3]
    store["EW"], times["EW"] = (best_ew[3], best_ew[4]), (0.0, best_ew[5])
    R["ew"] = {"grid": ew_rows, "chosen": {"m": ew_m, "c": ew_c}}
    lap("ew")

    sub_of = lambda names: {n: (store[n][0][tune_rows], store[n][1][tune_rows]) for n in names}
    # EW in place of H2 in the four-component blend; weights re-chosen on TUNE VTOP5
    store_t = sub_of(["EW", "MF", "PPR", "H2", "AR"])
    specs_ew = [fuse_spec({"MF": a / 10, "PPR": b / 10, "EW": c / 10}) for a in range(11) for b in range(11 - a) for c in range(11 - a - b)]
    cur = np.array([tune_eval(sp, store_t) for sp in specs_ew])
    best_i = int(cur[:, 3].argmax())
    w_ewb = specs_ew[best_i]["w"]
    R["ew_blend"] = {"weights": w_ewb, "popularity_weight": round(1 - sum(w_ewb.values()), 1), "tune_by_variant": dict(zip(VARS, cur[best_i].tolist()))}
    lap("ew_blend")

    # ================================================================== 1. ceiling diagnostics (VTOP5)
    print("candidates / ceiling", flush=True)
    cand5 = rerank.candidates(store, pop_norm, n_u, n_p, names=["H2", "PPR", "MF", "AR", "P-air"])
    cand6 = rerank.candidates(store, pop_norm, n_u, n_p, names=rerank.ORDER)
    T5 = T["VTOP5"]

    def share(M, rows):
        tr = T5[rows]
        return float((tr & M[rows]).sum() / tr.sum())
    allrows = np.flatnonzero(nonempty["VTOP5"])
    rep5 = report_rows[nonempty["VTOP5"][report_rows]]
    R["ceiling"] = {"truth_items_all": int(T5[allrows].sum()), "truth_items_report": int(T5[rep5].sum()),
                    "union_candidates_per_user_mean_5comp": float(cand5[allrows].sum(axis=1).mean()),
                    "union_candidates_per_user_mean_6comp": float(cand6[allrows].sum(axis=1).mean())}
    for label, M in (("a_watched_at_all_in_TRAIN", Tall), ("b_completed_ge80pct_in_TRAIN", Tdone),
                     ("c_in_top50_union_H2_PPR_MF_AR_Pair", cand5), ("c6_in_top50_union_plus_EW", cand6)):
        R["ceiling"][label] = {"all_val_users": share(M, allrows), "report_users": share(M, rep5)}

    def oracle_ap(cand, v):
        h = np.minimum(K, (T[v] & cand).sum(axis=1))
        den = np.minimum(K, T[v].sum(axis=1))
        return np.where(den > 0, h / np.maximum(den, 1), 0.0)
    lap("ceiling")

    # ================================================================== 3. reranker
    print("reranker features", flush=True)
    t = time.time()
    ri, ci, X, feat_s = rerank.make_features(ctx, store, pop_norm, cand6)
    y = T5[ri, ci].astype(np.int8)
    tune_truth = tune_rows[nonempty["VTOP5"][tune_rows]]
    best_rr, rr_all, rr_split, fit_s = rerank.fit(X, y, ri, ci, tune_truth, T5, SEED)
    model = best_rr["model"]
    top_rr = rerank.top5(model, X, ri, ci, report_rows)
    R["reranker"] = {"candidates_pairs": int(len(ri)), "positives": int(y.sum()), "features": list(X.columns), "feature_seconds": round(feat_s, 1),
                     "fit_seconds": round(fit_s, 1), "split": rr_split, "chosen_objective": best_rr["objective"],
                     "objectives": [{k: v for k, v in r.items() if k not in ("model",)} for r in rr_all],
                     "top10_importance_gain": best_rr["top_importance"],
                     "final_rule": "the fitted model is applied, without re-fitting, to features computed from TRAIN window = weeks 21-30 "
                                   "(same 10-week length as the training window 11-20) plus the 31-42 schedule for n_target_airings and the target pool"}
    lap("reranker")

    # ================================================================== rows (REPORT users)
    tops = {}
    tops["P"] = np.tile(np.argsort(-ctx["P"], kind="stable")[:K], (len(report_rows), 1))
    full_store = {n: store[n] for n in store}
    n_rep = len(report_rows)
    t = time.time(); tops["H2"] = top_of(fuse_spec({"H2": 1.0}), {n: (a[report_rows], b[report_rows]) for n, (a, b) in full_store.items()}, n_rep)
    rep_store = {n: (a[report_rows], b[report_rows]) for n, (a, b) in full_store.items()}
    tops["EW"] = top_of(fuse_spec({"EW": 1.0}), rep_store, n_rep)
    tops["step-6 blend"] = top_of(fuse_spec(w6), rep_store, n_rep)
    tops["EW-blend"] = top_of(fuse_spec(w_ewb), rep_store, n_rep)
    tops["reranker"] = top_rr
    assert abs(metrics.ap5_matrix(tops["step-6 blend"], T5[report_rows])[nonempty["VTOP5"][report_rows]].mean()
               - J6["rows"]["step-6 FINAL (VTOP5-tuned w)"]["VTOP5"]["report"]) < 1e-9, "step-6 blend not reproduced"
    cost = lambda names: (round(sum(times[n][0] for n in names), 2), round(sum(times[n][1] for n in names), 2))
    costs = {"P": (0.0, 0.0), "H2": cost(["H2"]), "EW": cost(["EW"]), "step-6 blend": cost(["H2", "MF", "PPR"]), "EW-blend": cost(["EW", "MF", "PPR"]),
             "reranker": (round(sum(times[n][0] for n in rerank.ORDER if n in times) + fit_s, 2),
                          round(sum(times[n][1] for n in rerank.ORDER if n in times) + feat_s, 2))}
    rows, aps = {}, {}
    for name, tp in tops.items():
        rows[name], aps[name] = {}, {}
        for v in VARS:
            mk = nonempty[v][report_rows]
            ap = metrics.ap5_matrix(tp, T[v][report_rows])
            aps[name][v] = (ap, mk)
            rows[name][v] = {"report": float(ap[mk].mean())}
        mk = nonempty["VTOP5"][report_rows]
        rows[name]["stats_vtop5_report_users"] = {"mean_train_popularity": float(ctx["P"][tp[mk]].mean()),
                                                  "repeat_share_train_positive": float(Tpos[report_rows[mk][:, None], tp[mk]].mean())}
        rows[name]["fit_s"], rows[name]["score_s"] = costs[name]
    for label, cand in (("oracle (6-component top-50 union)", cand6), ("oracle (5-component top-50 union)", cand5)):
        rows[label] = {}
        for v in VARS:
            mk = nonempty[v][report_rows]
            rows[label][v] = {"report": float(oracle_ap(cand, v)[report_rows][mk].mean())}
    R["rows"] = rows

    # ================================================================== 4. bootstrap, shipping rule
    new_names = ["EW", "EW-blend", "reranker"]
    best_new = max(new_names, key=lambda n: rows[n]["VTOP5"]["report"])
    rng = np.random.default_rng(SEED)
    sig = {}
    for nm in new_names:
        sig[nm] = {}
        for v in VARS:
            (a1, mk), (a0, _) = aps[nm][v], aps["step-6 blend"][v]
            d = (a1 - a0)[mk]
            bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            sig[nm][v] = {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
    ship = bool(rows[best_new]["VTOP5"]["report"] > rows["step-6 blend"]["VTOP5"]["report"] and sig[best_new]["VTOP5"]["ci95"][0] > 0)
    R["significance_vs_step6_blend"] = sig
    R["decision"] = {"best_new_strategy": best_new, "ship_new": ship,
                     "rule": "ship only if the best new strategy beats the step-6 blend on REPORT VTOP5 with the 95% CI of the paired difference excluding zero"}
    lap("evaluation")

    # ================================================================== final file (only if the rule says ship)
    if ship:
        print("final run:", best_new, flush=True)
        users_f = data.load_submission_users()
        pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
        if best_new == "reranker":
            ctxf = make_ctx(["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
            store_f, _ = build_store(ctxf, ["H2", "PPR", "MF", "AR"], cfg)
            store_f["EW"], _ = build_store(ctxf, ["EW"], cfg, (ew_m, ew_c))[0]["EW"], None
            candf = rerank.candidates(store_f, ctxf["pop_norm"], len(users_f), len(pool_f), names=rerank.ORDER)
            rif, cif, Xf, _ = rerank.make_features(ctxf, store_f, ctxf["pop_norm"], candf)
            top = rerank.top5(model, Xf, rif, cif, np.arange(len(users_f)))
            used_comps = rerank.ORDER[:-1]
        else:
            ctxf = make_ctx(["11-20", "21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
            spec = fuse_spec(w_ewb) if best_new == "EW-blend" else fuse_spec({"EW": 1.0})
            need = [n for n, w in spec["w"].items() if w > 0]
            store_f, _ = build_store(ctxf, [n for n in need if n != "EW"], cfg)
            if "EW" in need:
                store_f["EW"] = build_store(ctxf, ["EW"], cfg, (ew_m, ew_c))[0]["EW"]
            top = s5.make_top_fn(ctxf["pop_norm"])(spec, store_f, len(users_f))
            used_comps = need
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / "submission_STEP7.csv"
        write_submission(preds, users_f, path)
        pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
        R["final"] = {**s3.check_file(path, users_f, pool_f), "strategy": best_new, "train_window": ctxf["periods"],
                      "weights": w_ewb if best_new == "EW-blend" else None,
                      "users_without_component_data": {n: int((~store_f[n][1]).sum()) for n in used_comps if n in store_f},
                      "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / K for u in users_f]))}
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP7.csv" not in txt:
            desc = {"reranker": f"step 7: LightGBM reranker ({best_rr['objective']}) over top-50 union of H2, EW, PPR, MF, AR, P-air",
                    "EW-blend": "step 7: P-air + MF + PPR + EW", "EW": "step 7: EW expected-watches scorer"}[best_new]
            hp = {"reranker": f"fitted on TUNE users (weeks 11-20 features, weeks 21-30 VTOP5 labels), best iteration {best_rr['best_iteration']}; applied to features from weeks 21-30; EW m {ew_m}, c {ew_c}",
                  "EW-blend": f"{w_ewb}, P-air {round(1 - sum(w_ewb.values()), 1)}; EW m {ew_m}, c {ew_c}", "EW": f"m {ew_m}, c {ew_c}"}[best_new]
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP7.csv | {desc} | {hp} | | |\n", encoding="utf-8")
    R["laps_seconds"] = laps
    R["runtime_s"] = round(time.time() - T0, 1)
    (OUT / "validation_step7.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_md(R)
    print(f"done in {R['runtime_s']} s; best new {best_new}; ship {ship}")


def write_md(R):
    f4 = lambda x: f"{x:.4f}"
    V = VARS
    ce, rows = R["ceiling"], R["rows"]
    md = ["# Validation, step 7", "",
          "Primary criterion VTOP5 (pre-declared in step 6); V300/V900/VHALF are robustness checks. Same TUNE/REPORT halves (seed 42), pool, repeats allowed, "
          "5 distinct ids, popularity fill (P-air). No Kaggle number is used for any choice. REPORT users with non-empty truth in a variant only.", "",
          "## 1. Ceiling diagnostics (VTOP5 truth, validation weeks 21-30)", "",
          f"Truth items: {ce['truth_items_all']:,} (all users with truth), {ce['truth_items_report']:,} (REPORT). Mean candidate-union size per user: "
          f"{ce['union_candidates_per_user_mean_5comp']:.0f} (5 components), {ce['union_candidates_per_user_mean_6comp']:.0f} (with EW).", "",
          "| share of truth items that ... | all validation users | REPORT |", "|---|---|---|"]
    for k, lab in (("a_watched_at_all_in_TRAIN", "(a) the user watched at all in TRAIN (weeks 11-20)"),
                   ("b_completed_ge80pct_in_TRAIN", "(b) the user completed >= 80% at least once in TRAIN"),
                   ("c_in_top50_union_H2_PPR_MF_AR_Pair", "(c) are in the union of the top-50 of H2, PPR, MF, AR and P-air"),
                   ("c6_in_top50_union_plus_EW", "(c') same union plus EW (reranker candidates)")):
        md.append(f"| {lab} | {f4(ce[k]['all_val_users'])} | {f4(ce[k]['report_users'])} |")
    md += ["", "## 2. EW (expected watches)", "",
           "| m | recency c | global completion rate | TUNE V300 / V900 / VHALF / VTOP5 |", "|---|---|---|---|"]
    for g in R["ew"]["grid"]:
        md.append(f"| {g['m']} | {g['c']} | {g['global_rate']:.4f} | " + " / ".join(f4(g[f"tune_{v}"]) for v in V) + " |")
    eb = R["ew_blend"]
    md += ["", f"Chosen on TUNE VTOP5: m = {R['ew']['chosen']['m']}, c = {R['ew']['chosen']['c']}. EW-blend (EW in place of H2; weights re-chosen on TUNE VTOP5): "
               f"{eb['weights']}, P-air {eb['popularity_weight']} (TUNE by variant {[round(x, 4) for x in eb['tune_by_variant'].values()]}).", "",
           "## 3. Learned reranker", "",
           f"Candidates: top-{rerank.TOPN} union of {', '.join(rerank.ORDER)}; {R['reranker']['candidates_pairs']:,} (user, show) pairs, {R['reranker']['positives']:,} positives (VTOP5). "
           f"Fit on TUNE users only: {R['reranker']['split']}; seed {SEED}; early stopping on the held-out 20%. Objective chosen by held-out MAP@5 (VTOP5):", "",
           "| objective | best iteration | held-out MAP@5 (VTOP5) |", "|---|---|---|"]
    for o in R["reranker"]["objectives"]:
        md.append(f"| {o['objective']} | {o['best_iteration']} | {f4(o['holdout_map5_vtop5'])} |")
    md += ["", f"Chosen: **{R['reranker']['chosen_objective']}**. Top-10 feature importances (gain share): {R['reranker']['top10_importance_gain']}.", "",
           f"Final-run rule: {R['reranker']['final_rule']}.", "",
           "## 4. REPORT MAP@5", "", "| strategy | " + " | ".join(V) + " | mean | fit s | score s | mean train popularity / repeat share (VTOP5 REPORT users) |", "|---|" + "---|" * (len(V) + 4)]
    for n, r in rows.items():
        vals = [r[v]["report"] for v in V]
        st = r.get("stats_vtop5_report_users")
        md.append(f"| {n} | " + " | ".join(f4(x) for x in vals) + f" | {f4(np.mean(vals))} | {r.get('fit_s', '')} | {r.get('score_s', '')} | "
                  + (f"{st['mean_train_popularity']:.0f} / {st['repeat_share_train_positive']:.2f}" if st else "") + " |")
    md += ["", "Oracle rows: candidates ranked perfectly, AP@5 = min(5, hits in union) / min(5, |truth|); they are the ceiling of any reranker over those candidates.", "",
           "## 5. Paired bootstrap vs the step-6 blend (REPORT, 2,000 resamples, seed 42): mean AP@5 difference [95% CI]", "",
           "| new strategy | " + " | ".join(V) + " |", "|---|" + "---|" * len(V)]
    for n, d in R["significance_vs_step6_blend"].items():
        md.append(f"| {n} | " + " | ".join(f"{d[v]['mean_diff']:+.4f} [{d[v]['ci95'][0]:+.4f}, {d[v]['ci95'][1]:+.4f}]" for v in V) + " |")
    dc = R["decision"]
    md += ["", "## 6. Shipping decision", "", f"Rule: {dc['rule']}. Best new strategy on REPORT VTOP5: **{dc['best_new_strategy']}**; ship: **{dc['ship_new']}**.", ""]
    if "final" in R:
        f = R["final"]
        md += [f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}",
               f"- TRAIN window {f['train_window']}; users without component data (popularity fallback): {f['users_without_component_data']}; repeat share {f['repeat_share_train_positive']:.3f}"]
    else:
        md += ["No new file written; `submission_FINAL_vtop5.csv` (step 6) stays final."]
    md += ["", f"Runtime: {R['runtime_s']} s (stage clocks: {R['laps_seconds']})"]
    (OUT / "validation_step7.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

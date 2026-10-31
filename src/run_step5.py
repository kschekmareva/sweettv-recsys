"""Step 5: airing-level VTOP5 truth, H2 own-history scorer, pool option, hybrids, robust choice, final submission.
Usage: python src/run_step5.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import time

import numpy as np
import pandas as pd

import assoc
import baselines
import data
import h2
import hybrid
import metrics
import mf
import ppr
import run_step3 as s3
import run_step4 as s4
from run_validation import write_submission

SEED, THR, K = 42, 300, 5
VARS = ["V300", "V900", "VHALF", "VTOP5"]
W_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
MF_ALPHAS = (0.3, 0.5, 1, 3)
MF_BASE = dict(factors=32, reg=0.1)
MF4_CFG = dict(factors=32, reg=0.1, alpha=1)        # step-4 choice, kept for the unchanged step-4 blend
PPR_ALPHA, AR_MODE = 0.3, "max"                      # step-4 choices
BLEND4 = {"MF4": 0.4, "PPR": 0.4}                    # step-4 blend; popularity (P-air) gets the remaining 0.2
LONG_SEC, FILMS = 10000, "Фильмы"
OPTIONS = {"none": (), "no_films": ("films",), "no_long": ("long",), "no_films_no_long": ("films", "long")}
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
fuse_eps = s4.fuse_eps


def pool_attrs(sched, w0, w1, pool):
    """Category and median airing duration (s) of each pool show in the pool-defining weeks."""
    m = (sched["tv_show_id"] != 0) & sched["iso_week"].between(w0, w1) & (sched["iso_year"] == 2020)
    d = sched[m].assign(dur=sched["end"] - sched["start"])
    cat = d.groupby("tv_show_id")["tv_show_category"].first().reindex(pool).values
    med = d.groupby("tv_show_id")["dur"].median().reindex(pool).values
    return cat, med


def allowed_mask(option, cat, med):
    ok = np.ones(len(cat), dtype=bool)
    if "films" in option:
        ok &= cat != FILMS
    if "long" in option:
        ok &= med < LONG_SEC
    return ok


def fuse_spec(w):
    return {"kind": "fuse", "w": dict(w)}


def used_comps(spec):
    return sorted({n for n, w in spec["w"].items() if w > 0} | ({"H2"} if spec["kind"] == "h2first" else set()))


def make_top_fn(pop_norm):
    """top_of(spec, store) -> (rows, 5) column indices of the pool the store was built on."""
    def top_of(spec, store, n):
        def fused(w, k):
            act = [(c, x) for c, x in w.items() if x > 0]
            S = fuse_eps(pop_norm, [store[c] for c, _ in act], [x for _, x in act]) if act else \
                np.tile((pop_norm + s4.EPS * pop_norm)[None, :], (n, 1)).astype(np.float32)
            return hybrid.topk(S, k)
        if spec["kind"] == "fuse":
            return fused(spec["w"], K)
        top_b = fused(spec["w"], 10)                       # H2 first, remaining slots from the blend
        top_h = fused({"H2": 1.0}, K)
        Sn_h, valid_h = store["H2"]
        has = (np.take_along_axis(Sn_h, top_h, axis=1) > 0) & valid_h[:, None]
        out = np.empty((n, K), dtype=np.int64)
        for i in range(n):
            sel = [int(top_h[i, j]) for j in range(K) if has[i, j]][:spec["n"]]
            for x in top_b[i]:
                if len(sel) == K:
                    break
                if int(x) not in sel:
                    sel.append(int(x))
            out[i] = sel
        return out
    return top_of


def main():
    t_all = time.time()
    R = {"seed": SEED, "variants": VARS, "idea_source": "kinahem, 'gornyaki final solution' (Kaggle notebook), "
         "https://www.kaggle.com/code/kinahem/gornyaki-final-solution - concept only, no code copied"}
    np.random.seed(SEED)

    # ================================================================== data, truth variants
    inter, _ = data.build_interactions()
    recent = data.recent_interactions()
    train_all = data.positives(inter, ["11-20"], 0)
    train_pos = data.positives(inter, ["11-20"], THR)
    air_tr, air_val = data.airings(["11-20"]), data.airings(["21-30"])
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300], "V900": val[val["overlap_sec"] >= 900], "VHALF": val[val["max_frac"] >= 0.5]}
    vpairs["VTOP5"], n_shows_80 = h2.vtop5_truth(air_val)
    sched_val = data.load_schedule("21-30")
    pool_full = data.pool_ids(sched_val, 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    users = sorted(set(int(u) for v in vpairs.values() for u in v["user_id"].unique()) | set(sub_users))
    uix, pix = pd.Index(np.asarray(users, dtype=np.uint64)), pd.Index(pool_full)
    n_u, n_pf = len(users), len(pool_full)
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_pf) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    exact5 = T["VTOP5"].sum(axis=1) == 5
    Tpos, Tall = s4.bool_matrix(train_pos, uix, pix, n_u, n_pf), s4.bool_matrix(train_all, uix, pix, n_u, n_pf)

    perm = np.random.default_rng(SEED).permutation(len(eval_a))      # the step-3/4 halves
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    sub_rows = uix.get_indexer(np.asarray(sub_users, dtype=np.uint64))
    tune_v = {v: tune_rows[nonempty[v][tune_rows]] for v in VARS}
    report_v = {v: report_rows[nonempty[v][report_rows]] for v in VARS}
    full_v = {v: np.flatnonzero(nonempty[v]) for v in VARS}
    Tt = {v: T[v][tune_rows] for v in VARS}
    mt = {v: nonempty[v][tune_rows] for v in VARS}
    tune4 = lambda top: np.array([metrics.ap5_matrix(top, Tt[v])[mt[v]].mean() for v in VARS])

    # airing table vs interaction table (consistency of the new join level)
    g = air_val.groupby(["user_id", "tv_show_id"])["view_pct"].max().reset_index().merge(
        val[["user_id", "tv_show_id", "max_frac"]], on=["user_id", "tv_show_id"])
    R["airing_consistency"] = {"airing_rows_val": int(len(air_val)), "user_show_pairs": int(len(g)), "interaction_pairs_val": int(len(val)),
                               "share_max_pct_equals_max_frac": float((np.abs(g["view_pct"] - g["max_frac"]) < 1e-5).mean()),
                               "airings_with_pct_ge_0.8": int((air_val["view_pct"] >= 0.8).sum())}
    R["sizes"] = {"pool_full": n_pf, "users_universe": n_u, "tune_users_V300": len(tune_rows), "report_users_V300": len(report_rows),
                  "submission_users": len(sub_users)}
    R["variant_stats"] = {}
    for v in VARS:
        pp = vpairs[v].groupby("user_id").size()
        R["variant_stats"][v] = {
            "positive_pairs": int(len(vpairs[v])), "users_with_truth": int(nonempty[v].sum()),
            "tune_users": int(len(tune_v[v])), "report_users": int(len(report_v[v])),
            "submission_users_with_truth": int(nonempty[v][sub_rows].sum()),
            "positives_per_user_quantiles": {str(q): float(x) for q, x in pp.quantile([.1, .25, .5, .75, .9]).items()},
            "positives_per_user_mean": float(pp.mean())}
    R["variant_stats"]["VTOP5"].update({
        "users_with_exactly_5_truth_items": int(exact5.sum()),
        "report_users_exactly_5": int(exact5[report_v["VTOP5"]].sum()),
        "shows_with_ge1_airing_pct80_per_user_quantiles": {str(q): float(x) for q, x in n_shows_80.quantile([.1, .25, .5, .75, .9]).items()}})

    # ================================================================== popularity on the full pool, H2 grid (c on TUNE VTOP5)
    P_full = hybrid.pop_counts(train_pos, pool_full)
    air_cnt_full = s4.airing_counts(sched_val, 21, 30, pool_full)
    Pair_full = P_full * np.log1p(air_cnt_full)
    pn_full = hybrid.minmax_vec(Pair_full)
    n_air_full = h2.train_airing_counts(pool_full, ("11-20",))
    train_end_val = h2.sec_of("2020-05-18")
    prep = h2.prepare(air_tr)
    print("H2 c grid", flush=True)
    h2_grid, best_h2 = [], None
    for c in h2.C_GRID:
        S, valid, ts = h2.scores(prep, n_air_full, users, pool_full, train_end_val, c)
        Sn = hybrid.minmax_rows(S)
        m4 = tune4(hybrid.topk(fuse_eps(pn_full, [(Sn[tune_rows], valid[tune_rows])], [1.0]), K))
        h2_grid.append({"c": c, **{f"tune_{v}": float(x) for v, x in zip(VARS, m4)}, "score_s": round(ts, 2)})
        print("  ", h2_grid[-1], flush=True)
        if best_h2 is None or m4[3] > best_h2[0]:
            best_h2 = (m4[3], c, Sn[tune_rows], valid[tune_rows])
    c_best = best_h2[1]
    R["h2"] = {"c_grid": h2_grid, "c_best": c_best, "prepare_s": round(prep["seconds"], 2), "airings_pct80_train": prep["airings"],
               "definition": "sum over airings with view_pct>=0.8 of r*c^(days_before_train_end/60), r=5 if view_pct=1 else 2+2*view_pct^2, "
                             "times log10(1+airings of the show in the TRAIN schedule); train end 2020-05-18 00:00 (validation), pure-H2 fallback and ties: P-air"}

    # ================================================================== candidate-pool option (decided on TUNE with P and H2)
    cat_full, med_full = pool_attrs(sched_val, 21, 30, pool_full)
    opt_res = {}
    for name, opt in OPTIONS.items():
        ok = allowed_mask(opt, cat_full, med_full)
        vec = np.where(ok, P_full, -np.inf)
        top_p = np.tile(np.argsort(-vec, kind="stable")[:K], (len(tune_rows), 1))
        S = fuse_eps(pn_full, [(best_h2[2], best_h2[3])], [1.0])
        S[:, ~ok] = -np.inf
        opt_res[name] = {"pool_size": int(ok.sum()), "P": tune4(top_p), "H2": tune4(hybrid.topk(S, K))}
    qual = {}
    for name in OPTIONS:
        if name == "none":
            continue
        dP, dH = opt_res[name]["P"] - opt_res["none"]["P"], opt_res[name]["H2"] - opt_res["none"]["H2"]
        qual[name] = bool(dP[3] > 0 and dH[3] > 0 and min(dP[:3].min(), dH[:3].min()) >= -0.005)
    adopt = max((n for n in qual if qual[n]), key=lambda n: (opt_res[n]["P"][3] - opt_res["none"]["P"][3] + opt_res[n]["H2"][3] - opt_res["none"]["H2"][3]), default="none")
    R["pool_option"] = {"table_tune": {n: {"pool_size": r["pool_size"], "P": r["P"].tolist(), "H2": r["H2"].tolist()} for n, r in opt_res.items()},
                        "variants_order": VARS, "qualifies": qual, "adopted": adopt,
                        "rule": "adopt only if both P and H2 improve on TUNE VTOP5 and neither loses more than 0.005 on V300/V900/VHALF; "
                                "among qualifying options take the largest summed VTOP5 gain; programme duration = median airing duration in the pool-defining weeks"}
    print("pool option:", adopt, qual, flush=True)
    ok = allowed_mask(OPTIONS[adopt], cat_full, med_full)
    col_map = np.flatnonzero(ok)
    pool = pool_full[col_map]
    P_v, Pair_v = P_full[col_map], Pair_full[col_map]
    pop_norm = hybrid.minmax_vec(Pair_v)
    n_air_tr = n_air_full[col_map]
    top_of = make_top_fn(pop_norm)

    def tune_eval(spec, store_t):
        return tune4(col_map[top_of(spec, store_t, len(tune_rows))])

    # ================================================================== components on the (possibly restricted) pool
    comps = {}
    S, valid, ts = h2.scores(prep, n_air_tr, users, pool, train_end_val, c_best)
    t = time.time(); comps["H2"] = {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": prep["seconds"], "score_s": ts + time.time() - t}
    sub_of = lambda c: (c["Sn"][tune_rows], c["valid"][tune_rows])
    print("MF grid", flush=True)
    mf_rows, best_mf, mf4 = [], None, None
    for a in MF_ALPHAS:
        cfg = {**MF_BASE, "alpha": a}
        c = s3.mf_component(train_all, cfg, users, pool)
        curve = np.array([tune_eval(fuse_spec({"MF": w}), {"MF": sub_of(c)}) for w in W_GRID])
        pure = tune_eval(fuse_spec({"MF": 1.0}), {"MF": sub_of(c)})
        sc = float(curve.max(axis=0).mean())
        mf_rows.append({**cfg, "fit_s": round(c["fit_s"], 2), "score_s": round(c["score_s"], 2), "hybrid_tune_mean_over_variants": sc,
                        "best_w_per_variant": [W_GRID[int(i)] for i in curve.argmax(axis=0)],
                        "pure_tune": {v: float(x) for v, x in zip(VARS, pure)}})
        print("  ", mf_rows[-1], flush=True)
        if best_mf is None or sc > best_mf[0]:
            best_mf = (sc, cfg, c)
        if a == MF4_CFG["alpha"]:
            mf4 = c
    comps["MF"], comps["MF4"] = best_mf[2], mf4
    print("PPR / AR", flush=True)
    gr = ppr.build_graph(train_pos)
    S, valid, info = ppr.run(gr, users, pool, PPR_ALPHA)
    t = time.time(); comps["PPR"] = {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": gr["build_s"], "score_s": info["seconds"] + time.time() - t}
    R["ppr"] = {k: info[k] for k in ("alpha", "iterations_max", "converged_all_chunks", "final_max_l1_change")}
    am = assoc.fit(train_pos, pool)
    S, valid, ts = assoc.score(am, train_pos, users, AR_MODE)
    t = time.time(); comps["AR"] = {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": am["build_s"], "score_s": ts + time.time() - t}
    R["mf"] = {"grid": mf_rows, "chosen": best_mf[1], "step4_config_kept_for_blend": MF4_CFG, "same_as_step4": best_mf[1]["alpha"] == MF4_CFG["alpha"]}
    store_t = {n: sub_of(c) for n, c in comps.items()}
    store_f = {n: (c["Sn"], c["valid"]) for n, c in comps.items()}

    # ================================================================== weight grids on TUNE
    print("weight grids", flush=True)
    fam = {
        "H2+P-air": [fuse_spec({"H2": w}) for w in W_GRID],
        "H2+PPR": [fuse_spec({"H2": w, "PPR": round(1 - w, 1)}) for w in W_GRID],
        "H2+AR": [fuse_spec({"H2": w, "AR": round(1 - w, 1)}) for w in W_GRID],
        "H2+MF": [fuse_spec({"H2": w, "MF": round(1 - w, 1)}) for w in W_GRID],
        "P-air+MF+PPR+H2": [fuse_spec({"MF": a / 10, "PPR": b / 10, "H2": c / 10}) for a in range(11) for b in range(11 - a) for c in range(11 - a - b)],
        "H2-first + step-4 blend": [{"kind": "h2first", "n": n, "w": BLEND4} for n in range(K + 1)],
    }
    curves = {}
    for name, specs in fam.items():
        curves[name] = np.array([tune_eval(s, store_t) for s in specs])
        print("  ", name, len(specs), "best TUNE per variant", curves[name].max(axis=0).round(4), flush=True)

    # ================================================================== rows
    rows_out = {}
    pidx = {int(p): i for i, p in enumerate(pool)}

    def add_row(name, tops, fit_s, score_s, weights=None, kind="pure"):
        m = {}
        for v in VARS:
            ap = metrics.ap5_matrix(tops[v], T[v])
            r = full_v[v]
            d = {**{"tune": float(ap[tune_v[v]].mean()), "report": float(ap[report_v[v]].mean()), "full": float(ap[r].mean()),
                    "submission": float(ap[sub_rows].mean())},
                 "mean_train_popularity": float(P_full[tops[v][r]].mean()),
                 "repeat_share_train_positive": float(Tpos[r[:, None], tops[v][r]].mean()),
                 "repeat_share_any_pair": float(Tall[r[:, None], tops[v][r]].mean())}
            if v == "VTOP5":
                d["report_exact5"] = float(ap[report_v[v][exact5[report_v[v]]]].mean())
                d["full_exact5"] = float(ap[r[exact5[r]]].mean())
            d["_ap"] = ap
            m[v] = d
        rows_out[name] = {"metrics": m, "fit_s": round(fit_s, 2), "score_s": round(score_s, 2), "weights": weights, "kind": kind}

    def spec_tops(spec_by_v):
        tops, fuse_s, used = {}, 0.0, set()
        for v in VARS:
            t = time.time()
            tops[v] = col_map[top_of(spec_by_v[v], store_f, n_u)]
            fuse_s = max(fuse_s, time.time() - t)
            used |= set(used_comps(spec_by_v[v]))
        return tops, sum(comps[n]["fit_s"] for n in used), sum(comps[n]["score_s"] for n in used) + fuse_s

    same = lambda x: {v: x for v in VARS}
    for name, vec in (("P", P_v), ("P-air", Pair_v)):
        add_row(name, same(col_map[np.tile(np.argsort(-vec, kind="stable")[:K], (n_u, 1))]), 0.0, 0.0)
    t = time.time(); baselines.own_history_lists(train_pos, pool); fit_h = time.time() - t
    t = time.time(); preds_h = baselines.recommend_history(train_pos, pool, users, K); sc_h = time.time() - t
    add_row("H", same(col_map[np.array([[pidx[x] for x in preds_h[u]] for u in users])]), fit_h, sc_h)
    for n in ("H2", "MF", "PPR", "AR"):
        tops, f, s_ = spec_tops(same(fuse_spec({n: 1.0})))
        add_row(n, tops, f, s_, {n: 1.0})
    tops, f, s_ = spec_tops(same(fuse_spec({"MF4": 0.4, "PPR": 0.4})))
    add_row("step-4 blend", tops, f, s_, {"P-air": 0.2, "MF4": 0.4, "PPR": 0.4}, "fixed")
    shared = {}
    for name, specs in fam.items():
        cur = curves[name]
        sv = {v: specs[int(cur[:, j].argmax())] for j, v in enumerate(VARS)}
        tops, f, s_ = spec_tops(sv)
        add_row(name, tops, f, s_, {v: (sv[v]["w"] if sv[v]["kind"] == "fuse" else {"n_H2_first": sv[v]["n"], **sv[v]["w"]}) for v in VARS}, "hybrid")
        ss = specs[int(cur.mean(axis=1).argmax())]
        tops, f, s_ = spec_tops(same(ss))
        w_ss = ss["w"] if ss["kind"] == "fuse" else {"n_H2_first": ss["n"], **ss["w"]}
        add_row(f"{name} [shared w]", tops, f, s_, w_ss, "shared")
        shared[name] = {"weights": w_ss, "spec": ss, "tune_mean": float(cur.mean(axis=1).max())}

    # ================================================================== ranking, winner, bootstrap
    ranked = [n for n, r in rows_out.items() if r["kind"] != "shared"]
    rk = {v: pd.Series({n: rows_out[n]["metrics"][v]["report"] for n in ranked}).rank(ascending=False, method="min") for v in VARS}
    avg_rank = {n: float(np.mean([rk[v][n] for v in VARS])) for n in ranked}
    mean_report = {n: float(np.mean([rows_out[n]["metrics"][v]["report"] for v in VARS])) for n in ranked}
    winner = min(ranked, key=lambda n: (avg_rank[n], -mean_report[n]))
    ship = f"{winner} [shared w]" if f"{winner} [shared w]" in rows_out else winner
    R["ranking"] = {"avg_rank_report": avg_rank, "winner": winner, "shipped_row": ship,
                    "rule": "winner = lowest average REPORT rank over the 4 variants (per-variant-tuned weights); shipped = the single weight set "
                            "maximising the mean over the 4 variants of TUNE MAP@5 (ties: first in grid)"}
    rng = np.random.default_rng(SEED)
    R["significance"] = {"strategy": ship, "resamples": 2000, "vs": {}}
    for ref in ("P", "step-4 blend"):
        out = {}
        for v in VARS:
            idx = report_v[v]
            d = rows_out[ship]["metrics"][v]["_ap"][idx] - rows_out[ref]["metrics"][v]["_ap"][idx]
            bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            out[v] = {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
        R["significance"]["vs"][ref] = out

    # ================================================================== final submissions
    print("final submissions", flush=True)
    SUBDIR.mkdir(parents=True, exist_ok=True)
    train_all_f = data.positives(inter, list(data.PERIODS), 0)
    train_pos_f = data.positives(inter, list(data.PERIODS), THR)
    air_f = data.airings(["11-20", "21-30"])
    sched_f = data.load_schedule("31-42")
    pool_f_full = data.pool_ids(sched_f, 31, 42)
    cat_f, med_f = pool_attrs(sched_f, 31, 42, pool_f_full)
    col_f = np.flatnonzero(allowed_mask(OPTIONS[adopt], cat_f, med_f))
    pool_f = pool_f_full[col_f]
    users_f = data.load_submission_users()
    pop_norm_f = hybrid.minmax_vec(hybrid.pop_counts(train_pos_f, pool_f) * np.log1p(s4.airing_counts(sched_f, 31, 42, pool_f)))
    top_of_f = make_top_fn(pop_norm_f)
    ship_spec = rows_out[ship]["weights"]
    if ship != winner:
        ship_spec_obj = shared[winner]["spec"]
    elif winner == "step-4 blend":
        ship_spec_obj = fuse_spec(BLEND4)
    elif winner in ("H2", "MF", "PPR", "AR"):
        ship_spec_obj = fuse_spec({winner: 1.0})
    else:                                              # P / P-air (plain popularity); H is not supported in the final run
        ship_spec_obj = fuse_spec({})
    need = set(used_comps(ship_spec_obj)) | {"H2"}
    t = time.time()
    comps_f = {}
    if "H2" in need:
        prep_f = h2.prepare(air_f)
        S, valid, ts = h2.scores(prep_f, h2.train_airing_counts(pool_f, ("11-20", "21-30")), users_f, pool_f, h2.sec_of("2020-07-27"), c_best)
        comps_f["H2"] = {"Sn": hybrid.minmax_rows(S), "valid": valid}
    for n in need & {"MF", "MF4"}:
        cfg = best_mf[1] if n == "MF" else MF4_CFG
        comps_f[n] = s3.mf_component(train_all_f, cfg, users_f, pool_f)
    if "PPR" in need:
        g_f = ppr.build_graph(train_pos_f)
        S, valid, _ = ppr.run(g_f, users_f, pool_f, PPR_ALPHA)
        comps_f["PPR"] = {"Sn": hybrid.minmax_rows(S), "valid": valid}
    if "AR" in need:
        am_f = assoc.fit(train_pos_f, pool_f)
        S, valid, _ = assoc.score(am_f, train_pos_f, users_f, AR_MODE)
        comps_f["AR"] = {"Sn": hybrid.minmax_rows(S), "valid": valid}
    store_ff = {n: (c["Sn"], c["valid"]) for n, c in comps_f.items()}
    files = {}
    for tag, spec in (("H2", fuse_spec({"H2": 1.0})), ("CHOSEN_step5", ship_spec_obj)):
        top = top_of_f(spec, store_ff, len(users_f))
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / f"submission_{tag}.csv"
        write_submission(preds, users_f, path)
        files[tag] = {**s3.check_file(path, users_f, pool_f_full), "ids_in_restricted_pool": bool(all(set(p) <= set(int(x) for x in pool_f) for p in preds.values())),
                      "spec": spec, "users_without_component_data": {n: int((~comps_f[n]["valid"]).sum()) for n in used_comps(spec)}}
    R["final"] = {"pool_full": int(len(pool_f_full)), "pool_used": int(len(pool_f)), "pool_option": adopt, "c": c_best, "mf_cfg": best_mf[1],
                  "chosen": ship, "chosen_weights": ship_spec, "train_positive_pairs_weeks_11_30": int(len(train_pos_f)),
                  "build_seconds": round(time.time() - t, 1), "files": files}
    R["runtime"] = {"total_s": round(time.time() - t_all, 1)}

    # ================================================================== write
    for r in rows_out.values():
        for v in VARS:
            r["metrics"][v].pop("_ap", None)
    R["rows"] = {n: {k: r[k] for k in ("metrics", "fit_s", "score_s", "weights", "kind")} for n, r in rows_out.items()}
    R["tune_curves_best"] = {n: {"best_per_variant": c.max(axis=0).tolist(), "best_mean": float(c.mean(axis=1).max())} for n, c in curves.items()}
    write_md(R, rows_out, avg_rank)
    (OUT / "validation_step5.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"done in {R['runtime']['total_s']} s; winner {winner}; shipped {ship}; pool option {adopt}")


def write_md(R, rows, avg_rank):
    f4 = lambda x: f"{x:.4f}"
    V = VARS
    md = ["# Validation, step 5", "",
          f"Idea source for the airing-level truth and H2 (concept only, no code copied): {R['idea_source']}.", "",
          "Same TUNE/REPORT halves, seed 42, repeats allowed, 5 distinct ids, popularity fill. Only users with non-empty truth in a variant count for that variant "
          "(submission column: all 1,260 users, empty truth = 0). Fusion: per-user min-max, weights chosen on TUNE, 1e-6 popularity tie-break; "
          "invalid-component weight goes to popularity (P-air). Pure H2/MF/PPR/AR rows fall back to P-air.", "",
          "## 0. Airing-level join and truth variants", "",
          f"Airing table consistency (weeks 21-30): {R['airing_consistency']}", "",
          "| variant | positive pairs | users with truth | TUNE users | REPORT users | positives/user p10 / p25 / p50 / p75 / p90 | mean | submission users with truth |", "|---|---|---|---|---|---|---|---|"]
    for v in V:
        s = R["variant_stats"][v]
        q = s["positives_per_user_quantiles"]
        md.append(f"| {v} | {s['positive_pairs']:,} | {s['users_with_truth']:,} | {s['tune_users']:,} | {s['report_users']:,} | "
                  f"{q['0.1']:g} / {q['0.25']:g} / {q['0.5']:g} / {q['0.75']:g} / {q['0.9']:g} | {s['positives_per_user_mean']:.1f} | {s['submission_users_with_truth']} |")
    v5 = R["variant_stats"]["VTOP5"]
    sq = v5["shows_with_ge1_airing_pct80_per_user_quantiles"]
    pm = rows["P"]["metrics"]
    md += ["", f"VTOP5: {v5['users_with_exactly_5_truth_items']:,} users have exactly 5 truth items ({v5['report_users_exactly_5']} in REPORT). Shows with >= 1 airing at view_pct >= 0.8 per user "
               f"(before the top-5 cut) p10/25/50/75/90: {sq['0.1']:g} / {sq['0.25']:g} / {sq['0.5']:g} / {sq['0.75']:g} / {sq['0.9']:g}.", "",
           "P (popularity) MAP@5: " + "; ".join(f"{v}: REPORT {f4(pm[v]['report'])}, full {f4(pm[v]['full'])}, submission {f4(pm[v]['submission'])}" for v in V),
           f"; VTOP5 exactly-5 subset: REPORT {f4(pm['VTOP5']['report_exact5'])}, full {f4(pm['VTOP5']['full_exact5'])}.",
           "Plausibility check only (nothing tuned to it): Kaggle private score of P is 0.278; P under VTOP5 on the 1,260 submission users is "
           f"{f4(pm['VTOP5']['submission'])}.", "",
           "## 1. H2 (c chosen on TUNE VTOP5, pure H2)", "", "| c | TUNE V300 | V900 | VHALF | VTOP5 |", "|---|---|---|---|---|"]
    for g in R["h2"]["c_grid"]:
        md.append(f"| {g['c']} | " + " | ".join(f4(g[f"tune_{v}"]) for v in V) + " |")
    md += ["", f"Chosen c = {R['h2']['c_best']}. {R['h2']['definition']}. Airings with view_pct >= 0.8 in TRAIN: {R['h2']['airings_pct80_train']:,}; preparation {R['h2']['prepare_s']} s.", "",
           "## 2. Candidate-pool option (TUNE, P and H2)", "", "| option | pool | P: V300 / V900 / VHALF / VTOP5 | H2: V300 / V900 / VHALF / VTOP5 | qualifies |", "|---|---|---|---|---|"]
    for n, r in R["pool_option"]["table_tune"].items():
        md.append(f"| {n} | {r['pool_size']} | " + " / ".join(f4(x) for x in r["P"]) + " | " + " / ".join(f4(x) for x in r["H2"]) + f" | {R['pool_option']['qualifies'].get(n, '-')} |")
    md += ["", f"Adopted: **{R['pool_option']['adopted']}**. Rule: {R['pool_option']['rule']}.", ""]

    def table(metric, title):
        out = [f"## {title}", "", "| method | " + " | ".join(V) + " | mean | avg rank |", "|---|" + "---|" * (len(V) + 2)]
        for n, r in rows.items():
            vals = [r["metrics"][v][metric] for v in V]
            out.append(f"| {n} | " + " | ".join(f4(x) for x in vals) + f" | {f4(np.mean(vals))} | " + (f"{avg_rank[n]:.2f}" if n in avg_rank and metric == "report" else "") + " |")
        return out + [""]
    md += table("report", "3. MAP@5 on the REPORT half") + table("full", "4. MAP@5 on all validation users with truth") + \
          table("submission", "5. MAP@5 on the 1,260 submission users (partly overlaps TUNE)")
    md += ["VTOP5 on the subset with exactly 5 truth items (REPORT / full): " + "; ".join(f"{n}: {f4(r['metrics']['VTOP5']['report_exact5'])} / {f4(r['metrics']['VTOP5']['full_exact5'])}" for n, r in rows.items() if r['kind'] != 'shared'), "",
           f"Winner by average REPORT rank: **{R['ranking']['winner']}**; shipped row: **{R['ranking']['shipped_row']}**. Rule: {R['ranking']['rule']}.", "",
           "## 6. Hyperparameters", "", "### MF (32 factors, reg 0.1; selection = mean over the 4 variants of the best P-air + MF TUNE MAP@5)", "",
           "| alpha | fit s | score s | hybrid TUNE mean | best w per variant | pure MF TUNE " + " / ".join(V) + " |", "|---|---|---|---|---|---|"]
    for g in R["mf"]["grid"]:
        md.append(f"| {g['alpha']} | {g['fit_s']} | {g['score_s']} | {f4(g['hybrid_tune_mean_over_variants'])} | {g['best_w_per_variant']} | " + " / ".join(f4(g['pure_tune'][v]) for v in V) + " |")
    md += ["", f"Chosen MF: {R['mf']['chosen']} (step-4 config alpha=1 is kept as MF4 for the unchanged step-4 blend; identical to the chosen one: {R['mf']['same_as_step4']}). "
               f"PPR: alpha {R['ppr']['alpha']}, {R['ppr']['iterations_max']} iterations, converged {R['ppr']['converged_all_chunks']}. AR: mode max, min support 20 (step-4 choices, not re-tuned). "
               "The older MF configs (alpha 10/40 etc.) were not re-run.", "",
           "## 7. Weights (component weights; P-air gets the remainder)", "", "| row | " + " | ".join(V) + " | shared set (shipped rule) |", "|---|---|---|---|---|---|"]
    for n, r in rows.items():
        if r["kind"] == "hybrid":
            md.append(f"| {n} | " + " | ".join(str(r["weights"][v]) for v in V) + f" | {rows[n + ' [shared w]']['weights']} |")
    md += ["", "## 8. Cost and recommendation statistics (users with truth)", "",
           "Cells: mean TRAIN popularity (distinct TRAIN positive users) of recommended items / repeat share (already a TRAIN positive for the user). Times for ~4.8k users; hybrids include components.", "",
           "| row | fit s | score s | " + " | ".join(V) + " |", "|---|---|---|" + "---|" * len(V)]
    for n, r in rows.items():
        md.append(f"| {n} | {r['fit_s']} | {r['score_s']} | " + " | ".join(f"{r['metrics'][v]['mean_train_popularity']:.0f} / {r['metrics'][v]['repeat_share_train_positive']:.2f}" for v in V) + " |")
    for ref in ("P", "step-4 blend"):
        md += ["", f"## 9. Significance: {R['significance']['strategy']} vs {ref} on REPORT (paired bootstrap, 2000 resamples, seed {SEED})", "",
               "| variant | users | mean AP@5 difference | 95% CI |", "|---|---|---|---|"]
        for v in V:
            d = R["significance"]["vs"][ref][v]
            md.append(f"| {v} | {d['n_users']} | {d['mean_diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    fin = R["final"]
    md += ["", "## 10. Final submissions (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning)", "",
           f"Pool {fin['pool_full']:,} (used: {fin['pool_used']:,}, option {fin['pool_option']}); H2 c = {fin['c']}; chosen: {fin['chosen']} with {fin['chosen_weights']}.", ""]
    for tag, f in fin["files"].items():
        md.append(f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, ids in final pool {f['all_ids_in_final_pool']}, "
                  f"ids in used pool {f['ids_in_restricted_pool']}; spec {f['spec']}; users without component data (popularity fallback): {f['users_without_component_data']}")
    md += ["", f"Runtime: {R['runtime']['total_s']} s"]
    (OUT / "validation_step5.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Step 4: truth variants, PPR, association rules, popularity variants, hybrids, robustness, final submissions.
Usage: python src/run_step4.py
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
import cbf
import data
import hybrid
import metrics
import mf
import ppr
import run_step3 as s3
from run_validation import write_submission

SEED, THR, K = 42, 300, 5
VARIANTS = ["V300", "V900", "VHALF"]
W_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
EPS = 1e-6           # popularity tie-break added to every fused score
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
MF_EXTRA = [dict(factors=32, reg=0.1, alpha=1), dict(factors=32, reg=0.1, alpha=3)]
COMP_NAMES = ["H", "MF", "PPR", "AR"]          # components allowed in blends
SINGLE_NAMES = COMP_NAMES + ["CBF"]


def fuse_eps(pop_norm, parts, weights):
    out = hybrid.fuse(pop_norm, parts, weights)
    out += EPS * pop_norm[None, :]
    return out


def comp_H(train_pos, users, pool):
    """Own-history component: log1p(overlap/60) of the user's own in-pool TRAIN positives."""
    t0 = time.time()
    uix, pix = pd.Index(np.asarray(users, dtype=np.uint64)), pd.Index(np.asarray(pool))
    tp = train_pos[train_pos["tv_show_id"].isin(pool)]
    r, c = uix.get_indexer(tp["user_id"].values), pix.get_indexer(tp["tv_show_id"].values)
    ok = r >= 0
    S = np.zeros((len(users), len(pool)), dtype=np.float32)
    S[r[ok], c[ok]] = np.log1p(tp["overlap_sec"].values[ok] / 60.0)
    valid = S.max(axis=1) > 0
    return {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": 0.0, "score_s": time.time() - t0}


def airing_counts(sched, w0, w1, pool):
    m = (sched["tv_show_id"] != 0) & sched["iso_week"].between(w0, w1) & (sched["iso_year"] == 2020)
    return sched.loc[m].groupby("tv_show_id").size().reindex(np.asarray(pool)).fillna(0).values.astype(np.float32)


def pop_vectors(train_pos, recent_pos, pool, airings):
    p = hybrid.pop_counts(train_pos, pool)
    return {"P": p, "P-rec": hybrid.pop_counts(recent_pos, pool), "P-air": p * np.log1p(airings)}


def bool_matrix(df, uix, pix, n_users, n_pool):
    r, c = uix.get_indexer(df["user_id"].values), pix.get_indexer(df["tv_show_id"].values)
    ok = (r >= 0) & (c >= 0)
    M = np.zeros((n_users, n_pool), dtype=bool)
    M[r[ok], c[ok]] = True
    return M


def build_final_comp(name, ctx, sel, items):
    if name == "H":
        return comp_H(ctx["train_pos"], ctx["users"], ctx["pool"])
    if name == "MF":
        return s3.mf_component(ctx["train_all"], sel["mf_cfg"], ctx["users"], ctx["pool"])
    if name == "PPR":
        g = ppr.build_graph(ctx["train_pos"])
        S, valid, info = ppr.run(g, ctx["users"], ctx["pool"], sel["ppr_alpha"])
        return {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": g["build_s"], "score_s": info["seconds"], "info": info}
    if name == "AR":
        m = assoc.fit(ctx["train_pos"], ctx["pool"])
        S, valid, ts = assoc.score(m, ctx["train_pos"], ctx["users"], sel["ar_mode"])
        return {"Sn": hybrid.minmax_rows(S), "valid": valid, "fit_s": m["build_s"], "score_s": ts, "info": {k: v for k, v in m.items() if isinstance(v, (int, float))}}
    if name == "CBF":
        return s3.cbf_component(items[:4], items[4], ctx["train_pos"], ctx["users"], ctx["pool"])
    raise KeyError(name)


def main():
    t_all = time.time()
    R, tm = {"seed": SEED, "variants": VARIANTS, "eps_tiebreak": EPS}, {}
    np.random.seed(SEED)

    # ================================================================== data and truth variants
    inter, _ = data.build_interactions()
    recent = data.recent_interactions()
    train_all = data.positives(inter, ["11-20"], 0)
    train_pos = data.positives(inter, ["11-20"], THR)
    recent_pos = data.positives(recent, ["17-20"], THR)
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300], "V900": val[val["overlap_sec"] >= 900], "VHALF": val[val["max_frac"] >= 0.5]}
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    all_truth_users = set(int(u) for v in vpairs.values() for u in v["user_id"].unique())
    users = sorted(all_truth_users | set(sub_users))
    uix, pix = pd.Index(np.asarray(users, dtype=np.uint64)), pd.Index(pool)
    n_u, n_p = len(users), len(pool)
    T = {v: bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARIANTS}
    nonempty = {v: T[v].any(axis=1) for v in VARIANTS}
    Tpos, Tall = bool_matrix(train_pos, uix, pix, n_u, n_p), bool_matrix(train_all, uix, pix, n_u, n_p)

    # TUNE / REPORT: exactly the step-3 halves (same permutation of the sorted V300 users, seed 42)
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    sub_rows = uix.get_indexer(np.asarray(sub_users, dtype=np.uint64))
    tune_v = {v: tune_rows[nonempty[v][tune_rows]] for v in VARIANTS}
    report_v = {v: report_rows[nonempty[v][report_rows]] for v in VARIANTS}
    full_v = {v: np.flatnonzero(nonempty[v]) for v in VARIANTS}
    Tt = {v: T[v][tune_rows] for v in VARIANTS}
    mt = {v: nonempty[v][tune_rows] for v in VARIANTS}
    R["sizes"] = {"pool": n_p, "users_universe": n_u, "V300_users": len(eval_a), "tune_users_V300": len(tune_rows), "report_users_V300": len(report_rows),
                  "VHALF_users_in_neither_half": int(len(full_v["VHALF"]) - np.isin(full_v["VHALF"], np.r_[tune_rows, report_rows]).sum()),
                  "submission_users": len(sub_users), "train_labelled_pairs": len(train_all), "train_positive_pairs": len(train_pos),
                  "recent_positive_pairs_17_20": len(recent_pos)}
    R["variant_stats"] = {}
    for v in VARIANTS:
        pp = vpairs[v].groupby("user_id").size()
        R["variant_stats"][v] = {
            "positive_pairs": int(len(vpairs[v])), "users_with_truth": int(nonempty[v].sum()),
            "tune_users": int(len(tune_v[v])), "report_users": int(len(report_v[v])),
            "submission_users_with_truth": int(nonempty[v][sub_rows].sum()),
            "positives_per_user_quantiles": {str(q): float(x) for q, x in pp.quantile([.1, .25, .5, .75, .9]).items()},
            "positives_per_user_mean": float(pp.mean())}

    pop_counts = hybrid.pop_counts(train_pos, pool)     # standard popularity, used for the "mean popularity" statistic

    def maps(ap, v):
        return {"tune": float(ap[tune_v[v]].mean()), "report": float(ap[report_v[v]].mean()),
                "full": float(ap[full_v[v]].mean()), "submission": float(ap[sub_rows].mean())}

    # ================================================================== popularity variants
    sched_val = data.load_schedule("21-30")
    air_val = airing_counts(sched_val, 21, 30, pool)
    popv = pop_vectors(train_pos, recent_pos, pool, air_val)
    pop_top = {n: np.tile(np.argsort(-vec, kind="stable")[:K], (n_u, 1)) for n, vec in popv.items()}
    # tune score of each popularity variant, mean over truth variants
    pop_tune = {n: np.mean([metrics.ap5_matrix(pop_top[n][tune_rows], Tt[v])[mt[v]].mean() for v in VARIANTS]) for n in popv}
    pop_best = max(pop_tune, key=pop_tune.get)
    pop_norm = hybrid.minmax_vec(popv[pop_best])
    R["popularity_variants"] = {"tune_mean_over_variants": {k: float(x) for k, x in pop_tune.items()}, "best": pop_best}
    print("pop variants (TUNE mean over variants):", pop_tune, "-> best", pop_best, flush=True)

    # ================================================================== tuning engine
    def tune_grid(subs, combos):
        out = np.zeros((len(combos), 3))
        for i, wd in enumerate(combos):
            act = [(n, w) for n, w in wd.items() if w > 0]
            if act:
                S = fuse_eps(pop_norm, [subs[n] for n, _ in act], [w for _, w in act])
            else:
                S = np.tile((pop_norm + EPS * pop_norm)[None, :], (len(tune_rows), 1)).astype(np.float32)
            top = hybrid.topk(S, K)
            for j, v in enumerate(VARIANTS):
                out[i, j] = metrics.ap5_matrix(top, Tt[v])[mt[v]].mean()
        return out

    def sub_of(comp):
        return (comp["Sn"][tune_rows], comp["valid"][tune_rows])

    def config_score(name, comp):
        curve = tune_grid({name: sub_of(comp)}, [{name: w} for w in W_GRID])
        return float(curve.max(axis=0).mean()), curve

    comps, sel, cfg_tables = {}, {}, {}

    # ---- H and CBF
    comps["H"] = comp_H(train_pos, users, pool)
    t = time.time(); meta = cbf.load_metadata(); items = cbf.build_items(meta); items = (*items, time.time() - t)
    comps["CBF"] = s3.cbf_component(items[:4], items[4], train_pos, users, pool)
    R["cbf_features"] = items[3]

    # ---- MF grid (config chosen on the pop+MF hybrid TUNE score, mean over truth variants)
    print("MF grid", flush=True)
    rows, best = [], None
    for cfg in mf.GRID + MF_EXTRA:
        c = s3.mf_component(train_all, cfg, users, pool)
        sc, curve = config_score("MF", c)
        pure = tune_grid({"MF": sub_of(c)}, [{"MF": 1.0}])[0]
        rows.append({**cfg, "fit_s": round(c["fit_s"], 2), "score_s": round(c["score_s"], 2), "hybrid_tune_mean": sc,
                     "hybrid_best_w_per_variant": [W_GRID[int(i)] for i in curve.argmax(axis=0)],
                     "pure_tune": {v: float(x) for v, x in zip(VARIANTS, pure)}})
        print("  ", rows[-1], flush=True)
        if best is None or sc > best[0]:
            best = (sc, cfg, c)
    cfg_tables["mf"] = rows
    sel["mf_cfg"], comps["MF"] = best[1], best[2]

    # ---- PPR
    print("PPR", flush=True)
    g = ppr.build_graph(train_pos)
    rows, best = [], None
    for a in ppr.ALPHAS:
        S, valid, info = ppr.run(g, users, pool, a)
        t = time.time(); Sn = hybrid.minmax_rows(S); sec = info["seconds"] + time.time() - t
        c = {"Sn": Sn, "valid": valid, "fit_s": g["build_s"], "score_s": sec, "info": info}
        sc, curve = config_score("PPR", c)
        pure = tune_grid({"PPR": sub_of(c)}, [{"PPR": 1.0}])[0]
        rows.append({**{k: info[k] for k in ("alpha", "iterations_max", "iterations_per_chunk", "converged_all_chunks", "final_max_l1_change")},
                     "seconds": info["seconds"], "hybrid_tune_mean": sc,
                     "hybrid_best_w_per_variant": [W_GRID[int(i)] for i in curve.argmax(axis=0)],
                     "pure_tune": {v: float(x) for v, x in zip(VARIANTS, pure)}})
        print("  ", rows[-1], flush=True)
        if best is None or sc > best[0]:
            best = (sc, a, c)
    cfg_tables["ppr"] = rows
    R["ppr_graph"] = {"users": g["n_users"], "items": g["n_items"], "edges": g["nnz"], "build_s": round(g["build_s"], 2)}
    sel["ppr_alpha"], comps["PPR"] = best[1], best[2]

    # ---- AR
    print("AR", flush=True)
    am = assoc.fit(train_pos, pool)
    rows, best = [], None
    for mode in assoc.MODES:
        S, valid, ts = assoc.score(am, train_pos, users, mode)
        t = time.time(); Sn = hybrid.minmax_rows(S); sec = ts + time.time() - t
        c = {"Sn": Sn, "valid": valid, "fit_s": am["build_s"], "score_s": sec}
        sc, curve = config_score("AR", c)
        pure = tune_grid({"AR": sub_of(c)}, [{"AR": 1.0}])[0]
        rows.append({"mode": mode, "fit_s": round(am["build_s"], 2), "score_s": round(sec, 2), "hybrid_tune_mean": sc,
                     "hybrid_best_w_per_variant": [W_GRID[int(i)] for i in curve.argmax(axis=0)],
                     "pure_tune": {v: float(x) for v, x in zip(VARIANTS, pure)}})
        print("  ", rows[-1], flush=True)
        if best is None or sc > best[0]:
            best = (sc, mode, c)
    cfg_tables["ar"] = rows
    R["ar_rules"] = {k: v for k, v in am.items() if isinstance(v, (int, float))}
    sel["ar_mode"], comps["AR"] = best[1], best[2]
    R["configs"] = cfg_tables
    R["selected"] = {**sel, "pop_best": pop_best, "rule": "config with the best mean over the 3 truth variants of the max-over-w TUNE MAP@5 of pop_best + component"}
    R["coverage_valid_users_V300"] = {n: int(c["valid"][full_v["V300"]].sum()) for n, c in comps.items()}

    # ================================================================== weight grids on TUNE
    print("weight grids", flush=True)
    subs = {n: sub_of(c) for n, c in comps.items()}
    single_curves = {}
    for n in SINGLE_NAMES:
        single_curves[n] = tune_grid({n: subs[n]}, [{n: w} for w in W_GRID])
    pairs = [(a, b) for i, a in enumerate(COMP_NAMES) for b in COMP_NAMES[i + 1:]]
    pair_combos = [(a / 10, b / 10) for a in range(11) for b in range(11 - a)]
    pair_curves = {}
    for (x, y) in pairs:
        pair_curves[(x, y)] = tune_grid({x: subs[x], y: subs[y]}, [{x: wa, y: wb} for wa, wb in pair_combos])
        print("  pair", x, y, "best tune per variant", pair_curves[(x, y)].max(axis=0), flush=True)

    # ================================================================== rows (methods)
    comp_cost = lambda names: (sum(comps[n]["fit_s"] for n in names), sum(comps[n]["score_s"] for n in names))
    rows_out = {}

    def add_row(name, tops, fit_s, score_s, weights=None, kind="pure"):
        m = {}
        for v in VARIANTS:
            ap = metrics.ap5_matrix(tops[v], T[v])
            r = full_v[v]
            m[v] = {**maps(ap, v), "mean_train_popularity": float(pop_counts[tops[v][r]].mean()),
                    "repeat_share_train_positive": float(Tpos[r[:, None], tops[v][r]].mean()),
                    "repeat_share_any_pair": float(Tall[r[:, None], tops[v][r]].mean())}
            m[v]["_ap"] = ap
        rows_out[name] = {"metrics": m, "fit_s": round(fit_s, 2), "score_s": round(score_s, 2), "weights": weights, "kind": kind}

    def top_full(wd):
        act = [(n, w) for n, w in wd.items() if w > 0]
        t = time.time()
        S = fuse_eps(pop_norm, [(comps[n]["Sn"], comps[n]["valid"]) for n, _ in act], [w for _, w in act]) if act else \
            np.tile((pop_norm + EPS * pop_norm)[None, :], (n_u, 1)).astype(np.float32)
        top = hybrid.topk(S, K)
        return top, time.time() - t

    same = lambda top: {v: top for v in VARIANTS}
    # pure popularity variants and H reference (step 2 definition)
    for n in ("P", "P-rec", "P-air"):
        add_row(n, same(pop_top[n]), 0.0, 0.0)
    t = time.time(); hist = baselines.own_history_lists(train_pos, pool); fit_h = time.time() - t
    t = time.time(); preds_h = baselines.recommend_history(train_pos, pool, users, K); sc_h = time.time() - t
    pidx = {int(p): i for i, p in enumerate(pool)}
    add_row("H", same(np.array([[pidx[x] for x in preds_h[u]] for u in users])), fit_h, sc_h)
    # pure personal methods (popularity fallback + tie-break from pop_best)
    for n, label in (("MF", "MF"), ("CBF", "CBF"), ("PPR", "PPR"), ("AR", "AR")):
        top, ts = top_full({n: 1.0})
        f, s_ = comp_cost([n])
        add_row(label, same(top), f, s_ + ts, {n: 1.0})
    # single-component hybrids: per-variant weights and one shared weight
    shared = {}
    for n in SINGLE_NAMES:
        cur = single_curves[n]
        w_v = {v: W_GRID[int(cur[:, j].argmax())] for j, v in enumerate(VARIANTS)}
        tops, ts_tot = {}, 0.0
        for v in VARIANTS:
            tops[v], ts = top_full({n: w_v[v]})
            ts_tot = max(ts_tot, ts)
        f, s_ = comp_cost([n])
        add_row(f"pop+{n}", tops, f, s_ + ts_tot, {v: {n: w_v[v]} for v in VARIANTS}, "hybrid")
        ws = W_GRID[int(cur.mean(axis=1).argmax())]
        top, ts = top_full({n: ws})
        add_row(f"pop+{n} [shared w]", same(top), f, s_ + ts, {n: ws}, "shared")
        shared[f"pop+{n}"] = {n: ws}
    # best two-component blend (three components in total with popularity)
    best_v, tops = {}, {}
    for j, v in enumerate(VARIANTS):
        b = max(((p, i) for p in pairs for i in range(len(pair_combos))), key=lambda pi: pair_curves[pi[0]][pi[1], j])
        best_v[v] = {"pair": b[0], "w": pair_combos[b[1]]}
        tops[v], ts_b = top_full({b[0][0]: pair_combos[b[1]][0], b[0][1]: pair_combos[b[1]][1]})
    used = {n for v in VARIANTS for n in best_v[v]["pair"]}
    f, s_ = comp_cost(sorted(used))
    add_row("pop+blend2", tops, f, s_ + ts_b,
            {v: {best_v[v]["pair"][0]: best_v[v]["w"][0], best_v[v]["pair"][1]: best_v[v]["w"][1]} for v in VARIANTS}, "hybrid")
    bs = max(((p, i) for p in pairs for i in range(len(pair_combos))), key=lambda pi: pair_curves[pi[0]][pi[1]].mean())
    wd = {bs[0][0]: pair_combos[bs[1]][0], bs[0][1]: pair_combos[bs[1]][1]}
    top, ts = top_full(wd)
    f, s_ = comp_cost(list(bs[0]))
    add_row("pop+blend2 [shared w]", same(top), f, s_ + ts, wd, "shared")
    shared["pop+blend2"] = wd

    # ================================================================== ranking and winner
    ranked = [n for n, r in rows_out.items() if r["kind"] != "shared"]
    rk = {v: pd.Series({n: rows_out[n]["metrics"][v]["report"] for n in ranked}).rank(ascending=False, method="min") for v in VARIANTS}
    avg_rank = {n: float(np.mean([rk[v][n] for v in VARIANTS])) for n in ranked}
    mean_report = {n: float(np.mean([rows_out[n]["metrics"][v]["report"] for v in VARIANTS])) for n in ranked}
    winner = min(ranked, key=lambda n: (avg_rank[n], -mean_report[n]))
    ship_name = f"{winner} [shared w]" if f"{winner} [shared w]" in rows_out else winner
    R["ranking"] = {"avg_rank_report": avg_rank, "rank_per_variant": {v: {n: float(x) for n, x in rk[v].items()} for v in VARIANTS},
                    "winner": winner, "shipped_row": ship_name,
                    "shipping_rule": "winner = lowest average rank over the 3 truth variants on REPORT (per-variant-tuned weights); "
                                     "shipped weights = the single weight set (same for all variants) maximising the mean over the 3 variants of TUNE MAP@5"}
    ship_weights = rows_out[ship_name]["weights"]

    # ================================================================== significance (shipped vs P on REPORT)
    rng = np.random.default_rng(SEED)
    sig = {}
    for v in VARIANTS:
        idx = report_v[v]
        d = rows_out[ship_name]["metrics"][v]["_ap"][idx] - rows_out["P"]["metrics"][v]["_ap"][idx]
        bi = rng.integers(0, len(d), size=(2000, len(d)))
        bm = d[bi].mean(axis=1)
        sig[v] = {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}
    R["significance"] = {"strategy": ship_name, "vs": "P", "resamples": 2000, "per_variant": sig}

    # ================================================================== final submissions
    print("final submissions", flush=True)
    SUBDIR.mkdir(parents=True, exist_ok=True)
    train_all_f = data.positives(inter, list(data.PERIODS), 0)
    train_pos_f = data.positives(inter, list(data.PERIODS), THR)
    recent_f = data.positives(recent, ["27-30"], THR)
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    users_f = data.load_submission_users()
    air_f = airing_counts(data.load_schedule("31-42"), 31, 42, pool_f)
    popv_f = pop_vectors(train_pos_f, recent_f, pool_f, air_f)
    pop_norm_f = hybrid.minmax_vec(popv_f[pop_best])
    ctx = {"train_all": train_all_f, "train_pos": train_pos_f, "pool": pool_f, "users": users_f}
    need = {"PPR", "AR"} | set(ship_weights)
    t = time.time()
    comps_f = {n: build_final_comp(n, ctx, sel, items) for n in sorted(need)}
    files = {}

    def final_file(tag, wd):
        parts = [(comps_f[n]["Sn"], comps_f[n]["valid"]) for n, w in wd.items() if w > 0]
        S = fuse_eps(pop_norm_f, parts, [w for w in wd.values() if w > 0])
        top = hybrid.topk(S, K)
        preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / f"submission_{tag}.csv"
        write_submission(preds, users_f, path)
        files[tag] = {**s3.check_file(path, users_f, pool_f), "weights": wd,
                      "invalid_users": {n: int((~comps_f[n]["valid"]).sum()) for n in wd}}
    final_file("PPR", {"PPR": 1.0})
    final_file("AR", {"AR": 1.0})
    ship_wd = dict(ship_weights or {})
    if ship_wd and set(ship_wd) == set(VARIANTS):    # per-variant row (only if a non-shared row won and has no shared twin)
        ship_wd = ship_wd["V300"]
    final_file("CHOSEN", ship_wd)
    R["final"] = {"pool_size": int(len(pool_f)), "pop_variant": pop_best, "chosen_strategy": ship_name, "chosen_weights": ship_wd,
                  "ppr_alpha": sel["ppr_alpha"], "ar_mode": sel["ar_mode"], "mf_cfg": sel["mf_cfg"],
                  "train_positive_pairs_weeks_11_30": len(train_pos_f), "build_seconds": round(time.time() - t, 1), "files": files}
    tm["total_s"] = round(time.time() - t_all, 1)
    R["runtime"] = tm

    # ================================================================== write
    for r in rows_out.values():
        for v in VARIANTS:
            r["metrics"][v].pop("_ap", None)
    R["rows"] = {n: {k: r[k] for k in ("metrics", "fit_s", "score_s", "weights", "kind")} for n, r in rows_out.items()}
    R["single_curves_tune"] = {n: {"w": W_GRID, "map_by_variant": single_curves[n].tolist()} for n in SINGLE_NAMES}
    R["blend2_best_per_variant"] = {v: {"pair": list(best_v[v]["pair"]), "w": list(best_v[v]["w"]),
                                        "tune": float(pair_curves[best_v[v]["pair"]][pair_combos.index(best_v[v]["w"]), j])}
                                    for j, v in enumerate(VARIANTS)}
    R["shared_weights"] = shared
    write_md(R, rows_out, ranked, avg_rank, winner, ship_name, full_v)

    def _d(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    (OUT / "validation_step4.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=_d), encoding="utf-8")
    print(f"done in {tm['total_s']} s; winner {winner}; shipped {ship_name}")


def write_md(R, rows, ranked, avg_rank, winner, ship_name, full_v):
    f4 = lambda x: f"{x:.4f}"
    V = VARIANTS
    md = ["# Validation, step 4", "",
          "Training interactions as in step 3 (TRAIN weeks 11-20, positives >= 300 s). Only the validation truth changes. "
          "TUNE/REPORT halves are the step-3 halves of the V300 users; for V900/VHALF only users with non-empty truth in that variant count "
          "(submission column: all 1,260 users, empty truth = 0). Fusion adds a 1e-6 popularity tie-break; users for whom a component is invalid get popularity.", "",
          "## 0. Truth variants", "", "| variant | positive pairs | users with truth | TUNE users | REPORT users | positives/user p10 / p25 / p50 / p75 / p90 | mean | submission users with truth |", "|---|---|---|---|---|---|---|---|"]
    for v in V:
        s = R["variant_stats"][v]
        q = s["positives_per_user_quantiles"]
        md.append(f"| {v} | {s['positive_pairs']:,} | {s['users_with_truth']:,} | {s['tune_users']:,} | {s['report_users']:,} | "
                  f"{q['0.1']:g} / {q['0.25']:g} / {q['0.5']:g} / {q['0.75']:g} / {q['0.9']:g} | {s['positives_per_user_mean']:.1f} | {s['submission_users_with_truth']} |")
    md += ["", "P (popularity) MAP@5: " + "; ".join(f"{v}: REPORT {f4(rows['P']['metrics'][v]['report'])}, full {f4(rows['P']['metrics'][v]['full'])}, submission {f4(rows['P']['metrics'][v]['submission'])}" for v in V), "",
           f"Popularity variants (mean TUNE MAP@5 over the 3 variants): {R['popularity_variants']['tune_mean_over_variants']}; best = **{R['popularity_variants']['best']}** (base of all hybrids).", ""]

    def table(metric, title):
        out = [f"## {title}", "", "| method | " + " | ".join(V) + " | mean | avg rank |", "|---|" + "---|" * (len(V) + 2)]
        for n, r in rows.items():
            vals = [r["metrics"][v][metric] for v in V]
            ar = f"{avg_rank[n]:.2f}" if n in avg_rank and metric == "report" else ""
            out.append(f"| {n} | " + " | ".join(f4(x) for x in vals) + f" | {f4(np.mean(vals))} | {ar} |")
        return out + [""]
    md += table("report", "1. MAP@5 on the REPORT half") + table("full", "2. MAP@5 on all validation users with truth") + \
          table("submission", "3. MAP@5 on the 1,260 submission users (partly overlaps TUNE)")
    md += [f"Winner by average REPORT rank: **{winner}** (avg rank {avg_rank[winner]:.2f}). Shipped row: **{ship_name}**. Rule: {R['ranking']['shipping_rule']}.", ""]
    md += ["## 4. Hyperparameters", "", "### MF grid (pure-MF TUNE MAP per variant; selection = mean over variants of best pop+MF TUNE MAP)", "",
           "| factors | reg | alpha | fit s | score s | pop+MF TUNE mean | best w per variant | pure MF TUNE V300 / V900 / VHALF |", "|---|---|---|---|---|---|---|---|"]
    for g in R["configs"]["mf"]:
        md.append(f"| {g['factors']} | {g['reg']} | {g['alpha']} | {g['fit_s']} | {g['score_s']} | {f4(g['hybrid_tune_mean'])} | {g['hybrid_best_w_per_variant']} | "
                  + " / ".join(f4(g['pure_tune'][v]) for v in V) + " |")
    md += ["", f"Chosen MF: {R['selected']['mf_cfg']}", "", "### PPR (alpha = link-following probability; power iteration, tol 1e-6, max 100)", "",
           "| alpha | max iterations | converged (all chunks) | final max L1 change | seconds | pop+PPR TUNE mean | best w per variant | pure PPR TUNE V300 / V900 / VHALF |", "|---|---|---|---|---|---|---|---|"]
    for g in R["configs"]["ppr"]:
        md.append(f"| {g['alpha']} | {g['iterations_max']} | {g['converged_all_chunks']} | {g['final_max_l1_change']:.2e} | {g['seconds']} | {f4(g['hybrid_tune_mean'])} | "
                  f"{g['hybrid_best_w_per_variant']} | " + " / ".join(f4(g['pure_tune'][v]) for v in V) + " |")
    pg = R["ppr_graph"]
    md += ["", f"Chosen alpha: {R['selected']['ppr_alpha']}. Graph: {pg['users']:,} users, {pg['items']:,} programmes, {pg['edges']:,} edges.", "",
           "### Association rules (min support 20 users)", "",
           "| mode | fit s | score s | pop+AR TUNE mean | best w per variant | pure AR TUNE V300 / V900 / VHALF |", "|---|---|---|---|---|---|"]
    for g in R["configs"]["ar"]:
        md.append(f"| {g['mode']} | {g['fit_s']} | {g['score_s']} | {f4(g['hybrid_tune_mean'])} | {g['hybrid_best_w_per_variant']} | " + " / ".join(f4(g['pure_tune'][v]) for v in V) + " |")
    ar = R["ar_rules"]
    md += ["", f"Rules passing min support: {ar['rules_all']:,} (A != B); with a pool consequent: {ar['rules_pool_consequent']:,}; of those with lift > 1: {ar['rules_lift_gt1_pool']:,}; "
               f"antecedent programmes with at least one rule: {ar['antecedents_with_rule']:,} of {ar['items_in_train']:,} TRAIN-positive programmes. Chosen mode: {R['selected']['ar_mode']}.", "",
           "### CBF (step-3 definition, no grid)", f"Feature blocks: {R['cbf_features']['block_dims']}.", "",
           "## 5. Hybrid weights (weight of each component; popularity gets the remainder)", "",
           "| row | V300 | V900 | VHALF | shared weight set (shipped rule) |", "|---|---|---|---|---|"]
    for n, r in rows.items():
        if r["kind"] == "hybrid":
            md.append(f"| {n} | " + " | ".join(str(r["weights"][v]) for v in V) + f" | {R['shared_weights'][n]} |")
    md += ["", "Best two-component blend per variant on TUNE: " + "; ".join(f"{v}: {b['pair']} w={b['w']} (TUNE {f4(b['tune'])})" for v, b in R["blend2_best_per_variant"].items()), "",
           "## 6. Cost and recommendation statistics (users with truth; popularity = distinct TRAIN positive users)", "",
           "Cells: mean TRAIN popularity of recommended items / repeat share (item already a TRAIN positive for the user). Times are for ~4.8k users x 6,361 items; hybrids include components.", "",
           "| row | fit s | score s | " + " | ".join(V) + " |", "|---|---|---|" + "---|" * len(V)]
    for n, r in rows.items():
        md.append(f"| {n} | {r['fit_s']} | {r['score_s']} | " + " | ".join(f"{r['metrics'][v]['mean_train_popularity']:.0f} / {r['metrics'][v]['repeat_share_train_positive']:.2f}" for v in V) + " |")
    s = R["significance"]
    md += ["", f"## 7. Significance: {s['strategy']} vs P on REPORT (paired bootstrap, {s['resamples']} resamples, seed {SEED})", "",
           "| variant | users | mean AP@5 difference | 95% CI |", "|---|---|---|---|"]
    for v in V:
        d = s["per_variant"][v]
        md.append(f"| {v} | {d['n_users']} | {d['mean_diff']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    fin = R["final"]
    md += ["", "## 8. Final submissions (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning)", "",
           f"Popularity base: {fin['pop_variant']} (P-rec = weeks 27-30; P-air uses the 31-42 schedule only for airing counts); pool {fin['pool_size']:,}; "
           f"chosen strategy {fin['chosen_strategy']} with weights {fin['chosen_weights']}.", ""]
    for tag, f in fin["files"].items():
        md.append(f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, "
                  f"ids in final pool {f['all_ids_in_final_pool']}; weights {f['weights']}; users without component data (popularity fallback): {f['invalid_users']}")
    md += ["", f"Runtime: {R['runtime']['total_s']} s"]
    (OUT / "validation_step4.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Step 3: MF (ALS), CBF, hybrids on the step-2 validation protocol; then final submissions.
Usage: python src/run_step3.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # implicit recommends single-threaded BLAS

import json
import time

import numpy as np
import pandas as pd

import baselines
import cbf
import data
import hybrid
import metrics
import mf
from run_validation import write_submission, users_to_truth

SEED, THR, K = 42, 300, 5
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
W_GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
np.random.seed(SEED)


# ------------------------------------------------------------------ components
def mf_component(train_all, cfg, users, pool):
    fm = mf.fit_als(train_all, cfg, SEED)
    S, valid, item_ok, t_score = mf.score(fm, users, pool)
    t = time.time()
    Sn = hybrid.minmax_rows(S)
    return {"Sn": Sn, "valid": valid, "item_ok": item_ok, "fit_s": fm["fit_s"], "score_s": t_score + time.time() - t,
            "train_users": fm["n_users"], "train_items": fm["n_items"], "train_nnz": fm["nnz"]}


def cbf_component(items, fit_s, train_pos, users, pool):
    ids, X, _, _ = items
    S, valid, t_score = cbf.score(ids, X, train_pos, pool, users)
    t = time.time()
    Sn = hybrid.minmax_rows(S)
    return {"Sn": Sn, "valid": valid, "fit_s": fit_s, "score_s": t_score + time.time() - t}


def fused_top(pop_norm, comps, spec, rows=None):
    """Top-5 pool indices for `rows` of the component matrices under spec {'w_mf','w_cbf'}."""
    parts, ws = [], []
    for key, w in (("mf", spec.get("w_mf", 0.0)), ("cbf", spec.get("w_cbf", 0.0))):
        if w > 0:
            c = comps[key]
            parts.append((c["Sn"] if rows is None else c["Sn"][rows], c["valid"] if rows is None else c["valid"][rows]))
            ws.append(w)
    n = len(comps["mf"]["valid"]) if rows is None else len(rows)
    if not parts:
        S = np.broadcast_to(pop_norm[None, :], (n, len(pop_norm))).astype(np.float32)
    else:
        S = hybrid.fuse(pop_norm, parts, ws)
    return hybrid.topk(S, K)


def n_fallback(comps, spec, rows):
    bad = np.zeros(len(rows), dtype=bool)
    for key, w in (("mf", spec.get("w_mf", 0.0)), ("cbf", spec.get("w_cbf", 0.0))):
        if w > 0:
            bad |= ~comps[key]["valid"][rows]
    return int(bad.sum())


def check_file(path, users, pool):
    back = pd.read_csv(path, dtype=str)
    tmpl = pd.read_csv(data.DATA / data.SUB_FILE, dtype=str)
    pool_set = set(int(x) for x in pool)
    lists = back["tv_show_id"].map(lambda s: [int(x) for x in s.split(" ")])
    return {"file": str(path.relative_to(data.ROOT)), "header_ok": list(back.columns) == list(tmpl.columns),
            "rows": int(len(back)), "user_order_ok": bool((back["user_id"] == tmpl["user_id"]).all()),
            "five_distinct_ids": bool(lists.map(lambda l: len(l) == K and len(set(l)) == K).all()),
            "all_ids_in_final_pool": bool(lists.map(lambda l: set(l) <= pool_set).all())}


def main():
    t_all = time.time()
    R, tm = {"seed": SEED, "threshold_sec": THR}, {}

    # ------------------------------------------------------------------ data (step-2 protocol)
    inter, _ = data.build_interactions()
    train_all = data.positives(inter, ["11-20"], 0)       # all labelled TRAIN pairs (overlap >= 1 s)
    train_pos = data.positives(inter, ["11-20"], THR)
    val_pos = data.positives(inter, ["21-30"], THR)
    pool = data.pool_ids(data.load_schedule("21-30"), 21, 30)
    sub_users = data.load_submission_users()
    truth = users_to_truth(val_pos)
    eval_a = sorted(truth)
    users = sorted(set(eval_a) | set(sub_users))
    uix = pd.Index(users)
    eval_rows, sub_rows = uix.get_indexer(eval_a), uix.get_indexer(sub_users)
    pix = pd.Index(pool)
    R["sizes"] = {"pool": len(pool), "users_union": len(users), "eval_users": len(eval_a), "submission_users": len(sub_users),
                  "train_labelled_pairs": len(train_all), "train_positive_pairs": len(train_pos)}

    def bool_matrix(df):
        r = uix.get_indexer(df["user_id"].astype(np.uint64).astype(object).map(int).values)
        c = pix.get_indexer(df["tv_show_id"].values)
        ok = (r >= 0) & (c >= 0)
        M = np.zeros((len(users), len(pool)), dtype=bool)
        M[r[ok], c[ok]] = True
        return M
    T, Tpos, Tall = bool_matrix(val_pos), bool_matrix(train_pos), bool_matrix(train_all)

    perm = np.random.default_rng(SEED).permutation(len(eval_rows))
    half = len(eval_rows) // 2
    tune_rows, report_rows = np.sort(eval_rows[perm[:half]]), np.sort(eval_rows[perm[half:]])
    R["split"] = {"tune_users": len(tune_rows), "report_users": len(report_rows),
                  "submission_users_in_tune": int(np.isin(sub_rows, tune_rows).sum()),
                  "submission_users_in_report": int(np.isin(sub_rows, report_rows).sum())}

    def maps(ap):
        return {"tune": float(ap[tune_rows].mean()), "report": float(ap[report_rows].mean()),
                "full": float(ap[eval_rows].mean()), "submission": float(ap[sub_rows].mean())}

    def ap_of(top):
        return metrics.ap5_matrix(top, T)

    pop = hybrid.pop_counts(train_pos, pool)
    pop_norm = hybrid.minmax_vec(pop)

    def rec_stats(top):
        rec = top[eval_rows]
        r = eval_rows[:, None]
        return {"mean_train_popularity": float(pop[rec].mean()),
                "repeat_share_train_positive": float(Tpos[r, rec].mean()),
                "repeat_share_any_train_pair": float(Tall[r, rec].mean())}

    # ------------------------------------------------------------------ references P, H
    methods = {}   # name -> dict(top, maps, stats, fit_s, score_s, fallback)
    pidx = {int(p): i for i, p in enumerate(pool)}
    lists_to_top = lambda preds: np.array([[pidx[x] for x in preds[u]] for u in users])
    t = time.time(); rank = baselines.pop_ranking(train_pos, pool); fit_p = time.time() - t
    t = time.time(); preds_p = baselines.recommend_popularity(train_pos, pool, users, K); sc_p = time.time() - t
    t = time.time(); hist = baselines.own_history_lists(train_pos, pool); fit_h = time.time() - t
    t = time.time(); preds_h = baselines.recommend_history(train_pos, pool, users, K); sc_h = time.time() - t
    top_p, top_h = lists_to_top(preds_p), lists_to_top(preds_h)
    # vectorised AP == loop AP
    loop = np.array([metrics.ap_at_k(preds_h[u], truth.get(u, set()), K) for u in users])
    assert np.abs(loop - ap_of(top_h)).max() < 1e-12
    h_fill = lambda rows: int(sum(len(hist.get(users[i], [])) < K for i in rows))
    methods["P"] = dict(top=top_p, fit_s=fit_p, score_s=sc_p, fallback_eval=0, fallback_sub=0, note="is the popularity list")
    methods["H"] = dict(top=top_h, fit_s=fit_h, score_s=sc_h, fallback_eval=h_fill(eval_rows), fallback_sub=h_fill(sub_rows),
                        note="lists with < 5 own in-pool TRAIN positives (rest filled by P)")

    # ------------------------------------------------------------------ MF grid
    print("MF grid", flush=True)
    grid, best_mf = [], None
    for cfg in mf.GRID:
        c = mf_component(train_all, cfg, users, pool)
        spec_mf = {"w_mf": 1.0}
        comps = {"mf": c}
        top = fused_top(pop_norm, comps, spec_mf)
        ap = ap_of(top)
        m = maps(ap)
        hy = [(w, float(metrics.ap5_matrix(fused_top(pop_norm, comps, {"w_mf": w}, tune_rows), T[tune_rows]).mean()))
              for w in W_GRID]
        bw = max(hy, key=lambda x: x[1])
        row = {**cfg, "fit_s": round(c["fit_s"], 2), "score_s": round(c["score_s"], 2),
               "map_tune": m["tune"], "map_report": m["report"], "map_full": m["full"], "map_submission": m["submission"],
               "hybrid_best_w_on_tune": bw[0], "hybrid_tune_map": bw[1]}
        grid.append(row)
        print("  ", row, flush=True)
        if best_mf is None or m["tune"] > best_mf["tune"]:
            best_mf = {"cfg": cfg, "tune": m["tune"], "comp": c}
    R["mf_grid"] = grid
    R["mf_best_cfg"] = best_mf["cfg"]
    comps = {"mf": best_mf["comp"]}
    R["mf_best_coverage"] = {"pool_items_with_factors": int(comps["mf"]["item_ok"].sum()),
                             "users_valid_eval": int(comps["mf"]["valid"][eval_rows].sum()),
                             "users_valid_submission": int(comps["mf"]["valid"][sub_rows].sum())}

    # ------------------------------------------------------------------ CBF
    print("CBF", flush=True)
    t = time.time()
    meta = cbf.load_metadata()
    items = cbf.build_items(meta)
    cbf_fit = time.time() - t
    ids, X, noncat, info = items
    in_pool = pd.Index(ids).get_indexer(pool) >= 0
    info["share_labelled_with_non_category_feature_all"] = float(noncat.mean())
    info["share_pool_with_non_category_feature"] = float(noncat[pd.Index(ids).get_indexer(pool)].mean())
    info["pool_items_with_metadata_row"] = int(in_pool.sum())
    R["cbf_features"] = {k: (v if not isinstance(v, dict) else v) for k, v in info.items()}
    comps["cbf"] = cbf_component(items, cbf_fit, train_pos, users, pool)
    R["cbf_coverage"] = {"users_valid_eval": int(comps["cbf"]["valid"][eval_rows].sum()),
                         "users_valid_submission": int(comps["cbf"]["valid"][sub_rows].sum())}

    # ------------------------------------------------------------------ hybrids: tune on TUNE
    print("hybrid grids", flush=True)
    Tt = T[tune_rows]
    tune_map = lambda spec: float(metrics.ap5_matrix(fused_top(pop_norm, comps, spec, tune_rows), Tt).mean())
    g_mf = [(w, tune_map({"w_mf": w})) for w in W_GRID]
    g_cbf = [(w, tune_map({"w_cbf": w})) for w in W_GRID]
    g_3 = [((a / 10, b / 10), tune_map({"w_mf": a / 10, "w_cbf": b / 10})) for a in range(11) for b in range(11 - a)]
    best = lambda g: max(g, key=lambda x: x[1])
    w_mf, w_cbf, (w3_mf, w3_cbf) = best(g_mf)[0], best(g_cbf)[0], best(g_3)[0]
    R["hybrid_tuning_on_tune"] = {"pop_mf": g_mf, "pop_cbf": g_cbf, "three_way_best5": sorted(g_3, key=lambda x: -x[1])[:5],
                                  "chosen": {"w_mf": w_mf, "w_cbf": w_cbf, "three_way_w_mf": w3_mf, "three_way_w_cbf": w3_cbf}}

    specs = {
        "MF": {"w_mf": 1.0}, "CBF": {"w_cbf": 1.0},
        "Hybrid pop+MF": {"w_mf": w_mf}, "Hybrid pop+CBF": {"w_cbf": w_cbf},
        "Hybrid pop+MF+CBF": {"w_mf": w3_mf, "w_cbf": w3_cbf},
    }
    for name, sp in specs.items():
        t = time.time()
        top = fused_top(pop_norm, comps, sp)
        fuse_s = time.time() - t
        parts = [k for k, w in (("mf", sp.get("w_mf", 0)), ("cbf", sp.get("w_cbf", 0))) if w > 0]
        methods[name] = dict(
            top=top, fit_s=sum(comps[k]["fit_s"] for k in parts),
            score_s=sum(comps[k]["score_s"] for k in parts) + fuse_s,
            fallback_eval=n_fallback(comps, sp, eval_rows), fallback_sub=n_fallback(comps, sp, sub_rows),
            note="lists where a used component is invalid for the user (its weight goes to popularity)", spec=sp)

    # ------------------------------------------------------------------ report
    table = {}
    for name, m in methods.items():
        ap = ap_of(m["top"])
        table[name] = {**maps(ap), **rec_stats(m["top"]), "fit_s": round(m["fit_s"], 2), "score_s": round(m["score_s"], 2),
                       "lists_needing_pop_fill_eval": m["fallback_eval"], "lists_needing_pop_fill_submission": m["fallback_sub"],
                       "fill_definition": m["note"], "spec": m.get("spec")}
    R["table"] = table
    ranking = sorted(table, key=lambda n: -table[n]["report"])
    best_name = ranking[0]
    best_hyb = max([n for n in table if n.startswith("Hybrid")], key=lambda n: table[n]["report"])
    R["best_by_report"] = best_name
    R["best_hybrid_by_report"] = best_hyb
    check_p2 = {"P_full_matches_step2": abs(table["P"]["full"] - 0.6358) < 5e-5, "H_full_matches_step2": abs(table["H"]["full"] - 0.5760) < 5e-5}
    R["step2_reproduced"] = check_p2

    # ------------------------------------------------------------------ final submissions
    print("final submissions", flush=True)
    SUBDIR.mkdir(parents=True, exist_ok=True)
    train_all_f = data.positives(inter, list(data.PERIODS), 0)
    train_pos_f = data.positives(inter, list(data.PERIODS), THR)
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    users_f = data.load_submission_users()
    pop_f = hybrid.pop_counts(train_pos_f, pool_f)
    pop_norm_f = hybrid.minmax_vec(pop_f)
    t = time.time()
    cmf = mf_component(train_all_f, best_mf["cfg"], users_f, pool_f)
    ccb = cbf_component(items, cbf_fit, train_pos_f, users_f, pool_f)
    comps_f = {"mf": cmf, "cbf": ccb}
    sub_specs = {"P": None, "MF": specs["MF"], "CBF": specs["CBF"],
                 "HYBRID": specs[best_hyb]}
    files = {}
    for tag, sp in sub_specs.items():
        if sp is None:
            preds = baselines.recommend_popularity(train_pos_f, pool_f, users_f, K)
        else:
            top = fused_top(pop_norm_f, comps_f, sp)
            preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
        path = SUBDIR / f"submission_{tag}.csv"
        write_submission(preds, users_f, path)
        files[tag] = {**check_file(path, users_f, pool_f), "spec": sp, "hybrid_name": best_hyb if tag == "HYBRID" else None}
    R["final"] = {"pool_size": int(len(pool_f)), "train_positive_pairs_weeks_11_30": len(train_pos_f),
                  "train_labelled_pairs_weeks_11_30": len(train_all_f),
                  "mf_pool_items_with_factors": int(cmf["item_ok"].sum()),
                  "pool_items_with_popularity_gt0": int((pop_f > 0).sum()),
                  "mf_users_valid": int(cmf["valid"].sum()), "cbf_users_valid": int(ccb["valid"].sum()),
                  "submission_users": len(users_f), "mf_cfg": best_mf["cfg"], "build_seconds": round(time.time() - t, 1),
                  "files": files}
    tm["total_s"] = round(time.time() - t_all, 1)
    R["runtime"] = tm

    # ------------------------------------------------------------------ write md / json
    f4 = lambda x: f"{x:.4f}"
    md = ["# Validation, step 3: MF, CBF, hybrids", "",
          f"Protocol of step 2 (TRAIN weeks 11-20, truth weeks 21-30, threshold {THR} s, pool {len(pool):,}, repeats allowed, seed {SEED}). "
          f"Validation users with truth: {len(eval_a):,} -> TUNE {len(tune_rows):,} / REPORT {len(report_rows):,} (random 50/50, seed {SEED}). "
          f"Submission users: {len(sub_users):,} ({R['split']['submission_users_in_tune']} in TUNE, {R['split']['submission_users_in_report']} in REPORT, rest have empty truth), "
          "so the submission column partly overlaps the tuning half.", "",
          "## MAP@5", "", "| method | REPORT half | full validation | 1,260 submission users | TUNE half |", "|---|---|---|---|---|"]
    names = {"P": "P popularity", "H": "H own-history (diagnostic)", "MF": "MF (ALS)", "CBF": "CBF"}
    for n, r in table.items():
        label = names.get(n, n)
        sp = r["spec"]
        if n.startswith("Hybrid"):
            label += " (" + ", ".join(f"w_{k}={v}" for k, v in sp.items()) + ")"
        if n == "MF":
            label += f" f={best_mf['cfg']['factors']} reg={best_mf['cfg']['reg']} alpha={best_mf['cfg']['alpha']}"
        md.append(f"| {label} | {f4(r['report'])} | {f4(r['full'])} | {f4(r['submission'])} | {f4(r['tune'])} |")
    md += ["", f"Best by REPORT half: **{best_name}** ({f4(table[best_name]['report'])}). Best hybrid by REPORT: {best_hyb}. "
               f"Step-2 reproduction check: {check_p2}", "",
           "## MF grid (15 iterations, pure MF with popularity for users without TRAIN history; config chosen by TUNE MAP@5)", "",
           "| factors | reg | alpha | fit s | score s | TUNE | REPORT | full | submission | best hybrid w (TUNE) | hybrid TUNE |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in grid:
        md.append(f"| {g['factors']} | {g['reg']} | {g['alpha']} | {g['fit_s']} | {g['score_s']} | {f4(g['map_tune'])} | {f4(g['map_report'])} | "
                  f"{f4(g['map_full'])} | {f4(g['map_submission'])} | {g['hybrid_best_w_on_tune']} | {f4(g['hybrid_tune_map'])} |")
    md += ["", f"Chosen MF config: {best_mf['cfg']}. ALS trained on {comps['mf']['train_users']:,} users x {comps['mf']['train_items']:,} items, "
               f"{comps['mf']['train_nnz']:,} observed pairs; pool items with factors: {R['mf_best_coverage']['pool_items_with_factors']:,} of {len(pool):,}.", "",
           "## Hybrid weights chosen on TUNE", "",
           f"- pop+MF: w_mf = {w_mf} (TUNE MAP@5 {f4(best(g_mf)[1])}); curve: " + ", ".join(f"{w}:{f4(v)}" for w, v in g_mf),
           f"- pop+CBF: w_cbf = {w_cbf} (TUNE {f4(best(g_cbf)[1])}); curve: " + ", ".join(f"{w}:{f4(v)}" for w, v in g_cbf),
           f"- pop+MF+CBF: w_mf = {w3_mf}, w_cbf = {w3_cbf}, w_pop = {round(1 - w3_mf - w3_cbf, 1)} (TUNE {f4(best(g_3)[1])})", "",
           "## CBF features", "",
           f"- items {info['n_items']:,}; block dims {info['block_dims']}",
           f"- metadata coverage: share of labelled programmes with at least one non-category feature = {info['share_labelled_with_non_category_feature_all']:.4f} "
           f"(pool: {info['share_pool_with_non_category_feature']:.4f}); with genre {info['share_with_genre']:.4f}, people {info['share_with_people']:.4f}, decade {info['share_with_decade']:.4f}",
           f"- users with a non-empty CBF profile: eval {R['cbf_coverage']['users_valid_eval']:,} / {len(eval_a):,}, submission {R['cbf_coverage']['users_valid_submission']:,} / {len(sub_users):,}", "",
           "## Cost and recommendation statistics", "",
           "Fit/score seconds are for the 4.8k validation users x 6,361 items on this machine; hybrids include their components. "
           "Statistics are over the validation users with truth: mean TRAIN popularity (distinct TRAIN positive users) of recommended items; "
           "repeat share = share of recommended items already a TRAIN positive (>= 300 s) / having any labelled TRAIN pair for that user.", "",
           "| method | fit s | score s | mean train popularity | repeat share (TRAIN positive) | repeat share (any pair) | lists needing pop fill (eval) | (submission) |", "|---|---|---|---|---|---|---|---|"]
    for n, r in table.items():
        md.append(f"| {names.get(n, n)} | {r['fit_s']} | {r['score_s']} | {r['mean_train_popularity']:.1f} | {r['repeat_share_train_positive']:.4f} | "
                  f"{r['repeat_share_any_train_pair']:.4f} | {r['lists_needing_pop_fill_eval']} | {r['lists_needing_pop_fill_submission']} |")
    md += ["", "Fill definition: " + "; ".join(f"{n}: {r['fill_definition']}" for n, r in table.items() if n in ("P", "H", "MF", "Hybrid pop+MF")), "",
           "## Final submissions (TRAIN weeks 11-30, pool = weeks 31-42, no re-tuning)", "",
           f"- final pool {len(pool_f):,} programmes; MF factors exist for {R['final']['mf_pool_items_with_factors']:,} of them; popularity > 0 for {R['final']['pool_items_with_popularity_gt0']:,}",
           f"- TRAIN positive pairs {len(train_pos_f):,}; MF valid users {R['final']['mf_users_valid']} / {len(users_f)}; CBF valid users {R['final']['cbf_users_valid']} / {len(users_f)}", ""]
    for tag, f in files.items():
        md.append(f"- {f['file']}: header {f['header_ok']}, rows {f['rows']}, user order {f['user_order_ok']}, 5 distinct ids {f['five_distinct_ids']}, "
                  f"ids in final pool {f['all_ids_in_final_pool']}; spec {f['spec']}" + (f" ({f['hybrid_name']})" if f['hybrid_name'] else ""))
    md += ["", f"Runtime: {tm['total_s']} s"]
    (OUT / "validation_step3.md").write_text("\n".join(md), encoding="utf-8")

    def _d(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return str(o)
    (OUT / "validation_step3.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=_d), encoding="utf-8")
    print(f"done in {tm['total_s']} s")


if __name__ == "__main__":
    main()

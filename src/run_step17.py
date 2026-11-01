"""Step 17: last improvement round on the inference-window lever, evaluated on the step-15 validation analogue.

Baseline = per-week reranker fitted on weeks 16-20 features/candidates (labels VTOP5 of weeks 21-30), inference with weeks 11-20 (step-15 setting).
Levers: (1) H2/EW decay c at inference, (2) rank fusion of the weeks 16-20 and 11-20 inference scores (weight w on 11-20), (3) candidate depth K per component.
Every setting is chosen on the fold-internal TRAINING users only; a lever is kept for the combination only if BOTH folds choose the same non-baseline value.
Ship rule (pre-declared): the combination beats the baseline on VTOP5 in both folds and the pooled paired-bootstrap CI excludes zero.
Usage: python src/run_step17.py
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

SEED, K5 = 42, 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
C_GRID, W_GRID, K_GRID = (0.3, 0.5, 0.7, 1.0), (0.5, 0.7, 0.85, 1.0), (50, 80, 120)
BASE = {"c": 0.5, "w": 1.0, "K": 50}
FOLDS = ("A", "B")


def boot(d, rng):
    bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    return {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}


def top_from_scores(sc, ri, ci, rows):
    """Top-5 pool columns per user in `rows` (sorted ascending) from precomputed pair scores."""
    m = np.isin(ri, rows)
    r, c, s = ri[m], ci[m], sc[m]
    o = np.lexsort((-s, r))
    r, c = r[o], c[o]
    first = np.r_[0, np.flatnonzero(np.diff(r)) + 1]
    assert len(first) == len(rows) and (np.diff(np.r_[first, len(r)]) >= 5).all()
    return np.stack([c[first + j] for j in range(5)], axis=1)


def rank_norm(sc, ri):
    """Within-user rank normalisation in (0, 1]: best candidate = 1, worst = 1/n."""
    d = pd.DataFrame({"r": ri, "s": sc})
    rk = d.groupby("r")["s"].rank(ascending=False, method="first").values
    n = d.groupby("r")["s"].transform("size").values
    return ((n - rk + 1) / n).astype(np.float64)


def fuse(sa, ria, cia, sb, rib, cib, w, n_p):
    A = pd.Series(rank_norm(sa, ria), index=ria.astype(np.int64) * n_p + cia)
    B = pd.Series(rank_norm(sb, rib), index=rib.astype(np.int64) * n_p + cib)
    idx = A.index.union(B.index)
    f = w * B.reindex(idx).fillna(0).values + (1 - w) * A.reindex(idx).fillna(0).values
    return f, (idx.values // n_p).astype(np.int64), (idx.values % n_p).astype(np.int64)


def main():
    T0 = time.time()
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    J15 = json.loads((OUT / "validation_step15.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_m = J7["ew"]["chosen"]["m"]
    R = {"seed": SEED, "primary": "VTOP5", "grids": {"c": C_GRID, "w": W_GRID, "K": K_GRID}, "baseline": BASE}

    # ================================================================== setup (as step 15)
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
    ctx5 = window.restrict(ctx10, 16, 20, uc_pairs)
    uix, pix = ctx10["uix"], ctx10["pix"]
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    ne = {v: T[v].any(axis=1) for v in VARS}
    T5, ne5 = T["VTOP5"], ne["VTOP5"]
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    folds = {"A": {"train": tune_rows[ne5[tune_rows]], "eval": report_rows}, "B": {"train": report_rows[ne5[report_rows]], "eval": tune_rows}}

    def split(rows):
        p = np.random.default_rng(SEED).permutation(rows)
        n_va = int(round(0.2 * len(p)))
        return np.sort(p[n_va:]), np.sort(p[:n_va])

    def ap(top, rows, v="VTOP5"):
        return metrics.ap5_matrix(top, T[v][rows])[ne[v][rows]]

    # ---- stores: PPR/MF/AR do not depend on c or K; H2 and EW are rebuilt per decay c
    base10, _ = s7.build_store(ctx10, ["PPR", "MF", "AR"], cfg)
    base5, _ = s7.build_store(ctx5, ["PPR", "MF", "AR"], cfg)
    stores = {}

    def get_store(name, ctx, base, c):
        if (name, c) not in stores:
            st = dict(base)
            st["H2"] = s7.build_store(ctx, ["H2"], {**cfg, "h2_c": c})[0]["H2"]
            st["EW"] = s7.build_store(ctx, ["EW"], cfg, (ew_m, c))[0]["EW"]
            stores[(name, c)] = st
        return stores[(name, c)]

    def feats(ctx, store, K, wf):
        cand = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER, topn=K)
        ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
        print(f"  features K={K}: {len(ri):,} pairs ({time.time() - T0:.0f} s)", flush=True)
        return ri, ci, rerank2.per_week(X, wf, 10)

    # ---- baseline models (K=50, c=0.5) and scores of both windows
    F5 = {K: None for K in K_GRID}
    F10 = {}
    models = {}

    def fit_models(K):
        ri5, ci5, X5 = feats(ctx5, get_store("5", ctx5, base5, 0.5), K, 5)
        y5 = T5[ri5, ci5].astype(int)
        F5[K] = (ri5, ci5, X5)
        for f in FOLDS:
            tr, va = split(folds[f]["train"])
            models[(f, K)] = rerank2.train(X5, y5, ri5, tr, va, rerank2.BASE_PARAMS)

    fit_models(50)
    res = {f: {"baseline": {}} for f in FOLDS}
    sc10, ct = {}, {}                                     # scores of the baseline models on (ri, ci, X) of a setting, keyed by setting
    cols = list(F5[50][2].columns)

    def infer_scores(f, K, c, win):
        """Pair scores of the K-model of fold f on window `win` ('10' = weeks 11-20, '5' = weeks 16-20) with decay c."""
        key = (win, K, c)
        if key not in ct:
            if win == "10":
                ct[key] = feats(ctx10, get_store("10", ctx10, base10, c), K, 10)
            else:
                ct[key] = F5[K] if (K in F5 and F5[K] is not None and c == 0.5) else feats(ctx5, get_store("5", ctx5, base5, c), K, 5)
        ri, ci, X = ct[key]
        return models[(f, K)].predict(X[cols]), ri, ci

    def evaluate(f, K, c, w):
        """VTOP5 per-user arrays on training and evaluation users for the setting (K, c, w); also the robustness truths on evaluation users."""
        sb, rib, cib = infer_scores(f, K, c, "10")
        if w < 1.0:
            sa, ria, cia = infer_scores(f, K, c, "5")
            s, ri, ci = fuse(sa, ria, cia, sb, rib, cib, w, n_p)
        else:
            s, ri, ci = sb, rib, cib
        out = {}
        for part in ("train", "eval"):
            rows = folds[f][part]
            top = top_from_scores(s, ri, ci, rows)
            out[part] = {v: ap(top, rows, v) for v in (VARS if part == "eval" else ("VTOP5",))}
        return out

    def summ(o):
        return {"train_VTOP5": float(o["train"]["VTOP5"].mean()), **{"eval_" + v: float(a.mean()) for v, a in o["eval"].items()}}

    # ---- baseline sanity: must equal the step-15 analogue (b)
    base_eval = {f: evaluate(f, 50, 0.5, 1.0) for f in FOLDS}
    for f in FOLDS:
        res[f]["baseline"] = summ(base_eval[f])
        assert abs(res[f]["baseline"]["eval_VTOP5"] - J15["table"][f]["b_5w_fit_10w_infer"]["VTOP5"]) < 1e-9, (f, res[f]["baseline"])
    print("baseline reproduces step-15 analogue (b):", {f: round(res[f]["baseline"]["eval_VTOP5"], 4) for f in FOLDS}, flush=True)

    # ================================================================== 1. inference-time decay c
    for f in FOLDS:
        res[f]["item1_c"] = {str(c): summ(evaluate(f, 50, c, 1.0)) for c in C_GRID}
    print("item 1 done", {f: {c: round(v["train_VTOP5"], 4) for c, v in res[f]["item1_c"].items()} for f in FOLDS}, flush=True)
    # ================================================================== 2. rank fusion of the two inference windows
    for f in FOLDS:
        res[f]["item2_w"] = {str(w): summ(evaluate(f, 50, 0.5, w)) for w in W_GRID}
    print("item 2 done", {f: {w: round(v["train_VTOP5"], 4) for w, v in res[f]["item2_w"].items()} for f in FOLDS}, flush=True)
    # ================================================================== 3. candidate depth K (training and inference)
    for K in K_GRID[1:]:
        fit_models(K)
    for f in FOLDS:
        res[f]["item3_K"] = {str(K): summ(evaluate(f, K, 0.5, 1.0)) for K in K_GRID}
    print("item 3 done", {f: {k: round(v["train_VTOP5"], 4) for k, v in res[f]["item3_K"].items()} for f in FOLDS}, flush=True)

    # ================================================================== choice on fold-internal training users, consensus, combination
    def pick(f, item, grid, base):
        d = res[f][item]
        best = max(grid, key=lambda g: (d[str(g)]["train_VTOP5"], g == base))
        return best if d[str(best)]["train_VTOP5"] > d[str(base)]["train_VTOP5"] else base
    chosen = {f: {"c": pick(f, "item1_c", C_GRID, BASE["c"]), "w": pick(f, "item2_w", W_GRID, BASE["w"]), "K": pick(f, "item3_K", K_GRID, BASE["K"])} for f in FOLDS}
    cons = {k: (chosen["A"][k] if chosen["A"][k] == chosen["B"][k] else BASE[k]) for k in BASE}
    R["chosen_on_training_users"], R["consensus"] = chosen, cons
    print("chosen", chosen, "consensus", cons, flush=True)
    comb_eval = {f: evaluate(f, cons["K"], cons["c"], cons["w"]) for f in FOLDS}
    for f in FOLDS:
        res[f]["combination"] = summ(comb_eval[f])
    rng = np.random.default_rng(SEED)
    diff = {f: comb_eval[f]["eval"]["VTOP5"] - base_eval[f]["eval"]["VTOP5"] for f in FOLDS}
    R["bootstrap_combination_minus_baseline_vtop5"] = {"A": boot(diff["A"], rng), "B": boot(diff["B"], rng), "pooled": boot(np.r_[diff["A"], diff["B"]], rng)}
    R["bootstrap_robustness_pooled"] = {v: boot(np.r_[comb_eval["A"]["eval"][v] - base_eval["A"]["eval"][v], comb_eval["B"]["eval"][v] - base_eval["B"]["eval"][v]], rng) for v in VARS}
    # single levers on evaluation users (reporting only, no choice is made from them)
    R["levers_alone_pooled_vs_baseline"] = {}
    for item, grid, mk in (("item1_c", C_GRID, lambda g: (50, g, 1.0)), ("item2_w", W_GRID, lambda g: (50, 0.5, g)), ("item3_K", K_GRID, lambda g: (g, 0.5, 1.0))):
        R["levers_alone_pooled_vs_baseline"][item] = {}
        for g in grid:
            dd = np.concatenate([evaluate(f, *mk(g))["eval"]["VTOP5"] - base_eval[f]["eval"]["VTOP5"] for f in FOLDS])
            R["levers_alone_pooled_vs_baseline"][item][str(g)] = boot(dd, rng)
    R["table"] = res
    both = all(res[f]["combination"]["eval_VTOP5"] > res[f]["baseline"]["eval_VTOP5"] for f in FOLDS)
    ci_ok = R["bootstrap_combination_minus_baseline_vtop5"]["pooled"]["ci95"][0] > 0
    ship = bool(both and ci_ok and cons != BASE)
    R["rule"] = ("Ship only if the combination of the levers chosen in both folds on training users beats the step-15 analogue baseline on VTOP5 in BOTH folds "
                 "and the pooled paired-bootstrap 95% CI excludes zero.")
    R["decision"] = {"combination_beats_baseline_in_both_folds": bool(both), "pooled_ci_excludes_zero": bool(ci_ok), "consensus_differs_from_baseline": cons != BASE, "shipped": ship}
    print("decision", R["decision"], flush=True)
    R["validation_seconds"] = round(time.time() - T0, 1)

    # ================================================================== real final run (only if shipped)
    if ship:
        R["final"] = final_run(cons, cfg, ew_m, ctx10, uc_pairs, inter, cat_series, users, n_p, T5, folds, J15, stores, base10, get_store, feats)
    R["runtime_s"] = round(time.time() - T0, 1)
    (OUT / "validation_step17.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    write_reports(R)
    print(json.dumps({k: R[k] for k in ("consensus", "decision", "runtime_s") if k in R}, indent=1, default=str))
    if ship:
        print(json.dumps(R["final"], indent=1, default=str))


def final_run(cons, cfg, ew_m, ctx10, uc_pairs, inter, cat_series, users, n_p, T5, folds, J15, stores, base10, get_store, feats):
    """Model as in step 15 (per-week, weeks 11-20 features, all validation users, 142 trees; candidates top-K), inference with the chosen settings."""
    K, c, w = cons["K"], cons["c"], cons["w"]
    ri, ci, X = feats(ctx10, get_store("10", ctx10, base10, 0.5), K, 10)
    y = T5[ri, ci].astype(int)
    all_train = np.sort(np.r_[folds["A"]["train"], folds["B"]["train"]])
    model = rerank2.train(X, y, ri, all_train, None, rerank2.BASE_PARAMS, rounds=J15["final_model"]["trees"])
    users_f = data.load_submission_users()
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    nf, npf = len(users_f), len(pool_f)

    def window_scores(periods, wf):
        ctxf = s7.make_ctx(periods, "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
        st, _ = s7.build_store(ctxf, ["PPR", "MF", "AR"], cfg)
        st["H2"] = s7.build_store(ctxf, ["H2"], {**cfg, "h2_c": c})[0]["H2"]
        st["EW"] = s7.build_store(ctxf, ["EW"], cfg, (ew_m, c))[0]["EW"]
        cand = rerank.candidates(st, ctxf["pop_norm"], nf, npf, names=rerank.ORDER, topn=K)
        rif, cif, Xf, _ = rerank.make_features(ctxf, st, ctxf["pop_norm"], cand)
        Xf = rerank2.per_week(Xf, wf, 12)[list(X.columns)]
        return model.predict(Xf), rif, cif, ctxf, st
    sb, rib, cib, ctxf, st = window_scores(["11-20", "21-30"], 20)
    if w < 1.0:
        sa, ria, cia, _, _ = window_scores(["21-30"], 10)
        s, rf, cf = fuse(sa, ria, cia, sb, rib, cib, w, npf)
    else:
        s, rf, cf = sb, rib, cib
    top = top_from_scores(s, rf, cf, np.arange(nf))
    preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
    path = SUBDIR / "submission_STEP17.csv"
    write_submission(preds, users_f, path)
    prev = pd.read_csv(SUBDIR / "submission_STEP15.csv", dtype=str).set_index("user_id")["tv_show_id"]
    shared = [len(set(preds[u]) & set(int(x) for x in prev[str(u)].split())) / 5 for u in users_f]
    pos = ctxf["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
    return {"settings": cons, "model_trees": J15["final_model"]["trees"], "fit_users": int(len(all_train)), **s3.check_file(path, users_f, pool_f),
            "users_without_component_data": {n: int((~st[n][1]).sum()) for n in st},
            "repeat_share_train_positive": float(np.mean([len(set(preds[u]) & pos.get(np.uint64(u), set())) / 5 for u in users_f])),
            "share_of_ids_shared_with_STEP15": float(np.mean(shared)), "share_of_users_with_identical_list": float(np.mean([x == 1.0 for x in shared]))}


def write_reports(R):
    res, d, cons = R["table"], R["decision"], R["consensus"]
    base = {f: res[f]["baseline"]["eval_VTOP5"] for f in FOLDS}
    md = ["# Step 17 - last improvement round on the inference-window lever", "",
          "Baseline = step-15 validation analogue (b): per-week reranker fitted on features/candidates of weeks 16-20 (labels VTOP5 of weeks 21-30), inference with features/candidates of weeks 11-20 "
          f"(VTOP5 fold A {base['A']:.4f}, fold B {base['B']:.4f}, mean {(base['A'] + base['B']) / 2:.4f}; reproduced here exactly). Seed 42, 2-fold swap, primary VTOP5. "
          "Every setting is chosen on the fold-internal TRAINING users only (the users the fold's model was fitted on); a setting is part of the combination only if BOTH folds choose the same non-baseline value.", "",
          "## 1. Inference-time decay c of H2 and EW (model unchanged)", "", "| c per 60 days | fold | VTOP5 training users (selection) | VTOP5 evaluation users |", "|---|---|---|---|"]
    for key, item, nm in (("c", "item1_c", "1"), ("w", "item2_w", "2"), ("K", "item3_K", "3")):
        if key == "w":
            md += ["", "## 2. Rank fusion of the weeks 16-20 and 11-20 inference scores (weight w on 11-20; w = 1 is the baseline)", "", "| w | fold | VTOP5 training users (selection) | VTOP5 evaluation users |", "|---|---|---|---|"]
        if key == "K":
            md += ["", "## 3. Candidate depth K per component (training and inference)", "", "| K | fold | VTOP5 training users (selection) | VTOP5 evaluation users |", "|---|---|---|---|"]
        for g in res["A"][item]:
            for f in FOLDS:
                r = res[f][item][g]
                md.append(f"| {g} | {f} | {r['train_VTOP5']:.4f} | {r['eval_VTOP5']:.4f} |")
    md += ["", "Single levers, pooled evaluation users, difference to the baseline (paired bootstrap 95% CI; reporting only, nothing was chosen from these):", "", "| lever | value | mean diff | 95% CI |", "|---|---|---|---|"]
    for item, dd in R["levers_alone_pooled_vs_baseline"].items():
        for g, b in dd.items():
            md.append(f"| {item} | {g} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "## 4. Choice on training users and combination", "", f"- Chosen per fold on training users: {R['chosen_on_training_users']}.", f"- Consensus (same value in both folds, else baseline): {cons}.", "",
           "| setting | fold | VTOP5 | V300 | V900 | VHALF |", "|---|---|---|---|---|---|"]
    for f in FOLDS:
        for lab, k in (("baseline", "baseline"), ("combination", "combination")):
            r = res[f][k]
            md.append(f"| {lab} | {f} | {r['eval_VTOP5']:.4f} | {r['eval_V300']:.4f} | {r['eval_V900']:.4f} | {r['eval_VHALF']:.4f} |")
    bs = R["bootstrap_combination_minus_baseline_vtop5"]
    md += ["", "Paired bootstrap (2,000 resamples, seed 42), combination - baseline, VTOP5:", "", "| scope | users | mean diff | 95% CI |", "|---|---|---|---|"]
    for k in ("A", "B", "pooled"):
        b = bs[k]
        md.append(f"| {k} | {b['n_users']} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "Robustness, pooled combination - baseline:", "", "| truth | mean diff | 95% CI |", "|---|---|---|"]
    for v in VARS:
        b = R["bootstrap_robustness_pooled"][v]
        md.append(f"| {v} | {b['mean_diff']:+.4f} | [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}] |")
    md += ["", "## 5. Pre-declared rule and decision", "", R["rule"], "",
           f"- Combination beats baseline in both folds: {d['combination_beats_baseline_in_both_folds']}; pooled CI excludes zero: {d['pooled_ci_excludes_zero']}; consensus differs from baseline: {d['consensus_differs_from_baseline']}.",
           f"- **Shipped: {'yes, submission_STEP17.csv' if d['shipped'] else 'no - submission_STEP15.csv stays the final file; nothing else was changed'}**.", ""]
    if d["shipped"]:
        fi = R["final"]
        md += ["## Shipped file `outputs/submissions/submission_STEP17.csv`", "",
               f"- Settings {fi['settings']}; model as step 15 (per-week, weeks 11-20 features, {fi['fit_users']:,} users, {fi['model_trees']} trees); inference weeks 11-30 (fusion partner: weeks 21-30 if w < 1).",
               f"- Format: header {fi['header_ok']}, rows {fi['rows']}, user order {fi['user_order_ok']}, 5 distinct ids {fi['five_distinct_ids']}, ids in final pool {fi['all_ids_in_final_pool']}.",
               f"- Users without component data (popularity fallback): {fi['users_without_component_data']}; repeat share {fi['repeat_share_train_positive']:.3f}.",
               f"- Share of ids identical to submission_STEP15.csv: {fi['share_of_ids_shared_with_STEP15']:.3f} (identical lists for {fi['share_of_users_with_identical_list']:.1%} of users).", ""]
    b = bs["pooled"]
    md += ["## Summary", "",
           f"The settings chosen on training users were c = {R['chosen_on_training_users']['A']['c']} / {R['chosen_on_training_users']['B']['c']}, w = {R['chosen_on_training_users']['A']['w']} / {R['chosen_on_training_users']['B']['w']}, "
           f"K = {R['chosen_on_training_users']['A']['K']} / {R['chosen_on_training_users']['B']['K']} (folds A / B); the consensus was {cons}. "
           f"The combination scored VTOP5 {res['A']['combination']['eval_VTOP5']:.4f} and {res['B']['combination']['eval_VTOP5']:.4f} against the baseline {base['A']:.4f} and {base['B']:.4f}, "
           f"pooled difference {b['mean_diff']:+.4f} with 95% CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]. "
           f"By the pre-declared rule the result was {'shipped as submission_STEP17.csv' if d['shipped'] else 'not shipped, so submission_STEP15.csv remains the final file'}."]
    (OUT / "validation_step17.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if d["shipped"]:
        sm = SUBDIR / "SUBMISSIONS.md"
        txt = sm.read_text(encoding="utf-8")
        if "submission_STEP17.csv" not in txt:
            sm.write_text(txt.rstrip("\n") + f"\n| submission_STEP17.csv | step 17: step-15 model with inference settings {cons} | chosen on fold-internal training users; cannot be scored on validation | not submitted | not submitted |\n", encoding="utf-8")


if __name__ == "__main__":
    main()

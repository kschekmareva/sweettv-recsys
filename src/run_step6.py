"""Step 6: VTOP5 is the primary criterion. Evaluates the VTOP5-tuned weights of the four-component family (no new search),
writes submission_FINAL_vtop5.csv, appends a section to outputs/validation_step5.md and writes SUBMISSIONS.md.
Usage: python src/run_step6.py
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
import ppr
import run_step3 as s3
import run_step4 as s4
import run_step5 as s5
from run_validation import write_submission

SEED, THR, K = 42, 300, 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
FAMILY = "P-air+MF+PPR+H2"
RULE = ("Pre-declared rule from now on: VTOP5 is the PRIMARY validation criterion, because it matches the task shape "
        "(exactly 5 ids per user; truth = the user's most-completed programmes; definition taken from the cited public notebook). "
        "V300, V900 and VHALF are robustness checks. The Kaggle number was used once, only to check that P's VTOP5 score (0.270-0.296) "
        "is plausible against the leaderboard (0.278); no weights or hyperparameters are chosen from Kaggle.")


def main():
    t_all = time.time()
    J = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    w_vtop5 = {k: v for k, v in J["rows"][FAMILY]["weights"]["VTOP5"].items()}
    w_shared = {k: v for k, v in J["rows"][f"{FAMILY} [shared w]"]["weights"].items()}
    c_best, mf_cfg, mf4_cfg = J["h2"]["c_best"], J["mf"]["chosen"], J["mf"]["step4_config_kept_for_blend"]
    assert J["pool_option"]["adopted"] == "none"           # pool rule unchanged (no film filter)
    print("weights VTOP5-tuned", w_vtop5, "| step-5 shared", w_shared, "| c", c_best, "| MF", mf_cfg, flush=True)

    # ------------------------------------------------------------------ validation setup (identical to step 5)
    inter, _ = data.build_interactions()
    train_all = data.positives(inter, ["11-20"], 0)
    train_pos = data.positives(inter, ["11-20"], THR)
    air_tr, air_val = data.airings(["11-20"]), data.airings(["21-30"])
    val = inter[inter["period"] == "21-30"]
    vpairs = {"V300": val[val["overlap_sec"] >= 300], "V900": val[val["overlap_sec"] >= 900], "VHALF": val[val["max_frac"] >= 0.5]}
    vpairs["VTOP5"], _ = h2.vtop5_truth(air_val)
    sched_val = data.load_schedule("21-30")
    pool = data.pool_ids(sched_val, 21, 30)
    sub_users = data.load_submission_users()
    eval_a = sorted(int(u) for u in vpairs["V300"]["user_id"].unique())
    users = sorted(set(int(u) for v in vpairs.values() for u in v["user_id"].unique()) | set(sub_users))
    uix, pix = pd.Index(np.asarray(users, dtype=np.uint64)), pd.Index(pool)
    n_u, n_p = len(users), len(pool)
    T = {v: s4.bool_matrix(vpairs[v], uix, pix, n_u, n_p) for v in VARS}
    nonempty = {v: T[v].any(axis=1) for v in VARS}
    Tpos = s4.bool_matrix(train_pos, uix, pix, n_u, n_p)
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    sub_rows = uix.get_indexer(np.asarray(sub_users, dtype=np.uint64))
    report_v = {v: report_rows[nonempty[v][report_rows]] for v in VARS}
    full_v = {v: np.flatnonzero(nonempty[v]) for v in VARS}

    P = hybrid.pop_counts(train_pos, pool)
    pop_norm = hybrid.minmax_vec(P * np.log1p(s4.airing_counts(sched_val, 21, 30, pool)))
    top_of = s5.make_top_fn(pop_norm)
    prep = h2.prepare(air_tr)
    S, valid, _ = h2.scores(prep, h2.train_airing_counts(pool, ("11-20",)), users, pool, h2.sec_of("2020-05-18"), c_best)
    store = {"H2": (hybrid.minmax_rows(S), valid)}
    for name, cfg in (("MF", mf_cfg), ("MF4", mf4_cfg)):
        c = s3.mf_component(train_all, cfg, users, pool)
        store[name] = (c["Sn"], c["valid"])
    g = ppr.build_graph(train_pos)
    S, valid, _ = ppr.run(g, users, pool, J["ppr"]["alpha"])
    store["PPR"] = (hybrid.minmax_rows(S), valid)

    specs = {
        "P": None,
        "step-4 blend": s5.fuse_spec(s5.BLEND4),
        "step-5 blend (shared w)": s5.fuse_spec(w_shared),
        "step-6 FINAL (VTOP5-tuned w)": s5.fuse_spec(w_vtop5),
    }
    rows, aps = {}, {}
    for name, spec in specs.items():
        top = np.tile(np.argsort(-P, kind="stable")[:K], (n_u, 1)) if spec is None else top_of(spec, store, n_u)
        rows[name], aps[name] = {}, {}
        for v in VARS:
            ap = metrics.ap5_matrix(top, T[v])
            aps[name][v] = ap
            r = full_v[v]
            rows[name][v] = {"report": float(ap[report_v[v]].mean()), "full": float(ap[r].mean()), "submission": float(ap[sub_rows].mean()),
                             "repeat_share_train_positive_full": float(Tpos[r[:, None], top[r]].mean()),
                             "mean_train_popularity_full": float(P[top[r]].mean())}
    # reproduction check against the step-5 report
    chk = {v: abs(rows["step-4 blend"][v]["report"] - J["rows"]["step-4 blend"]["metrics"][v]["report"]) for v in VARS}
    chk |= {f"shared_{v}": abs(rows["step-5 blend (shared w)"][v]["report"] - J["rows"][f"{FAMILY} [shared w]"]["metrics"][v]["report"]) for v in VARS}
    assert max(chk.values()) < 1e-9, chk
    # the VTOP5-tuned weights on VTOP5 must equal the per-variant row of step 5
    assert abs(rows["step-6 FINAL (VTOP5-tuned w)"]["VTOP5"]["report"] - J["rows"][FAMILY]["metrics"]["VTOP5"]["report"]) < 1e-9

    rng = np.random.default_rng(SEED)
    sig = {}
    for ref in ("P", "step-4 blend", "step-5 blend (shared w)"):
        sig[ref] = {}
        for v in VARS:
            idx = report_v[v]
            d = aps["step-6 FINAL (VTOP5-tuned w)"][v][idx] - aps[ref][v][idx]
            bm = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            sig[ref][v] = {"n_users": int(len(d)), "mean_diff": float(d.mean()), "ci95": [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]}

    # ------------------------------------------------------------------ final file: TRAIN 11-30, pool 31-42
    print("final", flush=True)
    train_all_f = data.positives(inter, list(data.PERIODS), 0)
    train_pos_f = data.positives(inter, list(data.PERIODS), THR)
    sched_f = data.load_schedule("31-42")
    pool_f = data.pool_ids(sched_f, 31, 42)
    users_f = data.load_submission_users()
    pop_norm_f = hybrid.minmax_vec(hybrid.pop_counts(train_pos_f, pool_f) * np.log1p(s4.airing_counts(sched_f, 31, 42, pool_f)))
    prep_f = h2.prepare(data.airings(["11-20", "21-30"]))
    S, valid, _ = h2.scores(prep_f, h2.train_airing_counts(pool_f, ("11-20", "21-30")), users_f, pool_f, h2.sec_of("2020-07-27"), c_best)
    sf = {"H2": (hybrid.minmax_rows(S), valid)}
    c = s3.mf_component(train_all_f, mf_cfg, users_f, pool_f)
    sf["MF"] = (c["Sn"], c["valid"])
    gf = ppr.build_graph(train_pos_f)
    S, valid, _ = ppr.run(gf, users_f, pool_f, J["ppr"]["alpha"])
    sf["PPR"] = (hybrid.minmax_rows(S), valid)
    spec_f = s5.fuse_spec(w_vtop5)
    top = s5.make_top_fn(pop_norm_f)(spec_f, sf, len(users_f))
    preds = {u: [int(x) for x in pool_f[top[i]]] for i, u in enumerate(users_f)}
    path = SUBDIR / "submission_FINAL_vtop5.csv"
    write_submission(preds, users_f, path)
    check = s3.check_file(path, users_f, pool_f)
    pos_f = train_pos_f.groupby("user_id")["tv_show_id"].agg(set)
    rep = [len(set(preds[u]) & pos_f.get(np.uint64(u), set())) / K for u in users_f]
    prev = pd.read_csv(SUBDIR / "submission_CHOSEN_step5.csv", dtype=str)
    overlap = float(np.mean([len(set(a.split()) & set(b.split())) / K for a, b in zip(prev["tv_show_id"], [" ".join(map(str, preds[u])) for u in users_f])]))
    final = {"file": check["file"], "checks": check, "weights": {**w_vtop5, "P-air": round(1 - sum(w_vtop5.values()), 1)}, "pool_size": int(len(pool_f)),
             "users_without_component_data": {n: int((~sf[n][1]).sum()) for n in ("H2", "MF", "PPR")},
             "users_with_any_missing_component": int((~(sf["H2"][1] & sf["MF"][1] & sf["PPR"][1])).sum()),
             "repeat_share_train_positive": float(np.mean(rep)), "mean_overlap_with_step5_chosen_file": overlap,
             "h2_c": c_best, "mf_cfg": mf_cfg, "ppr_alpha": J["ppr"]["alpha"], "popularity_base": "P-air (31-42 schedule used only for airing counts)"}

    R = {"rule": RULE, "weights_vtop5_tuned": w_vtop5, "weights_step5_shared": w_shared, "rows": rows, "significance": sig,
         "reproduction_abs_diff_max": max(chk.values()), "final": final, "runtime_s": round(time.time() - t_all, 1)}
    (OUT / "validation_step6.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")

    # ------------------------------------------------------------------ append section to validation_step5.md
    f4 = lambda x: f"{x:.4f}"
    wf = lambda w: ", ".join(f"{k} {v}" for k, v in w.items()) + f", P-air {round(1 - sum(w.values()), 1)}"
    sec = ["## Step 6 - decision rule", "", RULE, "",
           f"Family: P-air + MF + PPR + H2 (components as in step 5: H2 c = {c_best}, MF {mf_cfg}, PPR alpha {J['ppr']['alpha']}, popularity base P-air, no pool filter). "
           f"Weights chosen on TUNE under VTOP5 in step 5 (no new search): **{wf(w_vtop5)}**. Step-5 shared-weight set: {wf(w_shared)}. Step-4 blend: P-air 0.2, MF4 0.4, PPR 0.4.", "",
           "REPORT MAP@5 (same TUNE/REPORT halves, seed 42):", "", "| strategy | VTOP5 (primary) | V300 | V900 | VHALF |", "|---|---|---|---|---|"]
    for n, r in rows.items():
        sec.append(f"| {n} | " + " | ".join(f4(r[v]["report"]) for v in ("VTOP5", "V300", "V900", "VHALF")) + " |")
    sec += ["", "Paired bootstrap of the step-6 strategy (VTOP5-tuned weights), 2,000 resamples, seed 42, REPORT users with non-empty truth: mean AP@5 difference [95% CI]", "",
            "| vs | VTOP5 (primary) | V300 | V900 | VHALF |", "|---|---|---|---|---|"]
    for ref, d in sig.items():
        sec.append(f"| {ref} | " + " | ".join(f"{d[v]['mean_diff']:+.4f} [{d[v]['ci95'][0]:+.4f}, {d[v]['ci95'][1]:+.4f}]" for v in ("VTOP5", "V300", "V900", "VHALF")) + " |")
    fr = rows["step-6 FINAL (VTOP5-tuned w)"]
    sec += ["", f"Repeat share (recommended items already a TRAIN positive for the user, validation users with truth): "
            + ", ".join(f"{v} {fr[v]['repeat_share_train_positive_full']:.2f}" for v in VARS) + ".", "",
            "Final file (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning, no film filter):", "",
            f"- {final['file']}: header {check['header_ok']}, rows {check['rows']}, user order {check['user_order_ok']}, 5 distinct ids {check['five_distinct_ids']}, ids in final pool {check['all_ids_in_final_pool']}",
            f"- weights {wf(w_vtop5)}; pool {final['pool_size']:,} programmes",
            f"- fallback to popularity (no data for the component): {final['users_without_component_data']}; users with at least one missing component: {final['users_with_any_missing_component']} of {len(users_f)}",
            f"- repeat share on the final lists (items that are TRAIN-positive for the user): {final['repeat_share_train_positive']:.3f}",
            f"- mean overlap with submission_CHOSEN_step5.csv: {overlap:.3f} of 5 ids per user (as a share)", ""]
    p5 = OUT / "validation_step5.md"
    txt = p5.read_text(encoding="utf-8")
    if "## Step 6 - decision rule" in txt:
        txt = txt[:txt.index("## Step 6 - decision rule")].rstrip() + "\n\n"
    p5.write_text(txt.rstrip() + "\n\n" + "\n".join(sec), encoding="utf-8")

    # ------------------------------------------------------------------ SUBMISSIONS.md
    j3 = json.loads((OUT / "validation_step3.json").read_text(encoding="utf-8"))
    j4 = json.loads((OUT / "validation_step4.json").read_text(encoding="utf-8"))
    mf3, mf4 = j3["mf_best_cfg"], j4["selected"]["mf_cfg"]
    items = [
        ("submission_P.csv", "step 3: global popularity (P)", "positives (>= 300 s) of weeks 11-30, distinct users, pool 31-42"),
        ("submission_MF.csv", "step 3: implicit ALS, pure (popularity fallback)", f"factors {mf3['factors']}, reg {mf3['reg']}, alpha {mf3['alpha']}, 15 iterations, seed 42"),
        ("submission_CBF.csv", "step 3: content-based (cosine to user profile)", "category + genre + TF-IDF people + decade blocks; profile = positives weighted by log1p(overlap/60)"),
        ("submission_HYBRID.csv", "step 3: popularity + MF score fusion", f"MF as above, w_MF {j3['hybrid_tuning_on_tune']['chosen']['w_mf']}, popularity {round(1 - j3['hybrid_tuning_on_tune']['chosen']['w_mf'], 1)}"),
        ("submission_PPR.csv", "step 4: personalised PageRank, pure", f"alpha {J['ppr']['alpha']} (link-following), tol 1e-6, fallback P-air"),
        ("submission_AR.csv", "step 4: item-to-item association rules, pure", f"confidence, mode {j4['selected']['ar_mode']}, min support 20 users, fallback P-air"),
        ("submission_CHOSEN.csv", "step 4: blend P-air + MF + PPR", f"P-air 0.2, MF 0.4 (factors {mf4['factors']}, reg {mf4['reg']}, alpha {mf4['alpha']}), PPR 0.4 (alpha {J['ppr']['alpha']})"),
        ("submission_H2.csv", "step 5: H2 own-history (airings with view_pct >= 0.8), pure", f"r = 5 if view_pct = 1 else 2 + 2*view_pct^2, decay c {c_best} per 60 days, x log10(1 + airings in TRAIN), fallback P-air"),
        ("submission_CHOSEN_step5.csv", "step 5: P-air + MF + PPR + H2, shared weights", f"{wf(w_shared)}; MF alpha {mf_cfg['alpha']}, PPR alpha {J['ppr']['alpha']}, H2 c {c_best}"),
        ("submission_FINAL_vtop5.csv", "step 6: P-air + MF + PPR + H2, weights tuned on VTOP5", f"{wf(w_vtop5)}; MF alpha {mf_cfg['alpha']}, PPR alpha {J['ppr']['alpha']}, H2 c {c_best}"),
    ]
    missing = [f for f, _, _ in items if not (SUBDIR / f).exists()]
    assert not missing, missing
    lines = ["# Submissions", "",
             "All files: TRAIN = viewing weeks 11-30, candidate pool = labelled programmes airing in weeks 31-42, 5 distinct ids per user, template format. "
             "Fill in the two Kaggle columns after each submission.", "",
             "| file | method | weights / hyperparameters | Kaggle public | Kaggle private |", "|---|---|---|---|---|"]
    lines += [f"| {f} | {m} | {h} | | |" for f, m, h in items]
    (SUBDIR / "SUBMISSIONS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"done in {R['runtime_s']} s")


if __name__ == "__main__":
    main()

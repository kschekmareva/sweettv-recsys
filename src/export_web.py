"""Exports web/data/web_data.json for the local web demo (web/index.html). No new modelling: the validation set-up, the fold reranker
and the final-run lists are rebuilt exactly as in make_examples.py (the step-7 report is reproduced by an assertion), then written for the browser.
Usage: python src/export_web.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import re

import numpy as np
import pandas as pd

import cbf
import data
import h2
import metrics
import rerank
import rerank2
import run_step3 as s3
import run_step4 as s4
import run_step5 as s5
import run_step7 as s7

SEED, K = 42, 5
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
WEB = data.ROOT / "web" / "data"
METHODS = ["P", "P-air", "H2", "CBF", "MF", "PPR", "AR", "step-6 blend", "step-7 reranker"]


def parse_handcheck():
    t = (OUT / "handcheck.md").read_text(encoding="utf-8")
    raw = re.search(r"```\n(.+?)\n```", t).group(1).split(",")
    rows = []
    for line in t.splitlines():
        if re.match(r"\| \d+ \| ", line):
            c = [x.strip() for x in line.strip("|").split("|")]
            rows.append({"channel_id": c[0], "channel_title": c[1], "start_time": c[2], "duration": int(c[3]), "tv_show_id": c[4], "tv_show_title": c[5],
                         "code_overlap_sec": int(c[6]), "code_view_pct": float(c[7])})
    user = re.search(r"user A \((\d+)\)", t).group(1)
    return {"user_id": user, "view": dict(zip(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"], raw)), "schedule": rows}


def parse_kaggle():
    out = []
    for line in (SUBDIR / "SUBMISSIONS.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("| submission_"):
            c = [x.strip() for x in line.strip("|").split("|")]
            out.append({"file": c[0], "method": c[1], "public": c[-2], "private": c[-1]})
    return out


def main():
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    w6 = dict(J5["rows"]["P-air+MF+PPR+H2"]["weights"]["VTOP5"])

    # ---- validation set-up (identical to make_examples.py / step 7)
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
    ctx = s7.make_ctx(["11-20"], "21-30", (21, 30), users, pool, uc_pairs, inter, cat_series)
    uix = ctx["uix"]
    T5 = s4.bool_matrix(vpairs["VTOP5"], uix, ctx["pix"], n_u, n_p)
    ne5 = T5.any(axis=1)
    perm = np.random.default_rng(SEED).permutation(len(eval_a))
    half = len(eval_a) // 2
    tune_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[:half]], dtype=np.uint64)))
    report_rows = np.sort(uix.get_indexer(np.asarray([eval_a[i] for i in perm[half:]], dtype=np.uint64)))
    store, _ = s7.build_store(ctx, ["H2", "PPR", "MF", "AR"], cfg)
    store["EW"] = s7.build_store(ctx, ["EW"], cfg, ew_cfg)[0]["EW"]
    items = cbf.build_items(cbf.load_metadata())
    cb = s3.cbf_component(items, 0.0, ctx["train_pos"], users, pool)
    store["CBF"] = (cb["Sn"], cb["valid"])
    top_of = s5.make_top_fn(ctx["pop_norm"])
    spec = lambda w: s5.fuse_spec(w)
    tops = {"P": np.tile(np.argsort(-ctx["P"], kind="stable")[:K], (n_u, 1)),
            "P-air": np.tile(np.argsort(-ctx["pop_norm"], kind="stable")[:K], (n_u, 1)),
            "H2": top_of(spec({"H2": 1.0}), store, n_u), "CBF": top_of(spec({"CBF": 1.0}), store, n_u), "MF": top_of(spec({"MF": 1.0}), store, n_u),
            "PPR": top_of(spec({"PPR": 1.0}), store, n_u), "AR": top_of(spec({"AR": 1.0}), store, n_u), "step-6 blend": top_of(spec(w6), store, n_u)}
    cand = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
    y = T5[ri, ci].astype(int)
    memb = {n: rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=[n]) for n in rerank.ORDER}   # which method's top-50 holds each programme
    seen_val = ctx["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
    tune_truth = tune_rows[ne5[tune_rows]]
    p = np.random.default_rng(SEED).permutation(tune_truth)
    n_va = int(round(0.2 * len(p)))
    model = rerank2.train(X, y, ri, np.sort(p[n_va:]), np.sort(p[:n_va]), rerank2.BASE_PARAMS)   # fitted on TUNE users only
    top_rr = rerank.top5(model, X, ri, ci, report_rows)
    rr_tops = np.zeros((n_u, K), dtype=int)
    rr_tops[report_rows] = top_rr
    tops["step-7 reranker"] = rr_tops
    chk = metrics.ap5_matrix(top_rr, T5[report_rows])[ne5[report_rows]].mean()
    assert abs(chk - J7["rows"]["reranker"]["VTOP5"]["report"]) < 1e-9, chk

    # ---- REPORT users with VTOP5 truth: truth (with completed-airing counts), Top-5 per method, Python AP@5
    rep = report_rows[ne5[report_rows]]
    val_done = air_val[air_val["view_pct"] >= 0.8].groupby(["user_id", "tv_show_id"]).size()
    vt = vpairs["VTOP5"]
    truth_by_user = {int(u): [int(s) for s in g] for u, g in vt.groupby("user_id")["tv_show_id"]}
    vusers, ids_used = {}, set()
    for r in rep:
        uid = users[r]
        truth = truth_by_user[uid]
        tset = set(truth)
        lists = {m: [int(pool[x]) for x in tops[m][r]] for m in METHODS}
        sv = seen_val.get(np.uint64(uid), set())
        shown = set(truth).union(*[set(l) for l in lists.values()])
        vusers[str(uid)] = {"truth": [[i, int(val_done.get((np.uint64(uid), i), 0))] for i in truth], "top": lists,
                            "ap": {m: metrics.ap_at_k(lists[m], tset, K) for m in METHODS},
                            "seen": sorted(int(i) for i in shown if i in sv),
                            "rr_from": [[n for n in rerank.ORDER if memb[n][r, tops["step-7 reranker"][r][k]]] for k in range(K)]}
        ids_used |= tset
        for l in lists.values():
            ids_used |= set(l)
    vmap = {m: float(np.mean([u["ap"][m] for u in vusers.values()])) for m in METHODS}
    assert abs(vmap["step-7 reranker"] - J7["rows"]["reranker"]["VTOP5"]["report"]) < 1e-9

    # ---- final run lists (TRAIN weeks 11-30, pool 31-42); reranker column = the committed final file
    users_f = data.load_submission_users()
    tmpl = pd.read_csv(data.DATA / "submission.csv", dtype=str)["user_id"].tolist()
    assert [str(u) for u in users_f] == tmpl
    pool_f = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    ctx_f = s7.make_ctx(["11-20", "21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)
    store_f, _ = s7.build_store(ctx_f, ["H2", "PPR", "MF", "AR"], cfg)
    cbf_f = s3.cbf_component(items, 0.0, ctx_f["train_pos"], users_f, pool_f)
    store_f["CBF"] = (cbf_f["Sn"], cbf_f["valid"])
    top_of_f = s5.make_top_fn(ctx_f["pop_norm"])
    nf = len(users_f)
    fin = {"P": np.tile(np.argsort(-ctx_f["P"], kind="stable")[:K], (nf, 1)), "P-air": np.tile(np.argsort(-ctx_f["pop_norm"], kind="stable")[:K], (nf, 1)),
           "H2": top_of_f(spec({"H2": 1.0}), store_f, nf), "CBF": top_of_f(spec({"CBF": 1.0}), store_f, nf), "MF": top_of_f(spec({"MF": 1.0}), store_f, nf),
           "PPR": top_of_f(spec({"PPR": 1.0}), store_f, nf), "AR": top_of_f(spec({"AR": 1.0}), store_f, nf), "step-6 blend": top_of_f(spec(w6), store_f, nf)}
    final_file = pd.read_csv(SUBDIR / "submission_STEP7_all.csv", dtype=str).set_index("user_id")["tv_show_id"]
    flists = {}
    for m in METHODS[:-1]:
        flists[m] = [[int(pool_f[x]) for x in fin[m][i]] for i in range(nf)]
    flists["step-7 reranker"] = [[int(x) for x in final_file[str(u)].split()] for u in users_f]
    for l in flists.values():
        for row in l:
            ids_used |= set(row)
    seen_f = ctx_f["train_pos"].groupby("user_id")["tv_show_id"].agg(set)
    fseen = []
    for i, u in enumerate(users_f):
        sv = seen_f.get(np.uint64(u), set())
        fseen.append(sorted({x for m in METHODS for x in flists[m][i] if x in sv}))
    ctx_r = s7.make_ctx(["21-30"], "31-42", (31, 42), users_f, pool_f, uc_pairs, inter, cat_series)    # window of the final reranker
    store_r, _ = s7.build_store(ctx_r, ["H2", "PPR", "MF", "AR"], cfg)
    store_r["EW"] = s7.build_store(ctx_r, ["EW"], cfg, ew_cfg)[0]["EW"]
    memb_r = {n: rerank.candidates(store_r, ctx_r["pop_norm"], nf, len(pool_f), names=[n]) for n in rerank.ORDER}
    pix_f = {int(x): j for j, x in enumerate(pool_f)}
    frr_from = [[[n for n in rerank.ORDER if memb_r[n][i, pix_f[x]]] for x in flists["step-7 reranker"][i]] for i in range(nf)]
    print("final reranker items outside every top-50 list:", sum(1 for r in frr_from for b in r if not b))

    # ---- history: completed airings (view_pct >= 0.8) by category, weeks 11-20 and 21-30 (counts only)
    show_users = set(vusers) | set(tmpl)
    hist = {u: {"11-20": {}, "21-30": {}} for u in show_users}
    cat_of = titles["tv_show_category"]
    for per in ("11-20", "21-30"):
        a = data.airings([per])
        a = a[a["view_pct"] >= 0.8]
        a = a.assign(cat=cat_of.reindex(a["tv_show_id"]).values)
        a = a[a["cat"].notna()]
        g = a.groupby(["user_id", "cat"]).size()
        for (u, c), n in g.items():
            if str(int(u)) in hist:
                hist[str(int(u))][per][c] = int(n)

    ttl = titles["tv_show_title"]
    pool_ids = [int(x) for x in pool_f]
    ids_used |= set(pool_ids)
    ex_md = (OUT / "examples.md").read_text(encoding="utf-8")
    ex = {t: re.search(rf"## User {t}: (\d+)", ex_md).group(1) for t in "AB"}
    W = {"meta": {"note": "derived from competition data; generated by src/export_web.py", "methods": METHODS, "categories": cat_list,
                  "n_report_users": len(vusers), "n_pool_validation": int(n_p)},
         "programmes": {str(i): [str(ttl.get(i, "?")), str(cat_of.get(i, "?"))] for i in sorted(ids_used)},
         "examples": ex,
         "validation": {"map_python": vmap, "users": vusers},
         "final": {"template": tmpl, "methods": METHODS, "lists": flists, "final_step7_all": flists["step-7 reranker"], "seen": fseen, "rr_from": frr_from},
         "history": hist, "pool_final": pool_ids, "kaggle": parse_kaggle(), "handcheck": parse_handcheck()}
    WEB.mkdir(parents=True, exist_ok=True)
    p_out = WEB / "web_data.json"
    p_out.write_text(json.dumps(W, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {p_out}: {p_out.stat().st_size / 1e6:.2f} MB; REPORT users {len(vusers)}; MAP@5 python:", {m: round(v, 4) for m, v in vmap.items()})


if __name__ == "__main__":
    main()

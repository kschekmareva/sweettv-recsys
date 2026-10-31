"""Writes outputs/examples.md (two example users, all methods, validation and final run) and outputs/handcheck.md (material for a hand check).
No model is changed or re-tuned; the step-7 reranker is rebuilt exactly as in step 7. Usage: python src/make_examples.py
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import itertools
import json

import numpy as np
import pandas as pd

import cbf
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

SEED, K = 42, 5
VARS = s5.VARS
OUT = data.ROOT / "outputs"
SUBDIR = OUT / "submissions"
METHODS = ["P", "P-air", "H2", "CBF", "MF", "PPR", "AR", "step-6 blend", "step-7 reranker"]
NEWS = "Инфо"


def main():
    J5 = json.loads((OUT / "validation_step5.json").read_text(encoding="utf-8"))
    J7 = json.loads((OUT / "validation_step7.json").read_text(encoding="utf-8"))
    cfg = {"h2_c": J5["h2"]["c_best"], "mf": J5["mf"]["chosen"], "ppr_alpha": J5["ppr"]["alpha"], "ar_mode": "max"}
    ew_cfg = (J7["ew"]["chosen"]["m"], J7["ew"]["chosen"]["c"])
    w6 = dict(J5["rows"]["P-air+MF+PPR+H2"]["weights"]["VTOP5"])

    # ================================================================== validation setup (identical to step 7)
    views = data.load_views()
    uc_pairs = views[["user_id", "vsetv_id", "iso_week"]].drop_duplicates()
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
    sub_rows = uix.get_indexer(np.asarray(sub_users, dtype=np.uint64))
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
    # step-7 reranker: fold model fitted on TUNE users only (the REPORT users were not seen)
    cand = rerank.candidates(store, ctx["pop_norm"], n_u, n_p, names=rerank.ORDER)
    ri, ci, X, _ = rerank.make_features(ctx, store, ctx["pop_norm"], cand)
    y = T5[ri, ci].astype(int)
    tune_truth = tune_rows[ne5[tune_rows]]
    p = np.random.default_rng(SEED).permutation(tune_truth)
    n_va = int(round(0.2 * len(p)))
    model = rerank2.train(X, y, ri, np.sort(p[n_va:]), np.sort(p[:n_va]), rerank2.BASE_PARAMS)
    top_rr = rerank.top5(model, X, ri, ci, report_rows)
    rr_tops = np.zeros((n_u, K), dtype=int)
    rr_tops[report_rows] = top_rr
    tops["step-7 reranker"] = rr_tops
    chk = metrics.ap5_matrix(top_rr, T5[report_rows])[ne5[report_rows]].mean()
    assert abs(chk - J7["rows"]["reranker"]["VTOP5"]["report"]) < 1e-9, chk          # reproduces the step-7 report

    # ================================================================== choice of the two users (REPORT half and submission users, so a final-run list exists)
    cands = np.array(sorted(set(report_rows) & set(sub_rows[sub_rows >= 0]) & set(np.flatnonzero(ne5))))
    a_ri, a_ci, a_cat = ctx["air_ri"], ctx["air_ci"], ctx["air_cat"]
    done = ctx["air"]["view_pct"].values >= 0.8
    news_code = cat_list.index(NEWS)
    stats = []
    for r in cands:
        m = (a_ri == r) & done
        cats = a_cat[m]
        n = int(m.sum())
        news = float((cats == news_code).mean()) if n else 1.0
        non = pd.Series(cats[cats != news_code]).value_counts()
        stats.append({"row": int(r), "completed": n, "news_share": news, "top_non_news_cat": cat_list[int(non.index[0])] if len(non) else "", "top_non_news_share": float(non.iloc[0] / n) if len(non) and n else 0.0})
    st = pd.DataFrame(stats)
    heavy = st[(st["completed"] >= st["completed"].quantile(0.75)) & (st["news_share"] <= 0.2)]
    rowA = int(heavy.sort_values(["top_non_news_share", "completed"], ascending=False).iloc[0]["row"])
    lists = {r: {m: [int(pool[x]) for x in tops[m][r]] for m in METHODS} for r in cands}
    dis = {r: np.mean([1 - len(set(lists[r][a]) & set(lists[r][b])) / len(set(lists[r][a]) | set(lists[r][b])) for a, b in itertools.combinations(METHODS, 2)]) for r in cands if r != rowA}
    rowB = int(max(dis, key=lambda r: (dis[r], -r)))
    rows_ex = {"A": rowA, "B": rowB}
    ttl = titles["tv_show_title"]
    cat = titles["tv_show_category"]
    val_done = air_val[air_val["view_pct"] >= 0.8].groupby(["user_id", "tv_show_id"]).size()
    tr_done = pd.DataFrame({"u": ctx["air_ri"][done & (ctx["air_ci"] >= 0)], "s": ctx["pool"][ctx["air_ci"][done & (ctx["air_ci"] >= 0)]]}).groupby(["u", "s"]).size()

    def fmt(i):
        return f"{i} — {ttl.get(i, '?')} — {cat.get(i, '?')}"

    # ================================================================== final-run lists for the same users (TRAIN weeks 11-30, pool 31-42)
    users_f = data.load_submission_users()
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

    # ================================================================== examples.md
    md = ["# Two example users", "",
          "Both users are in the REPORT half of the validation users (not used for tuning) **and** among the 1,260 submission users, so a final-run list exists for them. "
          "Validation = features from weeks 11-20, truth from weeks 21-30 (VTOP5: the user's top-5 programmes by number of airings watched to >= 80%). "
          "Pure methods fall back to / break ties with P-air. The reranker row is the fold model fitted on the TUNE users only, i.e. it did not see these users. "
          "HIT marks an id that is in the user's VTOP5 truth.", "",
          "Selection rules (deterministic, no peeking at scores): **user A** = among those users, completed-airing count (view_pct >= 0.8, weeks 11-20) in the top quarter, "
          f"at most 20% of the completed airings in the category '{NEWS}' (news/info), and then the largest share of completed airings in a single non-news category (clear taste). "
          "**user B** = among the others, the largest mean pairwise Jaccard distance between the Top-5 lists of the nine methods (the methods disagree most).", ""]
    for tag, r in rows_ex.items():
        uid = users[r]
        truth = [int(s) for s in vpairs["VTOP5"][vpairs["VTOP5"]["user_id"] == np.uint64(uid)]["tv_show_id"]]
        tset = set(truth)
        s_ = st[st["row"] == r].iloc[0]
        md += [f"## User {tag}: {uid}", "",
               f"TRAIN (weeks 11-20): {int(s_['completed'])} completed airings; share in '{NEWS}' {s_['news_share']:.2f}; most completed non-news category: {s_['top_non_news_cat']} ({s_['top_non_news_share']:.2f} of completed airings). "
               f"Mean Jaccard distance between the nine methods' lists: {np.mean([1 - len(set(lists[r][a]) & set(lists[r][b])) / len(set(lists[r][a]) | set(lists[r][b])) for a, b in itertools.combinations(METHODS, 2)]):.2f}.", "",
               "### VTOP5 truth (weeks 21-30)", "", "| rank | tv_show_id | title | category | completed airings in weeks 21-30 | completed airings in weeks 11-20 |", "|---|---|---|---|---|---|"]
        for k, i in enumerate(truth, 1):
            md.append(f"| {k} | {i} | {ttl.get(i, '?')} | {cat.get(i, '?')} | {int(val_done.get((np.uint64(uid), i), 0))} | {int(tr_done.get((r, i), 0))} |")
        md += ["", "### Validation Top-5 per method and AP@5", "", "| method | AP@5 | Top-5 (HIT = in the truth) |", "|---|---|---|"]
        for m in METHODS:
            lst = lists[r][m]
            md.append(f"| {m} | {metrics.ap_at_k(lst, tset, K):.4f} | " + "<br>".join(f"{k}. {'**HIT** ' if i in tset else ''}{fmt(i)}" for k, i in enumerate(lst, 1)) + " |")
        md += ["", "### Final-run Top-5 per method (TRAIN weeks 11-30, pool of weeks 31-42; no truth exists)", "",
               "The reranker row is the final model `submission_STEP7_all.csv` (features from weeks 21-30); the other rows use the history of weeks 11-30.", "",
               "| method | Top-5 |", "|---|---|"]
        fi = users_f.index(uid)
        for m in METHODS:
            lst = [int(x) for x in final_file[str(uid)].split()] if m == "step-7 reranker" else [int(pool_f[x]) for x in fin[m][fi]]
            md.append(f"| {m} | " + "<br>".join(f"{k}. {fmt(i)}" for k, i in enumerate(lst, 1)) + " |")
        md.append("")
    (OUT / "examples.md").write_text("\n".join(md), encoding="utf-8")

    # ================================================================== handcheck.md for user A
    uidA = users[rowA]
    truthA = [int(s) for s in vpairs["VTOP5"][vpairs["VTOP5"]["user_id"] == np.uint64(uidA)]["tv_show_id"]]
    listA = lists[rowA]["step-7 reranker"]
    apA = metrics.ap_at_k(listA, set(truthA), K)
    raw = pd.concat([c[c["user_id"] == str(uidA)] for c in pd.read_csv(data.DATA / data.VIEW_FILE, dtype=str, chunksize=1_000_000)])
    raw["s"] = pd.to_datetime(raw["start_time"], format="%Y-%m-%d %H:%M:%S")
    raw["e"] = pd.to_datetime(raw["stop_time"], format="%Y-%m-%d %H:%M:%S")
    raw = raw[(raw["s"] >= "2020-05-18") & (raw["s"] < "2020-07-27")].sort_values("s")
    sch = pd.read_csv(data.DATA / "export_arh_21-30-final.csv", dtype=str, usecols=["channel_id", "channel_title", "start_time", "duration", "tv_show_title", "tv_show_id"])
    sch["s"] = pd.to_datetime(sch["start_time"], format="%d.%m.%Y %H:%M:%S")
    sch["e"] = sch["s"] + pd.to_timedelta(sch["duration"].astype(int), unit="s")
    pick = None
    for _, v in raw.iterrows():
        o = sch[(sch["channel_id"] == v["vsetv_id"]) & (sch["s"] < v["e"]) & (sch["e"] > v["s"])]
        if len(o) in (2, 3) and (o["tv_show_id"] != "0").any() and 900 <= int(v["duraton"]) <= 3600:
            pick = (v, o)
            break
    assert pick is not None
    v, o = pick
    vv = data.load_views()
    one = vv[(vv["user_id"] == np.uint64(uidA)) & (vv["vsetv_id"] == int(v["vsetv_id"])) & (vv["start"] == int((v["s"] - pd.Timestamp("2020-01-01")).total_seconds())) &
             (vv["stop"] == int((v["e"] - pd.Timestamp("2020-01-01")).total_seconds()))].reset_index(drop=True)
    vidx, show, ov, air, adur, achan, astart, _ = data.join_views_schedule(one, data.load_schedule("21-30"))
    code = {(int(c), int(s_)): (int(x), int(sh)) for c, s_, x, sh in zip(achan, astart, ov, show)}
    ad = air_val[air_val["user_id"] == np.uint64(uidA)].set_index(["channel_id", "start"])
    hc = ["# Hand-check material (user A)", "",
          "Only inputs and the code's final numbers are shown; no intermediate arithmetic.", "",
          f"## (a) Step-7 reranker on validation for user A ({uidA})", "",
          f"- Top-5 (tv_show_id, in rank order): {listA}", f"- VTOP5 truth (tv_show_id, rank order of the truth): {truthA}",
          f"- AP@5 computed by the code: **{apA:.6f}**", "",
          "## (b) One raw view row and the schedule rows it overlaps", "",
          "Raw view row (data/dataset11-30.csv, columns user_id, vsetv_id, start_time, stop_time, duraton):", "",
          "```", f"{v['user_id']},{v['vsetv_id']},{v['start_time']},{v['stop_time']},{v['duraton']}", "```", "",
          "Schedule rows of data/export_arh_21-30-final.csv on channel_id = vsetv_id that overlap the view interval "
          "(columns channel_id, channel_title, start_time, duration in seconds, tv_show_id, tv_show_title), with the code's results:", "",
          "| channel_id | channel_title | start_time | duration | tv_show_id | tv_show_title | overlap seconds with this view (code) | view_pct of this airing for the user (code) |", "|---|---|---|---|---|---|---|---|"]
    for _, r_ in o.iterrows():
        key = (int(r_["channel_id"]), int((r_["s"] - pd.Timestamp("2020-01-01")).total_seconds()))
        ovs = code.get(key, (None, None))[0]
        vp = float(ad.loc[key, "view_pct"]) if (key in ad.index and r_["tv_show_id"] != "0") else None
        hc.append(f"| {r_['channel_id']} | {r_['channel_title']} | {r_['start_time']} | {r_['duration']} | {r_['tv_show_id']} | {r_['tv_show_title']} | {ovs} | "
                  + (f"{vp:.6f}" if vp is not None else "n/a (tv_show_id = 0, not an item)") + " |")
    hc += ["", "Note: view_pct is computed per airing from all of the user's views of that airing (overlap seconds summed over those views, capped at the airing duration), "
               "so it can exceed what this single view contributes."]
    (OUT / "handcheck.md").write_text("\n".join(hc) + "\n", encoding="utf-8")
    print("A =", uidA, "B =", users[rowB], "AP@5 (reranker, A) =", round(apA, 6))


if __name__ == "__main__":
    main()

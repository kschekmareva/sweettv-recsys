"""Step 2: temporal validation of baselines. Usage: python src/run_validation.py

Writes outputs/protocol.md, outputs/validation_step2.md, outputs/validation_step2.json.
`final_run(method, out_path)` is the TRAIN = weeks 11-30 pipeline that writes a submission-format CSV.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import baselines
import data
import metrics

OUT = data.ROOT / "outputs"
THR = 300
K = 5

PROTOCOL = f"""# Protocol (step 2)

1. **Times.** View log and schedules are used exactly as given (no timezone shift). Schedule end = start_time + duration seconds.
   user_id is read as uint64 (never float).
2. **Join.** A view matches a schedule row when vsetv_id == channel_id and the intervals overlap;
   overlap_sec = min(stop_view, end_sched) - max(start_view, start_sched) > 0. View weeks 11-20 use export_arh_11-20,
   weeks 21-30 use export_arh_21-30 (period = ISO week of the view's start_time). Vectorised per-channel interval search
   (src/data.py). Rows with tv_show_id == 0 are unlabelled airtime and are dropped (share of matched seconds reported).
   Views on channels absent from the schedule are dropped (counts reported).
3. **Interaction table** per (period, user_id, tv_show_id): total overlap_sec, n_views (distinct views touching the programme),
   n_days (distinct view start days); cached at outputs/cache/interactions.pkl. A pair is POSITIVE when total overlap_sec >= {THR}
   (summed over the periods used).
4. **Split.** TRAIN = views of ISO weeks 11-20, VALIDATION = views of ISO weeks 21-30. Models see TRAIN only.
   Validation truth per user = set of positive tv_show_ids in weeks 21-30.
5. **Pool (validation)** = distinct labelled tv_show_ids with a schedule row (file 21-30) starting in ISO weeks 21-30 (Monday tail of
   week 31 excluded). Recommending something the user already watched in TRAIN is allowed (all methods). Every list has exactly {K}
   distinct ids from the pool; short lists are filled from the popularity ranking.
6. **Metric.** MAP@{K}: AP@{K} = sum_k P@k * rel(k) / min({K}, |truth|); empty truth -> 0. Reported (a) over users with non-empty
   validation truth, (b) over the 1,260 submission users (empty truth counts 0).
7. **Baselines.** (P) popularity = number of distinct TRAIN users with a positive pair, restricted to the pool, ties by ascending
   tv_show_id, same list for everyone. (H) own-history diagnostic = user's own TRAIN positives in the pool ranked by total overlap
   seconds (ties: ascending id), filled with (P).
8. **Final run.** TRAIN = weeks 11-30 (periods summed, positive if total >= {THR} s), pool = labelled programmes with schedule
   rows starting in ISO weeks 31-42 of export_arh_31-42. No viewing data from weeks 31-42 is used.
"""


def users_to_truth(pos):
    return {int(u): set(int(x) for x in g["tv_show_id"]) for u, g in pos.groupby("user_id")}


def hand_check(preds, truth, user, ttl, label):
    lines = [f"**Method {label}** top-5:"]
    for i, p in enumerate(preds[user], 1):
        t = ttl.loc[p] if p in ttl.index else None
        lines.append(f"  {i}. id={p} | {t['tv_show_title'] if t is not None else '?'} | {t['tv_show_category'] if t is not None else '?'}")
    pred = preds[user]
    t = truth[user]
    rel = metrics.rel_vector(pred, t, K)
    lines.append(f"  truth set ({len(t)}): {sorted(t)}")
    lines.append(f"  rel vector: {rel}")
    hits, total, steps = 0, 0.0, []
    for k, r in enumerate(rel, 1):
        hits += r
        pk = hits / k
        steps.append(f"k={k}: hits={hits}, P@{k}={hits}/{k}={pk:.4f}, rel={r} -> {pk * r:.4f}")
        total += pk * r
    lines.extend("  " + s for s in steps)
    denom = min(K, len(t))
    lines.append(f"  AP@5 = {total:.4f} / min(5,{len(t)})={denom} = {total / denom:.6f}  "
                 f"(function: {metrics.ap_at_k(pred, t, K):.6f})")
    return lines


def write_submission(preds, users, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("user_id,tv_show_id\n")
        for u in users:
            assert len(preds[u]) == K and len(set(preds[u])) == K
            f.write(f"{u}," + " ".join(str(x) for x in preds[u]) + "\n")


def final_run(method, out_path, thr=THR):
    """TRAIN = weeks 11-30, pool = labelled programmes airing in ISO weeks 31-42. Writes the template format."""
    inter, _ = data.build_interactions()
    train_pos = data.positives(inter, list(data.PERIODS), thr)
    pool = data.pool_ids(data.load_schedule("31-42"), 31, 42)
    users = data.load_submission_users()
    preds = baselines.METHODS[method](train_pos, pool, users, K)
    write_submission(preds, users, out_path)
    return preds, pool


def main():
    t_all = time.time()
    tm = {}
    R = {"threshold_sec": THR, "k": K}
    md = ["# Validation, step 2", ""]

    print("metric tests", flush=True)
    R["metric_tests"] = metrics.run_tests()

    t = time.time()
    inter, jstats = data.build_interactions()
    tm["interactions_load_or_build_s"] = round(time.time() - t, 1)
    R["join"] = jstats

    sub_users = data.load_submission_users()
    train_pos = data.positives(inter, ["11-20"], THR)
    val_pos = data.positives(inter, ["21-30"], THR)
    sched_val = data.load_schedule("21-30")
    pool = data.pool_ids(sched_val, 21, 30)
    pool_set = set(int(x) for x in pool)
    ttl = data.titles()

    # ---- diagnostics
    truth_all = users_to_truth(val_pos)
    val_in_pool = val_pos["tv_show_id"].isin(pool)
    kt = val_pos.merge(train_pos[["user_id", "tv_show_id"]].assign(rep=1), on=["user_id", "tv_show_id"], how="left")
    train_users = set(int(u) for u in train_pos["user_id"])
    train_pool_users = set(int(u) for u in train_pos.loc[train_pos["tv_show_id"].isin(pool), "user_id"])
    eval_a = sorted(truth_all)
    cold = lambda us, ref: sum(u not in ref for u in us)
    sub_truth_nonempty = [u for u in sub_users if u in truth_all]
    D = {
        "pool_size": len(pool),
        "train_positive_pairs": len(train_pos), "train_positive_users": len(train_users),
        "val_positive_pairs": len(val_pos), "val_positive_users": len(truth_all),
        "val_positive_pairs_in_pool": int(val_in_pool.sum()),
        "share_val_positives_in_pool": float(val_in_pool.mean()),
        "repeat_rate_val_positive_also_train_positive": float(kt["rep"].notna().mean()),
        "repeat_pairs": int(kt["rep"].notna().sum()),
        "repeat_rate_among_users_with_train_positive": float(
            kt.loc[kt["user_id"].isin(list(train_users)), "rep"].notna().mean()),
        "val_pos_per_user_quantiles": {str(q): float(v) for q, v in
                                       val_pos.groupby("user_id").size().quantile([.1, .25, .5, .75, .9]).items()},
        "train_pos_per_user_quantiles": {str(q): float(v) for q, v in
                                         train_pos.groupby("user_id").size().quantile([.1, .25, .5, .75, .9]).items()},
        "eval_a_users": len(eval_a),
        "eval_a_cold_no_train_positive": cold(eval_a, train_users),
        "eval_a_cold_no_train_positive_in_pool": cold(eval_a, train_pool_users),
        "submission_users": len(sub_users),
        "submission_users_nonempty_truth": len(sub_truth_nonempty),
        "submission_cold_no_train_positive": cold(sub_users, train_users),
        "submission_cold_no_train_positive_in_pool": cold(sub_users, train_pool_users),
        "submission_cold_with_nonempty_truth": cold(sub_truth_nonempty, train_users),
    }
    R["diagnostics"] = D

    # ---- baselines
    users_union = sorted(set(eval_a) | set(sub_users))
    res, preds_all = {}, {}
    for name, fn in baselines.METHODS.items():
        t = time.time()
        preds = fn(train_pos, pool, users_union, K)
        assert all(len(p) == K and len(set(p)) == K and set(p) <= pool_set for p in preds.values())
        preds_all[name] = preds
        hist_users = [u for u in eval_a if u in train_pool_users]
        cold_users = [u for u in eval_a if u not in train_pool_users]
        res[name] = {
            "MAP5_a_users_with_truth": metrics.map_at_k(preds, truth_all, eval_a, K),
            "MAP5_b_submission_users": metrics.map_at_k(preds, truth_all, sub_users, K),
            "MAP5_a_subset_with_in_pool_train_history": metrics.map_at_k(preds, truth_all, hist_users, K) if hist_users else None,
            "MAP5_a_subset_without_in_pool_train_history": metrics.map_at_k(preds, truth_all, cold_users, K) if cold_users else None,
            "n_a_with_history": len(hist_users), "n_a_without_history": len(cold_users),
            "seconds": round(time.time() - t, 2),
        }
    R["results"] = res
    rank = baselines.pop_ranking(train_pos, pool)
    R["popularity_top10"] = [
        {"tv_show_id": int(r.tv_show_id), "n_train_users": int(r.n_users),
         "title": str(ttl.loc[r.tv_show_id, "tv_show_title"]), "category": str(ttl.loc[r.tv_show_id, "tv_show_category"])}
        for r in rank.head(10).itertuples()]

    # ---- hand check
    cand = [u for u in sub_users if len(truth_all.get(u, ())) >= 5 and u in train_pool_users]
    hc_lines = []
    if cand:
        u = cand[0]
        hc_lines.append(f"Submission user {u} (first in template order with >=5 validation positives and >=1 in-pool TRAIN positive); "
                        f"validation positives: {len(truth_all[u])}")
        for name in ["P", "H"]:
            hc_lines += hand_check(preds_all[name], truth_all, u, ttl, name)
    R["hand_check_user"] = cand[0] if cand else None
    R["hand_check_lines"] = hc_lines
    print("\n".join(hc_lines), flush=True)

    # ---- final-run format self-test (not a submission)
    chk = data.CACHE / "_final_run_format_check_P.csv"
    t = time.time()
    fp, fpool = final_run("P", chk)
    back = pd.read_csv(chk, dtype=str)
    tmpl = pd.read_csv(data.DATA / data.SUB_FILE, dtype=str)
    R["final_run_check"] = {
        "file": str(chk.relative_to(data.ROOT)), "pool_size_weeks_31_42": int(len(fpool)),
        "header_equals_template": list(back.columns) == list(tmpl.columns),
        "rows": int(len(back)), "user_order_equals_template": bool((back["user_id"] == tmpl["user_id"]).all()),
        "all_rows_five_distinct_ids": bool(back["tv_show_id"].map(lambda s: len(set(s.split(" "))) == K).all()),
        "all_ids_in_final_pool": bool(back["tv_show_id"].map(lambda s: set(map(int, s.split(" "))) <= set(map(int, fpool))).all()),
        "train_positive_pairs_weeks_11_30": int(len(data.positives(inter, list(data.PERIODS), THR))),
        "seconds": round(time.time() - t, 1),
    }
    tm["total_s"] = round(time.time() - t_all, 1)
    R["runtime"] = tm

    # ---- write
    (OUT / "protocol.md").write_text(PROTOCOL, encoding="utf-8")
    md += ["## Metric unit tests", ""] + [f"- {'OK' if c['ok'] else 'FAIL'}: {c['case']}: AP@5={c['ap5']:.6f} (expected {c['expected']:.6f})"
                                          for c in R["metric_tests"]] + [""]
    md += ["## Join (view -> schedule), per period", ""]
    for key in data.PERIODS:
        s = jstats[key]
        md += [f"### views of weeks {key}", "",
               f"- views: {s['views']:,}; distinct users {s['distinct_users_in_views']:,}",
               f"- dropped, channel absent from schedule: {s['views_channel_absent_from_schedule']:,} views "
               f"({s['views_channel_absent_from_schedule'] / s['views']:.4f}), {s['distinct_vsetv_id_absent']} distinct vsetv_id, "
               f"{s['view_seconds_channel_absent']:,} s of {s['view_seconds']:,} view seconds",
               f"- channel present but no overlapping schedule row: {s['views_channel_present_no_overlap']:,} views",
               f"- views with any overlap: {s['views_with_any_overlap']:,}; with a labelled overlap: {s['views_with_labelled_overlap']:,} "
               f"({s['views_with_labelled_overlap'] / s['views']:.4f} of all views)",
               f"- matched seconds: {s['matched_seconds_all']:,}; on tv_show_id == 0: {s['matched_seconds_unlabelled_id0']:,} "
               f"(**{s['share_matched_seconds_unlabelled']:.4f}**); on labelled: {s['matched_seconds_labelled']:,}",
               f"- labelled user-programme pairs (interaction rows): {s['interaction_pairs_labelled']:,}", ""]
    md += ["## Diagnostics", ""] + [f"- {k}: {v}" for k, v in D.items()] + [""]
    md += [f"## Results (threshold {THR} s, k={K}, pool = {len(pool)} programmes airing in weeks 21-30)", "",
           "| method | MAP@5 (a) users with truth | MAP@5 (b) 1,260 submission users | (a) with in-pool TRAIN history | (a) without |",
           "|---|---|---|---|---|"]
    names = {"P": "P popularity", "H": "H own-history"}
    f4 = lambda x: "n/a" if x is None else f"{x:.4f}"
    for k, r in res.items():
        md.append(f"| {names[k]} | {r['MAP5_a_users_with_truth']:.4f} (n={len(eval_a)}) | {r['MAP5_b_submission_users']:.4f} (n={len(sub_users)}) | "
                  f"{f4(r['MAP5_a_subset_with_in_pool_train_history'])} (n={r['n_a_with_history']}) | "
                  f"{f4(r['MAP5_a_subset_without_in_pool_train_history'])} (n={r['n_a_without_history']}) |")
    md += ["", "Popularity top-10 (distinct TRAIN users with a positive pair):", ""]
    md += [f"- {r['tv_show_id']} | {r['title']} | {r['category']} | {r['n_train_users']} users" for r in R["popularity_top10"]]
    md += ["", "## Hand check", "", "```"] + hc_lines + ["```", "",
           "## Final-run format self-test (not a submission)", ""] + [f"- {k}: {v}" for k, v in R["final_run_check"].items()]
    md += ["", "## Runtime", ""] + [f"- {k}: {v}" for k, v in tm.items()]
    md.append(f"- interactions build (when not cached): {jstats.get('build_seconds')} s")
    (OUT / "validation_step2.md").write_text("\n".join(md), encoding="utf-8")

    def _d(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return str(o)
    (OUT / "validation_step2.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=_d), encoding="utf-8")
    print(f"done in {tm['total_s']} s")


if __name__ == "__main__":
    main()

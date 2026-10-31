"""Packaging: requirements.txt, outputs/environment.md, outputs/report_tables.md/.csv and the Kaggle columns of SUBMISSIONS.md.
All numbers are copied from the step reports (outputs/*.json); nothing is recomputed. Usage: python src/make_docs.py
"""
import ast
import json
import os
import platform
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
SUB = OUT / "submissions"
L = lambda name: json.loads((OUT / name).read_text(encoding="utf-8"))
f4 = lambda x: "n/a" if x is None else f"{x:.4f}"

KAGGLE = {  # provided by the project owner (public, private)
    "submission_P.csv": ("0.28234", "0.27809"),
    "submission_FINAL_vtop5.csv": ("0.41541", "0.40202"),
    "submission_STEP7.csv": ("0.42744", "0.41984"),
    "submission_STEP7_all.csv": ("0.42914", "0.42244"),
}


# ------------------------------------------------------------------ requirements.txt and environment.md
def used_packages():
    local = {p.stem for p in (ROOT / "src").glob("*.py")}
    dist = {"sklearn": "scikit-learn"}
    names = set()
    for p in list((ROOT / "src").glob("*.py")) + [ROOT / "download.py"]:
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module and node.level == 0 else []
            for m in mods:
                top = m.split(".")[0]
                if top not in sys.stdlib_module_names and top not in local:
                    names.add(dist.get(top, top))
    return sorted(names)


def write_requirements():
    pins = [f"{n}=={metadata.version(n)}" for n in used_packages()]
    (ROOT / "requirements.txt").write_text("\n".join(pins) + "\n", encoding="utf-8")
    return pins


def runtimes():
    rt = []
    rt.append(("1 EDA (src/eda.py)", L("eda.json")["runtime_seconds"]))
    rt.append(("2 interactions, protocol, baselines (src/run_validation.py; includes building the join cache)", L("validation_step2.json")["runtime"]["total_s"]))
    rt.append(("3 MF, CBF, hybrids (src/run_step3.py)", L("validation_step3.json")["runtime"]["total_s"]))
    rt.append(("4 truth variants, PPR, AR, hybrids (src/run_step4.py)", L("validation_step4.json")["runtime"]["total_s"]))
    rt.append(("5 VTOP5, H2, pool option, blends (src/run_step5.py)", L("validation_step5.json")["runtime"]["total_s"]))
    rt.append(("6 decision rule, VTOP5-tuned blend (src/run_step6.py)", L("validation_step6.json")["runtime_s"]))
    rt.append(("7 ceiling, EW, reranker (src/run_step7.py)", L("validation_step7.json")["runtime_s"]))
    rt.append(("8 reranker improvements, 2-fold swap (src/run_step8.py)", L("validation_step8.json")["runtime_s"]))
    rt.append(("8b seed bagging, rank fusion (src/run_step8b.py)", L("validation_step8b.json")["runtime_s"]))
    rt.append(("10 (8c) H2 decay grid, multi-scale features, learning curve (src/run_step10.py)", L("validation_step10.json")["runtime_s"]))
    rt.append(("11 final refit on all validation users (src/run_step11.py)", L("validation_step11.json")["runtime_s"]))
    rep = OUT / "reproducibility.json"
    if rep.exists():
        r = json.loads(rep.read_text(encoding="utf-8"))
        rt.append(("9 reproduction of the final file, with cache rebuild (src/make_final.py --rebuild)", r["make_final_seconds_with_rebuild"]))
    return rt


def write_environment(pins):
    try:
        import psutil
        ram = f"{psutil.virtual_memory().total / 2**30:.2f} GB"
    except ImportError:
        ram = "n/a"
    md = ["# Environment and runtimes", "",
          f"- OS: {platform.platform()} (Windows 10 Home Single Language, build 10.0.19045)",
          f"- Python: {sys.version.split()[0]} ({platform.python_implementation()})",
          f"- CPU: {platform.processor()}, {os.cpu_count()} logical cores (LightGBM and PPR use 4 threads; BLAS is limited to 1 thread)",
          f"- RAM: {ram}",
          "- Packages (exact versions, also in requirements.txt): " + ", ".join(pins), "",
          "Runtimes copied from the step reports (seconds, wall clock on this machine):", "", "| step | seconds |", "|---|---|"]
    md += [f"| {n} | {s:.1f} |" for n, s in runtimes()]
    md += ["", "The join cache (outputs/cache/interactions_v3.pkl) is rebuilt from ./data in about 80 s; later steps reuse it."]
    (OUT / "environment.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ report tables
def method_rows():
    J4, J5, J7 = L("validation_step4.json"), L("validation_step5.json"), L("validation_step7.json")
    g5 = lambda n, v: J5["rows"][n]["metrics"][v]["report"]
    g4 = lambda n, v: J4["rows"][n]["metrics"][v]["report"]
    g7 = lambda n, v: J7["rows"][n][v]["report"]
    return [
        ("P (popularity)", "step 5", g5("P", "VTOP5"), g5("P", "V300")),
        ("P-rec (last 4 TRAIN weeks)", "step 4 (VTOP5 not computed)", None, g4("P-rec", "V300")),
        ("P-air (popularity x log1p(target airings))", "step 5", g5("P-air", "VTOP5"), g5("P-air", "V300")),
        ("H (own history, step 2)", "step 5", g5("H", "VTOP5"), g5("H", "V300")),
        ("H2 (completed airings, decay)", "step 5", g5("H2", "VTOP5"), g5("H2", "V300")),
        ("CBF (content-based)", "step 4 (VTOP5 not computed)", None, g4("CBF", "V300")),
        ("MF (ALS, alpha 0.3, pure)", "step 5", g5("MF", "VTOP5"), g5("MF", "V300")),
        ("PPR (alpha 0.3, pure)", "step 5", g5("PPR", "VTOP5"), g5("PPR", "V300")),
        ("AR (rules, max, pure)", "step 5", g5("AR", "VTOP5"), g5("AR", "V300")),
        ("step-4 blend (P-air 0.2, MF 0.4, PPR 0.4)", "step 5", g5("step-4 blend", "VTOP5"), g5("step-4 blend", "V300")),
        ("step-6 blend (P-air 0.2, MF 0.2, PPR 0.1, H2 0.5)", "step 7", g7("step-6 blend", "VTOP5"), g7("step-6 blend", "V300")),
        ("step-7 reranker (LightGBM, fit on TUNE)", "step 7", g7("reranker", "VTOP5"), g7("reranker", "V300")),
        ("oracle (perfect ranking of the 6-component top-50 union)", "step 7", g7("oracle (6-component top-50 union)", "VTOP5"), g7("oracle (6-component top-50 union)", "V300")),
    ]


def write_tables():
    J2, J3, J5, J6, J7 = L("validation_step2.json"), L("validation_step3.json"), L("validation_step5.json"), L("validation_step6.json"), L("validation_step7.json")
    J8, J8b, J10, J11, E = L("validation_step8.json"), L("validation_step8b.json"), L("validation_step10.json"), L("validation_step11.json"), L("eda.json")
    rows = method_rows()
    (OUT / "report_tables.csv").write_text("method,source,VTOP5_REPORT,V300_REPORT\n" + "\n".join(
        f'"{m}","{s}",{"" if a is None else f"{a:.4f}"},{"" if b is None else f"{b:.4f}"}' for m, s, a, b in rows) + "\n", encoding="utf-8")
    md = ["# Report tables", "", "All numbers are copied from the step reports in outputs/ (no recomputation). REPORT = the 50% of validation users not used for tuning "
          "(2,400 users for V300, 2,334 with VTOP5 truth); validation = features from weeks 11-20, truth from weeks 21-30.", "",
          "## (a) Method comparison on the REPORT half", "", "| method | VTOP5 (primary) | V300 | source |", "|---|---|---|---|"]
    md += [f"| {m} | {f4(a)} | {f4(b)} | {s} |" for m, s, a, b in rows]
    # (b) Kaggle
    pv = {"submission_P.csv": ("P (global popularity)", J5["rows"]["P"]["metrics"]["VTOP5"]["report"]),
          "submission_FINAL_vtop5.csv": ("P-air + MF + PPR + H2 blend, weights tuned on VTOP5", J6["rows"]["step-6 FINAL (VTOP5-tuned w)"]["VTOP5"]["report"]),
          "submission_STEP7.csv": ("step-7 reranker (LightGBM, fit on TUNE users)", J7["rows"]["reranker"]["VTOP5"]["report"]),
          "submission_STEP7_all.csv": ("step-7 reranker refit on all validation users (FINAL)", None)}
    md += ["", "## (b) Kaggle results (entered by the project owner) and validation VTOP5", "", "| file | method | Kaggle public | Kaggle private | validation VTOP5 (REPORT) |", "|---|---|---|---|---|"]
    for fn, (pub, pri) in KAGGLE.items():
        md.append(f"| {fn} | {pv[fn][0]} | {pub} | {pri} | {f4(pv[fn][1]) if pv[fn][1] is not None else 'not scorable (no held-out users); same recipe on TUNE-fit folds: ' + f4(J11['fold_vtop5_reproduced']['A']) + ' / ' + f4(J11['fold_vtop5_reproduced']['B'])} |")
    diffs = {fn: float(KAGGLE[fn][1]) - pv[fn][1] for fn in KAGGLE if pv[fn][1] is not None}
    md += ["", "The three files that have a validation VTOP5 are ordered the same on validation, Kaggle public and Kaggle private (P < FINAL_vtop5 < STEP7). "
               "Kaggle private minus validation VTOP5 (REPORT): " + ", ".join(f"{fn} {d:+.4f}" for fn, d in diffs.items()) + ". "
               "This agreement was observed after the fact; no weight or hyperparameter was chosen from a Kaggle number.", ""]
    # (c) hyperparameters
    sys.path.insert(0, str(ROOT / "src"))
    import rerank
    import rerank2
    bp = {k: v for k, v in rerank2.BASE_PARAMS.items() if k not in ("verbose",)}
    md += ["## (c) Final hyperparameters per method", "", "| method | hyperparameters |", "|---|---|",
           "| P | distinct users with a positive pair (>= 300 s total overlap), restricted to the pool; ties by ascending tv_show_id |",
           "| P-rec | same as P, last 4 TRAIN weeks only (weeks 17-20 in validation, 27-30 in the final run) |",
           "| P-air | P x log1p(number of airings of the show in the target-period schedule; the schedule is used only to know what airs) |",
           "| H | own TRAIN positives in the pool ranked by total overlap seconds, filled with P |",
           f"| H2 | airings with view_pct >= 0.8; r = 5 if view_pct = 1 else 2 + 2*view_pct^2; decay c^(days_before_train_end/60), c = {J5['h2']['c_best']}; x log10(1 + airings of the show in the TRAIN schedule) |",
           f"| CBF | blocks {J3['cbf_features']['block_dims'] if 'cbf_features' in J3 else ''} (category, genre, TF-IDF people, decade), L2-normalised; user profile = positives weighted by log1p(overlap/60); cosine |",
           f"| MF | implicit ALS, {J5['mf']['chosen']['factors']} factors, reg {J5['mf']['chosen']['reg']}, alpha {J5['mf']['chosen']['alpha']}, 15 iterations, seed 42; weight log1p(overlap/60) over all labelled TRAIN pairs (steps 3/4 used alpha 10 / 1) |",
           f"| PPR | bipartite user-programme graph from TRAIN positives, edge weight log1p(overlap/60), alpha {J5['ppr']['alpha']}, power iteration tol 1e-6, max 100 iterations |",
           "| AR | item-to-item rules from TRAIN positive sets, confidence, user score = max over the user's positives, min support 20 users |",
           f"| EW | smoothed completion rate x target airings, m = {J7['ew']['chosen']['m']}, recency c = {J7['ew']['chosen']['c']} |",
           "| step-4 blend | per-user min-max fusion: P-air 0.2, MF (alpha 1) 0.4, PPR 0.4 |",
           f"| step-6 blend | per-user min-max fusion: P-air 0.2, MF {J6['weights_vtop5_tuned']['MF']}, PPR {J6['weights_vtop5_tuned']['PPR']}, H2 {J6['weights_vtop5_tuned']['H2']} (weights tuned on TUNE under VTOP5) |",
           f"| step-7 reranker (FINAL) | LightGBM lambdarank, binary labels = VTOP5 truth, candidates = union of the top-{rerank.TOPN} of {', '.join(rerank.ORDER)}, {len(J7['reranker']['features'])} features, params {bp}; 142 trees (mean best iteration of the two folds: {J11['fold_best_iterations']['A']} and {J11['fold_best_iterations']['B']}), fitted on all {J11['fit_users']:,} validation users |", ""]
    # (d) data / join statistics
    fl = E["files"]
    j2 = J2["join"]
    d2 = J2["diagnostics"]
    md += ["## (d) Data and join statistics", "", "| item | value |", "|---|---|",
           "| view log rows / distinct users | " + f"{fl['dataset11-30.csv']['rows']:,} / {E['view_log']['distinct_users']:,} |",
           "| schedule rows (weeks 11-20 / 21-30 / 31-42) | " + " / ".join(f"{fl[f]['rows']:,}" for f in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv")) + " |",
           "| share of schedule rows with tv_show_id != 0 (11-20 / 21-30 / 31-42) | " + " / ".join(f"{E['labelling'][k]['share_rows_labelled']:.3f}" for k in ("11-20", "21-30", "31-42")) + " |",
           "| submission users present in the view log | " + f"{E['submission']['users_in_view_log']:,} of {E['submission']['rows']:,} |"]
    for k in ("11-20", "21-30"):
        s = j2[k]
        md.append(f"| views of weeks {k}: total / channel absent from schedule / channel present but no overlapping row | {s['views']:,} / {s['views_channel_absent_from_schedule']:,} ({s['views_channel_absent_from_schedule'] / s['views']:.3f}) / {s['views_channel_present_no_overlap']:,} |")
        md.append(f"| views of weeks {k}: with a labelled overlap; share of matched seconds on tv_show_id = 0 | {s['views_with_labelled_overlap']:,} ({s['views_with_labelled_overlap'] / s['views']:.3f}); {s['share_matched_seconds_unlabelled']:.3f} |")
    md += [f"| interaction pairs (user, programme) weeks 11-20 / 21-30 | {j2['11-20']['interaction_pairs_labelled']:,} / {j2['21-30']['interaction_pairs_labelled']:,} |",
           f"| positive pairs (>= 300 s) TRAIN / VALIDATION | {d2['train_positive_pairs']:,} / {d2['val_positive_pairs']:,} |",
           f"| validation pool / final pool (labelled programmes airing in weeks 21-30 / 31-42) | {d2['pool_size']:,} / {J3['final']['pool_size']:,} |",
           f"| repeat rate (validation positives that were also TRAIN positives for the same user) | {d2['repeat_rate_val_positive_also_train_positive']:.3f} |",
           f"| cold start in validation (no TRAIN positive): users with truth / submission users | {d2['eval_a_cold_no_train_positive']} / {d2['submission_cold_no_train_positive']} |",
           f"| final file STEP7_all: submission users without data in weeks 21-30, per component (popularity fallback) | {J11['final']['users_without_component_data']} |",
           f"| VTOP5 truth: users / positive pairs / users with exactly 5 truth items | {J5['variant_stats']['VTOP5']['users_with_truth']:,} / {J5['variant_stats']['VTOP5']['positive_pairs']:,} / {J5['variant_stats']['VTOP5']['users_with_exactly_5_truth_items']:,} |", ""]
    # (e) steps 8, 8b, 10
    a8, t8b, t10 = J8["ablation"], J8b["table"], J10["table"]
    base = a8["S7: step-7 reranker"]["mean_VTOP5"]
    md += ["## (e) Improvement attempts after step 7 (2-fold swap, mean VTOP5 over the two folds; step-7 reranker = " + f4(base) + ")", "",
           "| step | what was tried | mean VTOP5 | change vs step 7 | shipped |", "|---|---|---|---|---|"]
    for n, a in a8.items():
        if n.startswith("S7"):
            continue
        md.append(f"| 8 | {n} (cumulative) | {f4(a['mean_VTOP5'])} | {a['mean_VTOP5'] - base:+.4f} | no |")
    for n, t in t8b.items():
        m = (t["A_VTOP5"] + t["B_VTOP5"]) / 2
        if n.startswith("step7"):
            continue
        md.append(f"| 8b | {n} | {f4(m)} | {m - base:+.4f} | no |")
    mm = t10["step7 + multi-scale features"]
    md.append(f"| 10 | step-7 + multi-scale recency features | {f4((mm['A_VTOP5'] + mm['B_VTOP5']) / 2)} | {(mm['A_VTOP5'] + mm['B_VTOP5']) / 2 - base:+.4f} | no |")
    h = J10["h2_grid"]["chosen"]
    md += ["", f"Step 8 pooled bootstrap of the full new reranker vs step 7 on VTOP5: {J8['bootstrap_new_vs_step7']['VTOP5']['pooled']['mean_diff']:+.4f} "
               f"[{J8['bootstrap_new_vs_step7']['VTOP5']['pooled']['ci95'][0]:+.4f}, {J8['bootstrap_new_vs_step7']['VTOP5']['pooled']['ci95'][1]:+.4f}]. "
               f"Step 10 H2 decay grid (c in 0.5/0.3/0.2/0.1 x T in 60/30/14 days): chosen c = {h['c']}, T = {h['T']}, i.e. unchanged.", "",
           "Step-10 learning curve (step-7 features, VTOP5 on the other fold; users in the training fold):", "", "| share of training-fold users | fold A | fold B |", "|---|---|---|"]
    for fr, d in J10["learning_curve"]["vtop5_by_fraction"].items():
        md.append(f"| {float(fr):.0%} ({d['A_users']} / {d['B_users']} users) | {f4(d['A'])} | {f4(d['B'])} |")
    md += ["", "100% > 50% in both folds, so by the pre-declared rule the final model was refit on all validation users (step 11, `submission_STEP7_all.csv`)."]
    (OUT / "report_tables.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ SUBMISSIONS.md: Kaggle columns
def update_submissions():
    p = SUB / "SUBMISSIONS.md"
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\| (submission_[^ ]+\.csv) \|(.*)\| \| \|$", line)
        if m:
            fn = m.group(1)
            pub, pri = KAGGLE.get(fn, ("not submitted", "not submitted"))
            line = f"| {fn} |{m.group(2)}| {pub} | {pri} |"
        out.append(line)
    text = "\n".join(out)
    if "Final submission" not in text:
        text = text.replace("Fill in the two Kaggle columns after each submission.",
                            "Kaggle columns filled by the project owner; files marked \"not submitted\" were never uploaded.\n\n"
                            "**Final submission (FINAL) = `submission_STEP7_all.csv`** (public 0.42914, private 0.42244).")
    p.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    pins = write_requirements()
    write_environment(pins)
    write_tables()
    update_submissions()
    print("requirements:", pins)

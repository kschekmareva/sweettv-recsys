"""Step 1: exploratory data analysis for the sweet.tv TV programme recommender.

Reads ./data (read-only) and writes outputs/eda.md and outputs/eda.json.
Usage: python src/eda.py
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

VIEW_FILE = "dataset11-30.csv"
SCHED_FILES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
SUB_FILE = "submission.csv"
SEED = 42
SAMPLE_FRAC = 0.02
QS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99, 1.0]

R = {}      # results -> eda.json
MD = []     # lines -> eda.md


def md(*lines):
    MD.extend(lines)


def quantiles(s):
    q = s.quantile(QS)
    return {("min" if k == 0 else "max" if k == 1 else f"p{int(round(k * 100))}"): float(v)
            for k, v in q.items()}


def fmt_q(d):
    return ", ".join(f"{k}={v:g}" for k, v in d.items())


def head_str(df, n=3, width=40):
    h = df.head(n).copy()
    for c in h.columns:
        if h[c].dtype == object:
            h[c] = h[c].astype(str).str.slice(0, width)
    return h.to_string(index=False)


def iso_week(ts):
    ic = ts.isocalendar()
    return {"date": str(ts), "iso_year": int(ic[0]), "iso_week": int(ic[1]), "weekday": int(ic[2])}


# ---------------------------------------------------------------- 0. environment
env = {
    "python": sys.version.split()[0],
    "pandas": pd.__version__,
    "numpy": np.__version__,
    "platform": platform.platform(),
    "processor": platform.processor(),
    "cpu_count": os.cpu_count(),
}
try:
    import psutil
    env["ram_total_gb"] = round(psutil.virtual_memory().total / 2**30, 2)
except ImportError:
    env["ram_total_gb"] = None
R["environment"] = env

# ---------------------------------------------------------------- 1. load files
print("loading view log ...", flush=True)
views = pd.read_csv(
    DATA / VIEW_FILE,
    dtype={"user_id": "uint64", "vsetv_id": "int32", "start_time": str, "stop_time": str, "duraton": "int64"},
)
raw_view_head = views.head(3).copy()
view_fmt = "%Y-%m-%d %H:%M:%S"
view_str_len = {c: views[c].str.len().value_counts().to_dict() for c in ["start_time", "stop_time"]}
views["start_time"] = pd.to_datetime(views["start_time"], format=view_fmt, errors="coerce")
views["stop_time"] = pd.to_datetime(views["stop_time"], format=view_fmt, errors="coerce")

print("loading schedules ...", flush=True)
sched_fmt = "%d.%m.%Y %H:%M:%S"
scheds, sched_raw_head, sched_meta = {}, {}, {}
for key, fn in SCHED_FILES.items():
    df = pd.read_csv(DATA / fn, dtype={"start_time": str}, low_memory=False)
    sched_raw_head[key] = df.head(3).copy()
    lens = df["start_time"].str.len().value_counts().to_dict()
    df["start_time"] = pd.to_datetime(df["start_time"], format=sched_fmt, errors="coerce")
    sched_meta[key] = {"start_time_str_lengths": {int(k): int(v) for k, v in lens.items()},
                       "start_time_unparsed": int(df["start_time"].isna().sum())}
    df["stop_time"] = df["start_time"] + pd.to_timedelta(df["duration"], unit="s")
    scheds[key] = df

sub = pd.read_csv(DATA / SUB_FILE, dtype=str)

md("# sweet.tv TV programme recommender: EDA (step 1)", "")
md("## 0. Environment", "")
for k, v in env.items():
    md(f"- {k}: {v}")
md("- runtime: see the end of this file", "")

md("## 1. Files: columns, dtypes, row counts, first rows", "")
files_info = {}


def describe_file(name, df_parsed, raw_head):
    files_info[name] = {
        "rows": int(len(df_parsed)),
        "columns": list(df_parsed.columns),
        "dtypes_after_load": {c: str(t) for c, t in df_parsed.dtypes.items()},
        "null_counts": {c: int(v) for c, v in df_parsed.isna().sum().items()},
    }
    md(f"### {name}", "", f"- rows: {len(df_parsed):,}",
       f"- dtypes (as loaded by this script): " + ", ".join(f"`{c}`:{t}" for c, t in df_parsed.dtypes.items()),
       "- null counts: " + ", ".join(f"`{c}`:{v}" for c, v in df_parsed.isna().sum().items() if v), "",
       "First 3 rows (raw strings; long text cut to 40 chars):", "", "```", head_str(raw_head), "```", "")


describe_file(VIEW_FILE, views, raw_view_head)
for key, fn in SCHED_FILES.items():
    describe_file(fn, scheds[key].drop(columns="stop_time"), sched_raw_head[key])
describe_file(SUB_FILE, sub, sub.head(3))
R["files"] = files_info

# duration units: check whether consecutive programmes on a channel abut when duration is read as seconds
abut = {}
for key, df in scheds.items():
    d = df.sort_values(["channel_id", "start_time"])
    same = d["channel_id"].eq(d["channel_id"].shift(-1))
    nxt = d["start_time"].shift(-1)
    gap = (nxt - d["stop_time"]).dt.total_seconds()[same]
    abut[key] = {
        "share_next_start_eq_start_plus_duration_s": float((gap == 0).mean()),
        "share_gap_negative_overlap": float((gap < 0).mean()),
        "share_gap_positive": float((gap > 0).mean()),
        "gap_seconds_quantiles": quantiles(gap),
        "duration_seconds_quantiles": quantiles(df["duration"]),
        "duration_all_multiples_of_60": bool((df["duration"] % 60 == 0).all()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_channel_start": int(df.duplicated(["channel_id", "start_time"]).sum()),
    }
    sched_meta[key].update(abut[key])

R["time_formats"] = {
    "view_log": {"format": view_fmt, "string_lengths": {c: {int(k): int(v) for k, v in d.items()} for c, d in view_str_len.items()},
                 "unparsed_start": int(views["start_time"].isna().sum()),
                 "unparsed_stop": int(views["stop_time"].isna().sum()),
                 "timezone_in_string": False},
    "schedules": {"start_time_format": sched_fmt, "timezone_in_string": False, "per_file": sched_meta},
}
md("### Time formats", "",
   f"- View log `start_time`/`stop_time`: strings `{view_fmt}` (e.g. `{raw_view_head['start_time'].iloc[0]}`), "
   f"no timezone/offset in the string. String lengths: {view_str_len}. "
   f"Unparsed with this format: start={R['time_formats']['view_log']['unparsed_start']}, stop={R['time_formats']['view_log']['unparsed_stop']}.",
   f"- Schedule `start_time`: strings `{sched_fmt}` (day.month.year, e.g. `{sched_raw_head['11-20']['start_time'].iloc[0]}`), "
   "no timezone/offset in the string.",
   "- Schedule `duration`: integer. Unit check: if duration is in seconds, next programme start on the same channel "
   "should equal start + duration:", "")
for key, a in abut.items():
    md(f"  - {key}: next_start == start+duration(s) for {a['share_next_start_eq_start_plus_duration_s']:.4f} of consecutive pairs; "
       f"overlap(gap<0) {a['share_gap_negative_overlap']:.4f}; gap>0 {a['share_gap_positive']:.4f}; "
       f"all durations multiples of 60: {a['duration_all_multiples_of_60']}; duration quantiles (s): {fmt_q(a['duration_seconds_quantiles'])}; "
       f"exact duplicate rows: {a['exact_duplicate_rows']}; duplicate (channel_id,start_time): {a['duplicate_channel_start']}; "
       f"unparsed start_time: {sched_meta[key]['start_time_unparsed']}")
md("")

# ---------------------------------------------------------------- 2. time ranges / ISO weeks
views["iso_year"] = views["start_time"].dt.isocalendar().year.astype("int16")
views["iso_week"] = views["start_time"].dt.isocalendar().week.astype("int16")
tr = {"view_log": {
    "start_min": iso_week(views["start_time"].min()), "start_max": iso_week(views["start_time"].max()),
    "stop_min": iso_week(views["stop_time"].min()), "stop_max": iso_week(views["stop_time"].max()),
    "rows_per_iso_year_week": {f"{y}-W{w:02d}": int(n) for (y, w), n in views.groupby(["iso_year", "iso_week"]).size().items()},
}}
for key, df in scheds.items():
    wk = df["start_time"].dt.isocalendar()
    tr[key] = {"start_min": iso_week(df["start_time"].min()), "start_max": iso_week(df["start_time"].max()),
               "stop_max": iso_week(df["stop_time"].max()),
               "rows_per_iso_year_week": {f"{y}-W{w:02d}": int(n) for (y, w), n in df.groupby([wk.year, wk.week]).size().items()}}
R["time_ranges"] = tr
views_in_11_30 = views["iso_year"].eq(2020) & views["iso_week"].between(11, 30)
R["time_ranges"]["view_log"]["share_rows_iso2020_weeks_11_30"] = float(views_in_11_30.mean())

md("## 2. Time ranges and ISO weeks", "")
for name, t in tr.items():
    md(f"- **{name}**: start min {t['start_min']['date']} (ISO {t['start_min']['iso_year']}-W{t['start_min']['iso_week']}, weekday {t['start_min']['weekday']}), "
       f"start max {t['start_max']['date']} (ISO {t['start_max']['iso_year']}-W{t['start_max']['iso_week']}, weekday {t['start_max']['weekday']})"
       + (f"; stop max {t['stop_max']['date']}" if "stop_max" in t else ""))
md("", f"- Share of view-log rows with start_time in ISO 2020 weeks 11-30: {R['time_ranges']['view_log']['share_rows_iso2020_weeks_11_30']:.6f}", "",
   "Rows per ISO week (view log):", "", "```",
   " ".join(f"{k}:{v}" for k, v in tr["view_log"]["rows_per_iso_year_week"].items()), "```", "")
for key in SCHED_FILES:
    md(f"Rows per ISO week (schedule {key}):", "", "```", " ".join(f"{k}:{v}" for k, v in tr[key]["rows_per_iso_year_week"].items()), "```", "")

# ---------------------------------------------------------------- 3. channel key
md("## 3. Channel key: vsetv_id vs channel_id", "")
period_of_view = np.where(views["iso_week"] <= 20, "11-20", "21-30")
views["period"] = period_of_view
ch = {}
vset_all = set(views["vsetv_id"].unique())
for key in ["11-20", "21-30"]:
    chan = set(scheds[key]["channel_id"].unique())
    v = views[views["period"] == key]
    ch[key] = {
        "view_rows": int(len(v)),
        "distinct_vsetv_id": int(v["vsetv_id"].nunique()),
        "distinct_channel_id": len(chan),
        "share_rows_vsetv_in_channel_id": float(v["vsetv_id"].isin(chan).mean()),
        "share_distinct_vsetv_in_channel_id": float(np.mean([x in chan for x in v["vsetv_id"].unique()])),
    }
chan_test = set(scheds["31-42"]["channel_id"].unique())
ch["31-42"] = {"distinct_channel_id": len(chan_test),
               "share_distinct_vsetv_in_test_channel_id": float(np.mean([x in chan_test for x in vset_all]))}
ch["vsetv_id_range"] = [int(views["vsetv_id"].min()), int(views["vsetv_id"].max())]
ch["channel_id_range"] = [int(min(min(s["channel_id"]) for s in scheds.values())), int(max(max(s["channel_id"]) for s in scheds.values()))]
missing = views.loc[~views["vsetv_id"].isin(set(scheds["11-20"]["channel_id"]) | set(scheds["21-30"]["channel_id"])), "vsetv_id"]
ch["top_vsetv_ids_not_in_any_training_schedule"] = {int(k): int(v) for k, v in missing.value_counts().head(15).items()}
ch["channel_id_to_titles"] = {int(k): sorted(set(v)) for k, v in
                              pd.concat(scheds.values())[["channel_id", "channel_title"]].drop_duplicates()
                              .groupby("channel_id")["channel_title"]}
ch["channel_ids_with_multiple_titles"] = {k: v for k, v in ch["channel_id_to_titles"].items() if len(v) > 1}
R["channel_key"] = ch
for key in ["11-20", "21-30"]:
    c = ch[key]
    md(f"- Period {key}: {c['view_rows']:,} view rows, {c['distinct_vsetv_id']} distinct vsetv_id, schedule has {c['distinct_channel_id']} channel_id. "
       f"Share of view rows whose vsetv_id is a channel_id of the same-period schedule: **{c['share_rows_vsetv_in_channel_id']:.4f}**; "
       f"share of distinct vsetv_id found: {c['share_distinct_vsetv_in_channel_id']:.4f}.")
md(f"- Test schedule has {ch['31-42']['distinct_channel_id']} channel_id; share of all distinct vsetv_id found there: {ch['31-42']['share_distinct_vsetv_in_test_channel_id']:.4f}.",
   f"- vsetv_id range {ch['vsetv_id_range']}, channel_id range {ch['channel_id_range']}.",
   f"- Most frequent vsetv_id not present in either training schedule (id: view rows): {ch['top_vsetv_ids_not_in_any_training_schedule']}",
   f"- channel_ids with more than one channel_title: {ch['channel_ids_with_multiple_titles'] or 'none'}", "")

# ---------------------------------------------------------------- 4. labelling
md("## 4. Programme labelling", "")
lab = {}
train_labelled = set()
for key in ["11-20", "21-30"]:
    df = scheds[key]
    train_labelled |= set(df.loc[df["tv_show_id"] != 0, "tv_show_id"])
for key, df in scheds.items():
    m = df["tv_show_id"] != 0
    lab[key] = {
        "rows": int(len(df)),
        "share_rows_labelled": float(m.mean()),
        "share_airtime_labelled": float(df.loc[m, "duration"].sum() / df["duration"].sum()),
        "distinct_labelled_tv_show_id": int(df.loc[m, "tv_show_id"].nunique()),
        "tv_show_id_null": int(df["tv_show_id"].isna().sum()),
        "category_counts_all_rows": {str(k): int(v) for k, v in df["tv_show_category"].value_counts(dropna=False).items()},
        "category_counts_labelled_rows": {str(k): int(v) for k, v in df.loc[m, "tv_show_category"].value_counts(dropna=False).items()},
        "distinct_category_per_labelled_show_max": int(df.loc[m].groupby("tv_show_id")["tv_show_category"].nunique().max()),
    }
test = scheds["31-42"]
tm = test["tv_show_id"] != 0
test_ids = set(test.loc[tm, "tv_show_id"])
in_train = test["tv_show_id"].isin(train_labelled) & tm
lab["test_vs_train"] = {
    "test_distinct_labelled": len(test_ids),
    "train_distinct_labelled_union": len(train_labelled),
    "test_labelled_also_in_train": len(test_ids & train_labelled),
    "share_test_distinct_in_train": len(test_ids & train_labelled) / len(test_ids),
    "share_test_labelled_rows_in_train": float(in_train.sum() / tm.sum()),
    "share_test_labelled_airtime_in_train": float(test.loc[in_train, "duration"].sum() / test.loc[tm, "duration"].sum()),
}
R["labelling"] = lab
for key in SCHED_FILES:
    l = lab[key]
    md(f"- {key}: {l['rows']:,} rows, tv_show_id != 0 for **{l['share_rows_labelled']:.4f}** of rows ({l['share_airtime_labelled']:.4f} of airtime), "
       f"{l['distinct_labelled_tv_show_id']:,} distinct labelled programmes; null tv_show_id: {l['tv_show_id_null']}; "
       f"max distinct categories per labelled show: {l['distinct_category_per_labelled_show_max']}")
tvt = lab["test_vs_train"]
md("", f"- Test (31-42) labelled programmes: {tvt['test_distinct_labelled']:,}; also labelled in training schedules: "
       f"**{tvt['test_labelled_also_in_train']:,}** ({tvt['share_test_distinct_in_train']:.4f}); "
       f"they cover {tvt['share_test_labelled_rows_in_train']:.4f} of labelled test rows and {tvt['share_test_labelled_airtime_in_train']:.4f} of labelled test airtime.", "",
   "tv_show_category counts (all rows / labelled rows):", "")
cats = sorted(set().union(*[lab[k]["category_counts_all_rows"] for k in SCHED_FILES]))
md("| category | " + " | ".join(f"{k} all | {k} labelled" for k in SCHED_FILES) + " |",
   "|---|" + "---|---|" * len(SCHED_FILES))
for c in cats:
    md(f"| {c} | " + " | ".join(f"{lab[k]['category_counts_all_rows'].get(c, 0)} | {lab[k]['category_counts_labelled_rows'].get(c, 0)}" for k in SCHED_FILES) + " |")
md("")

# ---------------------------------------------------------------- 5. view log
md("## 5. View log", "")
calc = (views["stop_time"] - views["start_time"]).dt.total_seconds()
diff = views["duraton"] - calc
vl = {
    "rows": int(len(views)),
    "distinct_users": int(views["user_id"].nunique()),
    "distinct_users_weeks_11_20": int(views.loc[views["period"] == "11-20", "user_id"].nunique()),
    "distinct_users_weeks_21_30": int(views.loc[views["period"] == "21-30", "user_id"].nunique()),
    "duraton_quantiles": quantiles(views["duraton"]),
    "duraton_mean": float(views["duraton"].mean()),
    "share_duraton_eq_stop_minus_start": float((diff == 0).mean()),
    "duraton_minus_calc_value_counts_top": {int(k): int(v) for k, v in diff.value_counts().head(10).items()},
    "share_duraton_le_0": float((views["duraton"] <= 0).mean()),
    "exact_duplicate_rows": int(views.duplicated(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"]).sum()),
    "views_per_user_quantiles": quantiles(views.groupby("user_id").size()),
    "stop_crosses_midnight_share": float((views["stop_time"].dt.date != views["start_time"].dt.date).mean()),
}
# Ukraine switched to summer time on 2020-03-29 03:00 local (clocks jump to 04:00)
dst = pd.Timestamp("2020-03-29 03:00:00")
mism = diff != 0
vl["mismatch_rows"] = int(mism.sum())
vl["mismatch_rows_spanning_2020_03_29_03_00"] = int((mism & (views["start_time"] < dst) & (views["stop_time"] > dst)).sum())
R["view_log"] = vl
md(f"- rows {vl['rows']:,}; distinct users **{vl['distinct_users']:,}** (weeks 11-20: {vl['distinct_users_weeks_11_20']:,}, weeks 21-30: {vl['distinct_users_weeks_21_30']:,})",
   f"- duraton (s): {fmt_q(vl['duraton_quantiles'])}; mean {vl['duraton_mean']:.1f}; share <= 0: {vl['share_duraton_le_0']:.6f}",
   f"- duraton == stop_time - start_time for **{vl['share_duraton_eq_stop_minus_start']:.6f}** of rows; most common (duraton - computed) values: {vl['duraton_minus_calc_value_counts_top']}",
   f"- of the {vl['mismatch_rows']} mismatching rows, {vl['mismatch_rows_spanning_2020_03_29_03_00']} span 2020-03-29 03:00 (Ukrainian DST switch): "
   "view-log timestamps are local wall-clock time with DST, and duraton is true elapsed seconds",
   f"- exact duplicate rows: {vl['exact_duplicate_rows']:,}; share of views crossing midnight: {vl['stop_crosses_midnight_share']:.4f}",
   f"- views per user: {fmt_q(vl['views_per_user_quantiles'])}", "")

# ---------------------------------------------------------------- 6. submission
md("## 6. Submission template", "")
with open(DATA / SUB_FILE, encoding="utf-8") as f:
    first_lines = [next(f).rstrip("\n") for _ in range(3)]
sub_users = sub["user_id"].astype("uint64")
pred_tokens = sub["tv_show_id"].str.split(" ")
u_all = set(views["user_id"].unique())
u_1120 = set(views.loc[views["period"] == "11-20", "user_id"].unique())
u_2130 = set(views.loc[views["period"] == "21-30", "user_id"].unique())
su = set(sub_users)
sb = {
    "first_lines": first_lines,
    "rows": int(len(sub)),
    "distinct_users": int(sub_users.nunique()),
    "header": list(sub.columns),
    "tokens_per_row_counts": {int(k): int(v) for k, v in pred_tokens.str.len().value_counts().items()},
    "distinct_prediction_strings": sub["tv_show_id"].unique().tolist()[:5],
    "users_in_view_log": len(su & u_all),
    "users_with_view_weeks_21_30": len(su & u_2130),
    "users_with_view_weeks_11_20": len(su & u_1120),
    "users_with_view_in_both": len(su & u_1120 & u_2130),
}
R["submission"] = sb
md(f"- first lines: `{first_lines}`",
   f"- header: {sb['header']}; field delimiter `,`; second field holds {list(sb['tokens_per_row_counts'])} space-separated ids per row (counts {sb['tokens_per_row_counts']}); "
   f"template values: {sb['distinct_prediction_strings']}",
   f"- rows {sb['rows']:,}, distinct users {sb['distinct_users']:,}",
   f"- submission users present in view log: **{sb['users_in_view_log']:,}**; with >=1 view in weeks 21-30: **{sb['users_with_view_weeks_21_30']:,}**; "
   f"in weeks 11-20: **{sb['users_with_view_weeks_11_20']:,}**; in both: {sb['users_with_view_in_both']:,}", "")

# ---------------------------------------------------------------- 7. join estimate
print("join sample ...", flush=True)
md(f"## 7. Join estimate ({SAMPLE_FRAC:.0%} sample of view rows, seed {SEED})", "")
samp = views.sample(frac=SAMPLE_FRAC, random_state=SEED).reset_index(drop=True)
samp["view_idx"] = np.arange(len(samp))


def overlap_pairs(v, s):
    """All (view, schedule) pairs on the same channel with positive time overlap."""
    out = []
    for cid, vg in v.groupby("vsetv_id"):
        sg = s[s["channel_id"] == cid].sort_values("start_time")
        if sg.empty:
            continue
        st = sg["start_time"].values
        en = np.maximum.accumulate(sg["stop_time"].values)  # monotone bound, exact filter below
        lo = np.searchsorted(en, vg["start_time"].values, side="right")
        hi = np.searchsorted(st, vg["stop_time"].values, side="left")
        n = hi - lo
        keep = n > 0
        if not keep.any():
            continue
        vi = np.repeat(vg["view_idx"].values[keep], n[keep])
        si = np.concatenate([np.arange(a, b) for a, b in zip(lo[keep], hi[keep])])
        out.append(pd.DataFrame({"view_idx": vi, "sched_pos": sg.index.values[si]}))
    if not out:
        return pd.DataFrame(columns=["view_idx", "sched_pos"])
    return pd.concat(out, ignore_index=True)


pairs_all = []
for key in ["11-20", "21-30"]:
    s = scheds[key]
    v = samp[samp["period"] == key]
    p = overlap_pairs(v, s)
    p = p.merge(v[["view_idx", "start_time", "stop_time"]], on="view_idx")
    p = p.join(s[["start_time", "stop_time", "duration", "tv_show_id"]].rename(
        columns={"start_time": "p_start", "stop_time": "p_stop"}), on="sched_pos")
    p["overlap_s"] = (np.minimum(p["stop_time"], p["p_stop"]) - np.maximum(p["start_time"], p["p_start"])).dt.total_seconds()
    p = p[p["overlap_s"] > 0]
    pairs_all.append(p)
pairs = pd.concat(pairs_all, ignore_index=True)
labelled = pairs[pairs["tv_show_id"] != 0].copy()
labelled["overlap_frac"] = labelled["overlap_s"] / labelled["duration"].replace(0, np.nan)

n_s = len(samp)
any_match = samp["view_idx"].isin(pairs["view_idx"])
lab_match = samp["view_idx"].isin(labelled["view_idx"])
zero_dur_view = samp["duraton"] <= 0
jn = {
    "sample_rows": n_s,
    "share_views_overlap_any_schedule_row": float(any_match.mean()),
    "share_views_overlap_labelled": float(lab_match.mean()),
    "share_views_with_duraton_le_0_in_sample": float(zero_dur_view.mean()),
    "share_views_overlap_labelled_among_positive_duration": float(lab_match[~zero_dur_view].mean()),
    "share_views_whose_channel_absent_from_period_schedule": float(np.mean(np.concatenate([
        ~samp.loc[samp["period"] == k, "vsetv_id"].isin(set(scheds[k]["channel_id"])).values for k in ["11-20", "21-30"]]))),
    "schedule_rows_matched_per_view_quantiles": quantiles(pairs.groupby("view_idx").size()),
    "labelled_pairs": int(len(labelled)),
    "labelled_overlap_seconds_quantiles": quantiles(labelled["overlap_s"]),
    "labelled_overlap_over_programme_duration_quantiles": quantiles(labelled["overlap_frac"].dropna()),
    "labelled_overlap_frac_share_ge": {str(t): float((labelled["overlap_frac"] >= t).mean()) for t in [0.05, 0.1, 0.25, 0.5, 0.8]},
    "labelled_overlap_seconds_share_ge": {str(t): float((labelled["overlap_s"] >= t).mean()) for t in [60, 300, 600, 1800]},
}

# alignment check (timezone): share of views starting within +-60 s of some programme start on the
# same channel, for shifts of the view clock by -3..+3 h. A peak at shift 0 suggests the same clock.
def near_start_share(shift_h):
    hits, tot = 0, 0
    for key in ["11-20", "21-30"]:
        s = scheds[key]
        v = samp[(samp["period"] == key) & (samp["duraton"] > 0)]
        for cid, vg in v.groupby("vsetv_id"):
            st = np.sort(s.loc[s["channel_id"] == cid, "start_time"].values)
            tot += len(vg)
            if len(st) == 0:
                continue
            t = (vg["start_time"] + pd.Timedelta(hours=shift_h)).values
            i = np.clip(np.searchsorted(st, t), 1, len(st) - 1)
            d = np.minimum(np.abs(t - st[i - 1]), np.abs(st[i] - t))
            hits += int((d <= np.timedelta64(60, "s")).sum())
    return hits / tot


jn["view_start_within_60s_of_programme_start_by_shift_h"] = {str(h): near_start_share(h) for h in range(-3, 4)}
R["join_sample"] = jn
md(f"- sample rows: {n_s:,}",
   f"- share of sampled views overlapping >=1 schedule row (same channel, by period): {jn['share_views_overlap_any_schedule_row']:.4f}",
   f"- share overlapping >=1 **labelled** programme (tv_show_id != 0): **{jn['share_views_overlap_labelled']:.4f}** "
   f"(among views with duraton > 0: {jn['share_views_overlap_labelled_among_positive_duration']:.4f}; views with duraton <= 0 in sample: {jn['share_views_with_duraton_le_0_in_sample']:.4f})",
   f"- share of sampled views whose channel is absent from the period schedule: {jn['share_views_whose_channel_absent_from_period_schedule']:.4f}",
   f"- schedule rows matched per matched view: {fmt_q(jn['schedule_rows_matched_per_view_quantiles'])}",
   f"- labelled (view, programme) pairs: {jn['labelled_pairs']:,}",
   f"- overlap seconds (labelled pairs): {fmt_q(jn['labelled_overlap_seconds_quantiles'])}",
   f"- overlap / programme duration (labelled pairs): {fmt_q(jn['labelled_overlap_over_programme_duration_quantiles'])}",
   f"- share of labelled pairs with overlap/duration >= t: {jn['labelled_overlap_frac_share_ge']}",
   f"- share of labelled pairs with overlap seconds >= t: {jn['labelled_overlap_seconds_share_ge']}",
   f"- clock alignment check, share of views (duraton>0) starting within 60 s of a programme start when the view clock is shifted by h hours: "
   f"{ {k: round(v, 4) for k, v in jn['view_start_within_60s_of_programme_start_by_shift_h'].items()} }", "")

# ---------------------------------------------------------------- write
R["runtime_seconds"] = round(time.time() - T0, 1)
md(f"---", f"Runtime: {R['runtime_seconds']} s")
(OUT / "eda.md").write_text("\n".join(MD), encoding="utf-8")


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (set, np.ndarray)):
        return list(o)
    return str(o)


(OUT / "eda.json").write_text(json.dumps(R, ensure_ascii=False, indent=2, default=_default), encoding="utf-8")
print(f"done in {R['runtime_seconds']} s -> outputs/eda.md, outputs/eda.json")

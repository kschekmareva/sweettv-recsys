"""Loading, view/schedule join and interaction table.

Times are used as given (no timezone shift). Schedule end = start + duration seconds.
user_id is always uint64 (never float).
"""
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "outputs" / "cache"

VIEW_FILE = "dataset11-30.csv"
SCHEDULES = {
    "11-20": "export_arh_11-20-final.csv",
    "21-30": "export_arh_21-30-final.csv",
    "31-42": "export_arh_31-42-final.csv",
}
SUB_FILE = "submission.csv"
# view-log period -> (first ISO week, last ISO week); each uses the schedule file of the same name
PERIODS = {"11-20": (11, 20), "21-30": (21, 30)}
# extra sub-periods for recency popularity: label -> (schedule file key, first ISO week, last ISO week)
RECENT = {"17-20": ("11-20", 17, 20), "27-30": ("21-30", 27, 30)}

EPOCH = np.datetime64("2020-01-01T00:00:00")
K = 1 << 26  # > seconds range of the data (~2.1e7), used to pack (channel, time) into one int64
DAY = 86400


def _secs(dt: pd.Series) -> np.ndarray:
    """datetime Series -> int64 seconds since 2020-01-01."""
    return ((dt.values.astype("datetime64[s]") - EPOCH) // np.timedelta64(1, "s")).astype(np.int64)


def load_views() -> pd.DataFrame:
    df = pd.read_csv(
        DATA / VIEW_FILE,
        dtype={"user_id": "uint64", "vsetv_id": "int32", "start_time": str, "stop_time": str, "duraton": "int32"},
    )
    st = pd.to_datetime(df["start_time"], format="%Y-%m-%d %H:%M:%S")
    sp = pd.to_datetime(df["stop_time"], format="%Y-%m-%d %H:%M:%S")
    iso = st.dt.isocalendar()
    assert (iso["year"] == 2020).all()
    return pd.DataFrame({
        "user_id": df["user_id"].values,
        "vsetv_id": df["vsetv_id"].values,
        "start": _secs(st).astype(np.int32),
        "stop": _secs(sp).astype(np.int32),
        "iso_week": iso["week"].values.astype(np.int8),
    })


def load_schedule(key: str) -> pd.DataFrame:
    df = pd.read_csv(
        DATA / SCHEDULES[key],
        usecols=["channel_id", "start_time", "duration", "tv_show_title", "tv_show_id", "tv_show_category"],
        dtype={"start_time": str},
    )
    st = pd.to_datetime(df["start_time"], format="%d.%m.%Y %H:%M:%S")
    iso = st.dt.isocalendar()
    start = _secs(st)
    out = df[["channel_id", "tv_show_id", "tv_show_title", "tv_show_category"]].copy()
    out["start"] = start
    out["end"] = start + df["duration"].values.astype(np.int64)
    out["iso_year"] = iso["year"].values.astype(np.int16)
    out["iso_week"] = iso["week"].values.astype(np.int8)
    return out


def pool_ids(sched: pd.DataFrame, w0: int, w1: int) -> np.ndarray:
    """Distinct labelled tv_show_ids airing in ISO weeks w0..w1 (by programme start), sorted."""
    m = (sched["tv_show_id"] != 0) & sched["iso_week"].between(w0, w1) & (sched["iso_year"] == 2020)
    return np.sort(sched.loc[m, "tv_show_id"].unique())


def titles() -> pd.DataFrame:
    """tv_show_id -> (title, category), first occurrence over all schedule files."""
    parts = [load_schedule(k) for k in SCHEDULES]
    t = pd.concat(parts)
    t = t[t["tv_show_id"] != 0].drop_duplicates("tv_show_id")
    return t.set_index("tv_show_id")[["tv_show_title", "tv_show_category"]]


def load_submission_users() -> list:
    sub = pd.read_csv(DATA / SUB_FILE, dtype=str)
    return [int(x) for x in sub["user_id"]]


def join_views_schedule(v: pd.DataFrame, s: pd.DataFrame):
    """Vectorised interval join on (vsetv_id == channel_id) with positive overlap.

    Schedule rows are sorted by (channel, start); a running max of packed end keys makes the
    candidate range of each view a contiguous slice found with two searchsorted calls.
    Returns (vidx, show, overlap_sec, airing, airing_duration) arrays for ALL overlapping pairs
    (incl. tv_show_id == 0; `airing` = row of the start-sorted schedule) and a stats dict.
    """
    s = s.sort_values(["channel_id", "start"], kind="stable").reset_index(drop=True)
    ch = s["channel_id"].values.astype(np.int64) * K
    sk = ch + s["start"].values
    ek = np.maximum.accumulate(ch + s["end"].values)
    s_start, s_end, s_show = s["start"].values, s["end"].values, s["tv_show_id"].values

    present = v["vsetv_id"].isin(s["channel_id"].unique()).values
    vp = np.flatnonzero(present)  # indices into v of views on channels in the schedule
    vc = v["vsetv_id"].values[vp].astype(np.int64) * K
    v_start = v["start"].values[vp].astype(np.int64)
    v_stop = v["stop"].values[vp].astype(np.int64)
    lo = np.searchsorted(ek, vc + v_start, side="right")
    hi = np.searchsorted(sk, vc + v_stop, side="left")
    n = np.maximum(hi - lo, 0)
    rep = np.repeat(np.arange(len(vp)), n)
    si = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + np.repeat(lo, n)
    ov = np.minimum(v_stop[rep], s_end[si]) - np.maximum(v_start[rep], s_start[si])
    keep = ov > 0
    vidx, show, air, ov = vp[rep[keep]], s_show[si[keep]], si[keep], ov[keep]
    adur = (s_end - s_start)[air]
    achan, astart = s["channel_id"].values[air], s_start[air]

    dur = (v["stop"].values.astype(np.int64) - v["start"].values)
    matched_views = np.unique(vidx)
    absent_ch = v.loc[~present, "vsetv_id"]
    stats = {
        "views": int(len(v)),
        "view_seconds": int(dur.sum()),
        "views_channel_absent_from_schedule": int((~present).sum()),
        "view_seconds_channel_absent": int(dur[~present].sum()),
        "distinct_vsetv_id_absent": int(absent_ch.nunique()),
        "views_channel_present_no_overlap": int(present.sum() - len(matched_views)),
        "views_with_any_overlap": int(len(matched_views)),
        "pairs_all": int(len(vidx)),
        "matched_seconds_all": int(ov.sum()),
        "matched_seconds_unlabelled_id0": int(ov[show == 0].sum()),
        "matched_seconds_labelled": int(ov[show != 0].sum()),
    }
    stats["share_matched_seconds_unlabelled"] = stats["matched_seconds_unlabelled_id0"] / stats["matched_seconds_all"]
    stats["views_with_labelled_overlap"] = int(len(np.unique(vidx[show != 0])))
    return vidx, show, ov, air, adur, achan, astart, stats


def _interactions(label: str, sched_key: str, w0: int, w1: int, views: pd.DataFrame):
    v = views[views["iso_week"].between(w0, w1)].reset_index(drop=True)
    s = load_schedule(sched_key)
    vidx, show, ov, air, adur, achan, astart, stats = join_views_schedule(v, s)
    lab = show != 0
    g = (pd.DataFrame({"vidx": vidx[lab], "show": show[lab], "ov": ov[lab]})
         .groupby(["vidx", "show"], sort=False, as_index=False)["ov"].sum())  # one row per (view, programme)
    ucode, uniq = pd.factorize(v["user_id"].values)
    g["ucode"] = ucode[g["vidx"].values]
    g["day"] = (v["start"].values[g["vidx"].values] // DAY).astype(np.int32)
    agg = g.groupby(["ucode", "show"], as_index=False).agg(overlap_sec=("ov", "sum"), n_views=("ov", "size"))
    days = (g[["ucode", "show", "day"]].drop_duplicates().groupby(["ucode", "show"]).size()
            .rename("n_days").reset_index())
    agg = agg.merge(days, on=["ucode", "show"])
    # largest share of a single airing the user watched: per (user, airing) overlap / duration, capped at 1
    a = pd.DataFrame({"ucode": ucode[vidx[lab]], "air": air[lab], "ov": ov[lab], "dur": adur[lab], "show": show[lab]})
    ga = a.groupby(["ucode", "air"], as_index=False).agg(ov=("ov", "sum"), dur=("dur", "first"), show=("show", "first"))
    ga["frac"] = np.minimum(ga["ov"] / ga["dur"], 1.0)
    mfrac = ga.groupby(["ucode", "show"], as_index=False)["frac"].max().rename(columns={"frac": "max_frac"})
    agg = agg.merge(mfrac, on=["ucode", "show"])
    # airing-level table: overlap of one user on one airing (channel_id + start_time) summed over all their views
    gb = (pd.DataFrame({"ucode": ucode[vidx[lab]], "ch": achan[lab], "st": astart[lab], "ov": ov[lab], "dur": adur[lab], "show": show[lab]})
          .groupby(["ucode", "ch", "st"], as_index=False).agg(ov=("ov", "sum"), dur=("dur", "max"), show=("show", "first")))
    airings = pd.DataFrame({
        "user_id": uniq[gb["ucode"].values].astype(np.uint64), "tv_show_id": gb["show"].values.astype(np.int64),
        "channel_id": gb["ch"].values.astype(np.int32), "start": gb["st"].values.astype(np.int32),
        "dur": gb["dur"].values.astype(np.int32), "ov": gb["ov"].values.astype(np.int32),
        "view_pct": np.minimum(gb["ov"].values / gb["dur"].values, 1.0).astype(np.float32)})
    inter = pd.DataFrame({
        "period": label,
        "user_id": uniq[agg["ucode"].values].astype(np.uint64),
        "tv_show_id": agg["show"].values.astype(np.int64),
        "overlap_sec": agg["overlap_sec"].values.astype(np.int64),
        "n_views": agg["n_views"].values.astype(np.int32),
        "n_days": agg["n_days"].values.astype(np.int16),
        "max_frac": agg["max_frac"].values.astype(np.float32),
    })
    stats["distinct_users_in_views"] = int(len(uniq))
    stats["interaction_pairs_labelled"] = int(len(inter))
    return inter, stats, airings


def _build_cache(force: bool = False) -> dict:
    f = CACHE / "interactions_v3.pkl"
    if f.exists() and not force:
        with open(f, "rb") as fh:
            return pickle.load(fh)
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    views = load_views()
    parts, stats, recent, air = [], {}, [], {}
    for key, (w0, w1) in PERIODS.items():
        inter, st, air[key] = _interactions(key, key, w0, w1, views)
        parts.append(inter)
        stats[key] = st
        print(f"  joined {key}: {st['pairs_all']:,} pairs, {len(inter):,} labelled user-programme pairs", flush=True)
    for label, (sk, w0, w1) in RECENT.items():
        inter, st, _ = _interactions(label, sk, w0, w1, views)
        recent.append(inter)
        stats[label] = st
        print(f"  joined {label}: {len(inter):,} labelled user-programme pairs", flush=True)
    stats["build_seconds"] = round(time.time() - t0, 1)
    out = {"inter": pd.concat(parts, ignore_index=True), "stats": stats, "recent": pd.concat(recent, ignore_index=True), "airings": air}
    with open(f, "wb") as fh:
        pickle.dump(out, fh)
    return out


def build_interactions(force: bool = False):
    """Per (period, user_id, tv_show_id): overlap_sec, n_views, n_days, max_frac. Cached in outputs/cache."""
    c = _build_cache(force)
    return c["inter"], c["stats"]


def recent_interactions() -> pd.DataFrame:
    """Same table for the sub-periods in RECENT (weeks 17-20 and 27-30)."""
    return _build_cache()["recent"]


def airings(periods) -> pd.DataFrame:
    """Airing-level table (user_id, tv_show_id, channel_id, start, dur, ov, view_pct) for the given periods.
    With several periods the rows of the same (user, channel, start) airing are merged."""
    parts = [_build_cache()["airings"][k] for k in periods]
    if len(parts) == 1:
        return parts[0]
    a = pd.concat(parts, ignore_index=True)
    g = a.groupby(["user_id", "channel_id", "start"], as_index=False).agg(
        tv_show_id=("tv_show_id", "first"), dur=("dur", "max"), ov=("ov", "sum"))
    g["view_pct"] = np.minimum(g["ov"] / g["dur"], 1.0).astype(np.float32)
    return g


def positives(inter: pd.DataFrame, periods, thr: int = 300) -> pd.DataFrame:
    """User-programme pairs whose total overlap over `periods` is >= thr seconds."""
    d = inter[inter["period"].isin(periods)]
    g = d.groupby(["user_id", "tv_show_id"], as_index=False).agg(
        overlap_sec=("overlap_sec", "sum"), n_views=("n_views", "sum"), n_days=("n_days", "sum"),
        max_frac=("max_frac", "max"))
    return g[g["overlap_sec"] >= thr].reset_index(drop=True)

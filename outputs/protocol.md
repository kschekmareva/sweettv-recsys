# Protocol (step 2)

1. **Times.** View log and schedules are used exactly as given (no timezone shift). Schedule end = start_time + duration seconds.
   user_id is read as uint64 (never float).
2. **Join.** A view matches a schedule row when vsetv_id == channel_id and the intervals overlap;
   overlap_sec = min(stop_view, end_sched) - max(start_view, start_sched) > 0. View weeks 11-20 use export_arh_11-20,
   weeks 21-30 use export_arh_21-30 (period = ISO week of the view's start_time). Vectorised per-channel interval search
   (src/data.py). Rows with tv_show_id == 0 are unlabelled airtime and are dropped (share of matched seconds reported).
   Views on channels absent from the schedule are dropped (counts reported).
3. **Interaction table** per (period, user_id, tv_show_id): total overlap_sec, n_views (distinct views touching the programme),
   n_days (distinct view start days); cached at outputs/cache/interactions.pkl. A pair is POSITIVE when total overlap_sec >= 300
   (summed over the periods used).
4. **Split.** TRAIN = views of ISO weeks 11-20, VALIDATION = views of ISO weeks 21-30. Models see TRAIN only.
   Validation truth per user = set of positive tv_show_ids in weeks 21-30.
5. **Pool (validation)** = distinct labelled tv_show_ids with a schedule row (file 21-30) starting in ISO weeks 21-30 (Monday tail of
   week 31 excluded). Recommending something the user already watched in TRAIN is allowed (all methods). Every list has exactly 5
   distinct ids from the pool; short lists are filled from the popularity ranking.
6. **Metric.** MAP@5: AP@5 = sum_k P@k * rel(k) / min(5, |truth|); empty truth -> 0. Reported (a) over users with non-empty
   validation truth, (b) over the 1,260 submission users (empty truth counts 0).
7. **Baselines.** (P) popularity = number of distinct TRAIN users with a positive pair, restricted to the pool, ties by ascending
   tv_show_id, same list for everyone. (H) own-history diagnostic = user's own TRAIN positives in the pool ranked by total overlap
   seconds (ties: ascending id), filled with (P).
8. **Final run.** TRAIN = weeks 11-30 (periods summed, positive if total >= 300 s), pool = labelled programmes with schedule
   rows starting in ISO weeks 31-42 of export_arh_31-42. No viewing data from weeks 31-42 is used.

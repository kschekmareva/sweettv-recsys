# Hand-check material (user A)

Only inputs and the code's final numbers are shown; no intermediate arithmetic.

## (a) Step-7 reranker on validation for user A (5409901949727416037)

- Top-5 (tv_show_id, in rank order): [12001682, 12002805, 12001734, 12002856, 12002378]
- VTOP5 truth (tv_show_id, rank order of the truth): [12001682, 12001697, 12002809, 12001734, 12002355]
- AP@5 computed by the code: **0.333333**

## (b) One raw view row and the schedule rows it overlaps

Raw view row (data/dataset11-30.csv, columns user_id, vsetv_id, start_time, stop_time, duraton):

```
5409901949727416037,1495,2020-05-19 16:13:56,2020-05-19 16:43:49,1793
```

Schedule rows of data/export_arh_21-30-final.csv on channel_id = vsetv_id that overlap the view interval (columns channel_id, channel_title, start_time, duration in seconds, tv_show_id, tv_show_title), with the code's results:

| channel_id | channel_title | start_time | duration | tv_show_id | tv_show_title | overlap seconds with this view (code) | view_pct of this airing for the user (code) |
|---|---|---|---|---|---|---|---|
| 1495 | NickToons | 19.05.2020 16:00:00 | 1200 | 12001732 | М/с 'Пингвины из Мадагаскара', 2 сезон, 17 с. 'Захваченные чужим / Другая выдра'. | 364 | 0.303333 |
| 1495 | NickToons | 19.05.2020 16:20:00 | 1500 | 12001732 | М/с 'Пингвины из Мадагаскара', 2 сезон, 18 с. 'Фактор офицера Икс / Любовь причиняет боль'. | 1429 | 0.952667 |

Note: view_pct is computed per airing from all of the user's views of that airing (overlap seconds summed over those views, capped at the airing duration), so it can exceed what this single view contributes.

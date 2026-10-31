# sweet.tv TV programme recommender: EDA (step 1)

## 0. Environment

- python: 3.11.9
- pandas: 2.3.3
- numpy: 2.3.3
- platform: Windows-10-10.0.19045-SP0
- processor: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel
- cpu_count: 4
- ram_total_gb: 7.65
- runtime: see the end of this file

## 1. Files: columns, dtypes, row counts, first rows

### dataset11-30.csv

- rows: 5,821,987
- dtypes (as loaded by this script): `user_id`:uint64, `vsetv_id`:int32, `start_time`:datetime64[ns], `stop_time`:datetime64[ns], `duraton`:int64
- null counts: 

First 3 rows (raw strings; long text cut to 40 chars):

```
             user_id  vsetv_id          start_time           stop_time  duraton
 3398461054087191302         6 2020-03-09 00:00:02 2020-03-09 00:01:53      111
17226860011138219284       353 2020-03-09 00:00:02 2020-03-09 00:03:26      204
18417531283109304442       332 2020-03-09 00:00:05 2020-03-09 00:01:26       81
```

### export_arh_11-20-final.csv

- rows: 1,074,235
- dtypes (as loaded by this script): `channel_id`:int64, `channel_title`:object, `start_time`:datetime64[ns], `duration`:int64, `tv_show_title`:object, `tv_show_id`:int64, `tv_show_category`:object, `tv_show_genre_1`:object, `tv_show_genre_2`:object, `tv_show_genre_3`:object, `year_of_production`:object, `director`:object, `actors`:object
- null counts: `tv_show_genre_1`:620089, `tv_show_genre_2`:928009, `tv_show_genre_3`:1054059, `year_of_production`:959500, `director`:974050, `actors`:973861

First 3 rows (raw strings; long text cut to 40 chars):

```
 channel_id channel_title          start_time  duration                          tv_show_title  tv_show_id tv_show_category     tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors
          3           1+1 09.03.2020 06:00:00      6000      'Жизнь известных людей', 2 сезон.           0           Другое                 nan             nan             nan                nan      nan    nan
          3           1+1 09.03.2020 07:40:00      5400                         'ТСН. Неделя'.     2400480             Инфо общество и политика             nan             nan               1997      nan    nan
          3           1+1 09.03.2020 09:10:00      8400 'Голос страны', 10 сезон, 8 эп. 'Бои'.      700475      Развлечения              музыка             nan             nan               2010      nan    nan
```

### export_arh_21-30-final.csv

- rows: 1,074,394
- dtypes (as loaded by this script): `channel_id`:int64, `channel_title`:object, `start_time`:datetime64[ns], `duration`:int64, `tv_show_title`:object, `tv_show_id`:int64, `tv_show_category`:object, `tv_show_genre_1`:object, `tv_show_genre_2`:object, `tv_show_genre_3`:object, `year_of_production`:object, `director`:object, `actors`:object
- null counts: `tv_show_genre_1`:616370, `tv_show_genre_2`:924020, `tv_show_genre_3`:1050668, `year_of_production`:954382, `director`:969487, `actors`:968834

First 3 rows (raw strings; long text cut to 40 chars):

```
 channel_id channel_title          start_time  duration                            tv_show_title  tv_show_id tv_show_category          tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors
          3           1+1 18.05.2020 06:30:00      1800 'Завтрак с 1+1'. Информационно-развлекат       20088      Развлечения информация (комплексная)             nan             nan              1996-      nan    nan
          3           1+1 18.05.2020 07:00:00       600                                     ТСН.     2400480             Инфо      общество и политика             nan             nan               1997      nan    nan
          3           1+1 18.05.2020 07:10:00      3000 'Завтрак с 1+1'. Информационно-развлекат       20088      Развлечения информация (комплексная)             nan             nan              1996-      nan    nan
```

### export_arh_31-42-final.csv

- rows: 1,295,378
- dtypes (as loaded by this script): `channel_id`:int64, `channel_title`:object, `start_time`:datetime64[ns], `duration`:int64, `tv_show_title`:object, `tv_show_id`:int64, `tv_show_category`:object, `tv_show_genre_1`:object, `tv_show_genre_2`:object, `tv_show_genre_3`:object, `year_of_production`:object, `director`:object, `actors`:object
- null counts: `tv_show_genre_1`:737871, `tv_show_genre_2`:1113598, `tv_show_genre_3`:1266621, `year_of_production`:1151706, `director`:1170189, `actors`:1169519

First 3 rows (raw strings; long text cut to 40 chars):

```
 channel_id channel_title          start_time  duration                            tv_show_title  tv_show_id tv_show_category          tv_show_genre_1 tv_show_genre_2 tv_show_genre_3 year_of_production director actors
          3           1+1 27.07.2020 06:30:00      1800 'Завтрак с 1+1'. Информационно-развлекат       20088      Развлечения информация (комплексная)             nan             nan              1996-      nan    nan
          3           1+1 27.07.2020 07:00:00       600                                     ТСН.     2400480             Инфо      общество и политика             nan             nan               1997      nan    nan
          3           1+1 27.07.2020 07:10:00      3000 'Завтрак с 1+1'. Информационно-развлекат       20088      Развлечения информация (комплексная)             nan             nan              1996-      nan    nan
```

### submission.csv

- rows: 1,260
- dtypes (as loaded by this script): `user_id`:object, `tv_show_id`:object
- null counts: 

First 3 rows (raw strings; long text cut to 40 chars):

```
            user_id tv_show_id
8377619604347126107  0 0 0 0 0
8381667675275833309  0 0 0 0 0
8387147770138767246  0 0 0 0 0
```

### Time formats

- View log `start_time`/`stop_time`: strings `%Y-%m-%d %H:%M:%S` (e.g. `2020-03-09 00:00:02`), no timezone/offset in the string. String lengths: {'start_time': {19: 5821987}, 'stop_time': {19: 5821987}}. Unparsed with this format: start=0, stop=0.
- Schedule `start_time`: strings `%d.%m.%Y %H:%M:%S` (day.month.year, e.g. `09.03.2020 06:00:00`), no timezone/offset in the string.
- Schedule `duration`: integer. Unit check: if duration is in seconds, next programme start on the same channel should equal start + duration:

  - 11-20: next_start == start+duration(s) for 0.9996 of consecutive pairs; overlap(gap<0) 0.0000; gap>0 0.0004; all durations multiples of 60: True; duration quantiles (s): min=0, p10=300, p25=600, p50=1500, p75=3000, p90=5400, p99=12600, max=86400; exact duplicate rows: 0; duplicate (channel_id,start_time): 1; unparsed start_time: 0
  - 21-30: next_start == start+duration(s) for 1.0000 of consecutive pairs; overlap(gap<0) 0.0000; gap>0 0.0000; all durations multiples of 60: True; duration quantiles (s): min=60, p10=300, p25=600, p50=1500, p75=3000, p90=5400, p99=13200, max=86400; exact duplicate rows: 0; duplicate (channel_id,start_time): 0; unparsed start_time: 0
  - 31-42: next_start == start+duration(s) for 1.0000 of consecutive pairs; overlap(gap<0) 0.0000; gap>0 0.0000; all durations multiples of 60: True; duration quantiles (s): min=60, p10=300, p25=600, p50=1500, p75=3000, p90=5400, p99=13500, max=43200; exact duplicate rows: 0; duplicate (channel_id,start_time): 0; unparsed start_time: 0

## 2. Time ranges and ISO weeks

- **view_log**: start min 2020-03-09 00:00:02 (ISO 2020-W11, weekday 1), start max 2020-07-26 23:58:42 (ISO 2020-W30, weekday 7); stop max 2020-07-26 23:59:58
- **11-20**: start min 2020-03-08 23:00:00 (ISO 2020-W10, weekday 7), start max 2020-05-18 11:45:00 (ISO 2020-W21, weekday 1); stop max 2020-05-18 12:00:00
- **21-30**: start min 2020-05-18 00:00:00 (ISO 2020-W21, weekday 1), start max 2020-07-27 11:55:00 (ISO 2020-W31, weekday 1); stop max 2020-07-27 12:00:00
- **31-42**: start min 2020-07-27 00:00:00 (ISO 2020-W31, weekday 1), start max 2020-10-19 11:55:00 (ISO 2020-W43, weekday 1); stop max 2020-10-19 12:00:00

- Share of view-log rows with start_time in ISO 2020 weeks 11-30: 1.000000

Rows per ISO week (view log):

```
2020-W11:407493 2020-W12:452314 2020-W13:465318 2020-W14:428820 2020-W15:412698 2020-W16:406317 2020-W17:417854 2020-W18:385637 2020-W19:336042 2020-W20:232680 2020-W21:182044 2020-W22:192998 2020-W23:162882 2020-W24:156928 2020-W25:138807 2020-W26:126940 2020-W27:105223 2020-W28:271172 2020-W29:261698 2020-W30:278122
```

Rows per ISO week (schedule 11-20):

```
2020-W10:86 2020-W11:106036 2020-W12:108177 2020-W13:107363 2020-W14:106527 2020-W15:108126 2020-W16:107697 2020-W17:107721 2020-W18:107505 2020-W19:105473 2020-W20:107093 2020-W21:2431
```

Rows per ISO week (schedule 21-30):

```
2020-W21:104798 2020-W22:107274 2020-W23:107657 2020-W24:107449 2020-W25:107104 2020-W26:107544 2020-W27:106878 2020-W28:107528 2020-W29:107640 2020-W30:108093 2020-W31:2429
```

Rows per ISO week (schedule 31-42):

```
2020-W31:105719 2020-W32:108256 2020-W33:107039 2020-W34:106862 2020-W35:107028 2020-W36:107492 2020-W37:107558 2020-W38:108258 2020-W39:108512 2020-W40:108763 2020-W41:108843 2020-W42:108649 2020-W43:2399
```

## 3. Channel key: vsetv_id vs channel_id

- Period 11-20: 3,945,173 view rows, 199 distinct vsetv_id, schedule has 400 channel_id. Share of view rows whose vsetv_id is a channel_id of the same-period schedule: **0.9711**; share of distinct vsetv_id found: 0.9598.
- Period 21-30: 1,876,814 view rows, 198 distinct vsetv_id, schedule has 403 channel_id. Share of view rows whose vsetv_id is a channel_id of the same-period schedule: **0.9784**; share of distinct vsetv_id found: 0.9697.
- Test schedule has 404 channel_id; share of all distinct vsetv_id found there: 0.9703.
- vsetv_id range [3, 1531], channel_id range [3, 1541].
- Most frequent vsetv_id not present in either training schedule (id: view rows): {1174: 86883, 1510: 9357, 1509: 8546, 1508: 7491, 149: 1302}
- channel_ids with more than one channel_title: {75: ['Киев', 'Київ'], 222: ['MTV 90s', 'MTV Rocks International'], 224: ['MTV Dance International', 'club MTV International'], 227: ['MTV 80s', 'VH1 Classic Europe'], 288: ['Russian Extreme (HD REX)', 'Русский экстрим'], 1186: ['Kyiv.Live', 'Rabinovich TV'], 1249: ['UA:Перший', 'UA:Перший (спутник)'], 1319: ['OBOZ TV', 'OBOZREVATEL'], 1379: ['Travel Guide-TV', 'Travel Guide-TV HD']}

## 4. Programme labelling

- 11-20: 1,074,235 rows, tv_show_id != 0 for **0.1076** of rows (0.1597 of airtime), 5,847 distinct labelled programmes; null tv_show_id: 0; max distinct categories per labelled show: 1
- 21-30: 1,074,394 rows, tv_show_id != 0 for **0.1131** of rows (0.1690 of airtime), 6,363 distinct labelled programmes; null tv_show_id: 0; max distinct categories per labelled show: 1
- 31-42: 1,295,378 rows, tv_show_id != 0 for **0.1118** of rows (0.1667 of airtime), 6,636 distinct labelled programmes; null tv_show_id: 0; max distinct categories per labelled show: 1

- Test (31-42) labelled programmes: 6,636; also labelled in training schedules: **5,199** (0.7835); they cover 0.9564 of labelled test rows and 0.9272 of labelled test airtime.

tv_show_category counts (all rows / labelled rows):

| category | 11-20 all | 11-20 labelled | 21-30 all | 21-30 labelled | 31-42 all | 31-42 labelled |
|---|---|---|---|---|---|---|
| Детям | 104593 | 47862 | 104913 | 46585 | 130440 | 55130 |
| Другое | 484058 | 527 | 480431 | 497 | 572099 | 548 |
| Инфо | 50585 | 4487 | 51039 | 4485 | 62860 | 5503 |
| Музыка | 29200 | 0 | 28843 | 0 | 34441 | 0 |
| Познавательное | 221981 | 2792 | 225108 | 4212 | 273595 | 4367 |
| Развлечения | 8011 | 6887 | 7792 | 6603 | 9570 | 8214 |
| Сериалы | 72031 | 31153 | 72081 | 33288 | 86663 | 40590 |
| Спорт | 62197 | 10 | 60163 | 6 | 72002 | 2 |
| Фильмы | 41579 | 21895 | 44024 | 25829 | 53708 | 30420 |

## 5. View log

- rows 5,821,987; distinct users **4,914** (weeks 11-20: 4,876, weeks 21-30: 4,820)
- duraton (s): min=61, p10=80, p25=133, p50=396, p75=1452, p90=3911, p99=14079.1, max=342145; mean 1480.5; share <= 0: 0.000000
- duraton == stop_time - start_time for **0.999980** of rows; most common (duraton - computed) values: {0: 5821870, -3600: 117}
- of the 117 mismatching rows, 117 span 2020-03-29 03:00 (Ukrainian DST switch): view-log timestamps are local wall-clock time with DST, and duraton is true elapsed seconds
- exact duplicate rows: 17; share of views crossing midnight: 0.0206
- views per user: min=1, p10=223.3, p25=476, p50=874, p75=1547, p90=2429.7, p99=5348.31, max=38114

## 6. Submission template

- first lines: `['user_id,tv_show_id', '8377619604347126107,0 0 0 0 0', '8381667675275833309,0 0 0 0 0']`
- header: ['user_id', 'tv_show_id']; field delimiter `,`; second field holds [5] space-separated ids per row (counts {5: 1260}); template values: ['0 0 0 0 0']
- rows 1,260, distinct users 1,260
- submission users present in view log: **1,260**; with >=1 view in weeks 21-30: **1,246**; in weeks 11-20: **1,256**; in both: 1,242

## 7. Join estimate (2% sample of view rows, seed 42)

- sample rows: 116,440
- share of sampled views overlapping >=1 schedule row (same channel, by period): 0.9292
- share overlapping >=1 **labelled** programme (tv_show_id != 0): **0.5074** (among views with duraton > 0: 0.5074; views with duraton <= 0 in sample: 0.0000)
- share of sampled views whose channel is absent from the period schedule: 0.0265
- schedule rows matched per matched view: min=1, p10=1, p25=1, p50=1, p75=1, p90=2, p99=6, max=94
- labelled (view, programme) pairs: 76,654
- overlap seconds (labelled pairs): min=1, p10=79, p25=155, p50=523, p75=1500, p90=3300, p99=7200, max=27600
- overlap / programme duration (labelled pairs): min=6.94444e-05, p10=0.0127778, p25=0.0296233, p50=0.115833, p75=0.452118, p90=1, p99=1, max=1
- share of labelled pairs with overlap/duration >= t: {'0.05': 0.655569180995121, '0.1': 0.5273436480809873, '0.25': 0.35755472643306285, '0.5': 0.23200354841234638, '0.8': 0.14865499517311556}
- share of labelled pairs with overlap seconds >= t: {'60': 0.9731129490959376, '300': 0.6174107026378272, '600': 0.47247371304824276, '1800': 0.21162626868787016}
- clock alignment check, share of views (duraton>0) starting within 60 s of a programme start when the view clock is shifted by h hours: {'-3': 0.0353, '-2': 0.0363, '-1': 0.0342, '0': 0.0354, '1': 0.0326, '2': 0.0328, '3': 0.0321}

---
Runtime: 72.4 s
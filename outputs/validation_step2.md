# Validation, step 2

## Metric unit tests

- OK: perfect list: AP@5=1.000000 (expected 1.000000)
- OK: hits at k=2 and k=4, |truth|=2: AP@5=0.500000 (expected 0.500000)
- OK: single hit at k=5, |truth|=6 -> denominator 5: AP@5=0.040000 (expected 0.040000)
- OK: empty truth: AP@5=0.000000 (expected 0.000000)

## Join (view -> schedule), per period

### views of weeks 11-20

- views: 3,945,173; distinct users 4,876
- dropped, channel absent from schedule: 113,854 views (0.0289), 8 distinct vsetv_id, 123,503,248 s of 4,929,817,616 view seconds
- channel present but no overlapping schedule row: 255,954 views
- views with any overlap: 3,575,365; with a labelled overlap: 1,897,799 (0.4810 of all views)
- matched seconds: 4,478,653,208; on tv_show_id == 0: 2,023,786,859 (**0.4519**); on labelled: 2,454,866,349
- labelled user-programme pairs (interaction rows): 857,222

### views of weeks 21-30

- views: 1,876,814; distinct users 4,820
- dropped, channel absent from schedule: 40,549 views (0.0216), 6 distinct vsetv_id, 51,994,348 s of 3,689,818,808 view seconds
- channel present but no overlapping schedule row: 4,503 views
- views with any overlap: 1,831,762; with a labelled overlap: 1,065,602 (0.5678 of all views)
- matched seconds: 3,626,800,856; on tv_show_id == 0: 1,507,741,879 (**0.4157**); on labelled: 2,119,058,977
- labelled user-programme pairs (interaction rows): 658,368

## Diagnostics

- pool_size: 6361
- train_positive_pairs: 582559
- train_positive_users: 4834
- val_positive_pairs: 478219
- val_positive_users: 4799
- val_positive_pairs_in_pool: 478219
- share_val_positives_in_pool: 1.0
- repeat_rate_val_positive_also_train_positive: 0.211298589140122
- repeat_pairs: 101047
- repeat_rate_among_users_with_train_positive: 0.21231483621540984
- val_pos_per_user_quantiles: {'0.1': 19.0, '0.25': 42.0, '0.5': 81.0, '0.75': 139.0, '0.9': 205.0}
- train_pos_per_user_quantiles: {'0.1': 30.0, '0.25': 61.0, '0.5': 107.0, '0.75': 162.0, '0.9': 224.0}
- eval_a_users: 4799
- eval_a_cold_no_train_positive: 56
- eval_a_cold_no_train_positive_in_pool: 59
- submission_users: 1260
- submission_users_nonempty_truth: 1246
- submission_cold_no_train_positive: 10
- submission_cold_no_train_positive_in_pool: 10
- submission_cold_with_nonempty_truth: 8

## Results (threshold 300 s, k=5, pool = 6361 programmes airing in weeks 21-30)

| method | MAP@5 (a) users with truth | MAP@5 (b) 1,260 submission users | (a) with in-pool TRAIN history | (a) without |
|---|---|---|---|---|
| P popularity | 0.6358 (n=4799) | 0.7431 (n=1260) | 0.6394 (n=4740) | 0.3463 (n=59) |
| H own-history | 0.5760 (n=4799) | 0.6797 (n=1260) | 0.5788 (n=4740) | 0.3463 (n=59) |

Popularity top-10 (distinct TRAIN users with a positive pair):

- 2400480 | 'ТСН. Неделя'. | Инфо | 4390 users
- 240081 | Факты. День. | Инфо | 4098 users
- 2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо | 3698 users
- 6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное | 3671 users
- 2400467 | 'Викна-Новости'. | Инфо | 3590 users
- 10002514 | Т/с 'Кухня', 1 сезон, 1-4 с. | Сериалы | 3115 users
- 200352 | 'Вечерний квартал'. | Развлечения | 3012 users
- 6700482 | 'Секретные материалы'. | Другое | 2969 users
- 20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения | 2814 users
- 700389 | Кто сверху?, 6 сезон, 7 эп. | Развлечения | 2769 users

## Hand check

```
Submission user 8377619604347126107 (first in template order with >=5 validation positives and >=1 in-pool TRAIN positive); validation positives: 92
**Method P** top-5:
  1. id=2400480 | 'ТСН. Неделя'. | Инфо
  2. id=240081 | Факты. День. | Инфо
  3. id=2400508 | Сегодня. Итоги с Олегом Панютой. | Инфо
  4. id=6500479 | 'Мир наизнанку. Бразилия', 36 эп. | Познавательное
  5. id=2400467 | 'Викна-Новости'. | Инфо
  truth set (92): [20088, 100076, 200337, 200338, 200344, 200352, 200432, 200477, 200506, 240081, 400294, 400335, 400356, 400361, 400387, 400426, 500315, 500331, 500346, 700369, 700389, 1100411, 1100412, 2400467, 2400480, 2400508, 2500430, 5200303, 5800509, 5900256, 5900347, 5900348, 6000483, 6200371, 6200392, 6300336, 6500479, 6600437, 6700482, 10001499, 10002514, 10002515, 10002697, 10002947, 10003011, 10003167, 12002172, 12002896, 12002955, 12003004, 90021642, 90025700, 90026021, 90030050, 90032868, 90034561, 90036372, 90037203, 90038464, 90041065, 90041275, 90041425, 90041910, 90042851, 90044740, 90044905, 90045291, 90046928, 90046966, 90048220, 90048751, 90050976, 90052055, 90053203, 90056386, 90059058, 90062152, 90063447, 90069026, 90070560, 90073996, 90077794, 90078182, 90079260, 90079286, 90079688, 90079742, 90081012, 90081922, 90082182, 90082234, 90082814]
  rel vector: [1, 1, 1, 1, 1]
  k=1: hits=1, P@1=1/1=1.0000, rel=1 -> 1.0000
  k=2: hits=2, P@2=2/2=1.0000, rel=1 -> 1.0000
  k=3: hits=3, P@3=3/3=1.0000, rel=1 -> 1.0000
  k=4: hits=4, P@4=4/4=1.0000, rel=1 -> 1.0000
  k=5: hits=5, P@5=5/5=1.0000, rel=1 -> 1.0000
  AP@5 = 5.0000 / min(5,92)=5 = 1.000000  (function: 1.000000)
**Method H** top-5:
  1. id=2400480 | 'ТСН. Неделя'. | Инфо
  2. id=240081 | Факты. День. | Инфо
  3. id=20088 | 'Завтрак с 1+1'. Информационно-развлекательная программа. | Развлечения
  4. id=200337 | 'Утро с 'Интером'. | Развлечения
  5. id=200352 | 'Вечерний квартал'. | Развлечения
  truth set (92): [20088, 100076, 200337, 200338, 200344, 200352, 200432, 200477, 200506, 240081, 400294, 400335, 400356, 400361, 400387, 400426, 500315, 500331, 500346, 700369, 700389, 1100411, 1100412, 2400467, 2400480, 2400508, 2500430, 5200303, 5800509, 5900256, 5900347, 5900348, 6000483, 6200371, 6200392, 6300336, 6500479, 6600437, 6700482, 10001499, 10002514, 10002515, 10002697, 10002947, 10003011, 10003167, 12002172, 12002896, 12002955, 12003004, 90021642, 90025700, 90026021, 90030050, 90032868, 90034561, 90036372, 90037203, 90038464, 90041065, 90041275, 90041425, 90041910, 90042851, 90044740, 90044905, 90045291, 90046928, 90046966, 90048220, 90048751, 90050976, 90052055, 90053203, 90056386, 90059058, 90062152, 90063447, 90069026, 90070560, 90073996, 90077794, 90078182, 90079260, 90079286, 90079688, 90079742, 90081012, 90081922, 90082182, 90082234, 90082814]
  rel vector: [1, 1, 1, 1, 1]
  k=1: hits=1, P@1=1/1=1.0000, rel=1 -> 1.0000
  k=2: hits=2, P@2=2/2=1.0000, rel=1 -> 1.0000
  k=3: hits=3, P@3=3/3=1.0000, rel=1 -> 1.0000
  k=4: hits=4, P@4=4/4=1.0000, rel=1 -> 1.0000
  k=5: hits=5, P@5=5/5=1.0000, rel=1 -> 1.0000
  AP@5 = 5.0000 / min(5,92)=5 = 1.000000  (function: 1.000000)
```

## Final-run format self-test (not a submission)

- file: outputs\cache\_final_run_format_check_P.csv
- pool_size_weeks_31_42: 6632
- header_equals_template: True
- rows: 1260
- user_order_equals_template: True
- all_rows_five_distinct_ids: True
- all_ids_in_final_pool: True
- train_positive_pairs_weeks_11_30: 962976
- seconds: 12.0

## Runtime

- interactions_load_or_build_s: 50.6
- total_s: 94.5
- interactions build (when not cached): 50.4 s
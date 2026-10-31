# Validation, step 3: MF, CBF, hybrids

Protocol of step 2 (TRAIN weeks 11-20, truth weeks 21-30, threshold 300 s, pool 6,361, repeats allowed, seed 42). Validation users with truth: 4,799 -> TUNE 2,399 / REPORT 2,400 (random 50/50, seed 42). Submission users: 1,260 (610 in TUNE, 636 in REPORT, rest have empty truth), so the submission column partly overlaps the tuning half.

## MAP@5

| method | REPORT half | full validation | 1,260 submission users | TUNE half |
|---|---|---|---|---|
| P popularity | 0.6322 | 0.6358 | 0.7431 | 0.6395 |
| H own-history (diagnostic) | 0.5746 | 0.5760 | 0.6797 | 0.5773 |
| MF (ALS) f=32 reg=0.1 alpha=10 | 0.1391 | 0.1449 | 0.1270 | 0.1507 |
| CBF | 0.0410 | 0.0434 | 0.0495 | 0.0458 |
| Hybrid pop+MF (w_mf=0.6) | 0.6339 | 0.6373 | 0.7444 | 0.6407 |
| Hybrid pop+CBF (w_cbf=0.1) | 0.6323 | 0.6359 | 0.7432 | 0.6396 |
| Hybrid pop+MF+CBF (w_mf=0.6, w_cbf=0.0) | 0.6339 | 0.6373 | 0.7444 | 0.6407 |

Best by REPORT half: **Hybrid pop+MF** (0.6339). Best hybrid by REPORT: Hybrid pop+MF. Step-2 reproduction check: {'P_full_matches_step2': True, 'H_full_matches_step2': True}

## MF grid (15 iterations, pure MF with popularity for users without TRAIN history; config chosen by TUNE MAP@5)

| factors | reg | alpha | fit s | score s | TUNE | REPORT | full | submission | best hybrid w (TUNE) | hybrid TUNE |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.01 | 10 | 11.26 | 0.95 | 0.1465 | 0.1376 | 0.1420 | 0.1256 | 0.6 | 0.6409 |
| 32 | 0.01 | 40 | 9.27 | 0.46 | 0.0640 | 0.0564 | 0.0602 | 0.0472 | 0.0 | 0.6395 |
| 32 | 0.1 | 10 | 3.4 | 0.41 | 0.1507 | 0.1391 | 0.1449 | 0.1270 | 0.6 | 0.6407 |
| 32 | 0.1 | 40 | 3.29 | 0.39 | 0.0655 | 0.0572 | 0.0613 | 0.0505 | 0.0 | 0.6395 |
| 64 | 0.01 | 10 | 3.57 | 0.42 | 0.1106 | 0.1024 | 0.1065 | 0.0889 | 0.7 | 0.6411 |
| 64 | 0.01 | 40 | 3.66 | 0.36 | 0.0581 | 0.0541 | 0.0561 | 0.0478 | 0.0 | 0.6395 |
| 64 | 0.1 | 10 | 3.45 | 0.32 | 0.1099 | 0.1051 | 0.1075 | 0.0899 | 0.6 | 0.6408 |
| 64 | 0.1 | 40 | 3.63 | 0.28 | 0.0597 | 0.0544 | 0.0570 | 0.0498 | 0.0 | 0.6395 |

Chosen MF config: {'factors': 32, 'reg': 0.1, 'alpha': 10}. ALS trained on 4,841 users x 2,976 items, 857,222 observed pairs; pool items with factors: 2,198 of 6,361.

## Hybrid weights chosen on TUNE

- pop+MF: w_mf = 0.6 (TUNE MAP@5 0.6407); curve: 0.0:0.6395, 0.1:0.6395, 0.2:0.6395, 0.3:0.6395, 0.4:0.6399, 0.5:0.6405, 0.6:0.6407, 0.7:0.6401, 0.8:0.6316, 0.9:0.3801, 1.0:0.1507
- pop+CBF: w_cbf = 0.1 (TUNE 0.6396); curve: 0.0:0.6395, 0.1:0.6396, 0.2:0.6293, 0.3:0.6004, 0.4:0.5404, 0.5:0.4144, 0.6:0.2080, 0.7:0.1187, 0.8:0.0942, 0.9:0.0747, 1.0:0.0458
- pop+MF+CBF: w_mf = 0.6, w_cbf = 0.0, w_pop = 0.4 (TUNE 0.6407)

## CBF features

- items 9,671; block dims {'category': 8, 'genre': 65, 'people_tfidf': 16796, 'decade': 12}
- metadata coverage: share of labelled programmes with at least one non-category feature = 0.9996 (pool: 0.9997); with genre 0.9761, people 0.9859, decade 0.9986
- users with a non-empty CBF profile: eval 4,743 / 4,799, submission 1,250 / 1,260

## Cost and recommendation statistics

Fit/score seconds are for the 4.8k validation users x 6,361 items on this machine; hybrids include their components. Statistics are over the validation users with truth: mean TRAIN popularity (distinct TRAIN positive users) of recommended items; repeat share = share of recommended items already a TRAIN positive (>= 300 s) / having any labelled TRAIN pair for that user.

| method | fit s | score s | mean train popularity | repeat share (TRAIN positive) | repeat share (any pair) | lists needing pop fill (eval) | (submission) |
|---|---|---|---|---|---|---|---|
| P popularity | 0.18 | 0.19 | 3889.4 | 0.8022 | 0.8466 | 0 | 0 |
| H own-history (diagnostic) | 1.63 | 2.48 | 2387.4 | 0.9836 | 0.9838 | 98 | 16 |
| MF (ALS) | 3.4 | 0.75 | 1082.3 | 0.3681 | 0.4511 | 53 | 10 |
| CBF | 9.05 | 1.98 | 381.4 | 0.2328 | 0.2502 | 56 | 10 |
| Hybrid pop+MF | 3.4 | 1.15 | 3883.1 | 0.8029 | 0.8474 | 53 | 10 |
| Hybrid pop+CBF | 9.05 | 2.25 | 3889.4 | 0.8023 | 0.8466 | 56 | 10 |
| Hybrid pop+MF+CBF | 3.4 | 0.76 | 3883.1 | 0.8029 | 0.8474 | 53 | 10 |

Fill definition: P: is the popularity list; H: lists with < 5 own in-pool TRAIN positives (rest filled by P); MF: lists where a used component is invalid for the user (its weight goes to popularity); Hybrid pop+MF: lists where a used component is invalid for the user (its weight goes to popularity)

## Final submissions (TRAIN weeks 11-30, pool = weeks 31-42, no re-tuning)

- final pool 6,632 programmes; MF factors exist for 3,111 of them; popularity > 0 for 3,072
- TRAIN positive pairs 962,976; MF valid users 1258 / 1260; CBF valid users 1258 / 1260

- outputs\submissions\submission_P.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; spec None
- outputs\submissions\submission_MF.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; spec {'w_mf': 1.0}
- outputs\submissions\submission_CBF.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; spec {'w_cbf': 1.0}
- outputs\submissions\submission_HYBRID.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; spec {'w_mf': 0.6} (Hybrid pop+MF)

Runtime: 169.3 s

## Choice of strategy

By REPORT-half MAP@5 the best strategy is the pop+MF hybrid (w_mf = 0.6, 0.6339), but it beats plain popularity (0.6322) by only 0.0017; this gap was not tested for significance and is small relative to the TUNE/REPORT difference of P itself (0.6395 vs 0.6322). The personal methods alone are far worse than popularity (MF 0.139, CBF 0.041), because the validation truth is dominated by programmes almost everyone watches (P's lists have mean train popularity 3,889 vs 1,082 for MF and 381 for CBF). The pop+CBF hybrid is essentially popularity (w_cbf = 0.1), and the three-way blend selected w_cbf = 0 on TUNE, so it is identical to pop+MF. The best MF config sits at the edge of the grid (smallest alpha = 10, reg 0.1, 32 factors), so a lower alpha was not explored.

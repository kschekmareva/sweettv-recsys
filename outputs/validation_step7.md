# Validation, step 7

Primary criterion VTOP5 (pre-declared in step 6); V300/V900/VHALF are robustness checks. Same TUNE/REPORT halves (seed 42), pool, repeats allowed, 5 distinct ids, popularity fill (P-air). No Kaggle number is used for any choice. REPORT users with non-empty truth in a variant only.

## 1. Ceiling diagnostics (VTOP5 truth, validation weeks 21-30)

Truth items: 21,689 (all users with truth), 10,891 (REPORT). Mean candidate-union size per user: 97 (5 components), 132 (with EW).

| share of truth items that ... | all validation users | REPORT |
|---|---|---|
| (a) the user watched at all in TRAIN (weeks 11-20) | 0.7233 | 0.7227 |
| (b) the user completed >= 80% at least once in TRAIN | 0.5590 | 0.5640 |
| (c) are in the union of the top-50 of H2, PPR, MF, AR and P-air | 0.7925 | 0.7898 |
| (c') same union plus EW (reranker candidates) | 0.8169 | 0.8120 |

## 2. EW (expected watches)

| m | recency c | global completion rate | TUNE V300 / V900 / VHALF / VTOP5 |
|---|---|---|---|
| 1 | 1.0 | 0.0032 | 0.3681 / 0.3481 / 0.3392 / 0.2251 |
| 3 | 1.0 | 0.0032 | 0.3698 / 0.3492 / 0.3425 / 0.2294 |
| 10 | 1.0 | 0.0032 | 0.3580 / 0.3392 / 0.3359 / 0.2308 |
| 1 | 0.5 | 0.0031 | 0.3678 / 0.3481 / 0.3400 / 0.2249 |
| 3 | 0.5 | 0.0031 | 0.3685 / 0.3488 / 0.3426 / 0.2289 |
| 10 | 0.5 | 0.0031 | 0.3589 / 0.3401 / 0.3381 / 0.2327 |

Chosen on TUNE VTOP5: m = 10, c = 0.5. EW-blend (EW in place of H2; weights re-chosen on TUNE VTOP5): {'MF': 0.3, 'PPR': 0.2, 'EW': 0.3}, P-air 0.2 (TUNE by variant [0.685, 0.6342, 0.5644, 0.3479]).

## 3. Learned reranker

Candidates: top-50 union of H2, EW, PPR, MF, AR, P-air; 633,881 (user, show) pairs, 17,717 positives (VTOP5). Fit on TUNE users only: {'train_users': 1847, 'holdout_users': 462, 'train_pairs': 242987}; seed 42; early stopping on the held-out 20%. Objective chosen by held-out MAP@5 (VTOP5):

| objective | best iteration | held-out MAP@5 (VTOP5) |
|---|---|---|
| lambdarank | 174 | 0.4132 |
| binary | 197 | 0.4094 |

Chosen: **lambdarank**. Top-10 feature importances (gain share): {'r_H2': 0.4476, 's_H2': 0.147, 'show_n_target_air': 0.0555, 's_P-air': 0.0498, 'r_MF': 0.0317, 'r_AR': 0.0263, 'r_PPR': 0.0254, 'u_share_show_channel': 0.0234, 'show_duration': 0.0228, 'r_P-air': 0.019}.

Final-run rule: the fitted model is applied, without re-fitting, to features computed from TRAIN window = weeks 21-30 (same 10-week length as the training window 11-20) plus the 31-42 schedule for n_target_airings and the target pool.

## 4. REPORT MAP@5

| strategy | V300 | V900 | VHALF | VTOP5 | mean | fit s | score s | mean train popularity / repeat share (VTOP5 REPORT users) |
|---|---|---|---|---|---|---|---|---|
| P | 0.6322 | 0.5686 | 0.4745 | 0.2699 | 0.4863 | 0.0 | 0.0 | 3889 / 0.82 |
| H2 | 0.5834 | 0.5346 | 0.5134 | 0.3482 | 0.4949 | 0.02 | 0.2 | 2593 / 0.96 |
| EW | 0.3584 | 0.3348 | 0.3310 | 0.2285 | 0.3132 | 0.0 | 0.47 | 1537 / 0.53 |
| step-6 blend | 0.6548 | 0.6043 | 0.5554 | 0.3630 | 0.5444 | 3.03 | 18.88 | 3100 / 0.97 |
| EW-blend | 0.6802 | 0.6281 | 0.5533 | 0.3433 | 0.5512 | 3.01 | 19.16 | 3179 / 0.97 |
| reranker | 0.6819 | 0.6281 | 0.5954 | 0.4047 | 0.5775 | 17.75 | 31.2 | 3021 / 0.92 |
| oracle (6-component top-50 union) | 0.9589 | 0.9414 | 0.9055 | 0.7959 | 0.9004 |  |  |  |
| oracle (5-component top-50 union) | 0.9575 | 0.9391 | 0.9007 | 0.7742 | 0.8929 |  |  |  |

Oracle rows: candidates ranked perfectly, AP@5 = min(5, hits in union) / min(5, |truth|); they are the ceiling of any reranker over those candidates.

## 5. Paired bootstrap vs the step-6 blend (REPORT, 2,000 resamples, seed 42): mean AP@5 difference [95% CI]

| new strategy | V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|
| EW | -0.2965 [-0.3079, -0.2849] | -0.2695 [-0.2798, -0.2595] | -0.2244 [-0.2338, -0.2146] | -0.1345 [-0.1423, -0.1266] |
| EW-blend | +0.0254 [+0.0197, +0.0310] | +0.0239 [+0.0181, +0.0296] | -0.0021 [-0.0074, +0.0034] | -0.0197 [-0.0241, -0.0158] |
| reranker | +0.0270 [+0.0209, +0.0335] | +0.0238 [+0.0179, +0.0296] | +0.0400 [+0.0341, +0.0461] | +0.0417 [+0.0368, +0.0468] |

## 6. Shipping decision

Rule: ship only if the best new strategy beats the step-6 blend on REPORT VTOP5 with the 95% CI of the paired difference excluding zero. Best new strategy on REPORT VTOP5: **reranker**; ship: **True**.

- outputs\submissions\submission_STEP7.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True
- TRAIN window ['21-30']; users without component data (popularity fallback): {'H2': 23, 'EW': 14, 'PPR': 14, 'MF': 14, 'AR': 14}; repeat share 0.924

Runtime: 221.7 s (stage clocks: {'setup': 49.0, 'components': 73.7, 'ew': 79.5, 'ew_blend': 154.7, 'ceiling': 158.8, 'reranker': 186.7, 'evaluation': 189.3})

## Winner

The LightGBM reranker (lambdarank) reaches 0.4047 on REPORT VTOP5 against 0.3630 for the step-6 blend (+0.0417, 95% CI [+0.0368, +0.0468]), and it also beats the step-6 blend on the three robustness variants (+0.0270, +0.0238, +0.0400; all CIs exclude zero), so the shipping rule is met. The candidate union holds 81.7% of VTOP5 truth items while only 72.3% were watched at all in TRAIN and 55.9% were completed at least once, and a perfect reranker over the union would score 0.7959, so the reranker closes about 0.04 of a gap of about 0.43 above the step-6 blend. EW alone is weak (VTOP5 0.2285 against 0.2699 for P) and EW in place of H2 in the blend is better on V300 and V900 (+0.0254, +0.0239) but worse on VTOP5 (-0.0197, CI [-0.0241, -0.0158]). The reranker's top feature by gain is the H2 rank (0.4476) followed by the H2 score (0.1470); its recommendations have repeat share 0.92 against 0.97 for the step-6 blend. For the final file the model is applied without re-fitting to features from weeks 21-30, where 14 submission users have no data for most components (23 for H2) and fall back to popularity, and where the target window (weeks 31-42, 12 weeks) is longer than the 10 weeks used in training, which shifts the raw n_target_airings feature (4th-largest gain share, 0.0555).

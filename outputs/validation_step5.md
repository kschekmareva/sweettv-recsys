# Validation, step 5

Idea source for the airing-level truth and H2 (concept only, no code copied): kinahem, 'gornyaki final solution' (Kaggle notebook), https://www.kaggle.com/code/kinahem/gornyaki-final-solution - concept only, no code copied.

Same TUNE/REPORT halves, seed 42, repeats allowed, 5 distinct ids, popularity fill. Only users with non-empty truth in a variant count for that variant (submission column: all 1,260 users, empty truth = 0). Fusion: per-user min-max, weights chosen on TUNE, 1e-6 popularity tie-break; invalid-component weight goes to popularity (P-air). Pure H2/MF/PPR/AR rows fall back to P-air.

## 0. Airing-level join and truth variants

Airing table consistency (weeks 21-30): {'airing_rows_val': 1188848, 'user_show_pairs': 658368, 'interaction_pairs_val': 658368, 'share_max_pct_equals_max_frac': 1.0, 'airings_with_pct_ge_0.8': 295883}

| variant | positive pairs | users with truth | TUNE users | REPORT users | positives/user p10 / p25 / p50 / p75 / p90 | mean | submission users with truth |
|---|---|---|---|---|---|---|---|
| V300 | 478,219 | 4,799 | 2,399 | 2,400 | 19 / 42 / 81 / 139 / 205 | 99.6 | 1246 |
| V900 | 364,507 | 4,780 | 2,388 | 2,392 | 13 / 29 / 60 / 108 / 161 | 76.3 | 1245 |
| VHALF | 209,769 | 4,729 | 2,354 | 2,375 | 6 / 14 / 30 / 59 / 103 | 44.4 | 1244 |
| VTOP5 | 21,689 | 4,643 | 2,309 | 2,334 | 4 / 5 / 5 / 5 / 5 | 4.7 | 1238 |

VTOP5: 4,030 users have exactly 5 truth items (2019 in REPORT). Shows with >= 1 airing at view_pct >= 0.8 per user (before the top-5 cut) p10/25/50/75/90: 4 / 8 / 17 / 38 / 77.

P (popularity) MAP@5: V300: REPORT 0.6322, full 0.6358, submission 0.7431; V900: REPORT 0.5686, full 0.5727, submission 0.6737; VHALF: REPORT 0.4745, full 0.4822, submission 0.5535; VTOP5: REPORT 0.2699, full 0.2709, submission 0.2959
; VTOP5 exactly-5 subset: REPORT 0.2852, full 0.2854.
Plausibility check only (nothing tuned to it): Kaggle private score of P is 0.278; P under VTOP5 on the 1,260 submission users is 0.2959.

## 1. H2 (c chosen on TUNE VTOP5, pure H2)

| c | TUNE V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|
| 1.0 | 0.5799 | 0.5362 | 0.5157 | 0.3452 |
| 0.8 | 0.5814 | 0.5375 | 0.5170 | 0.3461 |
| 0.5 | 0.5836 | 0.5394 | 0.5177 | 0.3473 |

Chosen c = 0.5. sum over airings with view_pct>=0.8 of r*c^(days_before_train_end/60), r=5 if view_pct=1 else 2+2*view_pct^2, times log10(1+airings of the show in the TRAIN schedule); train end 2020-05-18 00:00 (validation), pure-H2 fallback and ties: P-air. Airings with view_pct >= 0.8 in TRAIN: 319,978; preparation 0.03 s.

## 2. Candidate-pool option (TUNE, P and H2)

| option | pool | P: V300 / V900 / VHALF / VTOP5 | H2: V300 / V900 / VHALF / VTOP5 | qualifies |
|---|---|---|---|---|
| none | 6361 | 0.6395 / 0.5768 / 0.4900 / 0.2720 | 0.5836 / 0.5394 / 0.5177 / 0.3473 | - |
| no_films | 1031 | 0.6395 / 0.5768 / 0.4900 / 0.2720 | 0.5923 / 0.5458 / 0.5228 / 0.3505 | False |
| no_long | 5908 | 0.6395 / 0.5768 / 0.4900 / 0.2720 | 0.5877 / 0.5430 / 0.5209 / 0.3488 | False |
| no_films_no_long | 937 | 0.6395 / 0.5768 / 0.4900 / 0.2720 | 0.5965 / 0.5494 / 0.5261 / 0.3520 | False |

Adopted: **none**. Rule: adopt only if both P and H2 improve on TUNE VTOP5 and neither loses more than 0.005 on V300/V900/VHALF; among qualifying options take the largest summed VTOP5 gain; programme duration = median airing duration in the pool-defining weeks.

## 3. MAP@5 on the REPORT half

| method | V300 | V900 | VHALF | VTOP5 | mean | avg rank |
|---|---|---|---|---|---|---|
| P | 0.6322 | 0.5686 | 0.4745 | 0.2699 | 0.4863 | 9.25 |
| P-air | 0.6315 | 0.5678 | 0.4806 | 0.2748 | 0.4887 | 9.25 |
| H | 0.5746 | 0.5388 | 0.4406 | 0.2361 | 0.4476 | 12.75 |
| H2 | 0.5834 | 0.5346 | 0.5134 | 0.3482 | 0.4949 | 9.50 |
| MF | 0.5199 | 0.4540 | 0.3458 | 0.1525 | 0.3680 | 14.00 |
| PPR | 0.6028 | 0.5665 | 0.4646 | 0.2580 | 0.4730 | 11.25 |
| AR | 0.6365 | 0.5721 | 0.4851 | 0.2777 | 0.4929 | 7.75 |
| step-4 blend | 0.6909 | 0.6408 | 0.5429 | 0.3180 | 0.5482 | 3.25 |
| H2+P-air | 0.6400 | 0.5787 | 0.5383 | 0.3594 | 0.5291 | 4.50 |
| H2+P-air [shared w] | 0.6384 | 0.5814 | 0.5345 | 0.3464 | 0.5252 |  |
| H2+PPR | 0.6241 | 0.5836 | 0.5229 | 0.3482 | 0.5197 | 6.50 |
| H2+PPR [shared w] | 0.6215 | 0.5790 | 0.5196 | 0.3341 | 0.5135 |  |
| H2+AR | 0.6519 | 0.5921 | 0.5399 | 0.3587 | 0.5356 | 3.75 |
| H2+AR [shared w] | 0.6466 | 0.5894 | 0.5369 | 0.3478 | 0.5302 |  |
| H2+MF | 0.6187 | 0.5646 | 0.5293 | 0.3526 | 0.5163 | 7.75 |
| H2+MF [shared w] | 0.6141 | 0.5631 | 0.5293 | 0.3517 | 0.5146 |  |
| P-air+MF+PPR+H2 | 0.6921 | 0.6395 | 0.5566 | 0.3630 | 0.5628 | 1.50 |
| P-air+MF+PPR+H2 [shared w] | 0.6895 | 0.6395 | 0.5505 | 0.3345 | 0.5535 |  |
| H2-first + step-4 blend | 0.6909 | 0.6408 | 0.5429 | 0.3451 | 0.5549 | 3.00 |
| H2-first + step-4 blend [shared w] | 0.6909 | 0.6408 | 0.5429 | 0.3180 | 0.5482 |  |

## 4. MAP@5 on all validation users with truth

| method | V300 | V900 | VHALF | VTOP5 | mean | avg rank |
|---|---|---|---|---|---|---|
| P | 0.6358 | 0.5727 | 0.4822 | 0.2709 | 0.4904 |  |
| P-air | 0.6352 | 0.5720 | 0.4884 | 0.2758 | 0.4929 |  |
| H | 0.5760 | 0.5433 | 0.4459 | 0.2363 | 0.4504 |  |
| H2 | 0.5835 | 0.5370 | 0.5155 | 0.3478 | 0.4960 |  |
| MF | 0.5229 | 0.4563 | 0.3481 | 0.1529 | 0.3700 |  |
| PPR | 0.6050 | 0.5716 | 0.4720 | 0.2573 | 0.4765 |  |
| AR | 0.6402 | 0.5763 | 0.4928 | 0.2806 | 0.4975 |  |
| step-4 blend | 0.6967 | 0.6473 | 0.5520 | 0.3198 | 0.5540 |  |
| H2+P-air | 0.6429 | 0.5825 | 0.5407 | 0.3589 | 0.5313 |  |
| H2+P-air [shared w] | 0.6386 | 0.5824 | 0.5387 | 0.3476 | 0.5268 |  |
| H2+PPR | 0.6257 | 0.5889 | 0.5271 | 0.3478 | 0.5224 |  |
| H2+PPR [shared w] | 0.6227 | 0.5836 | 0.5252 | 0.3340 | 0.5164 |  |
| H2+AR | 0.6528 | 0.5935 | 0.5428 | 0.3597 | 0.5372 |  |
| H2+AR [shared w] | 0.6475 | 0.5912 | 0.5406 | 0.3487 | 0.5320 |  |
| H2+MF | 0.6171 | 0.5652 | 0.5303 | 0.3520 | 0.5161 |  |
| H2+MF [shared w] | 0.6129 | 0.5636 | 0.5303 | 0.3505 | 0.5143 |  |
| P-air+MF+PPR+H2 | 0.6975 | 0.6466 | 0.5647 | 0.3644 | 0.5683 |  |
| P-air+MF+PPR+H2 [shared w] | 0.6950 | 0.6466 | 0.5605 | 0.3369 | 0.5597 |  |
| H2-first + step-4 blend | 0.6967 | 0.6473 | 0.5520 | 0.3472 | 0.5608 |  |
| H2-first + step-4 blend [shared w] | 0.6967 | 0.6473 | 0.5520 | 0.3198 | 0.5540 |  |

## 5. MAP@5 on the 1,260 submission users (partly overlaps TUNE)

| method | V300 | V900 | VHALF | VTOP5 | mean | avg rank |
|---|---|---|---|---|---|---|
| P | 0.7431 | 0.6737 | 0.5535 | 0.2959 | 0.5665 |  |
| P-air | 0.7417 | 0.6726 | 0.5610 | 0.3013 | 0.5692 |  |
| H | 0.6797 | 0.6441 | 0.5283 | 0.2614 | 0.5284 |  |
| H2 | 0.6748 | 0.6243 | 0.5906 | 0.3848 | 0.5686 |  |
| MF | 0.5726 | 0.5006 | 0.3654 | 0.1444 | 0.3958 |  |
| PPR | 0.7132 | 0.6767 | 0.5590 | 0.2855 | 0.5586 |  |
| AR | 0.7424 | 0.6740 | 0.5641 | 0.3050 | 0.5714 |  |
| step-4 blend | 0.7964 | 0.7481 | 0.6280 | 0.3482 | 0.6301 |  |
| H2+P-air | 0.7437 | 0.6815 | 0.6209 | 0.3961 | 0.6106 |  |
| H2+P-air [shared w] | 0.7377 | 0.6768 | 0.6170 | 0.3806 | 0.6030 |  |
| H2+PPR | 0.7320 | 0.6945 | 0.6116 | 0.3848 | 0.6057 |  |
| H2+PPR [shared w] | 0.7295 | 0.6865 | 0.6121 | 0.3706 | 0.5997 |  |
| H2+AR | 0.7530 | 0.6868 | 0.6168 | 0.3941 | 0.6127 |  |
| H2+AR [shared w] | 0.7441 | 0.6852 | 0.6183 | 0.3785 | 0.6065 |  |
| H2+MF | 0.6992 | 0.6485 | 0.6011 | 0.3891 | 0.5845 |  |
| H2+MF [shared w] | 0.7019 | 0.6503 | 0.6011 | 0.3846 | 0.5845 |  |
| P-air+MF+PPR+H2 | 0.7976 | 0.7472 | 0.6419 | 0.3991 | 0.6465 |  |
| P-air+MF+PPR+H2 [shared w] | 0.7950 | 0.7472 | 0.6395 | 0.3700 | 0.6379 |  |
| H2-first + step-4 blend | 0.7964 | 0.7481 | 0.6280 | 0.3821 | 0.6386 |  |
| H2-first + step-4 blend [shared w] | 0.7964 | 0.7481 | 0.6280 | 0.3482 | 0.6301 |  |

VTOP5 on the subset with exactly 5 truth items (REPORT / full): P: 0.2852 / 0.2854; P-air: 0.2905 / 0.2908; H: 0.2482 / 0.2475; H2: 0.3695 / 0.3687; MF: 0.1533 / 0.1543; PPR: 0.2721 / 0.2706; AR: 0.2936 / 0.2954; step-4 blend: 0.3359 / 0.3368; H2+P-air: 0.3817 / 0.3807; H2+PPR: 0.3695 / 0.3687; H2+AR: 0.3810 / 0.3812; H2+MF: 0.3746 / 0.3733; P-air+MF+PPR+H2: 0.3855 / 0.3863; H2-first + step-4 blend: 0.3658 / 0.3675

Winner by average REPORT rank: **P-air+MF+PPR+H2**; shipped row: **P-air+MF+PPR+H2 [shared w]**. Rule: winner = lowest average REPORT rank over the 4 variants (per-variant-tuned weights); shipped = the single weight set maximising the mean over the 4 variants of TUNE MAP@5 (ties: first in grid).

## 6. Hyperparameters

### MF (32 factors, reg 0.1; selection = mean over the 4 variants of the best P-air + MF TUNE MAP@5)

| alpha | fit s | score s | hybrid TUNE mean | best w per variant | pure MF TUNE V300 / V900 / VHALF / VTOP5 |
|---|---|---|---|---|---|
| 0.3 | 3.33 | 0.46 | 0.5198 | [0.6, 0.6, 0.6, 0.5] | 0.5258 / 0.4587 / 0.3504 / 0.1533 |
| 0.5 | 3.42 | 0.42 | 0.5170 | [0.6, 0.6, 0.6, 0.6] | 0.5108 / 0.4466 / 0.3455 / 0.1557 |
| 1 | 3.59 | 0.4 | 0.5086 | [0.6, 0.6, 0.7, 0.7] | 0.4621 / 0.4051 / 0.3220 / 0.1511 |
| 3 | 3.18 | 0.4 | 0.5031 | [0.6, 0.6, 0.3, 0.7] | 0.3120 / 0.2690 / 0.2168 / 0.1131 |

Chosen MF: {'factors': 32, 'reg': 0.1, 'alpha': 0.3} (step-4 config alpha=1 is kept as MF4 for the unchanged step-4 blend; identical to the chosen one: False). PPR: alpha 0.3, 13 iterations, converged True. AR: mode max, min support 20 (step-4 choices, not re-tuned). The older MF configs (alpha 10/40 etc.) were not re-run.

## 7. Weights (component weights; P-air gets the remainder)

| row | V300 | V900 | VHALF | VTOP5 | shared set (shipped rule) |
|---|---|---|---|---|---|
| H2+P-air | {'H2': 0.3} | {'H2': 0.3} | {'H2': 0.6} | {'H2': 0.8} | {'H2': 0.5} |
| H2+PPR | {'H2': 0.3, 'PPR': 0.7} | {'H2': 0.2, 'PPR': 0.8} | {'H2': 0.5, 'PPR': 0.5} | {'H2': 1.0, 'PPR': 0.0} | {'H2': 0.4, 'PPR': 0.6} |
| H2+AR | {'H2': 0.1, 'AR': 0.9} | {'H2': 0.2, 'AR': 0.8} | {'H2': 0.4, 'AR': 0.6} | {'H2': 0.6, 'AR': 0.4} | {'H2': 0.3, 'AR': 0.7} |
| H2+MF | {'H2': 0.5, 'MF': 0.5} | {'H2': 0.6, 'MF': 0.4} | {'H2': 0.7, 'MF': 0.3} | {'H2': 0.8, 'MF': 0.2} | {'H2': 0.7, 'MF': 0.3} |
| P-air+MF+PPR+H2 | {'MF': 0.3, 'PPR': 0.4, 'H2': 0.0} | {'MF': 0.3, 'PPR': 0.4, 'H2': 0.1} | {'MF': 0.2, 'PPR': 0.3, 'H2': 0.2} | {'MF': 0.2, 'PPR': 0.1, 'H2': 0.5} | {'MF': 0.3, 'PPR': 0.4, 'H2': 0.1} |
| H2-first + step-4 blend | {'n_H2_first': 0, 'MF4': 0.4, 'PPR': 0.4} | {'n_H2_first': 0, 'MF4': 0.4, 'PPR': 0.4} | {'n_H2_first': 0, 'MF4': 0.4, 'PPR': 0.4} | {'n_H2_first': 3, 'MF4': 0.4, 'PPR': 0.4} | {'n_H2_first': 0, 'MF4': 0.4, 'PPR': 0.4} |

## 8. Cost and recommendation statistics (users with truth)

Cells: mean TRAIN popularity (distinct TRAIN positive users) of recommended items / repeat share (already a TRAIN positive for the user). Times for ~4.8k users; hybrids include components.

| row | fit s | score s | V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|---|---|
| P | 0.0 | 0.0 | 3889 / 0.80 | 3889 / 0.80 | 3889 / 0.81 | 3889 / 0.81 |
| P-air | 0.0 | 0.0 | 3889 / 0.80 | 3889 / 0.80 | 3889 / 0.81 | 3889 / 0.81 |
| H | 0.5 | 0.68 | 2387 / 0.98 | 2386 / 0.98 | 2392 / 0.99 | 2401 / 0.99 |
| H2 | 0.03 | 0.61 | 2603 / 0.95 | 2600 / 0.95 | 2595 / 0.95 | 2586 / 0.96 |
| MF | 3.33 | 0.82 | 2676 / 0.88 | 2674 / 0.88 | 2674 / 0.89 | 2671 / 0.89 |
| PPR | 0.09 | 19.27 | 2559 / 0.98 | 2558 / 0.98 | 2565 / 0.99 | 2576 / 0.99 |
| AR | 0.76 | 2.54 | 3805 / 0.82 | 3805 / 0.82 | 3804 / 0.82 | 3803 / 0.83 |
| step-4 blend | 3.68 | 19.73 | 3232 / 0.98 | 3231 / 0.98 | 3236 / 0.98 | 3244 / 0.98 |
| H2+P-air | 0.03 | 0.86 | 3709 / 0.84 | 3708 / 0.84 | 3298 / 0.90 | 2957 / 0.94 |
| H2+P-air [shared w] | 0.03 | 1.53 | 3435 / 0.88 | 3434 / 0.88 | 3433 / 0.88 | 3432 / 0.89 |
| H2+PPR | 0.12 | 20.75 | 2599 / 0.98 | 2599 / 0.98 | 2595 / 0.98 | 2586 / 0.96 |
| H2+PPR [shared w] | 0.12 | 48.96 | 2595 / 0.98 | 2594 / 0.98 | 2599 / 0.98 | 2607 / 0.99 |
| H2+AR | 0.79 | 4.09 | 3707 / 0.84 | 3588 / 0.86 | 3329 / 0.90 | 3059 / 0.93 |
| H2+AR [shared w] | 0.79 | 3.96 | 3462 / 0.88 | 3461 / 0.88 | 3458 / 0.88 | 3454 / 0.89 |
| H2+MF | 3.36 | 1.46 | 2803 / 0.94 | 2777 / 0.94 | 2741 / 0.95 | 2683 / 0.96 |
| H2+MF [shared w] | 3.36 | 1.34 | 2745 / 0.95 | 2742 / 0.95 | 2741 / 0.95 | 2738 / 0.96 |
| P-air+MF+PPR+H2 | 3.45 | 20.12 | 3350 / 0.98 | 3196 / 0.98 | 3304 / 0.97 | 3096 / 0.96 |
| P-air+MF+PPR+H2 [shared w] | 3.45 | 20.15 | 3196 / 0.98 | 3196 / 0.98 | 3201 / 0.98 | 3207 / 0.98 |
| H2-first + step-4 blend | 3.71 | 20.24 | 3232 / 0.98 | 3231 / 0.98 | 3236 / 0.98 | 2943 / 0.98 |
| H2-first + step-4 blend [shared w] | 3.71 | 20.25 | 3232 / 0.98 | 3231 / 0.98 | 3236 / 0.98 | 3244 / 0.98 |

## 9. Significance: P-air+MF+PPR+H2 [shared w] vs P on REPORT (paired bootstrap, 2000 resamples, seed 42)

| variant | users | mean AP@5 difference | 95% CI |
|---|---|---|---|
| V300 | 2400 | +0.0574 | [+0.0492, +0.0660] |
| V900 | 2392 | +0.0709 | [+0.0627, +0.0792] |
| VHALF | 2375 | +0.0760 | [+0.0671, +0.0842] |
| VTOP5 | 2334 | +0.0645 | [+0.0574, +0.0715] |

## 9. Significance: P-air+MF+PPR+H2 [shared w] vs step-4 blend on REPORT (paired bootstrap, 2000 resamples, seed 42)

| variant | users | mean AP@5 difference | 95% CI |
|---|---|---|---|
| V300 | 2400 | -0.0014 | [-0.0048, +0.0022] |
| V900 | 2392 | -0.0013 | [-0.0050, +0.0024] |
| VHALF | 2375 | +0.0076 | [+0.0040, +0.0114] |
| VTOP5 | 2334 | +0.0165 | [+0.0135, +0.0196] |

## 10. Final submissions (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning)

Pool 6,632 (used: 6,632, option none); H2 c = 0.5; chosen: P-air+MF+PPR+H2 [shared w] with {'MF': 0.3, 'PPR': 0.4, 'H2': 0.1}.

- outputs\submissions\submission_H2.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True, ids in used pool True; spec {'kind': 'fuse', 'w': {'H2': 1.0}}; users without component data (popularity fallback): {'H2': 5}
- outputs\submissions\submission_CHOSEN_step5.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True, ids in used pool True; spec {'kind': 'fuse', 'w': {'MF': 0.3, 'PPR': 0.4, 'H2': 0.1}}; users without component data (popularity fallback): {'H2': 5, 'MF': 2, 'PPR': 2}

Runtime: 293.3 s

## Winner

The four-component blend (P-air 0.2, MF 0.3, PPR 0.4, H2 0.1 as the shipped shared weight set) has the best average REPORT rank (1.50 with per-variant weights) and beats P in every variant, by +0.057 / +0.071 / +0.076 / +0.065 AP@5 (V300 / V900 / VHALF / VTOP5), all with 95% CIs that exclude zero. Against the step-4 blend its shared-weight version is not distinguishable on V300 (-0.0014, CI [-0.0048, +0.0022]) and V900 (-0.0013, CI [-0.0050, +0.0024]) but is better on VHALF (+0.0076) and VTOP5 (+0.0165), where the CIs exclude zero. H2 alone is the best single component under VTOP5 (REPORT 0.3482 vs P 0.2699 and H 0.2361), but it is weaker than P on V300 and V900 (0.5834 vs 0.6322, 0.5346 vs 0.5686), which is why the shared weight on H2 stays at 0.1. With weights tuned on VTOP5 alone the same family reaches 0.3630 on VTOP5 REPORT (H2 weight 0.5) against 0.3345 for the shared set.

## Step 6 - decision rule

Pre-declared rule from now on: VTOP5 is the PRIMARY validation criterion, because it matches the task shape (exactly 5 ids per user; truth = the user's most-completed programmes; definition taken from the cited public notebook). V300, V900 and VHALF are robustness checks. The Kaggle number was used once, only to check that P's VTOP5 score (0.270-0.296) is plausible against the leaderboard (0.278); no weights or hyperparameters are chosen from Kaggle.

Family: P-air + MF + PPR + H2 (components as in step 5: H2 c = 0.5, MF {'factors': 32, 'reg': 0.1, 'alpha': 0.3}, PPR alpha 0.3, popularity base P-air, no pool filter). Weights chosen on TUNE under VTOP5 in step 5 (no new search): **MF 0.2, PPR 0.1, H2 0.5, P-air 0.2**. Step-5 shared-weight set: MF 0.3, PPR 0.4, H2 0.1, P-air 0.2. Step-4 blend: P-air 0.2, MF4 0.4, PPR 0.4.

REPORT MAP@5 (same TUNE/REPORT halves, seed 42):

| strategy | VTOP5 (primary) | V300 | V900 | VHALF |
|---|---|---|---|---|
| P | 0.2699 | 0.6322 | 0.5686 | 0.4745 |
| step-4 blend | 0.3180 | 0.6909 | 0.6408 | 0.5429 |
| step-5 blend (shared w) | 0.3345 | 0.6895 | 0.6395 | 0.5505 |
| step-6 FINAL (VTOP5-tuned w) | 0.3630 | 0.6548 | 0.6043 | 0.5554 |

Paired bootstrap of the step-6 strategy (VTOP5-tuned weights), 2,000 resamples, seed 42, REPORT users with non-empty truth: mean AP@5 difference [95% CI]

| vs | VTOP5 (primary) | V300 | V900 | VHALF |
|---|---|---|---|---|
| P | +0.0931 [+0.0855, +0.1013] | +0.0227 [+0.0136, +0.0320] | +0.0357 [+0.0265, +0.0450] | +0.0809 [+0.0716, +0.0898] |
| step-4 blend | +0.0450 [+0.0396, +0.0508] | -0.0361 [-0.0429, -0.0293] | -0.0366 [-0.0435, -0.0294] | +0.0125 [+0.0059, +0.0189] |
| step-5 blend (shared w) | +0.0286 [+0.0241, +0.0331] | -0.0347 [-0.0401, -0.0288] | -0.0352 [-0.0412, -0.0297] | +0.0049 [-0.0009, +0.0103] |

Repeat share (recommended items already a TRAIN positive for the user, validation users with truth): V300 0.96, V900 0.96, VHALF 0.96, VTOP5 0.96.

Final file (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning, no film filter):

- outputs\submissions\submission_FINAL_vtop5.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True
- weights MF 0.2, PPR 0.1, H2 0.5, P-air 0.2; pool 6,632 programmes
- fallback to popularity (no data for the component): {'H2': 5, 'MF': 2, 'PPR': 2}; users with at least one missing component: 5 of 1260
- repeat share on the final lists (items that are TRAIN-positive for the user): 0.991
- mean overlap with submission_CHOSEN_step5.csv: 0.759 of 5 ids per user (as a share)

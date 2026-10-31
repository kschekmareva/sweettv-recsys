# Validation, step 4

Training interactions as in step 3 (TRAIN weeks 11-20, positives >= 300 s). Only the validation truth changes. TUNE/REPORT halves are the step-3 halves of the V300 users; for V900/VHALF only users with non-empty truth in that variant count (submission column: all 1,260 users, empty truth = 0). Fusion adds a 1e-6 popularity tie-break; users for whom a component is invalid get popularity.

## 0. Truth variants

| variant | positive pairs | users with truth | TUNE users | REPORT users | positives/user p10 / p25 / p50 / p75 / p90 | mean | submission users with truth |
|---|---|---|---|---|---|---|---|
| V300 | 478,219 | 4,799 | 2,399 | 2,400 | 19 / 42 / 81 / 139 / 205 | 99.6 | 1246 |
| V900 | 364,507 | 4,780 | 2,388 | 2,392 | 13 / 29 / 60 / 108 / 161 | 76.3 | 1245 |
| VHALF | 209,769 | 4,729 | 2,354 | 2,375 | 6 / 14 / 30 / 59 / 103 | 44.4 | 1244 |

P (popularity) MAP@5: V300: REPORT 0.6322, full 0.6358, submission 0.7431; V900: REPORT 0.5686, full 0.5727, submission 0.6737; VHALF: REPORT 0.4745, full 0.4822, submission 0.5535

Popularity variants (mean TUNE MAP@5 over the 3 variants): {'P': 0.5687366829074222, 'P-rec': 0.5673033087881653, 'P-air': 0.5704906859113609}; best = **P-air** (base of all hybrids).

## 1. MAP@5 on the REPORT half

| method | V300 | V900 | VHALF | mean | avg rank |
|---|---|---|---|---|---|
| P | 0.6322 | 0.5686 | 0.4745 | 0.5584 | 9.00 |
| P-rec | 0.6350 | 0.5715 | 0.4675 | 0.5580 | 8.33 |
| P-air | 0.6315 | 0.5678 | 0.4806 | 0.5599 | 9.00 |
| H | 0.5746 | 0.5388 | 0.4406 | 0.5180 | 12.00 |
| MF | 0.4605 | 0.4029 | 0.3150 | 0.3928 | 13.00 |
| CBF | 0.0431 | 0.0329 | 0.0185 | 0.0315 | 14.00 |
| PPR | 0.6028 | 0.5665 | 0.4646 | 0.5446 | 11.00 |
| AR | 0.6365 | 0.5721 | 0.4851 | 0.5646 | 5.67 |
| pop+H | 0.6849 | 0.6351 | 0.5379 | 0.6193 | 2.33 |
| pop+H [shared w] | 0.6849 | 0.6351 | 0.5379 | 0.6193 |  |
| pop+MF | 0.6443 | 0.5817 | 0.4986 | 0.5748 | 4.00 |
| pop+MF [shared w] | 0.6443 | 0.5817 | 0.4966 | 0.5742 |  |
| pop+PPR | 0.6861 | 0.6326 | 0.5367 | 0.6185 | 2.67 |
| pop+PPR [shared w] | 0.6861 | 0.6348 | 0.5367 | 0.6192 |  |
| pop+AR | 0.6364 | 0.5725 | 0.4852 | 0.5647 | 5.33 |
| pop+AR [shared w] | 0.6364 | 0.5722 | 0.4852 | 0.5646 |  |
| pop+CBF | 0.6350 | 0.5712 | 0.4806 | 0.5623 | 7.33 |
| pop+CBF [shared w] | 0.6350 | 0.5712 | 0.4784 | 0.5616 |  |
| pop+blend2 | 0.6909 | 0.6408 | 0.5443 | 0.6254 | 1.00 |
| pop+blend2 [shared w] | 0.6909 | 0.6408 | 0.5429 | 0.6249 |  |

## 2. MAP@5 on all validation users with truth

| method | V300 | V900 | VHALF | mean | avg rank |
|---|---|---|---|---|---|
| P | 0.6358 | 0.5727 | 0.4822 | 0.5636 |  |
| P-rec | 0.6380 | 0.5751 | 0.4749 | 0.5626 |  |
| P-air | 0.6352 | 0.5720 | 0.4884 | 0.5652 |  |
| H | 0.5760 | 0.5433 | 0.4459 | 0.5217 |  |
| MF | 0.4613 | 0.4040 | 0.3185 | 0.3946 |  |
| CBF | 0.0456 | 0.0361 | 0.0211 | 0.0343 |  |
| PPR | 0.6050 | 0.5716 | 0.4720 | 0.5495 |  |
| AR | 0.6402 | 0.5763 | 0.4928 | 0.5698 |  |
| pop+H | 0.6883 | 0.6392 | 0.5467 | 0.6247 |  |
| pop+H [shared w] | 0.6883 | 0.6392 | 0.5467 | 0.6247 |  |
| pop+MF | 0.6467 | 0.5839 | 0.5036 | 0.5781 |  |
| pop+MF [shared w] | 0.6467 | 0.5839 | 0.5025 | 0.5777 |  |
| pop+PPR | 0.6893 | 0.6377 | 0.5457 | 0.6242 |  |
| pop+PPR [shared w] | 0.6893 | 0.6387 | 0.5457 | 0.6246 |  |
| pop+AR | 0.6402 | 0.5766 | 0.4928 | 0.5699 |  |
| pop+AR [shared w] | 0.6402 | 0.5763 | 0.4928 | 0.5698 |  |
| pop+CBF | 0.6380 | 0.5753 | 0.4884 | 0.5672 |  |
| pop+CBF [shared w] | 0.6380 | 0.5753 | 0.4857 | 0.5664 |  |
| pop+blend2 | 0.6967 | 0.6473 | 0.5536 | 0.6326 |  |
| pop+blend2 [shared w] | 0.6967 | 0.6473 | 0.5520 | 0.6320 |  |

## 3. MAP@5 on the 1,260 submission users (partly overlaps TUNE)

| method | V300 | V900 | VHALF | mean | avg rank |
|---|---|---|---|---|---|
| P | 0.7431 | 0.6737 | 0.5535 | 0.6568 |  |
| P-rec | 0.7454 | 0.6759 | 0.5438 | 0.6551 |  |
| P-air | 0.7417 | 0.6726 | 0.5610 | 0.6585 |  |
| H | 0.6797 | 0.6441 | 0.5283 | 0.6174 |  |
| MF | 0.4961 | 0.4362 | 0.3260 | 0.4195 |  |
| CBF | 0.0505 | 0.0396 | 0.0219 | 0.0374 |  |
| PPR | 0.7132 | 0.6767 | 0.5590 | 0.6497 |  |
| AR | 0.7424 | 0.6740 | 0.5641 | 0.6602 |  |
| pop+H | 0.7913 | 0.7418 | 0.6249 | 0.7193 |  |
| pop+H [shared w] | 0.7913 | 0.7418 | 0.6249 | 0.7193 |  |
| pop+MF | 0.7509 | 0.6813 | 0.5707 | 0.6676 |  |
| pop+MF [shared w] | 0.7509 | 0.6813 | 0.5745 | 0.6689 |  |
| pop+PPR | 0.7905 | 0.7413 | 0.6227 | 0.7182 |  |
| pop+PPR [shared w] | 0.7905 | 0.7397 | 0.6227 | 0.7176 |  |
| pop+AR | 0.7425 | 0.6761 | 0.5642 | 0.6609 |  |
| pop+AR [shared w] | 0.7425 | 0.6741 | 0.5642 | 0.6602 |  |
| pop+CBF | 0.7467 | 0.6781 | 0.5610 | 0.6620 |  |
| pop+CBF [shared w] | 0.7467 | 0.6781 | 0.5585 | 0.6611 |  |
| pop+blend2 | 0.7964 | 0.7481 | 0.6325 | 0.7257 |  |
| pop+blend2 [shared w] | 0.7964 | 0.7481 | 0.6280 | 0.7241 |  |

Winner by average REPORT rank: **pop+blend2** (avg rank 1.00). Shipped row: **pop+blend2 [shared w]**. Rule: winner = lowest average rank over the 3 truth variants on REPORT (per-variant-tuned weights); shipped weights = the single weight set (same for all variants) maximising the mean over the 3 variants of TUNE MAP@5.

## 4. Hyperparameters

### MF grid (pure-MF TUNE MAP per variant; selection = mean over variants of best pop+MF TUNE MAP)

| factors | reg | alpha | fit s | score s | pop+MF TUNE mean | best w per variant | pure MF TUNE V300 / V900 / VHALF |
|---|---|---|---|---|---|---|---|
| 32 | 0.01 | 10 | 10.1 | 1.01 | 0.5740 | [0.2, 0.2, 0.5] | 0.1465 / 0.1245 / 0.0987 |
| 32 | 0.01 | 40 | 10.17 | 0.95 | 0.5718 | [0.5, 0.5, 0.6] | 0.0641 / 0.0524 / 0.0400 |
| 32 | 0.1 | 10 | 9.82 | 1.1 | 0.5740 | [0.2, 0.2, 0.5] | 0.1507 / 0.1280 / 0.1026 |
| 32 | 0.1 | 40 | 9.95 | 0.95 | 0.5721 | [0.4, 0.4, 0.6] | 0.0655 / 0.0540 / 0.0419 |
| 64 | 0.01 | 10 | 11.01 | 0.93 | 0.5734 | [0.3, 0.2, 0.5] | 0.1106 / 0.0917 / 0.0715 |
| 64 | 0.01 | 40 | 7.29 | 0.4 | 0.5716 | [0.4, 0.4, 0.6] | 0.0581 / 0.0489 / 0.0373 |
| 64 | 0.1 | 10 | 3.62 | 0.37 | 0.5735 | [0.3, 0.3, 0.4] | 0.1100 / 0.0911 / 0.0714 |
| 64 | 0.1 | 40 | 3.71 | 0.41 | 0.5718 | [0.4, 0.4, 0.6] | 0.0597 / 0.0502 / 0.0381 |
| 32 | 0.1 | 1 | 3.28 | 0.44 | 0.5814 | [0.6, 0.6, 0.7] | 0.4621 / 0.4051 / 0.3220 |
| 32 | 0.1 | 3 | 3.36 | 0.4 | 0.5758 | [0.6, 0.6, 0.3] | 0.3120 / 0.2690 / 0.2168 |

Chosen MF: {'factors': 32, 'reg': 0.1, 'alpha': 1}

### PPR (alpha = link-following probability; power iteration, tol 1e-6, max 100)

| alpha | max iterations | converged (all chunks) | final max L1 change | seconds | pop+PPR TUNE mean | best w per variant | pure PPR TUNE V300 / V900 / VHALF |
|---|---|---|---|---|---|---|---|
| 0.3 | 13 | True | 4.08e-07 | 19.1 | 0.6300 | [0.6, 0.7, 0.6] | 0.6073 / 0.5767 / 0.4795 |
| 0.5 | 22 | True | 9.92e-07 | 31.4 | 0.6270 | [0.7, 0.7, 0.7] | 0.6453 / 0.6104 / 0.5128 |
| 0.7 | 42 | True | 9.97e-07 | 59.9 | 0.6190 | [0.8, 0.8, 0.8] | 0.6736 / 0.6279 / 0.5305 |
| 0.85 | 91 | True | 9.23e-07 | 131.5 | 0.6041 | [1.0, 1.0, 1.0] | 0.6743 / 0.6184 / 0.5198 |

Chosen alpha: 0.3. Graph: 4,834 users, 2,905 programmes, 582,559 edges.

### Association rules (min support 20 users)

| mode | fit s | score s | pop+AR TUNE mean | best w per variant | pure AR TUNE V300 / V900 / VHALF |
|---|---|---|---|---|---|
| sum | 0.91 | 2.44 | 0.5709 | [1.0, 1.0, 0.0] | 0.6395 / 0.5770 / 0.4888 |
| max | 0.91 | 2.31 | 0.5751 | [1.0, 0.9, 1.0] | 0.6439 / 0.5805 / 0.5005 |

Rules passing min support: 1,157,654 (A != B); with a pool consequent: 905,226; of those with lift > 1: 877,625; antecedent programmes with at least one rule: 2,348 of 2,905 TRAIN-positive programmes. Chosen mode: max.

### CBF (step-3 definition, no grid)
Feature blocks: {'category': 8, 'genre': 65, 'people_tfidf': 16796, 'decade': 12}.

## 5. Hybrid weights (weight of each component; popularity gets the remainder)

| row | V300 | V900 | VHALF | shared weight set (shipped rule) |
|---|---|---|---|---|
| pop+H | {'H': 0.6} | {'H': 0.6} | {'H': 0.6} | {'H': 0.6} |
| pop+MF | {'MF': 0.6} | {'MF': 0.6} | {'MF': 0.7} | {'MF': 0.6} |
| pop+PPR | {'PPR': 0.6} | {'PPR': 0.7} | {'PPR': 0.6} | {'PPR': 0.6} |
| pop+AR | {'AR': 1.0} | {'AR': 0.9} | {'AR': 1.0} | {'AR': 1.0} |
| pop+CBF | {'CBF': 0.1} | {'CBF': 0.1} | {'CBF': 0.0} | {'CBF': 0.1} |
| pop+blend2 | {'MF': 0.4, 'PPR': 0.4} | {'MF': 0.4, 'PPR': 0.4} | {'H': 0.4, 'MF': 0.3} | {'MF': 0.4, 'PPR': 0.4} |

Best two-component blend per variant on TUNE: V300: ['MF', 'PPR'] w=[0.4, 0.4] (TUNE 0.7025); V900: ['MF', 'PPR'] w=[0.4, 0.4] (TUNE 0.6538); VHALF: ['H', 'MF'] w=[0.4, 0.3] (TUNE 0.5631)

## 6. Cost and recommendation statistics (users with truth; popularity = distinct TRAIN positive users)

Cells: mean TRAIN popularity of recommended items / repeat share (item already a TRAIN positive for the user). Times are for ~4.8k users x 6,361 items; hybrids include components.

| row | fit s | score s | V300 | V900 | VHALF |
|---|---|---|---|---|---|
| P | 0.0 | 0.0 | 3889 / 0.80 | 3889 / 0.80 | 3889 / 0.81 |
| P-rec | 0.0 | 0.0 | 3889 / 0.80 | 3889 / 0.80 | 3889 / 0.81 |
| P-air | 0.0 | 0.0 | 3889 / 0.80 | 3889 / 0.80 | 3889 / 0.81 |
| H | 0.51 | 0.56 | 2387 / 0.98 | 2386 / 0.98 | 2392 / 0.99 |
| MF | 3.28 | 0.75 | 2432 / 0.82 | 2428 / 0.82 | 2423 / 0.83 |
| CBF | 0.92 | 4.47 | 408 / 0.24 | 404 / 0.24 | 401 / 0.24 |
| PPR | 0.11 | 19.54 | 2559 / 0.98 | 2558 / 0.98 | 2565 / 0.99 |
| AR | 0.91 | 2.66 | 3805 / 0.82 | 3805 / 0.82 | 3804 / 0.82 |
| pop+H | 0.0 | 1.04 | 3265 / 0.98 | 3265 / 0.98 | 3272 / 0.98 |
| pop+H [shared w] | 0.0 | 0.98 | 3265 / 0.98 | 3265 / 0.98 | 3272 / 0.98 |
| pop+MF | 3.28 | 1.03 | 3705 / 0.84 | 3705 / 0.84 | 3556 / 0.85 |
| pop+MF [shared w] | 3.28 | 1.01 | 3705 / 0.84 | 3705 / 0.84 | 3706 / 0.84 |
| pop+PPR | 0.11 | 19.82 | 3315 / 0.98 | 3159 / 0.98 | 3322 / 0.98 |
| pop+PPR [shared w] | 0.11 | 19.78 | 3315 / 0.98 | 3315 / 0.98 | 3322 / 0.98 |
| pop+AR | 0.91 | 2.92 | 3805 / 0.82 | 3852 / 0.81 | 3805 / 0.82 |
| pop+AR [shared w] | 0.91 | 2.88 | 3805 / 0.82 | 3805 / 0.82 | 3805 / 0.82 |
| pop+CBF | 0.92 | 4.74 | 3844 / 0.81 | 3844 / 0.81 | 3889 / 0.81 |
| pop+CBF [shared w] | 0.92 | 4.71 | 3844 / 0.81 | 3844 / 0.81 | 3844 / 0.82 |
| pop+blend2 | 3.39 | 20.42 | 3232 / 0.98 | 3231 / 0.98 | 3323 / 0.98 |
| pop+blend2 [shared w] | 3.39 | 20.03 | 3232 / 0.98 | 3231 / 0.98 | 3236 / 0.98 |

## 7. Significance: pop+blend2 [shared w] vs P on REPORT (paired bootstrap, 2000 resamples, seed 42)

| variant | users | mean AP@5 difference | 95% CI |
|---|---|---|---|
| V300 | 2400 | +0.0588 | [+0.0510, +0.0672] |
| V900 | 2392 | +0.0723 | [+0.0645, +0.0804] |
| VHALF | 2375 | +0.0684 | [+0.0604, +0.0765] |

## 8. Final submissions (TRAIN weeks 11-30, pool weeks 31-42, no re-tuning)

Popularity base: P-air (P-rec = weeks 27-30; P-air uses the 31-42 schedule only for airing counts); pool 6,632; chosen strategy pop+blend2 [shared w] with weights {'MF': 0.4, 'PPR': 0.4}.

- outputs\submissions\submission_PPR.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; weights {'PPR': 1.0}; users without component data (popularity fallback): {'PPR': 2}
- outputs\submissions\submission_AR.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; weights {'AR': 1.0}; users without component data (popularity fallback): {'AR': 2}
- outputs\submissions\submission_CHOSEN.csv: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True; weights {'MF': 0.4, 'PPR': 0.4}; users without component data (popularity fallback): {'MF': 2, 'PPR': 2}

Runtime: 626.0 s

## Winner

The MF + PPR + popularity blend (shared weights 0.4 / 0.4 / 0.2, popularity base P-air) has the highest REPORT MAP@5 in all three truth variants (0.6909 / 0.6408 / 0.5429 vs P 0.6322 / 0.5686 / 0.4745), and its paired improvement over P is +0.059 / +0.072 / +0.068 with 95% CIs that exclude zero. The personal components alone do not beat popularity by much or at all (REPORT V300: MF 0.4605, PPR 0.6028, AR 0.6365 vs P 0.6322), so the gain comes from fusing them with popularity. The simpler pop+H (0.6849 / 0.6351 / 0.5379) and pop+PPR (shared w: 0.6861 / 0.6348 / 0.5367) are within about 0.006 of the blend; that gap was not tested for significance. All hybrids that include H or PPR recommend items the user already watched in TRAIN for about 98% of slots (repeat share 0.98), and this was not tested against a truth that excludes already-watched programmes.

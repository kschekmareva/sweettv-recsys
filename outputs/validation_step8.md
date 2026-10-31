# Validation, step 8

Primary criterion VTOP5; V300/V900/VHALF are robustness checks. Same pool, repeats allowed, 5 distinct ids, popularity fill, seed 42. Evaluation is a 2-fold swap: fold A = fit on TUNE users, score REPORT users; fold B = fit on REPORT users, score TUNE users; 'mean' = average of the two folds. Caveat: step-5/7 component hyperparameters (H2 c, MF, EW) were chosen on TUNE, so fold B's features carry a small optimism. No Kaggle number is used.

Step-7 reproduction (fold A, VTOP5): 0.4047 vs 0.4047 in the step-7 report (abs diff 0.00e+00).

## 1. Candidate sources (VTOP5 truth)

Share of VTOP5 truth items the user never watched in TRAIN: 0.2767. Mean candidates per user: {'step7_union_6comp': 131.84492784837389, 'ch_slot_only': 29.396726254576784, 'union_with_ch_slot': 149.04716777945293}.

| candidate set | recall of all truth items (all users / REPORT) | recall of truth items unseen in TRAIN (all users / REPORT) |
|---|---|---|
| step7_union_6comp | 0.8169 / 0.8120 | 0.3959 / 0.3762 |
| ch_slot_only | 0.3673 / 0.3712 | 0.0926 / 0.0937 |
| union_with_ch_slot | 0.8348 / 0.8305 | 0.4562 / 0.4381 |

Oracle ceiling on REPORT (perfect ranking of the candidates): step7_union_6comp: V300 0.9589, V900 0.9414, VHALF 0.9055, VTOP5 0.7959; union_with_ch_slot: V300 0.9652, V900 0.9500, VHALF 0.9171, VTOP5 0.8147

## 2. Ablation (cumulative changes, 2-fold swap)

| step | fold A VTOP5 | fold B VTOP5 | mean VTOP5 | change in mean VTOP5 | mean V300 | mean V900 | mean VHALF | fit s (2 folds) |
|---|---|---|---|---|---|---|---|---|
| S7: step-7 reranker | 0.4047 | 0.4065 | 0.4056 |  | 0.6852 | 0.6318 | 0.6005 | 23.7 |
| +1 per-week features | 0.4047 | 0.4065 | 0.4056 | +0.0000 | 0.6852 | 0.6318 | 0.6005 | 19.2 |
| +2 CH-SLOT candidates and features | 0.4007 | 0.4034 | 0.4021 | -0.0035 | 0.6846 | 0.6306 | 0.5970 | 13.1 |
| +3 extra features | 0.4035 | 0.4050 | 0.4043 | +0.0022 | 0.6852 | 0.6309 | 0.6001 | 17.8 |
| +4 graded labels | 0.4026 | 0.4032 | 0.4029 | -0.0014 | 0.6703 | 0.6174 | 0.5901 | 24.1 |
| +5 tuned LightGBM (small search) | 0.4003 | 0.4010 | 0.4006 | -0.0022 | 0.6682 | 0.6158 | 0.5877 | 522.7 |

Step +1 divides count/seconds features by the number of weeks of their window; in validation both windows have 10 weeks, so it is a constant rescaling and trees are unchanged - its effect is on the final run, where the target window has 12 weeks.

Per fold, all four variants:

| model | fold | V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|---|
| S7: step-7 reranker | A | 0.6819 | 0.6281 | 0.5954 | 0.4047 |
| S7: step-7 reranker | B | 0.6885 | 0.6355 | 0.6056 | 0.4065 |
| +5 tuned LightGBM (small search) | A | 0.6675 | 0.6140 | 0.5841 | 0.4003 |
| +5 tuned LightGBM (small search) | B | 0.6689 | 0.6175 | 0.5913 | 0.4010 |

## 3. LightGBM search (inner 20% split of each fold's training users; holdout MAP@5 VTOP5)

- fold A best three: leaves 31, lr 0.03, min_child 100, ff 0.7 -> 0.4100 (157 trees); leaves 31, lr 0.03, min_child 20, ff 1.0 -> 0.4099 (140 trees); leaves 31, lr 0.1, min_child 20, ff 1.0 -> 0.4094 (61 trees)
- fold B best three: leaves 63, lr 0.1, min_child 100, ff 1.0 -> 0.4143 (55 trees); leaves 15, lr 0.1, min_child 100, ff 0.7 -> 0.4138 (113 trees); leaves 15, lr 0.03, min_child 20, ff 0.7 -> 0.4126 (300 trees)

Final config (best mean over the two folds): {'num_leaves': 31, 'learning_rate': 0.03, 'min_data_in_leaf': 20, 'feature_fraction': 1.0}, 176 trees. Rule: config with the best mean inner-split MAP@5 (VTOP5) over the two folds; rounds = mean best iteration of that config.

## 4. Paired bootstrap, new reranker vs step-7 reranker (2,000 resamples, seed 42): mean AP@5 difference [95% CI]

| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |
|---|---|---|---|
| V300 | -0.0143 [-0.0178, -0.0111] | -0.0196 [-0.0236, -0.0155] | -0.0170 [-0.0196, -0.0144] |
| V900 | -0.0141 [-0.0174, -0.0107] | -0.0180 [-0.0219, -0.0142] | -0.0160 [-0.0186, -0.0135] |
| VHALF | -0.0112 [-0.0146, -0.0082] | -0.0142 [-0.0184, -0.0103] | -0.0127 [-0.0154, -0.0102] |
| VTOP5 | -0.0044 [-0.0073, -0.0014] | -0.0055 [-0.0086, -0.0020] | -0.0049 [-0.0071, -0.0027] |

## 5. Feature importances, top 15 (gain share, average of the two fold models)

r_H2 0.5544, s_H2 0.0835, s_P-air 0.0492, show_n_target_air 0.0389, show_completion_rate 0.0333, r_PPR 0.0196, r_MF 0.0193, slot_rank 0.0188, r_AR 0.0181, u_share_show_channel 0.0147, us_days_since 0.0121, s_EW 0.0119, u_completed 0.0116, show_n_train_air 0.0093, s_PPR 0.0092

## 6. Shipping decision

Rule: ship only if the new reranker beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero. Ship: **False**.

No new file written; `submission_STEP7.csv` stays the latest reranker file.

Runtime: 743.4 s (stage clocks: {'setup': 67.3, 'components': 98.8, 'candidates': 100.6, 'features': 115.8, 'ablation': 214.9, 'search': 742.8, 'evaluation': 743.4})

## Conclusion

The new reranker is slightly worse than the step-7 reranker on VTOP5 in both folds (-0.0044 and -0.0055; pooled -0.0049, 95% CI [-0.0071, -0.0027]), and also on V300, V900 and VHALF (pooled -0.0170, -0.0160, -0.0127), so the shipping rule is not met and no new file was written. In the cumulative ablation (mean VTOP5 over the two folds) per-week scaling changed nothing (+0.0000, a constant rescaling in validation), CH-SLOT candidates and features changed it by -0.0035, the extra features by +0.0022, graded labels by -0.0014 (mean V300 fell from 0.6852 to 0.6703) and the tuned LightGBM by -0.0022, ending at 0.4006 against 0.4056 for step 7. CH-SLOT did raise candidate recall, from 0.8169 to 0.8348 overall and from 0.3959 to 0.4562 for truth items the user never watched in TRAIN, and raised the VTOP5 oracle ceiling from 0.7959 to 0.8147, but the reranker did not turn that into better top-5 lists. The inner-split search picked configurations with holdout MAP@5 of 0.4100 (fold A) and 0.4143 (fold B) that scored 0.4003 and 0.4010 on the other fold, and the H2 rank alone carries 0.5544 of the gain.

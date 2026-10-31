# Validation, step 10 (8c)

Primary criterion VTOP5; same pool, repeats allowed, 5 distinct ids, popularity fill, seed 42; 2-fold swap (fold A: fit TUNE, score REPORT; fold B: fit REPORT, score TUNE). Starts from the step-7 reranker (step-7 features and settings). No Kaggle number is used.

Step-7 reproduction (fold A VTOP5): 0.4047 vs 0.4047.

## 1. H2 decay grid (H2 alone, TUNE users; decay = c ** (days / T))

| c | T (days) | decay per day | TUNE V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|---|---|
| 0.5 | 60 | 0.0116 | 0.5836 | 0.5394 | 0.5177 | 0.3473 |
| 0.5 | 30 | 0.0231 | 0.5810 | 0.5375 | 0.5155 | 0.3437 |
| 0.5 | 14 | 0.0495 | 0.5669 | 0.5245 | 0.5005 | 0.3317 |
| 0.3 | 60 | 0.0201 | 0.5818 | 0.5377 | 0.5158 | 0.3441 |
| 0.3 | 30 | 0.0401 | 0.5729 | 0.5298 | 0.5063 | 0.3363 |
| 0.3 | 14 | 0.086 | 0.5440 | 0.5018 | 0.4762 | 0.3117 |
| 0.2 | 60 | 0.0268 | 0.5807 | 0.5374 | 0.5148 | 0.3426 |
| 0.2 | 30 | 0.0536 | 0.5641 | 0.5218 | 0.4979 | 0.3298 |
| 0.2 | 14 | 0.115 | 0.5278 | 0.4862 | 0.4594 | 0.2973 |
| 0.1 | 60 | 0.0384 | 0.5747 | 0.5313 | 0.5077 | 0.3372 |
| 0.1 | 30 | 0.0768 | 0.5484 | 0.5065 | 0.4810 | 0.3160 |
| 0.1 | 14 | 0.1645 | 0.5090 | 0.4689 | 0.4404 | 0.2801 |

Chosen on TUNE VTOP5: c = 0.5, T = 60. Step-5 value: c = 0.5, T = 60. c ** (days / T) = exp(-days * ln(1/c) / T), so (c, T) pairs with equal decay_per_day are identical.

## 2. Multi-scale features and 2-fold swap

Added columns: ['s_H2_c1.0_T60', 'r_H2_c1.0_T60', 'us_completed_last2w', 'us_completed_last4w', 'us_days_since_last_completed']. H2 (c=0.5, T=60) is already in the step-7 features; the added H2 configs are the others in {1.0, 0.5, best}.

| model | VTOP5 fold A | VTOP5 fold B | mean VTOP5 | mean V300 | mean V900 | mean VHALF | fit s (2 folds) |
|---|---|---|---|---|---|---|---|
| step7 features | 0.4047 | 0.4065 | 0.4056 | 0.6852 | 0.6318 | 0.6005 | 20.5 |
| step7 + multi-scale features | 0.4017 | 0.4032 | 0.4024 | 0.6827 | 0.6285 | 0.5967 | 11.5 |

## 3. Learning curve (step7 features; VTOP5 on the other fold)

| share of training-fold users | fold A (users) | fold B (users) |
|---|---|---|
| 25% | 0.3845 (577) | 0.3911 (584) |
| 50% | 0.3980 (1154) | 0.3992 (1167) |
| 100% | 0.4047 (2309) | 0.4065 (2334) |

Rule: if 100% > 50% in both folds the final model is fitted on ALL validation users with the same settings and number of trees; otherwise the step-7 final fitting (TUNE-fitted model) is kept. Outcome: 100% > 50% in both folds = **True**.

## 4. Paired bootstrap, step7 + multi-scale vs step 7 (2,000 resamples, seed 42): mean AP@5 difference [95% CI]

| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |
|---|---|---|---|
| V300 | -0.0016 [-0.0047, +0.0013] | -0.0033 [-0.0057, -0.0008] | -0.0024 [-0.0044, -0.0005] |
| V900 | -0.0025 [-0.0054, +0.0004] | -0.0042 [-0.0067, -0.0018] | -0.0033 [-0.0053, -0.0014] |
| VHALF | -0.0035 [-0.0063, -0.0007] | -0.0041 [-0.0065, -0.0017] | -0.0038 [-0.0056, -0.0019] |
| VTOP5 | -0.0030 [-0.0056, -0.0006] | -0.0033 [-0.0052, -0.0012] | -0.0031 [-0.0048, -0.0015] |

Feature importances, top 15 (gain share, mean of the two fold models of the multi-scale model): r_H2_c1.0_T60 0.3264, r_H2 0.2966, s_P-air 0.0514, show_n_target_air 0.0489, r_MF 0.0404, s_H2 0.024, show_duration 0.0215, u_share_show_channel 0.0195, s_H2_c1.0_T60 0.0193, r_PPR 0.0177, r_AR 0.015, us_days_since 0.0149, s_AR 0.0127, us_completed 0.0088, s_MF 0.0083

## 5. Decision

Rule: ship only if the new reranker (step-7 + multi-scale features) beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero. Ship: **False**.

Not shipped; no submission file was written and nothing else was changed.

Runtime: 137.4 s (stage clocks {'setup': 47.3, 'h2_grid': 52.8, 'features': 91.7, 'fits': 123.6, 'learning_curve': 136.8, 'evaluation': 137.4})

## Conclusion

The H2 grid does not move off its edge: a faster decay is monotonically worse (VTOP5 on TUNE 0.3473 at c = 0.5, T = 60 against 0.2801 at c = 0.1, T = 14), so the step-5 value stays. Adding multi-scale recency features lowers VTOP5 in both folds (-0.0030 and -0.0033; pooled -0.0031, 95% CI [-0.0048, -0.0015]) and the three robustness variants too (pooled -0.0024, -0.0033, -0.0038), so the shipping rule is not met and no file was written. The learning curve rises with training users (fold A 0.3845, 0.3980, 0.4047 at 25%, 50%, 100%; fold B 0.3911, 0.3992, 0.4065), so 100% > 50% in both folds; that rule only applies if a model is shipped, and no model was, so the step-7 file keeps its original fitting on TUNE users.


## Step 11 - learning-curve rule applied

Rule (step 10, item 3): if 100% > 50% in both folds the final model is fitted on ALL validation users with the same settings and number of trees; otherwise the step-7 final fitting (TUNE-fitted model) is kept Outcome: 100% > 50% in both folds (A 0.4047 > 0.3980, B 0.4065 > 0.3992), best features = step-7 features.

- Model: step-7 reranker (step-7 features, lambdarank, step-7 hyperparameters, seed 42) fitted on all 4,643 validation users with VTOP5 truth (TUNE + REPORT), 142 trees = mean of the best iterations of the two folds (174 and 110); no early stopping, no re-fitting on test. The two fold fits were re-run only to recover the iterations and reproduce step 10 exactly (VTOP5 0.4047 and 0.4065).
- Final run as for submission_STEP7.csv: features from weeks 21-30 + the 31-42 schedule. File `outputs\submissions\submission_STEP7_all.csv`: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True.
- Users without component data (popularity fallback): {'H2': 23, 'PPR': 14, 'MF': 14, 'AR': 14, 'EW': 14}; repeat share 0.927; share of ids shared with submission_STEP7.csv 0.912 (identical lists for 59.6% of users).
- submission_STEP7_all.csv is the final submission. It cannot be scored on validation, because no held-out users remain; the decision rests on the learning curve (VTOP5 rises with training users in both folds), which was observed up to the size of one fold and is extrapolated to twice that size.

# Validation, step 8b

Seed/fold bagging of the step-7 reranker (step-7 features and settings, no step-8 changes): 5 seeds x the 2-fold swap (fold A: fit TUNE, score REPORT; fold B: fit REPORT, score TUNE). Each user's score is the average of the rank-normalised scores of the 5 models that did not see that user. Primary criterion VTOP5. No Kaggle number is used.

Step-7 reproduction (fold A, VTOP5, seed 42): 0.4047 vs 0.4047 in the step-7 report.

## 1. MAP@5 (mean over the two folds; fold A / fold B for VTOP5)

| model | VTOP5 fold A | VTOP5 fold B | mean VTOP5 | mean V300 | mean V900 | mean VHALF |
|---|---|---|---|---|---|---|
| step7 (single seed 42) | 0.4047 | 0.4065 | 0.4056 | 0.6852 | 0.6318 | 0.6005 |
| bagged (w=0) | 0.4032 | 0.4073 | 0.4052 | 0.6848 | 0.6316 | 0.6000 |
| bagged + step-6 fusion (w chosen out-of-bag) | 0.4032 | 0.4069 | 0.4050 | 0.6866 | 0.6330 | 0.6012 |

Fusion weight on the step-6 blend (grid [0.0, 0.1, 0.2, 0.3]), chosen out-of-bag inside each fold: {'A': 0.0, 'B': 0.1}. w is chosen on out-of-bag predictions of the fold's training users (users in a seed's inner 20% holdout, scored by that seed's model only). Out-of-bag selection MAP@5 (VTOP5) by w: {"A": {"0.0": 0.39872659780503555, "0.1": 0.396464923606628, "0.2": 0.3945176099275518, "0.3": 0.3923445233483968}, "B": {"0.0": 0.4063985096557514, "0.1": 0.4071568010075567, "0.2": 0.4045068919675343, "0.3": 0.40079677441925554}} on [1549, 1588] users.

## 2. Paired bootstrap vs step 7 (2,000 resamples, seed 42): mean AP@5 difference [95% CI]

**bagged (w=0)**

| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |
|---|---|---|---|
| V300 | -0.0001 [-0.0028, +0.0025] | -0.0006 [-0.0028, +0.0016] | -0.0003 [-0.0021, +0.0014] |
| V900 | -0.0002 [-0.0027, +0.0025] | -0.0003 [-0.0024, +0.0019] | -0.0002 [-0.0021, +0.0016] |
| VHALF | -0.0015 [-0.0042, +0.0010] | +0.0005 [-0.0016, +0.0028] | -0.0005 [-0.0023, +0.0012] |
| VTOP5 | -0.0015 [-0.0039, +0.0007] | +0.0008 [-0.0011, +0.0029] | -0.0003 [-0.0018, +0.0013] |

**bagged + step-6 fusion (w chosen out-of-bag)**

| variant | fold A (REPORT users) | fold B (TUNE users) | pooled |
|---|---|---|---|
| V300 | -0.0001 [-0.0027, +0.0026] | +0.0029 [+0.0006, +0.0051] | +0.0014 [-0.0003, +0.0030] |
| V900 | -0.0002 [-0.0029, +0.0026] | +0.0026 [+0.0004, +0.0049] | +0.0012 [-0.0006, +0.0030] |
| VHALF | -0.0015 [-0.0043, +0.0010] | +0.0029 [+0.0008, +0.0052] | +0.0007 [-0.0010, +0.0024] |
| VTOP5 | -0.0015 [-0.0039, +0.0008] | +0.0004 [-0.0016, +0.0025] | -0.0006 [-0.0021, +0.0010] |

## 3. Decision

Rule: ship only if the best of bagging / bagging+fusion beats step 7 on VTOP5 in BOTH folds and the pooled 95% CI excludes zero. Best of 1-2 by mean VTOP5: **bagged (w=0)**. Ship: **False**.

Not shipped; no submission file was written and nothing else was changed.

Runtime: 180.4 s (all model fits 91.8 s; stage clocks {'setup': 84.9, 'bagging': 176.7, 'fusion': 179.1})
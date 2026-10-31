# Report tables

All numbers are copied from the step reports in outputs/ (no recomputation). REPORT = the 50% of validation users not used for tuning (2,400 users for V300, 2,334 with VTOP5 truth); validation = features from weeks 11-20, truth from weeks 21-30.

## (a) Method comparison on the REPORT half

| method | VTOP5 (primary) | V300 | source |
|---|---|---|---|
| P (popularity) | 0.2699 | 0.6322 | step 5 |
| P-rec (last 4 TRAIN weeks) | n/a | 0.6350 | step 4 (VTOP5 not computed) |
| P-air (popularity x log1p(target airings)) | 0.2748 | 0.6315 | step 5 |
| H (own history, step 2) | 0.2361 | 0.5746 | step 5 |
| H2 (completed airings, decay) | 0.3482 | 0.5834 | step 5 |
| CBF (content-based) | n/a | 0.0431 | step 4 (VTOP5 not computed) |
| MF (ALS, alpha 0.3, pure) | 0.1525 | 0.5199 | step 5 |
| PPR (alpha 0.3, pure) | 0.2580 | 0.6028 | step 5 |
| AR (rules, max, pure) | 0.2777 | 0.6365 | step 5 |
| step-4 blend (P-air 0.2, MF 0.4, PPR 0.4) | 0.3180 | 0.6909 | step 5 |
| step-6 blend (P-air 0.2, MF 0.2, PPR 0.1, H2 0.5) | 0.3630 | 0.6548 | step 7 |
| step-7 reranker (LightGBM, fit on TUNE) | 0.4047 | 0.6819 | step 7 |
| oracle (perfect ranking of the 6-component top-50 union) | 0.7959 | 0.9589 | step 7 |

## (b) Kaggle results (entered by the project owner) and validation VTOP5

| file | method | Kaggle public | Kaggle private | validation VTOP5 (REPORT) |
|---|---|---|---|---|
| submission_P.csv | P (global popularity) | 0.28234 | 0.27809 | 0.2699 |
| submission_FINAL_vtop5.csv | P-air + MF + PPR + H2 blend, weights tuned on VTOP5 | 0.41541 | 0.40202 | 0.3630 |
| submission_STEP7.csv | step-7 reranker (LightGBM, fit on TUNE users) | 0.42744 | 0.41984 | 0.4047 |
| submission_STEP7_all.csv | step-7 reranker refit on all validation users (FINAL) | 0.42914 | 0.42244 | not scorable (no held-out users); same recipe on TUNE-fit folds: 0.4047 / 0.4065 |

The three files that have a validation VTOP5 are ordered the same on validation, Kaggle public and Kaggle private (P < FINAL_vtop5 < STEP7). Kaggle private minus validation VTOP5 (REPORT): submission_P.csv +0.0082, submission_FINAL_vtop5.csv +0.0390, submission_STEP7.csv +0.0151. This agreement was observed after the fact; no weight or hyperparameter was chosen from a Kaggle number.

## (c) Final hyperparameters per method

| method | hyperparameters |
|---|---|
| P | distinct users with a positive pair (>= 300 s total overlap), restricted to the pool; ties by ascending tv_show_id |
| P-rec | same as P, last 4 TRAIN weeks only (weeks 17-20 in validation, 27-30 in the final run) |
| P-air | P x log1p(number of airings of the show in the target-period schedule; the schedule is used only to know what airs) |
| H | own TRAIN positives in the pool ranked by total overlap seconds, filled with P |
| H2 | airings with view_pct >= 0.8; r = 5 if view_pct = 1 else 2 + 2*view_pct^2; decay c^(days_before_train_end/60), c = 0.5; x log10(1 + airings of the show in the TRAIN schedule) |
| CBF | blocks {'category': 8, 'genre': 65, 'people_tfidf': 16796, 'decade': 12} (category, genre, TF-IDF people, decade), L2-normalised; user profile = positives weighted by log1p(overlap/60); cosine |
| MF | implicit ALS, 32 factors, reg 0.1, alpha 0.3, 15 iterations, seed 42; weight log1p(overlap/60) over all labelled TRAIN pairs (steps 3/4 used alpha 10 / 1) |
| PPR | bipartite user-programme graph from TRAIN positives, edge weight log1p(overlap/60), alpha 0.3, power iteration tol 1e-6, max 100 iterations |
| AR | item-to-item rules from TRAIN positive sets, confidence, user score = max over the user's positives, min support 20 users |
| EW | smoothed completion rate x target airings, m = 10, recency c = 0.5 |
| step-4 blend | per-user min-max fusion: P-air 0.2, MF (alpha 1) 0.4, PPR 0.4 |
| step-6 blend | per-user min-max fusion: P-air 0.2, MF 0.2, PPR 0.1, H2 0.5 (weights tuned on TUNE under VTOP5) |
| step-7 reranker (FINAL) | LightGBM lambdarank, binary labels = VTOP5 truth, candidates = union of the top-50 of H2, EW, PPR, MF, AR, P-air, 33 features, params {'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [5], 'learning_rate': 0.05, 'num_leaves': 31, 'min_data_in_leaf': 50, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 1.0, 'seed': 42, 'num_threads': 4, 'deterministic': True, 'force_row_wise': True}; 142 trees (mean best iteration of the two folds: 174 and 110), fitted on all 4,643 validation users |

## (d) Data and join statistics

| item | value |
|---|---|
| view log rows / distinct users | 5,821,987 / 4,914 |
| schedule rows (weeks 11-20 / 21-30 / 31-42) | 1,074,235 / 1,074,394 / 1,295,378 |
| share of schedule rows with tv_show_id != 0 (11-20 / 21-30 / 31-42) | 0.108 / 0.113 / 0.112 |
| submission users present in the view log | 1,260 of 1,260 |
| views of weeks 11-20: total / channel absent from schedule / channel present but no overlapping row | 3,945,173 / 113,854 (0.029) / 255,954 |
| views of weeks 11-20: with a labelled overlap; share of matched seconds on tv_show_id = 0 | 1,897,799 (0.481); 0.452 |
| views of weeks 21-30: total / channel absent from schedule / channel present but no overlapping row | 1,876,814 / 40,549 (0.022) / 4,503 |
| views of weeks 21-30: with a labelled overlap; share of matched seconds on tv_show_id = 0 | 1,065,602 (0.568); 0.416 |
| interaction pairs (user, programme) weeks 11-20 / 21-30 | 857,222 / 658,368 |
| positive pairs (>= 300 s) TRAIN / VALIDATION | 582,559 / 478,219 |
| validation pool / final pool (labelled programmes airing in weeks 21-30 / 31-42) | 6,361 / 6,632 |
| repeat rate (validation positives that were also TRAIN positives for the same user) | 0.211 |
| cold start in validation (no TRAIN positive): users with truth / submission users | 56 / 10 |
| final file STEP7_all: submission users without data in weeks 21-30, per component (popularity fallback) | {'H2': 23, 'PPR': 14, 'MF': 14, 'AR': 14, 'EW': 14} |
| VTOP5 truth: users / positive pairs / users with exactly 5 truth items | 4,643 / 21,689 / 4,030 |

## (e) Improvement attempts after step 7 (2-fold swap, mean VTOP5 over the two folds; step-7 reranker = 0.4056)

| step | what was tried | mean VTOP5 | change vs step 7 | shipped |
|---|---|---|---|---|
| 8 | +1 per-week features (cumulative) | 0.4056 | +0.0000 | no |
| 8 | +2 CH-SLOT candidates and features (cumulative) | 0.4021 | -0.0035 | no |
| 8 | +3 extra features (cumulative) | 0.4043 | -0.0013 | no |
| 8 | +4 graded labels (cumulative) | 0.4029 | -0.0027 | no |
| 8 | +5 tuned LightGBM (small search) (cumulative) | 0.4006 | -0.0049 | no |
| 8b | bagged (w=0) | 0.4052 | -0.0003 | no |
| 8b | bagged + step-6 fusion (w chosen out-of-bag) | 0.4050 | -0.0006 | no |
| 10 | step-7 + multi-scale recency features | 0.4024 | -0.0031 | no |

Step 8 pooled bootstrap of the full new reranker vs step 7 on VTOP5: -0.0049 [-0.0071, -0.0027]. Step 10 H2 decay grid (c in 0.5/0.3/0.2/0.1 x T in 60/30/14 days): chosen c = 0.5, T = 60, i.e. unchanged.

Step-10 learning curve (step-7 features, VTOP5 on the other fold; users in the training fold):

| share of training-fold users | fold A | fold B |
|---|---|---|
| 25% (577 / 584 users) | 0.3845 | 0.3911 |
| 50% (1154 / 1167 users) | 0.3980 | 0.3992 |
| 100% (2309 / 2334 users) | 0.4047 | 0.4065 |

100% > 50% in both folds, so by the pre-declared rule the final model was refit on all validation users (step 11, `submission_STEP7_all.csv`).

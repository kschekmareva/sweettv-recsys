# Submissions

All files: candidate pool = labelled programmes airing in weeks 31-42, 5 distinct ids per user, template format. History used: viewing weeks 11-30 for the steps 3-6 files; the reranker files (`submission_STEP7.csv`, `submission_STEP7_all.csv`) take their features from weeks 21-30 (same window length as in training) and the 31-42 schedule. Kaggle columns filled by the project owner; files marked "not submitted" were never uploaded.

**Final submission (FINAL) = `submission_STEP7_all.csv`** (public 0.42914, private 0.42244).

| file | method | weights / hyperparameters | Kaggle public | Kaggle private |
|---|---|---|---|---|
| submission_P.csv | step 3: global popularity (P) | positives (>= 300 s) of weeks 11-30, distinct users, pool 31-42 | 0.28234 | 0.27809 |
| submission_MF.csv | step 3: implicit ALS, pure (popularity fallback) | factors 32, reg 0.1, alpha 10, 15 iterations, seed 42 | not submitted | not submitted |
| submission_CBF.csv | step 3: content-based (cosine to user profile) | category + genre + TF-IDF people + decade blocks; profile = positives weighted by log1p(overlap/60) | not submitted | not submitted |
| submission_HYBRID.csv | step 3: popularity + MF score fusion | MF as above, w_MF 0.6, popularity 0.4 | not submitted | not submitted |
| submission_PPR.csv | step 4: personalised PageRank, pure | alpha 0.3 (link-following), tol 1e-6, fallback P-air | not submitted | not submitted |
| submission_AR.csv | step 4: item-to-item association rules, pure | confidence, mode max, min support 20 users, fallback P-air | not submitted | not submitted |
| submission_CHOSEN.csv | step 4: blend P-air + MF + PPR | P-air 0.2, MF 0.4 (factors 32, reg 0.1, alpha 1), PPR 0.4 (alpha 0.3) | not submitted | not submitted |
| submission_H2.csv | step 5: H2 own-history (airings with view_pct >= 0.8), pure | r = 5 if view_pct = 1 else 2 + 2*view_pct^2, decay c 0.5 per 60 days, x log10(1 + airings in TRAIN), fallback P-air | not submitted | not submitted |
| submission_CHOSEN_step5.csv | step 5: P-air + MF + PPR + H2, shared weights | MF 0.3, PPR 0.4, H2 0.1, P-air 0.2; MF alpha 0.3, PPR alpha 0.3, H2 c 0.5 | not submitted | not submitted |
| submission_FINAL_vtop5.csv | step 6: P-air + MF + PPR + H2, weights tuned on VTOP5 | MF 0.2, PPR 0.1, H2 0.5, P-air 0.2; MF alpha 0.3, PPR alpha 0.3, H2 c 0.5 | 0.41541 | 0.40202 |
| submission_STEP7.csv | step 7: LightGBM reranker (lambdarank) over top-50 union of H2, EW, PPR, MF, AR, P-air | fitted on TUNE users (weeks 11-20 features, weeks 21-30 VTOP5 labels), best iteration 174; applied to features from weeks 21-30; EW m 10, c 0.5 | 0.42744 | 0.41984 |
| submission_STEP7_all.csv | step 11 (FINAL): step-7 reranker refit on ALL validation users (learning-curve rule), LightGBM lambdarank | step-7 features and settings, seed 42, 142 trees (mean best iteration of the two folds: 174 and 110); features from weeks 21-30 + 31-42 schedule; cannot be scored on validation | 0.42914 | 0.42244 |

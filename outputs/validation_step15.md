# Step 15 - use the full history at inference, validated by an analogue

Per-week reranker = step-7 reranker with window-dependent counts divided by the number of weeks of their window (`rerank2.per_week`), step-7 hyperparameters, seed 42, lambdarank, early stopping on 20% of the training users. Analogue of the final run: fit on features from weeks 16-20 only (labels VTOP5 of weeks 21-30, candidates from the same window), infer with features/candidates from (a) weeks 16-20 or (b) weeks 11-20 (H2 decay unchanged). 2-fold swap (A: fit TUNE, score REPORT; B: the reverse). Primary criterion VTOP5.

## Validation (MAP@5, users with non-empty truth of the respective type)

| variant | fold | V300 | V900 | VHALF | VTOP5 |
|---|---|---|---|---|---|
| (a) fit 16-20, infer 16-20 | A | 0.6761 | 0.6218 | 0.5864 | 0.3937 |
| (a) fit 16-20, infer 16-20 | B | 0.6746 | 0.6201 | 0.5904 | 0.3941 |
| (b) fit 16-20, infer 11-20 | A | 0.6775 | 0.6224 | 0.5894 | 0.4006 |
| (b) fit 16-20, infer 11-20 | B | 0.6804 | 0.6264 | 0.5983 | 0.4002 |
| reference: fit 11-20, infer 11-20 | A | 0.6819 | 0.6281 | 0.5954 | 0.4047 |
| reference: fit 11-20, infer 11-20 | B | 0.6885 | 0.6355 | 0.6056 | 0.4065 |

## Paired bootstrap 95% CI of (b) - (a), VTOP5 (2,000 resamples, seed 42)

| scope | users | mean diff | 95% CI |
|---|---|---|---|
| A | 2334 | +0.0069 | [+0.0030, +0.0108] |
| B | 2309 | +0.0061 | [+0.0021, +0.0100] |
| pooled | 4643 | +0.0065 | [+0.0036, +0.0094] |

Robustness, pooled (b) - (a):

| truth | mean diff | 95% CI |
|---|---|---|
| V300 | +0.0036 | [+0.0005, +0.0068] |
| V900 | +0.0034 | [+0.0002, +0.0067] |
| VHALF | +0.0055 | [+0.0023, +0.0087] |
| VTOP5 | +0.0065 | [+0.0037, +0.0094] |

## Evaluation users without H2 data (no scheduled channel watched in the feature window, popularity fallback)

| fold | evaluation users with VTOP5 truth | (a) weeks 16-20 | (b) weeks 11-20 |
|---|---|---|---|
| A | 2334 | 96 | 48 |
| B | 2309 | 117 | 60 |

## Pre-declared rule and decision

(b) beats (a) on VTOP5 in BOTH folds and the pooled paired-bootstrap 95% CI of (b)-(a) excludes zero -> final run = per-week reranker trained on weeks 11-20 features, applied to features and candidates from weeks 11-30; otherwise per-week reranker applied to weeks 21-30 as in step 11.

- (b) > (a) in fold A: True; fold B: True; pooled CI excludes zero: True.
- **Branch taken: long window (weeks 11-30 features)**.

## Shipped file `outputs/submissions/submission_STEP15.csv`

- Model: per-week reranker, features from weeks 11-20, fitted on all 4,643 validation users with VTOP5 truth, 142 trees (mean of the fold best iterations {'A': 174, 'B': 110}). Inference features/candidates from a 20-week window, n_target_airings divided by 12 (weeks 31-42).
- Format: header True, rows 1260, user order True, 5 distinct ids True, ids in final pool True.
- Users without component data (popularity fallback): {'H2': 5, 'PPR': 2, 'MF': 2, 'AR': 2, 'EW': 1}.
- Repeat share (ids the user already watched >= 300 s in the feature window): 0.974.
- Share of ids identical to submission_STEP7_all.csv: 0.815 (identical lists for 30.3% of users).
- Runtime of this script: 247.9 s.

## Summary

Inferring with 10 weeks instead of the 5 weeks used for fitting gave VTOP5 0.4006 vs 0.3937 in fold A and 0.4002 vs 0.3941 in fold B; the pooled difference is +0.0065 with 95% CI [+0.0036, +0.0094]. Users without H2 data fell from 213 to 108 of 4643 evaluation users. The rule gave the branch 'long_window', so submission_STEP15.csv uses a 20-week inference window. Its ids coincide with submission_STEP7_all.csv for 81.5% of the slots; this file has no validation score and no Kaggle result yet.

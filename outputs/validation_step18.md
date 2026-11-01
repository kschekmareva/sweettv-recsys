# Step 18 - temporal data augmentation for the reranker

Extra training period: features and candidates from weeks 11-15 (every component fitted on weeks 11-15 only), labels = VTOP5 of weeks 16-20, candidate pool = programmes airing in weeks 16-20, per-week features (5 feature weeks, 5 target weeks); all users with such truth. Only weeks <= 20 are used, so validation weeks 21-30 cannot leak. Standard setting: features 11-20 -> truth 21-30, 2-fold swap (A: fit TUNE, score REPORT; B: the reverse). M0 = training-fold rows only; M1 = training-fold rows + all extra rows; tree count by early stopping (30 rounds) on a 20% inner split of the training-fold users. Seed 42.

## Extra-period rows

| item | value |
|---|---|
| users with VTOP5 truth in weeks 16-20 | 4,574 |
| pool (programmes airing in weeks 16-20) | 4,402 |
| candidate pairs / positive pairs (share) | 573,003 / 17,538 (0.0306) |
| truth items in the pool / retrieved by the candidate generators | 20,969 / 17,538 |
| users without H2 data in weeks 11-15 | 177 |
| standard rows (features 11-20, truth 21-30): pairs (positive share) | 633,881 (0.0280) |

## Validation (MAP@5 on the evaluation fold)

| model | fold | V300 | V900 | VHALF | VTOP5 | best iteration |
|---|---|---|---|---|---|---|
| M0 | A | 0.6819 | 0.6281 | 0.5954 | 0.4047 | 174 |
| M1 | A | 0.6831 | 0.6287 | 0.5943 | 0.4027 | 149 |
| M0 | B | 0.6885 | 0.6355 | 0.6056 | 0.4065 | 110 |
| M1 | B | 0.6886 | 0.6350 | 0.6039 | 0.4044 | 99 |

M0 reproduces the step-7 / step-15 reference (0.4047 and 0.4065).

## Paired bootstrap 95% CI of M1 - M0, VTOP5 (2,000 resamples, seed 42)

| scope | users | mean diff | 95% CI |
|---|---|---|---|
| A | 2334 | -0.0020 | [-0.0047, +0.0005] |
| B | 2309 | -0.0020 | [-0.0047, +0.0004] |
| pooled | 4643 | -0.0020 | [-0.0039, -0.0001] |

Robustness, pooled M1 - M0:

| truth | mean diff | 95% CI |
|---|---|---|
| V300 | +0.0007 | [-0.0014, +0.0028] |
| V900 | +0.0001 | [-0.0020, +0.0021] |
| VHALF | -0.0013 | [-0.0033, +0.0008] |
| VTOP5 | -0.0020 | [-0.0039, -0.0001] |

## Pre-declared rule and decision

Ship only if M1 (training-fold rows + all extra-period rows) beats M0 (training-fold rows only) on VTOP5 in BOTH folds and the pooled paired-bootstrap 95% CI of M1 - M0 excludes zero.

- M1 > M0 in fold A: False; fold B: False; pooled CI excludes zero: False.
- **Shipped: no - submission_STEP15.csv stays the final file; nothing else was changed**.

## Summary

M1 scored VTOP5 0.4027 / 0.4044 against M0 0.4047 / 0.4065 in folds A / B, pooled difference -0.0020 with 95% CI [-0.0039, -0.0001]. The extra period added 573,003 rows (3.06% positive) to 633,881 standard rows. By the pre-declared rule the result was not shipped, so submission_STEP15.csv remains the final file.

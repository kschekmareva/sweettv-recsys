# Step 17 - last improvement round on the inference-window lever

Baseline = step-15 validation analogue (b): per-week reranker fitted on features/candidates of weeks 16-20 (labels VTOP5 of weeks 21-30), inference with features/candidates of weeks 11-20 (VTOP5 fold A 0.4006, fold B 0.4002, mean 0.4004; reproduced here exactly). Seed 42, 2-fold swap, primary VTOP5. Every setting is chosen on the fold-internal TRAINING users only (the users the fold's model was fitted on); a setting is part of the combination only if BOTH folds choose the same non-baseline value.

## 1. Inference-time decay c of H2 and EW (model unchanged)

| c per 60 days | fold | VTOP5 training users (selection) | VTOP5 evaluation users |
|---|---|---|---|
| 0.3 | A | 0.4076 | 0.4004 |
| 0.3 | B | 0.4056 | 0.4005 |
| 0.5 | A | 0.4067 | 0.4006 |
| 0.5 | B | 0.4054 | 0.4002 |
| 0.7 | A | 0.4059 | 0.4012 |
| 0.7 | B | 0.4047 | 0.3991 |
| 1.0 | A | 0.4062 | 0.4001 |
| 1.0 | B | 0.4037 | 0.3983 |

## 2. Rank fusion of the weeks 16-20 and 11-20 inference scores (weight w on 11-20; w = 1 is the baseline)

| w | fold | VTOP5 training users (selection) | VTOP5 evaluation users |
|---|---|---|---|
| 0.5 | A | 0.4230 | 0.4000 |
| 0.5 | B | 0.4081 | 0.3999 |
| 0.7 | A | 0.4126 | 0.4010 |
| 0.7 | B | 0.4063 | 0.4014 |
| 0.85 | A | 0.4076 | 0.4005 |
| 0.85 | B | 0.4061 | 0.4007 |
| 1.0 | A | 0.4067 | 0.4006 |
| 1.0 | B | 0.4054 | 0.4002 |

## 3. Candidate depth K per component (training and inference)

| K | fold | VTOP5 training users (selection) | VTOP5 evaluation users |
|---|---|---|---|
| 50 | A | 0.4067 | 0.4006 |
| 50 | B | 0.4054 | 0.4002 |
| 80 | A | 0.4048 | 0.3998 |
| 80 | B | 0.4058 | 0.4002 |
| 120 | A | 0.4031 | 0.3983 |
| 120 | B | 0.4055 | 0.4014 |

Single levers, pooled evaluation users, difference to the baseline (paired bootstrap 95% CI; reporting only, nothing was chosen from these):

| lever | value | mean diff | 95% CI |
|---|---|---|---|
| item1_c | 0.3 | +0.0001 | [-0.0009, +0.0010] |
| item1_c | 0.5 | +0.0000 | [+0.0000, +0.0000] |
| item1_c | 0.7 | -0.0002 | [-0.0012, +0.0007] |
| item1_c | 1.0 | -0.0012 | [-0.0024, +0.0001] |
| item2_w | 0.5 | -0.0004 | [-0.0024, +0.0014] |
| item2_w | 0.7 | +0.0008 | [-0.0004, +0.0020] |
| item2_w | 0.85 | +0.0002 | [-0.0005, +0.0009] |
| item2_w | 1.0 | +0.0000 | [+0.0000, +0.0000] |
| item3_K | 50 | +0.0000 | [+0.0000, +0.0000] |
| item3_K | 80 | -0.0004 | [-0.0023, +0.0014] |
| item3_K | 120 | -0.0005 | [-0.0024, +0.0013] |

## 4. Choice on training users and combination

- Chosen per fold on training users: {'A': {'c': 0.3, 'w': 0.5, 'K': 50}, 'B': {'c': 0.3, 'w': 0.5, 'K': 80}}.
- Consensus (same value in both folds, else baseline): {'c': 0.3, 'w': 0.5, 'K': 50}.

| setting | fold | VTOP5 | V300 | V900 | VHALF |
|---|---|---|---|---|---|
| baseline | A | 0.4006 | 0.6775 | 0.6224 | 0.5894 |
| combination | A | 0.3995 | 0.6799 | 0.6262 | 0.5919 |
| baseline | B | 0.4002 | 0.6804 | 0.6264 | 0.5983 |
| combination | B | 0.3990 | 0.6802 | 0.6261 | 0.5977 |

Paired bootstrap (2,000 resamples, seed 42), combination - baseline, VTOP5:

| scope | users | mean diff | 95% CI |
|---|---|---|---|
| A | 2334 | -0.0011 | [-0.0038, +0.0017] |
| B | 2309 | -0.0012 | [-0.0040, +0.0016] |
| pooled | 4643 | -0.0011 | [-0.0032, +0.0009] |

Robustness, pooled combination - baseline:

| truth | mean diff | 95% CI |
|---|---|---|
| V300 | +0.0011 | [-0.0013, +0.0034] |
| V900 | +0.0017 | [-0.0007, +0.0041] |
| VHALF | +0.0009 | [-0.0015, +0.0032] |
| VTOP5 | -0.0011 | [-0.0032, +0.0009] |

## 5. Pre-declared rule and decision

Ship only if the combination of the levers chosen in both folds on training users beats the step-15 analogue baseline on VTOP5 in BOTH folds and the pooled paired-bootstrap 95% CI excludes zero.

- Combination beats baseline in both folds: False; pooled CI excludes zero: False; consensus differs from baseline: True.
- **Shipped: no - submission_STEP15.csv stays the final file; nothing else was changed**.

## Summary

The settings chosen on training users were c = 0.3 / 0.3, w = 0.5 / 0.5, K = 50 / 80 (folds A / B); the consensus was {'c': 0.3, 'w': 0.5, 'K': 50}. The combination scored VTOP5 0.3995 and 0.3990 against the baseline 0.4006 and 0.4002, pooled difference -0.0011 with 95% CI [-0.0032, +0.0009]. By the pre-declared rule the result was not shipped, so submission_STEP15.csv remains the final file.

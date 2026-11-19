# Fairness Metrics Report – ABC Bank drift model

Generated 2026-11-19 from the hold-out test set. 'Selected' = flagged for the reactivation offer (a benefit). Protected attributes are used only for this audit, never as model inputs.

Rules (Modules 1–2): disparate impact (group selection rate ÷ overall) within 0.8–1.25; equalized odds difference ≤ 0.1. Groups under 500 customers are reported but not judged.

## 1. Data bias in the modelling table

| attribute | group | customers | base_drift_rate | share_pct |
| --- | --- | --- | --- | --- |
| Age band | 18-24 | 3546 | 0.088 | 17.020 |
| Age band | 25-34 | 3050 | 0.105 | 14.640 |
| Age band | 35-44 | 4961 | 0.088 | 23.810 |
| Age band | 45-54 | 4882 | 0.085 | 23.430 |
| Age band | 55-64 | 2202 | 0.079 | 10.570 |
| Age band | 65+ | 1886 | 0.066 | 9.050 |
| Age band | <18 | 271 | 0.048 | 1.300 |
| Age band | SUPPRESSED | 38 | 0.026 | 0.180 |
| Sex (H/V as coded in source) | H | 9037 | 0.082 | 43.370 |
| Sex (H/V as coded in source) | V | 11799 | 0.089 | 56.630 |
| Foreign-born | N | 19945 | 0.085 | 95.720 |
| Foreign-born | S | 891 | 0.109 | 4.280 |

## 2. Champion model, by group

| attribute | group | customers | base_drift_rate | selection_rate | recall_tpr | false_positive_rate | disparate_impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age band | 18-24 | 716 | 0.074 | 0.295 | 0.717 | 0.261 | 0.782 | Check: DI outside 0.8-1.25 |
| Age band | 25-34 | 615 | 0.102 | 0.432 | 0.667 | 0.406 | 1.147 | OK |
| Age band | 35-44 | 995 | 0.086 | 0.453 | 0.779 | 0.422 | 1.203 | OK |
| Age band | 45-54 | 970 | 0.089 | 0.413 | 0.756 | 0.380 | 1.097 | OK |
| Age band | 55-64 | 451 | 0.082 | 0.346 | 0.730 | 0.312 | 0.918 | Too small to judge (<500) |
| Age band | 65+ | 359 | 0.089 | 0.228 | 0.438 | 0.208 | 0.606 | Too small to judge (<500) |
| Age band | <18 | 51 | 0.020 | 0.078 | 0.000 | 0.080 | 0.208 | Too small to judge (<500) |
| Age band | SUPPRESSED | 11 | 0.091 | 0.000 | 0.000 | 0.000 | 0.000 | Too small to judge (<500) |
| Sex (H/V as coded in source) | H | 1829 | 0.086 | 0.365 | 0.650 | 0.339 | 0.969 | OK |
| Sex (H/V as coded in source) | V | 2339 | 0.086 | 0.386 | 0.748 | 0.352 | 1.024 | OK |
| Foreign-born | N | 4018 | 0.087 | 0.378 | 0.702 | 0.347 | 1.002 | OK |
| Foreign-born | S | 150 | 0.067 | 0.360 | 0.800 | 0.329 | 0.955 | Too small to judge (<500) |

## 3. Summary per attribute (groups large enough to judge)

| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.650 | 0.162 | False | False | False |
| Sex (H/V as coded in source) | 2 | 0.946 | 0.098 | True | True | True |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

## 4. Fairlearn cross-check (all groups, including small ones)

| attribute | fairlearn_dp_ratio_all_groups | fairlearn_eo_difference_all_groups | fairlearn_selection_rate_max_gap |
| --- | --- | --- | --- |
| Age band | 0.000 | 0.779 | 0.453 |
| Sex (H/V as coded in source) | 0.946 | 0.098 | 0.021 |
| Foreign-born | 0.954 | 0.098 | 0.018 |

## 5. Mitigation: ExponentiatedGradient, equalized odds on age_band

| attribute | group | customers | base_drift_rate | selection_rate | recall_tpr | false_positive_rate | disparate_impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age band | 18-24 | 716 | 0.074 | 0.272 | 0.604 | 0.246 | 1.019 | OK |
| Age band | 25-34 | 615 | 0.102 | 0.275 | 0.365 | 0.265 | 1.028 | OK |
| Age band | 35-44 | 995 | 0.086 | 0.253 | 0.570 | 0.223 | 0.948 | OK |
| Age band | 45-54 | 970 | 0.089 | 0.281 | 0.395 | 0.270 | 1.053 | OK |
| Age band | 55-64 | 451 | 0.082 | 0.255 | 0.460 | 0.237 | 0.954 | Too small to judge (<500) |
| Age band | 65+ | 359 | 0.089 | 0.267 | 0.312 | 0.263 | 1.000 | Too small to judge (<500) |
| Age band | <18 | 51 | 0.020 | 0.235 | 0.000 | 0.240 | 0.880 | Too small to judge (<500) |
| Age band | SUPPRESSED | 11 | 0.091 | 0.182 | 0.000 | 0.200 | 0.680 | Too small to judge (<500) |
| Sex (H/V as coded in source) | H | 1829 | 0.086 | 0.264 | 0.433 | 0.248 | 0.988 | OK |
| Sex (H/V as coded in source) | V | 2339 | 0.086 | 0.270 | 0.480 | 0.250 | 1.009 | OK |
| Foreign-born | N | 4018 | 0.087 | 0.269 | 0.459 | 0.251 | 1.005 | OK |
| Foreign-born | S | 150 | 0.067 | 0.233 | 0.500 | 0.214 | 0.873 | Too small to judge (<500) |

| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.900 | 0.239 | True | False | False |
| Sex (H/V as coded in source) | 2 | 0.979 | 0.047 | True | True | True |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

Performance trade-off (test set):

| model | recall | precision | f1 | roc_auc | flagged_rate |
| --- | --- | --- | --- | --- | --- |
| champion | 0.705 | 0.161 | 0.262 | 0.745 | 0.377 |
| mitigated | 0.460 | 0.148 | 0.224 | 0.721 | 0.267 |

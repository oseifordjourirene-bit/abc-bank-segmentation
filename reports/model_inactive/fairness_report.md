# Fairness Metrics Report – ABC Bank drift model

Generated 2026-11-19 from the hold-out test set. 'Selected' = flagged for the reactivation offer (a benefit). Protected attributes are used only for this audit, never as model inputs.

Rules (Modules 1–2): disparate impact (group selection rate ÷ overall) within 0.8–1.25; equalized odds difference ≤ 0.1. Groups under 500 customers are reported but not judged.

## 1. Data bias in the modelling table

| attribute | group | customers | base_drift_rate | share_pct |
| --- | --- | --- | --- | --- |
| Age band | 18-24 | 3546 | 0.060 | 17.020 |
| Age band | 25-34 | 3050 | 0.043 | 14.640 |
| Age band | 35-44 | 4961 | 0.023 | 23.810 |
| Age band | 45-54 | 4882 | 0.017 | 23.430 |
| Age band | 55-64 | 2202 | 0.014 | 10.570 |
| Age band | 65+ | 1886 | 0.024 | 9.050 |
| Age band | <18 | 271 | 0.044 | 1.300 |
| Age band | SUPPRESSED | 38 | 0.026 | 0.180 |
| Sex (H/V as coded in source) | H | 9037 | 0.030 | 43.370 |
| Sex (H/V as coded in source) | V | 11799 | 0.030 | 56.630 |
| Foreign-born | N | 19945 | 0.030 | 95.720 |
| Foreign-born | S | 891 | 0.037 | 4.280 |

## 2. Champion model, by group

| attribute | group | customers | base_drift_rate | selection_rate | recall_tpr | false_positive_rate | disparate_impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age band | 18-24 | 711 | 0.063 | 0.781 | 0.933 | 0.770 | 2.601 | Check: DI outside 0.8-1.25 |
| Age band | 25-34 | 626 | 0.034 | 0.465 | 0.905 | 0.450 | 1.549 | Check: DI outside 0.8-1.25 |
| Age band | 35-44 | 985 | 0.024 | 0.187 | 0.542 | 0.178 | 0.622 | Check: DI outside 0.8-1.25 |
| Age band | 45-54 | 971 | 0.021 | 0.113 | 0.600 | 0.103 | 0.377 | Check: DI outside 0.8-1.25 |
| Age band | 55-64 | 451 | 0.011 | 0.115 | 0.400 | 0.112 | 0.384 | Too small to judge (<500) |
| Age band | 65+ | 365 | 0.022 | 0.088 | 0.500 | 0.078 | 0.292 | Too small to judge (<500) |
| Age band | <18 | 49 | 0.061 | 0.490 | 1.000 | 0.457 | 1.632 | Too small to judge (<500) |
| Age band | SUPPRESSED | 10 | 0.000 | 0.300 |  | 0.300 | 1.000 | Too small to judge (<500) |
| Sex (H/V as coded in source) | H | 1812 | 0.029 | 0.343 | 0.846 | 0.328 | 1.144 | OK |
| Sex (H/V as coded in source) | V | 2356 | 0.031 | 0.267 | 0.689 | 0.253 | 0.889 | OK |
| Foreign-born | N | 3983 | 0.030 | 0.299 | 0.767 | 0.285 | 0.997 | OK |
| Foreign-born | S | 185 | 0.032 | 0.319 | 0.500 | 0.313 | 1.063 | Too small to judge (<500) |

## 3. Summary per attribute (groups large enough to judge)

| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.145 | 0.667 | False | False | False |
| Sex (H/V as coded in source) | 2 | 0.778 | 0.157 | True | False | False |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

## 4. Fairlearn cross-check (all groups, including small ones)

| attribute | fairlearn_dp_ratio_all_groups | fairlearn_eo_difference_all_groups | fairlearn_selection_rate_max_gap |
| --- | --- | --- | --- |
| Age band | 0.112 | 1.000 | 0.693 |
| Sex (H/V as coded in source) | 0.778 | 0.157 | 0.076 |
| Foreign-born | 0.938 | 0.267 | 0.020 |

## 5. Mitigation: ExponentiatedGradient, equalized odds on age_band

| attribute | group | customers | base_drift_rate | selection_rate | recall_tpr | false_positive_rate | disparate_impact | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Age band | 18-24 | 711 | 0.063 | 0.335 | 0.333 | 0.335 | 1.133 | OK |
| Age band | 25-34 | 626 | 0.034 | 0.323 | 0.476 | 0.317 | 1.093 | OK |
| Age band | 35-44 | 985 | 0.024 | 0.284 | 0.500 | 0.279 | 0.963 | OK |
| Age band | 45-54 | 971 | 0.021 | 0.280 | 0.200 | 0.282 | 0.949 | OK |
| Age band | 55-64 | 451 | 0.011 | 0.262 | 0.400 | 0.260 | 0.886 | Too small to judge (<500) |
| Age band | 65+ | 365 | 0.022 | 0.293 | 0.250 | 0.294 | 0.993 | Too small to judge (<500) |
| Age band | <18 | 49 | 0.061 | 0.265 | 0.333 | 0.261 | 0.898 | Too small to judge (<500) |
| Age band | SUPPRESSED | 10 | 0.000 | 0.100 |  | 0.100 | 0.339 | Too small to judge (<500) |
| Sex (H/V as coded in source) | H | 1812 | 0.029 | 0.295 | 0.385 | 0.292 | 0.998 | OK |
| Sex (H/V as coded in source) | V | 2356 | 0.031 | 0.296 | 0.351 | 0.294 | 1.002 | OK |
| Foreign-born | N | 3983 | 0.030 | 0.296 | 0.367 | 0.294 | 1.003 | OK |
| Foreign-born | S | 185 | 0.032 | 0.276 | 0.333 | 0.274 | 0.933 | Too small to judge (<500) |

| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.837 | 0.300 | True | False | False |
| Sex (H/V as coded in source) | 2 | 0.996 | 0.033 | True | True | True |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

Performance trade-off (test set):

| model | recall | precision | f1 | roc_auc | flagged_rate |
| --- | --- | --- | --- | --- | --- |
| champion | 0.754 | 0.076 | 0.138 | 0.812 | 0.300 |
| mitigated | 0.365 | 0.037 | 0.068 | 0.712 | 0.295 |

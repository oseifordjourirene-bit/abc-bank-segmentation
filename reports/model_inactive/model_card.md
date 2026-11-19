# Model Card – abc-bank-drift-model-inactive (version 1)

## Model details
- **Owner:** Irene Osei Fordjour, ABC Bank Segmentation project (BAN6800)
- **Algorithm:** logistic_regression (scikit-learn pipeline: preprocessing + classifier)
- **Trained:** 2026-11-19T03:50:28+00:00 · MLflow run `c41085a4542544b6bcb878afb19759c0`
- **Decision threshold:** 0.5445 (chosen on cross-validated training predictions to reach recall ≥ 0.7)

## Intended use
- Rank active retail customers by their risk of drifting in the next three months, so the bank can send a reactivation offer early.
- Users: retail marketing and relationship managers. The score supports, not replaces, human decisions.
- **Out of scope:** credit decisions, pricing, account closure, or any adverse action against a customer.

## Data
- Santander Product Recommendation (Kaggle, 2016) as a proxy for ABC Bank; Module 3 clean, anonymised tables.
- Unit: one customer active in 2016-02. Features from 2015-12 to 2016-02; outcome measured at 2016-05 (3-month horizon).
- Target (`drift_3m`): Became inactive (activity flag 1 in February, 0 in May).
- 20,836 customers, drift rate 3.0%. Stratified 80/20 hold-out; 5-fold stratified cross-validation for tuning.
- Age, sex, nationality and province are excluded from the inputs.

## Performance (hold-out test set)
Acceptance criteria: recall ≥ 0.7 and ROC-AUC ≥ 0.75. The champion **meets** them.

| model | cv_roc_auc | roc_auc | recall | precision | f1 | accuracy | top20_capture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_majority |  | 0.500 | 0.000 | 0.000 | 0.000 | 0.970 | 0.200 |
| baseline_business_rule |  | 0.653 | 0.429 | 0.098 | 0.160 | 0.864 | 0.492 |
| logistic_regression | 0.783 | 0.812 | 0.754 | 0.076 | 0.138 | 0.715 | 0.651 |
| random_forest | 0.779 | 0.793 | 0.706 | 0.069 | 0.126 | 0.705 | 0.627 |
| lightgbm | 0.778 | 0.809 | 0.738 | 0.073 | 0.133 | 0.709 | 0.603 |

## Fairness
| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.145 | 0.667 | False | False | False |
| Sex (H/V as coded in source) | 2 | 0.778 | 0.157 | True | False | False |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

Full detail: `reports/model/fairness_report.md`.

## What the model learned (top SHAP features)
| feature | mean_abs_shap | mean_shap |
| --- | --- | --- |
| Products held | 0.689 | -0.069 |
| Tenure (months) | 0.319 | 0.011 |
| Months active (Dec-Feb) | 0.230 | -0.082 |
| Has direct debit | 0.115 | -0.022 |
| Months observed (Dec-Feb) | 0.108 | 0.057 |
| Products dropped (Dec-Feb) | 0.104 | -0.029 |
| Salary paid in | 0.074 | 0.006 |
| Income was missing | 0.067 | 0.011 |

## Robustness
| scenario | mean_change_in_risk | mean_abs_change | decisions_flipped_pct |
| --- | --- | --- | --- |
| Income -20% | -0.003 | 0.003 | 0.310 |
| Income +20% | 0.002 | 0.002 | 0.430 |
| Tenure -12 months | 0.010 | 0.010 | 1.580 |
| Tenure +12 months | -0.011 | 0.011 | 1.850 |
| Income reported missing | -0.025 | 0.025 | 3.960 |
| Random noise on numbers (5%) | 0.000 | 0.003 | 0.220 |
| Inactive one more month (large change) | 0.232 | 0.232 | 39.300 |

Seed stability: test ROC-AUC 0.812 ± 0.000 over 5 seeds.

## Limitations and risks
- Proxy data from a Spanish bank (2015–16); patterns must be revalidated on ABC Bank data before use in Ghana.
- One cohort and one 3-month window: seasonality and economic shocks are not captured.
- 'Activity' is the source bank's own flag; ABC Bank's definition (BoG dormancy rules) may differ.
- Counterfactual suggestions show associations, not proven causes; test offers with a control group.

## Monitoring after deployment
- Monthly: recall and precision on customers whose outcome is known; alert if recall < 0.7.
- Monthly: input drift (population stability index) on the top features; alert if PSI > 0.2.
- Quarterly: fairness table by age band, sex and foreign-born; alert if any judged group leaves 0.8–1.25.
- Retrain at least every 6 months or when an alert fires; new versions go through the MLflow registry.
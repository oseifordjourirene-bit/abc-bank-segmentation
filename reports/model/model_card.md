# Model Card – abc-bank-drift-model (version 1)

## Model details
- **Owner:** Irene Osei Fordjour, ABC Bank Segmentation project (BAN6800)
- **Algorithm:** random_forest (scikit-learn pipeline: preprocessing + classifier)
- **Trained:** 2026-11-19T03:41:12+00:00 · MLflow run `8b591270ea294e9da98d7a29489c5b2e`
- **Decision threshold:** 0.4653 (chosen on cross-validated training predictions to reach recall ≥ 0.7)

## Intended use
- Rank active retail customers by their risk of drifting in the next three months, so the bank can send a reactivation offer early.
- Users: retail marketing and relationship managers. The score supports, not replaces, human decisions.
- **Out of scope:** credit decisions, pricing, account closure, or any adverse action against a customer.

## Data
- Santander Product Recommendation (Kaggle, 2016) as a proxy for ABC Bank; Module 3 clean, anonymised tables.
- Unit: one customer active in 2016-02. Features from 2015-12 to 2016-02; outcome measured at 2016-05 (3-month horizon).
- Target (`drift_3m`): Became inactive, or dropped products without adding any (Module 3 drift rule).
- 20,836 customers, drift rate 8.6%. Stratified 80/20 hold-out; 5-fold stratified cross-validation for tuning.
- Age, sex, nationality and province are excluded from the inputs.

## Performance (hold-out test set)
Acceptance criteria: recall ≥ 0.7 and ROC-AUC ≥ 0.75. The champion **does NOT meet** them.

| model | cv_roc_auc | roc_auc | recall | precision | f1 | accuracy | top20_capture |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_majority |  | 0.500 | 0.000 | 0.000 | 0.000 | 0.914 | 0.200 |
| baseline_business_rule |  | 0.544 | 0.201 | 0.143 | 0.167 | 0.828 | 0.270 |
| logistic_regression | 0.732 | 0.721 | 0.688 | 0.150 | 0.246 | 0.637 | 0.473 |
| random_forest | 0.744 | 0.745 | 0.705 | 0.161 | 0.262 | 0.658 | 0.504 |
| lightgbm | 0.741 | 0.750 | 0.716 | 0.160 | 0.261 | 0.652 | 0.518 |

## Fairness
| attribute | groups_judged | demographic_parity_ratio | equalized_odds_difference | di_within_band | eo_within_limit | passes |
| --- | --- | --- | --- | --- | --- | --- |
| Age band | 4 | 0.650 | 0.162 | False | False | False |
| Sex (H/V as coded in source) | 2 | 0.946 | 0.098 | True | True | True |
| Foreign-born | 1 | 1.000 | 0.000 | True | True | True |

Full detail: `reports/model/fairness_report.md`.

## What the model learned (top SHAP features)
| feature | mean_abs_shap | mean_shap |
| --- | --- | --- |
| Has direct debit | 0.073 | -0.019 |
| Products added (Dec-Feb) | 0.055 | -0.022 |
| Tenure (months) | 0.035 | -0.009 |
| Has credit card | 0.024 | -0.007 |
| Products held | 0.022 | -0.003 |
| Months active (Dec-Feb) | 0.021 | -0.010 |
| Has credit product | 0.011 | -0.002 |
| Has term deposit | 0.007 | -0.001 |

## Robustness
| scenario | mean_change_in_risk | mean_abs_change | decisions_flipped_pct |
| --- | --- | --- | --- |
| Income -20% | 0.000 | 0.003 | 0.580 |
| Income +20% | -0.000 | 0.003 | 0.670 |
| Tenure -12 months | 0.008 | 0.012 | 1.660 |
| Tenure +12 months | -0.008 | 0.016 | 1.320 |
| Income reported missing | -0.001 | 0.005 | 1.100 |
| Random noise on numbers (5%) | 0.000 | 0.002 | 0.410 |
| Inactive one more month (large change) | 0.241 | 0.241 | 62.310 |

Seed stability: test ROC-AUC 0.746 ± 0.000 over 5 seeds.

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
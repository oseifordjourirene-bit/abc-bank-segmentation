# Model Card – abc-bank-drift-model (version 1)

_Updated 26 Nov 2026 for the final project: deployment, precision review, monitoring and retraining sections added._

## Model details
- **Owner:** Irene Osei Fordjour, ABC Bank Segmentation project (BAN6800)
- **Algorithm:** random_forest (scikit-learn pipeline: preprocessing + classifier)
- **Trained:** 2026-11-19T03:41:12+00:00 · MLflow run `8b591270ea294e9da98d7a29489c5b2e`
- **Decision threshold:** 0.4653 (chosen on cross-validated training predictions to reach recall ≥ 0.7)
- **Registry status:** version 1, alias `champion`, tag `stage = Rejected` (misses the AUC bar). Not approved for live decisions.
- **Live demo API:** https://abc-bank-drift-api.onrender.com/docs (public Santander data only; free tier sleeps after 15 idle minutes)

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

## Precision review and contact policy (final project)
Precision at the agreed threshold is 16% because only 8.6% of customers drift and the threshold catches 70% of them. The three algorithms reach similar PR-AUC (0.21–0.27 against a 0.086 baseline), so the lever is how many customers to contact (`python -m src.model.precision_review`).

| Contact level | Share flagged | Recall | Precision |
| --- | --- | --- | --- |
| Riskiest 5% | 5% | 23% | 39% |
| Riskiest 10% (Tier 1) | 10% | 35% | 30% |
| Riskiest 20% | 20% | 50% | 22% |
| Agreed threshold | 38% | 70% | 16% |

**Two-tier policy:** Tier 1, the riskiest 10%, gets a relationship-manager call (417 test customers, precision 30%); Tier 2, the rest of the flagged group, gets a low-cost digital nudge (1,154 customers, precision 11%). Together they keep 70% recall. The API returns each customer's `contact_tier`.

## Explanations and audit
Every `/predict` call returns and stores its top five factors (feature replacement against a typical customer), with a `request_id`, model version, inputs (behavioural only), score and decision (`api/explanation_store.py`). Retrieve with `GET /explanations/{request_id}`. Production retention: 7 years. The API refuses age, sex, nationality and province (HTTP 422).

## Monitoring after deployment
Run monthly by `flows/retrain_flow.py`; thresholds and owners in `docs/model_governance.md`.
- Data drift: PSI per input against the training cohort; watch at 0.10, alert at 0.25.
- Prediction drift: share of customers flagged; watch at ±5 points.
- Fairness drift: offer rate per age band and sex ÷ overall rate (groups ≥ 500); alert outside 0.8–1.25.
- Performance: recall, precision per tier and ROC-AUC once outcomes arrive (3 months later); alert if recall < 0.70 or AUC < 0.75.

**Findings, March–May 2016 cohorts:** inputs stable (highest PSI 0.03) and the flagged share moved less than 1.3 points. Fairness drifted: the 18–24 offer rate fell from 0.79 to 0.53 of the overall rate, and 65+ stayed at 0.60–0.67. A labelled simulated shock (income −20%, one fewer active month for 30% of customers) raised every alarm (PSI 0.80 and 0.92; flagged share 38% → 56%).

## Retraining and version history
`src/model/retrain.py` refits the champion's configuration on the latest data and registers a candidate only through four gates: recall ≥ 0.70, ROC-AUC ≥ 0.75, AUC no more than 0.01 below the champion, and every judged group within 0.8–1.25. Promotion always needs human sign-off by the model owner and Digital Governance and Risk.

| Version | Date | Status | Reason |
| --- | --- | --- | --- |
| 1 | 2026-11-19 | Champion, tagged Rejected | ROC-AUC 0.745 < 0.75; age band outside 0.8–1.25 |
| 2 | 2026-11-26 | Candidate, Rejected | Reproduces v1 (AUC 0.7449); same failed gates |

_Note: `src/model/run_all.py` regenerates this card from the Module 4 template; re-apply these final-project sections after retraining._

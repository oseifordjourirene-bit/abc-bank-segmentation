# Drift Model: Design and Validation Approach (Module 4)

Code: `src/model/`. Tests: `tests/test_model_*.py`, `tests/test_api.py`. Results come from each run and are written to `reports/model/` and MLflow.

## Predictive task

| Item | Decision |
| --- | --- |
| Business use | Send the reactivation offer before an active customer drifts towards dormancy |
| Unit of analysis | One customer, active in February 2016 (cutoff month) |
| Target `drift_3m` | 1 if by May 2016 the customer became inactive, or dropped products without adding any (Module 3 drift rule) |
| Horizon | 3 months after the cutoff |
| Features | December 2015 to February 2016 only (23 inputs: product holdings and changes, activity months, tenure, income, channel, segment, relation type) |
| Protected attributes | Age band, sex, foreign-born: fairness audit only, never inputs |
| Success criteria | Recall ≥ 0.70 (missing a drifter is the costlier error) and ROC-AUC ≥ 0.75 on the hold-out test set |
| Fairness rule | Disparate impact 0.8–1.25 per judged group; equalized odds difference ≤ 0.10; groups under 500 reported, not judged |

Customers already inactive in February are excluded: they belong to the "Drifting inactive" segment and get a different treatment.

## Leakage prevention

- Features are built from months up to the cutoff only; a test changes post-cutoff data and checks the features do not move.
- The outcome is measured only after the cutoff (February to May).
- Rare channels are grouped, and scaling is fitted, inside the scikit-learn pipeline on training folds only.
- The decision threshold is chosen on cross-validated training predictions, never on the test set.
- The champion is chosen by cross-validated ROC-AUC; the test set is used once, for the final report.

## Train/test protocol

- Stratified 80/20 hold-out (drift rate kept equal in both parts).
- 5-fold stratified cross-validation for tuning.
- Baselines: majority class, random prior, and a business rule (flag if a product was dropped or the customer was inactive in December or January).

## Models and tuning

| Model | Why | Tuning |
| --- | --- | --- |
| Logistic regression | Transparent, coefficient-based reference | C on a log scale |
| Random forest | Non-linear, robust to outliers | Trees, depth, leaf size, features per split |
| LightGBM | Strong on tabular data, fast | Leaves, learning rate, trees, leaf size, sampling, L2 |

Coarse random search, then a small grid around the best values (coarse-to-fine). Class imbalance is handled with balanced class weights and a recall-based threshold. Every candidate is an MLflow child run; the champion is registered as `abc-bank-drift-model` with alias `champion` and a `stage` tag (`Staging` if it meets the success criteria, `Rejected` if not).

## Explainability (Module 2 framework)

- Global: SHAP summary and importance plots, SHAP dependence for the top feature.
- Local: SHAP waterfall and LIME for three customers (caught drifter, borderline, low risk), compared for agreement.
- Counterfactual: smallest set of realistic offers (e-account, direct debit, card, salary account) that would take a flagged customer below the threshold; only actionable features change. An offer-effects table shows the average change in risk per offer.

## Fairness and mitigation

Fairlearn metrics by group: selection rate, recall (true positive rate), false positive rate, disparate impact, demographic parity ratio and equalized odds difference. If a judged attribute breaks a rule, Fairlearn's ExponentiatedGradient (equalized odds constraint) is trained using that attribute during training only; before and after results and the performance trade-off are reported.

## Sensitivity analysis

Income ±20%, tenure ±12 months, income reported missing, 5% noise, and one extra inactive month: the share of offer decisions that flip. Seed stability: retrain with five seeds and compare test ROC-AUC.

## Monitoring hooks

Monthly recall and precision on matured outcomes; input drift (PSI) on top features; quarterly fairness table; retrain at least every six months.

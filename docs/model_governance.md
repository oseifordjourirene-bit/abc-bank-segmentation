# Model governance: monitoring, incidents and retirement

Applies to the ABC Bank drift model (registered as `abc-bank-drift-model`). Owner: the model owner
named in the Ethical AI Charter (Module 1). Oversight: Digital Governance and Risk (chair), Internal
Control, Compliance and the Data Protection Officer.

## 1. Monthly monitoring (`python -m flows.retrain_flow`, 3rd working day of each month)
| Check | Measure | Watch | Alert | Action on alert |
| --- | --- | --- | --- | --- |
| Data drift | PSI per input vs the training cohort | 0.10 | 0.25 | Retrain a candidate; investigate the source system |
| Prediction drift | Share of customers flagged | ±5 points | — | Retrain a candidate; check campaign capacity |
| Fairness drift | Offer rate per age band and sex ÷ overall rate (groups ≥ 500) | change ≥ 0.10 | outside 0.8–1.25 | Open a fairness incident (section 3) |
| Performance (3 months later) | Recall, precision at each tier, ROC-AUC | recall < 0.72 | recall < 0.70 or AUC < 0.75 | Retrain; review the definition of drift |
| Business outcome | Reactivation rate vs control group | — | no uplift after 2 cycles | Pause campaign; review with Marketing |

## 2. Retraining and promotion (`src/model/retrain.py`)
A candidate is refit with the champion's settings on the latest data and tested on an untouched split.
Gates: recall ≥ 0.70, ROC-AUC ≥ 0.75, AUC no more than 0.01 below the champion, every judged group inside
0.8–1.25. A passing model is tagged `Candidate`; it goes live only when the model owner and Digital
Governance and Risk sign off and the `champion` alias is moved. Every decision is logged in MLflow.

## 3. Incident response for ethical failures
| Step | Who | When |
| --- | --- | --- |
| Detect: monitoring alert, complaint, staff report or audit finding | Anyone | — |
| Log the incident and notify the model owner and Digital Governance and Risk | Model owner | 1 working day |
| Contain: pause the affected tier or segment (scores still logged, no offers sent) | Model owner | 2 working days |
| Investigate with the explanation store and fairness tables | Data and analytics team, Internal Control | 10 working days |
| Fix: retrain, change threshold or tier, or retire (section 4) | Model owner, approved by the ethics body | — |
| Tell affected customers where an offer was unfairly withheld; report to the DPO | Compliance, DPO | as required by Act 843 |
| Close with a lessons-learned note in the RAID log | Model owner | — |

## 4. Decommissioning
Retire the model when it fails its gates for two consecutive quarters, the business need ends, a better
approved model replaces it, or the ethics body orders it. Steps: announce the date to users; switch
campaign lists back to the previous process; disable the API key and endpoint; set the MLflow version tag
`stage=Archived`; keep the model, explanation store and fairness reports for 7 years for audit, then
delete them under the bank's retention schedule; record the decision in the RAID log.

## 5. Explanation store
Every prediction is saved with its inputs (behavioural only), score, decision, contact tier, model version
and top factors. Access: Internal Control and the model owner. Retention: 7 years. The free Render demo
resets on restart; production uses the bank's governed database in Ghana (CISD 2026; Act 843).

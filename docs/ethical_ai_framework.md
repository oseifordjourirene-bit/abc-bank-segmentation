# Ethical AI framework: ABC Bank drift early-warning model

Version 1.0 · 26 Nov 2026 · Owner: model owner (Digital Transformation) · Approver: Digital Governance and Risk

This framework turns the Module 1 Ethical AI Charter into controls that run in the code, and points to the
evidence for each. It matches Section 8 of the integrated report. Operating detail (monitoring thresholds,
incident response, decommissioning) is in [`model_governance.md`](model_governance.md).

## 1. Principles, controls and evidence
| Charter principle | Control | Where it runs | Evidence |
| --- | --- | --- | --- |
| Fairness | Protected traits (age, sex, nationality, province) are never model inputs | `src/model/config.py`; API schema refuses extra fields | API returns HTTP 422 if `age` is sent (CI smoke test) |
| Fairness | Offer rate per age band and sex checked against 0.8–1.25 of overall rate | `src/model/fairness.py`, `src/monitoring/monitor.py` (monthly) | `reports/model/fairness_report.md`; `reports/monitoring/fairness_drift.csv` |
| Fairness | Retrained models cannot be promoted if any judged group leaves the band | `src/model/retrain.py` gates | `reports/retraining/` (both candidates rejected) |
| Transparency | Every score returned with its top five factors | `src/model/explain_local.py`, `api/main.py` | Live API response (`top_factors`) |
| Transparency | Global and local explanations; plain-language dashboard | SHAP notebook, LIME, dashboard tabs | `notebooks/04_shap_analysis.ipynb`; live dashboard |
| Accountability | Named model owner; human sign-off for every promotion | MLflow registry tags; `retrain.py` never moves the champion alias | Registry: v1 and v2 tagged Rejected |
| Accountability | Every decision stored for audit | `api/explanation_store.py` | `GET /explanations/{request_id}` |
| Privacy | Salted HMAC-SHA256 keys; protected traits in a restricted table; k ≥ 5 | `src/anonymise.py` (Module 3) | `docs/anonymisation_plan.md`; pipeline audit log |
| Privacy | No customer records in public tools; production data stays in Ghana | Dashboard uses aggregates and fictional profiles only | `docs/data_governance.md` |
| Inclusion | Every flagged customer gets an offer; tiers change the channel, not who is served | `src/model/precision_review.py` | `reports/model/precision_tiers.csv` |

## 2. Algorithmic impact assessment
| Impact area | Risk | Likelihood / severity | Control (built or required before use) |
| --- | --- | --- | --- |
| Unequal access to a benefit | Young or older customers offered less (18–24 at 0.53x by May 2016) | High / Medium | Monthly fairness drift alert; tier pause; human review |
| Wasted or intrusive contact | About 6 offers per drifter at the agreed threshold | High / Low | Two-tier policy (built); contact cap and opt-out (required) |
| Misuse for adverse decisions | Score reused for credit or closures | Low / High | Charter bans it; scores released only as campaign lists |
| Privacy | Profiling without notice; data leaving Ghana | Medium / High | Act 843 notice and right to object; Ghana hosting; pseudonymised keys |
| Opacity | Staff cannot explain an offer | Medium / Medium | Top factors on every prediction; explanation store |
| Model decay | Behaviour shifts after training | Medium / Medium | PSI alerts; gated retraining; decommissioning plan |

Method follows Reisman et al. (2018). Reassess before the pilot and whenever the drift definition, data source or use changes.

## 3. Regulatory compliance
Ghana's Data Protection Act, 2012 (Act 843) and the Bank of Ghana cyber and information security directive apply.
GDPR and CCPA are used as international benchmarks. HIPAA does not apply: no health data is processed.

| Requirement | Ghana (applies) | GDPR / CCPA (benchmark) | How the project meets it |
| --- | --- | --- | --- |
| Lawful basis and notice | Act 843: consent and notice | GDPR Art. 6, 13; CCPA notice | Required before pilot: consent flag checked before targeting; privacy notice covering profiling |
| Right to object to profiling | Act 843: data-subject rights | GDPR Art. 21–22; CCPA opt-out | Required before pilot: opt-out list removes customers from scoring output |
| Data minimisation | Act 843 principles | GDPR Art. 5(1)(c) | 23 behavioural inputs; no names, IDs or protected traits |
| Security and residency | BoG directive | GDPR Art. 32 | HMAC keys, k ≥ 5, role-based access; production hosted in Ghana |
| Accountability and records | BoG directive; Act 843 | GDPR Art. 30, 35 (DPIA) | Audit log, MLflow, explanation store, this assessment |

## 4. Public trust statement
> ABC Bank uses an analytics model to spot customers who may be drifting away, so that we can offer help earlier.
> The model looks only at how you use your accounts, never at your age, sex, nationality or where you live.
> It can lead to an offer, never to a refusal, a higher price or a closed account. People review how fairly it
> works every month, and you can ask why you were contacted or opt out at any branch.

## 5. Ethical governance structure
| Body | Role in ongoing oversight | Cadence |
| --- | --- | --- |
| Model owner (Digital Transformation) | Runs monitoring; signs each release; opens incidents | Monthly |
| Digital Governance and Risk (chair) | Approves definition, fairness tolerance and promotions; can pause any tier | Monthly + on alert |
| Internal Control | Audits decisions through the explanation store and audit log | Quarterly |
| Compliance and DPO | Consent, profiling notices, Act 843 and BoG reporting | Quarterly |
| Retail and Marketing heads | Business owners: capacity, offer design, outcome review | Each campaign |

## 6. Review
This framework is reviewed every six months, after any incident, and before the pilot on ABC Bank data.
Changes need approval from Digital Governance and Risk and are recorded in the RAID log.

**References:** Republic of Ghana (2012) Data Protection Act (Act 843); Bank of Ghana (2026) Cyber and
information security directive; Regulation (EU) 2016/679 (GDPR); California Consumer Privacy Act of 2018;
Reisman, D., Schultz, J., Crawford, K., & Whittaker, M. (2018). *Algorithmic impact assessments*. AI Now Institute.

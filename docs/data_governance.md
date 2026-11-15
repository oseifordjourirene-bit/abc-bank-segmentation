# Data Governance Framework

Builds on the Module 1 Ethical AI Charter and the Module 2 Data Privacy Plan. Applies to every dataset the pipeline reads or writes.

## Roles

| Role | Held by (ABC Bank phase) | Responsibility |
| --- | --- | --- |
| Data owner | Head of Retail | Approves each use of customer data |
| Data steward | Data and analytics lead | Runs the pipeline, maintains quality rules and the data dictionary |
| Data Protection Officer | DPO | Approves anonymisation, reviews the audit log quarterly |
| Ethics body | Digital Governance and Risk (chair), Internal Control, Compliance, DPO | Reviews bias reports before any model release |
| Analysts and modellers | Data team | Use the analytics tables only |

## Access control

| Dataset | Contents | Who can read | Who can write |
| --- | --- | --- | --- |
| `data/raw/` | Source snapshots with original IDs | Data steward | Ingestion step only |
| `clean_monthly_anonymised` | Cleaned monthly data, pseudonymised keys, no protected attributes | Data team | Pipeline only |
| `customer_features` | One row per customer, behavioural features | Data team, modellers | Pipeline only |
| `restricted_audit_attributes` | Age band, sex, foreign-born, generalised province (k ≥ 5) | Ethics body, DPO, bias check step | Pipeline only |
| `logs/audit_log.jsonl` | Who accessed or changed which dataset, when | DPO, Internal Control | Append-only by the pipeline |

Least privilege applies: no role gets more than its task needs. Access is reviewed quarterly.

## Usage rules

- Data is used only for segmentation and product-matching analytics, the purpose customers consented to (Data Protection Act, 2012, Act 843).
- Protected attributes are never model inputs. They are used only to audit outcomes.
- No data leaves the approved environment or goes to third parties.
- In the ABC Bank phase, customer financial and personal data stays in Ghana (Bank of Ghana CISD 2026). Azure holds only the public Santander proxy data.

## Retention

| Dataset | Kept for | Then |
| --- | --- | --- |
| Raw extracts | Course duration (proxy data); bank records policy (ABC Bank phase) | Deleted |
| Processed tables | Until replaced by the next pipeline run | Overwritten, previous versions held in DVC |
| Training snapshots | Life of the model plus one year | Deleted |
| Audit log | At least 5 years | Archived |

## Quality and change control

- Great Expectations validates every run; a failed check stops the pipeline before any output is written.
- Code changes follow Git Flow with Jira keys in commit messages; data versions are tracked with DVC.
- Each run writes `reports/lineage.json` with row counts per stage, so any output traces back to its input.

# ABC Bank Customer Segmentation

Behavioural customer segmentation for a Ghanaian retail bank (ABC Bank, a pseudonym), built for the BAN6800 Business Analytics capstone at Nexford University.

## Goal
Understand customers by how they actually bank, group them into behavioural segments, and recommend the product each customer is most likely to need:

| Proposed segment | Offer |
| --- | --- |
| Salaried | Salary advance |
| Savers | Goal savings |
| Current-account holders | Debit card options |
| Long-term depositors | Treasury bill investments, deposit-secured loans |
| Drifting inactive | Reactivation offer |

The five segments are business hypotheses; clustering decides the final groupings.

## Data
[Santander Product Recommendation](https://www.kaggle.com/c/santander-product-recommendation) (Kaggle, 2016): real, anonymised monthly bank records, January 2015 to May 2016. Download `train_ver2.csv` after accepting the competition rules and place it in `data/raw/`. Raw data is tracked with DVC and never committed. See `docs/data_dictionary.md`.

## Repository structure
```
abc-bank-segmentation/
├── data/              # raw and processed data (not committed; DVC-tracked)
├── docs/              # data dictionary, pipeline, governance, anonymisation plan, RAID log
├── flows/             # Prefect orchestration DAG
├── notebooks/         # exploration and modelling
├── reports/           # bias report and charts
├── src/               # pipeline stages: ingest, clean, anonymise, features, validate, bias, audit
├── tests/             # pytest suite
├── Dockerfile
├── requirements.txt            # full project
├── requirements-pipeline.txt   # Module 3 pipeline only
└── README.md
```

## Run the data pipeline (Module 3)

1. Download `train_ver2.csv.zip` from the Kaggle competition page, then make the six-month sample:
   ```bash
   python src/make_sample.py          # run in the folder holding train_ver2.csv.zip
   ```
   Move `santander_sample.csv` into `data/raw/`.
2. Install and run (Python 3.11):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate             # macOS/Linux: source .venv/bin/activate
   pip install -r requirements-pipeline.txt
   set ANON_SALT=choose-a-long-secret  # PowerShell: $env:ANON_SALT="choose-a-long-secret"
   python -m flows.pipeline_flow
   pytest -q
   ```
3. Outputs: `data/processed/` (Parquet tables), `reports/bias/`, `logs/audit_log.jsonl`, and Great Expectations Data Docs at `gx/uncommitted/data_docs/local_site/index.html`.

### Docker
```bash
docker build -t abc-bank-pipeline .
docker run --rm -e ANON_SALT=choose-a-long-secret -v "%cd%/data:/app/data" -v "%cd%/reports:/app/reports" -v "%cd%/logs:/app/logs" -v "%cd%/gx:/app/gx" abc-bank-pipeline
```
(PowerShell: replace `%cd%` with `${PWD}`.)

See `docs/pipeline.md` for stages and lineage, `docs/data_governance.md` and `docs/anonymisation_plan.md` for governance.

## Project management
Work is planned on a Jira Scrum board: 24 tickets across five sprints, one sprint per course module. Commit messages quote the Jira ticket key (for example, `S2-02 Rename columns to English`).

## Branching
Git Flow: `main` (released), `develop`, `feature/*`, `release/*`, `hotfix/*`.

## Ethics
Age, sex, nationality and province are excluded from clustering and used only for fairness audits (Fairlearn). Predictions are explained with SHAP.

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
├── data/              # raw and processed data (DVC-tracked)
├── notebooks/         # exploration and modelling
├── src/               # reusable Python code
├── reports/           # figures and results
├── docs/              # data dictionary and project documents
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Project management
Work is planned on a Jira Scrum board: 24 tickets across five sprints, one sprint per course module. Commit messages quote the Jira ticket key (for example, `S2-02 Rename columns to English`).

## Branching
Git Flow: `main` (released), `develop`, `feature/*`, `release/*`, `hotfix/*`.

## Ethics
Age, sex, nationality and province are excluded from clustering and used only for fairness audits (Fairlearn). Predictions are explained with SHAP.

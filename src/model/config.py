"""Settings for the Module 4 drift model. Decisions agreed by the project owner are marked."""
from src.config import PROCESSED_DIR, REPORTS_DIR, ROOT

# Inputs: the Module 3 pipeline outputs
MONTHLY_FILE = PROCESSED_DIR / "clean_monthly_anonymised.parquet"
AUDIT_FILE = PROCESSED_DIR / "restricted_audit_attributes.parquet"

# Outputs
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "drift_model.joblib"
MODEL_META = MODEL_DIR / "model_meta.json"
OUT_DIR = REPORTS_DIR / "model"

# MLflow: local SQLite store so the Model Registry works on a laptop
MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT = "abc-bank-drift-model"
REGISTERED_MODEL = "abc-bank-drift-model"

# Time windows (leakage control): features up to the cutoff, outcome after it
FEATURE_MONTHS = ["2015-12-28", "2016-01-28", "2016-02-28"]
CUTOFF = "2016-02-28"
OUTCOME_MONTH = "2016-05-28"          # 3-month horizon

# Success criteria (project owner decisions, Module 4)
RECALL_TARGET = 0.70                  # missing a drifter is the costlier error
AUC_TARGET = 0.75
DI_BAND = (0.8, 1.25)                 # disparate impact band from Modules 1-3
EO_GAP_MAX = 0.10                     # equalized odds difference limit
MIN_GROUP_SIZE = 500                  # smaller groups are reported, not judged

TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42

# Used only to audit fairness, never as model inputs
SENSITIVE = {"age_band": "Age band", "sex": "Sex (H/V as coded in source)",
             "foreign_born_flag": "Foreign-born"}

NUMERIC_FEATURES = [
    "product_count", "products_added_2m", "products_dropped_2m", "months_active_3m",
    "months_observed_3m", "tenure_months", "household_income",
]
BINARY_FEATURES = [
    "has_current_account", "has_salary", "has_pension_credit", "has_term_deposit",
    "has_investments", "has_credit", "has_card", "has_e_account", "has_direct_debit",
    "has_junior_account", "income_was_missing", "new_customer_flag", "is_primary_customer",
]
CATEGORICAL_FEATURES = ["joining_channel", "bank_segment", "relation_type"]
FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES

# Features a customer or the bank can realistically change (used for counterfactuals)
ACTIONABLE = ["has_e_account", "has_direct_debit", "has_card", "has_salary",
              "products_added_2m", "product_count"]
TARGET = "drift_3m"
# Project owner decision: main target keeps the Module 3 drift rule
TARGET_RULES = {
    "inactive_or_dropping": "Became inactive, or dropped products without adding any (Module 3 drift rule)",
    "inactive": "Became inactive (activity flag 1 in February, 0 in May)",
}
DEFAULT_TARGET_RULE = "inactive_or_dropping"

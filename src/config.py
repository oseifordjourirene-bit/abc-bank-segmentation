"""Shared settings for the ABC Bank Segmentation data pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "santander_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"
AUDIT_LOG = LOG_DIR / "audit_log.jsonl"

# Spanish source columns -> English names (see docs/data_dictionary.md)
COLUMN_MAP = {
    "fecha_dato": "snapshot_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_status",
    "pais_residencia": "country_of_residence",
    "sexo": "sex",
    "age": "age",
    "fecha_alta": "join_date",
    "ind_nuevo": "new_customer_flag",
    "antiguedad": "tenure_months",
    "indrel": "primary_customer_flag",
    "ult_fec_cli_1t": "last_primary_date",
    "indrel_1mes": "customer_type_start_month",
    "tiprel_1mes": "relation_type_start_month",
    "indresi": "resident_flag",
    "indext": "foreign_born_flag",
    "conyuemp": "spouse_of_employee",
    "canal_entrada": "joining_channel",
    "indfall": "deceased_flag",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "active_customer_flag",
    "renta": "household_income",
    "segmento": "bank_segment",
    "ind_ahor_fin_ult1": "savings_account",
    "ind_aval_fin_ult1": "guarantees",
    "ind_cco_fin_ult1": "current_account",
    "ind_cder_fin_ult1": "derivatives_account",
    "ind_cno_fin_ult1": "payroll_account",
    "ind_ctju_fin_ult1": "junior_account",
    "ind_ctma_fin_ult1": "current_account_mas_particular",
    "ind_ctop_fin_ult1": "current_account_particular",
    "ind_ctpp_fin_ult1": "current_account_particular_plus",
    "ind_deco_fin_ult1": "short_term_deposit",
    "ind_deme_fin_ult1": "medium_term_deposit",
    "ind_dela_fin_ult1": "long_term_deposit",
    "ind_ecue_fin_ult1": "e_account",
    "ind_fond_fin_ult1": "funds",
    "ind_hip_fin_ult1": "mortgage",
    "ind_plan_fin_ult1": "pension_plan",
    "ind_pres_fin_ult1": "loan",
    "ind_reca_fin_ult1": "taxes",
    "ind_tjcr_fin_ult1": "credit_card",
    "ind_valo_fin_ult1": "securities",
    "ind_viv_fin_ult1": "home_account",
    "ind_nomina_ult1": "salary_credited",
    "ind_nom_pens_ult1": "pension_credited",
    "ind_recibo_ult1": "direct_debit",
}

PRODUCT_COLUMNS = [v for k, v in COLUMN_MAP.items() if k.endswith("ult1")]

# Columns dropped during cleaning (near-empty or constant; see data dictionary)
DROP_COLUMNS = ["spouse_of_employee", "last_primary_date", "address_type", "resident_flag"]

# Protected or proxy attributes: kept only in the restricted audit file, never as model features
AUDIT_ONLY_COLUMNS = ["sex", "age", "foreign_born_flag", "country_of_residence",
                      "province_code", "province_name"]

# Direct identifiers removed if they ever appear (none exist in the Santander data)
PII_COLUMNS = ["name", "full_name", "first_name", "last_name", "email", "phone",
               "address", "national_id", "ghana_card", "account_number", "date_of_birth"]

# Cleaning rules agreed by the project owner (Module 3)
MAX_AGE = 100                 # ages above this are treated as data errors
INCOME_CAP_QUANTILE = 0.99    # cap extreme incomes at the 99th percentile
MIN_GROUP_SIZE = 500          # Module 2 rule: compare groups only with 500+ customers
PARITY_BAND = (0.8, 1.25)     # Module 1/2 disparate impact band
K_TARGET = 5                  # k-anonymity target for the restricted audit file
PROVINCE_MIN = 500            # provinces with fewer customers are merged into OTHER

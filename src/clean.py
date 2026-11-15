"""Stage 2 - Cleaning: English names, correct types, filters, missing values, duplicates."""
import pandas as pd

from src.audit import audit
from src.config import (COLUMN_MAP, DROP_COLUMNS, INCOME_CAP_QUANTILE, MAX_AGE,
                        PRODUCT_COLUMNS)

NUMERIC_COLUMNS = ["age", "tenure_months", "household_income", "new_customer_flag",
                   "active_customer_flag", "primary_customer_flag"]


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_MAP)


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """Strip stray spaces, convert numbers and dates, standardise codes."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "NA": pd.NA})
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["household_income"] = df["household_income"].astype("Float64")
    for col in PRODUCT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df["join_date"] = pd.to_datetime(df["join_date"], errors="coerce")
    # '1', '1.0' and 1 all mean the same customer type
    df["customer_type_start_month"] = (df["customer_type_start_month"].astype("string")
                                       .str.replace(r"\.0$", "", regex=True))
    return df


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def apply_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove records that should not be segmented. Minors are kept (junior accounts)."""
    counts = {}
    rules = {
        "bank_employees": df["employee_status"].fillna("N") != "N",
        "deceased": df["deceased_flag"].fillna("N") == "S",
        f"age_over_{MAX_AGE}": df["age"] > MAX_AGE,
        "missing_id_or_date": df["customer_id"].isna() | df["snapshot_date"].isna(),
    }
    keep = pd.Series(True, index=df.index)
    for name, mask in rules.items():
        counts[name] = int((mask & keep).sum())
        keep &= ~mask
    return df[keep].copy(), counts


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.sort_values(["customer_id", "snapshot_date"]).drop_duplicates(
        subset=["customer_id", "snapshot_date"], keep="last")
    return df, before - len(df)


def handle_income(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Cap extreme incomes, then fill gaps with the overall median and flag them."""
    df = df.copy()
    cap = float(df["household_income"].quantile(INCOME_CAP_QUANTILE))
    capped = int((df["household_income"] > cap).sum())
    df["household_income"] = df["household_income"].clip(upper=cap)
    median = float(df["household_income"].median())
    df["income_was_missing"] = df["household_income"].isna().astype(int)
    filled = int(df["income_was_missing"].sum())
    df["household_income"] = df["household_income"].fillna(median)
    return df, {"income_cap": round(cap, 2), "rows_capped": capped,
                "income_median": round(median, 2), "rows_imputed": filled}


def fill_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["bank_segment", "joining_channel", "province_name", "province_code",
                "customer_type_start_month", "relation_type_start_month", "sex"]:
        df[col] = df[col].fillna("UNKNOWN")
    df["tenure_months"] = df["tenure_months"].clip(lower=0)
    return df


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Run every cleaning step in order and return the clean table plus a step report."""
    report = {"rows_raw": len(raw)}
    df = drop_unused_columns(fix_types(rename_columns(raw)))
    df, report["removed"] = apply_filters(df)
    df, report["duplicates_removed"] = remove_duplicates(df)
    df, report["income"] = handle_income(df)
    df = fill_categories(df).reset_index(drop=True)
    report["rows_clean"] = len(df)
    audit("transform", "clean_monthly", rows_in=len(raw), rows_out=len(df),
          columns=df.columns, details=report)
    return df, report

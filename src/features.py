"""Stage 4 - Transformation and integration: monthly snapshots -> one row per customer."""
import pandas as pd

from src.audit import audit
from src.config import PRODUCT_COLUMNS

CURRENT_ACCOUNTS = ["current_account", "current_account_mas_particular",
                    "current_account_particular", "current_account_particular_plus"]
TERM_DEPOSITS = ["short_term_deposit", "medium_term_deposit", "long_term_deposit"]
INVESTMENTS = ["funds", "securities", "pension_plan"]
CREDIT = ["loan", "mortgage", "credit_card"]


def build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Integrate six monthly snapshots into a customer-level analysis table for Module 4."""
    df = monthly.sort_values(["customer_key", "snapshot_date"])
    months = sorted(df["snapshot_date"].unique())
    latest_month, base_month = months[-1], months[-4] if len(months) >= 4 else months[0]

    latest = df[df["snapshot_date"] == latest_month].set_index("customer_key")
    base = df[df["snapshot_date"] == base_month].set_index("customer_key")[PRODUCT_COLUMNS]
    base = base.reindex(latest.index)

    out = pd.DataFrame(index=latest.index)
    out["product_count"] = latest[PRODUCT_COLUMNS].sum(axis=1)
    out["has_current_account"] = latest[CURRENT_ACCOUNTS].max(axis=1)
    out["has_salary"] = latest[["payroll_account", "salary_credited"]].max(axis=1)
    out["has_pension_credit"] = latest["pension_credited"]
    out["has_savings"] = latest["savings_account"]
    out["has_term_deposit"] = latest[TERM_DEPOSITS].max(axis=1)
    out["has_investments"] = latest[INVESTMENTS].max(axis=1)
    out["has_credit"] = latest[CREDIT].max(axis=1)
    out["has_card"] = latest["credit_card"]
    out["has_e_account"] = latest["e_account"]
    out["has_direct_debit"] = latest["direct_debit"]
    out["has_junior_account"] = latest["junior_account"]

    added = (latest[PRODUCT_COLUMNS] - base.fillna(0)).clip(lower=0).sum(axis=1)
    dropped = (base.fillna(0) - latest[PRODUCT_COLUMNS]).clip(lower=0).sum(axis=1)
    out["products_added_3m"] = added.astype(int)
    out["products_dropped_3m"] = dropped.astype(int)

    activity = df.pivot_table(index="customer_key", columns="snapshot_date",
                              values="active_customer_flag", aggfunc="max")
    out["months_observed"] = activity.notna().sum(axis=1).reindex(out.index)
    out["months_active_6m"] = activity.sum(axis=1).reindex(out.index).astype(int)
    out["active_now"] = latest["active_customer_flag"].astype(int)
    was_active = activity[base_month].reindex(out.index) if base_month in activity else None
    lost_activity = (was_active == 1).fillna(False) & (out["active_now"] == 0)
    only_dropping = (out["products_dropped_3m"] > 0) & (out["products_added_3m"] == 0)
    out["drift_signal"] = (lost_activity | only_dropping).astype(int)

    for col in ["tenure_months", "household_income", "income_was_missing",
                "new_customer_flag", "joining_channel", "bank_segment",
                "relation_type_start_month"]:
        out[col] = latest[col]
    out["snapshot_date"] = latest_month
    out = out.reset_index()
    audit("transform", "customer_features", rows_in=len(monthly), rows_out=len(out),
          columns=out.columns, details={"latest_month": str(pd.Timestamp(latest_month).date()),
                                        "base_month": str(pd.Timestamp(base_month).date())})
    return out

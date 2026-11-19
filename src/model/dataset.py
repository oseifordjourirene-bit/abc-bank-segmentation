"""Build the modelling table: features up to February 2016, drift outcome by May 2016.

Unit of analysis: one customer who was active in February 2016 (the cutoff month).
Target (drift_3m): by May 2016 the customer became inactive, or dropped products
without adding any. Same rule as the Module 3 drift_signal, measured after the cutoff.
"""
import pandas as pd

from src.audit import audit
from src.config import PRODUCT_COLUMNS
from src.features import CREDIT, CURRENT_ACCOUNTS, INVESTMENTS, TERM_DEPOSITS
from src.model.config import (AUDIT_FILE, CUTOFF, DEFAULT_TARGET_RULE, FEATURE_MONTHS, FEATURES,
                              MONTHLY_FILE, OUTCOME_MONTH, SENSITIVE, TARGET)


def load_inputs(monthly_path=MONTHLY_FILE, audit_path=AUDIT_FILE):
    return pd.read_parquet(monthly_path), pd.read_parquet(audit_path)


def _snapshot(monthly: pd.DataFrame, month: str) -> pd.DataFrame:
    return monthly[monthly["snapshot_date"] == pd.Timestamp(month)].set_index("customer_key")


def build_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Customer features using only the months up to and including the cutoff."""
    window = monthly[monthly["snapshot_date"].isin(pd.to_datetime(FEATURE_MONTHS))]
    assert window["snapshot_date"].max() <= pd.Timestamp(CUTOFF), "feature window leaks past cutoff"
    now = _snapshot(window, CUTOFF)
    start = _snapshot(window, FEATURE_MONTHS[0])[PRODUCT_COLUMNS].reindex(now.index).fillna(0)
    products_now = now[PRODUCT_COLUMNS]

    f = pd.DataFrame(index=now.index)
    f["product_count"] = products_now.sum(axis=1)
    f["products_added_2m"] = (products_now - start).clip(lower=0).sum(axis=1)
    f["products_dropped_2m"] = (start - products_now).clip(lower=0).sum(axis=1)
    activity = window.pivot_table(index="customer_key", columns="snapshot_date",
                                  values="active_customer_flag", aggfunc="max")
    f["months_active_3m"] = activity.sum(axis=1).reindex(f.index)
    f["months_observed_3m"] = activity.notna().sum(axis=1).reindex(f.index)
    f["tenure_months"] = now["tenure_months"]
    f["household_income"] = now["household_income"]

    f["has_current_account"] = now[CURRENT_ACCOUNTS].max(axis=1)
    f["has_salary"] = now[["payroll_account", "salary_credited"]].max(axis=1)
    f["has_pension_credit"] = now["pension_credited"]
    f["has_term_deposit"] = now[TERM_DEPOSITS].max(axis=1)
    f["has_investments"] = now[INVESTMENTS].max(axis=1)
    f["has_credit"] = now[CREDIT].max(axis=1)
    f["has_card"] = now["credit_card"]
    f["has_e_account"] = now["e_account"]
    f["has_direct_debit"] = now["direct_debit"]
    f["has_junior_account"] = now["junior_account"]
    f["income_was_missing"] = now["income_was_missing"]
    f["new_customer_flag"] = now["new_customer_flag"]
    f["is_primary_customer"] = (now["primary_customer_flag"] == 1).astype(int)

    f["joining_channel"] = now["joining_channel"]
    f["bank_segment"] = now["bank_segment"]
    f["relation_type"] = now["relation_type_start_month"]
    f["active_at_cutoff"] = now["active_customer_flag"]

    for col in f.columns:
        if col in ("joining_channel", "bank_segment", "relation_type"):
            f[col] = f[col].astype(str)
        else:
            f[col] = pd.to_numeric(f[col]).astype(float)
    return f


def build_target(monthly: pd.DataFrame, rule: str = DEFAULT_TARGET_RULE) -> pd.Series:
    """drift_3m from the cutoff month to the outcome month (after the feature window)."""
    before, after = _snapshot(monthly, CUTOFF), _snapshot(monthly, OUTCOME_MONTH)
    keys = before.index.intersection(after.index)
    before, after = before.loc[keys], after.loc[keys]
    added = (after[PRODUCT_COLUMNS] - before[PRODUCT_COLUMNS]).clip(lower=0).sum(axis=1)
    dropped = (before[PRODUCT_COLUMNS] - after[PRODUCT_COLUMNS]).clip(lower=0).sum(axis=1)
    lost_activity = (before["active_customer_flag"] == 1) & (after["active_customer_flag"] == 0)
    only_dropping = (dropped > 0) & (added == 0)
    if rule == "inactive":
        return lost_activity.astype(int).rename(TARGET)
    if rule != "inactive_or_dropping":
        raise ValueError(f"Unknown target rule: {rule}")
    return (lost_activity | only_dropping).astype(int).rename(TARGET)


def build_modelling_table(monthly: pd.DataFrame, audit_table: pd.DataFrame,
                          rule: str = DEFAULT_TARGET_RULE):
    """Return (X, y, sensitive, report). Sensitive attributes are kept apart from X."""
    features = build_features(monthly)
    target = build_target(monthly, rule)
    population = features[features["active_at_cutoff"] == 1]
    keys = population.index.intersection(target.index)

    X = population.loc[keys, FEATURES].copy()
    y = target.loc[keys].copy()
    attrs = _snapshot(audit_table, CUTOFF).reindex(keys)[list(SENSITIVE)].astype(str)
    attrs = attrs.replace({"nan": "UNKNOWN", "<NA>": "UNKNOWN"})

    report = {"target_rule": rule, "customers_active_at_cutoff": int(len(population)),
              "left_before_outcome_month": int(len(population) - len(keys)),
              "modelling_rows": int(len(keys)), "positives": int(y.sum()),
              "drift_rate": round(float(y.mean()), 4),
              "feature_window": f"{FEATURE_MONTHS[0]} to {CUTOFF}",
              "outcome_month": OUTCOME_MONTH, "n_features": len(FEATURES)}
    audit("transform", "modelling_table", rows_in=len(monthly), rows_out=len(X),
          columns=X.columns, details=report)
    return X, y, attrs, report


def split(X, y, s):
    """Stratified 80/20 hold-out, identical in training and in the SHAP notebook."""
    from sklearn.model_selection import train_test_split

    from src.model.config import RANDOM_STATE, TEST_SIZE
    parts = train_test_split(X, y, s, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    return [p.reset_index(drop=True) for p in parts]

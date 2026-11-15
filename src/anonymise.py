"""Stage 3 - Anonymisation: remove PII, pseudonymise IDs, split off protected attributes."""
import hashlib
import hmac
import os

import pandas as pd

from src.audit import audit
from src.config import AUDIT_ONLY_COLUMNS, K_TARGET, PII_COLUMNS, PROVINCE_MIN

AGE_BANDS = [0, 18, 25, 35, 45, 55, 65, 200]
AGE_LABELS = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]


def _salt() -> bytes:
    salt = os.environ.get("ANON_SALT")
    if not salt:
        raise RuntimeError("Set the ANON_SALT environment variable before running "
                           "(a long random secret, never committed to GitHub).")
    return salt.encode()


def pseudonymise_id(value, salt: bytes) -> str:
    """Keyed hash (HMAC-SHA256): stable across months, irreversible without the secret."""
    return hmac.new(salt, str(value).encode(), hashlib.sha256).hexdigest()[:16]


def remove_pii(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    found = [c for c in df.columns if c.lower() in PII_COLUMNS]
    return df.drop(columns=found), found


QUASI_IDENTIFIERS = ["age_band", "sex", "province_name"]


def k_anonymity(table: pd.DataFrame) -> int:
    """Smallest group size on the quasi-identifiers in the latest month (higher = safer)."""
    latest = table[table["snapshot_date"] == table["snapshot_date"].max()]
    return int(latest.groupby(QUASI_IDENTIFIERS, observed=True).size().min())


def generalise(table: pd.DataFrame) -> pd.DataFrame:
    """Generalisation plus local suppression until every group has at least K_TARGET people."""
    table["age_band"] = table["age_band"].astype(str)
    latest = table[table["snapshot_date"] == table["snapshot_date"].max()]
    sizes = latest["province_name"].value_counts()
    small = set(sizes[sizes < PROVINCE_MIN].index)
    table.loc[table["province_name"].isin(small), "province_name"] = "OTHER"

    latest = table[table["snapshot_date"] == table["snapshot_date"].max()]
    group_size = latest.groupby(QUASI_IDENTIFIERS, observed=True)["customer_key"].transform("size")
    at_risk = set(latest.loc[group_size < K_TARGET, "customer_key"])
    table.loc[table["customer_key"].isin(at_risk), ["province_name", "age_band"]] = "SUPPRESSED"
    return table


def anonymise(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (analytics table, restricted audit table, report)."""
    salt = _salt()
    df, pii_removed = remove_pii(df)
    df = df.copy()
    df["customer_key"] = df["customer_id"].map(lambda v: pseudonymise_id(v, salt))
    df["age_band"] = pd.cut(df["age"], bins=AGE_BANDS, labels=AGE_LABELS, right=False)
    df["join_year"] = df["join_date"].dt.year.astype("Int64")

    audit_cols = ["customer_key", "snapshot_date", "age_band", "sex", "foreign_born_flag",
                  "country_of_residence", "province_name"]
    audit_table = generalise(df[audit_cols].copy())

    analytics = df.drop(columns=["customer_id", "join_date", "age_band"] +
                        [c for c in AUDIT_ONLY_COLUMNS if c in df.columns])

    k_before = k_anonymity(df.assign(age_band=df["age_band"].astype(str))[audit_cols])
    k_after = k_anonymity(audit_table)
    report = {"pii_columns_removed": pii_removed, "id_method": "HMAC-SHA256 (salted)",
              "audit_only_columns": [c for c in audit_cols if c not in ("customer_key", "snapshot_date")],
              "exact_age_removed_from_audit_file": True,
              "k_anonymity_before": k_before, "k_anonymity_after": k_after,
              "customers_suppressed": int(audit_table.loc[audit_table["province_name"] == "SUPPRESSED",
                                                          "customer_key"].nunique())}
    audit("transform", "anonymised_analytics", rows_in=len(df), rows_out=len(analytics),
          columns=analytics.columns, details=report)
    return analytics, audit_table, report

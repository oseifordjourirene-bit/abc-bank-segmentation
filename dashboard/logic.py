"""Data and model logic for the stakeholder dashboard (no Streamlit code here, so it can be tested).

The dashboard uses only the trained model files and the aggregated Module 4 report tables.
No customer-level data is loaded or published.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # the saved pipelines refer to src.model.*
    sys.path.insert(0, str(ROOT))

MODELS = {
    "main": {
        "label": "Main model: drift = goes inactive OR drops products",
        "short": "Main model",
        "model": ROOT / "models" / "drift_model.joblib",
        "meta": ROOT / "models" / "model_meta.json",
        "reports": ROOT / "reports" / "model",
        "suffix": "",
    },
    "inactive": {
        "label": "Comparison model: drift = goes inactive only",
        "short": "Comparison model",
        "model": ROOT / "models" / "drift_model_inactive.joblib",
        "meta": ROOT / "models" / "model_meta_inactive.json",
        "reports": ROOT / "reports" / "model_inactive",
        "suffix": "_inactive",
    },
}

DI_BAND = (0.8, 1.25)
EO_GAP_MAX = 0.10
MIN_GROUP = 500
RECALL_TARGET, AUC_TARGET = 0.70, 0.75

PLAIN_NAMES = {
    "logistic_regression": "Logistic regression", "random_forest": "Random forest", "lightgbm": "LightGBM",
    "baseline_majority": "Guess 'nobody drifts'", "baseline_prior": "Random guess",
    "baseline_business_rule": "Simple branch rule",
}

# Plain-language meaning of the main drivers, per model (from the Module 4 SHAP and LIME results)
DRIVER_NOTES = {
    "main": {
        "Has direct debit": "Customers with a direct debit score higher, because they hold more products that could be dropped.",
        "Products added (Dec-Feb)": "Customers who recently took a new product score much higher: new products are often dropped again.",
        "Tenure (months)": "Newer customers score higher; long-standing customers score lower.",
        "Has credit card": "Card holders score higher, again because there is a product to drop.",
        "Products held": "More products held means a higher score under this definition of drift.",
        "Months active (Dec-Feb)": "Customers who skipped a month recently score higher.",
    },
    "inactive": {
        "Products held": "Customers with few products are far more likely to go quiet.",
        "Tenure (months)": "Newer customers are more likely to go quiet than long-standing ones.",
        "Months active (Dec-Feb)": "Customers who skipped a month recently are much more likely to go quiet.",
        "Has direct debit": "A direct debit keeps the account in use and lowers the score.",
        "Months observed (Dec-Feb)": "Customers seen for fewer months (newer records) score differently.",
        "Products dropped (Dec-Feb)": "Recent product drops change the score.",
    },
}

OFFERS = {"Open an e-account (digital banking)": "has_e_account",
          "Set up a direct debit": "has_direct_debit",
          "Take up a card": "has_card",
          "Move salary to ABC Bank": "has_salary"}

NUMERIC = ["product_count", "products_added_2m", "products_dropped_2m", "months_active_3m",
           "months_observed_3m", "tenure_months", "household_income"]
CATEGORICAL = ["joining_channel", "bank_segment", "relation_type"]


def load_meta(key: str) -> dict:
    return json.loads(MODELS[key]["meta"].read_text(encoding="utf-8"))


def load_model(key: str):
    return joblib.load(MODELS[key]["model"])


def load_table(key: str, name: str) -> pd.DataFrame:
    return pd.read_csv(MODELS[key]["reports"] / f"{name}.csv")


def load_personas() -> dict:
    return json.loads((ROOT / "dashboard" / "personas.json").read_text(encoding="utf-8"))


def to_frame(profile: dict, features: list) -> pd.DataFrame:
    row = pd.DataFrame([profile])[features].copy()
    for col in row.columns:
        row[col] = row[col].astype(str) if col in CATEGORICAL else row[col].astype(float)
    return row


def score(model, profile: dict, features: list) -> float:
    return float(model.predict_proba(to_frame(profile, features))[:, 1][0])


def _add_product(p: dict, col: str) -> dict:
    q = dict(p)
    q[col] = 1
    q["product_count"] = q["product_count"] + 1
    q["products_added_2m"] = q["products_added_2m"] + 1
    return q


def what_if_changes(profile: dict) -> dict:
    """Realistic single changes a relationship manager could discuss. Age, sex and nationality are never used."""
    changes = {}
    for offer, col in OFFERS.items():
        if int(profile.get(col, 0)) == 0:
            changes[offer] = _add_product(profile, col)
    if profile["months_active_3m"] < 3:
        changes["Active one more month"] = {**profile, "months_active_3m": profile["months_active_3m"] + 1}
    if profile["months_active_3m"] > 1:
        changes["Inactive one more month"] = {**profile, "months_active_3m": profile["months_active_3m"] - 1}
    if profile["products_dropped_2m"] == 0 and profile["product_count"] > 1:
        changes["Drops one product"] = {**profile, "product_count": profile["product_count"] - 1,
                                        "products_dropped_2m": 1}
    changes["One more year with the bank"] = {**profile, "tenure_months": profile["tenure_months"] + 12}
    return changes


def what_if_table(model, profile: dict, features: list, threshold: float) -> pd.DataFrame:
    base = score(model, profile, features)
    rows = []
    for name, changed in what_if_changes(profile).items():
        new = score(model, changed, features)
        rows.append({"change": name, "new_score": round(new, 3), "change_in_score": round(new - base, 3),
                     "crosses_threshold": (base >= threshold) != (new >= threshold)})
    return pd.DataFrame(rows).sort_values("change_in_score")


def plain_explanation(model, profile: dict, features: list, threshold: float) -> str:
    """One or two sentences a non-technical reader can use."""
    s = score(model, profile, features)
    status = "would be flagged for the reactivation offer" if s >= threshold else "would not be flagged"
    t = what_if_table(model, profile, features, threshold)
    down, up = t.iloc[0], t.iloc[-1]
    parts = [f"Score {s:.2f} against a threshold of {threshold:.2f}: this customer {status}."]
    if down["change_in_score"] < -0.01:
        parts.append(f"Biggest fall in the score: {down['change']} ({down['change_in_score']:+.2f}).")
    if up["change_in_score"] > 0.01:
        parts.append(f"Biggest rise in the score: {up['change']} ({up['change_in_score']:+.2f}).")
    return " ".join(parts)


def fairness_summary(groups: pd.DataFrame) -> pd.DataFrame:
    """Same rules as Module 4: judge only groups of 500+ customers."""
    rows = []
    for attr, sub in groups.groupby("attribute", sort=False):
        ok = sub[sub["customers"] >= MIN_GROUP]
        sel = ok["selection_rate"]
        di_ok = bool(ok["disparate_impact"].between(*DI_BAND).all())
        if len(ok) > 1:
            eo = float(max(ok["recall_tpr"].max() - ok["recall_tpr"].min(),
                           ok["false_positive_rate"].max() - ok["false_positive_rate"].min()))
            dp = float(sel.min() / sel.max()) if sel.max() > 0 else 1.0
        else:
            eo, dp = 0.0, 1.0
        rows.append({"Attribute": attr.replace(" (H/V as coded in source)", ""),
                     "Result": "Passes" if di_ok and eo <= EO_GAP_MAX else "Fails",
                     "Groups judged": len(ok),
                     "Offer rates in band (0.8-1.25)": "Yes" if di_ok else "No",
                     "Equal-treatment gap": round(eo, 4),
                     "Gap ≤ 0.10": "Yes" if eo <= EO_GAP_MAX else "No",
                     "Lowest ÷ highest offer rate": round(dp, 2)})
    return pd.DataFrame(rows)


def headline(meta: dict) -> dict:
    m, d = meta["test_metrics"], meta["data"]
    return {"customers": d["modelling_rows"], "drift_rate": d["drift_rate"], "recall": m["recall"],
            "precision": m["precision"], "auc": m["roc_auc"], "top20": m["top20_capture"],
            "flagged": m["flagged_rate"], "accepted": meta["acceptance_passed"],
            "threshold": meta["threshold"], "champion": PLAIN_NAMES.get(meta["champion"], meta["champion"]),
            "definition": d["target_definition"]}


# ---------- Final project: local explanations, precision review, monitoring ----------
MONITORING_DIR = ROOT / "reports" / "monitoring"
FEATURE_NAMES = {
    "product_count": "Products held", "products_added_2m": "Products added (last 2 months)",
    "products_dropped_2m": "Products dropped (last 2 months)", "months_active_3m": "Months active (of 3)",
    "months_observed_3m": "Months on record (of 3)", "tenure_months": "Months with the bank",
    "household_income": "Household income", "has_current_account": "Has current account",
    "has_salary": "Salary paid in", "has_pension_credit": "Pension paid in", "has_term_deposit": "Has term deposit",
    "has_investments": "Has investments", "has_credit": "Has a loan or mortgage", "has_card": "Has a card",
    "has_e_account": "Has e-account", "has_direct_debit": "Has direct debit",
    "has_junior_account": "Has junior account", "income_was_missing": "Income not recorded",
    "new_customer_flag": "New customer", "is_primary_customer": "Primary customer",
    "joining_channel": "Joining channel", "bank_segment": "Bank segment", "relation_type": "Relationship type",
}


def load_reference(key: str) -> dict:
    from src.model.explain_local import load_reference_profile
    return load_reference_profile(ROOT / "models" / f"reference_profile{MODELS[key]['suffix']}.json")


def local_explanation(model, profile: dict, features: list, reference: dict, top: int = 5) -> pd.DataFrame:
    """Top drivers of one customer's score (same method the API stores for audit)."""
    from src.model.explain_local import local_contributions
    rows = local_contributions(model, profile, features, reference, top)
    df = pd.DataFrame(rows, columns=["feature", "value", "typical_value", "contribution"])
    df["factor"] = df["feature"].map(FEATURE_NAMES).fillna(df["feature"])
    return df


def load_json(key: str, name: str) -> dict:
    return json.loads((MODELS[key]["reports"] / f"{name}.json").read_text(encoding="utf-8"))


def break_even_precision(cost_per_contact: float, value_if_retained: float, save_rate: float) -> float:
    """Contacting pays when precision x save_rate x value_if_retained > cost_per_contact."""
    if value_if_retained <= 0 or save_rate <= 0:
        return float("inf")
    return cost_per_contact / (value_if_retained * save_rate)


def load_monitoring(name: str) -> pd.DataFrame:
    return pd.read_csv(MONITORING_DIR / f"{name}.csv")


def monitoring_status(alerts: pd.DataFrame, month: str) -> dict:
    """Traffic light per check for one month: red if any alert of that type."""
    month_alerts = alerts[alerts["month"] == month]
    return {t: ("alert" if (month_alerts["type"] == t).any() else "ok")
            for t in ["data drift", "prediction drift", "fairness"]}

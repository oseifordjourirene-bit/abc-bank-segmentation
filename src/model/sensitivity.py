"""Sensitivity analysis: how much do predictions move when inputs change a little?"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.model.config import RANDOM_STATE  # noqa: E402


def _scenarios(X: pd.DataFrame) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    s = {
        "Income -20%": X.assign(household_income=X["household_income"] * 0.8),
        "Income +20%": X.assign(household_income=X["household_income"] * 1.2),
        "Tenure -12 months": X.assign(tenure_months=(X["tenure_months"] - 12).clip(lower=0)),
        "Tenure +12 months": X.assign(tenure_months=X["tenure_months"] + 12),
        "Income reported missing": X.assign(household_income=X["household_income"].median(),
                                            income_was_missing=1.0),
        "Random noise on numbers (5%)": X.assign(**{
            c: X[c] * (1 + rng.normal(0, 0.05, len(X)))
            for c in ["household_income", "tenure_months"]}),
        "Inactive one more month (large change)": X.assign(months_active_3m=(X["months_active_3m"] - 1).clip(lower=1)),
    }
    return s


def input_sensitivity(pipeline, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    base = pipeline.predict_proba(X)[:, 1]
    base_flag = base >= threshold
    rows = []
    for name, Xs in _scenarios(X).items():
        p = pipeline.predict_proba(Xs)[:, 1]
        rows.append({"scenario": name,
                     "mean_change_in_risk": round(float(np.mean(p - base)), 4),
                     "mean_abs_change": round(float(np.mean(np.abs(p - base))), 4),
                     "decisions_flipped_pct": round(100 * float(np.mean((p >= threshold) != base_flag)), 2)})
    return pd.DataFrame(rows)


def seed_stability(pipeline, X_train, y_train, X_test, y_test, seeds=(1, 2, 3, 4, 5)) -> pd.DataFrame:
    """Retrain with different random seeds: a robust model gives nearly the same AUC."""
    rows = []
    for seed in seeds:
        model = clone(pipeline)
        if "random_state" in model.named_steps["model"].get_params():
            model.set_params(model__random_state=seed)
        model.fit(X_train, y_train)
        rows.append({"seed": seed, "test_roc_auc": round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)})
    return pd.DataFrame(rows)


def sensitivity_figure(table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    t = table.iloc[::-1]
    ax.barh(t["scenario"], t["decisions_flipped_pct"], color="#1F3864")
    for y, v in enumerate(t["decisions_flipped_pct"]):
        ax.text(v + 0.1, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Customers whose offer decision changes (%)")
    ax.set_title("Sensitivity: effect of small input changes on decisions", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

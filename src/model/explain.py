"""Explainability: SHAP (global and local), LIME (local) and actionable counterfactuals."""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model.config import RANDOM_STATE  # noqa: E402

FRIENDLY = {
    "months_active_3m": "Months active (Dec-Feb)", "product_count": "Products held",
    "products_added_2m": "Products added (Dec-Feb)", "products_dropped_2m": "Products dropped (Dec-Feb)",
    "tenure_months": "Tenure (months)", "household_income": "Household income",
    "has_e_account": "Has e-account", "has_direct_debit": "Has direct debit",
    "has_card": "Has credit card", "has_salary": "Salary paid in", "has_current_account": "Has current account",
    "has_term_deposit": "Has term deposit", "has_investments": "Has investments", "has_credit": "Has credit product",
    "has_pension_credit": "Pension paid in", "has_junior_account": "Has junior account",
    "income_was_missing": "Income was missing", "new_customer_flag": "New customer",
    "is_primary_customer": "Primary customer", "months_observed_3m": "Months observed (Dec-Feb)",
}

# Offers the bank can actually make, and the features each one changes
ACTIONS = {"Open an e-account (digital banking)": "has_e_account",
           "Set up a direct debit": "has_direct_debit",
           "Take up a card": "has_card",
           "Move salary to ABC Bank": "has_salary"}


def friendly(name: str) -> str:
    if name in FRIENDLY:
        return FRIENDLY[name]
    for prefix, label in [("joining_channel_", "Channel "), ("bank_segment_", "Segment "),
                          ("relation_type_", "Relation type ")]:
        if name.startswith(prefix):
            return label + name[len(prefix):].replace("infrequent_sklearn", "(other)")
    return name


def transformed(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    Xt = pipeline.named_steps["prep"].transform(X)
    return Xt.rename(columns=friendly)


def shap_explanation(pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame):
    """SHAP values for the positive class (drift), with readable feature names."""
    import shap
    model = pipeline.named_steps["model"]
    Xb, Xe = transformed(pipeline, X_background), transformed(pipeline, X_explain)
    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, Xb)
    else:
        explainer = shap.TreeExplainer(model)
    exp = explainer(Xe)
    if exp.values.ndim == 3:            # some tree models return one set per class
        exp = exp[:, :, 1]
    return exp


def shap_global(exp, out_dir: Path) -> pd.DataFrame:
    import shap
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    shap.plots.beeswarm(exp, max_display=15, show=False)
    plt.title("SHAP summary: what pushes drift risk up (right) or down (left)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    plt.figure()
    shap.plots.bar(exp, max_display=15, show=False)
    plt.title("Global feature importance (mean |SHAP|)", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=180, bbox_inches="tight")
    plt.close("all")
    importance = pd.DataFrame({"feature": exp.feature_names,
                               "mean_abs_shap": np.abs(exp.values).mean(axis=0),
                               "mean_shap": exp.values.mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False).round(4)
    importance.to_csv(out_dir / "shap_importance.csv", index=False)
    return importance


def pick_examples(y_true, scores, threshold) -> dict:
    """Three customers to explain: a caught drifter, a borderline case and a low-risk customer."""
    s, y = pd.Series(np.asarray(scores)), pd.Series(np.asarray(y_true))
    caught = s[(y == 1) & (s >= threshold)]
    return {"High risk (correctly flagged)": int(caught.idxmax()) if len(caught) else int(s.idxmax()),
            "Borderline": int((s - threshold).abs().idxmin()),
            "Low risk": int(s.idxmin())}


def shap_local(exp, examples: dict, out_dir: Path) -> pd.DataFrame:
    import shap
    rows = []
    for label, i in examples.items():
        plt.figure()
        shap.plots.waterfall(exp[i], max_display=10, show=False)
        plt.title(f"SHAP local explanation: {label}", fontsize=10)
        plt.tight_layout()
        slug = label.split(" ")[0].lower()
        plt.savefig(out_dir / f"shap_local_{slug}.png", dpi=180, bbox_inches="tight")
        plt.close("all")
        top = np.argsort(-np.abs(exp.values[i]))[:5]
        for rank, j in enumerate(top, 1):
            rows.append({"case": label, "rank": rank, "feature": exp.feature_names[j],
                         "value": round(float(exp.data[i][j]), 3),
                         "shap": round(float(exp.values[i][j]), 4),
                         "effect": "raises risk" if exp.values[i][j] > 0 else "lowers risk"})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "shap_local_explanations.csv", index=False)
    return table


def lime_local(pipeline, X_train, X_explain, examples: dict, out_dir: Path) -> pd.DataFrame:
    """LIME explanations for the same customers, to check that SHAP and LIME agree."""
    from lime.lime_tabular import LimeTabularExplainer
    model = pipeline.named_steps["model"]
    Xtr, Xe = transformed(pipeline, X_train), transformed(pipeline, X_explain)
    cols = list(Xtr.columns)

    def predict_fn(a):
        return model.predict_proba(pd.DataFrame(a, columns=pipeline.named_steps["prep"].get_feature_names_out()))

    explainer = LimeTabularExplainer(Xtr.values, feature_names=cols, class_names=["Stays active", "Drifts"],
                                     mode="classification", discretize_continuous=True,
                                     random_state=RANDOM_STATE)
    rows = []
    for label, i in examples.items():
        exp = explainer.explain_instance(Xe.values[i], predict_fn, num_features=8)
        fig = exp.as_pyplot_figure()
        fig.set_size_inches(6.5, 3.8)
        plt.title(f"LIME local explanation: {label}", fontsize=10)
        plt.tight_layout()
        fig.savefig(out_dir / f"lime_local_{label.split(' ')[0].lower()}.png", dpi=180, bbox_inches="tight")
        plt.close("all")
        for rank, (rule, weight) in enumerate(exp.as_list(), 1):
            rows.append({"case": label, "rank": rank, "rule": rule, "weight": round(float(weight), 4),
                         "effect": "raises risk" if weight > 0 else "lowers risk"})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "lime_local_explanations.csv", index=False)
    return table


def counterfactuals(pipeline, X: pd.DataFrame, threshold: float, n_customers: int = 10) -> pd.DataFrame:
    """Smallest set of realistic offers that would take a flagged customer below the threshold.
    Only actionable features change; age, sex, income and tenure are never altered.
    Each new product also raises product_count and products_added_2m, so the profile stays consistent."""
    scores = pipeline.predict_proba(X)[:, 1]
    # Flagged customers closest to the threshold: the group an offer could realistically tip
    above = np.where(scores >= threshold)[0]
    flagged = above[np.argsort(scores[above])][: n_customers * 4]
    rows = []
    for i in flagged:
        base = X.iloc[[i]].copy()
        options = [a for a, col in ACTIONS.items() if base[col].iloc[0] == 0]
        found = None
        for size in range(1, min(3, len(options)) + 1):
            best = None
            for combo in combinations(options, size):
                cf = base.copy()
                for action in combo:
                    cf[ACTIONS[action]] = 1.0
                cf["product_count"] += size
                cf["products_added_2m"] += size
                new = float(pipeline.predict_proba(cf)[:, 1][0])
                if new < threshold and (best is None or new < best[1]):
                    best = (combo, new)
            if best:
                found = best
                break
        rows.append({"row": int(i), "risk_before": round(float(scores[i]), 3),
                     "change_needed": " + ".join(found[0]) if found else "No realistic offer set found",
                     "risk_after": round(found[1], 3) if found else None,
                     "offers_needed": len(found[0]) if found else None})
        if sum(r["risk_after"] is not None for r in rows) >= n_customers:
            break
    return pd.DataFrame(rows)


def offer_effects(pipeline, X: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Average change in predicted risk if each flagged customer took up one offer.
    Positive = the model expects MORE drift, which can happen when the target counts product drops."""
    scores = pipeline.predict_proba(X)[:, 1]
    flagged = X[scores >= threshold]
    rows = []
    for action, col in ACTIONS.items():
        eligible = flagged[flagged[col] == 0]
        if eligible.empty:
            continue
        cf = eligible.copy()
        cf[col] = 1.0
        cf["product_count"] += 1
        cf["products_added_2m"] += 1
        before, after = pipeline.predict_proba(eligible)[:, 1], pipeline.predict_proba(cf)[:, 1]
        rows.append({"offer": action, "flagged_customers_eligible": len(eligible),
                     "mean_change_in_risk": round(float(np.mean(after - before)), 4),
                     "share_moved_below_threshold": round(float(np.mean(after < threshold)), 4)})
    return pd.DataFrame(rows)

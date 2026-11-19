"""Write the Fairness Metrics Report and the Model Card from the numbers of the actual run."""
from datetime import date

import pandas as pd

from src.model.config import (AUC_TARGET, CUTOFF, DI_BAND, EO_GAP_MAX, FEATURE_MONTHS, MIN_GROUP_SIZE,
                              OUTCOME_MONTH, RECALL_TARGET)


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))
                                        for v in r.values) + " |")
    return "\n".join(lines)


def fairness_report(data_bias, before, before_summary, fairlearn_check, after, after_summary,
                    mitigated_attr, mitigated_metrics, champion_metrics) -> str:
    keep = ["attribute", "group", "customers", "base_drift_rate", "selection_rate", "recall_tpr",
            "false_positive_rate", "disparate_impact", "status"]
    out = [
        "# Fairness Metrics Report – ABC Bank drift model",
        "",
        f"Generated {date.today().isoformat()} from the hold-out test set. 'Selected' = flagged for the "
        "reactivation offer (a benefit). Protected attributes are used only for this audit, never as model inputs.",
        "",
        f"Rules (Modules 1–2): disparate impact (group selection rate ÷ overall) within {DI_BAND[0]}–{DI_BAND[1]}; "
        f"equalized odds difference ≤ {EO_GAP_MAX}. Groups under {MIN_GROUP_SIZE} customers are reported but not judged.",
        "",
        "## 1. Data bias in the modelling table",
        "",
        _md(data_bias),
        "",
        "## 2. Champion model, by group",
        "",
        _md(before[keep]),
        "",
        "## 3. Summary per attribute (groups large enough to judge)",
        "",
        _md(before_summary.drop(columns=["attr_key"])),
        "",
    ]
    if fairlearn_check is not None and len(fairlearn_check):
        out += ["## 4. Fairlearn cross-check (all groups, including small ones)", "", _md(fairlearn_check), ""]
    if after is not None:
        out += [f"## 5. Mitigation: ExponentiatedGradient, equalized odds on {mitigated_attr}", "",
                _md(after[keep]), "", _md(after_summary.drop(columns=["attr_key"])), "",
                "Performance trade-off (test set):", "",
                _md(pd.DataFrame([{"model": "champion", **champion_metrics},
                                  {"model": "mitigated", **mitigated_metrics}])[
                    ["model", "recall", "precision", "f1", "roc_auc", "flagged_rate"]]), ""]
    else:
        out += ["## 5. Mitigation", "", "All judged groups were within the rules, so no mitigation was applied.", ""]
    return "\n".join(out)


def model_card(meta: dict, comparison: pd.DataFrame, summary_tbl: pd.DataFrame, top_features: pd.DataFrame,
               sensitivity: pd.DataFrame, seeds: pd.DataFrame) -> str:
    m = meta["test_metrics"]
    passed = "meets" if meta["acceptance_passed"] else "does NOT meet"
    return "\n".join([
        f"# Model Card – {meta['registered_model']} (version {meta.get('model_version', 'n/a')})",
        "",
        "## Model details",
        f"- **Owner:** Irene Osei Fordjour, ABC Bank Segmentation project (BAN6800)",
        f"- **Algorithm:** {meta['champion']} (scikit-learn pipeline: preprocessing + classifier)",
        f"- **Trained:** {meta['trained_at']} · MLflow run `{meta.get('mlflow_run_id', 'n/a')}`",
        f"- **Decision threshold:** {meta['threshold']:.4f} (chosen on cross-validated training predictions "
        f"to reach recall ≥ {RECALL_TARGET})",
        "",
        "## Intended use",
        "- Rank active retail customers by their risk of drifting in the next three months, so the bank can "
        "send a reactivation offer early.",
        "- Users: retail marketing and relationship managers. The score supports, not replaces, human decisions.",
        "- **Out of scope:** credit decisions, pricing, account closure, or any adverse action against a customer.",
        "",
        "## Data",
        f"- Santander Product Recommendation (Kaggle, 2016) as a proxy for ABC Bank; Module 3 clean, anonymised tables.",
        f"- Unit: one customer active in {CUTOFF[:7]}. Features from {FEATURE_MONTHS[0][:7]} to {CUTOFF[:7]}; "
        f"outcome measured at {OUTCOME_MONTH[:7]} (3-month horizon).",
        f"- Target (`drift_3m`): {meta['data'].get('target_definition', '')}.",
        f"- {meta['data']['modelling_rows']:,} customers, drift rate {meta['data']['drift_rate']:.1%}. "
        "Stratified 80/20 hold-out; 5-fold stratified cross-validation for tuning.",
        "- Age, sex, nationality and province are excluded from the inputs.",
        "",
        "## Performance (hold-out test set)",
        f"Acceptance criteria: recall ≥ {RECALL_TARGET} and ROC-AUC ≥ {AUC_TARGET}. The champion **{passed}** them.",
        "",
        _md(comparison[["model", "cv_roc_auc", "roc_auc", "recall", "precision", "f1", "accuracy", "top20_capture"]]),
        "",
        "## Fairness",
        _md(summary_tbl.drop(columns=["attr_key"])),
        "",
        "Full detail: `reports/model/fairness_report.md`.",
        "",
        "## What the model learned (top SHAP features)",
        _md(top_features.head(8)),
        "",
        "## Robustness",
        _md(sensitivity),
        "",
        f"Seed stability: test ROC-AUC {seeds['test_roc_auc'].mean():.3f} ± {seeds['test_roc_auc'].std():.3f} over {len(seeds)} seeds.",
        "",
        "## Limitations and risks",
        "- Proxy data from a Spanish bank (2015–16); patterns must be revalidated on ABC Bank data before use in Ghana.",
        "- One cohort and one 3-month window: seasonality and economic shocks are not captured.",
        "- 'Activity' is the source bank's own flag; ABC Bank's definition (BoG dormancy rules) may differ.",
        "- Counterfactual suggestions show associations, not proven causes; test offers with a control group.",
        "",
        "## Monitoring after deployment",
        f"- Monthly: recall and precision on customers whose outcome is known; alert if recall < {RECALL_TARGET}.",
        "- Monthly: input drift (population stability index) on the top features; alert if PSI > 0.2.",
        f"- Quarterly: fairness table by age band, sex and foreign-born; alert if any judged group leaves {DI_BAND[0]}–{DI_BAND[1]}.",
        "- Retrain at least every 6 months or when an alert fires; new versions go through the MLflow registry.",
    ])

"""Precision review (response to Module 4 and Module 5 feedback).

Precision is low at the agreed threshold because only 8.6% of customers drift and the
threshold was set to catch 70% of them. This script measures, on the untouched test set:

1. precision, recall and lift at capacity-based cut-offs (riskiest 5%, 10%, ... of customers);
2. a two-tier contact policy: high-cost contact for the riskiest tier, low-cost digital
   nudges for the rest of the flagged group;
3. the precision-recall curve and PR-AUC against the base rate.

Run from the repository root (after the Module 3 pipeline and Module 4 training):
    python -m src.model.precision_review
Outputs go to reports/model/ and reports/model_inactive/ (aggregates only, no customer rows),
plus models/targeting_policy*.json and models/reference_profile*.json used by the API and dashboard.
"""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import REPORTS_DIR
from src.model.config import MODEL_DIR
from src.model.dataset import build_modelling_table, load_inputs, split
from src.model.explain_local import build_reference_profile, save_reference_profile

RULES = {"inactive_or_dropping": "", "inactive": "_inactive"}
CAPACITY_SHARES = [0.05, 0.10, 0.15, 0.20, 0.30]
TIER1_SHARE = 0.10


def _row(label, y, flagged):
    y = np.asarray(y)
    n, tp, pos = int(flagged.sum()), int(y[flagged].sum()), int(y.sum())
    prec = tp / n if n else 0.0
    return {"operating_point": label, "share_flagged": round(n / len(y), 4), "customers_flagged": n,
            "drifters_caught": tp, "recall": round(tp / pos, 4), "precision": round(prec, 4),
            "lift_vs_random": round(prec / y.mean(), 2) if n else 0.0,
            "offers_per_drifter": round(n / tp, 2) if tp else None}


def top_share_mask(p, share):
    """Flag the riskiest `share` of customers (ties broken by order, as a campaign list would be)."""
    n = int(round(share * len(p)))
    mask = np.zeros(len(p), dtype=bool)
    mask[np.argsort(-np.asarray(p), kind="stable")[:n]] = True
    return mask


def operating_points(y, p, threshold, shares=CAPACITY_SHARES) -> pd.DataFrame:
    rows = [_row(f"Riskiest {s:.0%}", y, top_share_mask(p, s)) for s in shares]
    rows.append(_row(f"Agreed threshold ({threshold:.3f})", y, np.asarray(p) >= threshold))
    return pd.DataFrame(rows)


def tiered_policy(y, p, threshold, tier1_share=TIER1_SHARE) -> pd.DataFrame:
    """Tier 1 = riskiest tier1_share (personal contact); Tier 2 = rest of the flagged group (digital)."""
    p = np.asarray(p)
    tier1 = top_share_mask(p, tier1_share)
    flagged = p >= threshold
    tier2 = flagged & ~tier1
    df = pd.DataFrame([_row("Tier 1: relationship-manager call", y, tier1),
                       _row("Tier 2: low-cost digital nudge", y, tier2),
                       _row("Both tiers together", y, tier1 | flagged)])
    df["recall"] = df["drifters_caught"] / int(np.asarray(y).sum())
    df["recall"] = df["recall"].round(4)
    return df


def pr_curve(y, p, points=120) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y, p)
    df = pd.DataFrame({"recall": rec[:-1], "precision": prec[:-1], "threshold": thr})
    idx = np.unique(np.linspace(0, len(df) - 1, points).astype(int))
    return df.iloc[idx].round(4).reset_index(drop=True)


def break_even_precision(cost_per_contact: float, value_if_retained: float, save_rate: float) -> float:
    """Lowest precision at which contacting customers pays for itself.

    A contact costs `cost_per_contact`; a contacted drifter is kept with probability `save_rate`,
    worth `value_if_retained`. Contacting pays when precision x save_rate x value > cost.
    """
    return cost_per_contact / (value_if_retained * save_rate)


def review(monthly, audit_table, rule: str) -> dict:
    suffix = RULES[rule]
    X, y, s, _ = build_modelling_table(monthly, audit_table, rule)
    X_train, X_test, _, y_test, _, _ = split(X, y, s)
    model = joblib.load(MODEL_DIR / f"drift_model{suffix}.joblib")
    meta = json.loads((MODEL_DIR / f"model_meta{suffix}.json").read_text(encoding="utf-8"))
    p = model.predict_proba(X_test[meta["features"]])[:, 1]

    out = REPORTS_DIR / f"model{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    points = operating_points(y_test, p, meta["threshold"])
    tiers = tiered_policy(y_test, p, meta["threshold"])
    curve = pr_curve(y_test, p)
    summary = {"target_rule": rule, "test_customers": int(len(y_test)), "drifters": int(y_test.sum()),
               "base_rate": round(float(y_test.mean()), 4),
               "pr_auc": round(float(average_precision_score(y_test, p)), 4),
               "pr_auc_lift": round(float(average_precision_score(y_test, p) / y_test.mean()), 2),
               "threshold": round(float(meta["threshold"]), 4), "tier1_share": TIER1_SHARE}
    points.to_csv(out / "precision_operating_points.csv", index=False)
    tiers.to_csv(out / "precision_tiers.csv", index=False)
    curve.to_csv(out / "pr_curve.csv", index=False)
    summary["tier1_score_cutoff"] = round(float(np.sort(p)[::-1][int(round(TIER1_SHARE * len(p))) - 1]), 4)
    (out / "precision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Typical-customer profile for local explanations (API explanation store and dashboard)
    save_reference_profile(build_reference_profile(X_train[meta["features"]]),
                           MODEL_DIR / f"reference_profile{suffix}.json")
    policy = {"threshold": summary["threshold"], "tier1_share": TIER1_SHARE,
              "tier1_score_cutoff": summary["tier1_score_cutoff"],
              "source": "test-set precision review (src/model/precision_review.py)"}
    (MODEL_DIR / f"targeting_policy{suffix}.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    _log_mlflow(rule, summary, out)
    return {"summary": summary, "points": points, "tiers": tiers}


def _log_mlflow(rule, summary, out):
    """Record the review as its own MLflow run, next to the training runs (skipped if MLflow is absent)."""
    try:
        import mlflow

        from src.model.config import EXPERIMENT, MLFLOW_URI
    except ImportError:
        return
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"precision_review_{rule}"):
        mlflow.set_tag("purpose", "precision review (final project)")
        mlflow.log_params({"target_rule": rule, "tier1_share": TIER1_SHARE})
        mlflow.log_metrics({k: v for k, v in summary.items() if isinstance(v, float)})
        for f in ["precision_operating_points.csv", "precision_tiers.csv", "pr_curve.csv"]:
            mlflow.log_artifact(str(out / f))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rule", choices=[*RULES, "both"], default="both")
    args = parser.parse_args()
    monthly, audit_table = load_inputs()
    for rule in (RULES if args.rule == "both" else [args.rule]):
        r = review(monthly, audit_table, rule)
        print(f"\n== {rule} ==  PR-AUC {r['summary']['pr_auc']} (base rate {r['summary']['base_rate']})")
        print(r["points"].to_string(index=False))
        print(r["tiers"].to_string(index=False))


if __name__ == "__main__":
    main()

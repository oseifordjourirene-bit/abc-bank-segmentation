"""Automated retraining with promotion gates (final project).

Trigger: the monthly flow (flows/retrain_flow.py) runs this when monitoring raises a
data-drift alert, or on the quarterly schedule. Steps:
1. Rebuild the modelling table from the latest pipeline outputs.
2. Refit the champion's exact configuration (same hyperparameters) on the new training split.
3. Re-derive the threshold from cross-validated training predictions (recall >= 0.70).
4. Score champion and candidate on the same untouched test split.
5. Apply the promotion gates below, log everything to MLflow and register the candidate
   with stage tag "Candidate" or "Rejected".

Retraining never replaces the live model by itself: moving the "champion" alias needs a
named human approver (model owner, with Digital Governance and Risk), as the Ethical AI
Charter requires.

Run: python -m src.model.retrain            (both target rules)
     python -m src.model.retrain --no-mlflow (dry run, e.g. in CI)
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import REPORTS_DIR
from src.model import evaluate as ev
from src.model.config import (AUC_TARGET, CV_FOLDS, EXPERIMENT, MLFLOW_URI, MODEL_DIR, RANDOM_STATE,
                              RECALL_TARGET, REGISTERED_MODEL)
from src.model.dataset import build_modelling_table, load_inputs, split

RULES = {"inactive_or_dropping": "", "inactive": "_inactive"}
MAX_AUC_DROP = 0.01            # candidate may not rank worse than the champion by more than this
OUT_DIR = REPORTS_DIR / "retraining"


def fairness_breaches(flagged: pd.Series, attrs: pd.DataFrame) -> list:
    from src.monitoring.monitor import fairness_rates
    fr = fairness_rates(flagged, attrs, "test")
    bad = fr[(fr["judged"]) & (fr["within_band"] == False)]  # noqa: E712
    return [f"{r.attribute} {r.group} {r.ratio_to_overall:.2f}x" for r in bad.itertuples()]


def decide(candidate: dict, champion: dict, breaches: list) -> dict:
    """Promotion gates. Every failed gate is listed so the approver sees why."""
    reasons = []
    if candidate["recall"] < RECALL_TARGET:
        reasons.append(f"recall {candidate['recall']:.3f} < {RECALL_TARGET}")
    if candidate["roc_auc"] < AUC_TARGET:
        reasons.append(f"ROC-AUC {candidate['roc_auc']:.3f} < {AUC_TARGET}")
    if candidate["roc_auc"] < champion["roc_auc"] - MAX_AUC_DROP:
        reasons.append(f"ROC-AUC {candidate['roc_auc']:.3f} worse than champion {champion['roc_auc']:.3f}")
    if breaches:
        reasons.append("fairness band breached: " + "; ".join(breaches))
    return {"stage": "Rejected" if reasons else "Candidate",
            "needs_human_approval": True, "failed_gates": reasons}


def retrain(monthly, audit_table, rule: str, use_mlflow: bool = True) -> dict:
    suffix = RULES[rule]
    X, y, s, data_report = build_modelling_table(monthly, audit_table, rule)
    X_tr, X_te, y_tr, y_te, _, s_te = split(X, y, s)
    champion = joblib.load(MODEL_DIR / f"drift_model{suffix}.joblib")
    meta = json.loads((MODEL_DIR / f"model_meta{suffix}.json").read_text(encoding="utf-8"))
    feats = meta["features"]

    candidate = clone(champion)
    oof = cross_val_predict(clone(champion), X_tr[feats], y_tr, method="predict_proba",
                            cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))[:, 1]
    threshold = ev.threshold_for_recall(y_tr, oof, RECALL_TARGET)
    candidate.fit(X_tr[feats], y_tr)

    cand_scores = candidate.predict_proba(X_te[feats])[:, 1]
    champ_scores = champion.predict_proba(X_te[feats])[:, 1]
    cand_m = {k: round(float(v), 4) for k, v in ev.classification_metrics(y_te, cand_scores, threshold).items()}
    champ_m = {k: round(float(v), 4)
               for k, v in ev.classification_metrics(y_te, champ_scores, meta["threshold"]).items()}
    s_te.index = X_te.index
    breaches = fairness_breaches(pd.Series(cand_scores >= threshold, index=X_te.index), s_te)
    decision = decide(cand_m, champ_m, breaches)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cand_dir = MODEL_DIR / "candidates"
    cand_dir.mkdir(exist_ok=True)
    cand_file = cand_dir / f"drift_model{suffix}_{stamp}.joblib"
    joblib.dump(candidate, cand_file)
    report = {"target_rule": rule, "run_at": stamp, "data": data_report, "threshold": round(threshold, 4),
              "candidate": cand_m, "champion": champ_m, "decision": decision,
              "candidate_file": cand_file.name}
    if use_mlflow:
        report["mlflow"] = _log_mlflow(candidate, X_tr[feats], report, suffix)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"retrain_report{suffix}.json").write_text(json.dumps(report, indent=2, default=str),
                                                         encoding="utf-8")
    return report


def _log_mlflow(model, X_tr, report, suffix) -> dict:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    registered = REGISTERED_MODEL + suffix.replace("_", "-")
    with mlflow.start_run(run_name=f"retrain_{report['target_rule']}_{report['run_at']}") as run:
        mlflow.set_tags({"role": "retraining candidate", "target_rule": report["target_rule"]})
        mlflow.log_params({"threshold": report["threshold"], "rows": report["data"]["modelling_rows"]})
        mlflow.log_metrics({f"candidate_{k}": v for k, v in report["candidate"].items()})
        mlflow.log_metrics({f"champion_{k}": v for k, v in report["champion"].items()})
        info = mlflow.sklearn.log_model(model, "model", input_example=X_tr.head(3),
                                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        run_id = run.info.run_id
    mv = mlflow.register_model(info.model_uri, registered)
    client = MlflowClient()
    client.set_model_version_tag(registered, mv.version, "stage", report["decision"]["stage"])
    client.set_model_version_tag(registered, mv.version, "failed_gates",
                                 "; ".join(report["decision"]["failed_gates"]) or "none")
    client.set_model_version_tag(registered, mv.version, "approval", "pending human sign-off")
    return {"run_id": run_id, "registered_model": registered, "version": str(mv.version)}


def main():
    parser = argparse.ArgumentParser(description="Retrain with promotion gates")
    parser.add_argument("--rule", choices=[*RULES, "both"], default="both")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    monthly, audit_table = load_inputs()
    for rule in (RULES if args.rule == "both" else [args.rule]):
        r = retrain(monthly, audit_table, rule, use_mlflow=not args.no_mlflow)
        print(f"{rule}: candidate AUC {r['candidate']['roc_auc']}, recall {r['candidate']['recall']} -> "
              f"{r['decision']['stage']} {r['decision']['failed_gates']}")


if __name__ == "__main__":
    main()

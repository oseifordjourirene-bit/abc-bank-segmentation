"""Module 4: train, validate, explain and audit the drift model, tracked in MLflow.

Run from the repository root, after the Module 3 pipeline:
    python -m src.model.run_all
Then:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (open http://127.0.0.1:5000)
"""
import json
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.audit import audit
from src.model import evaluate as ev
from src.model import explain, fairness, reporting, sensitivity
from src.model.config import (AUC_TARGET, CV_FOLDS, EXPERIMENT, FEATURES, MLFLOW_URI, MODEL_DIR,
                              OUT_DIR, RANDOM_STATE, RECALL_TARGET,
                              REGISTERED_MODEL, SENSITIVE, TARGET, TARGET_RULES,
                              DEFAULT_TARGET_RULE)
from src.model.dataset import build_modelling_table, load_inputs, split
from src.model.models import baseline_models, candidate_models, fine_grid, rule_baseline_scores


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _num(d):
    return {k: float(v) for k, v in d.items() if isinstance(v, (int, float, np.floating, np.integer))}


def tune(name, pipeline, space, n_iter, fine_keys, X, y, mlflow):
    """Coarse random search, then a small grid around the best values. Every candidate is logged."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    coarse = RandomizedSearchCV(pipeline, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
                                random_state=RANDOM_STATE, n_jobs=-1, refit=False)
    coarse.fit(X, y)
    best = coarse.best_params_
    fine = GridSearchCV(pipeline, fine_grid(best, fine_keys), scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    fine.fit(X, y)
    for stage, search in (("coarse", coarse), ("fine", fine)):
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-{stage}-{i:02d}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                mlflow.log_metrics({"cv_roc_auc": float(res["mean_test_score"][i]),
                                    "cv_roc_auc_std": float(res["std_test_score"][i])})
                mlflow.set_tags({"stage": stage, "model_family": name})
    return fine


def run(strict: bool = True, rule: str = DEFAULT_TARGET_RULE):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    suffix = "" if rule == DEFAULT_TARGET_RULE else f"_{rule}"
    out_dir = OUT_DIR.parent / f"model{suffix}"
    model_file = MODEL_DIR / f"drift_model{suffix}.joblib"
    meta_file = MODEL_DIR / f"model_meta{suffix}.json"
    registered = REGISTERED_MODEL + suffix.replace("_", "-")
    out_dir.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    # 1. Data
    _log("Building modelling table from Module 3 outputs")
    monthly, audit_table = load_inputs()
    X, y, s, data_report = build_modelling_table(monthly, audit_table, rule)
    data_report["target_definition"] = TARGET_RULES[rule]
    _log(f"{len(X):,} customers, drift rate {y.mean():.3f}")
    data_bias = fairness.group_table(y, y, s)[["attribute", "group", "customers", "base_drift_rate"]]
    data_bias["share_pct"] = (100 * data_bias["customers"] / len(X)).round(2)
    data_bias.to_csv(out_dir / "data_bias.csv", index=False)

    X_tr, X_te, y_tr, y_te, s_tr, s_te = split(X, y, s)
    split_info = {"train_rows": len(X_tr), "test_rows": len(X_te),
                  "train_drift_rate": round(float(y_tr.mean()), 4), "test_drift_rate": round(float(y_te.mean()), 4)}

    rows, curves, fitted, infos = [], {}, {}, {}

    # 2. Baselines
    for name, model in baseline_models().items():
        with mlflow.start_run(run_name=name):
            model.fit(X_tr[["product_count"]], y_tr)
            scores = model.predict_proba(X_te[["product_count"]])[:, 1]
            m = ev.classification_metrics(y_te, scores, 0.5)
            mlflow.log_params({"strategy": model.strategy})
            mlflow.log_metrics(_num(m))
            mlflow.set_tag("role", "baseline")
            rows.append({"model": name, "cv_roc_auc": np.nan, **m})
    with mlflow.start_run(run_name="baseline_business_rule"):
        rule_te = rule_baseline_scores(X_te)
        m = ev.classification_metrics(y_te, rule_te, 0.5)   # flag if either warning sign is present
        mlflow.log_params({"rule": "dropped a product OR inactive in Dec/Jan"})
        mlflow.log_metrics(_num(m))
        mlflow.set_tag("role", "baseline")
        rows.append({"model": "baseline_business_rule", "cv_roc_auc": np.nan, **m})
        curves["Business rule"] = rule_te

    # 3. Candidate models with tuning
    for name, (pipe, space, n_iter, fine_keys) in candidate_models().items():
        _log(f"Tuning {name}")
        with mlflow.start_run(run_name=name) as run_:
            mlflow.set_tags({"role": "candidate", "target": TARGET, "target_rule": rule})
            mlflow.log_params({"features": len(FEATURES), "cv_folds": CV_FOLDS, **split_info})
            search = tune(name, pipe, space, n_iter, fine_keys, X_tr, y_tr, mlflow)
            best = search.best_estimator_
            oof = cross_val_predict(clone(best), X_tr, y_tr, method="predict_proba",
                                    cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE))[:, 1]
            threshold = ev.threshold_for_recall(y_tr, oof, RECALL_TARGET)
            scores = best.predict_proba(X_te)[:, 1]
            m = ev.classification_metrics(y_te, scores, threshold)
            m["cv_roc_auc"] = float(search.best_score_)
            mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
            mlflow.log_metrics({f"test_{k}" if k != "cv_roc_auc" else k: v for k, v in _num(m).items()})
            mlflow.log_figure(ev.confusion_figure(y_te, scores, threshold, f"{name}: confusion matrix (test)"),
                              "plots/confusion_matrix.png")
            mlflow.log_figure(ev.roc_figure(y_te, {name: scores}), "plots/roc_curve.png")
            info = mlflow.sklearn.log_model(best, "model", input_example=X_tr.head(3), serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                                            signature=infer_signature(X_tr.head(50), best.predict_proba(X_tr.head(50))))
            rows.append({"model": name, **m})
            curves[name] = scores
            fitted[name] = (best, threshold, scores)
            infos[name] = (run_.info.run_id, info.model_uri)

    comparison = pd.DataFrame(rows)
    num_cols = comparison.select_dtypes("number").columns
    comparison[num_cols] = comparison[num_cols].round(4)
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    # 4. Champion = best cross-validated ROC-AUC (chosen without looking at the test set)
    champion = max(fitted, key=lambda n: comparison.set_index("model").loc[n, "cv_roc_auc"])
    model, threshold, scores = fitted[champion]
    test_m = comparison.set_index("model").loc[champion].to_dict()
    accepted = bool(test_m["recall"] >= RECALL_TARGET and test_m["roc_auc"] >= AUC_TARGET)
    _log(f"Champion: {champion} (test recall {test_m['recall']:.3f}, AUC {test_m['roc_auc']:.3f}, accepted={accepted})")

    for fig, fname in [(ev.roc_figure(y_te, curves), "roc_curves.png"),
                       (ev.pr_figure(y_te, curves, RECALL_TARGET), "precision_recall.png"),
                       (ev.confusion_figure(y_te, scores, threshold, f"Champion ({champion}): confusion matrix"),
                        "confusion_matrix.png"),
                       (ev.comparison_figure([r for r in rows if r["model"] in fitted]), "model_comparison.png")]:
            fig.savefig(out_dir / fname, dpi=180)

    run_id, model_uri = infos[champion]
    version = None
    try:
        mv = mlflow.register_model(model_uri, registered)
        version = mv.version
        client = MlflowClient()
        client.set_registered_model_alias(registered, "champion", version)
        client.set_model_version_tag(registered, version, "stage", "Staging" if accepted else "Rejected")
        client.set_model_version_tag(registered, version, "acceptance_passed", str(accepted))
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        _log(f"Model registry skipped: {exc}")

    # 5. Fairness
    _log("Fairness audit")
    y_pred = (scores >= threshold).astype(int)
    before = fairness.group_table(y_te, y_pred, s_te)
    before_sum = fairness.summary(before)
    fl_check = fairness.fairlearn_crosscheck(y_te.values, y_pred, s_te, require=strict)
    before.to_csv(out_dir / "fairness_by_group.csv", index=False)
    fairness.fairness_figure(before).savefig(out_dir / "fairness_recall_by_group.png", dpi=180)
    after = after_sum = mitigated_attr = None
    mitigated_m = {}
    failing = before_sum[~before_sum["passes"]]
    if len(failing):
        mitigated_attr = failing.sort_values("equalized_odds_difference").iloc[-1]["attr_key"]
        _log(f"Mitigating on {mitigated_attr}")
        try:
            with mlflow.start_run(run_name=f"{champion}-mitigated-{mitigated_attr}"):
                y_mit, s_mit = fairness.mitigate(model, X_tr, y_tr, s_tr, X_te, mitigated_attr)
                after = fairness.group_table(y_te, y_mit, s_te)
                after_sum = fairness.summary(after)
                mitigated_m = ev.classification_metrics(y_te, y_mit.astype(float), 0.5)
                if len(np.unique(s_mit)) > 1:
                    mitigated_m["roc_auc"] = float(roc_auc_score(y_te, s_mit))
                mlflow.log_params({"method": "ExponentiatedGradient", "constraint": "EqualizedOdds",
                                   "sensitive_attribute": mitigated_attr})
                mlflow.log_metrics({f"test_{k}": v for k, v in _num(mitigated_m).items()})
                for _, r in after_sum.iterrows():
                    mlflow.log_metric(f"eo_diff_{r['attr_key']}", r["equalized_odds_difference"])
                after.to_csv(out_dir / "fairness_by_group_mitigated.csv", index=False)
                fig = fairness.fairness_figure(before, after, " before and after mitigation")
                fig.savefig(out_dir / "fairness_mitigation.png", dpi=180)
                mlflow.log_figure(fig, "plots/fairness_mitigation.png")
        except ImportError:
            if strict:
                raise
            _log("Fairlearn not installed: mitigation skipped")
    report_md = reporting.fairness_report(data_bias, before, before_sum, fl_check, after, after_sum,
                                          mitigated_attr, mitigated_m, test_m)
    (out_dir / "fairness_report.md").write_text(report_md, encoding="utf-8")

    # 6. Explainability
    _log("Explainability (SHAP, LIME, counterfactuals)")
    top = pd.DataFrame(columns=["feature", "mean_abs_shap"])
    examples = explain.pick_examples(y_te, scores, threshold)
    try:
        sample = X_te.sample(min(1000, len(X_te)), random_state=RANDOM_STATE).index
        rows_ = sorted(set(sample) | set(examples.values()))
        X_exp = X_te.loc[rows_].reset_index(drop=True)
        pos = {label: rows_.index(i) for label, i in examples.items()}
        exp_all = explain.shap_explanation(model, X_tr.sample(min(300, len(X_tr)), random_state=RANDOM_STATE), X_exp)
        top = explain.shap_global(exp_all, out_dir)
        explain.shap_local(exp_all, pos, out_dir)
        explain.lime_local(model, X_tr, X_exp, pos, out_dir)
        pd.DataFrame([{"case": k, "test_row": v, "risk": round(float(scores[v]), 3), "actual_drift": int(y_te[v])}
                      for k, v in examples.items()]).to_csv(out_dir / "explained_customers.csv", index=False)
    except ImportError:
        if strict:
            raise
        _log("SHAP/LIME not installed: skipped")
    cfs = explain.counterfactuals(model, X_te, threshold)
    cfs.to_csv(out_dir / "counterfactuals.csv", index=False)
    explain.offer_effects(model, X_te, threshold).to_csv(out_dir / "offer_effects.csv", index=False)

    # 7. Sensitivity
    _log("Sensitivity analysis")
    sens = sensitivity.input_sensitivity(model, X_te, threshold)
    seeds = sensitivity.seed_stability(model, X_tr, y_tr, X_te, y_te)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    seeds.to_csv(out_dir / "seed_stability.csv", index=False)
    sensitivity.sensitivity_figure(sens).savefig(out_dir / "sensitivity.png", dpi=180)

    # 8. Log champion evidence, save the model and write the model card
    meta = {"registered_model": registered, "target_rule": rule, "model_version": version, "champion": champion,
            "threshold": float(threshold), "features": FEATURES, "target": TARGET,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mlflow_run_id": run_id, "acceptance_passed": accepted, "data": data_report,
            "split": split_info, "test_metrics": {k: test_m[k] for k in
                                                  ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                                                   "top20_capture", "flagged_rate"]},
            "protected_attributes_excluded": list(SENSITIVE)}
    joblib.dump(model, model_file)
    meta_file.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    card = reporting.model_card(meta, comparison[comparison["model"].isin(list(fitted) + ["baseline_business_rule",
                                                                                           "baseline_majority"])],
                                before_sum, top, sens, seeds)
    (out_dir / "model_card.md").write_text(card, encoding="utf-8")
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({"champion": "true", "acceptance_passed": str(accepted)})
        mlflow.log_metric("decision_threshold", float(threshold))
        for f in out_dir.glob("*"):
            mlflow.log_artifact(str(f), "reports")
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    audit("write", "drift_model", rows_in=len(X), details={"champion": champion, "version": version,
                                                         "accepted": accepted, "threshold": float(threshold)},
          path=model_file)
    _log(f"Done. Model saved to {model_file}; reports in {out_dir}")
    return meta


if __name__ == "__main__":
    chosen = sys.argv[sys.argv.index("--target") + 1] if "--target" in sys.argv else DEFAULT_TARGET_RULE
    run(strict="--lenient" not in sys.argv, rule=chosen)


"""Monthly monitoring: data drift, prediction drift, fairness drift and performance.

Each month the newest cohort (customers active at that month's end, described by their
last three months) is scored with the deployed models and compared with the training
cohort (February 2016). Outcomes arrive three months later, so performance can only be
measured once labels exist: in this dataset, for the February cohort.

A labelled, simulated stress cohort (income -20%, one fewer active month for 30% of
customers) shows how the alerts behave when the data really does shift.

Run from the repository root after the pipeline and training:
    python -m src.monitoring.monitor
Outputs: reports/monitoring/*.csv, summary.json and (if Evidently is installed)
evidently_data_drift.html. All outputs are aggregates: no customer rows leave this step.
"""
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from src.config import REPORTS_DIR
from src.model.config import (CATEGORICAL_FEATURES, DI_BAND, MIN_GROUP_SIZE, MODEL_DIR, NUMERIC_FEATURES,
                              SENSITIVE)
from src.model.dataset import _snapshot, build_features, build_modelling_table, split

OUT_DIR = REPORTS_DIR / "monitoring"
MODELS = {"main": ("inactive_or_dropping", ""), "inactive_only": ("inactive", "_inactive")}
REFERENCE = ("2016-02", ["2015-12-28", "2016-01-28", "2016-02-28"], "2016-02-28")
COHORTS = [("2016-03", ["2016-01-28", "2016-02-28", "2016-03-28"], "2016-03-28"),
           ("2016-04", ["2016-02-28", "2016-03-28", "2016-04-28"], "2016-04-28"),
           ("2016-05", ["2016-03-28", "2016-04-28", "2016-05-28"], "2016-05-28")]
PSI_WATCH, PSI_ALERT = 0.10, 0.25          # common industry rule of thumb (Siddiqi, 2006)
FLAG_RATE_WATCH = 0.05                     # 5-point move in the share of customers flagged
STRESS_LABEL = "Simulated stress"


# ---------- drift measures ----------
def psi(reference, current, bins: int = 10, categorical: bool = False) -> float:
    """Population Stability Index between two samples of one feature."""
    eps = 1e-4
    ref, cur = pd.Series(reference).dropna(), pd.Series(current).dropna()
    if categorical or ref.nunique() <= bins:          # few distinct values: compare shares directly
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return psi(ref, cur, categorical=True)
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float, watch: float, alert: float) -> str:
    return "alert" if value >= alert else "watch" if value >= watch else "ok"


def data_drift(ref: pd.DataFrame, cur: pd.DataFrame, month: str) -> pd.DataFrame:
    rows = []
    for col in ref.columns:
        cat = col in CATEGORICAL_FEATURES
        value = psi(ref[col], cur[col], categorical=cat)
        row = {"month": month, "feature": col, "psi": round(value, 4), "status": status(value, PSI_WATCH, PSI_ALERT)}
        if col in NUMERIC_FEATURES:
            row["ks_pvalue"] = round(float(ks_2samp(ref[col], cur[col]).pvalue), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def fairness_rates(flagged: pd.Series, attrs: pd.DataFrame, month: str) -> pd.DataFrame:
    """Offer (flag) rate per protected group, compared with the overall rate (disparate impact)."""
    overall = float(flagged.mean())
    rows = []
    for attr in SENSITIVE:
        for group, idx in attrs.groupby(attr).groups.items():
            n = len(idx)
            rate = float(flagged.loc[idx].mean())
            ratio = rate / overall if overall else np.nan
            judged = n >= MIN_GROUP_SIZE and group != "UNKNOWN"
            rows.append({"month": month, "attribute": attr, "group": group, "customers": n,
                         "offer_rate": round(rate, 4), "ratio_to_overall": round(ratio, 3), "judged": judged,
                         "within_band": bool(DI_BAND[0] <= ratio <= DI_BAND[1]) if judged else None})
    return pd.DataFrame(rows)


# ---------- cohorts ----------
def cohort(monthly, audit_table, months, cutoff, features):
    f = build_features(monthly, months, cutoff)
    f = f[f["active_at_cutoff"] == 1]
    attrs = _snapshot(audit_table, cutoff).reindex(f.index)[list(SENSITIVE)].astype(str)
    attrs = attrs.replace({"nan": "UNKNOWN", "<NA>": "UNKNOWN"})
    return f[features], attrs


def stress(X: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Simulated shock: incomes fall 20%, and 30% of customers lose one active month."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    out["household_income"] = out["household_income"] * 0.8
    hit = rng.random(len(out)) < 0.30
    out.loc[hit, "months_active_3m"] = (out.loc[hit, "months_active_3m"] - 1).clip(lower=1)
    return out


def evidently_report(ref: pd.DataFrame, cur: pd.DataFrame, path) -> bool:
    """Optional Evidently AI data-drift report (HTML of distributions only)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        print("Evidently not installed: skipping the HTML report (pip install 'evidently>=0.7,<0.8').")
        return False
    snapshot = Report([DataDriftPreset()]).run(current_data=cur, reference_data=ref)
    snapshot.save_html(str(path))
    return True


# ---------- main run ----------
def run(monthly, audit_table, out_dir=OUT_DIR, with_evidently=True) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    drift_rows, pred_rows, fair_rows, perf_rows, alerts = [], [], [], [], []
    for key, (rule, suffix) in MODELS.items():
        model = joblib.load(MODEL_DIR / f"drift_model{suffix}.joblib")
        meta = json.loads((MODEL_DIR / f"model_meta{suffix}.json").read_text(encoding="utf-8"))
        feats, thr = meta["features"], meta["threshold"]

        # Performance on the labelled reference cohort (test split only, never training rows)
        X, y, s, _ = build_modelling_table(monthly, audit_table, rule)
        _, X_te, _, y_te, _, _ = split(X, y, s)
        p_te = model.predict_proba(X_te[feats])[:, 1]
        perf_rows.append({"model": key, "month": REFERENCE[0], "labels": "available (May 2016)",
                          "roc_auc": round(roc_auc_score(y_te, p_te), 4),
                          "recall": round(recall_score(y_te, p_te >= thr), 4),
                          "precision": round(precision_score(y_te, p_te >= thr), 4)})

        ref_X, ref_attrs = cohort(monthly, audit_table, REFERENCE[1], REFERENCE[2], feats)
        ref_flag_rate = None
        cohorts = [(REFERENCE[0], ref_X, ref_attrs)]
        for month, months, cutoff in COHORTS:
            cohorts.append((month, *cohort(monthly, audit_table, months, cutoff, feats)))
        cohorts.append((STRESS_LABEL, stress(cohorts[-1][1]), cohorts[-1][2]))

        for month, Xc, attrs in cohorts:
            p = pd.Series(model.predict_proba(Xc)[:, 1], index=Xc.index)
            flagged = p >= thr
            flag_rate = float(flagged.mean())
            ref_flag_rate = flag_rate if ref_flag_rate is None else ref_flag_rate
            pred_rows.append({"model": key, "month": month, "customers": len(Xc),
                              "mean_score": round(float(p.mean()), 4), "flagged_rate": round(flag_rate, 4),
                              "flag_rate_change": round(flag_rate - ref_flag_rate, 4)})
            fr = fairness_rates(flagged, attrs, month)
            fr.insert(0, "model", key)
            fair_rows.append(fr)
            if month != REFERENCE[0]:
                dd = data_drift(ref_X, Xc, month)
                dd.insert(0, "model", key)
                drift_rows.append(dd)
                for _, r in dd[dd["status"] == "alert"].iterrows():
                    alerts.append({"model": key, "month": month, "type": "data drift",
                                   "detail": f"{r['feature']} PSI {r['psi']:.2f} (alert at {PSI_ALERT})"})
                if abs(flag_rate - ref_flag_rate) >= FLAG_RATE_WATCH:
                    alerts.append({"model": key, "month": month, "type": "prediction drift",
                                   "detail": f"flagged share {flag_rate:.1%} vs {ref_flag_rate:.1%} at training"})
            for _, r in fr[(fr["judged"]) & (fr["within_band"] == False)].iterrows():  # noqa: E712
                alerts.append({"model": key, "month": month, "type": "fairness",
                               "detail": f"{r['attribute']} {r['group']}: offer rate {r['ratio_to_overall']:.2f}x "
                                         f"overall (band {DI_BAND[0]}-{DI_BAND[1]})"})
        if with_evidently and key == "main":
            evidently_report(ref_X, cohorts[-2][1], out_dir / "evidently_data_drift.html")

    tables = {"data_drift": pd.concat(drift_rows), "prediction_drift": pd.DataFrame(pred_rows),
              "fairness_drift": pd.concat(fair_rows), "performance": pd.DataFrame(perf_rows),
              "alerts": pd.DataFrame(alerts, columns=["model", "month", "type", "detail"])}
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    summary = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "reference_cohort": REFERENCE[0], "cohorts": [c[0] for c in COHORTS] + [STRESS_LABEL],
               "psi_watch": PSI_WATCH, "psi_alert": PSI_ALERT, "fairness_band": list(DI_BAND),
               "alerts_by_type": tables["alerts"].groupby("type").size().to_dict()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tables


def main():
    from src.model.dataset import load_inputs
    monthly, audit_table = load_inputs()
    tables = run(monthly, audit_table)
    print(tables["prediction_drift"].to_string(index=False))
    print(tables["alerts"].to_string(index=False))


if __name__ == "__main__":
    main()

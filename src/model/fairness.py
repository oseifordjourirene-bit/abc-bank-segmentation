"""Fairness metrics (Fairlearn) and mitigation for the drift model.

"Selected" means flagged for the reactivation offer. The offer is a benefit, so a group
that is selected less often, or whose drifters are caught less often, is disadvantaged.
Protected attributes are used here only; they are never model inputs.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.model.config import DI_BAND, EO_GAP_MAX, MIN_GROUP_SIZE, RANDOM_STATE, SENSITIVE  # noqa: E402


def _rates(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    pos, neg = y_true == 1, y_true == 0
    return {"selection_rate": y_pred.mean() if len(y_pred) else np.nan,
            "tpr": y_pred[pos].mean() if pos.any() else np.nan,
            "fpr": y_pred[neg].mean() if neg.any() else np.nan,
            "base_rate": y_true.mean() if len(y_true) else np.nan}


def group_table(y_true, y_pred, sensitive: pd.DataFrame) -> pd.DataFrame:
    """One row per protected group: size, drift base rate, selection rate, recall, FPR, DI ratio."""
    y_true, y_pred = pd.Series(np.asarray(y_true)), pd.Series(np.asarray(y_pred))
    overall = _rates(y_true, y_pred)
    rows = []
    for attr, label in SENSITIVE.items():
        groups = sensitive[attr].reset_index(drop=True)
        for g in sorted(groups.unique()):
            m = (groups == g).values
            r = _rates(y_true[m], y_pred[m])
            n = int(m.sum())
            di = r["selection_rate"] / overall["selection_rate"] if overall["selection_rate"] else np.nan
            if n < MIN_GROUP_SIZE:
                status = f"Too small to judge (<{MIN_GROUP_SIZE})"
            elif not DI_BAND[0] <= di <= DI_BAND[1]:
                status = f"Check: DI outside {DI_BAND[0]}-{DI_BAND[1]}"
            else:
                status = "OK"
            rows.append({"attribute": label, "attr_key": attr, "group": str(g), "customers": n,
                         "base_drift_rate": r["base_rate"], "selection_rate": r["selection_rate"],
                         "recall_tpr": r["tpr"], "false_positive_rate": r["fpr"],
                         "disparate_impact": di, "status": status})
    return pd.DataFrame(rows).round(4)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per attribute, over groups large enough to judge: DP ratio, EO difference, pass/fail."""
    rows = []
    for attr, sub in table.groupby("attr_key", sort=False):
        ok = sub[sub["customers"] >= MIN_GROUP_SIZE]
        sel, tpr, fpr = ok["selection_rate"], ok["recall_tpr"], ok["false_positive_rate"]
        dp_ratio = float(sel.min() / sel.max()) if len(ok) > 1 and sel.max() > 0 else 1.0
        eo_diff = float(max(tpr.max() - tpr.min(), fpr.max() - fpr.min())) if len(ok) > 1 else 0.0
        di_ok = bool(ok["disparate_impact"].between(*DI_BAND).all())
        rows.append({"attribute": SENSITIVE[attr], "attr_key": attr, "groups_judged": len(ok),
                     "demographic_parity_ratio": round(dp_ratio, 4),
                     "equalized_odds_difference": round(eo_diff, 4),
                     "di_within_band": di_ok, "eo_within_limit": eo_diff <= EO_GAP_MAX,
                     "passes": di_ok and eo_diff <= EO_GAP_MAX})
    return pd.DataFrame(rows)


def fairlearn_crosscheck(y_true, y_pred, sensitive: pd.DataFrame, require: bool = True) -> pd.DataFrame:
    """Same metrics computed by Fairlearn, as an independent check (all groups included)."""
    try:
        from fairlearn.metrics import (MetricFrame, demographic_parity_ratio,
                                       equalized_odds_difference, false_positive_rate,
                                       selection_rate, true_positive_rate)
    except ImportError:
        if require:
            raise
        return pd.DataFrame()
    rows = []
    for attr, label in SENSITIVE.items():
        s = sensitive[attr].values
        mf = MetricFrame(metrics={"selection_rate": selection_rate, "tpr": true_positive_rate,
                                  "fpr": false_positive_rate},
                         y_true=y_true, y_pred=y_pred, sensitive_features=s)
        rows.append({"attribute": label,
                     "fairlearn_dp_ratio_all_groups": demographic_parity_ratio(y_true, y_pred, sensitive_features=s),
                     "fairlearn_eo_difference_all_groups": equalized_odds_difference(y_true, y_pred, sensitive_features=s),
                     "fairlearn_selection_rate_max_gap": float(mf.difference()["selection_rate"])})
    return pd.DataFrame(rows).round(4)


def mitigate(pipeline, X_train, y_train, s_train, X_test, attr: str):
    """Fairlearn ExponentiatedGradient with an equalized-odds constraint on one attribute.
    The protected attribute is used during training only, never when scoring a customer."""
    from fairlearn.reductions import EqualizedOdds, ExponentiatedGradient
    from sklearn.base import clone
    prep = clone(pipeline.named_steps["prep"]).fit(X_train, y_train)
    mitigator = ExponentiatedGradient(clone(pipeline.named_steps["model"]),
                                      constraints=EqualizedOdds(), max_iter=40)
    mitigator.fit(prep.transform(X_train), np.asarray(y_train), sensitive_features=s_train[attr].values)
    Xt = prep.transform(X_test)
    y_pred = np.asarray(mitigator.predict(Xt, random_state=RANDOM_STATE)).astype(int)
    try:
        scores = mitigator._pmf_predict(Xt)[:, 1]
    except Exception:  # noqa: BLE001 - older/newer Fairlearn versions
        scores = y_pred.astype(float)
    return y_pred, scores


def fairness_figure(before: pd.DataFrame, after: pd.DataFrame | None = None, title_suffix=""):
    """Recall (TPR) per group: are each group's drifters caught equally often?"""
    attrs = list(before["attribute"].unique())
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.4 * len(attrs), 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [max(2, (before["attribute"] == a).sum()) for a in attrs]})
    for ax, attr in zip(np.atleast_1d(axes), attrs):
        b = before[before["attribute"] == attr]
        x = np.arange(len(b))
        w = 0.38 if after is not None else 0.6
        colours = ["#1F3864" if c >= MIN_GROUP_SIZE else "#B8C2D6" for c in b["customers"]]
        ax.bar(x - (w / 2 if after is not None else 0), b["recall_tpr"], w, color=colours, label="Before")
        if after is not None:
            a = after[after["attribute"] == attr].set_index("group").reindex(b["group"])
            ax.bar(x + w / 2, a["recall_tpr"], w, color="#E8A33D", label="After mitigation")
        ax.set_xticks(x, b["group"], fontsize=8)
        ax.set_title(attr, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Recall: share of drifters caught")
    if after is not None:
        np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Recall by protected group{title_suffix} (pale bars = group under {MIN_GROUP_SIZE}, not judged)",
                 fontsize=9, color="#444")
    fig.tight_layout()
    return fig

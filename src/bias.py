"""Stage 6 - Bias detection: representation bias across protected groups (Fairlearn)."""
from pathlib import Path

import pandas as pd

from src.audit import audit
from src.config import MIN_GROUP_SIZE, PARITY_BAND, REPORTS_DIR

SENSITIVE = {"sex": "Sex (H/V as coded in source)", "age_band": "Age band",
             "foreign_born_flag": "Foreign-born"}


def _pandas_rates(frame: pd.DataFrame, attr: str) -> pd.DataFrame:
    g = frame.groupby(attr, observed=True)
    return pd.DataFrame({"count": g.size(), "active_rate": g["active_now"].mean()})


def _fairlearn_rates(frame: pd.DataFrame, attr: str) -> pd.DataFrame:
    from fairlearn.metrics import MetricFrame, count, selection_rate
    mf = MetricFrame(metrics={"count": count, "active_rate": selection_rate},
                     y_true=frame["active_now"], y_pred=frame["active_now"],
                     sensitive_features=frame[attr])
    return mf.by_group


def representation_report(features: pd.DataFrame, audit_table: pd.DataFrame,
                          require_fairlearn: bool = True,
                          out_dir: Path = REPORTS_DIR / "bias") -> tuple[pd.DataFrame, dict]:
    latest = audit_table[audit_table["snapshot_date"] == audit_table["snapshot_date"].max()]
    frame = features.merge(latest.drop(columns=["snapshot_date"]), on="customer_key", how="left")
    frame["foreign_born_flag"] = frame["foreign_born_flag"].fillna("UNKNOWN")

    try:
        import fairlearn  # noqa: F401
        engine = "fairlearn"
    except ImportError:
        if require_fairlearn:
            raise
        engine = "pandas (fairlearn not installed)"

    rows, total = [], len(frame)
    overall = {"active_rate": frame["active_now"].mean(),
               "product_count": frame["product_count"].mean(),
               "drift_rate": frame["drift_signal"].mean()}
    for attr, label in SENSITIVE.items():
        rates = (_fairlearn_rates if engine == "fairlearn" else _pandas_rates)(frame, attr)
        extra = frame.groupby(attr, observed=True).agg(
            product_count=("product_count", "mean"), drift_rate=("drift_signal", "mean"))
        rates = rates.join(extra)
        for group, r in rates.iterrows():
            n = int(r["count"])
            row = {"attribute": label, "group": str(group), "customers": n,
                   "share_pct": round(100 * n / total, 2),
                   "active_rate": round(float(r["active_rate"]), 3),
                   "active_ratio": round(float(r["active_rate"]) / overall["active_rate"], 2),
                   "avg_products": round(float(r["product_count"]), 2),
                   "drift_rate": round(float(r["drift_rate"]), 3)}
            if n < MIN_GROUP_SIZE:
                row["status"] = f"Too small to compare (<{MIN_GROUP_SIZE})"
            elif not PARITY_BAND[0] <= row["active_ratio"] <= PARITY_BAND[1]:
                row["status"] = "Check: outside 0.8-1.25"
            else:
                row["status"] = "OK"
            rows.append(row)

    report = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_dir / "representation_report.csv", index=False)
    _chart(report, out_dir / "representation_chart.png")
    summary = {"engine": engine, "customers": total,
               "groups_checked": len(report),
               "groups_flagged": int((report["status"] != "OK").sum()),
               "overall_active_rate": round(overall["active_rate"], 3)}
    audit("bias_check", "customer_features", rows_in=total, details=summary)
    return report, summary


def _chart(report: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attrs = report["attribute"].unique()
    order = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+", "SUPPRESSED"]
    report = report.assign(_k=report["group"].map(lambda g: order.index(g) if g in order else -1))
    report = report.sort_values(["_k", "group"])
    top = max(1.6, report["active_ratio"].max() * 1.15)
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.2 * len(attrs), 3.4), sharey=True,
                             gridspec_kw={"width_ratios": [max(2, (report["attribute"] == a).sum()) for a in attrs]})
    for ax, attr in zip(axes, attrs):
        sub = report[report["attribute"] == attr]
        colours = ["#1F3864" if s == "OK" else "#E8A33D" if s.startswith("Too") else "#C0392B"
                   for s in sub["status"]]
        ax.bar(sub["group"], sub["active_ratio"], color=colours)
        ax.axhspan(PARITY_BAND[0], PARITY_BAND[1], color="#2E7D32", alpha=0.08)
        ax.axhline(1, color="#555", lw=0.8)
        ax.set_title(attr, fontsize=10)
        ax.set_ylim(0, top)
        ax.tick_params(axis="x", labelsize=8, rotation=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Active rate vs overall (ratio)")
    fig.suptitle("Representation check: green band = 0.8-1.25; amber = group under 500",
                 fontsize=9, color="#444")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

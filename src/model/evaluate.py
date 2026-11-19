"""Metrics, decision threshold and evaluation charts."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,  # noqa: E402
                             f1_score, precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)

NAVY, GOLD, GREY = "#1F3864", "#E8A33D", "#8C96A8"


def threshold_for_recall(y_true, scores, target: float) -> float:
    """Highest threshold that still catches at least `target` of drifters (fewest wasted offers)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def top_share_capture(y_true, scores, share: float = 0.2) -> float:
    """Share of all drifters found by contacting the riskiest `share` of customers."""
    y_true = np.asarray(y_true)
    n = max(1, int(round(len(scores) * share)))
    top = np.argsort(-np.asarray(scores))[:n]
    return float(y_true[top].sum() / max(1, y_true.sum()))


def classification_metrics(y_true, scores, threshold: float) -> dict:
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    single_class = len(np.unique(scores)) < 2
    return {"threshold": round(float(threshold), 4),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": 0.5 if single_class else roc_auc_score(y_true, scores),
            "pr_auc": float(np.mean(y_true)) if single_class else average_precision_score(y_true, scores),
            "flagged_rate": float(y_pred.mean()),
            "top20_capture": 0.2 if single_class else top_share_capture(y_true, scores, 0.2)}


def roc_figure(y_true, curves: dict, title="ROC curve, hold-out test set"):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colours = [NAVY, GOLD, "#2E7D32", "#C0392B", GREY]
    for (name, scores), c in zip(curves.items(), colours):
        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_true, scores):.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def pr_figure(y_true, curves: dict, recall_target: float):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    colours = [NAVY, GOLD, "#2E7D32", "#C0392B", GREY]
    for (name, scores), c in zip(curves.items(), colours):
        p, r, _ = precision_recall_curve(y_true, scores)
        ax.plot(r, p, color=c, lw=2, label=f"{name} (AP {average_precision_score(y_true, scores):.3f})")
    ax.axvline(recall_target, color="#C0392B", ls=":", lw=1.2, label=f"Recall target {recall_target:.2f}")
    ax.axhline(float(np.mean(y_true)), color=GREY, ls="--", lw=1, label="Drift rate (no skill)")
    ax.set_xlabel("Recall (share of drifters caught)")
    ax.set_ylabel("Precision (share of offers that reach a drifter)")
    ax.set_title("Precision-recall curve, hold-out test set", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def confusion_figure(y_true, scores, threshold: float, title: str):
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["Stays active", "Drifts"], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def comparison_figure(rows: list):
    """Bar chart of test ROC-AUC and recall per model, with the acceptance lines."""
    from src.model.config import AUC_TARGET, RECALL_TARGET
    names = [r["model"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(x - 0.2, [r["roc_auc"] for r in rows], 0.4, color=NAVY, label="ROC-AUC")
    ax.bar(x + 0.2, [r["recall"] for r in rows], 0.4, color=GOLD, label="Recall")
    ax.axhline(AUC_TARGET, color=NAVY, ls=":", lw=1)
    ax.axhline(RECALL_TARGET, color="#B9770E", ls=":", lw=1)
    ax.set_xticks(x, [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the hold-out test set (dotted = acceptance bars)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig

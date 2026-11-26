"""Lightweight local explanations shared by the API explanation store and the dashboard.

Method (feature replacement, also called occlusion): each input is replaced, one at a time,
by its typical value in the training data (median for numbers, most common value otherwise),
and the change in the score is recorded. A positive contribution means this customer's actual
value pushes the score up compared with a typical customer.

It needs only scikit-learn, so it runs inside the small API container. The full SHAP and LIME
analyses from Module 4 remain the reference explanations (notebooks/04_shap_analysis.ipynb).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_FILE = ROOT / "models" / "reference_profile.json"
CATEGORICAL = ("joining_channel", "bank_segment", "relation_type")


def build_reference_profile(X_train: pd.DataFrame) -> dict:
    """Typical values from the training split only (aggregate values, no customer records)."""
    ref = {}
    for col in X_train.columns:
        if col in CATEGORICAL or X_train[col].nunique() <= 2:
            ref[col] = X_train[col].mode().iloc[0]
        else:
            ref[col] = float(X_train[col].median())
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in ref.items()}


def save_reference_profile(ref: dict, path: Path = REFERENCE_FILE) -> Path:
    path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    return path


def load_reference_profile(path: Path = REFERENCE_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _frame(profile: dict, features: list) -> pd.DataFrame:
    row = pd.DataFrame([profile])[features]
    for col in features:
        row[col] = row[col].astype(str) if col in CATEGORICAL else row[col].astype(float)
    return row


def local_contributions(model, profile: dict, features: list, reference: dict, top: int = 5) -> list:
    """Return the top drivers as [{feature, value, typical_value, contribution}], largest first."""
    base = float(model.predict_proba(_frame(profile, features))[:, 1][0])
    variants = []
    for col in features:
        if profile[col] == reference[col]:
            continue
        variants.append((col, {**profile, col: reference[col]}))
    if not variants:
        return []
    batch = pd.concat([_frame(v, features) for _, v in variants], ignore_index=True)
    scores = model.predict_proba(batch)[:, 1]
    out = [{"feature": col, "value": profile[col], "typical_value": reference[col],
            "contribution": round(base - float(s), 4)} for (col, _), s in zip(variants, scores)]
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out[:top]

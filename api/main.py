"""FastAPI service for the ABC Bank drift model.

Run from the repository root:
    uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs and try POST /predict.

The API does not accept age, sex, nationality or province: the model never uses them.
Every prediction is stored with its explanation (api/explanation_store.py) for audit:
GET /explanations/{request_id} returns why a decision was made.
"""
import json
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from api import explanation_store
from src.model.explain_local import load_reference_profile, local_contributions

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "models" / "drift_model.joblib"
META_FILE = ROOT / "models" / "model_meta.json"
REFERENCE_FILE = ROOT / "models" / "reference_profile.json"
POLICY_FILE = ROOT / "models" / "targeting_policy.json"

app = FastAPI(title="ABC Bank drift model", version="1.0",
              description="Scores an active customer's risk of drifting in the next 3 months.")
_state: dict = {}


def _load():
    if "model" not in _state:
        if not MODEL_FILE.exists():
            raise HTTPException(503, "Model not trained yet: run python -m src.model.run_all")
        _state["model"] = joblib.load(MODEL_FILE)
        _state["meta"] = json.loads(META_FILE.read_text(encoding="utf-8"))
        _state["reference"] = load_reference_profile(REFERENCE_FILE) if REFERENCE_FILE.exists() else None
        _state["policy"] = json.loads(POLICY_FILE.read_text(encoding="utf-8")) if POLICY_FILE.exists() else {}
    return _state["model"], _state["meta"]


class Customer(BaseModel):
    """Customer profile at the end of the latest month (behaviour over the last three months)."""
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "product_count": 2, "products_added_2m": 0, "products_dropped_2m": 1, "months_active_3m": 2,
        "months_observed_3m": 3, "tenure_months": 48, "household_income": 95000,
        "has_current_account": 1, "has_salary": 0, "has_pension_credit": 0, "has_term_deposit": 0,
        "has_investments": 0, "has_credit": 0, "has_card": 0, "has_e_account": 1, "has_direct_debit": 0,
        "has_junior_account": 0, "income_was_missing": 0, "new_customer_flag": 0, "is_primary_customer": 1,
        "joining_channel": "KHE", "bank_segment": "02 - PARTICULARES", "relation_type": "A"}})

    product_count: int = Field(ge=0, le=30)
    products_added_2m: int = Field(ge=0, le=30)
    products_dropped_2m: int = Field(ge=0, le=30)
    months_active_3m: int = Field(ge=1, le=3)
    months_observed_3m: int = Field(ge=1, le=3)
    tenure_months: int = Field(ge=0, le=400)
    household_income: float = Field(ge=0)
    has_current_account: Literal[0, 1]
    has_salary: Literal[0, 1]
    has_pension_credit: Literal[0, 1]
    has_term_deposit: Literal[0, 1]
    has_investments: Literal[0, 1]
    has_credit: Literal[0, 1]
    has_card: Literal[0, 1]
    has_e_account: Literal[0, 1]
    has_direct_debit: Literal[0, 1]
    has_junior_account: Literal[0, 1]
    income_was_missing: Literal[0, 1]
    new_customer_flag: Literal[0, 1]
    is_primary_customer: Literal[0, 1]
    joining_channel: str
    bank_segment: str
    relation_type: str


class Factor(BaseModel):
    feature: str
    value: float | str
    typical_value: float | str
    contribution: float


class Prediction(BaseModel):
    drift_probability: float
    flag_for_reactivation_offer: bool
    contact_tier: str
    threshold: float
    model: str
    model_version: str | None
    request_id: str
    top_factors: list[Factor]


def contact_tier(score: float, threshold: float, policy: dict) -> str:
    """Precision review (final project): riskiest tier gets a personal call, the rest a digital nudge."""
    if score < threshold:
        return "none"
    tier1 = policy.get("tier1_score_cutoff")
    return "tier 1: relationship-manager call" if tier1 is not None and score >= tier1 \
        else "tier 2: digital nudge"


@app.get("/health")
def health():
    model, meta = _load()
    return {"status": "ok", "model": meta["champion"], "version": meta.get("model_version"),
            "trained_at": meta["trained_at"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    model, meta = _load()
    row = pd.DataFrame([customer.model_dump()])[meta["features"]]
    for col in row.columns:
        if col not in ("joining_channel", "bank_segment", "relation_type"):
            row[col] = row[col].astype(float)
    p = float(model.predict_proba(row)[:, 1][0])
    inputs = customer.model_dump()
    factors = local_contributions(model, inputs, meta["features"], _state["reference"]) \
        if _state.get("reference") else []
    version = None if meta.get("model_version") is None else str(meta["model_version"])
    tier = contact_tier(p, meta["threshold"], _state.get("policy", {}))
    request_id = explanation_store.save({
        "model": meta["champion"], "model_version": version, "threshold": meta["threshold"], "score": p,
        "flagged": p >= meta["threshold"], "contact_tier": tier, "method": "feature replacement vs typical customer",
        "inputs": inputs, "top_factors": factors})
    return Prediction(drift_probability=round(p, 4), flag_for_reactivation_offer=p >= meta["threshold"],
                      contact_tier=tier, threshold=round(meta["threshold"], 4), model=meta["champion"],
                      model_version=version, request_id=request_id, top_factors=factors)


@app.get("/explanations/{request_id}")
def get_explanation(request_id: str):
    """Audit: retrieve a stored decision and its explanation."""
    record = explanation_store.get(request_id)
    if record is None:
        raise HTTPException(404, "No stored prediction with that request_id")
    return record


@app.get("/explanations")
def list_explanations(limit: int = Query(20, ge=1, le=200)):
    """Audit: the most recent stored decisions, newest first."""
    return explanation_store.recent(limit)

"""Module 4: the /predict endpoint scores a customer and refuses protected attributes."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import api.main as api_main  # noqa: E402
from src.model.config import BINARY_FEATURES, CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES  # noqa: E402
from src.model.models import candidate_models  # noqa: E402

EXAMPLE = api_main.Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    X = pd.DataFrame({c: rng.integers(1, 4, 300).astype(float) for c in NUMERIC_FEATURES + BINARY_FEATURES})
    for c in CATEGORICAL_FEATURES:
        X[c] = rng.choice(["KHE", "02 - PARTICULARES", "A"], 300)
    y = rng.integers(0, 2, 300)
    model = candidate_models()["logistic_regression"][0].fit(X[FEATURES], y)
    joblib.dump(model, tmp_path / "m.joblib")
    (tmp_path / "meta.json").write_text(json.dumps({"features": FEATURES, "threshold": 0.5,
                                                    "champion": "test", "model_version": "1",
                                                    "trained_at": "now"}))
    monkeypatch.setattr(api_main, "MODEL_FILE", tmp_path / "m.joblib")
    monkeypatch.setattr(api_main, "META_FILE", tmp_path / "meta.json")
    api_main._state.clear()
    return TestClient(api_main.app)


def test_predict_returns_probability_and_decision(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["drift_probability"] <= 1
    assert body["flag_for_reactivation_offer"] == (body["drift_probability"] >= 0.5)


def test_protected_attributes_are_rejected(client):
    assert client.post("/predict", json={**EXAMPLE, "age": 30}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "sex": "H"}).status_code == 422


def test_prediction_is_stored_with_its_explanation(client, tmp_path, monkeypatch):
    monkeypatch.setenv("EXPLANATION_DB", str(tmp_path / "store.db"))
    body = client.post("/predict", json=EXAMPLE).json()
    assert body["contact_tier"] in {"none", "tier 1: relationship-manager call", "tier 2: digital nudge"}
    stored = client.get(f"/explanations/{body['request_id']}")
    assert stored.status_code == 200
    assert stored.json()["score"] == pytest.approx(body["drift_probability"], abs=1e-4)
    assert "age" not in stored.json()["inputs"]
    assert client.get("/explanations/not-a-real-id").status_code == 404

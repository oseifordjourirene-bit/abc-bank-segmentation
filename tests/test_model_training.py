"""Module 4: thresholds, tuning helpers, counterfactuals and sensitivity on synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.model import evaluate as ev
from src.model import explain, sensitivity
from src.model.config import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.model.models import candidate_models, fine_grid


@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame({c: rng.integers(0, 4, n).astype(float) for c in NUMERIC_FEATURES})
    X["months_active_3m"] = rng.integers(1, 4, n).astype(float)
    X["household_income"] = rng.normal(100000, 20000, n).clip(0)
    for c in BINARY_FEATURES:
        X[c] = rng.integers(0, 2, n).astype(float)
    for c in CATEGORICAL_FEATURES:
        X[c] = rng.choice(["A", "B", "C"], n)
    logit = -2 + 1.2 * (X["months_active_3m"] < 3) + 0.8 * X["products_dropped_2m"] - 0.9 * X["has_direct_debit"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def test_threshold_reaches_recall_target():
    y = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    s = np.array([.1, .2, .9, .8, .3, .4, .35, .7, .05, .15])
    t = ev.threshold_for_recall(y, s, 0.75)
    assert ev.classification_metrics(y, s, t)["recall"] >= 0.75


def test_fine_grid_brackets_best_value():
    grid = fine_grid({"model__C": 1.0, "model__max_depth": 10, "model__n_estimators": 200},
                     ["model__C", "model__max_depth"])
    assert grid["model__C"][0] < 1.0 < grid["model__C"][-1]
    assert 10 in grid["model__max_depth"] and grid["model__n_estimators"] == [200]


def test_every_candidate_trains_and_beats_chance(synthetic):
    X, y = synthetic
    for name, (pipe, *_rest) in candidate_models().items():
        pipe.fit(X, y)
        auc = ev.classification_metrics(y, pipe.predict_proba(X)[:, 1], 0.5)["roc_auc"]
        assert auc > 0.6, name


def test_counterfactuals_change_only_actionable_features(synthetic):
    X, y = synthetic
    pipe = candidate_models()["logistic_regression"][0].fit(X, y)
    t = float(np.quantile(pipe.predict_proba(X)[:, 1], 0.7))
    cfs = explain.counterfactuals(pipe, X, t, n_customers=5)
    assert len(cfs) > 0
    found = cfs.dropna(subset=["risk_after"])
    assert (found["risk_after"] < t).all()
    effects = explain.offer_effects(pipe, X, t)
    assert set(effects["offer"]) <= set(explain.ACTIONS)


def test_sensitivity_reports_every_scenario(synthetic):
    X, y = synthetic
    pipe = candidate_models()["logistic_regression"][0].fit(X, y)
    table = sensitivity.input_sensitivity(pipe, X, 0.5)
    assert len(table) == 7 and table["decisions_flipped_pct"].between(0, 100).all()

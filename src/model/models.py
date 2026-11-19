"""Candidate models, preprocessing and coarse-to-fine tuning."""
import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.model.config import (BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                              RANDOM_STATE)


def _onehot():
    # Rare channels are grouped as "infrequent" using the training data only (no leakage)
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                         sparse_output=False)


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """Linear models get log-income and scaling; tree models get raw, readable values."""
    if scale:
        numeric = Pipeline([("log_income", FunctionTransformer(_log_income, feature_names_out="one-to-one")),
                            ("scale", StandardScaler())])
    else:
        numeric = "passthrough"
    pre = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                             ("bin", "passthrough", BINARY_FEATURES),
                             ("cat", _onehot(), CATEGORICAL_FEATURES)],
                            verbose_feature_names_out=False)
    return pre.set_output(transform="pandas")


def _log_income(X):
    X = X.copy()
    if "household_income" in X:
        X["household_income"] = np.log1p(X["household_income"].clip(lower=0))
    return X


def _gradient_boosting():
    """LightGBM if installed (laptop); scikit-learn's histogram boosting otherwise."""
    try:
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(objective="binary", class_weight="balanced", subsample_freq=1,
                               random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
        coarse = {"model__num_leaves": randint(8, 64), "model__learning_rate": loguniform(0.02, 0.2),
                  "model__n_estimators": randint(150, 600), "model__min_child_samples": randint(20, 150),
                  "model__subsample": uniform(0.7, 0.3), "model__colsample_bytree": uniform(0.6, 0.4),
                  "model__reg_lambda": loguniform(0.01, 10)}
        return "lightgbm", model, coarse, ["model__num_leaves", "model__learning_rate"]
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        coarse = {"model__max_leaf_nodes": randint(8, 64), "model__learning_rate": loguniform(0.02, 0.2),
                  "model__max_iter": randint(150, 600), "model__min_samples_leaf": randint(20, 150),
                  "model__l2_regularization": loguniform(0.01, 10)}
        return "hist_gradient_boosting", model, coarse, ["model__max_leaf_nodes", "model__learning_rate"]


def candidate_models():
    """name -> (pipeline, coarse search space, n_iter, params refined in the fine stage)."""
    gb_name, gb_model, gb_space, gb_fine = _gradient_boosting()
    return {
        "logistic_regression": (
            Pipeline([("prep", make_preprocessor(scale=True)),
                      ("model", LogisticRegression(class_weight="balanced", max_iter=2000))]),
            {"model__C": loguniform(1e-3, 10)}, 8, ["model__C"]),
        "random_forest": (
            Pipeline([("prep", make_preprocessor(scale=False)),
                      ("model", RandomForestClassifier(class_weight="balanced_subsample",
                                                       random_state=RANDOM_STATE, n_jobs=-1))]),
            {"model__n_estimators": randint(150, 400), "model__max_depth": randint(4, 16),
             "model__min_samples_leaf": randint(5, 60), "model__max_features": uniform(0.2, 0.6)},
            10, ["model__max_depth", "model__min_samples_leaf"]),
        gb_name: (
            Pipeline([("prep", make_preprocessor(scale=False)), ("model", gb_model)]),
            gb_space, 12, gb_fine),
    }


def baseline_models():
    return {"baseline_majority": DummyClassifier(strategy="most_frequent"),
            "baseline_prior": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)}


def rule_baseline_scores(X):
    """Business rule a branch might use today: flag customers already slowing down."""
    score = (X["products_dropped_2m"] > 0).astype(float) + (X["months_active_3m"] < 3).astype(float)
    return score / 2


def fine_grid(best_params: dict, keys: list) -> dict:
    """Small grid around the best coarse values (coarse-to-fine tuning)."""
    grid = {}
    for k, v in best_params.items():
        if k not in keys:
            grid[k] = [v]
        elif isinstance(v, (int, np.integer)):
            grid[k] = sorted({max(2, int(v * 0.7)), int(v), int(v * 1.3) + 1})
        else:
            grid[k] = sorted({float(v) * 0.6, float(v), float(v) * 1.5})
    return grid

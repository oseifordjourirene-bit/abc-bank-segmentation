"""Module 4: fairness metrics match hand calculations and Fairlearn."""
import numpy as np
import pandas as pd
import pytest

from src.model import fairness


def _data():
    n = 1200
    s = pd.DataFrame({"age_band": ["18-24"] * 600 + ["35-44"] * 600,
                      "sex": ["H", "V"] * 600, "foreign_born_flag": ["N"] * 1100 + ["S"] * 100})
    y = np.array([1, 0, 0, 0] * 300)
    pred = np.concatenate([np.array([1, 0, 0, 0] * 150), np.array([1, 1, 0, 0] * 150)])
    return y, pred, s, n


def test_group_rates_and_small_groups():
    y, pred, s, n = _data()
    table = fairness.group_table(y, pred, s)
    young = table[(table.attr_key == "age_band") & (table.group == "18-24")].iloc[0]
    assert young["selection_rate"] == pytest.approx(0.25) and young["recall_tpr"] == pytest.approx(1.0)
    assert young["disparate_impact"] == pytest.approx(0.25 / 0.375, abs=1e-3)
    assert table[(table.attr_key == "foreign_born_flag") & (table.group == "S")]["status"].iloc[0].startswith("Too small")
    summ = fairness.summary(table).set_index("attr_key")
    assert not summ.loc["age_band", "passes"]
    assert summ.loc["age_band", "equalized_odds_difference"] == pytest.approx(1 / 3, abs=1e-3)


def test_matches_fairlearn():
    pytest.importorskip("fairlearn")
    from fairlearn.metrics import equalized_odds_difference
    y, pred, s, _ = _data()
    mine = fairness.summary(fairness.group_table(y, pred, s)).set_index("attr_key")
    theirs = equalized_odds_difference(y, pred, sensitive_features=s["age_band"])
    assert mine.loc["age_band", "equalized_odds_difference"] == pytest.approx(theirs, abs=1e-4)

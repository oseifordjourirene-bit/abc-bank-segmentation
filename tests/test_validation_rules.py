"""Checks that the Great Expectations rules are well formed and catch bad data."""
import pytest

from src.clean import clean
from src.validate import EXPECTATIONS, run_great_expectations


def test_every_rule_names_a_real_great_expectations_class():
    gx = pytest.importorskip("great_expectations")
    for name, _ in EXPECTATIONS:
        assert hasattr(gx.expectations, name), name


def test_suite_passes_clean_data_and_fails_bad_data(raw_factory, monkeypatch, tmp_path):
    pytest.importorskip("great_expectations")
    monkeypatch.setattr("src.validate.ROOT", tmp_path)
    monkeypatch.setattr("src.validate.audit", lambda *a, **k: None)
    rows = [{"ncodpers": str(i)} for i in range(1, 6001)]
    good, _ = clean(raw_factory(rows))
    assert run_great_expectations(good)["success"]

    bad = good.copy()
    bad.loc[0, "tenure_months"] = -5
    assert not run_great_expectations(bad)["success"]

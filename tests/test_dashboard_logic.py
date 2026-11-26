"""Module 5: dashboard logic (runs on the saved models and aggregated report tables)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import logic as L  # noqa: E402

PROTECTED = {"age", "age_band", "sex", "foreign_born_flag", "province_name", "country_of_residence"}


@pytest.mark.parametrize("key", list(L.MODELS))
def test_personas_score_between_0_and_1(key):
    model, meta = L.load_model(key), L.load_meta(key)
    for profile in L.load_personas().values():
        assert 0 <= L.score(model, profile, meta["features"]) <= 1
        assert "Score" in L.plain_explanation(model, profile, meta["features"], meta["threshold"])


def test_no_protected_attribute_in_personas_or_what_if():
    for profile in L.load_personas().values():
        assert not PROTECTED & set(profile)
        for changed in L.what_if_changes(profile).values():
            assert not PROTECTED & set(changed)


def test_fairness_summary_matches_module_4():
    main = L.fairness_summary(L.load_table("main", "fairness_by_group")).set_index("Attribute")
    assert main.loc["Age band", "Result"] == "Fails"
    assert main.loc["Age band", "Equal-treatment gap"] == pytest.approx(0.1615, abs=0.0005)
    assert main.loc["Sex", "Result"] == "Passes"
    inactive = L.fairness_summary(L.load_table("inactive", "fairness_by_group")).set_index("Attribute")
    assert inactive.loc["Sex", "Result"] == "Fails"

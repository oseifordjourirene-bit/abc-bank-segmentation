"""Module 4: the modelling table must not leak future information."""
import pandas as pd
import pytest

from src.anonymise import anonymise
from src.clean import clean
from src.model.config import FEATURES, SENSITIVE
from src.model.dataset import build_features, build_modelling_table, build_target

MONTHS = ["2015-12-28", "2016-01-28", "2016-02-28", "2016-03-28", "2016-04-28", "2016-05-28"]


def _monthly(raw_factory, customers):
    """customers: {id: [(active, products) per month]} -> anonymised monthly + audit tables."""
    rows = []
    for cid, history in customers.items():
        for month, (active, products) in zip(MONTHS, history):
            row = {"ncodpers": str(cid), "fecha_dato": month, "ind_actividad_cliente": str(active)}
            row.update({p: "1" for p in products})
            rows.append(row)
    clean_df, _ = clean(raw_factory(rows))
    analytics, audit_table, _ = anonymise(clean_df)
    return analytics, audit_table


CA, CARD = "ind_cco_fin_ult1", "ind_tjcr_fin_ult1"


@pytest.fixture
def data(raw_factory):
    return _monthly(raw_factory, {
        1: [(1, [CA])] * 6,                                        # stays active: 0
        2: [(1, [CA])] * 3 + [(0, [CA])] * 3,                      # goes inactive: 1
        3: [(1, [CA, CARD])] * 3 + [(1, [CA])] * 3,                # drops card: 1 (inactive_or_dropping)
        4: [(0, [CA])] * 6,                                        # inactive at cutoff: excluded
    })


def test_targets_follow_the_agreed_rules(data):
    monthly, _ = data
    both = build_target(monthly, "inactive_or_dropping").sort_index()
    inactive = build_target(monthly, "inactive").sort_index()
    assert both.sum() == 2 and inactive.sum() == 1
    with pytest.raises(ValueError):
        build_target(monthly, "made_up_rule")


def test_population_is_active_at_cutoff_and_protected_attributes_are_not_features(data):
    monthly, audit_table = data
    X, y, s, report = build_modelling_table(monthly, audit_table)
    assert len(X) == 3 and report["modelling_rows"] == 3
    assert list(X.columns) == FEATURES
    for col in list(SENSITIVE) + ["age", "province_name", "customer_key"]:
        assert col not in X.columns
    assert list(s.columns) == list(SENSITIVE)


def test_features_ignore_months_after_the_cutoff(data):
    monthly, _ = data
    before = build_features(monthly)
    changed = monthly.copy()
    after_cutoff = changed["snapshot_date"] > pd.Timestamp("2016-02-28")
    changed.loc[after_cutoff, "active_customer_flag"] = 0
    changed.loc[after_cutoff, "credit_card"] = 1
    pd.testing.assert_frame_equal(before, build_features(changed))

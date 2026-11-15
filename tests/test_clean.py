import pandas as pd

from src.clean import clean


def test_columns_renamed_and_types_fixed(raw_factory):
    df, _ = clean(raw_factory([{"ncodpers": "1"}]))
    assert "customer_id" in df.columns and "ncodpers" not in df.columns
    assert df.loc[0, "age"] == 35            # ' 35' stripped and converted
    assert df.loc[0, "tenure_months"] == 60  # '     60' stripped and converted
    assert df.loc[0, "customer_type_start_month"] == "1"  # '1.0' standardised


def test_filters_remove_staff_deceased_and_impossible_ages_but_keep_minors(raw_factory):
    raw = raw_factory([
        {"ncodpers": "1"},
        {"ncodpers": "2", "ind_empleado": "A"},   # bank employee
        {"ncodpers": "3", "indfall": "S"},        # deceased
        {"ncodpers": "4", "age": " 116"},         # age over 100
        {"ncodpers": "5", "age": " 12"},          # minor: kept (junior account)
    ])
    df, report = clean(raw)
    assert sorted(df["customer_id"].tolist()) == [1, 5]
    assert report["removed"] == {"bank_employees": 1, "deceased": 1, "age_over_100": 1,
                                 "missing_id_or_date": 0}


def test_duplicates_removed(raw_factory):
    df, report = clean(raw_factory([{"ncodpers": "1"}, {"ncodpers": "1"}]))
    assert len(df) == 1 and report["duplicates_removed"] == 1


def test_missing_income_filled_with_overall_median_and_flagged(raw_factory):
    raw = raw_factory([{"ncodpers": str(i), "renta": v}
                       for i, v in enumerate(["10", "20", "30", None], start=1)])
    df, report = clean(raw)
    filled = df[df["customer_id"] == 4].iloc[0]
    assert filled["income_was_missing"] == 1
    assert filled["household_income"] == report["income"]["income_median"]
    assert df["household_income"].notna().all()


def test_unused_columns_dropped(raw_factory):
    df, _ = clean(raw_factory([{"ncodpers": "1"}]))
    for col in ["spouse_of_employee", "last_primary_date", "address_type", "resident_flag"]:
        assert col not in df.columns

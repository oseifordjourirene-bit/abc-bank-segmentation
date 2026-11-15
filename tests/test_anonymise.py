import pandas as pd
import pytest

from src.anonymise import anonymise, pseudonymise_id, remove_pii
from src.clean import clean


def test_pseudonymised_key_is_stable_and_depends_on_secret():
    assert pseudonymise_id(123, b"a") == pseudonymise_id(123, b"a")
    assert pseudonymise_id(123, b"a") != pseudonymise_id(123, b"b")
    assert "123" not in pseudonymise_id(123, b"a")


def test_analytics_table_has_no_identifiers_or_protected_attributes(raw_factory):
    clean_df, _ = clean(raw_factory([{"ncodpers": str(i)} for i in range(1, 11)]))
    analytics, audit_table, report = anonymise(clean_df)
    for col in ["customer_id", "age", "sex", "province_name", "join_date", "foreign_born_flag"]:
        assert col not in analytics.columns
    assert "customer_key" in analytics.columns
    assert "age" not in audit_table.columns          # exact age never stored
    assert report["k_anonymity_after"] >= 5


def test_pii_columns_removed_with_faker_data():
    faker = pytest.importorskip("faker")
    fake = faker.Faker("en_GB")
    df = pd.DataFrame({"name": [fake.name() for _ in range(5)],
                       "email": [fake.email() for _ in range(5)],
                       "phone": [fake.phone_number() for _ in range(5)],
                       "product_count": range(5)})
    cleaned, removed = remove_pii(df)
    assert set(removed) == {"name", "email", "phone"}
    assert list(cleaned.columns) == ["product_count"]


def test_salt_required(raw_factory, monkeypatch):
    clean_df, _ = clean(raw_factory([{"ncodpers": "1"}]))
    monkeypatch.delenv("ANON_SALT")
    with pytest.raises(RuntimeError):
        anonymise(clean_df)

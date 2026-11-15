from src.anonymise import anonymise
from src.clean import clean
from src.features import build_features


def _history(raw_factory, months):
    rows = []
    for month, products, active in months:
        row = {"ncodpers": "1", "fecha_dato": month, "ind_actividad_cliente": active}
        row.update({p: "1" for p in products})
        rows.append(row)
    clean_df, _ = clean(raw_factory(rows))
    analytics, _, _ = anonymise(clean_df)
    return build_features(analytics).iloc[0]


def test_product_count_and_changes(raw_factory):
    f = _history(raw_factory, [
        ("2016-02-28", ["ind_cco_fin_ult1", "ind_tjcr_fin_ult1"], "1"),
        ("2016-03-28", ["ind_cco_fin_ult1"], "1"),
        ("2016-04-28", ["ind_cco_fin_ult1"], "1"),
        ("2016-05-28", ["ind_cco_fin_ult1", "ind_nomina_ult1"], "1"),
    ])
    assert f["product_count"] == 2
    assert f["products_added_3m"] == 1      # salary credited added
    assert f["products_dropped_3m"] == 1    # credit card dropped
    assert f["has_salary"] == 1 and f["has_current_account"] == 1


def test_drift_signal_when_customer_goes_inactive(raw_factory):
    f = _history(raw_factory, [
        ("2016-02-28", ["ind_cco_fin_ult1"], "1"),
        ("2016-03-28", ["ind_cco_fin_ult1"], "1"),
        ("2016-04-28", ["ind_cco_fin_ult1"], "0"),
        ("2016-05-28", ["ind_cco_fin_ult1"], "0"),
    ])
    assert f["active_now"] == 0 and f["drift_signal"] == 1
    assert f["months_active_6m"] == 2

import json

from src.anonymise import anonymise
from src.bias import representation_report
from src.clean import clean
from src.features import build_features


def test_every_stage_writes_an_audit_entry(raw_factory, isolated_audit_log):
    clean_df, _ = clean(raw_factory([{"ncodpers": str(i)} for i in range(1, 6)]))
    analytics, _, _ = anonymise(clean_df)
    build_features(analytics)
    entries = [json.loads(line) for line in isolated_audit_log.read_text().splitlines()]
    assert [e["dataset"] for e in entries] == ["clean_monthly", "anonymised_analytics",
                                               "customer_features"]
    assert all(e["user"] and e["run_id"] and e["timestamp"] for e in entries)


def test_small_groups_are_flagged_not_compared(raw_factory, tmp_path):
    rows = [{"ncodpers": str(i), "sexo": "H" if i <= 600 else "V"} for i in range(1, 651)]
    clean_df, _ = clean(raw_factory(rows))
    analytics, audit_table, _ = anonymise(clean_df)
    features = build_features(analytics)
    report, _ = representation_report(features, audit_table, require_fairlearn=False,
                                      out_dir=tmp_path)
    sex = report[report["attribute"].str.startswith("Sex")].set_index("group")
    assert sex.loc["H", "status"] == "OK"
    assert sex.loc["V", "status"].startswith("Too small")
    assert (tmp_path / "representation_chart.png").exists()

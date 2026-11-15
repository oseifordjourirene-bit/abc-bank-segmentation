"""Prefect orchestration DAG for the ABC Bank Segmentation data pipeline.

ingest -> clean -> anonymise -> features -> validate -> bias check -> publish

Run from the repository root:
    python -m flows.pipeline_flow
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

from src.anonymise import anonymise  # noqa: E402
from src.audit import RUN_ID, audit  # noqa: E402
from src.bias import representation_report  # noqa: E402
from src.clean import clean  # noqa: E402
from src.config import PROCESSED_DIR, RAW_FILE, REPORTS_DIR  # noqa: E402
from src.features import build_features  # noqa: E402
from src.ingest import ingest  # noqa: E402
from src.validate import run_great_expectations  # noqa: E402


@task(name="1. Ingest raw data", retries=2, retry_delay_seconds=10)
def ingest_task(path: str) -> pd.DataFrame:
    return ingest(Path(path))


@task(name="2. Clean and standardise")
def clean_task(raw: pd.DataFrame):
    return clean(raw)


@task(name="3. Validate with Great Expectations")
def validate_task(clean_df: pd.DataFrame) -> dict:
    summary = run_great_expectations(clean_df)
    if not summary["success"]:
        raise ValueError(f"Validation failed: {summary['failed']} expectation(s); "
                         f"see {summary['data_docs']}")
    return summary


@task(name="4. Anonymise (PII removal, pseudonymised keys)")
def anonymise_task(clean_df: pd.DataFrame):
    return anonymise(clean_df)


@task(name="5. Build customer features")
def features_task(analytics: pd.DataFrame) -> pd.DataFrame:
    return build_features(analytics)


@task(name="6. Bias detection (Fairlearn)")
def bias_task(features: pd.DataFrame, audit_table: pd.DataFrame) -> dict:
    _, summary = representation_report(features, audit_table, require_fairlearn=True)
    return summary


@task(name="7. Publish outputs")
def publish_task(features: pd.DataFrame, analytics: pd.DataFrame, audit_table: pd.DataFrame,
                 lineage: dict) -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {"customer_features": features, "clean_monthly_anonymised": analytics,
               "restricted_audit_attributes": audit_table}
    paths = {}
    for name, frame in outputs.items():
        path = PROCESSED_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        audit("write", name, rows_out=len(frame), columns=frame.columns, path=path)
        paths[name] = str(path)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "lineage.json").write_text(json.dumps(lineage, indent=2, default=str))
    return paths


@flow(name="abc-bank-segmentation-pipeline", log_prints=True)
def abc_bank_pipeline(raw_path: str = str(RAW_FILE)) -> dict:
    logger = get_run_logger()
    raw = ingest_task(raw_path)
    clean_df, clean_report = clean_task(raw)
    validation = validate_task(clean_df)
    analytics, audit_table, anon_report = anonymise_task(clean_df)
    features = features_task(analytics)
    bias = bias_task(features, audit_table)
    lineage = {"run_id": RUN_ID, "raw_rows": len(raw), "cleaning": clean_report,
               "validation": validation, "anonymisation": anon_report,
               "feature_rows": len(features), "bias": bias}
    paths = publish_task(features, analytics, audit_table, lineage)
    logger.info("Pipeline finished: %s customers ready for Module 4", len(features))
    return {"lineage": lineage, "outputs": paths}


if __name__ == "__main__":
    abc_bank_pipeline()

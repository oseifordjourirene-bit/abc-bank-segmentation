"""Stage 5 - Validation: Great Expectations suite on the clean monthly table.

The rules live in EXPECTATIONS (one source of truth). Great Expectations runs them and
builds Data Docs (HTML) at gx/uncommitted/data_docs/local_site/index.html.
"""
import pandas as pd

from src.audit import audit
from src.config import MAX_AGE, PRODUCT_COLUMNS, ROOT

SUITE_NAME = "clean_monthly_suite"

EXPECTATIONS = (
    [("ExpectTableRowCountToBeBetween", {"min_value": 5000}),
     ("ExpectCompoundColumnsToBeUnique", {"column_list": ["customer_id", "snapshot_date"]})]
    + [("ExpectColumnValuesToNotBeNull", {"column": c}) for c in
       ["customer_id", "snapshot_date", "household_income", "tenure_months",
        "active_customer_flag", "age"]]
    + [("ExpectColumnValuesToBeBetween", {"column": "age", "min_value": 0, "max_value": MAX_AGE}),
       ("ExpectColumnValuesToBeBetween", {"column": "tenure_months", "min_value": 0, "max_value": 400}),
       ("ExpectColumnValuesToBeBetween", {"column": "household_income", "min_value": 0}),
       ("ExpectColumnValuesToBeInSet", {"column": "sex", "value_set": ["H", "V", "UNKNOWN"]}),
       ("ExpectColumnValuesToBeInSet", {"column": "relation_type_start_month",
                                        "value_set": ["A", "I", "P", "R", "UNKNOWN"]}),
       ("ExpectColumnValuesToBeInSet", {"column": "active_customer_flag", "value_set": [0, 1]}),
       ("ExpectColumnValuesToBeInSet", {"column": "income_was_missing", "value_set": [0, 1]})]
    + [("ExpectColumnValuesToBeInSet", {"column": c, "value_set": [0, 1]}) for c in PRODUCT_COLUMNS]
)


def run_great_expectations(df: pd.DataFrame) -> dict:
    import great_expectations as gx

    context = gx.get_context(mode="file", project_root_dir=str(ROOT))
    source = context.data_sources.add_or_update_pandas(name="abc_bank_pipeline")
    asset = source.add_dataframe_asset(name="clean_monthly")
    batch_definition = asset.add_batch_definition_whole_dataframe("full_table")

    suite = gx.ExpectationSuite(
        name=SUITE_NAME,
        expectations=[getattr(gx.expectations, name)(**kwargs) for name, kwargs in EXPECTATIONS])
    suite = context.suites.add_or_update(suite)

    validation = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="clean_monthly_validation", data=batch_definition, suite=suite))
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(name="pipeline_checkpoint", validation_definitions=[validation],
                      actions=[gx.checkpoint.UpdateDataDocsAction(name="update_data_docs")],
                      result_format={"result_format": "SUMMARY"}))

    result = checkpoint.run(batch_parameters={"dataframe": df})
    run = list(result.run_results.values())[0]
    stats = run.statistics
    summary = {"success": bool(result.success),
               "evaluated": int(stats["evaluated_expectations"]),
               "passed": int(stats["successful_expectations"]),
               "failed": int(stats["unsuccessful_expectations"]),
               "data_docs": str(ROOT / "gx" / "uncommitted" / "data_docs" / "local_site" / "index.html")}
    audit("validate", "clean_monthly", rows_in=len(df), details=summary)
    return summary

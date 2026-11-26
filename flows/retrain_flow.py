"""Monthly MLOps flow: monitor, then retrain only when needed (Prefect).

    python -m flows.retrain_flow                 # one run now
Scheduling (bank environment): a Prefect deployment with cron "0 6 3 * *" (03rd of each month,
after the month-end extract and the Module 3 pipeline have run).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prefect import flow, task  # noqa: E402

from src.model.dataset import load_inputs  # noqa: E402
from src.monitoring import monitor  # noqa: E402


@task(name="Monitor drift, predictions and fairness")
def monitor_task():
    monthly, audit_table = load_inputs()
    tables = monitor.run(monthly, audit_table)
    real = tables["alerts"][tables["alerts"]["month"] != monitor.STRESS_LABEL]
    return real.to_dict("records")


@task(name="Retrain with promotion gates")
def retrain_task():
    from src.model import retrain
    monthly, audit_table = load_inputs()
    return [retrain.retrain(monthly, audit_table, rule) for rule in retrain.RULES]


@flow(name="abc-bank-monthly-monitoring", log_prints=True)
def monthly_mlops(force_retrain: bool = False, quarter_end: bool = False):
    alerts = monitor_task()
    drift = [a for a in alerts if a["type"] in ("data drift", "prediction drift")]
    fairness = [a for a in alerts if a["type"] == "fairness"]
    if fairness:
        print(f"{len(fairness)} fairness alert(s): open an incident for Digital Governance and Risk "
              "(docs/model_governance.md, incident response).")
    if drift or force_retrain or quarter_end:
        reports = retrain_task()
        for r in reports:
            print(f"{r['target_rule']}: {r['decision']['stage']} {r['decision']['failed_gates']}")
    else:
        print("No drift alert: model kept, no retraining this month.")


if __name__ == "__main__":
    monthly_mlops()

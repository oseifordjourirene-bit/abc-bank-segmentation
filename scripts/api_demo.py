"""API demo for the stakeholder deck: score three fictional customers through the FastAPI /predict endpoint.

1. Start the API in one terminal:   uvicorn api.main:app --reload
2. Run this in another terminal:     python scripts/api_demo.py
The output is printed and saved to reports/api_demo_output.md.
"""
import json
from pathlib import Path

import httpx

URL = "http://127.0.0.1:8000"
ROOT = Path(__file__).resolve().parents[1]
personas = json.loads((ROOT / "dashboard" / "personas.json").read_text(encoding="utf-8"))

lines = ["# API demo: POST /predict", "", f"Endpoint: {URL}/predict", "",
         "| Fictional customer | HTTP | Score | Flagged for offer | Model |", "| --- | --- | --- | --- | --- |"]
with httpx.Client(base_url=URL, timeout=10) as client:
    health = client.get("/health").json()
    for name, profile in personas.items():
        r = client.post("/predict", json=profile)
        body = r.json()
        lines.append(f"| {name} | {r.status_code} | {body['drift_probability']:.3f} | "
                     f"{'Yes' if body['flag_for_reactivation_offer'] else 'No'} | "
                     f"{body['model']} v{body['model_version']} |")
    # The API must refuse protected attributes
    bad = {**next(iter(personas.values())), "age": 30}
    r = client.post("/predict", json=bad)
    lines += ["", f"Request with `age` added: HTTP {r.status_code} ({r.json()['detail'][0]['msg']})",
              "", f"Health check: {health}"]

text = "\n".join(lines)
print(text)
(ROOT / "reports" / "api_demo_output.md").write_text(text + "\n", encoding="utf-8")

# API demo: POST /predict

Endpoint: http://127.0.0.1:8000/predict

| Fictional customer | HTTP | Score | Flagged for offer | Model |
| --- | --- | --- | --- | --- |
| Profile A: just added products, fully active | 200 | 0.849 | Yes | random_forest v1 |
| Profile B: one account, quiet until February | 200 | 0.780 | Yes | random_forest v1 |
| Profile C: long-standing salary customer | 200 | 0.606 | Yes | random_forest v1 |

Request with `age` added: HTTP 422 (Extra inputs are not permitted)

Health check: {'status': 'ok', 'model': 'random_forest', 'version': 1, 'trained_at': '2026-11-19T03:41:12+00:00'}

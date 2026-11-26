"""Explanation store: every prediction is saved with its explanation, for audit.

One SQLite row per /predict call: when it was made, by which model version, the inputs
(behavioural features only: the API refuses protected attributes and takes no customer ID),
the score, the decision and the top drivers of that score. Internal Control can retrieve
any decision by its request_id and see why it was made.

Location: the EXPLANATION_DB environment variable (default logs/explanations.db).
On the free Render demo the file resets when the service restarts or sleeps; in production the
same table would live in the bank's governed database with a 7-year retention rule.
"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = """CREATE TABLE IF NOT EXISTS explanations (
    request_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model TEXT, model_version TEXT,
    threshold REAL, score REAL, flagged INTEGER, contact_tier TEXT, method TEXT,
    inputs TEXT, top_factors TEXT)"""


def db_path() -> Path:
    return Path(os.environ.get("EXPLANATION_DB", ROOT / "logs" / "explanations.db"))


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(SCHEMA)
    return con


def save(record: dict) -> str:
    request_id = record.get("request_id") or str(uuid.uuid4())
    row = {**record, "request_id": request_id,
           "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "inputs": json.dumps(record["inputs"]), "top_factors": json.dumps(record["top_factors"]),
           "flagged": int(record["flagged"])}
    cols = ["request_id", "created_at", "model", "model_version", "threshold", "score", "flagged",
            "contact_tier", "method", "inputs", "top_factors"]
    with _connect() as con:
        con.execute(f"INSERT INTO explanations ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                    [row.get(c) for c in cols])
    return request_id


def _decode(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["inputs"], d["top_factors"] = json.loads(d["inputs"]), json.loads(d["top_factors"])
    d["flagged"] = bool(d["flagged"])
    return d


def get(request_id: str) -> dict | None:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM explanations WHERE request_id = ?", (request_id,)).fetchone()
    return _decode(row) if row else None


def recent(limit: int = 20) -> list[dict]:
    with _connect() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM explanations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_decode(r) for r in rows]

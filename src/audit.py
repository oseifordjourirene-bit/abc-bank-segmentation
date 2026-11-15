"""Privacy audit logging: one JSON line for every data access and transformation."""
import getpass
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import AUDIT_LOG

RUN_ID = uuid.uuid4().hex[:12]


def _get_logger(log_path: Path = AUDIT_LOG) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"abc_bank.audit.{log_path}")
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def file_hash(path: Path) -> str:
    """SHA-256 of a file, so any later change to a dataset can be detected."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(action: str, dataset: str, rows_in=None, rows_out=None, columns=None,
          details=None, path: Path = None, log_path: Path = AUDIT_LOG) -> dict:
    """Record who touched which dataset, when, and what changed.

    action: read | write | transform | validate | bias_check
    """
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "user": getpass.getuser(),
        "action": action,
        "dataset": dataset,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "columns": len(columns) if columns is not None else None,
        "sha256": file_hash(path) if path is not None and Path(path).exists() else None,
        "details": details or {},
    }
    _get_logger(log_path).info(json.dumps(entry, default=str))
    return entry

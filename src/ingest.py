"""Stage 1 - Ingestion: read the raw Santander snapshot file exactly as delivered."""
from pathlib import Path

import pandas as pd

from src.audit import audit
from src.config import COLUMN_MAP, RAW_FILE


def ingest(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read every column as text so no value is silently altered on load."""
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing = set(COLUMN_MAP) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {sorted(missing)}")
    audit("read", "raw_santander_sample", rows_out=len(df), columns=df.columns,
          path=path, details={"file": Path(path).name})
    return df

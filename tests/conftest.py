import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLUMN_MAP  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_audit_log(tmp_path, monkeypatch):
    """Send audit entries from tests to a temporary file, not the real log."""
    import src.audit as audit_module
    original = audit_module.audit

    def _audit(*args, **kwargs):
        kwargs["log_path"] = tmp_path / "audit_test.jsonl"
        return original(*args, **kwargs)

    for module in ["src.ingest", "src.clean", "src.anonymise", "src.features", "src.bias",
                   "src.model.dataset"]:
        monkeypatch.setattr(f"{module}.audit", _audit, raising=False)
    monkeypatch.setenv("ANON_SALT", "test-salt")
    yield tmp_path / "audit_test.jsonl"


def make_raw(rows):
    """Build a raw-format (Spanish columns, all text) frame from partial row dicts."""
    base = {k: "0" for k in COLUMN_MAP}
    base.update({"fecha_dato": "2016-05-28", "ind_empleado": "N", "pais_residencia": "ES",
                 "sexo": "H", "age": " 35", "fecha_alta": "2010-01-01", "ind_nuevo": "0",
                 "antiguedad": "     60", "indrel": "1", "ult_fec_cli_1t": None,
                 "indrel_1mes": "1.0", "tiprel_1mes": "A", "indresi": "S", "indext": "N",
                 "conyuemp": None, "canal_entrada": "KHE", "indfall": "N", "tipodom": "1",
                 "cod_prov": "28", "nomprov": "MADRID", "ind_actividad_cliente": " 1",
                 "renta": "50000.0", "segmento": "02 - PARTICULARES"})
    return pd.DataFrame([{**base, **r} for r in rows], dtype="string")


@pytest.fixture
def raw_factory():
    return make_raw

"""Rutas y parámetros compartidos del proyecto."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "alumnos.csv"
MODEL_PATH = PROJECT_ROOT / "modelo_riesgo.pkl"

FEATURE_COLUMNS = ("asistencia", "promedio", "horas_estudio")
TARGET_COLUMN = "riesgo"
REQUIRED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)
EXPECTED_DATASET_RECORDS = 20

FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "asistencia": (0.0, 100.0),
    "promedio": (0.0, 10.0),
    "horas_estudio": (0.0, 30.0),
}

RANDOM_FOREST_TREES = 100
RANDOM_STATE = 42
TEST_SIZE = 0.2


def resolve_groq_api_key(
    streamlit_secrets: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Obtiene la clave del entorno o de los secretos de Streamlit."""
    environment = environ if environ is not None else os.environ

    secret_candidate = None
    if streamlit_secrets is not None:
        try:
            secret_candidate = streamlit_secrets.get("GROQ_API_KEY")
        except Exception:
            # Streamlit lanza una excepción si no existe secrets.toml.
            secret_candidate = None

    for candidate in (environment.get("GROQ_API_KEY"), secret_candidate):
        if candidate is None:
            continue

        value = str(candidate).strip()
        if value and value != "COLOCA_AQUI_TU_CLAVE":
            return value

    return None
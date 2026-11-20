"""Lectura, normalización y validación del dataset académico."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pandas as pd

from src.config import FEATURE_RANGES, REQUIRED_COLUMNS, TARGET_COLUMN


class DataValidationError(ValueError):
    """Indica que el archivo CSV no satisface el esquema del laboratorio."""


def _read_single_column_fallback(path: Path) -> pd.DataFrame:
    """Repara archivos donde cada línea CSV quedó agrupada en una columna."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise DataValidationError("No fue posible leer el archivo CSV.") from exc

    rows = list(csv.reader(io.StringIO(text)))
    if not rows or not rows[0] or "," not in rows[0][0]:
        raise DataValidationError("El archivo CSV está vacío o tiene un formato inválido.")

    split_rows = [row[0].split(",") if len(row) == 1 else row for row in rows]
    headers = [cell.strip().strip('"').strip("'") for cell in split_rows[0]]
    if not headers or any(not header for header in headers):
        raise DataValidationError("El encabezado del archivo CSV está incompleto.")

    normalized_rows: list[list[str]] = []
    for row in split_rows[1:]:
        if len(row) != len(headers):
            raise DataValidationError("El archivo CSV contiene filas con columnas inconsistentes.")
        normalized_rows.append([cell.strip() for cell in row])
    return pd.DataFrame(normalized_rows, columns=headers)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("El archivo CSV está vacío.") from exc
    except pd.errors.ParserError:
        return _read_single_column_fallback(path)
    except (OSError, UnicodeError) as exc:
        raise DataValidationError("No fue posible leer el archivo CSV.") from exc

    if len(frame.columns) == 1 and "," in str(frame.columns[0]):
        return _read_single_column_fallback(path)
    return frame


def load_dataset(
    path: str | Path,
    expected_rows: int | None = None,
) -> pd.DataFrame:
    """Carga el CSV, elimina filas incompletas y valida datos y etiquetas."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No se encontró el dataset: {csv_path}")

    frame = _read_csv(csv_path)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if len(set(frame.columns)) != len(frame.columns):
        raise DataValidationError("El CSV contiene nombres de columnas duplicados.")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise DataValidationError(
            "Faltan columnas requeridas: " + ", ".join(missing_columns) + "."
        )
    if expected_rows is not None and len(frame) != expected_rows:
        raise DataValidationError(
            f"Se esperaban {expected_rows} registros originales; se encontraron {len(frame)}."
        )

    source_values = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    numeric_values = source_values.apply(pd.to_numeric, errors="coerce")
    invalid_values = source_values.notna() & numeric_values.isna()
    if invalid_values.any().any():
        invalid_columns = [
            column for column in REQUIRED_COLUMNS if invalid_values[column].any()
        ]
        raise DataValidationError(
            "Hay datos no numéricos en: " + ", ".join(invalid_columns) + "."
        )

    prepared = numeric_values.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    if prepared.empty:
        raise DataValidationError("No hay registros completos para entrenar.")

    for column, (minimum, maximum) in FEATURE_RANGES.items():
        if not prepared[column].between(minimum, maximum).all():
            raise DataValidationError(
                f"La columna {column} debe contener valores entre {minimum:g} y {maximum:g}."
            )

    labels = set(prepared[TARGET_COLUMN].unique().tolist())
    if not labels.issubset({0, 1}):
        raise DataValidationError("La columna riesgo solo admite las etiquetas 0 y 1.")
    if labels != {0, 1}:
        raise DataValidationError("El dataset debe incluir ejemplos de ambas clases de riesgo.")
    return prepared

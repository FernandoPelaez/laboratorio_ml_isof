"""Entrenamiento y serialización del clasificador Random Forest."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    MODEL_PATH,
    RANDOM_FOREST_TREES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


class TrainingError(ValueError):
    """Indica que el dataset no puede usarse para entrenar el modelo."""


@dataclass(frozen=True)
class TrainingReport:
    total_samples: int
    train_samples: int
    test_samples: int
    accuracy: float
    model_path: Path


def train_and_save(
    frame: pd.DataFrame,
    model_path: str | Path = MODEL_PATH,
) -> TrainingReport:
    """Divide los datos, ajusta el modelo y guarda el artefacto de forma segura."""
    required = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise TrainingError("Faltan columnas para entrenar: " + ", ".join(missing) + ".")
    if len(frame) < 5:
        raise TrainingError("Se requieren al menos cinco registros para dividir los datos.")
    if set(frame[TARGET_COLUMN].unique().tolist()) != {0, 1}:
        raise TrainingError("El entrenamiento requiere las clases 0 y 1.")

    features = frame.loc[:, list(FEATURE_COLUMNS)]
    target = frame[TARGET_COLUMN]
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    except ValueError as exc:
        raise TrainingError("No fue posible dividir el dataset para entrenamiento.") from exc

    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_TREES,
        random_state=RANDOM_STATE,
    )
    try:
        model.fit(x_train, y_train)
        accuracy = float(accuracy_score(y_test, model.predict(x_test)))
    except (ValueError, TypeError) as exc:
        raise TrainingError("El modelo no pudo completar el entrenamiento.") from exc

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.stem}_",
            suffix=".pkl",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        joblib.dump(model, temporary_path)
        os.replace(temporary_path, destination)
    except OSError as exc:
        raise TrainingError("No fue posible guardar el modelo entrenado.") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return TrainingReport(
        total_samples=len(frame),
        train_samples=len(x_train),
        test_samples=len(x_test),
        accuracy=accuracy,
        model_path=destination,
    )

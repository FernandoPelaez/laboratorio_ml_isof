"""Carga del modelo persistido y predicción de indicadores estudiantiles."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURE_RANGES, MODEL_PATH


class PredictionError(ValueError):
    """Indica una entrada inválida o un error al ejecutar la inferencia."""


@dataclass(frozen=True)
class PredictionResult:
    risk_class: int
    label: str
    low_risk_probability: float
    high_risk_probability: float
    predicted_class_probability: float


def load_model(path: str | Path = MODEL_PATH) -> Any:
    """Carga únicamente el artefacto local generado por este proyecto."""
    model_path = Path(path)
    if not model_path.is_file():
        raise PredictionError("No se encontró el modelo. Ejecuta python entrenar.py.")
    try:
        return joblib.load(model_path)
    except Exception as exc:
        raise PredictionError("No se pudo cargar el modelo entrenado.") from exc


def _validate_indicators(indicators: dict[str, object]) -> dict[str, float]:
    missing = [name for name in FEATURE_COLUMNS if name not in indicators]
    if missing:
        raise PredictionError("Faltan indicadores: " + ", ".join(missing) + ".")

    values: dict[str, float] = {}
    for name in FEATURE_COLUMNS:
        try:
            value = float(indicators[name])
        except (TypeError, ValueError) as exc:
            raise PredictionError(f"El valor de {name} debe ser numérico.") from exc
        if not math.isfinite(value):
            raise PredictionError(f"El valor de {name} debe ser finito.")
        minimum, maximum = FEATURE_RANGES[name]
        if not minimum <= value <= maximum:
            raise PredictionError(
                f"El valor de {name} debe estar entre {minimum:g} y {maximum:g}."
            )
        values[name] = value
    return values


def predict_student(
    model: Any,
    indicators: dict[str, object],
) -> PredictionResult:
    """Predice la clase y asigna probabilidades mediante model.classes_."""
    values = _validate_indicators(indicators)
    input_frame = pd.DataFrame([values], columns=list(FEATURE_COLUMNS))
    try:
        predicted = int(model.predict(input_frame)[0])
        classes = [int(class_value) for class_value in model.classes_]
        probabilities = model.predict_proba(input_frame)[0]
    except Exception as exc:
        raise PredictionError("No fue posible calcular el diagnóstico.") from exc

    if set(classes) != {0, 1} or predicted not in classes:
        raise PredictionError("El modelo no contiene las dos clases de riesgo esperadas.")
    if len(probabilities) != len(classes):
        raise PredictionError("El modelo devolvió probabilidades incompletas.")

    probability_by_class = {
        class_value: float(probabilities[index])
        for index, class_value in enumerate(classes)
    }
    low_probability = probability_by_class[0]
    high_probability = probability_by_class[1]
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in (low_probability, high_probability)
    ):
        raise PredictionError("El modelo devolvió probabilidades inválidas.")

    return PredictionResult(
        risk_class=predicted,
        label="Riesgo alto" if predicted == 1 else "Bajo riesgo",
        low_risk_probability=low_probability,
        high_risk_probability=high_probability,
        predicted_class_probability=probability_by_class[predicted],
    )

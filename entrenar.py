"""Punto de entrada para entrenar y guardar el modelo del laboratorio."""

from __future__ import annotations

import sys

from src.config import DATASET_PATH, EXPECTED_DATASET_RECORDS, MODEL_PATH
from src.data.csv_loader import DataValidationError, load_dataset
from src.ml.trainer import TrainingError, train_and_save


def main() -> int:
    print(f"1. Cargando dataset: {DATASET_PATH.name}")
    try:
        dataset = load_dataset(DATASET_PATH, expected_rows=EXPECTED_DATASET_RECORDS)
        print(f"   Dataset válido: {len(dataset)} registros y cuatro columnas requeridas.")
        print("2. Entrenando RandomForestClassifier...")
        report = train_and_save(dataset, MODEL_PATH)
    except (DataValidationError, TrainingError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(
        f"3. Entrenamiento finalizado: {report.train_samples} registros de entrenamiento, "
        f"{report.test_samples} de prueba."
    )
    print(f"   Exactitud de prueba (descriptiva): {report.accuracy:.2f}")
    print(
        f"4. Modelo guardado en {report.model_path.name} "
        f"({report.model_path.stat().st_size} bytes)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

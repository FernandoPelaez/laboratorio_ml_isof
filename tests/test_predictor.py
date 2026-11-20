"""Pruebas de carga, validación y clasificación."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.config import DATASET_PATH
from src.data.csv_loader import load_dataset
from src.ml.predictor import PredictionError, load_model, predict_student
from src.ml.trainer import train_and_save


class PredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.model_path = Path(self.temporary_directory.name) / "model.pkl"
        train_and_save(load_dataset(DATASET_PATH), self.model_path)
        self.model = load_model(self.model_path)
        self.indicators = {
            "asistencia": 85.5,
            "promedio": 8.2,
            "horas_estudio": 12.0,
        }

    def test_real_model_returns_valid_class_and_probabilities(self) -> None:
        result = predict_student(self.model, self.indicators)
        self.assertIn(result.risk_class, (0, 1))
        self.assertGreaterEqual(result.low_risk_probability, 0.0)
        self.assertLessEqual(result.low_risk_probability, 1.0)
        self.assertGreaterEqual(result.high_risk_probability, 0.0)
        self.assertLessEqual(result.high_risk_probability, 1.0)
        self.assertAlmostEqual(
            result.low_risk_probability + result.high_risk_probability, 1.0
        )

    def test_probability_lookup_uses_classes_order(self) -> None:
        reversed_model = SimpleNamespace(
            classes_=[1, 0],
            predict=lambda frame: [1],
            predict_proba=lambda frame: [[0.8, 0.2]],
        )
        result = predict_student(reversed_model, self.indicators)
        self.assertEqual(result.risk_class, 1)
        self.assertAlmostEqual(result.high_risk_probability, 0.8)
        self.assertAlmostEqual(result.low_risk_probability, 0.2)
        self.assertAlmostEqual(result.predicted_class_probability, 0.8)

    def test_missing_indicator_is_rejected(self) -> None:
        with self.assertRaises(PredictionError):
            predict_student(self.model, {"asistencia": 80, "promedio": 7})

    def test_out_of_range_indicator_is_rejected(self) -> None:
        with self.assertRaises(PredictionError):
            predict_student(self.model, {**self.indicators, "promedio": 11})

    def test_model_loading_rejects_missing_artifact(self) -> None:
        with self.assertRaises(PredictionError):
            load_model(Path(self.temporary_directory.name) / "absent.pkl")

    def test_model_loading_reads_saved_artifact(self) -> None:
        loaded = load_model(self.model_path)
        self.assertEqual(set(loaded.classes_), {0, 1})


if __name__ == "__main__":
    unittest.main()

"""Pruebas del entrenamiento y la persistencia del modelo."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import joblib

from src.config import DATASET_PATH
from src.data.csv_loader import load_dataset
from src.ml.trainer import TrainingError, train_and_save


class TrainingTests(unittest.TestCase):
    def test_trains_and_serializes_model_in_temporary_directory(self) -> None:
        frame = load_dataset(DATASET_PATH)
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "temporary_model.pkl"
            report = train_and_save(frame, model_path)
            self.assertEqual(report.total_samples, 20)
            self.assertEqual(report.train_samples, 16)
            self.assertEqual(report.test_samples, 4)
            self.assertTrue(model_path.is_file())
            self.assertGreater(model_path.stat().st_size, 0)
            model = joblib.load(model_path)
            self.assertEqual(set(model.classes_), {0, 1})
            self.assertEqual(model.n_estimators, 100)
            self.assertEqual(model.random_state, 42)

    def test_rejects_a_dataset_with_too_few_records(self) -> None:
        frame = load_dataset(DATASET_PATH).head(4)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TrainingError):
                train_and_save(frame, Path(directory) / "model.pkl")


if __name__ == "__main__":
    unittest.main()

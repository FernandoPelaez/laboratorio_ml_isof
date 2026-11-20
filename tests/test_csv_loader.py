"""Pruebas de lectura, normalización y validación del CSV."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config import DATASET_PATH
from src.data.csv_loader import DataValidationError, load_dataset


class CsvLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.csv_path = Path(self.temporary_directory.name) / "dataset.csv"

    def test_original_dataset_has_twenty_records_and_four_required_columns(self) -> None:
        frame = load_dataset(DATASET_PATH)
        self.assertEqual(len(frame), 20)
        self.assertEqual(
            list(frame.columns),
            ["asistencia", "promedio", "horas_estudio", "riesgo"],
        )

    def test_expected_row_count_is_checked_before_cleaning(self) -> None:
        self.csv_path.write_text(
            "asistencia,promedio,horas_estudio,riesgo\n85,8,12,0\n,,,\n60,6,4,1\n",
            encoding="utf-8",
        )
        with self.assertRaises(DataValidationError):
            load_dataset(self.csv_path, expected_rows=2)

    def test_normalizes_headers_and_drops_incomplete_rows(self) -> None:
        self.csv_path.write_text(
            " Asistencia , Promedio , Horas_Estudio , Riesgo \n"
            "85,8,12,0\n,,,\n60,6,4,1\n",
            encoding="utf-8",
        )
        frame = load_dataset(self.csv_path)
        self.assertEqual(len(frame), 2)
        self.assertEqual(
            list(frame.columns),
            ["asistencia", "promedio", "horas_estudio", "riesgo"],
        )

    def test_repairs_rows_quoted_as_single_csv_fields(self) -> None:
        self.csv_path.write_text(
            '"asistencia,promedio,horas_estudio,riesgo"\n'
            '"85,8,12,0"\n"60,6,4,1"\n',
            encoding="utf-8",
        )
        self.assertEqual(load_dataset(self.csv_path).shape, (2, 4))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_dataset(Path(self.temporary_directory.name) / "missing.csv")

    def test_missing_columns_are_rejected(self) -> None:
        self.csv_path.write_text(
            "asistencia,promedio,riesgo\n80,8,0\n60,6,1\n", encoding="utf-8"
        )
        with self.assertRaises(DataValidationError):
            load_dataset(self.csv_path)

    def test_non_numeric_values_are_rejected(self) -> None:
        self.csv_path.write_text(
            "asistencia,promedio,horas_estudio,riesgo\n85,8,doce,0\n60,6,4,1\n",
            encoding="utf-8",
        )
        with self.assertRaises(DataValidationError):
            load_dataset(self.csv_path)

    def test_out_of_range_values_are_rejected(self) -> None:
        self.csv_path.write_text(
            "asistencia,promedio,horas_estudio,riesgo\n105,8,12,0\n60,6,4,1\n",
            encoding="utf-8",
        )
        with self.assertRaises(DataValidationError):
            load_dataset(self.csv_path)


if __name__ == "__main__":
    unittest.main()

"""Pruebas para la lectura segura de secretos opcionales."""

from __future__ import annotations

import unittest

from src.config import resolve_groq_api_key


class MissingSecrets:
    """Simula el proxy de Streamlit cuando no existe secrets.toml."""

    def __bool__(self) -> bool:
        raise RuntimeError("No secrets found")

    def get(self, key: str):
        raise RuntimeError("No secrets found")


class ConfigTests(unittest.TestCase):
    def test_missing_streamlit_secrets_are_treated_as_unconfigured(self) -> None:
        self.assertIsNone(
            resolve_groq_api_key(streamlit_secrets=MissingSecrets(), environ={})
        )

    def test_environment_key_works_when_streamlit_secrets_are_missing(self) -> None:
        result = resolve_groq_api_key(
            streamlit_secrets=MissingSecrets(),
            environ={"GROQ_API_KEY": "test-key"},
        )
        self.assertEqual(result, "test-key")

    def test_streamlit_secret_is_used_when_present(self) -> None:
        result = resolve_groq_api_key(
            streamlit_secrets={"GROQ_API_KEY": "test-key"},
            environ={},
        )
        self.assertEqual(result, "test-key")


if __name__ == "__main__":
    unittest.main()
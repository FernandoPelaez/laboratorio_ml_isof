"""Pruebas del cliente Groq con un SDK simulado."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.integrations.groq_client import (
    ConnectionFailureError,
    EmptyGroqResponseError,
    GroqAPIClient,
    InvalidCredentialsError,
    MissingApiKeyError,
    NoChatModelAvailableError,
    RateLimitError,
)


class ModelUnavailable(Exception):
    status_code = 404


class APIConnectionError(Exception):
    pass


class RateLimitErrorFromSdk(Exception):
    status_code = 429


class AuthenticationErrorFromSdk(Exception):
    status_code = 401


class FakeSDK:
    def __init__(self, models, response=None, completion_error=None, model_list_error=None):
        self._models = models
        self._model_list_error = model_list_error
        self.models = SimpleNamespace(list=self._list_models)
        self.used_models = []
        self.calls = []
        self._response = response or SimpleNamespace(
            model="llama-compatible",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Orientación útil."))],
        )
        self._completion_error = completion_error
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create_completion)
        )

    def _list_models(self):
        if self._model_list_error is not None:
            raise self._model_list_error
        return SimpleNamespace(data=[SimpleNamespace(id=name) for name in self._models])

    def _create_completion(self, **kwargs):
        self.used_models.append(kwargs["model"])
        self.calls.append(kwargs)
        if self._completion_error is not None:
            error = self._completion_error
            self._completion_error = None
            raise error
        return self._response


class GroqClientTests(unittest.TestCase):
    def test_missing_key_is_reported(self) -> None:
        with self.assertRaises(MissingApiKeyError):
            GroqAPIClient(None)

    def test_llama_is_prioritized_and_request_is_built(self) -> None:
        sdk = FakeSDK(["whisper-large", "gemma-text", "llama-chat"])
        client = GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk)
        result = client.generate([{"role": "user", "content": "Hola"}])
        self.assertEqual(sdk.used_models, ["llama-chat"])
        self.assertEqual(sdk.calls[0]["messages"][0]["content"], "Hola")
        self.assertEqual(sdk.calls[0]["temperature"], 0.4)
        self.assertEqual(sdk.calls[0]["max_completion_tokens"], 700)
        self.assertEqual(result.model_name, "llama-compatible")
        self.assertEqual(result.content, "Orientación útil.")

    def test_unavailable_model_falls_back(self) -> None:
        sdk = FakeSDK(["llama-a", "gemma-b"], completion_error=ModelUnavailable())
        result = GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
            [{"role": "user", "content": "Hola"}]
        )
        self.assertEqual(sdk.used_models, ["llama-a", "gemma-b"])
        self.assertEqual(result.content, "Orientación útil.")

    def test_models_for_other_tasks_are_excluded(self) -> None:
        sdk = FakeSDK(["whisper-large", "text-embedding", "tts-model"])
        with self.assertRaises(NoChatModelAvailableError):
            GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
                [{"role": "user", "content": "Hola"}]
            )

    def test_empty_response_is_reported(self) -> None:
        empty = SimpleNamespace(
            model="llama-chat",
            choices=[SimpleNamespace(message=SimpleNamespace(content="  "))],
        )
        sdk = FakeSDK(["llama-chat"], response=empty)
        with self.assertRaises(EmptyGroqResponseError):
            GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
                [{"role": "user", "content": "Hola"}]
            )

    def test_connection_failure_is_translated_without_details(self) -> None:
        sdk = FakeSDK(["llama-chat"], completion_error=APIConnectionError("private detail"))
        with self.assertRaises(ConnectionFailureError) as context:
            GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
                [{"role": "user", "content": "Hola"}]
            )
        self.assertNotIn("private detail", str(context.exception))

    def test_rate_limit_is_translated(self) -> None:
        sdk = FakeSDK(["llama-chat"], completion_error=RateLimitErrorFromSdk())
        with self.assertRaises(RateLimitError):
            GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
                [{"role": "user", "content": "Hola"}]
            )

    def test_invalid_credentials_are_translated_without_key(self) -> None:
        sdk = FakeSDK(["llama-chat"], model_list_error=AuthenticationErrorFromSdk())
        with self.assertRaises(InvalidCredentialsError) as context:
            GroqAPIClient("test-key", client_factory=lambda **kwargs: sdk).generate(
                [{"role": "user", "content": "Hola"}]
            )
        self.assertNotIn("test-key", str(context.exception))


if __name__ == "__main__":
    unittest.main()

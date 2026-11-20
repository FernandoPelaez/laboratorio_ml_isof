"""Cliente acotado para listar modelos y solicitar texto a Groq."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

EXCLUDED_MODEL_TERMS = (
    "audio", "embed", "guard", "image", "playai", "speech",
    "transcrib", "tts", "vision", "whisper",
)
MAX_MODEL_CANDIDATES = 5


class GroqIntegrationError(RuntimeError):
    """Error de integración presentado sin detalles sensibles del proveedor."""


class MissingApiKeyError(GroqIntegrationError):
    """La clave de Groq no está configurada."""


class InvalidCredentialsError(GroqIntegrationError):
    """Groq rechazó las credenciales configuradas."""


class RateLimitError(GroqIntegrationError):
    """La cuenta alcanzó un límite de solicitudes."""


class ConnectionFailureError(GroqIntegrationError):
    """No fue posible conectar con Groq."""


class RequestTimeoutError(GroqIntegrationError):
    """La solicitud a Groq agotó el tiempo de espera."""


class NoChatModelAvailableError(GroqIntegrationError):
    """No se encontró un modelo compatible con generación de texto."""


class EmptyGroqResponseError(GroqIntegrationError):
    """Groq no devolvió contenido de texto."""


@dataclass(frozen=True)
class GroqResponse:
    model_name: str
    content: str


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _translate_error(error: Exception) -> GroqIntegrationError:
    """Convierte excepciones de red y SDK en mensajes seguros para la interfaz."""
    name = type(error).__name__.lower()
    status = _status_code(error)
    if "auth" in name or status in (401, 403):
        return InvalidCredentialsError("Groq rechazó la clave. Revisa GROQ_API_KEY.")
    if "ratelimit" in name or status == 429:
        return RateLimitError("Groq alcanzó un límite de solicitudes. Inténtalo más tarde.")
    if "timeout" in name or isinstance(error, TimeoutError):
        return RequestTimeoutError("La solicitud a Groq tardó demasiado.")
    if "connection" in name or isinstance(error, ConnectionError):
        return ConnectionFailureError("No fue posible conectar con Groq.")
    return GroqIntegrationError("Ocurrió un error al solicitar recomendaciones a Groq.")


def _is_model_selection_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    return _status_code(error) in (400, 404, 410) or any(
        token in name for token in ("badrequest", "invalidrequest", "notfound")
    )


class GroqAPIClient:
    """Encapsula el SDK oficial; se importa solo cuando se configura una clave."""

    def __init__(
        self,
        api_key: str | None,
        client_factory: Any | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise MissingApiKeyError(
                "No se configuró GROQ_API_KEY. Añádela a los secretos de Streamlit."
            )
        if client_factory is None:
            try:
                from groq import Groq
            except ImportError as exc:
                raise GroqIntegrationError(
                    "Falta el SDK de Groq. Instala las dependencias del proyecto."
                ) from exc
            client_factory = Groq

        try:
            self._client = client_factory(
                api_key=api_key.strip(),
                timeout=20.0,
                max_retries=1,
            )
        except Exception as exc:
            logger.warning("No fue posible inicializar Groq (tipo=%s).", type(exc).__name__)
            raise _translate_error(exc) from exc

    @staticmethod
    def _model_id(model: Any) -> str:
        if isinstance(model, str):
            return model.strip()
        return str(getattr(model, "id", "")).strip()

    def _chat_model_candidates(self) -> list[str]:
        try:
            available_models = self._client.models.list().data
        except Exception as exc:
            logger.warning("No fue posible consultar modelos Groq (tipo=%s).", type(exc).__name__)
            raise _translate_error(exc) from exc

        candidates = []
        for model in available_models:
            model_id = self._model_id(model)
            if not model_id or getattr(model, "active", None) is False:
                continue
            if any(term in model_id.lower() for term in EXCLUDED_MODEL_TERMS):
                continue
            candidates.append(model_id)

        llama_models = sorted({model for model in candidates if "llama" in model.lower()})
        other_models = sorted(set(candidates) - set(llama_models))
        ordered = [*llama_models, *other_models]
        if not ordered:
            raise NoChatModelAvailableError(
                "La cuenta no muestra modelos candidatos para generar texto."
            )
        return ordered[:MAX_MODEL_CANDIDATES]

    def generate(self, messages: Sequence[Mapping[str, str]]) -> GroqResponse:
        """Prueba candidatos activos y prioriza Llama cuando está disponible."""
        for model_name in self._chat_model_candidates():
            try:
                completion = self._client.chat.completions.create(
                    model=model_name,
                    messages=[dict(message) for message in messages],
                    temperature=0.4,
                    max_completion_tokens=700,
                )
            except Exception as exc:
                if _is_model_selection_error(exc):
                    logger.info("Un modelo listado no aceptó chat completions.")
                    continue
                logger.warning("Falló la solicitud de Groq (tipo=%s).", type(exc).__name__)
                raise _translate_error(exc) from exc

            choices = getattr(completion, "choices", None)
            content = None
            if choices:
                message = getattr(choices[0], "message", None)
                content = getattr(message, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise EmptyGroqResponseError(
                    "Groq devolvió una respuesta vacía. Inténtalo nuevamente."
                )
            return GroqResponse(
                model_name=str(getattr(completion, "model", None) or model_name),
                content=content.strip(),
            )
        raise NoChatModelAvailableError("Ningún modelo listado aceptó solicitudes de chat.")

"""Punto de entrada y coordinador de la aplicación Streamlit."""

from __future__ import annotations

import logging

import streamlit as st

from src.config import resolve_groq_api_key
from src.integrations.groq_client import GroqAPIClient, GroqIntegrationError
from src.ml.predictor import PredictionError, load_model, predict_student
from src.services.tutoring import request_tutoring_advice
from src.ui.dashboard import (
    render_dashboard,
    render_error,
    render_groq_setup_message,
    render_prediction,
    render_tutoring,
)

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Evaluador Predictivo de Riesgo Académico",
    page_icon="📘",
    layout="wide",
)


@st.cache_resource
def _get_model():
    return load_model()


def _configured_api_key() -> str | None:
    try:
        secrets = st.secrets
    except Exception:
        secrets = None
    return resolve_groq_api_key(streamlit_secrets=secrets)


def main() -> None:
    inputs = render_dashboard()
    if inputs is None:
        return

    try:
        prediction = predict_student(_get_model(), inputs.as_dict())
    except PredictionError as exc:
        render_error(str(exc))
        return
    except Exception as exc:
        logger.error("Falló la evaluación (tipo=%s).", type(exc).__name__)
        render_error("No fue posible completar la evaluación. Revisa el modelo entrenado.")
        return

    render_prediction(prediction)
    api_key = _configured_api_key()
    if not api_key:
        render_groq_setup_message(
            "El diagnóstico de Machine Learning está disponible. Para generar tutoría, "
            "configura GROQ_API_KEY en .streamlit/secrets.toml o como variable de entorno."
        )
        return

    with st.spinner("Generando orientación académica con Groq..."):
        try:
            advice = request_tutoring_advice(
                indicators=inputs.as_dict(),
                prediction=prediction,
                client=GroqAPIClient(api_key=api_key),
            )
        except GroqIntegrationError as exc:
            render_error(str(exc))
        except Exception as exc:
            logger.error("Falló la tutoría (tipo=%s).", type(exc).__name__)
            render_error("No fue posible generar la tutoría. El diagnóstico sigue disponible.")
        else:
            render_tutoring(advice)


if __name__ == "__main__":
    main()

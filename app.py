"""Punto de entrada y coordinador de las páginas Streamlit."""

from __future__ import annotations

import logging

import streamlit as st

from src.config import resolve_groq_api_key
from src.integrations.groq_client import GroqAPIClient, GroqIntegrationError
from src.ml.predictor import PredictionError, load_model, predict_student
from src.services.tutoring import request_tutoring_advice
from src.ui.dashboard import render_dashboard, render_error
from src.ui.results import EvaluationResults, render_results

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Evaluador de Riesgo Académico",
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


def _evaluation_page() -> None:
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
        render_error(
            "No fue posible completar la evaluación. Revisa el modelo entrenado."
        )
        return

    advice = None
    setup_message = None
    tutoring_error = None
    api_key = _configured_api_key()

    if not api_key:
        setup_message = (
            "El diagnóstico de Machine Learning está disponible. Para generar "
            "tutoría, configura GROQ_API_KEY en .streamlit/secrets.toml o "
            "como variable de entorno."
        )
    else:
        with st.spinner("Generando orientación académica con Groq..."):
            try:
                advice = request_tutoring_advice(
                    indicators=inputs.as_dict(),
                    prediction=prediction,
                    client=GroqAPIClient(api_key=api_key),
                )
            except GroqIntegrationError as exc:
                logger.error("Falló la tutoría (tipo=%s).", type(exc).__name__)
                tutoring_error = str(exc)
            except Exception as exc:
                logger.error("Falló la tutoría (tipo=%s).", type(exc).__name__)
                tutoring_error = (
                    "No fue posible generar la tutoría. "
                    "El diagnóstico sigue disponible."
                )

    st.session_state["evaluation_results"] = EvaluationResults(
        prediction=prediction,
        tutoring_advice=advice,
        groq_setup_message=setup_message,
        tutoring_error=tutoring_error,
    )
    st.switch_page(_RESULTS_PAGE)


def _results_page() -> None:
    result = st.session_state.get("evaluation_results")
    if result is None:
        st.switch_page(_FORM_PAGE)

    if render_results(result):
        st.switch_page(_FORM_PAGE)


_FORM_PAGE = st.Page(
    _evaluation_page,
    title="Evaluación",
    url_path="evaluacion",
    icon=":material/edit_note:",
    help="Ingresa los indicadores académicos y solicita una evaluación de riesgo.",
    default=True,
)
_RESULTS_PAGE = st.Page(
    _results_page,
    title="Resultados",
    url_path="resultados",
    icon=":material/analytics:",
    help="Consulta el diagnóstico, las probabilidades y la orientación de tutoría.",
)


def main() -> None:
    navigation = st.navigation(
        [_FORM_PAGE, _RESULTS_PAGE],
        position="hidden",
    )
    navigation.run()


if __name__ == "__main__":
    main()
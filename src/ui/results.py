"""Pantalla exclusiva para mostrar el resultado de una evaluación."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from src.ml.predictor import PredictionResult
from src.services.tutoring import TutoringAdvice


@dataclass(frozen=True)
class EvaluationResults:
    """Resultado predictivo y estado opcional de la tutoría con Groq."""

    prediction: PredictionResult
    tutoring_advice: TutoringAdvice | None = None
    groq_setup_message: str | None = None
    tutoring_error: str | None = None


def _render_prediction(result: PredictionResult) -> None:
    """Presenta el diagnóstico y las probabilidades del clasificador."""
    st.subheader("Diagnóstico del modelo predictivo")

    if result.risk_class == 1:
        st.error("Riesgo Alto de Deserción")
        selected_name = (
            "Probabilidad del clasificador para la clase de alto riesgo"
        )
    else:
        st.success("Bajo Riesgo Académico")
        selected_name = (
            "Probabilidad del clasificador para la clase de bajo riesgo"
        )

    st.metric(
        selected_name,
        f"{result.predicted_class_probability:.1%}",
    )

    low_column, high_column = st.columns(2)
    low_column.metric(
        "Clase 0 · Bajo riesgo",
        f"{result.low_risk_probability:.1%}",
    )
    high_column.metric(
        "Clase 1 · Alto riesgo",
        f"{result.high_risk_probability:.1%}",
    )

    st.caption(
        "Estas probabilidades corresponden al clasificador entrenado con "
        "20 registros de ejemplo; no son una estimación validada de "
        "deserción ni una certeza."
    )


def _render_tutoring(advice: TutoringAdvice) -> None:
    """Presenta el modelo seleccionado y la tutoría generada."""
    st.subheader("Plan de tutoría automatizado")
    st.caption(f"Respuesta generada con Groq · Modelo: {advice.model_name}")
    st.markdown(advice.content)


def render_results(result: EvaluationResults) -> bool:
    """Muestra únicamente los resultados y permite volver al formulario."""
    st.title("Resultado de la evaluación")
    _render_prediction(result.prediction)

    if result.tutoring_advice is not None:
        _render_tutoring(result.tutoring_advice)
    elif result.groq_setup_message:
        st.info(result.groq_setup_message)

    if result.tutoring_error:
        st.error(result.tutoring_error)

    return st.button(
        "Volver al formulario",
        type="secondary",
        width="content",
        key="back_to_form_button",
    )
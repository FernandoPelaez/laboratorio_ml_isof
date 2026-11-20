"""Componentes visuales y controles de la aplicación Streamlit."""

from __future__ import annotations
from dataclasses import dataclass
import streamlit as st
from src.ml.predictor import PredictionResult
from src.services.tutoring import TutoringAdvice


@st.dialog(
    "Bienvenido al Evaluador de Riesgo Académico",
    width="small",
    dismissible=False,
)
def _render_welcome_dialog() -> None:
    """Muestra las indicaciones iniciales de la aplicación."""
    st.write(
        "Ingresa la asistencia, el promedio y las horas de estudio para "
        "obtener una evaluación orientativa y recomendaciones de tutoría. "
        "El resultado se basa en datos de ejemplo y no sustituye la "
        "valoración de un docente."
    )

    if st.button(
        "Comenzar",
        type="primary",
        width="content",
        key="welcome_start_button",
    ):
        st.session_state["welcome_dialog_dismissed"] = True
        st.rerun()


@dataclass(frozen=True)
class StudentInputs:
    """Indicadores ingresados para evaluar a un estudiante."""

    asistencia: float
    promedio: float
    horas_estudio: float

    def as_dict(self) -> dict[str, float]:
        """Devuelve los indicadores en formato de diccionario."""
        return {
            "asistencia": self.asistencia,
            "promedio": self.promedio,
            "horas_estudio": self.horas_estudio,
        }


def render_dashboard() -> StudentInputs | None:
    """Muestra el encabezado y el formulario de evaluación."""
    if not st.session_state.get("welcome_dialog_dismissed", False):
        st.html(
            """
            <style>
            @media (max-width: 640px) {
                [role="dialog"] {
                    position: fixed !important;
                    top: 50vh !important;
                    top: 50dvh !important;
                    left: 50vw !important;
                    transform: translate(-50%, -50%) !important;
                    max-height: calc(100dvh - 2rem) !important;
                    overflow-y: auto !important;
                }
            }
            </style>
            """
        )
        _render_welcome_dialog()
        return None

    st.title("Evaluador Predictivo de Riesgo Académico")
    st.write(
        "Evalúa indicadores académicos con un clasificador Random Forest y, "
        "si Groq está configurado, solicita una orientación de tutoría."
    )
    st.subheader("Parámetros del estudiante")

    with st.form("student_evaluation"):
        left_column, right_column = st.columns(2)

        with left_column:
            asistencia = st.slider(
                "Porcentaje de asistencia",
                min_value=0,
                max_value=100,
                value=80,
                format="%d%%",
            )
            promedio = st.number_input(
                "Promedio académico",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                format="%.1f",
            )

        with right_column:
            horas_estudio = st.slider(
                "Horas de estudio semanales",
                min_value=0,
                max_value=30,
                value=8,
            )
            st.caption("Introduce valores dentro de los rangos indicados.")

        submitted = st.form_submit_button(
            "Evaluar Estudiante",
            type="primary",
            width="content",
        )

    if not submitted:
        return None

    return StudentInputs(
        float(asistencia),
        float(promedio),
        float(horas_estudio),
    )


def render_prediction(result: PredictionResult) -> None:
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


def render_tutoring(advice: TutoringAdvice) -> None:
    """Presenta el modelo seleccionado y la tutoría generada."""
    st.subheader("Plan de tutoría automatizado")
    st.caption(f"Respuesta generada con Groq · Modelo: {advice.model_name}")
    st.markdown(advice.content)


def render_groq_setup_message(message: str) -> None:
    """Informa que el diagnóstico funciona aunque Groq no esté configurado."""
    st.info(message)


def render_error(message: str) -> None:
    """Muestra un mensaje de error comprensible."""
    st.error(message)
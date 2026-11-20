"""Componentes visuales y controles del formulario de evaluación."""

from __future__ import annotations
from dataclasses import dataclass

import streamlit as st


@st.dialog(
    "¡Hola, te damos la bienvenida!",
    width="small",
    dismissible=False,
)
def _render_welcome_dialog() -> None:
    """Muestra el saludo y la nota informativa inicial."""
    st.write(
        "Esta herramienta ofrece una orientación sobre el riesgo académico "
        "mediante inteligencia artificial. Los resultados son orientativos "
        "y no sustituyen la valoración de un docente."
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
    """Muestra el saludo y el formulario de evaluación."""
    if not st.session_state.get("welcome_dialog_dismissed", False):
        st.html(
            """
            <style>
                /* Posición del modal en computadoras. */
                @media (min-width: 769px) {
                    [data-testid="stDialog"] {
                        padding-top: clamp(
                            10rem,
                            35dvh,
                            24rem
                        ) !important;
                    }
                }

                /* Ajuste original para dispositivos móviles. */
                @media (max-width: 640px) {
                    [data-testid="stDialog"] {
                        padding-top: clamp(
                            6rem,
                            20dvh,
                            10rem
                        ) !important;
                    }
                }
            </style>
            """
        )
        _render_welcome_dialog()
        return None

    st.title("Evaluador de Riesgo Académico")
    st.write(
        "Ingresa la asistencia, el promedio académico y las horas de estudio "
        "semanales para evaluar el riesgo académico mediante Random Forest. "
        "Si Groq está configurado, también recibirás recomendaciones de tutoría."
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
                key="student_attendance",
                persist_state="session",
            )

            promedio = st.number_input(
                "Promedio académico",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                format="%.1f",
                key="student_average",
                persist_state="session",
            )

        with right_column:
            horas_estudio = st.slider(
                "Horas de estudio semanales",
                min_value=0,
                max_value=30,
                value=8,
                key="student_study_hours",
                persist_state="session",
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


def render_error(message: str) -> None:
    """Muestra un mensaje de error comprensible."""
    st.error(message)
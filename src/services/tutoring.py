"""Construye las solicitudes para la tutoría académica automatizada."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from src.integrations.groq_client import GroqAPIClient
from src.ml.predictor import PredictionResult


@dataclass(frozen=True)
class TutoringAdvice:
    """Respuesta de tutoría generada por el modelo de Groq."""

    model_name: str
    content: str


def build_tutoring_messages(
    indicators: Mapping[str, float],
    prediction: PredictionResult,
) -> list[dict[str, str]]:
    """Prepara los indicadores y las instrucciones para la tutoría."""

    if prediction.risk_class == 1:
        diagnosis = "Riesgo Alto de Deserción"
    else:
        diagnosis = "Bajo Riesgo Académico"

    system_prompt = (
        "Actúas como tutor académico universitario, respetuoso y práctico. "
        "Responde en español. Explica las relaciones entre los indicadores y el "
        "diagnóstico de manera general, sin afirmar causas exactas ni atribuir "
        "importancia a variables que el clasificador no explicó. Aclara que el "
        "resultado proviene de un modelo entrenado con pocos datos de ejemplo. "
        "Conserva las unidades indicadas: las horas de estudio son semanales. "
        "Nunca las describas como mensuales ni inventes otra frecuencia."
    )

    user_prompt = (
        "Indicadores del estudiante:\n"
        f"- Asistencia: {indicators['asistencia']:.1f}%\n"
        f"- Promedio: {indicators['promedio']:.1f}/10\n"
        f"- Horas de estudio: {indicators['horas_estudio']:.1f} horas por semana\n"
        f"- Diagnóstico del modelo predictivo: {diagnosis}\n\n"
        "Redacta una respuesta breve en Markdown con exactamente estas secciones:\n"
        "## 1. Análisis de Indicadores\n"
        "Relaciona los valores con el diagnóstico de forma general. "
        "Al mencionar las horas de estudio, indica que son por semana; "
        "no cambies esa unidad.\n\n"
        "## 2. Plan de Acción (3 pasos)\n"
        "Propón exactamente tres acciones concretas para mejorar o mantener "
        "el rendimiento. No inventes datos del estudiante."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def request_tutoring_advice(
    indicators: Mapping[str, float],
    prediction: PredictionResult,
    client: GroqAPIClient,
) -> TutoringAdvice:
    """Solicita al cliente de Groq una recomendación académica."""

    response = client.generate(
        build_tutoring_messages(indicators, prediction)
    )

    return TutoringAdvice(
        model_name=response.model_name,
        content=response.content,
    )
# Respuestas al cuestionario de reflexión pedagógica

## 1. Funcionamiento del algoritmo

El clasificador Random Forest aprende relaciones entre las variables de entrada y las etiquetas del dataset durante el entrenamiento. Está formado por varios árboles de decisión que generan resultados y combinan sus votos para clasificar un registro nuevo. Por eso, para evaluar un caso no se escriben reglas manuales específicas con sentencias if / else: el modelo aplica los patrones que aprendió de los ejemplos de entrenamiento. La calidad de esa clasificación depende de la cantidad, representatividad y calidad de los datos usados.

## 2. Limpieza de datos

Un registro con valores faltantes no proporciona al modelo toda la información requerida para entrenar o evaluar una fila. La operación dropna elimina las filas incompletas en las columnas necesarias, evitando errores durante la conversión numérica o el ajuste del clasificador. En esta aplicación, los valores no numéricos escritos en campos no vacíos se reportan como errores en lugar de descartarse silenciosamente; las filas con campos requeridos vacíos se eliminan.

## 3. Persistencia del modelo

Después del entrenamiento, joblib serializa el clasificador ajustado en modelo_riesgo.pkl. La aplicación carga ese artefacto para ejecutar predict y predict_proba, por lo que no necesita repetir la división de datos ni ajustar los árboles cada vez que se presiona el botón. Esta separación reduce el tiempo de respuesta y mantiene estable el modelo de inferencia hasta que se ejecute nuevamente el script de entrenamiento.

## 4. Rol de Groq API

Scikit-learn calcula la clase de riesgo y las probabilidades del clasificador a partir de asistencia, promedio y horas de estudio. Groq recibe esos indicadores y el diagnóstico para producir una explicación pedagógica breve y un plan de tres acciones. La API no modifica la predicción ni determina las probabilidades: es una capa opcional de asesoría en lenguaje natural. Si no se configura una clave o falla el servicio, el diagnóstico de Machine Learning sigue disponible.

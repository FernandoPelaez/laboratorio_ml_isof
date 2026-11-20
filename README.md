# Proyecto ML ISOF

## Descripción

Aplicación académica que utiliza los veinte registros de alumnos.csv para entrenar un clasificador Random Forest. El artefacto modelo_riesgo.pkl se usa en una página Streamlit que evalúa indicadores y puede solicitar tutoría en español a Groq.

## Tecnologías

Python, pandas, scikit-learn, joblib, Streamlit y el SDK oficial de Groq. Las dependencias principales y las de persistencia del modelo están fijadas en requirements.txt.

## Arquitectura

app.py coordina el flujo de Streamlit; entrenar.py inicia el entrenamiento. Los módulos de src separan configuración, CSV, ML, Groq, tutoría y presentación. Las pruebas usan unittest y mocks.

    proyecto_ml_isof/
    ├── app.py
    ├── entrenar.py
    ├── alumnos.csv
    ├── modelo_riesgo.pkl
    ├── requirements.txt
    ├── .gitignore
    ├── README.md
    ├── respuestas_cuestionario.md
    ├── .streamlit/
    │   └── secrets.toml.example
    ├── src/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── data/
    │   │   ├── __init__.py
    │   │   └── csv_loader.py
    │   ├── ml/
    │   │   ├── __init__.py
    │   │   ├── trainer.py
    │   │   └── predictor.py
    │   ├── integrations/
    │   │   ├── __init__.py
    │   │   └── groq_client.py
    │   ├── services/
    │   │   ├── __init__.py
    │   │   └── tutoring.py
    │   └── ui/
    │       ├── __init__.py
    │       └── dashboard.py
    └── tests/
        ├── __init__.py
        ├── test_csv_loader.py
        ├── test_training.py
        ├── test_predictor.py
        └── test_groq_client.py

## Flujo de Machine Learning

El cargador normaliza encabezados, repara filas agrupadas en una columna, convierte datos y elimina registros incompletos. Entrenar.py valida que el CSV tenga veinte filas originales. El entrenador utiliza asistencia, promedio y horas_estudio como entradas, riesgo como objetivo, test_size=0.2, random_state=42 y RandomForestClassifier(n_estimators=100, random_state=42). El predictor carga el modelo persistido y usa classes_ para asignar las probabilidades.

La evaluación del conjunto de prueba describe únicamente una división de veinte ejemplos; no demuestra precisión para estudiantes reales.

## Flujo de Groq

Streamlit → servicio de tutoría → cliente Groq → API de Groq. El cliente consulta modelos activos, excluye nombres asociados a tareas no conversacionales, prioriza Llama y prueba hasta cinco candidatos. El listado por sí solo no confirma compatibilidad con chat: la solicitud de Chat Completions valida cada candidato y permite intentar otro ante un rechazo del modelo.

## Instalación y ejecución en PowerShell

El artefacto entregado fue generado con Python 3.14.7, scikit-learn 1.9.1, NumPy 2.5.5, SciPy 1.18.1 y joblib 1.6.0. Usa esas versiones para cargar el modelo serializado.

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    python entrenar.py
    python -m streamlit run app.py

La salida de predict_proba no es una estimación validada de deserción ni una certeza académica.

## Configuración de Groq

Obtén una clave desde https://console.groq.com/keys. Para desarrollo, copia .streamlit/secrets.toml.example a .streamlit/secrets.toml y coloca la clave en GROQ_API_KEY, o configura la variable de entorno del mismo nombre. El archivo real está protegido por .gitignore. La evaluación ML funciona aunque Groq no esté configurado.

En Streamlit Community Cloud, selecciona Python 3.14 en Advanced settings y configura GROQ_API_KEY en Secrets. Sube alumnos.csv y modelo_riesgo.pkl a GitHub, selecciona app.py y realiza el despliegue desde tu cuenta.

## Pruebas

    python -m unittest discover -s tests -v

Las pruebas de Groq usan un SDK simulado y no requieren una clave. Para comprobar una llamada real se necesita una clave válida y modelos habilitados.

## Solución de problemas

- Si no existe modelo_riesgo.pkl, ejecuta python entrenar.py.
- Si el modelo no carga, verifica Python y las versiones fijadas en requirements.txt.
- Si falla la tutoría, revisa GROQ_API_KEY y la disponibilidad de modelos.
- Si el CSV no valida, comprueba nombres, valores y rangos de las columnas.

## Limitaciones y cumplimiento

El modelo usa veinte registros de ejemplo y no está validado para decisiones institucionales. La integración con Groq solo redacta recomendaciones; no altera el resultado del clasificador.

Los siete apartados están cubiertos: conceptos y librerías en la documentación; dataset original; entrenamiento y artefacto; aplicación Streamlit con Groq opcional; instrucciones de despliegue; y las cuatro respuestas en respuestas_cuestionario.md. La publicación y una llamada autenticada real quedan pendientes de las cuentas y credenciales del usuario.

## Verificación

Se ejecutó python entrenar.py: el modelo se creó con 20 filas (16 para entrenamiento y 4 para prueba) y pesa 64,137 bytes. La exactitud de prueba fue 0.75 y solo describe esta división pequeña. Pasaron 24 pruebas con python -m unittest discover -s tests -v; pip check no reportó dependencias rotas. Streamlit AppTest confirmó la página, el título y el botón Evaluar Estudiante sin excepciones. La llamada autenticada real a Groq y el despliegue público quedan pendientes de configurar una clave y las cuentas correspondientes.

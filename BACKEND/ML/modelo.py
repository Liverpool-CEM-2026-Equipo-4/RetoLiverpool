import joblib
import json
import os
import pandas as pd

from dotenv import load_dotenv
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

from google import genai
from google.genai import types


# ==================================================
# MODELOS PYDANTIC
# ==================================================

class renovoPredictionModel(BaseModel):
    revenue: float
    antiguedad_marca: int
    calificacion_promedio_productos: float
    participacion_mercado: float
    participacion_mercado_promedio: float
    crecimiento_total_sales: float


class GeminiChatModel(BaseModel):
    prompt: str


# ==================================================
# VARIABLES DE ENTORNO
# ==================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY")


# ==================================================
# CONEXIÓN POSTGRESQL
# ==================================================

conn_string = (
    f"host={DB_HOST} "
    f"port={DB_PORT} "
    f"dbname={DB_NAME} "
    f"user={DB_USER} "
    f"password={DB_PASSWORD}"
)

pool = ConnectionPool(
    conninfo=conn_string
)


# ==================================================
# CLIENTE GEMINI
# ==================================================

client = genai.Client(
    api_key=GOOGLE_GENAI_API_KEY
)


# ==================================================
# ESQUEMA DE BASE DE DATOS
# ==================================================

def get_schema_description() -> str:

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute("""
                SELECT table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
            """)

            rows = cur.fetchall()

    tables: Dict[str, Any] = {}

    for row in rows:

        table_name = row["table_name"]

        col = f"{row['column_name']} ({row['data_type']})"

        tables.setdefault(table_name, []).append(col)

    lines = ["Esquema de la base de datos:"]

    for table, cols in tables.items():

        lines.append(f"- {table}:")

        for col in cols:

            lines.append(f"  - {col}")

    return "\n".join(lines)


# ==================================================
# VALIDACIÓN CONSULTAS SQL
# ==================================================

def _is_safe_query(query: str) -> bool:

    normalized = query.strip().lower()

    forbidden = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "truncate",
        "create"
    )

    if not normalized.startswith("select"):
        return False

    return not any(word in normalized for word in forbidden)


# ==================================================
# TOOL CONSULTA SQL
# ==================================================

def query_database(query: str) -> str:
    """
    Ejecuta una consulta SQL de solo lectura
    y devuelve los resultados en formato JSON.
    """

    if not _is_safe_query(query):

        return "Error: Solo se permiten consultas SELECT."

    try:

        with pool.connection() as conn:

            conn.row_factory = dict_row

            with conn.cursor() as cur:

                cur.execute(query)

                rows = cur.fetchall()

                return json.dumps(rows, default=str)

    except Exception as e:

        return f"Error al ejecutar consulta: {str(e)}"


# ==================================================
# TOOL MACHINE LEARNING
# ==================================================

def predict_renovo_tool(
    antiguedad_marca: int,
    participacion_mercado_promedio: float,
    calificacion_promedio_productos: float,
    participacion_mercado: float,
    revenue: float,
    crecimiento_total_sales: float
) -> str:
    """
    Genera una predicción de renovación de marcas
    utilizando un modelo de machine learning entrenado
    con variables históricas, financieras y operativas.
    """

    input_data = pd.DataFrame(
    [
        [
            participacion_mercado_promedio,
            revenue,
            antiguedad_marca,
            calificacion_promedio_productos,
            participacion_mercado,
            crecimiento_total_sales
        ]
    ],
    columns=[
        "Avg Market Share",
        "Revenue",
        "Antigüedad de la Marca",
        "Calificación Promedio de Productos",
        "Participación de Mercado (%)",
        "Crecimiento Total Sales"
    ]
)
    prediction = app.state.model.predict(input_data)[0]

    return f"Predicción de renovación de marcas: {prediction}"


# ==================================================
# CHAT GEMINI
# ==================================================

def chat_with_gemini(
    prompt: str,
    schema_description: str
) -> str:

    MODEL = "gemini-2.5-flash"

    SYSTEM_PROMPT = """
Eres un asistente especializado en análisis de datos
del Data Warehouse de Liverpool.

Ayudas a analizar:
- ventas
- tendencias
- marcas
- participación de mercado

Puedes consultar la base de datos y generar predicciones.
"""

    config = types.GenerateContentConfig(
        system_instruction=f"{SYSTEM_PROMPT}\n{schema_description}",
        tools=[
            query_database,
            predict_renovo_tool
        ],
    )

    response = client.models.generate_content(
        model=MODEL,
        config=config,
        contents=prompt,
    )

    return response.text


# ==================================================
# APP FASTAPI
# ==================================================

app = FastAPI()


# ==================================================
# CARGA MODELO ML
# ==================================================

app.state.model = joblib.load(
    "decision_tree_model.joblib"
)

app.state.schema_description = (
    get_schema_description()
)

print("Servidor iniciado correctamente")
print(app.state.model.feature_names_in_)


# ==================================================
# ENDPOINT HOME
# ==================================================

@app.get("/")
def index():

    return {
        "mensaje": "Backend Liverpool funcionando"
    }


# ==================================================
# GET 1 - MARCAS
# ==================================================

@app.get("/marcas")
def get_marcas():

    query = '''
    SELECT *
    FROM "D_marca"
    LIMIT 50
    '''

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()


# ==================================================
# GET 2 - TOP VENTAS
# ==================================================

@app.get("/top_ventas")
def get_top_ventas():

    query = '''
    SELECT *
    FROM "Hecho_ventas"
    ORDER BY revenue DESC
    LIMIT 10
    '''

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()


# ==================================================
# GET 3 - TOP TRENDS
# ==================================================

@app.get("/top_trends")
def get_top_trends():

    query = '''
    SELECT *
    FROM "Hecho_trends"
    ORDER BY trend DESC
    LIMIT 10
    '''

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()


# ==================================================
# GET 4 - DETALLE MARCAS
# ==================================================

@app.get("/detalle_marcas")
def get_detalle_marcas():

    query = '''
    SELECT
        dm.denominacion,
        d.no_registro,
        d.estatus_marca,
        d.tipo
    FROM "Detalle_marca" d
    JOIN "D_marca" dm
        ON d.id_marca = dm.id_marca
    LIMIT 20
    '''

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()


# ==================================================
# GET 5 - CLASES
# ==================================================

@app.get("/clases")
def get_clases():

    query = '''
    SELECT *
    FROM "D_clase"
    '''

    with pool.connection() as conn:

        conn.row_factory = dict_row

        with conn.cursor() as cur:

            cur.execute(query)

            return cur.fetchall()


# ==================================================
# POST MACHINE LEARNING
# ==================================================

@app.post("/predict_renovo")
def predict_renovo(
    data: renovoPredictionModel
):

    pred_df = pd.DataFrame([{
        "Avg Market Share": data.participacion_mercado_promedio,
        "Revenue": data.revenue,
        "Antigüedad de la Marca": data.antiguedad_marca,
        "Calificación Promedio de Productos": data.calificacion_promedio_productos,
        "Participación de Mercado (%)": data.participacion_mercado,
        "Crecimiento Total Sales": data.crecimiento_total_sales
    }])

    prediction = app.state.model.predict(pred_df)[0]

    return {
        "prediccion": str(prediction),
    }


# ==================================================
# POST GEMINI
# ==================================================

@app.post("/chat")
def chat(prompt: GeminiChatModel):

    response = chat_with_gemini(
        prompt.prompt,
        app.state.schema_description
    )

    return {
        "response": response
    }


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
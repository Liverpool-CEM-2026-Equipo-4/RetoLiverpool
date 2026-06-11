# =========================================================
# MULTIAGENT ORCHESTRATION SYSTEM - LIVERPOOL LEGAL PROJECT
# =========================================================

from google.adk.agents import Agent
import google.genai as genai
from dotenv import load_dotenv
from pinecone import Pinecone
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import json
import logging
import re

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================================================
# ENV
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

AGENT_MODEL = "gemini-2.5-flash-lite"

# =========================================================
# CLIENTS
# =========================================================

genai_client = genai.Client(api_key=GOOGLE_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# =========================================================
# HELPERS
# =========================================================

def json_response(data) -> str:
    return json.dumps(data, default=str, ensure_ascii=False, indent=2)


def get_postgres_connection():
    """Conecta a PostgreSQL con logging mejorado."""
    try:
        logger.info(f"Conectando a {POSTGRES_HOST}:{POSTGRES_PORT}...")
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=8,
        )
        conn.set_session(readonly=True, autocommit=True)
        logger.info("Conexion exitosa a PostgreSQL")
        return conn
    except psycopg2.OperationalError as e:
        logger.error(f"Error de conexion PostgreSQL: {str(e)}")
        raise


def generate_embedding(text: str) -> list:
    result = genai_client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text,
    )

    return list(result.embeddings[0].values)[:1024]


def clean_sql(sql: str) -> str:
    sql = sql.strip()

    if sql.endswith(";"):
        sql = sql[:-1].strip()

    return sql


def is_safe_readonly_sql(sql: str) -> bool:
    sql_clean = clean_sql(sql)
    sql_lower = sql_clean.lower()

    if not (sql_lower.startswith("select") or sql_lower.startswith("with")):
        return False

    blocked_patterns = [
        r"\binsert\b",
        r"\bupdate\b",
        r"\bdelete\b",
        r"\bdrop\b",
        r"\balter\b",
        r"\btruncate\b",
        r"\bcreate\b",
        r"\bgrant\b",
        r"\brevoke\b",
        r"\bcopy\b",
        r"\bexecute\b",
        r"\bcall\b",
        r"\bdo\b",
        r"\bmerge\b",
        r"\bvacuum\b",
        r"\banalyze\b",
        r"\brefresh\b",
        r"\bset\b",
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, sql_lower):
            return False

    if ";" in sql_clean:
        return False

    return True

# =========================================================
# RESEARCHER TOOLS
# =========================================================

def search_legal_documents(query: str, top_k: int = 8) -> str:
    """
    Busca documentos legales en Pinecone.
    """

    logger.info(f"[RESEARCHER] search_legal_documents: {query}")

    try:
        top_k = min(max(int(top_k), 1), 20)

        query_embedding = generate_embedding(query)

        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            include_values=False,
        )

        matches = results.get("matches", [])

        if not matches:
            return json_response(
                {
                    "message": "No se encontraron documentos legales relacionados.",
                    "query": query,
                }
            )

        output = []

        for match in matches:
            metadata = match.get("metadata", {})

            output.append(
                {
                    "content": metadata.get("page_content", metadata.get("text", ""))[:1200],
                    "source": metadata.get("source", metadata.get("file_name", "Unknown")),
                    "page": metadata.get("page", metadata.get("page_number", None)),
                    "relevance": round(match.get("score", 0), 4),
                }
            )

        return json_response(output)

    except Exception as e:
        logger.error(f"[RESEARCHER] Error: {str(e)}", exc_info=True)
        return json_response({"error": str(e)})


def analyze_legal_issue(entity_or_topic: str) -> str:
    """
    Analiza problemas legales usando Pinecone.
    """

    logger.info(f"[RESEARCHER] analyze_legal_issue: {entity_or_topic}")

    try:
        query = (
            f"problema legal litigio demanda oposicion riesgo juridico expediente "
            f"renovacion marca contrato {entity_or_topic}"
        )

        results = search_legal_documents(query=query, top_k=10)

        return json_response(
            {
                "entity_or_topic": entity_or_topic,
                "analysis_query": query,
                "documents": json.loads(results),
            }
        )

    except Exception as e:
        logger.error(f"[RESEARCHER] Error: {str(e)}", exc_info=True)
        return json_response({"error": str(e)})


def compare_legal_documents(topic: str) -> str:
    """
    Compara documentos legales relacionados.
    """

    query = f"comparacion documentos legales expedientes contratos marcas {topic}"

    return search_legal_documents(query=query, top_k=12)


researcher_agent = Agent(
    name="Researcher",
    model=AGENT_MODEL,
    description=(
        "Especialista legal. Consulta Pinecone para documentos legales, contratos, "
        "litigios, problemas legales, expedientes y riesgos juridicos."
    ),
    instruction="""
Eres el especialista legal del proyecto Liverpool.

Tu UNICA mision: responder preguntas sobre asuntos legales.

CUANDO EL USUARIO PREGUNTE SOBRE:
- Litigios, demandas, expedientes, conflictos
- Problemas legales, riesgos juridicos
- Contratos, derechos de propiedad intelectual
- Renovacion, cancelacion u oposicion de marcas
- Cualquier tema juridico

INMEDIATAMENTE:
1. Usa search_legal_documents() o analyze_legal_issue() para buscar en Pinecone
2. Extrae la informacion relevante
3. RESPONDE DIRECTAMENTE AL USUARIO con los hallazgos
4. No menciones que usaste herramientas
5. No hagas menciones a agentes, coordinadores o procesos internos

FORMATO DE RESPUESTA:
- Dale la respuesta de forma natural y directa
- Incluye los documentos relevantes encontrados
- Explica el riesgo legal si lo hay
- Da recomendaciones

Responde SIEMPRE en espanol.
Responde DIRECTAMENTE. Sin prefacios.
Tu respuesta ES la respuesta final del sistema.
""",
    tools=[
        search_legal_documents,
        analyze_legal_issue,
        compare_legal_documents,
    ],
)

# =========================================================
# DATA ANALYST TOOLS
# =========================================================

def get_database_schema() -> str:
    """
    Obtiene esquema de PostgreSQL.
    """

    conn = None

    try:
        sql = """
            SELECT
                table_name,
                column_name,
                data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
            ORDER BY table_name, ordinal_position
        """

        conn = get_postgres_connection()

        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return json_response([dict(row) for row in rows])

    except Exception as e:
        logger.error(f"[DATA] Error: {str(e)}", exc_info=True)

        if conn:
            conn.close()

        return json_response({"error": str(e)})


def run_readonly_sql(sql: str, limit: int = 100) -> str:
    """
    Ejecuta SQL SELECT seguro.
    """

    conn = None

    try:
        limit = min(max(int(limit), 1), 500)

        sql_clean = clean_sql(sql)

        if not is_safe_readonly_sql(sql_clean):
            return json_response(
                {
                    "error": "SQL no permitido.",
                    "sql": sql,
                }
            )

        wrapped_sql = f"""
            SELECT *
            FROM (
                {sql_clean}
            ) AS readonly_query
            LIMIT %s
        """

        conn = get_postgres_connection()

        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(wrapped_sql, [limit])

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return json_response([dict(row) for row in rows])

    except Exception as e:
        logger.error(f"[DATA] Error: {str(e)}", exc_info=True)

        if conn:
            conn.close()

        return json_response({"error": str(e)})


def query_brands(search_term: str = "", status: str = "", limit: int = 50) -> str:
    """
    Busca marcas en PostgreSQL.
    """

    conn = None

    try:
        limit = min(max(int(limit), 1), 200)

        where_clauses = []
        params = []

        if search_term:
            where_clauses.append(
                "(dm.denominacion ILIKE %s OR dm.titular ILIKE %s)"
            )

            params.extend(
                [
                    f"%{search_term}%",
                    f"%{search_term}%",
                ]
            )

        if status:
            where_clauses.append(
                "det.estatus_marca ILIKE %s"
            )

            params.append(f"%{status}%")

        where_sql = ""

        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT DISTINCT
                dm.id_marca,
                dm.denominacion,
                dm.titular,
                det.estatus_marca
            FROM "D_marca" dm
            LEFT JOIN "Detalle_marca" det
                ON dm.id_marca = det.id_marca
            {where_sql}
            ORDER BY dm.denominacion
            LIMIT %s
        """

        params.append(limit)

        conn = get_postgres_connection()

        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(sql, params)

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return json_response([dict(row) for row in rows])

    except Exception as e:
        logger.error(f"[DATA] Error: {str(e)}", exc_info=True)

        if conn:
            conn.close()

        return json_response({"error": str(e)})


def count_brands(limit: int = 20) -> str:
    """
    Cuenta las marcas que mas se repiten.
    """

    conn = None

    try:
        limit = min(max(int(limit), 1), 100)

        sql = """
            SELECT
                dm.denominacion,
                dm.titular,
                COUNT(*) AS total_apariciones
            FROM "D_marca" dm
            LEFT JOIN "Detalle_marca" det
                ON dm.id_marca = det.id_marca
            GROUP BY dm.denominacion, dm.titular
            ORDER BY total_apariciones DESC
            LIMIT %s
        """

        conn = get_postgres_connection()

        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(sql, [limit])

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return json_response([dict(row) for row in rows])

    except Exception as e:
        logger.error(f"[DATA] Error: {str(e)}", exc_info=True)

        if conn:
            conn.close()

        return json_response({"error": str(e)})


def get_brand_stats(brand_name: str) -> str:
    """
    Obtiene estadisticas de marca.
    """

    conn = None

    try:
        sql = """
            SELECT 
                dm.denominacion,
                dm.titular,
                COUNT(hv.id_ventas) AS total_sales,
                SUM(hv.revenue) AS total_revenue,
                AVG(hv.revenue) AS avg_revenue,
                AVG(hv.calificacion_promedio) AS avg_rating,
                SUM(hv.numero_devoluciones) AS total_returns,
                AVG(hv.avg_market_share) AS avg_market_share
            FROM "D_marca" dm
            JOIN "Detalle_marca" det
                ON dm.id_marca = det.id_marca
            JOIN "Hecho_ventas" hv
                ON det.id_detalle = hv.id_detalle
            WHERE dm.denominacion ILIKE %s
            GROUP BY dm.denominacion, dm.titular
            ORDER BY total_revenue DESC NULLS LAST
        """

        conn = get_postgres_connection()

        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(sql, [f"%{brand_name}%"])

        rows = cur.fetchall()

        cur.close()
        conn.close()

        return json_response([dict(row) for row in rows])

    except Exception as e:
        logger.error(f"[DATA] Error: {str(e)}", exc_info=True)

        if conn:
            conn.close()

        return json_response({"error": str(e)})


def get_top_brands_by_metric(metric: str = "revenue", limit: int = 10) -> str:
    """
    Ranking de marcas.
    """

    metric_map = {
        "revenue": "SUM(hv.revenue)",
        "ventas": "COUNT(hv.id_ventas)",
        "sales": "COUNT(hv.id_ventas)",
        "market_share": "AVG(hv.avg_market_share)",
        "rating": "AVG(hv.calificacion_promedio)",
        "returns": "SUM(hv.numero_devoluciones)",
    }

    metric_sql = metric_map.get(metric.lower(), "SUM(hv.revenue)")

    sql = f"""
        SELECT
            dm.denominacion,
            dm.titular,
            {metric_sql} AS metric_value
        FROM "D_marca" dm
        JOIN "Detalle_marca" det
            ON dm.id_marca = det.id_marca
        JOIN "Hecho_ventas" hv
            ON det.id_detalle = hv.id_detalle
        GROUP BY dm.denominacion, dm.titular
        ORDER BY metric_value DESC NULLS LAST
        LIMIT {limit}
    """

    return run_readonly_sql(sql)


data_analyst_agent = Agent(
    name="DataAnalyst",
    model=AGENT_MODEL,
    description=(
        "Especialista en PostgreSQL y datos comerciales de marcas."
    ),
    instruction="""
Eres el especialista en DATOS COMERCIALES del proyecto Liverpool.

Tu UNICA mision: responder preguntas sobre datos comerciales.

CUANDO EL USUARIO PREGUNTE SOBRE:
- Revenue, ingresos, ventas
- Market share, tendencias
- Ratings, calificaciones
- Estadisticas de marcas
- Ranking de marcas
- Cualquier metrica comercial

INMEDIATAMENTE:
1. Usa las herramientas (query_brands, get_brand_stats, get_top_brands_by_metric, etc)
2. Extrae los datos relevantes
3. RESPONDE DIRECTAMENTE AL USUARIO con los datos
4. No menciones que usaste herramientas
5. No hagas menciones a agentes, coordinadores o procesos internos

FORMATO DE RESPUESTA:
- Presenta los datos de forma clara
- Usa tablas o listas si es necesario
- Interpreta los datos comercialmente
- Da contexto

Responde SIEMPRE en espanol.
Responde DIRECTAMENTE. Sin prefacios.
Tu respuesta ES la respuesta final del sistema.
""",
    tools=[
        get_database_schema,
        run_readonly_sql,
        query_brands,
        count_brands,
        get_brand_stats,
        get_top_brands_by_metric,
    ],
)

# =========================================================
# REPORT GENERATOR
# =========================================================

def generate_brand_report(brand_name: str) -> str:
    stats = json.loads(get_brand_stats(brand_name))
    legal = json.loads(analyze_legal_issue(brand_name))

    return json_response(
        {
            "brand": brand_name,
            "commercial_stats": stats,
            "legal_findings": legal,
        }
    )


report_generator_agent = Agent(
    name="ReportGenerator",
    model=AGENT_MODEL,
    description="Generador de reportes.",
    instruction="""
Genera reportes ejecutivos legales y comerciales.
""",
    tools=[
        generate_brand_report,
    ],
)

# =========================================================
# COORDINATOR
# =========================================================

coordinator_agent = Agent(
    name="Coordinator",
    model=AGENT_MODEL,
    description="Coordinador principal multiagente.",
    instruction="""
Tu mision: ROUTTEAR SILENCIOSAMENTE la pregunta al agente correcto.

La respuesta del agente delegado DEBE SER la respuesta final.
NO AGREGUES NADA. NO EXPLIQUES NADA.

REGLAS DE ROUTEO AUTOMATICO:

1. PREGUNTAS LEGALES - PALABRAS CLAVE:
   litigio, demanda, expediente, contrato, legal, problema legal,
   conflicto, disputa, juicio, sentencia, recurso, derecho, propiedad,
   intelectual, defensa, proteccion, registro, renovacion, cancelacion,
   oposicion, juridico, case, lawsuit

   → DELEGA A: Researcher
   → La respuesta del Researcher ES la respuesta final

2. PREGUNTAS COMERCIALES - PALABRAS CLAVE:
   revenue, ingresos, ventas, sales, market share, rating, tendencias,
   top, ranking, estadisticas, promedio, total, cantidad, marca/marcas,
   cuantas, cuales

   Y NO contiene palabras legales

   → DELEGA A: DataAnalyst
   → La respuesta del DataAnalyst ES la respuesta final

3. SI PIDE REPORTES:
   → DELEGA A: ReportGenerator

4. SI PIDE DOCUMENTOS:
   → DELEGA A: DocumentManager

REGLA CRITICA:
- NO RESPONDAS TU
- DELEGA AL AGENTE CORRECTO
- LA RESPUESTA DEL AGENTE ES LA RESPUESTA FINAL
- SIN MENSAJES INTERMEDIOS
- SIN EXPLICACIONES
- SILENCIO TOTAL DEL COORDINADOR

El usuario NUNCA debe saber que hay un coordinador.
El usuario NUNCA debe ver "Esta consulta corresponde a..."
El usuario NUNCA debe ver nombres de agentes.

Solo ve la respuesta directa del especialista.

Responde en espanol.
""",
    sub_agents=[
        researcher_agent,
        data_analyst_agent,
        report_generator_agent,
    ],
)

# =========================================================
# ROOT AGENT
# =========================================================

root_agent = coordinator_agent

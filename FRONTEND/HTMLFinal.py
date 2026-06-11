"""
APP PRINCIPAL EN DASH: PORTAFOLIO MARCARIO LIVERPOOL
-----------------------------------------------------
Integra Dash + chatbot ADK.

Requisito:
- Este archivo debe poder importar root_agent.
"""

import asyncio
import dash
import joblib
import pandas as pd
import base64
import io
import socket

from dash import Dash, html, dcc, Input, Output, State, no_update
from dash import dash_table

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from LIVERPOOLADK.RagAgent.agent import root_agent

# -----------------------------------------------------------------------------
# MODELO DE PREDICCIÓN
# -----------------------------------------------------------------------------

MODEL_PATH = r"C:\Users\bbets\Documents\APP_WEB\ML\decision_tree_model.joblib"

modelo_prediccion = joblib.load(MODEL_PATH)

print("Modelo cargado correctamente")
print(modelo_prediccion.feature_names_in_)

# -----------------------------------------------------------------------------
# CONFIGURACIÓN EXTERNA DE ESTILOS Y SCRIPTS
# -----------------------------------------------------------------------------

external_stylesheets = []

external_scripts = [
    {"src": "https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"}
]


# -----------------------------------------------------------------------------
# INICIALIZACIÓN DE LA APP DASH
# -----------------------------------------------------------------------------

app = Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    external_scripts=external_scripts,
    title="Liverpool | Portafolio Marcario",
    update_title="Cargando portafolio marcario...",
    use_pages=False,
    suppress_callback_exceptions=True,
)


# -----------------------------------------------------------------------------
# CONFIGURACIÓN DEL CHATBOT ADK
# -----------------------------------------------------------------------------

APP_NAME = "liverpool_dash_chatbot"
USER_ID = "dash_user"
SESSION_ID = "dash_session"

session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

asyncio.run(
    session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
)


async def call_agent_async(message: str) -> str:
    content = types.Content(
        role="user",
        parts=[types.Part(text=message)],
    )

    final_response = ""

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=SESSION_ID,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    return final_response or "No recibí respuesta del agente."


def call_agent(message: str) -> str:
    return asyncio.run(call_agent_async(message))


# -----------------------------------------------------------------------------
# DATA MOCK PARA PROTOTIPO
# -----------------------------------------------------------------------------

marcas = [
    {
        "nombre": "Liverpool",
        "riesgo": "Bajo",
        "renovacion": "2028",
        "tendencia": "+18%",
        "prediccion": "Renovar",
        "badge": "bg-emerald-100 text-emerald-700",
    },
    {
        "nombre": "Suburbia",
        "riesgo": "Medio",
        "renovacion": "2027",
        "tendencia": "+7%",
        "prediccion": "Renovar con monitoreo",
        "badge": "bg-orange-100 text-orange-700",
    },
    {
        "nombre": "Boutique Moda",
        "riesgo": "Alto",
        "renovacion": "2026",
        "tendencia": "-11%",
        "prediccion": "Revisar",
        "badge": "bg-rose-100 text-rose-700",
    },
    {
        "nombre": "Marketplace Liverpool",
        "riesgo": "Medio",
        "renovacion": "2029",
        "tendencia": "+24%",
        "prediccion": "Renovar",
        "badge": "bg-orange-100 text-orange-700",
    },
]

registro_marca = {
    "Revenue": 8941.078577753173,
    "Antigüedad de la Marca": 7.0,
    "Avg Market Share": 8.528,
    "Calificación Promedio de Productos": 4.02900644475073,
    "Participación de Mercado (%)": 60.0,
    "Crecimiento Total Sales": 0.15
}


# -----------------------------------------------------------------------------
# COMPONENTES REUTILIZABLES
# -----------------------------------------------------------------------------

def metric_card(label, value, helper, icon="✦"):
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.P(label, className="text-sm text-slate-500"),
                            html.P(value, className="mt-2 text-3xl font-bold text-slate-900"),
                            html.P(helper, className="mt-1 text-xs text-slate-400"),
                        ]
                    ),
                    html.Div(
                        icon,
                        className="rounded-2xl bg-gradient-to-br from-fuchsia-500 to-orange-400 p-3 text-white shadow-sm text-xl",
                    ),
                ],
                className="flex items-start justify-between gap-4",
            )
        ],
        className="rounded-2xl border-0 bg-white/85 p-5 shadow-sm backdrop-blur",
    )


def tableau_placeholder(title, subtitle, tableau_url):
    """
    Contenedor real para dashboards Tableau embebidos.
    """

    return html.Div(
        children=[
            html.Div(
                className="h-2 bg-gradient-to-r from-fuchsia-500 via-pink-300 to-orange-400"
            ),

            html.Div(
                children=[

                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.H3(
                                        title,
                                        className="text-lg font-bold text-slate-900"
                                    ),

                                    html.P(
                                        subtitle,
                                        className="text-sm text-slate-500"
                                    ),
                                ]
                            ),

                            html.Span(
                                "Tableau Live",
                                className=(
                                    "rounded-full bg-pink-50 px-3 py-1 "
                                    "text-xs font-semibold text-fuchsia-700"
                                ),
                            ),
                        ],
                        className="mb-5 flex items-center justify-between gap-3",
                    ),

                    html.Div(
                        children=[
                            html.Iframe(
                                src=tableau_url,
                                style={
                                    "width": "100%",
                                    "height": "720px",
                                    "border": "none",
                                    "borderRadius": "24px",
                                    "backgroundColor": "white",
                                },
                            )
                        ],
                        className=(
                            "overflow-hidden rounded-3xl border "
                            "border-pink-100 bg-white"
                        ),
                    ),
                ],
                className="p-6",
            ),
        ],
        className="overflow-hidden rounded-3xl border-0 bg-white shadow-sm",
    )

def manual_prediction_input(label, input_id, value, step="any"):
    return html.Div(
        children=[
            html.Label(label, className="mb-2 block text-sm font-semibold text-slate-700"),
            dcc.Input(
                id=input_id,
                type="text",
                value=str(value),
                placeholder="0.00",
                className="w-full rounded-2xl border border-pink-100 bg-pink-50/40 px-7 text-base outline-none focus:ring-4 focus:ring-pink-100",
                style={"height": "44px", "display": "flex", "alignItems": "center"},
            ),
        ]
    )


def chatbot_panel():
    return html.Div(
        children=[
            dcc.Store(id="chat-history", data=[]),
            dcc.Store(id="chat-open", data=False),
            dcc.Store(id="chat-pending-message", data=None),

            # Este intervalo permite que Dash renderice primero el mensaje
            # "Cargando..." y después ejecute la llamada al agente.
            dcc.Interval(
                id="chat-agent-trigger",
                interval=300,
                n_intervals=0,
                disabled=True,
            ),

            html.Button(
                "💬",
                id="chat-toggle",
                n_clicks=0,
                className=(
                    "fixed bottom-6 right-6 z-50 grid h-16 w-16 place-items-center "
                    "rounded-full bg-fuchsia-600 text-3xl text-white shadow-xl "
                    "hover:bg-fuchsia-700 transition-all duration-200"
                ),
            ),

            html.Div(
                id="chat-window",
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                children=[
                                    html.H3(
                                        "Asistente marcario",
                                        className="text-base font-bold text-slate-900",
                                    ),
                                    html.P(
                                        "Liverpool AI",
                                        className="text-xs text-slate-500",
                                    ),
                                ]
                            ),
                            html.Button(
                                "×",
                                id="chat-close",
                                n_clicks=0,
                                className="rounded-full px-3 py-1 text-xl font-bold text-slate-400 hover:bg-pink-50 hover:text-fuchsia-600",
                            ),
                        ],
                        className="flex items-center justify-between border-b border-pink-100 px-5 py-4",
                    ),
                    dcc.Loading(
                        children=[
                            html.Div(
                                id="chat-messages",
                                className="h-80 space-y-3 overflow-y-auto bg-slate-50 p-4",
                            ),
                    ]),
                    html.Div(
                        children=[
                            dcc.Textarea(
                                id="chat-input",
                                placeholder="Escribe tu pregunta...",
                                value="",
                                className=(
                                    "h-12 max-h-28 w-full resize-none rounded-2xl "
                                    "border border-pink-100 bg-white px-4 py-3 text-sm "
                                    "leading-5 outline-none focus:ring-4 focus:ring-pink-100 transition-all"
                                ),
                                style={
                                    "lineHeight": "20px",
                                    "paddingTop": "12px",
                                    "paddingBottom": "12px",
                                    "wordWrap": "break-word",
                                    "overflowWrap": "break-word",
                                    "whiteSpace": "pre-wrap",
                                },
                            ),
                            html.Button(
                                "Enviar",
                                id="chat-send",
                                n_clicks=0,
                                className="rounded-full bg-fuchsia-600 px-5 py-3 text-sm font-semibold text-white hover:bg-fuchsia-700 transition-all disabled:opacity-60",
                            ),
                        ],
                        className="flex items-end gap-2 border-t border-pink-100 bg-white p-4",
                    ),
                ],
                className=(
                    "fixed bottom-24 right-6 z-50 hidden w-[360px] overflow-hidden "
                    "rounded-3xl bg-white shadow-2xl ring-1 ring-pink-100 md:w-[420px] transition-all"
                ),
            ),
        ]
    )

# -----------------------------------------------------------------------------
# SECCIONES / PÁGINAS VISUALES
# -----------------------------------------------------------------------------

def hero_banner():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.P(
                        "Inteligencia legal + comercial",
                        className="mb-3 inline-flex rounded-full bg-white/20 px-4 py-1 text-sm font-semibold backdrop-blur",
                    ),
                    html.H1(
                        "Centro de renovación marcaria de Liverpool",
                        className="text-4xl font-black tracking-tight md:text-5xl",
                    ),
                    html.P(
                        "Explora y monitorea el desempeño del portafolio marcario de Liverpool mediante indicadores clave, análisis interactivos y herramientas predictivas que facilitan la toma de decisiones estratégicas.",
                        className="mt-4 max-w-2xl text-white/85",
                    ),
                ],
                className="max-w-3xl",
            )
        ],
        className="mb-8 overflow-hidden rounded-[2rem] bg-gradient-to-r from-fuchsia-600 via-pink-400 to-orange-400 p-8 text-white shadow-xl",
    )


def home_page():
    return html.Section(
        children=[
            hero_banner(),
            html.Div(
                children=[
                    metric_card("Marcas activas", "1,248", "Portafolio vigente", "✓"),
                    metric_card("Riesgo alto", "86", "Renovación o litigio", "!"),
                    metric_card("Marca más buscada", "Tabasco Dos Mil", "Google Trends", "⌕"),
                    metric_card("Marca mejor calificada", "Perisur", "Voz del consumidor", "✦"),
                ],
                className="grid gap-4 md:grid-cols-4",
            ),
            html.Div(
                children=[
                    tableau_placeholder(
                        "Riesgo de renovación de marcas",
                        "Mapa de calor por vigencia, valor comercial y criticidad legal",
                        "https://public.tableau.com/views/Priorizacindemarcasenriesgo-LIVERPOOL/Dashboard1?:embed=true&:showVizHome=no",
                    ),
                    tableau_placeholder(
                        "Picos de consulta en Google Trends",
                        "Interés de búsqueda por marca, temporalidad y territorio",
                        "https://public.tableau.com/views/Monitoreodetendencias-LIVERPOOL/Dashboard2?:embed=true&:showVizHome=no",
                    ),
                ],
                className="flex flex-col gap-6",
            ),
        ],
        className="space-y-6",
    )


def batch_upload_card():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div("⬆️", className="rounded-2xl bg-fuchsia-100 p-3 text-xl text-fuchsia-700"),
                    html.Div(
                        children=[
                            html.H3("Carga batch del modelo", className="text-lg font-bold text-slate-900"),
                            html.P(
                                "Sube un archivo para predecir varias marcas a la vez",
                                className="text-sm text-slate-500",
                            ),
                        ]
                    ),
                ],
                className="mb-5 flex items-center gap-3",
            ),
            html.Div(
                children=[
                    html.Div("⬆️", className="mx-auto text-4xl text-fuchsia-600"),
                    html.P("Arrastra el archivo del modelo o dataset", className="mt-3 font-semibold text-slate-800"),
                    html.P("Formatos sugeridos: .pkl, .joblib, .csv o .xlsx", className="mt-1 text-sm text-slate-500"),
                    dcc.Upload(
                        id="upload-prediccion-batch",
                        children=html.Button(
                            "Seleccionar archivo",
                            className="mt-5 rounded-full bg-fuchsia-600 px-6 py-2 text-white hover:bg-fuchsia-700",
                        ),
                        multiple=False,
                    ),
                ],
                className="rounded-3xl border border-dashed border-pink-200 bg-pink-50/60 p-6 text-center",
            ),
            html.Div(
                children=[
                    html.Button(
                        "Generar predicciones",
                        id="btn-generar-predicciones",
                        className="rounded-full bg-fuchsia-600 px-8 py-3 font-semibold text-white shadow-sm hover:bg-fuchsia-700",
                    ),
                ],
                className="mt-6 flex justify-center",
            ),
        
            #  DENTRO DEL LAYOUT
            html.Div(
                id="resultado-batch",
                className="mt-4 max-w-full overflow-x-auto"
            ),
        ],
        className="rounded-3xl border-0 bg-white p-6 shadow-sm",
    )

def manual_prediction_card():
    return html.Div(
        children=[
            html.Div(
                children=[
                    html.Div("◎", className="rounded-2xl bg-orange-100 p-3 text-xl text-orange-600"),
                    html.Div(
                        children=[
                            html.H3("Consulta individual", className="text-lg font-bold text-slate-900"),
                            html.P(
                                "Captura las variables comerciales para predecir renovación",
                                className="text-sm text-slate-500",
                            ),
                        ]
                    ),
                ],
                className="mb-5 flex items-center gap-3",
            ),
            html.Div(
                children=[
                    manual_prediction_input(
                        "Revenue", 
                        "input-revenue", 
                        registro_marca["Revenue"]
                    ),
                    manual_prediction_input(
                        "Antigüedad de la Marca",
                        "input-antiguedad-marca",
                        registro_marca["Antigüedad de la Marca"],
                    ),
                    manual_prediction_input(
                        "Avg Market Share",
                        "input-avg-market-share",
                        registro_marca["Avg Market Share"],
                    ),
                    manual_prediction_input(
                        "Calificación Promedio de Productos",
                        "input-calificacion-promedio",
                        registro_marca["Calificación Promedio de Productos"],
                    ),
                    manual_prediction_input(
                        "Participación de Mercado (%)",
                        "input-participacion-mercado",
                        registro_marca["Participación de Mercado (%)"],
                    ),
                    manual_prediction_input(
                        "Crecimiento Total Sales",
                        "input-crecimiento-total-sales",
                        registro_marca["Crecimiento Total Sales"],
                    ),
                ],
                className="grid gap-7 md:grid-cols-2",
            ),
            html.Div(
                children=[
                    html.Button(
                        "Predecir renovación",
                        id="btn-predecir-individual",
                        className="rounded-full bg-orange-500 px-6 py-3 font-semibold text-white hover:bg-orange-600",
                    ),
                    html.Div(
                        children=[
                            html.P(
                                "Resultado esperado",
                                className="text-xs font-semibold uppercase tracking-wide text-fuchsia-600",
                            ),
                            html.P(
                                "Pendiente de ejecutar modelo",
                                id="resultado-prediccion-individual",
                                className="mt-1 text-sm font-semibold text-slate-800",
                            ),
                        ],
                        className="rounded-2xl bg-pink-50 px-4 py-3",
                    ),
                ],
                className="mt-6 flex flex-col gap-3 md:flex-row md:items-center md:justify-between",
            ),
        ],
        className="rounded-3xl border-0 bg-white p-6 shadow-sm",
    )


def prediction_tableau_card():
    return html.Div(
        
            tableau_placeholder(
                "Resultados de predicción batch",
                "Visualización de los resultados obtenidos al cargar un dataset para predicción masiva",
                "https://public.tableau.com/views/Distribucindesugerenciasderenovacin-LIVERPOOL/Hoja1?:embed=true&:showVizHome=no",
            ),
        className="rounded-3xl border-0 bg-white p-6 shadow-sm",
    )


def predictivo_page():
    return html.Section(
        children=[
            hero_banner(),
            html.Div(
                children=[
                    batch_upload_card(),
                    manual_prediction_card(),
                ],
                className="flex flex-col gap-6",
            ),
            prediction_tableau_card(),
        ],
        className="space-y-6",
    )


def datos_page():
    cards = [
        ("📄", "Despacho legal", "Notas de criterio, recomendaciones y responsables por expediente."),
        ("⚠️", "Alertas regulatorias", "Cambios de estatus, vencimientos y expedientes críticos."),
        ("✓", "Soporte decisional", "Checklist para renovación, abandono, defensa o vigilancia."),
    ]

    return html.Section(
        children=[
            hero_banner(),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                icon,
                                className="mb-5 grid h-14 w-14 place-items-center rounded-2xl bg-gradient-to-br from-fuchsia-500 to-orange-400 text-xl text-white",
                            ),
                            html.H3(title, className="text-lg font-bold text-slate-900"),
                            html.P(description, className="mt-2 text-sm text-slate-500"),
                            html.Button(
                                "Consultar",
                                className="mt-6 rounded-full border border-pink-200 px-5 py-2 text-fuchsia-700 hover:bg-pink-50",
                            ),
                        ],
                        className="rounded-3xl border-0 bg-white p-6 shadow-sm",
                    )
                    for icon, title, description in cards
                ],
                className="grid gap-6 lg:grid-cols-3",
            ),
        ],
        className="space-y-6",
    )


# -----------------------------------------------------------------------------
# HEADER Y FOOTER
# -----------------------------------------------------------------------------

header = html.Header(
    children=[
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.Div(
                            "✦",
                            className="grid h-12 w-12 place-items-center rounded-2xl bg-gradient-to-br from-fuchsia-600 to-orange-400 text-xl text-white shadow-sm",
                        ),
                        html.Div(
                            children=[
                                html.P(
                                    "Liverpool",
                                    className="text-xs font-bold uppercase tracking-[0.22em] text-fuchsia-600",
                                ),
                                html.H1("Portafolio Marcario", className="text-xl font-black text-slate-950"),
                            ]
                        ),
                    ],
                    className="flex items-center gap-3",
                ),
                html.Nav(
                    children=[
                        dcc.Link(
                            "Inicio",
                            href="/",
                            className="rounded-full px-4 py-2 font-bold text-slate-700 hover:bg-pink-50",
                        ),
                        dcc.Link(
                            "Predictivo",
                            href="/predictivo",
                            className="rounded-full px-4 py-2 font-bold text-slate-700 hover:bg-pink-50",
                        ),
                        dcc.Link(
                            "Datos legales",
                            href="/datos",
                            className="rounded-full px-4 py-2 font-bold text-slate-700 hover:bg-pink-50",
                        ),
                    ],
                    className="flex flex-wrap items-center justify-center gap-2",
                ),
            ],
            className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-6 py-4 md:flex-row",
        )
    ],
    className="sticky top-0 z-30 border-b border-pink-100 bg-white/80 backdrop-blur-xl",
)

footer = html.Footer(
    children=[
        html.P("2026 Liverpool - Todos los derechos reservados"),
        html.P("Elaborado por Brenda Servín, Blanca Isidro, Karla Juárez, Max Lázaro y Lisset Segundo"),
    ],
    className="mt-12 p-4 text-center text-sm text-slate-500",
)


# -----------------------------------------------------------------------------
# LAYOUT
# -----------------------------------------------------------------------------

app.layout = html.Div(
    children=[
        dcc.Location(id="url"),
        header,

        html.Main(
            children=html.Div(id="page-content"),
            className="mx-auto min-h-screen max-w-7xl px-6 py-8",
        ),

        chatbot_panel(),

        footer,
    ],
    className="min-h-screen bg-gradient-to-br from-pink-50 via-white to-orange-50 pb-32 text-slate-900",
)


# -----------------------------------------------------------------------------
# CALLBACKS
# -----------------------------------------------------------------------------

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname == "/predictivo":
        return predictivo_page()

    if pathname == "/datos":
        return datos_page()

    return home_page()


@app.callback(
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def update_chat(send_clicks, textarea_submit, user_message, history):
    if not user_message:
        return no_update, no_update

    history = history or []

    clean_message = user_message.strip()

    if not clean_message:
        return no_update, ""

    history.append(
        {
            "role": "user",
            "content": clean_message,
        }
    )

    try:
        bot_response = call_agent(clean_message)
    except Exception as e:
        bot_response = f"Error al consultar el agente: {str(e)}"

    history.append(
        {
            "role": "assistant",
            "content": bot_response,
        }
    )

    return history, ""


@app.callback(
    Output("chat-messages", "children"),
    Input("chat-history", "data"),
)
def render_chat_messages(history):
    history = history or []

    messages = []

    for msg in history:
        is_user = msg["role"] == "user"
        dcc.Loading(
            messages.append(
                html.Div(
                    children=[
                        html.P(
                            "Tú" if is_user else "Asistente",
                            className="mb-1 text-xs font-bold uppercase tracking-wide text-slate-400",
                        ),
                        dcc.Markdown(
                            msg["content"],
                            className=(
                                "rounded-2xl px-4 py-3 text-sm whitespace-pre-wrap "
                                + (
                                    "bg-fuchsia-600 text-white"
                                    if is_user
                                    else "bg-white text-slate-700 shadow-sm"
                                )
                            ),
                        ),
                    ],
                    className="ml-auto max-w-[85%]" if is_user else "mr-auto max-w-[85%]",
                )
            )
        )

    return messages


@app.callback(
    Output("chat-open", "data"),
    Input("chat-toggle", "n_clicks"),
    Input("chat-close", "n_clicks"),
    State("chat-open", "data"),
    prevent_initial_call=True,
)
def toggle_chat(toggle_clicks, close_clicks, is_open):
    ctx = dash.callback_context

    if not ctx.triggered:
        return is_open

    button_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if button_id == "chat-close":
        return False

    if button_id == "chat-toggle":
        return not is_open

    return is_open


@app.callback(
    Output("chat-window", "className"),
    Input("chat-open", "data"),
)
def update_chat_window(is_open):
    base_class = (
        "fixed bottom-24 right-6 z-50 w-[360px] overflow-hidden "
        "rounded-3xl bg-white shadow-2xl ring-1 ring-pink-100 md:w-[420px]"
    )

    if is_open:
        return base_class

    return base_class + " hidden"

# -----------------------------------------------------------------------------
# PREDICCIÓN INDIVIDUAL
# -----------------------------------------------------------------------------

@app.callback(
    Output(
        "resultado-prediccion-individual",
        "children"
    ),

    Input(
        "btn-predecir-individual",
        "n_clicks"
    ),

    State(
        "input-revenue",
        "value"
    ),

    State(
        "input-antiguedad-marca",
        "value"
    ),

    State(
        "input-avg-market-share",
        "value"
    ),

    State(
        "input-calificacion-promedio",
        "value"
    ),

    State(
        "input-participacion-mercado",
        "value"
    ),

    State(
        "input-crecimiento-total-sales",
        "value"
    ),

    prevent_initial_call=True
)
def predecir_renovacion(
    n_clicks,
    revenue,
    antiguedad,
    avg_market_share,
    calificacion,
    participacion,
    crecimiento_total_sales
):

    try:
        # Convertir valores de texto a float, reemplazando comas por puntos
        def parse_number(value):
            if value is None or value == "":
                return 0
            if isinstance(value, str):
                value = value.replace(",", ".")
            return float(value)

        datos = pd.DataFrame([{

            "Avg Market Share": parse_number(avg_market_share),

            "Revenue": parse_number(revenue),

            "Antigüedad de la Marca": int(parse_number(antiguedad)),

            "Calificación Promedio de Productos": parse_number(calificacion),

            "Participación de Mercado (%)": parse_number(participacion),
            
            "Crecimiento Total Sales": parse_number(crecimiento_total_sales),

        }])

        prediccion = modelo_prediccion.predict(
            datos
        )[0]

        prediccion = modelo_prediccion.predict(datos)[0]

        print("Predicción obtenida:", prediccion)
        print("Tipo:", type(prediccion))
        print("Clases:", modelo_prediccion.classes_)

        proba = modelo_prediccion.predict_proba(datos)[0]
        print("Probabilidades:", proba)

        if str(prediccion) in ["1", "True", "Renovar"]:

            return (
                "✅ Se recomienda renovar"
            )

        else:

            return (
                "❌ No se recomienda renovar"
            )

    except Exception as e:

        return f"Error al generar predicción: {str(e)}"

def mapear_prediccion(valor):
    """Mapea valores numéricos de predicción a texto descriptivo"""
    if str(valor) in ["1", "True"]:
        return "Se recomienda renovar"
    else:
        return "No se recomienda renovar"


@app.callback(
    Output("resultado-batch", "children"),
    Input("btn-generar-predicciones", "n_clicks"),
    State("upload-prediccion-batch", "contents"),
    State("upload-prediccion-batch", "filename"),
    prevent_initial_call=True
)
def predecir_batch(n_clicks, contents, filename):

    if contents is None:
        return "⚠️ Debes cargar un archivo."

    try:

        content_type, content_string = contents.split(",")

        decoded = base64.b64decode(content_string)

        if filename.endswith(".csv"):

            df = pd.read_csv(
                io.StringIO(decoded.decode("utf-8"))
            )

        elif filename.endswith(".xlsx"):

            df = pd.read_excel(
                io.BytesIO(decoded)
            )

        else:

            return "⚠️ Formato no soportado."

        predicciones = modelo_prediccion.predict(df)

        df["Prediccion"] = [mapear_prediccion(p) for p in predicciones]

        # Crear tabla estilizada
        df_display = df.head(20).copy()

        return html.Div(
            [
                html.Div(
                    children=[
                        html.Div(
                            children=[
                                html.Span(
                                    f"✅ {len(df)} predicciones generadas",
                                    className="text-sm font-semibold text-emerald-700"
                                ),
                                html.P(
                                    f"Mostrando primeras 20 filas",
                                    className="mt-1 text-xs text-slate-500"
                                ),
                            ]
                        ),
                    ],
                    className="mb-4 flex items-start justify-between"
                ),
                dash_table.DataTable(
                    data=df_display.to_dict('records'),
                    columns=[
                        {"name": col, "id": col}
                        for col in df_display.columns
                    ],
                    style_cell={
                        "padding": "12px",
                        "textAlign": "left",
                        "fontFamily": "Inter, sans-serif",
                        "fontSize": "14px",
                        "color": "#334155",
                        "minWidth": "150px",
                    },
                    style_header={
                        "backgroundColor": "#f1f5f9",
                        "fontWeight": "600",
                        "color": "#1e293b",
                        "borderBottom": "2px solid #e2e8f0",
                        "textTransform": "capitalize",
                        "minWidth": "150px",
                    },
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "#ffffff",
                        },
                        {
                            "if": {"row_index": "even"},
                            "backgroundColor": "#f8fafc",
                        },
                        {
                            "if": {
                                "column_id": "Prediccion",
                                "filter_query": '{Prediccion} contains "Se recomienda renovar"'
                            },
                            "backgroundColor": "#f0fdf4",
                            "color": "#166534",
                            "fontWeight": "500",
                        },
                        {
                            "if": {
                                "column_id": "Prediccion",
                                "filter_query": '{Prediccion} contains "No se recomienda"'
                            },
                            "backgroundColor": "#fef2f2",
                            "color": "#991b1b",
                            "fontWeight": "500",
                        },
                    ],
                    style_table={
                        "overflowX": "auto",
                        "overflowY": "auto",
                        "borderRadius": "16px",
                        "border": "1px solid #e2e8f0",
                        "maxHeight": "600px",
                        "display": "block",
                    },
                    page_size=20,
                    sort_action="native",
                    filter_action="native",
                )
            ],
            className="rounded-2xl bg-white p-5 shadow-sm"
        )

    except Exception as e:

        return f"❌ Error: {str(e)}"

# -----------------------------------------------------------------------------
# EJECUCIÓN LOCAL
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    hostname = socket.gethostname()
    ip_local = socket.gethostbyname(hostname)

    print("\n" + "="*60)
    print(f"Acceso local:   http://127.0.0.1:8050")
    print(f"Acceso red LAN: http://{ip_local}:8050")
    print("="*60 + "\n")

    app.run(
        host="0.0.0.0",
        port=8050,
        debug=True
    )
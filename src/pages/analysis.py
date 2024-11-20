import sys
import os
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)

from src.utils.utils import load_processed_data
from src.app_init import app


df = load_processed_data()
available_years = sorted(df["year"].unique())

# Crear opciones para las productoras
prod_companies = [
    "Columbia Pictures",
    "Dune Entertainment",
    "New Line Cinema",
    "Paramount Pictures",
    "Relativity Media",
    "Village Roadshow Pictures",
    "Twentieth Century Fox Film Corporation",
    "Universal Pictures",
    "Walt Disney Pictures",
    "Warner Bros.",
    "Other_company",
]
prod_companies.sort()
options_dropdown_companies = []
for company in prod_companies:
    options_dropdown_companies.append({"label": company, "value": company})

# Crear opciones para los generos
df_genres = df.copy()
df_genres["genres_list"] = df_genres["genres"].str.split()
df_genres_aux = df_genres.explode("genres_list")
genres = df_genres_aux["genres_list"].unique().tolist()
genres.sort()
options_dropdown_genre = []
for genre in genres:
    options_dropdown_genre.append({"label": genre, "value": genre})

# Crear opciones para los idiomas
languages = df["original_language"].unique().tolist()
languages.sort()
options_dropdown_language = []
for language in languages:
    options_dropdown_language.append({"label": language, "value": language})


def bubbleGraph():
    bubble_layout = html.Div(
        [
            html.Div(
                [
                    html.H6("Año"),
                    dcc.Slider(
                        id="select_year2",
                        included=False,
                        updatemode="drag",
                        tooltip={"always_visible": True},
                        min=min(available_years),
                        max=max(available_years),
                        step=None,
                        value=max(available_years),
                        marks={
                            str(year): str(year) if (year % 5 == 0 or year % 10 == 0) else ""
                            for year in available_years
                        },
                    ),
                ],
                id="slider_year",
            ),
            dcc.Graph(id="bubble", config={"displayModeBar": "hover"}, style={"height": "500px", "width": "100%"}),
        ]
    )
    return bubble_layout


def filterGraph():
    layout_temporal = html.Div(
        children=[
            html.Div(  # fila
                children=[
                    html.Div(  # Bloque izquierdo
                        children=[
                            html.H3(
                                children=["Filtros"],
                                id="primer_grupo",
                                style={"display": "block", "text-align": "center"},
                            ),
                            html.P("Año:", style={"color": "black"}),
                            dcc.RangeSlider(
                                id="select_year1",  # any name you'd like to give it
                                marks=None,
                                step=1,
                                min=min(available_years),
                                max=max(available_years),
                                value=[min(available_years), max(available_years)],  # default value initially chosen
                                dots=False,  # True, False - insert dots, only when step>1
                                allowCross=False,  # True,False - Manage handle crossover
                                updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                included=True,  # True, False - highlight handle
                                tooltip={"always_visible": True, "placement": "bottom"},  # show current slider values
                            ),
                            html.P("Duración (mins):", style={"color": "black"}),
                            dcc.RangeSlider(
                                id="select_runtime",  # any name you'd like to give it
                                marks=None,
                                step=1,
                                min=df["runtime"].min(),
                                max=df["runtime"].max(),
                                value=[df["runtime"].min(), df["runtime"].max()],  # default value initially chosen
                                dots=False,  # True, False - insert dots, only when step>1
                                allowCross=False,  # True,False - Manage handle crossover
                                updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                included=True,  # True, False - highlight handle
                                tooltip={"always_visible": True, "placement": "bottom"},  # show current slider values
                            ),
                            html.P("Popularidad:", style={"color": "black"}),
                            dcc.RangeSlider(
                                id="select_popularity",  # any name you'd like to give it
                                marks=None,
                                step=1,
                                min=df["popularity"].min(),
                                max=df["popularity"].max(),
                                value=[
                                    df["popularity"].min(),
                                    df["popularity"].max(),
                                ],  # default value initially chosen
                                dots=False,  # True, False - insert dots, only when step>1
                                allowCross=False,  # True,False - Manage handle crossover
                                updatemode="mouseup",  # 'mouseup', 'drag' - update value method
                                included=True,  # True, False - highlight handle
                                tooltip={"always_visible": True, "placement": "bottom"},  # show current slider values
                            ),
                            html.P("Géneros:", style={"color": "black"}),
                            dcc.Dropdown(
                                options=options_dropdown_genre,
                                placeholder="Selecciona géneros",
                                id="dropdown_genre",
                                multi=True,
                                value=genres,
                                style={
                                    "display": "block",
                                    "width": "600px",
                                    "height": "120px",
                                    "margin-left": "10px",
                                    "margin-bottom": "10px",
                                },
                            ),
                            html.P("Productoras:", style={"color": "black"}),
                            dcc.Dropdown(
                                options=options_dropdown_companies,
                                placeholder="Selecciona productoras",
                                id="dropdown_company",
                                multi=True,
                                value=prod_companies,
                                style={
                                    "display": "block",
                                    "width": "600px",
                                    "height": "150px",
                                    "margin-left": "10px",
                                    "margin-bottom": "10px",
                                },
                            ),
                            html.P("Idiomas:", style={"color": "black"}),
                            dcc.Dropdown(
                                options=options_dropdown_language,
                                placeholder="Selecciona idiomas",
                                id="dropdown_language",
                                multi=True,
                                value=languages,
                                style={
                                    "display": "block",
                                    "width": "600px",
                                    "height": "100px",
                                    "margin-left": "10px",
                                    "margin-bottom": "10px",
                                },
                            ),
                        ],  # cierra children
                        style={
                            "width": "700px",
                            "height": "600px",
                            "display": "inline-block",
                        },
                    ),  # cierra div bloque izquierdo
                    html.Div(  # Bloque derecho
                        children=[dcc.Graph(id="dropdown_figure", style={"display": "none"})],
                        style={
                            "width": "700px",
                            "height": "600px",
                            "display": "inline-block",
                            "margin-top": "50px",
                        },
                    ),
                ]
            ),
        ],
        style={"font-family": "Arial"},
    )
    return layout_temporal


tab_style = {
    "color": "black",
    "fontSize": "1vw",
    "padding": "0.5vh",
    "backgroundColor": "white",
    "border-bottom": "1px white solid",
    "border-left": "1px white solid",
    "border-right": "1px white solid",
    "border-radius": "5px",
    "border-color": "#55A1FF",
    "height": "50%",
}

tab_selected_style = {
    "fontSize": "1vw",
    "color": "#black",
    "padding": "0.5vh",
    "fontWeight": "bold",
    "backgroundColor": "#D1DCFF",
    "border-bottom": "1px white solid",
    "border-left": "1px white solid",
    "border-right": "1px white solid",
    "border-radius": "5px",
    "height": "50%",
}

analysis_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            id="tabs_analysis",
                            value="evolucion",
                            vertical=True,
                            children=[
                                dcc.Tab(
                                    label="Evolución",
                                    value="evolucion",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                                dcc.Tab(
                                    label="Críticas",
                                    value="criticas",
                                    style=tab_style,
                                    selected_style=tab_selected_style,
                                ),
                            ],
                            style={"width": "auto"},
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(html.Div(id="tabs-example-content")),
            ]
        )
    ],
    fluid=True,
)


def get_conditions(column_name, dropdown):
    cond = df[column_name] == dropdown[0]
    for i in dropdown:
        cond = cond | (df[column_name] == i)
    return cond


@app.callback(Output("tabs-example-content", "children"), [Input("tabs_analysis", "value")])
def render_content(tab):
    if tab == "evolucion":
        return filterGraph()
    elif tab == "criticas":
        return bubbleGraph()


@app.callback(
    Output("dropdown_figure", "figure"),
    Output("dropdown_figure", "style"),
    Input("select_year1", "value"),
    Input("select_runtime", "value"),
    Input("select_popularity", "value"),
    Input("dropdown_genre", "value"),
    Input("dropdown_company", "value"),
    Input("dropdown_language", "value"),
)
def figure_dropdown(
    select_year_value,
    select_runtime_value,
    select_popularity_value,
    dropdown_genre_value,
    dropdown_company_value,
    dropdown_language_value,
):
    if (
        select_year_value
        and dropdown_genre_value
        and dropdown_company_value
        and dropdown_language_value
        and select_runtime_value
        and select_popularity_value
    ):
        # Filtrar el DataFrame según los filtros seleccionados
        df_aux = df.copy()

        # Filtro por año
        if select_year_value:
            df_aux = df_aux[(df_aux["year"] >= select_year_value[0]) & (df_aux["year"] <= select_year_value[1])]

        # Filtro por año
        if select_runtime_value:
            df_aux = df_aux[
                (df_aux["runtime"] >= select_runtime_value[0]) & (df_aux["runtime"] <= select_runtime_value[1])
            ]

        # Filtro por año
        if select_popularity_value:
            df_aux = df_aux[
                (df_aux["popularity"] >= select_popularity_value[0])
                & (df_aux["popularity"] <= select_popularity_value[1])
            ]

        # Filtro por géneros
        if dropdown_genre_value:
            # Si el valor de género es una lista, filtrar filas que contengan alguno de los géneros seleccionados
            df_aux = df_aux[df_aux["genres"].apply(lambda x: any(genre in x.split() for genre in dropdown_genre_value))]

        # Filtro por compañías
        if dropdown_company_value:
            # Mantener filas donde alguna columna de compañía tenga un valor de 1
            mask = df_aux[dropdown_company_value].any(axis=1)
            df_aux = df_aux[mask]

        # Filtro por idioma
        if dropdown_language_value:
            df_aux = df_aux[df_aux["original_language"].isin(dropdown_language_value)]

        # Agrupar por año y calcular la suma de las columnas de interés
        df_aux = df_aux.groupby("year")[["revenue", "budget", "gdp"]].mean().reset_index()

        # Crear la figura
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df_aux["year"],
                y=df_aux["revenue"],
                hoverinfo="x+y",
                mode="lines+markers",
                line=dict(width=1.5, color="#00379A"),
                name="Recaudación",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_aux["year"],
                y=df_aux["budget"],
                hoverinfo="x+y",
                mode="lines+markers",
                line=dict(width=1.5, color="#488BFF"),
                name="Budget",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_aux["year"],
                y=df_aux["gdp"],
                hoverinfo="x+y",
                mode="lines+markers",
                line=dict(width=1.5, color="#FFC861"),
                name="GDP USA",
            )
        )
        fig.update_layout(
            title=f"Recaudación según los filtros",
            xaxis_title="Año",
            yaxis_title="$",
            bargap=0.1,
            legend=dict(bgcolor="white", yanchor="top", y=0.99, xanchor="left", x=0.01),
            font=dict(size=8),
            plot_bgcolor="#F3F6FF",
            paper_bgcolor="#F3F6FF",
        )
        return (fig, {"display": "block"})
    else:
        return (go.Figure(data=[], layout={}), {"display": "none"})


@app.callback(
    Output("bubble", "figure"),
    [Input("select_year2", "value")],
)
def update_bubble(select_year_value):
    df_aux = df[df["year"] == select_year_value]

    # Dividir la columna 'genres' en múltiples géneros y 'explotar' las filas
    df_aux["genres"] = df_aux["genres"].str.split()
    df_aux = df_aux.explode("genres")

    # Paso 3: Agrupar por género y calcular la media de 'vote_average' y 'vote_count' y la suma de 'revenue'
    df_aux_scores = df_aux.groupby("genres")[["vote_average", "vote_count"]].mean().reset_index()
    df_aux_sales = df_aux.groupby("genres")["revenue"].mean().reset_index()

    # Unir los DataFrames de scores y ventas
    df_aux2 = pd.merge(df_aux_scores, df_aux_sales[["genres", "revenue"]], on="genres")

    # Paso 4: Crear el gráfico
    fig = px.scatter(
        df_aux2,
        x="vote_average",
        y="vote_count",
        size="revenue",
        color="genres",
        hover_name="genres",
        log_x=True,
        size_max=60,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )

    # Actualizar el diseño
    fig.update_layout(
        title=f"Puntuaciones y número de votos de usuarios según el Género y Recaudación media para el año {select_year_value}",
        xaxis_title="Puntuación Media",
        yaxis_title="Conteo votos",
        bargap=0.1,
        titlefont={"size": 15},
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
    )

    return fig

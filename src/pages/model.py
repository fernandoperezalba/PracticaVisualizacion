import os
import sys
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import pickle
import numpy as np

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)

from src.utils.utils import load_model_data
from src.app import app
from src.pages.analysis import options_dropdown_genre, options_dropdown_companies, options_dropdown_language
from src.preprocessing.data_processing import extract_date_related_features


# load model input data
df_X, df_y = load_model_data()

# load model and test metrics
model = pickle.load(open(os.path.join(basepath, "data/models/catboost_model.pkl"), "rb"))
df_metrics = pd.read_csv(os.path.join(basepath, "data/models/df_metrics.csv"))

# load gdp forecast data
gdp_forecast = pd.read_csv(os.path.join(basepath, "data/raw/GDP_forecast.csv"))

# Crear opciones para los países
prod_countries = ["Canada", "France", "Germany", "Other_country", "United Kingdom", "United States of America"]
prod_countries.sort()
options_dropdown_countries = []
for country in prod_countries:
    options_dropdown_countries.append({"label": country, "value": country})


def predict_vs_real_train():
    # Predict on training set
    y_pred = model.predict(df_X)
    df_pred_test = pd.DataFrame(y_pred)
    df_pred_test.columns = ["revenue"]
    df_pred_test["date"] = pd.to_datetime(df_X[["year", "month", "day"]]).reset_index(drop=True)
    df_y["date"] = pd.to_datetime(df_X[["year", "month", "day"]]).reset_index(drop=True)

    df_aux = df_pred_test.groupby(df_pred_test["date"].dt.year)["revenue"].mean().reset_index()[:-1]
    df_aux2 = df_y.groupby(df_y["date"].dt.year)["revenue"].mean().reset_index()[:-1]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_aux["date"],
            y=df_aux["revenue"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=1, color="#4D4D4D"),
            name="Recaudación predicha",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_aux2["date"],
            y=df_aux2["revenue"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=1, color="#00379A"),
            name="Recaudación real",
        )
    )

    fig.update_layout(
        title="Recaudación por año - real vs. predicción en entrenamiento",
        xaxis_title="Año",
        yaxis_title="Recaudación ($))",
        legend=dict(bgcolor="white", yanchor="top", y=0.99, xanchor="left", x=0.01),
        font=dict(size=8),
        margin=dict(l=0, r=0, b=0, t=30, pad=4),
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
    )

    return fig


# Variable importance in model
def varImportanceBar():
    importancia = pd.DataFrame()
    importancia["nombre"] = model.feature_names_
    importancia["importancia"] = model.feature_importances_
    importancia.set_index("nombre", inplace=True)
    importancia = importancia.sort_values(by="importancia", ascending=True)
    importancia = importancia.tail(10)
    data = [
        go.Bar(
            x=importancia["importancia"],
            y=importancia.index,
            marker_color="#488BFF",
            orientation="h",
            name="Importancia de variables",
        )
    ]

    layout = go.Layout(
        title="Importancia de variables",
        xaxis_title="Ganancia en el modelo",
        yaxis_title="Variable",
        margin=dict(l=0, r=0, b=0, t=30, pad=4),
        font=dict(size=8),
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
    )

    fig = go.Figure(data=data, layout=layout)

    return fig


# Model Layout
model_layout = html.Div(
    children=[
        html.Div(  # Primera fila modelo
            children=[
                html.H6(
                    children=["Análisis del modelo"],
                    id="titulo_primera_fila_modelo",
                    style={"font-size": "1.8vw", "text-align": "left", "display": "block", "color": "#363636"},
                ),
                html.Div(  # gráficos de esta fila
                    children=[
                        html.Div(  # Bloque izquierdo modelo
                            children=[
                                dcc.Graph(
                                    figure=predict_vs_real_train(),
                                    id="pred_vs_real_figure",
                                    style={
                                        "display": "block",
                                        "height": "20vw",
                                        "margin-top": "1%",
                                        "margin-left": "1%",
                                        "margin-right": "1%",
                                        "margin-bottom": "1%",
                                        "width": "95%",
                                    },
                                ),
                            ],
                            style={
                                "width": "46%",
                                "height": "22vw",
                                "border-style": "ridge",
                                "border-color": "#488BFF",
                                "display": "inline-block",
                                "padding-left": "1%",
                                "padding-bottom": "1%",
                                # "background-color": "#D1DCFF",
                            },
                        ),
                        html.Div(  # Bloque medio modelo
                            children=[
                                dcc.Graph(
                                    figure=varImportanceBar(),
                                    id="var_importance_bar_figure",
                                    style={
                                        "display": "block",
                                        "height": "20vw",
                                        "margin-top": "1%",
                                        "margin-left": "1%",
                                        "margin-right": "1%",
                                        "margin-bottom": "1%",
                                        "width": "95%",
                                    },
                                ),
                            ],
                            style={
                                "width": "48%",
                                "height": "22vw",
                                "border-style": "ridge",
                                "border-color": "#488BFF",
                                "display": "inline-block",
                                "margin-left": "1.2%",
                                "padding-left": "1%",
                                "padding-bottom": "1%",
                                # "background-color": "white",
                            },
                        ),
                        html.Div(  # Bloque derecho modelo
                            children=[
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.P("MAPE", style={"color": "#00379A"}),
                                                        html.P(
                                                            f"{df_metrics['mape'].iloc[0]:.2f}%",
                                                            style={
                                                                "textAlign": "center",
                                                                "color": "#00379A",
                                                                "fontSize": 18,
                                                                "margin-top": "-10px",
                                                                "fontWeight": "bold",
                                                            },
                                                        ),
                                                    ],
                                                    id="mape",
                                                    className="create_container1_1",
                                                ),
                                            ],
                                            width="auto",
                                        ),
                                    ],
                                    justify="between",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.P("R2", style={"color": "#00379A"}),
                                                        html.P(
                                                            f"{df_metrics['r2'].iloc[0]:.2f}",
                                                            style={
                                                                "textAlign": "center",
                                                                "color": "#00379A",
                                                                "fontSize": 18,
                                                                "margin-top": "-10px",
                                                                "fontWeight": "bold",
                                                            },
                                                        ),
                                                    ],
                                                    id="r2",
                                                    className="create_container1_1",
                                                ),
                                            ],
                                            width="auto",
                                        ),
                                    ],
                                    justify="between",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.P("MAE", style={"color": "#00379A"}),
                                                        html.P(
                                                            f"${df_metrics['mae'].iloc[0]/1000000:.0f}M",
                                                            style={
                                                                "textAlign": "center",
                                                                "color": "#00379A",
                                                                "fontSize": 18,
                                                                "margin-top": "-10px",
                                                                "fontWeight": "bold",
                                                            },
                                                        ),
                                                    ],
                                                    id="mae",
                                                    className="create_container1_1",
                                                ),
                                            ],
                                            width="auto",
                                        ),
                                    ],
                                    justify="between",
                                ),
                            ],
                            style={
                                "width": "10%",
                                "height": "22vw",
                                "border-style": "ridge",
                                "border-color": "#488BFF",
                                "display": "inline-block",
                                "margin-left": "1.2%",
                                "padding-left": "1%",
                                "padding-bottom": "1%",
                                # "background-color": "white",
                            },
                        ),
                    ],
                    style={"display": "flex"},
                ),
            ],
            className="create_container2",
        ),
        html.Div(  # Segunda fila modelo
            children=[
                html.H6(
                    children=["Predicción de recaudación de una película"],
                    id="titulo_segunda_fila_modelo",
                    style={"font-size": "1.8vw", "text-align": "left", "display": "block", "color": "#363636"},
                ),
                html.Div(  # elementos de esta fila
                    children=[
                        html.Div(  # Bloque izquierdo modelo
                            children=[
                                html.P(
                                    children=["Seleccione las características de la película:"],
                                    id="movie_pred",
                                    style={
                                        "display": "block",
                                        "text-align": "left",
                                    },
                                ),
                                html.Div(
                                    children=[
                                        html.Div(  # Selección fecha
                                            children=[
                                                html.P(
                                                    "Fecha: ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.DatePickerSingle(
                                                    id="selector_fecha",
                                                    date="2024-01-01",  # Fecha inicial
                                                    display_format="YYYY-MM-DD",  # Formato de la fecha mostrada
                                                    min_date_allowed="2024-01-01",  # Fecha mínima seleccionable
                                                    max_date_allowed="2034-12-31",  # Fecha máxima seleccionable
                                                    className="custom-date-picker",
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección budget
                                            children=[
                                                html.P(
                                                    "Budget ($miles): ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.Input(
                                                    id="selector_budget",
                                                    type="number",
                                                    placeholder="Introduzca budget",
                                                    min=1,
                                                    max=500000,
                                                    step=1,
                                                    style={
                                                        "width": "15vw",
                                                        "padding-top": "0%",
                                                        "padding-bottom": "0%",
                                                        "height": "2vw",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección géneros
                                            children=[
                                                html.P("Géneros: ", style={"height": "auto", "margin-bottom": "auto"}),
                                                dcc.Dropdown(
                                                    options=options_dropdown_genre,
                                                    placeholder="Seleccione géneros",
                                                    id="selector_generos",
                                                    multi=True,
                                                    style={
                                                        "display": "block",
                                                        "width": "15vw",
                                                        "height": "60px",
                                                        # "margin-left": "10px",
                                                        "margin-bottom": "10px",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección popularidad
                                            children=[
                                                html.P(
                                                    "Popularidad: ", style={"height": "auto", "margin-bottom": "auto"}
                                                ),
                                                dcc.Input(
                                                    id="selector_popu",
                                                    type="number",
                                                    placeholder="Introduzca popularidad",
                                                    min=0,
                                                    max=900,
                                                    step=1,
                                                    style={
                                                        "width": "15vw",
                                                        "padding-top": "0%",
                                                        "padding-bottom": "0%",
                                                        "height": "2vw",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección runtime
                                            children=[
                                                html.P(
                                                    "Runtime (mins): ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.Input(
                                                    id="selector_runtime",
                                                    type="number",
                                                    placeholder="Introduzca runtime",
                                                    min=1,
                                                    max=250,
                                                    step=1,
                                                    style={
                                                        "width": "15vw",
                                                        "padding-top": "0%",
                                                        "padding-bottom": "0%",
                                                        "height": "2vw",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección Productoras
                                            children=[
                                                html.P(
                                                    "Productoras: ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.Dropdown(
                                                    options=options_dropdown_companies,
                                                    placeholder="Seleccione productoras",
                                                    id="selector_prods",
                                                    multi=True,
                                                    style={
                                                        "display": "block",
                                                        "width": "15vw",
                                                        "height": "60px",
                                                        # "margin-left": "10px",
                                                        "margin-bottom": "10px",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección idioma
                                            children=[
                                                html.P(
                                                    "Idioma: ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.Dropdown(
                                                    options=options_dropdown_language,
                                                    placeholder="Seleccione idioma",
                                                    id="selector_idioma",
                                                    style={
                                                        "display": "block",
                                                        "width": "15vw",
                                                        "height": "3vw",
                                                        # "margin-left": "10px",
                                                        "margin-bottom": "10px",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección países
                                            children=[
                                                html.P("Países: ", style={"height": "auto", "margin-bottom": "auto"}),
                                                dcc.Dropdown(
                                                    options=options_dropdown_countries,
                                                    placeholder="Seleccione países",
                                                    id="selector_countries",
                                                    multi=True,
                                                    style={
                                                        "display": "block",
                                                        "width": "15vw",
                                                        "height": "60px",
                                                        # "margin-left": "10px",
                                                        "margin-bottom": "10px",
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Selección gdp
                                            children=[
                                                html.P(
                                                    "Expectativa economía: ",
                                                    style={"height": "auto", "margin-bottom": "auto"},
                                                ),
                                                dcc.RadioItems(
                                                    id="selector_gdp",
                                                    labelStyle={"display": "inline-block", "font-size": 14},
                                                    value="Esperada",
                                                    options=["Mal", "Esperada", "Bien"],
                                                    style={"padding-top": "5%"},
                                                    className="dcc_compon",
                                                    inputStyle={
                                                        "margin-left": "1px",  # Espacio entre el círculo y el texto
                                                    },
                                                ),
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                        html.Div(  # Botón
                                            children=[
                                                html.Button(  # botón
                                                    children=["Calcular"],
                                                    id="boton_predict",
                                                    title="Calcular",
                                                    n_clicks=0,
                                                    style={
                                                        "background-color": "#488BFF",
                                                        "color": "white",
                                                        "height": "2vw",
                                                        "width": "10vw",
                                                        "font-size": "1vw",
                                                        "border-style": "none",
                                                        "cursor": "pointer",
                                                        "margin-top": "2.8vw",
                                                        "textAlign": "center",
                                                        "line-height": "2vw",
                                                    },
                                                )
                                            ],
                                            style={"margin-right": "1.5%"},
                                        ),
                                    ],
                                    style={"display": "flex", "flexWrap": "wrap"},
                                ),
                            ],
                            style={
                                "width": "48%",
                                "height": "30vw",
                                "border-style": "ridge",
                                "border-color": "#488BFF",
                                "display": "inline-block",
                                "padding-left": "1%",
                                "padding-bottom": "1%",
                                # "background-color": "white",
                            },
                        ),
                        html.Div(  # Bloque derecho modelo
                            children=[],
                            id="prediction",
                            style={
                                "width": "48%",
                                "height": "30vw",
                                "border-style": "ridge",
                                "border-color": "#488BFF",
                                "display": "inline-block",
                                "margin-left": "1.2%",
                                "padding-left": "1%",
                                "padding-bottom": "1%",
                                # "background-color": "white",
                            },
                        ),
                    ],
                    style={"display": "flex", "flex-direction": "row"},
                ),
            ],
            className="create_container2",
        ),
    ]
)


@app.callback(
    Output("prediction", "children"),
    Output("prediction", "style"),
    Input("boton_predict", "n_clicks"),
    State("selector_fecha", "date"),
    State("selector_budget", "value"),
    State("selector_generos", "value"),
    State("selector_popu", "value"),
    State("selector_runtime", "value"),
    State("selector_prods", "value"),
    State("selector_idioma", "value"),
    State("selector_countries", "value"),
    State("selector_gdp", "value"),
)
def prediction_callback(
    n_clicks, fecha, budget, genres, popularity, runtime, prod_companies, language, prod_countries, gdp_expectation
):
    option_list = [
        fecha,
        budget,
        genres,
        popularity,
        runtime,
        prod_companies,
        language,
        prod_countries,
        gdp_expectation,
    ]
    if (None in option_list) or (option_list == []):
        return (html.H1(""), {"display": "none"})
    else:
        testX = pd.DataFrame(columns=df_X.columns, index=[0])
        testX["budget"] = budget
        testX["popularity"] = popularity
        testX["runtime"] = runtime
        testX["date"] = fecha
        testX = extract_date_related_features(testX, "date")
        testX.drop(columns="date", inplace=True)
        for genre in genres:
            testX[genre] = 1
        for company in prod_companies:
            testX[company] = 1
        for country in prod_countries:
            testX[country] = 1
        testX[language] = 1

        print(testX)

        gdp_forecast_year = gdp_forecast[gdp_forecast["Year"] == testX["year"].iloc[0]]["GDP_forecast"].iloc[0]

        if gdp_expectation == "Esperada":
            testX["gdp"] = gdp_forecast_year
        elif gdp_expectation == "Mal":
            testX["gdp"] = gdp_forecast_year * (1 - 0.05)
        else:
            testX["gdp"] = gdp_forecast_year * (1 + 0.05)

        testX.fillna(0, inplace=True)

        revenue_pred = model.predict(testX)[0]

        # Calcular la diferencia entre la recaudación predicha y el presupuesto
        difference = revenue_pred - budget * 1000

        # Determinar el color basado en la diferencia
        if difference < 0:
            difference_color = "red"  # Si la diferencia es negativa, el color será rojo
        elif 0 <= difference < budget * 1000:
            difference_color = (
                "orange"  # Si la recaudación es menor que el doble pero superior al presupuesto, color naranja
            )
        else:
            difference_color = "green"  # Si la recaudación es mayor al doble, el color será verde

        # Formatear la recaudación predicha con comas y dos decimales
        formatted_revenue = f"{revenue_pred:,.2f}"

        # Formatear la diferencia
        formatted_difference = f"{difference:,.2f}"

        # Devolver el layout con el mensaje de recaudación y la diferencia con el color
        return (
            html.Div(
                children=[
                    html.P(
                        ["Recaudación predicha:"], style={"text-align": "center", "font-size": "14", "color": "black"}
                    ),
                    html.H1(
                        children=[f"${formatted_revenue}"],
                        style={"text-align": "center", "font-size": "3vw", "color": "#00379A"},
                    ),
                    html.P(
                        ["Diferencia con el presupuesto:"],
                        style={"text-align": "center", "font-size": "14", "color": "black"},
                    ),
                    html.H3(
                        children=[f"${formatted_difference}"],
                        style={"text-align": "center", "font-size": "2vw", "color": difference_color},
                    ),
                ]
            ),
            {
                "width": "48%",
                "height": "30vw",
                "border-style": "ridge",
                "border-color": "#488BFF",
                "display": "inline-block",
                "margin-left": "1.2%",
                "padding-left": "1%",
                "padding-bottom": "1%",
                # "background-color": "white",
            },
        )

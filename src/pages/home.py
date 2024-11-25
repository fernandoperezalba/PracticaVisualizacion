from app import app
from src.utils.utils import load_processed_data
import os
import sys
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash_bootstrap_components as dbc
import numpy as np
import plotly.figure_factory as ff
from plotly.colors import n_colors

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)


df = load_processed_data()
available_years = sorted(df["year"].unique())


def topGenres():

    # Obtener top 10 géneros
    df["genres_list"] = df["genres"].str.split()
    genre_revenue_df = df.explode("genres_list")[["genres_list", "revenue"]]
    genre_avg_revenue = genre_revenue_df.groupby(
        "genres_list")["revenue"].mean().reset_index()
    top_10_genres_revenues = genre_avg_revenue.sort_values(
        by="revenue", ascending=True).tail(10)

    # Filtrar el DataFrame original para contener solo el top 10 géneros
    top_10_genres_names = top_10_genres_revenues["genres_list"]
    top_10_genres = genre_revenue_df[genre_revenue_df["genres_list"].isin(
        top_10_genres_names)]

    fig = go.Figure()
    # Añadir un box plot para cada género
    for genre in top_10_genres_names:
        genre_data = top_10_genres[top_10_genres["genres_list"]
                                   == genre]["revenue"]

        fig.add_trace(
            go.Box(
                x=genre_data,
                name=genre,
                boxmean=True,
                marker_color="#55A1FF",
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title="Top 10 Géneros de Películas",
        yaxis_title="Género",
        xaxis_title="Recaudación",
        template="plotly_white",
        width=800,
        showlegend=False,
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
        font=dict(size=8),
        margin=dict(l=0, r=0, b=0, t=40, pad=4),
    )

    return fig


def topDirectors():
    director_avg_revenue = df.groupby(
        "director")["revenue"].mean().reset_index()
    top10directors = director_avg_revenue.sort_values(by="revenue").tail(10)

    # Crear el gráfico de barras con plotly.graph_objects
    fig = go.Figure()

    fig.add_trace(
        go.Bar(y=top10directors["director"], x=top10directors["revenue"], marker=dict(
            color="#55A1FF"), orientation="h")
    )

    # Personalizar el gráfico
    fig.update_layout(
        title="Top 10 Directores",
        yaxis_title="Director",
        xaxis_title="Recaudación Media",
        width=800,
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
        font=dict(size=8),
        margin=dict(l=0, r=0, b=0, t=40, pad=4),
    )

    return fig


def correlationMap():
    df_aux = df[
        ["revenue", "budget", "popularity", "runtime", "vote_average",
            "vote_count", "year", "month", "weekday", "gdp"]
    ]
    df_num = df_aux.select_dtypes(include="number")
    corr = df_num.corr()
    z = round(corr, 4).values
    z = np.flip(z, axis=0)

    x = [x for x in df_num.columns]
    y = [x for x in df_num.columns]
    y.reverse()

    z_text = [[str(y) for y in x] for x in z]

    custom_colorscale = [
        [0.0, "#D1DCFF"],  # Color inicial
        [0.5, "#488BFF"],  # Color intermedio
        [1.0, "#00379A"],  # Color final
    ]

    heatmap = ff.create_annotated_heatmap(
        z, x=x, y=y, annotation_text=z_text, colorscale=custom_colorscale)

    heatmap.update_layout(
        title="Matriz de correlación",
        title_y=0.95,
        titlefont={"size": 15},
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
        font=dict(size=8),
        margin=dict(l=0, r=0, b=0, t=70, pad=4),
    )

    return heatmap


def productionCompanies():
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
    ]
    use_cols = prod_companies + ["revenue"]
    df_prod_companies = df[use_cols]
    df_long = df_prod_companies.melt(
        id_vars=["revenue"], value_vars=prod_companies, var_name="production_company", value_name="present"
    )

    # Filtrar filas donde 'present' es 1
    df_long = df_long[df_long["present"] == 1]

    # Step 2: Add a new column for the grouped category
    df_long["group"] = df_long["production_company"].replace(
        {"Twentieth Century Fox Film Corporation": "Disney Group",
            "Walt Disney Pictures": "Disney Group"}
    )

    # Step 3: Combine group and individual data
    # Create a combined DataFrame that includes both individual and group categories
    df_combined = pd.concat(
        [
            df_long,  # Individual companies
            df_long[df_long["group"] == "Disney Group"].assign(
                production_company="Disney Group"),  # Group category
        ]
    )

    # Step 4: Group by company and calculate metrics
    df_grouped = (
        df_combined.groupby("production_company")
        .agg(count=("revenue", "size"), avg_revenue=("revenue", "mean"))
        .reset_index()
    )

    rangeColors = n_colors("rgb(0,29,80)", "rgb(85,161,255)",
                           len(df_grouped), colortype="rgb")
    i = 0
    fig = go.Figure()
    # Add scatter points for each company
    for _, row in df_grouped.iterrows():
        position = "top center"
        if row["production_company"] in ["Universal Pictures", "Warner Bros."]:
            position = "top left"
        elif row["production_company"] == "Dune Entertainment":
            position = "top right"
        elif row["production_company"] == "Village Roadshow Pictures":
            position = "bottom right"

        fig.add_trace(
            go.Scatter(
                x=[row["count"]],
                y=[row["avg_revenue"]],
                mode="markers+text",
                text=[row["production_company"]],
                textposition=position,
                marker=dict(size=7),
                name=row["production_company"],
                marker_color=rangeColors[i],
            )
        )
        i += 1

    # Update layout
    fig.update_layout(
        title="Productoras: Número de Películas vs Recaudación Media",
        xaxis_title="Número de Películas",
        yaxis_title="Recaudación Media",
        legend_title="Productora",
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
        font=dict(size=10),
        margin=dict(l=0, r=0, b=40, t=50),
        showlegend=False,
    )

    return fig


home_layout = html.Div(
    [  # cuadro de mandos
        html.Div(
            [  # parte de arriba
                html.Div(
                    [  # fila 1.1 con slider
                        html.Div(
                            [
                                html.H6("Año"),
                                dcc.Slider(
                                    id="select_year",
                                    included=False,
                                    updatemode="drag",
                                    tooltip={"always_visible": True},
                                    min=min(available_years),
                                    max=max(available_years),
                                    step=None,
                                    value=max(available_years),
                                    marks={str(year): str(year) if (
                                        year % 5 == 0) else "" for year in available_years},
                                ),
                            ],
                            id="barra_year",
                        )
                    ]
                ),
                html.Div(
                    [  # fila 1.2 con KPIs
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.P("Recaudación Total", style={
                                                   "color": "#00379A"}),
                                            html.Div(id="valorRevenueTotal"),
                                        ],
                                        id="revenue_total",
                                        className="create_container1",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.P("Recaudación Media", style={
                                                   "color": "#00379A"}),
                                            html.Div(id="valorRevenueMean"),
                                        ],
                                        id="revenue_mean",
                                        className="create_container1",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.P("Recuento Películas",
                                                   style={"color": "#00379A"}),
                                            html.Div(id="valorCountMovies"),
                                        ],
                                        id="count_movies",
                                        className="create_container1",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.P("Duración Media", style={
                                                   "color": "#00379A"}),
                                            html.Div(id="valorRuntimeMean"),
                                        ],
                                        id="runtime_mean",
                                        className="create_container1",
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    html.Div(
                                        [
                                            html.P("Película nº1", style={
                                                   "color": "#00379A"}),
                                            html.Div(id="valorMovieTop"),
                                        ],
                                        id="top_movie",
                                        className="create_container1",
                                    ),
                                    width="auto",
                                ),
                            ],
                            justify="between",
                        )
                    ],
                    style={"display": "inline-block", "text-align": "center"},
                ),
            ],
            className="create_container2",
        ),
        html.Div(
            [  # gráfico top géneros
                dcc.Graph(
                    figure=topGenres(),
                    id="bar_chartGenres",
                    config={"displayModeBar": "hover"},
                    style={"height": "300px", "width": "100%"},
                )
            ],
            className="create_container2 three columns",
        ),
        html.Div(
            [  # gráfico de barras directores
                html.Div(
                    [  # barras directores
                        dcc.Graph(
                            figure=topDirectors(),
                            id="bar_chartDirectors",
                            config={"displayModeBar": "hover"},
                            style={"height": "300px", "width": "100%"},
                        ),
                    ],
                    style={"height": "300px"},
                )
            ],
            className="create_container2 three columns",
        ),
        html.Div(
            [  # gráfico tiempo
                # selector tiempo
                html.Div(
                    [
                        dcc.RadioItems(
                            id="radio_items1",
                            labelStyle={
                                "display": "inline-block", "font-size": 12},
                            value="year",
                            options=[
                                {"label": "Año", "value": "year"},
                                {"label": "Mes", "value": "month"},
                                {"label": "Día del año", "value": "day_of_year"},
                                {"label": "Día de la semana", "value": "weekday"},
                            ],
                            style={"text-align": "center"},
                            className="dcc_compon",
                        ),
                        dcc.Graph(
                            id="timeGraph",
                            config={"displayModeBar": "hover"},
                            style={"height": "250px", "width": "100%"},
                        ),
                    ],
                    style={"height": "300px"},
                )
            ],
            className="create_container2 six columns",
        ),
        html.Div(
            [  # matriz de correlaciones
                dcc.Graph(
                    figure=correlationMap(),
                    id="correlation",
                    config={"displayModeBar": "hover"},
                    style={"height": "300px", "width": "100%"},
                )
            ],
            className="create_container2 five columns",
            style={"width": "47.8%"},
        ),
        html.Div(
            [  # % cantidad juegos por plataforma
                dcc.Graph(
                    figure=productionCompanies(),
                    id="prodCompanies",
                    config={"displayModeBar": "hover"},
                    style={"height": "300px", "width": "100%"},
                )
            ],
            className="create_container2 five columns",
            style={"width": "47.8%"},
        ),
    ]
)


@app.callback(
    [
        Output("valorRevenueTotal", "children"),
        Output("valorRevenueMean", "children"),
        Output("valorCountMovies", "children"),
        Output("valorRuntimeMean", "children"),
        Output("valorMovieTop", "children"),
    ],
    [Input("select_year", "value")],
)
def update_text(select_year_value):
    df_year = df[df["year"] == select_year_value]
    revenueTotal = df_year["revenue"].sum() / 1000000
    revenueMean = df_year["revenue"].mean() / 1000000
    countMovies = df_year["revenue"].count()
    runtimeMean = df_year["runtime"].mean()
    topMovie = df_year.loc[df_year["revenue"].idxmax()]["title"]
    return [
        html.P(
            f"${revenueTotal:.2f} M",
            style={
                "textAlign": "center",
                "color": "#00379A",
                "fontSize": 20,
                "margin-top": "-10px",
                "fontWeight": "bold",
            },
        ),
        html.P(
            f"${revenueMean:.2f} M",
            style={
                "textAlign": "center",
                "color": "#00379A",
                "fontSize": 20,
                "margin-top": "-10px",
                "fontWeight": "bold",
            },
        ),
        html.P(
            f"{countMovies}",
            style={
                "textAlign": "center",
                "color": "#00379A",
                "fontSize": 20,
                "margin-top": "-10px",
                "fontWeight": "bold",
            },
        ),
        html.P(
            f"{runtimeMean:.2f} mins",
            style={
                "textAlign": "center",
                "color": "#00379A",
                "fontSize": 20,
                "margin-top": "-10px",
                "fontWeight": "bold",
            },
        ),
        html.P(
            f"{topMovie}",
            style={
                "textAlign": "center",
                "color": "#00379A",
                "fontSize": 20,
                "margin-top": "-10px",
                "fontWeight": "bold",
                "word-wrap": "break-word",  # Permite que el texto se divida si es necesario
                "word-break": "break-word",  # Rompe las palabras largas si es necesario
                "max-width": "250px",  # Ajusta el ancho máximo si es necesario
                "line-height": "1.1",  # Reduce el espaciado entre las líneas
                "margin": "0",  # Elimina el margen extra entre las líneas
                "textAlign": "center",  # Centra el texto
            },
        ),
    ]


@app.callback(Output("timeGraph", "figure"), [Input("radio_items1", "value")])
def timeGraph(radio_items1):
    df_aux = df.groupby(df[radio_items1])[
        ["revenue", "budget", "gdp"]].mean().reset_index()[:-1]
    radio_items1_labels = {"year": "Año", "month": "Mes",
                           "day_of_year": "Día del año", "weekday": "Día de la semana"}

    weekday_mapping = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }
    if radio_items1 == "weekday":
        df_aux[radio_items1] = df_aux[radio_items1].map(weekday_mapping)

    data = [
        go.Scatter(
            x=df_aux[radio_items1],
            y=df_aux["revenue"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=1.5, color="#00379A"),
            name="Recaudación",
        ),
        go.Scatter(
            x=df_aux[radio_items1],
            y=df_aux["budget"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=1.5, color="#488BFF"),
            name="Budget",
        ),
        go.Scatter(
            x=df_aux[radio_items1],
            y=df_aux["gdp"],
            hoverinfo="x+y",
            mode="lines+markers",
            line=dict(width=1.5, color="#FFC861"),
            name="GDP USA",
        ),
    ]
    layout = go.Layout(
        title="Evolución Temporal",
        xaxis_title=radio_items1_labels[radio_items1],
        yaxis_title="$",
        legend=dict(bgcolor="white", yanchor="top",
                    y=0.99, xanchor="left", x=0.01),
        font=dict(size=8),
        plot_bgcolor="#F3F6FF",
        paper_bgcolor="#F3F6FF",
        margin=dict(l=0, r=0, b=25, t=25, pad=4),
    )
    return {"data": data, "layout": layout}

import os
import sys
from dash import dcc, html

basepath = os.path.join(os.path.dirname(__file__), "../")
sys.path.insert(1, basepath)

# Import page layouts
from src.pages.home import home_layout
from src.pages.analysis import analysis_layout
from src.pages.model import model_layout
from src.preprocessing.data_processing import data_processing
from src.app import app
from src.modelling.train_model import train_model


df_plots, df_model = data_processing()
model = train_model(df_model)

# Initialize the Dash app
# app = dash.Dash(
#     __name__,
#     suppress_callback_exceptions=True,
#     external_stylesheets=[dbc.themes.LUX],
#     meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
# )

tabs_styles = {
    "flex-direction": "row",
}

tab_style = {
    "color": "#F4F4F4",
    "fontSize": "1.5vw",
    "padding": "0.5vh",
    "backgroundColor": "#55A1FF",
    "border-bottom": "1px white solid",
    "border-left": "1px white solid",
    "border-right": "1px white solid",
    "border-radius": "5px",
    "height": "10%",
}

tab_selected_style = {
    "fontSize": "1.5vw",
    "color": "#242424",
    "padding": "0.5vh",
    "fontWeight": "bold",
    "backgroundColor": "#D1DCFF",
    "border-bottom": "1px white solid",
    "border-left": "1px white solid",
    "border-right": "1px white solid",
    "border-radius": "5px",
    "height": "10%",
}


app.layout = html.Div([html.H1("Hello, Dash!")])

# html.Div(
#     [
#         html.Div(
#             [  # encabezado
#                 html.Div(
#                     children=[html.Img(src="assets/logo.png", height="50px")],
#                     className="one columns",
#                     id="title1",
#                 ),
#                 html.Div(
#                     [
#                         html.Div(
#                             [
#                                 html.H1(
#                                     "Análisis Películas",
#                                     className="fix_label",
#                                     style={"margin-bottom": "0px", "color": "#00379A", "fontWeight": "bold"},
#                                 ),
#                             ]
#                         )
#                     ],
#                     className="two-thirds column",
#                     id="title2",
#                 ),
#             ],
#             id="header",
#             className="row flex-display",
#             style={"margin-bottom": "0px"},
#         ),
#         html.Hr(style={"borderColor": "#660091"}),
#         html.Div(
#             [
#                 dcc.Tabs(
#                     id="tabs-styled-with-inline",
#                     value="home",
#                     children=[
#                         dcc.Tab(
#                             home_layout,
#                             label="Home",
#                             value="home",
#                             style=tab_style,
#                             selected_style=tab_selected_style,
#                         ),
#                         dcc.Tab(
#                             analysis_layout,
#                             label="Análisis",
#                             value="analysis",
#                             style=tab_style,
#                             selected_style=tab_selected_style,
#                         ),
#                         dcc.Tab(
#                             model_layout,
#                             label="Modelo",
#                             value="model",
#                             style=tab_style,
#                             selected_style=tab_selected_style,
#                         ),
#                     ],
#                     style=tabs_styles,
#                     colors={"border": None, "primary": None, "background": None},
#                 ),
#             ]
#         ),
#     ]
# )


if __name__ == "__main__":
    app.run_server(debug=True)

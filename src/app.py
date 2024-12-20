import dash
import dash_bootstrap_components as dbc

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
)
app.title = "Análisis Películas"

server = app.server

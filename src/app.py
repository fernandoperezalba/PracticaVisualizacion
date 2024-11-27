import dash
import dash_bootstrap_components as dbc
from dash import html

# Initialize the Dash app
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX],
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1.0"}],
)
app.title = "Análisis Películas"

# Expose the Flask server
server = app.server

# Placeholder layout
app.layout = html.Div([
    html.H1("Cargando aplicación..."),
    html.P("Espera mientras se inicializa la aplicación.")
])

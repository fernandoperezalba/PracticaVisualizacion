import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from src.modelling.train_model import train_model
from src.preprocessing.data_processing import data_processing
from src.pages.model import model_layout
from src.pages.analysis import analysis_layout
from src.pages.home import home_layout
import os
import sys


print(basepath)

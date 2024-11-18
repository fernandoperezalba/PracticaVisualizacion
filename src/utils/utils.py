import os
import sys
import pandas as pd

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)


def load_processed_data():
    df = pd.read_csv("data/processed/df_plots.csv")
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d")
    return df

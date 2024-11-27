import os
import sys
import pandas as pd

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)


def load_processed_data():
    csv_path = os.path.join(basepath, "data/processed/df_plots.csv")
    df = pd.read_csv(csv_path)
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d")
    return df


def load_model_data():
    csv_path = os.path.join(basepath, "data/processed/df_model.csv")
    df = pd.read_csv(csv_path)
    target = "revenue"
    df_X = df.drop(columns=[target, "vote_average", "vote_count"])
    df_y = df[[target]]
    return df_X, df_y

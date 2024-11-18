import os
import sys
import pandas as pd
import ast
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)


def extract_date_related_features(df, date_col="release_date"):
    df[date_col] = pd.to_datetime(df[date_col])

    # Extract date-related features
    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day
    df["weekday"] = df[date_col].dt.weekday  # 0 = Monday, 6 = Sunday
    df["day_of_year"] = df[date_col].dt.dayofyear

    return df


def process_production_country(df: pd.DataFrame) -> pd.DataFrame:
    df["production_countries"] = df["production_countries"].apply(lambda x: [d["name"] for d in ast.literal_eval(x)])

    # Contar la frecuencia de cada compañía en todas las filas
    all_countries = [country for countries_list in df["production_countries"] for country in countries_list]
    top_countries = [company for company, count in Counter(all_countries).most_common(5)]

    # Crear una lista de compañías que estén en el top 5 o "Otros"
    df["production_countries"] = df["production_countries"].apply(
        lambda x: [company if company in top_countries else "Other_country" for company in x]
    )

    # Aplicar One-Hot Encoding a las compañías seleccionadas (incluyendo "Otros")
    mlb = MultiLabelBinarizer()
    df_countries = pd.DataFrame(mlb.fit_transform(df["production_countries"]), columns=mlb.classes_, index=df.index)
    df_countries.head()

    return df_countries


def process_production_company(df: pd.DataFrame) -> pd.DataFrame:
    df["production_companies"] = df["production_companies"].apply(lambda x: [d["name"] for d in ast.literal_eval(x)])

    # Contar la frecuencia de cada compañía en todas las filas
    all_companies = [company for companies_list in df["production_companies"] for company in companies_list]
    top_companies = [company for company, count in Counter(all_companies).most_common(10)]

    # Crear una lista de compañías que estén en el top 10 o "Otros"
    df["production_companies"] = df["production_companies"].apply(
        lambda x: [company if company in top_companies else "Other_company" for company in x]
    )

    # Aplicar One-Hot Encoding a las compañías seleccionadas (incluyendo "Otros")
    mlb = MultiLabelBinarizer()
    df_companies = pd.DataFrame(mlb.fit_transform(df["production_companies"]), columns=mlb.classes_, index=df.index)
    df_companies.head()

    return df_companies


def process_original_language(df: pd.DataFrame) -> pd.DataFrame:
    # Contar la frecuencia de cada compañía en todas las filas
    top_languages = list(df["original_language"].value_counts().sort_values(ascending=False).head(5).index)

    # Crear una lista de compañías que estén en el top 5 o "Otros"
    df["original_language"] = df["original_language"].apply(
        lambda x: [f"{x}_language" if x in top_languages else "Other_language"]
    )

    # Aplicar One-Hot Encoding a las compañías seleccionadas (incluyendo "Otros")
    mlb = MultiLabelBinarizer()
    df_language = pd.DataFrame(mlb.fit_transform(df["original_language"]), columns=mlb.classes_, index=df.index)
    df_language.head()

    return df_language


def merge_gdp(df_final):
    gdp_data = pd.read_csv("data/raw/USGDP_1900-2024.csv", thousands=",")
    df_gdp = gdp_data[["Year", "Nominal GDP (million of Dollars)"]]
    df_gdp.columns = ["year", "gdp"]
    df_merged = df_final.merge(df_gdp, on="year", how="left")
    return df_merged


def data_processing():
    df = pd.read_csv("data/raw/movie_dataset.csv", index_col="index")

    # dropna and 0 values in budget and revenue (same as NaN in these columns)
    df = df.dropna()
    df = df[(df["budget"] != 0) & (df["revenue"] != 0)].reset_index(drop=True)

    # Remove unused columns:
    df.drop(
        columns=[
            "homepage",
            "id",
            "keywords",
            "overview",
            "status",
            "tagline",
            "original_title",
            "crew",
            "cast",
            "spoken_languages",
        ],
        inplace=True,
    )

    # separate genres into dummies
    df_genres = df["genres"].str.get_dummies(sep=" ")
    df_final = pd.concat([df, df_genres], axis=1)

    # separate language into dummies
    df_language = process_original_language(df)
    df_final = pd.concat([df_final, df_language], axis=1)

    # separate production companies and countries into dummies
    df_companies = process_production_company(df)
    df_final = pd.concat([df_final, df_companies], axis=1)

    df_countries = process_production_country(df)
    df_final = pd.concat([df_final, df_countries], axis=1)

    df_final = extract_date_related_features(df_final)
    df_final = merge_gdp(df_final)
    df_plots = df_final.copy()
    df_model = df_final.drop(
        columns=[
            "genres",
            "production_companies",
            "production_countries",
            "title",
            "director",
            "release_date",
            "original_language",
        ]
    )

    df_plots.to_csv("data/processed/df_plots.csv", index=False)
    df_model.to_csv("data/processed/df_model.csv", index=False)
    return df_plots, df_model


if __name__ == "__main__":
    df_plots, df_model = data_processing()

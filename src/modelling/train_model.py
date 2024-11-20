import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from catboost import CatBoostRegressor
import pickle

basepath = os.path.join(os.path.dirname(__file__), "../../")
sys.path.insert(1, basepath)


def train_model(df: pd.DataFrame):

    target = "revenue"
    df_X = df.drop(columns=[target, "vote_average", "vote_count"])
    df_y = df[[target]]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

    # Initialize the model
    model = CatBoostRegressor(random_state=42, verbose=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    df_metrics = pd.DataFrame({"mse": mse, "r2": r2, "mape": mape, "mae": mae}, index=[0])
    df_metrics.to_csv("data/models/df_metrics.csv", index=False)

    # Train on whole dataset
    model.fit(df_X, df_y)

    # Save trained model
    with open("data/models/catboost_model.pkl", "wb") as file:
        pickle.dump(model, file)

    return model


if __name__ == "__main__":
    df = pd.read_csv("data/processed/df_model.csv")
    model = train_model(df)

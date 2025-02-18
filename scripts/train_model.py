import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import requests
from datetime import datetime

# Constants
LOCATION = {"lat": 37.7749, "lon": -122.4194}  # Example: San Francisco
MODEL_PATH = "models/xgboost_model.pkl"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Weather_Prediction"
REFERENCE_DATA_PATH = "data/reference_data.csv"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def fetch_weather_data(LOCATION=LOCATION):
    API_KEY = os.environ.get("WEATHER_API_KEY")
    API_URL = os.environ.get("WEATHER_API_URL")
    FULL_API_URL = f"{API_URL}{LOCATION['lat']},{LOCATION['lon']}&apikey={API_KEY}"
    response = requests.get(FULL_API_URL)
    return response.json() if response.status_code == 200 else None

def preprocess_data(data):
    forecasts = data["timelines"]["hourly"]
    df = pd.DataFrame(forecasts)
    df["time"] = pd.to_datetime(df["time"])
    df["hour"] = df["time"].dt.hour
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    df["temperature"] = df["values"].apply(lambda x: x["temperature"])
    df.drop(["values", "time"], axis=1, inplace=True)
    return df

def train_model(LOCATION=LOCATION):
    data = fetch_weather_data(LOCATION)
    if not data:
        print("Error fetching data!")
        return
    
    df = preprocess_data(data)
    df.to_csv(REFERENCE_DATA_PATH, index=False)  # Save reference dataset
    X, y = df.drop(columns=["temperature"]), df["temperature"]

    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
    with mlflow.start_run():
        model.fit(X, y)
        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_artifact(REFERENCE_DATA_PATH, "reference_data")
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model trained and saved!")

if __name__ == "__main__":
    train_model()


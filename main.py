import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import streamlit as st
from datetime import datetime
import requests
from evidently.report import Report
from evidently.metrics import DataDriftTable

# Constants
LOCATION = {"lat": 37.7749, "lon": -122.4194}  # Example: San Francisco
MODEL_PATH = "models/xgboost_model.pkl"
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Weather_Prediction"
REFERENCE_DATA_PATH = "data/reference_data.csv"
PRODUCTION_DATA_PATH = "data/production_data.csv"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

@st.cache_data
def load_model():
    return joblib.load(MODEL_PATH)

def predict(hour, day, month):
    model = load_model()
    input_data = np.array([[hour, day, month]])
    prediction = model.predict(input_data)[0]
    
    # Log production data
    new_entry = pd.DataFrame([[hour, day, month]], columns=["hour", "day", "month"])
    new_entry.to_csv(PRODUCTION_DATA_PATH, mode="a", header=not os.path.exists(PRODUCTION_DATA_PATH), index=False)
    
    return float(prediction)

def check_drift():
    if not os.path.exists(REFERENCE_DATA_PATH) or not os.path.exists(PRODUCTION_DATA_PATH):
        st.warning("Reference or production data not available. Unable to check drift.")
        return None

    ref = pd.read_csv(REFERENCE_DATA_PATH)
    curr = pd.read_csv(PRODUCTION_DATA_PATH)
    
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=ref, current_data=curr)
    
    drift_metrics = {
        "drift_score": report.as_dict()["metrics"][0]["drift_score"],
        "drifted_features": report.as_dict()["metrics"][0]["number_of_drifted_columns"]
    }
    
    with mlflow.start_run():
        mlflow.log_metrics(drift_metrics)
    
    return drift_metrics

def main():
    st.title("Weather Forecasting App")
    st.write("Enter details to predict the temperature.")

    hour = st.slider("Hour", 0, 23, 12)
    day = st.slider("Day", 1, 31, datetime.now().day)
    month = st.slider("Month", 1, 12, datetime.now().month)

    if st.button("Predict Temperature"):
        result = predict(hour, day, month)
        st.success(f"Predicted Temperature: {result:.2f} °C")

    # Drift Monitoring Section
    st.sidebar.header("Drift Monitoring")
    if st.sidebar.button("Check Data Drift"):
        drift = check_drift()
        if drift:
            st.sidebar.metric("Overall Drift Score", f"{drift['drift_score']:.2f}")
            st.sidebar.metric("Drifted Features", drift["drifted_features"])
            
            if drift["drifted_features"] > 0:
                st.error("Drift detected! Consider retraining model")
        else:
            st.sidebar.warning("Unable to check drift. Ensure both reference and production data are available.")

    # Drift History
    if st.sidebar.checkbox("Show Drift History"):
        results = mlflow.search_runs()
        if not results.empty:
            history = results[["metrics.drift_score", "metrics.drifted_features"]]
            st.line_chart(history["metrics.drift_score"])
            st.bar_chart(history["metrics.drifted_features"])
        else:
            st.sidebar.info("No drift history available yet.")

if __name__ == "__main__":
    main()

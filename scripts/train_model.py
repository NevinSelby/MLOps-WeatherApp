import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MODEL_PATH = "models/xgboost_model.pkl"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "MLOPS_Weather_Prediction"
REFERENCE_DATA_PATH = "data/reference_data.csv"

# Set MLflow tracking
try:
    mlflow.create_experiment(EXPERIMENT_NAME)
except:
    pass
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def generate_synthetic_weather_data(lat, lon, days=30):
    """Generate synthetic weather data for training when API is not available"""
    print("Generating synthetic weather data for training...")
    
    data = []
    base_date = datetime.now() - timedelta(days=days)
    
    for day_offset in range(days):
        current_date = base_date + timedelta(days=day_offset)
        
        for hour in range(24):
            # Generate realistic temperature patterns
            base_temp = 15 + 10 * np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365)
            hourly_variation = 5 * np.sin(2 * np.pi * hour / 24)
            noise = np.random.normal(0, 2)
            
            temperature = base_temp + hourly_variation + noise
            
            data.append({
                'hour': hour,
                'day': current_date.day,
                'month': current_date.month,
                'lat': lat,
                'lon': lon,
                'temperature': temperature
            })
    
    return pd.DataFrame(data)

def fetch_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API"""
    # Open-Meteo API doesn't require API key
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for weather data
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m',
        'timezone': 'auto',
        'forecast_days': 7
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API request failed with status {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")
        return None

def preprocess_api_data(data, lat, lon):
    """Preprocess data from Open-Meteo API"""
    if not data or 'hourly' not in data:
        return None
    
    hourly_data = data['hourly']
    if not hourly_data or 'time' not in hourly_data or 'temperature_2m' not in hourly_data:
        return None
    
    processed_data = []
    times = hourly_data['time']
    temperatures = hourly_data['temperature_2m']
    
    for i in range(len(times)):
        if i < len(temperatures):
            dt = datetime.fromisoformat(times[i].replace('Z', '+00:00'))
            processed_data.append({
                'hour': dt.hour,
                'day': dt.day,
                'month': dt.month,
                'lat': lat,
                'lon': lon,
                'temperature': temperatures[i]
            })
    
    return pd.DataFrame(processed_data)

def train_model(lat, lon):
    """Train the XGBoost model with weather data"""
    print(f"Training model for location: {lat}, {lon}")
    
    # Try to get real data first
    api_data = fetch_weather_data(lat, lon)
    
    if api_data:
        df = preprocess_api_data(api_data, lat, lon)
        if df is None or df.empty:
            print("Failed to process API data. Using synthetic data.")
            df = generate_synthetic_weather_data(lat, lon)
    else:
        df = generate_synthetic_weather_data(lat, lon)
    
    # Save reference dataset
    os.makedirs(os.path.dirname(REFERENCE_DATA_PATH), exist_ok=True)
    df.to_csv(REFERENCE_DATA_PATH, index=False)
    print(f"Reference data saved to {REFERENCE_DATA_PATH}")
    
    # Prepare features and target
    X = df[['hour', 'day', 'month', 'lat', 'lon']]
    y = df['temperature']
    
    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    # Log training with MLflow
    with mlflow.start_run(run_name="model_training"):
        model.fit(X, y)
        
        # Log model parameters
        mlflow.log_params({
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "data_points": len(df)
        })
        
        # Log model metrics
        train_score = model.score(X, y)
        mlflow.log_metric("train_score", train_score)
        
        # Log the model
        mlflow.xgboost.log_model(model, "xgboost_model")
        mlflow.log_artifact(REFERENCE_DATA_PATH, "reference_data")
        
        print(f"Model training score: {train_score:.4f}")
    
    # Save model locally
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    return model

if __name__ == "__main__":
    # Default to San Francisco coordinates
    lat = float(os.environ.get('DEFAULT_LAT', 37.7749))
    lon = float(os.environ.get('DEFAULT_LON', -122.4194))
    
    print(f"Training model for coordinates: {lat}, {lon}")
    train_model(lat, lon)
    print("Training completed successfully!")

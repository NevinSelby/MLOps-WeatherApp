# WeatherMLOps - Streamlit Cloud Deployment

This is a weather prediction MLOps application deployed on Streamlit Cloud.

## Features

- 🌤️ Weather prediction using multiple ML models
- 📊 Data exploration and analysis
- 🔍 Data drift detection
- 🤖 Model comparison and evaluation
- 📈 Interactive visualizations

## Usage

1. Enter a location in the sidebar
2. Choose between Weather Prediction and Data Exploration
3. Generate forecasts or analyze your data
4. Monitor data drift and retrain models as needed

## Deployment

This app is automatically deployed on Streamlit Cloud. The deployment includes:

- All required Python packages
- MLflow for model tracking
- Interactive data visualization
- Real-time weather data integration

## Data Sources

- Open-Meteo API for real-time weather data
- Local reference and production datasets
- Geocoding via Nominatim

## Model Information

The app uses multiple machine learning models:
- XGBoost (primary)
- Random Forest
- Linear Regression

All models are trained on weather features including temperature, humidity, pressure, wind speed, and more.

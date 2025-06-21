#!/usr/bin/env python3
"""
WeatherMLOps - Streamlit Cloud Optimized Version
Weather prediction MLOps application with robust error handling for cloud deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import joblib
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="WeatherMLOps",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Constants
EXPERIMENT_NAME = "weather_prediction"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# Initialize MLflow with error handling
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_AVAILABLE = True
except Exception as e:
    st.warning(f"MLflow initialization failed: {str(e)}. Some features may be limited.")
    MLFLOW_AVAILABLE = False

@st.cache_data
def load_model():
    """Load the trained model with error handling"""
    try:
        model_path = "models/xgboost_model.pkl"
        if os.path.exists(model_path):
            return joblib.load(model_path)
        else:
            st.warning("Model file not found. Please train a model first.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def fetch_weather_data(lat, lon):
    """Fetch weather data from Open-Meteo API"""
    try:
        url = f"https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,cloud_cover",
            "timezone": "auto"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return None

def extract_weather_features(weather_data):
    """Extract features from weather data"""
    try:
        if not weather_data or 'hourly' not in weather_data:
            return None
        
        hourly_data = weather_data['hourly']
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly_data['time']),
            'temperature': hourly_data['temperature_2m'],
            'humidity': hourly_data['relative_humidity_2m'],
            'pressure': hourly_data['pressure_msl'],
            'wind_speed': hourly_data['wind_speed_10m'],
            'cloud_cover': hourly_data['cloud_cover']
        })
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Add location features
        df['lat'] = weather_data['latitude']
        df['lon'] = weather_data['longitude']
        
        # Handle missing values
        df = df.dropna()
        
        return df
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

def get_location_coordinates(location_name):
    """Get latitude and longitude from location name"""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
        
        geolocator = Nominatim(user_agent="weathermlops")
        location = geolocator.geocode(location_name)
        
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except ImportError:
        st.error("geopy package not available. Please install it.")
        return None, None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None, None, None
    except Exception as e:
        st.error(f"Error getting location coordinates: {str(e)}")
        return None, None, None

def make_prediction(model, features_df):
    """Make weather predictions"""
    try:
        if model is None or features_df is None or features_df.empty:
            return None
        
        # Prepare features for prediction
        feature_columns = ['hour', 'day', 'month', 'lat', 'lon', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
        
        # Check if all required columns exist
        missing_columns = [col for col in feature_columns if col not in features_df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        X = features_df[feature_columns].values
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create results DataFrame
        results_df = features_df.copy()
        results_df['predicted_temperature'] = predictions
        results_df['prediction_error'] = abs(results_df['temperature'] - results_df['predicted_temperature'])
        
        return results_df
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        return None

def create_visualizations(results_df, model=None):
    """Create interactive visualizations"""
    try:
        if results_df is None or results_df.empty:
            return
        
        # Temperature comparison
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['temperature'],
            mode='lines',
            name='Actual Temperature',
            line=dict(color='blue')
        ))
        fig_temp.add_trace(go.Scatter(
            x=results_df['timestamp'],
            y=results_df['predicted_temperature'],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='red', dash='dash')
        ))
        fig_temp.update_layout(
            title='Temperature Prediction vs Actual',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            height=400
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Prediction error distribution
        fig_error = px.histogram(
            results_df,
            x='prediction_error',
            nbins=20,
            title='Prediction Error Distribution'
        )
        fig_error.update_layout(height=400)
        st.plotly_chart(fig_error, use_container_width=True)
        
        # Feature importance (if available)
        if model is not None and hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': ['hour', 'day', 'month', 'lat', 'lon', 'humidity', 'pressure', 'wind_speed', 'cloud_cover'],
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title='Feature Importance'
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error creating visualizations: {str(e)}")

def show_data_exploration_page():
    """Show data exploration page"""
    st.markdown("## 📊 Data Exploration")
    
    # Load data
    ref_data = None
    prod_data = None
    
    try:
        if os.path.exists("data/reference_data.csv"):
            ref_data = pd.read_csv("data/reference_data.csv")
        if os.path.exists("data/production_data.csv"):
            prod_data = pd.read_csv("data/production_data.csv")
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return
    
    # Data overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Reference Data")
        if ref_data is not None:
            st.write(f"Records: {len(ref_data)}")
            st.write(f"Columns: {len(ref_data.columns)}")
            st.dataframe(ref_data.head())
        else:
            st.warning("Reference data not found")
    
    with col2:
        st.markdown("### Production Data")
        if prod_data is not None:
            st.write(f"Records: {len(prod_data)}")
            st.write(f"Columns: {len(prod_data.columns)}")
            st.dataframe(prod_data.head())
        else:
            st.warning("Production data not found")

def main():
    """Main application function"""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ WeatherMLOps</h1>
        <p>Intelligent Weather Forecasting with MLOps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Page navigation
    st.sidebar.markdown("## 📄 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Weather Prediction", "Data Exploration"]
    )
    
    if page == "Data Exploration":
        show_data_exploration_page()
        return
    
    # Weather Prediction Page
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Location input
    location_input = st.sidebar.text_input(
        "Enter location (city, country):",
        placeholder="e.g., New York, USA"
    )
    
    if st.sidebar.button("🌍 Get Weather Forecast", type="primary"):
        if not location_input:
            st.error("Please enter a location")
            return
        
        with st.spinner("Fetching weather data..."):
            # Get coordinates
            lat, lon, address = get_location_coordinates(location_input)
            
            if lat is None or lon is None:
                st.error("Could not find location. Please try a different location name.")
                return
            
            st.success(f"Location found: {address}")
            
            # Fetch weather data
            weather_data = fetch_weather_data(lat, lon)
            if not weather_data:
                st.error("Failed to fetch weather data")
                return
            
            # Extract features
            features_df = extract_weather_features(weather_data)
            if features_df is None or features_df.empty:
                st.error("Failed to extract weather features")
                return
            
            # Load model
            model = load_model()
            if model is None:
                st.error("Model not available. Please train a model first.")
                return
            
            # Make predictions
            results_df = make_prediction(model, features_df)
            if results_df is None:
                st.error("Failed to make predictions")
                return
            
            # Display results
            st.markdown("## 📈 Weather Forecast Results")
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Temperature",
                    f"{results_df['temperature'].mean():.1f}°C"
                )
            
            with col2:
                st.metric(
                    "Predicted Temperature",
                    f"{results_df['predicted_temperature'].mean():.1f}°C"
                )
            
            with col3:
                st.metric(
                    "Mean Error",
                    f"{results_df['prediction_error'].mean():.2f}°C"
                )
            
            with col4:
                st.metric(
                    "Max Error",
                    f"{results_df['prediction_error'].max():.2f}°C"
                )
            
            # Visualizations
            create_visualizations(results_df, model)
            
            # Data table
            st.markdown("### 📋 Detailed Results")
            st.dataframe(results_df[['timestamp', 'temperature', 'predicted_temperature', 'prediction_error']].head(10))
    
    # Sidebar information
    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ About")
    st.sidebar.markdown("""
    This app uses machine learning to predict weather patterns.
    
    **Features:**
    - Real-time weather data
    - ML-powered predictions
    - Interactive visualizations
    - Data exploration tools
    
    **Data Source:** Open-Meteo API
    """)

if __name__ == "__main__":
    main() 
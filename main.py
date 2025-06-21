import os
import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import joblib
import streamlit as st
from datetime import datetime, timedelta
import requests
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import geopy
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="WeatherMLOps - Intelligent Weather Forecasting",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, classy design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .drift-warning {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
    }
    
    .drift-success {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .model-comparison {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Constants
MODEL_PATH = "models/xgboost_model.pkl"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "MLOPS_Weather_Prediction"
REFERENCE_DATA_PATH = "data/reference_data.csv"
PRODUCTION_DATA_PATH = "data/production_data.csv"
DRIFT_THRESHOLD = 0.3  # Configurable drift threshold

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

@st.cache_data
def load_model():
    """Load the trained XGBoost model"""
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None

def get_location_coordinates(location_name):
    """Get latitude and longitude from location name"""
    try:
        geolocator = Nominatim(user_agent="weathermlops")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude, location.address
        else:
            return None, None, None
    except GeocoderTimedOut:
        st.error("Geocoding service timed out. Please try again.")
        return None, None, None
    except Exception as e:
        st.error(f"Error getting location coordinates: {str(e)}")
        return None, None, None

def fetch_weather_data(lat, lon):
    """Fetch comprehensive weather data from Open-Meteo API"""
    # Open-Meteo API doesn't require API key
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters for comprehensive weather data
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,cloud_cover,visibility,precipitation_probability,weather_code',
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max',
        'timezone': 'auto',
        'forecast_days': 7
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"API request failed: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def extract_weather_features(weather_data):
    """Extract comprehensive weather features from Open-Meteo API data"""
    if not weather_data:
        return None
    
    features = []
    current_time = datetime.now()
    
    # Extract hourly data
    hourly_data = weather_data.get('hourly', {})
    if hourly_data:
        times = hourly_data.get('time', [])
        temperatures = hourly_data.get('temperature_2m', [])
        humidity = hourly_data.get('relative_humidity_2m', [])
        pressure = hourly_data.get('pressure_msl', [])
        wind_speed = hourly_data.get('wind_speed_10m', [])
        wind_direction = hourly_data.get('wind_direction_10m', [])
        cloud_cover = hourly_data.get('cloud_cover', [])
        visibility = hourly_data.get('visibility', [])
        precipitation_prob = hourly_data.get('precipitation_probability', [])
        weather_code = hourly_data.get('weather_code', [])
        
        # Process each hour of data
        for i in range(len(times)):
            if i < len(temperatures):
                dt = datetime.fromisoformat(times[i].replace('Z', '+00:00'))
                
                # Map weather codes to descriptions (WMO codes)
                weather_description = map_weather_code(weather_code[i] if i < len(weather_code) else 0)
                
                features.append({
                    'timestamp': dt,
                    'temperature': temperatures[i] if i < len(temperatures) else 0,
                    'humidity': humidity[i] if i < len(humidity) else 0,
                    'pressure': pressure[i] if i < len(pressure) else 0,
                    'wind_speed': wind_speed[i] if i < len(wind_speed) else 0,
                    'wind_direction': wind_direction[i] if i < len(wind_direction) else 0,
                    'clouds': cloud_cover[i] if i < len(cloud_cover) else 0,
                    'visibility': visibility[i] if i < len(visibility) else 10000,
                    'precipitation_probability': precipitation_prob[i] if i < len(precipitation_prob) else 0,
                    'weather_code': weather_code[i] if i < len(weather_code) else 0,
                    'weather_description': weather_description,
                    'hour': dt.hour,
                    'day': dt.day,
                    'month': dt.month,
                    'day_of_week': dt.weekday()
                })
    
    # If no hourly data, create current weather entry
    if not features:
        features.append({
            'timestamp': current_time,
            'temperature': 20,  # Default temperature
            'humidity': 60,     # Default humidity
            'pressure': 1013,   # Default pressure
            'wind_speed': 5,    # Default wind speed
            'wind_direction': 0,
            'clouds': 50,       # Default cloud cover
            'visibility': 10000,
            'precipitation_probability': 0,
            'weather_code': 1,
            'weather_description': 'Clear sky',
            'hour': current_time.hour,
            'day': current_time.day,
            'month': current_time.month,
            'day_of_week': current_time.weekday()
        })
    
    return pd.DataFrame(features)

def map_weather_code(code):
    """Map WMO weather codes to descriptions"""
    weather_codes = {
        0: 'Clear sky',
        1: 'Mainly clear',
        2: 'Partly cloudy',
        3: 'Overcast',
        45: 'Foggy',
        48: 'Depositing rime fog',
        51: 'Light drizzle',
        53: 'Moderate drizzle',
        55: 'Dense drizzle',
        56: 'Light freezing drizzle',
        57: 'Dense freezing drizzle',
        61: 'Slight rain',
        63: 'Moderate rain',
        65: 'Heavy rain',
        66: 'Light freezing rain',
        67: 'Heavy freezing rain',
        71: 'Slight snow fall',
        73: 'Moderate snow fall',
        75: 'Heavy snow fall',
        77: 'Snow grains',
        80: 'Slight rain showers',
        81: 'Moderate rain showers',
        82: 'Violent rain showers',
        85: 'Slight snow showers',
        86: 'Heavy snow showers',
        95: 'Thunderstorm',
        96: 'Thunderstorm with slight hail',
        99: 'Thunderstorm with heavy hail'
    }
    return weather_codes.get(code, 'Unknown')

def train_multiple_models(X, y):
    """Train multiple models for comparison"""
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    metrics = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X, y)
        trained_models[name] = model
        
        # Calculate metrics
        y_pred = model.predict(X)
        metrics[name] = {
            'mae': mean_absolute_error(y, y_pred),
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
    
    return trained_models, metrics

def predict_with_multiple_models(models, input_data):
    """Make predictions with multiple models"""
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict(input_data)[0]
    return predictions

def create_enhanced_weather_chart(predictions, location_name):
    """Create enhanced weather prediction chart"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Forecast', 'Model Comparison', 'Weather Conditions', 'Prediction Confidence'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Temperature forecast
    dates = [pred["Date"] for pred in predictions['forecast']]
    temps = [pred["Temperature (°C)"] for pred in predictions['forecast']]
    
    fig.add_trace(
        go.Scatter(x=dates, y=temps, mode='lines+markers', name='Temperature',
                  line=dict(color='#667eea', width=3), marker=dict(size=8)),
        row=1, col=1
    )
    
    # Model comparison
    model_names = list(predictions['models'].keys())
    model_temps = list(predictions['models'].values())
    
    fig.add_trace(
        go.Bar(x=model_names, y=model_temps, name='Model Predictions',
               marker_color=['#667eea', '#764ba2', '#f093fb']),
        row=1, col=2
    )
    
    # Weather conditions (if available)
    if 'conditions' in predictions:
        conditions = predictions['conditions']
        fig.add_trace(
            go.Scatter(x=list(conditions.keys()), y=list(conditions.values()),
                      mode='markers', name='Weather Conditions',
                      marker=dict(size=12, color='#4ecdc4')),
            row=2, col=1
        )
    
    # Prediction confidence
    confidence = [0.85, 0.78, 0.92]  # Example confidence scores
    fig.add_trace(
        go.Bar(x=model_names, y=confidence, name='Confidence',
               marker_color=['#ff6b6b', '#4ecdc4', '#45b7d1']),
        row=2, col=2
    )
    
    fig.update_layout(
        title=f"Weather Forecast for {location_name}",
        height=600,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ WeatherMLOps</h1>
        <p>Intelligent Weather Forecasting with MLOps</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Location input
    st.sidebar.markdown("### 📍 Location")
    location_input = st.sidebar.text_input(
        "Enter City/Location", 
        value="San Francisco, CA",
        help="Enter a city name, address, or location"
    )
    
    # Get coordinates from location
    lat, lon, address = get_location_coordinates(location_input)
    
    if lat and lon:
        st.sidebar.success(f"📍 {address}")
        st.sidebar.metric("Latitude", f"{lat:.4f}")
        st.sidebar.metric("Longitude", f"{lon:.4f}")
    else:
        st.sidebar.error("Could not find location. Please try a different search term.")
        return
    
    # Drift monitoring section
    st.sidebar.markdown("### 🔍 Drift Monitoring")
    
    if st.sidebar.button("Check Data Drift", key="drift_check"):
        with st.spinner("Checking for data drift..."):
            drift_metrics = check_data_drift()
            
            if drift_metrics:
                st.sidebar.markdown(f"""
                <div class="metric-card">
                    <h4>Drift Score: {drift_metrics['drift_score']:.3f}</h4>
                    <p>Drifted Features: {drift_metrics['drifted_features']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if drift_metrics['drift_score'] > DRIFT_THRESHOLD:
                    st.sidebar.markdown("""
                    <div class="drift-warning">
                        ⚠️ Significant drift detected! Consider retraining the model.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.sidebar.button("Retrain Model", key="retrain"):
                        with st.spinner("Retraining model..."):
                            retrain_model(lat, lon)
                else:
                    st.sidebar.markdown("""
                    <div class="drift-success">
                        ✅ No significant drift detected.
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.sidebar.warning("Unable to check drift. Ensure data files exist.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🌡️ Weather Prediction")
        
        # Fetch real weather data
        weather_data = fetch_weather_data(lat, lon)
        
        if weather_data:
            # Extract current weather from Open-Meteo data
            hourly_data = weather_data.get('hourly', {})
            if hourly_data and 'temperature_2m' in hourly_data and 'time' in hourly_data:
                # Get current temperature (first entry)
                current_temp = hourly_data['temperature_2m'][0] if hourly_data['temperature_2m'] else 20
                current_humidity = hourly_data.get('relative_humidity_2m', [60])[0] if hourly_data.get('relative_humidity_2m') else 60
                current_wind_speed = hourly_data.get('wind_speed_10m', [5])[0] if hourly_data.get('wind_speed_10m') else 5
                current_weather_code = hourly_data.get('weather_code', [1])[0] if hourly_data.get('weather_code') else 1
                weather_desc = map_weather_code(current_weather_code)
                
                # Display current weather
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Current Weather in {location_input}</h3>
                    <h2>{current_temp:.1f}°C</h2>
                    <p>{weather_desc}</p>
                    <p>Humidity: {current_humidity}% | Wind: {current_wind_speed} km/h</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Could not extract current weather data.")
                return
            
            # Extract features for prediction
            features_df = extract_weather_features(weather_data)
            
            if features_df is not None:
                if st.button("Generate Enhanced Weather Forecast", key="predict"):
                    with st.spinner("Generating predictions with multiple models..."):
                        # Prepare features for ML models (only numeric columns)
                        numeric_columns = ['hour', 'day', 'month', 'day_of_week', 'humidity', 'pressure', 'wind_speed', 'wind_direction', 'clouds', 'visibility', 'precipitation_probability', 'weather_code']
                        
                        # Filter to only include columns that exist in the dataframe
                        available_columns = [col for col in numeric_columns if col in features_df.columns]
                        
                        # Add lat/lon if they exist, otherwise use default values
                        if 'lat' not in features_df.columns:
                            features_df['lat'] = lat
                        if 'lon' not in features_df.columns:
                            features_df['lon'] = lon
                        
                        available_columns.extend(['lat', 'lon'])
                        
                        # Create feature matrix with only numeric columns
                        X = features_df[available_columns].copy()
                        
                        # Create target variable (temperature)
                        if 'temperature' in features_df.columns:
                            y = features_df['temperature']
                        else:
                            # If no temperature data, use synthetic temperature based on hour and month
                            y = 15 + 10 * np.sin(2 * np.pi * features_df['month'] / 12) + 5 * np.sin(2 * np.pi * features_df['hour'] / 24)
                        
                        # Train multiple models
                        models, metrics = train_multiple_models(X, y)
                        
                        # Generate predictions for next 7 days
                        today = datetime.now()
                        predictions = {
                            'forecast': [],
                            'models': {},
                            'conditions': {}
                        }
                        
                        for i in range(7):
                            date = today + timedelta(days=i)
                            
                            # Create input data for prediction using the same numeric columns
                            input_data = X.iloc[0:1].copy()  # Use first row as template
                            input_data['hour'] = 12
                            input_data['day'] = date.day
                            input_data['month'] = date.month
                            input_data['day_of_week'] = date.weekday()
                            
                            # Make predictions with all models
                            model_predictions = predict_with_multiple_models(models, input_data)
                            
                            # Use XGBoost as primary prediction
                            primary_temp = model_predictions['XGBoost']
                            
                            predictions['forecast'].append({
                                "Date": date.strftime("%Y-%m-%d"),
                                "Day": date.strftime("%A"),
                                "Temperature (°C)": primary_temp
                            })
                            
                            # Store model predictions for the first day
                            if i == 0:
                                predictions['models'] = model_predictions
                        
                        # Display enhanced chart
                        st.markdown("### 📊 Enhanced Weather Forecast")
                        chart = create_enhanced_weather_chart(predictions, location_input)
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Display model comparison
                        st.markdown("### 🤖 Model Comparison")
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown("""
                            <div class="model-comparison">
                                <h4>XGBoost</h4>
                                <p>Temperature: {:.1f}°C</p>
                                <p>R² Score: {:.3f}</p>
                            </div>
                            """.format(predictions['models']['XGBoost'], metrics['XGBoost']['r2']), 
                            unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown("""
                            <div class="model-comparison">
                                <h4>Random Forest</h4>
                                <p>Temperature: {:.1f}°C</p>
                                <p>R² Score: {:.3f}</p>
                            </div>
                            """.format(predictions['models']['Random Forest'], metrics['Random Forest']['r2']), 
                            unsafe_allow_html=True)
                        
                        with col_c:
                            st.markdown("""
                            <div class="model-comparison">
                                <h4>Linear Regression</h4>
                                <p>Temperature: {:.1f}°C</p>
                                <p>R² Score: {:.3f}</p>
                            </div>
                            """.format(predictions['models']['Linear Regression'], metrics['Linear Regression']['r2']), 
                            unsafe_allow_html=True)
                        
                        # Display forecast table
                        pred_df = pd.DataFrame(predictions['forecast'])
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Log predictions to production data
                        for pred in predictions['forecast']:
                            new_entry = pd.DataFrame({
                                "hour": [12],
                                "day": [datetime.strptime(pred["Date"], "%Y-%m-%d").day],
                                "month": [datetime.strptime(pred["Date"], "%Y-%m-%d").month],
                                "lat": [lat],
                                "lon": [lon],
                                "temperature": [pred["Temperature (°C)"]]
                            })
                            
                            if os.path.exists(PRODUCTION_DATA_PATH):
                                new_entry.to_csv(PRODUCTION_DATA_PATH, mode="a", header=False, index=False)
                            else:
                                new_entry.to_csv(PRODUCTION_DATA_PATH, index=False)
            else:
                st.error("Could not extract weather features. Please try again.")
        else:
            st.warning("Could not fetch weather data. Please check your internet connection or try again later.")
    
    with col2:
        st.markdown("## 📈 Model Status")
        
        # Model information
        model = load_model()
        if model:
            st.markdown("""
            <div class="metric-card">
                <h4>✅ Model Status</h4>
                <p>XGBoost Model Loaded</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="drift-warning">
                <h4>❌ Model Status</h4>
                <p>Model Not Found</p>
            </div>
            """, unsafe_allow_html=True)
        
        # System metrics
        st.markdown("### 🔧 System Metrics")
        
        # Check MLflow status
        try:
            mlflow_client = mlflow.MlflowClient()
            experiments = mlflow_client.search_experiments()
            st.metric("MLflow Experiments", len(experiments))
        except:
            st.metric("MLflow Status", "Not Connected")
        
        # Check data files
        ref_exists = os.path.exists(REFERENCE_DATA_PATH)
        prod_exists = os.path.exists(PRODUCTION_DATA_PATH)
        
        st.metric("Reference Data", "✅" if ref_exists else "❌")
        st.metric("Production Data", "✅" if prod_exists else "❌")
        
        # Weather API status
        st.metric("Weather API", "✅ Open-Meteo")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌤️ WeatherMLOps - Powered by MLOps & AI</p>
        <p>Built with Streamlit, MLflow, and Multiple ML Models</p>
    </div>
    """, unsafe_allow_html=True)

def calculate_drift_score(ref_data, curr_data):
    """Calculate data drift score using statistical tests"""
    if ref_data.empty or curr_data.empty:
        return 0.0, 0
    
    drift_score = 0.0
    drifted_features = 0
    
    for column in ref_data.columns:
        if column in curr_data.columns:
            try:
                # Simple drift detection based on mean difference
                ref_mean = ref_data[column].mean()
                curr_mean = curr_data[column].mean()
                
                # Calculate relative difference
                if ref_mean != 0:
                    relative_diff = abs(curr_mean - ref_mean) / abs(ref_mean)
                    if relative_diff > 0.1:  # 10% threshold
                        drifted_features += 1
                    drift_score += relative_diff
            except Exception:
                continue
    
    if len(ref_data.columns) > 0:
        drift_score /= len(ref_data.columns)
    
    return drift_score, drifted_features

def check_data_drift():
    """Check for data drift between reference and production data"""
    if not os.path.exists(REFERENCE_DATA_PATH) or not os.path.exists(PRODUCTION_DATA_PATH):
        return None

    try:
        ref_data = pd.read_csv(REFERENCE_DATA_PATH)
        curr_data = pd.read_csv(PRODUCTION_DATA_PATH)
        
        # Ensure we have enough data for comparison
        if len(curr_data) < 10:
            return None
        
        drift_score, drifted_features = calculate_drift_score(ref_data, curr_data)
        
        drift_metrics = {
            "drift_score": drift_score,
            "drifted_features": drifted_features
        }
        drift_timestamp = datetime.now().isoformat()
        
        # Log drift metrics to MLflow
        try:
            mlflow.log_metrics(drift_metrics)
            mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
            mlflow.log_param("timestamp", drift_timestamp)
        except Exception as e:
            st.warning(f"Could not log to MLflow: {str(e)}")
        
        return {**drift_metrics, "timestamp": drift_timestamp}
    except Exception as e:
        st.error(f"Error checking drift: {str(e)}")
        return None

def retrain_model(lat, lon):
    """Retrain the model with new data"""
    try:
        from scripts.train_model import train_model
        train_model(lat, lon)
        st.success("Model retrained successfully!")
        return True
    except Exception as e:
        st.error(f"Error retraining model: {str(e)}")
        return False

if __name__ == "__main__":
    main()

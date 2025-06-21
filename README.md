# WeatherMLOps: Intelligent Weather Forecasting with MLOps

This project is a comprehensive MLOps application that demonstrates a full lifecycle of a machine learning model, from data fetching and training to deployment, monitoring, and retraining. It features an interactive web UI built with Streamlit.

## 🚀 Key Features

-   **Interactive Web UI**: A user-friendly interface built with Streamlit for weather forecasting and data analysis.
-   **Multi-Model Training**: Trains and compares several models (XGBoost, Random Forest, Linear Regression).
-   **Real-time Data**: Fetches and processes weather data from the Open-Meteo API.
-   **MLflow Integration**: For end-to-end experiment tracking, model logging, and metric comparison.
-   **Data Drift Detection**: Monitors for changes between reference and production datasets using Evidently AI.
-   **Automated Retraining**: A daily monitoring script to automatically detect drift and trigger model retraining.
-   **Cloud-Ready**: Optimized for deployment on Streamlit Cloud with robust error handling.

## MLOps Capabilities: Local vs. Streamlit Cloud

While the interactive Streamlit application is fully deployable to the cloud, the automated backend MLOps features are designed for a local or dedicated server environment. Here's a breakdown:

| Feature | ✅ Works on Streamlit Cloud? | 🏡 Best Used Locally | Explanation |
| :--- | :---: | :---: | :--- |
| **Interactive Streamlit App** | **Yes** | Yes | The user-facing web UI for predictions and data exploration is fully cloud-compatible. |
| **MLflow Experiment Tracking** | **Partial** | **Yes** | `mlflow.db` is ephemeral on the cloud; history is wiped on restarts. Locally, it's persistent. |
| **Data Drift Detection** | **Partial** | **Yes** | You can manually *explore* data via the UI, but the automated daily check will not run on the cloud. |
| **Automated Daily Retraining**| **No** | **Yes** | Streamlit Cloud cannot run the scheduled background scripts (`daily_monitor.py`) required for automation. |
| **Viewing MLflow UI** | **No** | **Yes** | The `mlflow ui` command requires a persistent database and must be run locally. |

### Why the Difference?
Streamlit Cloud provides an **ephemeral filesystem**, which means any files created or modified by the app (like `mlflow.db`) are deleted when the app restarts or redeploys. It is also designed to run a single command (`streamlit run ...`), so it cannot run background processes for automated monitoring.

## Local MLOps Workflow

To use the full MLOps capabilities, run the components on your local machine.

### 1. View Experiment History
Open a new terminal window and run the MLflow UI. This gives you a web dashboard to compare all your model training runs.
```bash
source venv/bin/activate
mlflow ui
```
Navigate to `http://127.0.0.1:5000` in your browser.

### 2. Run Automated Monitoring
To simulate a real-world MLOps pipeline, you can run the daily monitoring script. This script will check for data drift once a day and retrain the model if necessary.
```bash
source venv/bin/activate
python scripts/daily_monitor.py
```
Leave this script running in a terminal to perform its daily checks.

## 🚀 How to Run and Deploy

### 1. Running Locally
```bash
# Activate virtual environment
source venv/bin/activate

# Retrain the model (if you've made changes)
python scripts/train_model.py

# Run the Streamlit app
streamlit run main_cloud.py
```

### 2. Deploying to Streamlit Cloud
After pushing all files to GitHub (including the model and data files), deploy your app:

1.  **Go to**: [https://share.streamlit.io/](https://share.streamlit.io/)
2.  **Connect** your GitHub repository.
3.  **Set Main file path to**: `main_cloud.py`
4.  **Deploy!** The application will be live, offering predictions and data exploration, while the automated MLOps features can be run locally.

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

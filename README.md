# Weather Forecasting App with Data Drift Detection

## Overview

This project is a Streamlit-based web application that predicts temperature based on hour, day, and month inputs. It uses an XGBoost model for predictions and incorporates data drift detection to monitor model performance over time. The app is designed with MLOps principles in mind, featuring automated model retraining when significant data drift is detected.

## Features

- Temperature prediction based on time inputs
- Real-time data drift monitoring
- Automated model retraining via GitHub Actions
- MLflow integration for experiment tracking
- Streamlit-based user interface

## Project Structure

```
weather_forecasting_app/
│
├── .github/
│   └── workflows/
│       └── drift_check_and_retrain.yml
│
├── .streamlit/
│   └── secrets.toml
│
├── data/
│   ├── reference_data.csv
│   └── production_data.csv
│
├── models/
│   └── xgboost_model.pkl
│
├── scripts/
│   ├── drift_detection.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/weather_forecasting_app.git
cd weather_forecasting_app
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

1. Set up your .streamlit/secrets.toml file with your API keys:
```toml
WEATHER_API_KEY = "your_api_key_here"
WEATHER_API_URL = "your_api_url_here"
```

2. Ensure your MLFLOW_TRACKING_URI in both app.py and train_model.py points to your MLflow server.

## Usage

1. Train the initial model:
```bash
python scripts/train_model.py
```

2. Start the MLflow server:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

4. Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).

## Data Drift Detection and Model Retraining

The project includes a GitHub Actions workflow (`drift_check_and_retrain.yml`) that automatically checks for data drift and retrains the model if significant drift is detected. This ensures that the model remains accurate over time as new data is collected.

To enable this feature:
1. Ensure your GitHub repository is connected to your deployment platform.
2. Set up the necessary secrets in your GitHub repository settings for API access.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

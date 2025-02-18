# Not Your Basic Weather Prediction

## 📌 Project Overview

**"Not Your Basic Weather Prediction"** is an MLOps-powered weather forecasting application that enhances a simple weather prediction model with a full **MLOps pipeline**, including:
✅ **Automated Data Fetching** from an external weather API <br>
✅ **Data Preprocessing & Feature Engineering**<br>
✅ **Model Training using XGBoost**<br>
✅ **Model Tracking with MLflow**<br>
✅ **API Deployment via FastAPI**<br>
✅ **Interactive UI using Streamlit**<br>
✅ **Automated Model Retraining**<br>

This project showcases **end-to-end machine learning model development** and deployment using modern MLOps principles.

## 🛠️ Tech Stack

* **Python** (Main Language)
* **XGBoost** (ML Model)
* **Pandas & NumPy** (Data Processing)
* **MLflow** (Model Tracking)
* **FastAPI** (API Deployment)
* **Streamlit** (User Interface)
* **Uvicorn** (ASGI Server)
* **Joblib** (Model Persistence)

## 📁 Project Structure

```
📂 Not Your Basic Weather Prediction
│── 📄 main.py # Main application file (Train, Serve, UI)
│── 📄 config.py # Contains API keys & configuration
│── 📄 requirements.txt # Dependencies for deployment
│── 📄 xgboost_model.pkl # Trained model (generated after training)
│── 📂 models # Model storage directory
│── 📂 mlruns # MLflow tracking directory
```

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model

```bash
python main.py --train
```

* Fetches weather data
* Preprocesses and engineers features
* Trains an **XGBoost model**
* Logs experiment using **MLflow**
* Saves model as `xgboost_model.pkl`

### 3️⃣ Run the FastAPI Server

```bash
python main.py --serve
```

* Starts FastAPI on **http://127.0.0.1:8000/**
* Use **POST /predict** to get predictions

### 4️⃣ Run the Streamlit UI

```bash
python main.py --ui
```

* Provides an interactive interface for weather predictions

## 🌍 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Predict temperature for a given time |

Example request:

```json
{
  "hour": 12,
  "day": 10,
  "month": 2
}
```

Example response:

```json
{
  "predicted_temperature": 22.5
}
```

## 🎨 Streamlit UI

* Provides a **slider-based UI** for entering weather conditions
* Calls the **FastAPI backend** to fetch predictions
* Displays **predicted temperature** dynamically

## 🔄 Automating Model Retraining

To **retrain the model periodically**, run:

```bash
python main.py --train
```

This fetches new data and updates the model.

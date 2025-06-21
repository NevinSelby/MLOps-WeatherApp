#!/bin/bash

# WeatherMLOps Startup Script
echo "🌤️ Starting WeatherMLOps..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check if model exists, if not train it
if [ ! -f "models/xgboost_model.pkl" ]; then
    echo "Training initial model..."
    python scripts/train_model.py
fi

# Start MLflow server in background
echo "Starting MLflow server..."
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifacts &
MLFLOW_PID=$!

# Wait a moment for MLflow to start
sleep 3

# Start Streamlit app
echo "Starting Streamlit application..."
echo "🌤️ WeatherMLOps is running at: http://localhost:8501"
echo "📊 MLflow UI is available at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the application"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping WeatherMLOps..."
    kill $MLFLOW_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Start Streamlit
streamlit run main.py

# Cleanup when Streamlit exits
cleanup 
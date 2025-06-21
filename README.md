# WeatherMLOps - Weather Prediction with MLOps

A sophisticated weather prediction web application built with MLOps principles, featuring automated model training, drift detection, and real-time weather forecasting using multiple machine learning models.

## 🌟 Features

### Core Functionality
- **Multi-Model Weather Prediction**: Uses XGBoost, Random Forest, and Linear Regression models
- **Real-time Weather Data**: Fetches current weather data from Open-Meteo API
- **Location-based Predictions**: Accepts city names and automatically geocodes to coordinates
- **Interactive Dashboard**: Modern Streamlit UI with beautiful visualizations
- **Model Performance Comparison**: Side-by-side comparison of different ML models

### MLOps Capabilities
- **Automated Model Training**: Scheduled retraining with MLflow experiment tracking
- **Data Drift Detection**: Monitors data distribution changes and triggers retraining
- **Model Versioning**: Tracks model performance and versions using MLflow
- **Daily Monitoring**: Automated daily checks for data quality and model performance
- **Production-Ready**: Includes monitoring scripts and systemd service for automation

### Advanced Features
- **Rich Weather Features**: Extracts comprehensive weather features for better predictions
- **Performance Metrics**: Tracks MAE, MSE, RMSE, and R² scores
- **Interactive Visualizations**: Charts showing predictions vs actual values
- **Error Handling**: Robust error handling for API failures and edge cases

## 🏗️ Architecture

```
WeatherMLOps/
├── main.py                 # Main Streamlit application
├── scripts/
│   ├── train_model.py      # Model training script
│   ├── drift_detection.py  # Data drift detection
│   └── daily_monitor.py    # Daily monitoring automation
├── data/
│   ├── production_data.csv # Production weather data
│   └── reference_data.csv  # Reference data for drift detection
├── models/                 # Trained model artifacts
├── mlruns/                 # MLflow experiment tracking
├── requirements.txt        # Python dependencies
├── start.sh               # Application startup script
└── weather-mlops.service  # Systemd service for automation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd WeatherMLOps
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run main.py
   ```

The application will be available at `http://localhost:8501`

## 📖 Usage

### Web Interface

1. **Enter Location**: Type a city name (e.g., "New York", "London", "Tokyo")
2. **View Current Weather**: See real-time weather data for the location
3. **Get Predictions**: View temperature predictions from multiple ML models
4. **Compare Models**: Analyze performance metrics and predictions
5. **Monitor Drift**: Check data drift status and model health

### Command Line Tools

**Train Models**
```bash
python scripts/train_model.py
```

**Check Data Drift**
```bash
python scripts/drift_detection.py
```

**Run Daily Monitoring**
```bash
python scripts/daily_monitor.py
```

## 🔧 Configuration

### Environment Variables
- `OPENWEATHER_API_KEY`: OpenWeatherMap API key (optional, fallback to Open-Meteo)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (defaults to local)

### Model Configuration
Models are configured in `scripts/train_model.py`:
- **XGBoost**: Gradient boosting with hyperparameter tuning
- **Random Forest**: Ensemble method with 100 estimators
- **Linear Regression**: Baseline linear model

## 🤖 MLOps Automation

### Daily Monitoring Setup

1. **Install systemd service**
   ```bash
   sudo cp weather-mlops.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable weather-mlops
   sudo systemctl start weather-mlops
   ```

2. **Check service status**
   ```bash
   sudo systemctl status weather-mlops
   ```

### Automated Workflow
1. **Daily at 6 AM**: Data drift detection runs
2. **If drift detected**: Model retraining is triggered
3. **Performance monitoring**: Metrics are logged to MLflow
4. **Alert system**: Notifications for critical issues

## 📊 Data Sources

### Weather Data
- **Primary**: Open-Meteo API (free, no API key required)
- **Fallback**: OpenWeatherMap API (requires API key)
- **Features**: Temperature, humidity, pressure, wind speed, precipitation

### Data Processing
- **Feature Engineering**: Extracts time-based features, weather patterns
- **Data Validation**: Ensures data quality and consistency
- **Drift Detection**: Monitors distribution changes using statistical tests

## 🧪 Model Performance

### Evaluation Metrics
- **Mean Absolute Error (MAE)**: Average prediction error
- **Mean Squared Error (MSE)**: Squared prediction error
- **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors
- **R² Score**: Coefficient of determination

### Model Comparison
The application automatically compares:
- **XGBoost**: Best for complex patterns, highest accuracy
- **Random Forest**: Good balance of accuracy and interpretability
- **Linear Regression**: Baseline model for comparison

## 🔍 Monitoring & Logging

### MLflow Integration
- **Experiment Tracking**: All training runs are logged
- **Model Registry**: Version control for trained models
- **Artifact Storage**: Model files and metadata
- **Performance History**: Historical metrics and comparisons

### Logging Features
- **Training Logs**: Detailed training process information
- **Prediction Logs**: API calls and prediction results
- **Error Logs**: Exception handling and debugging information
- **Performance Logs**: Model accuracy and drift metrics

## 🛠️ Development

### Project Structure
```
├── main.py                 # Streamlit web application
├── scripts/
│   ├── train_model.py      # Model training and evaluation
│   ├── drift_detection.py  # Statistical drift detection
│   └── daily_monitor.py    # Automated monitoring
├── data/                   # Data storage and management
├── models/                 # Trained model artifacts
└── mlruns/                 # MLflow experiment data
```

### Adding New Models
1. Import the model in `scripts/train_model.py`
2. Add to the `models` dictionary
3. Update the evaluation loop
4. Test with `python scripts/train_model.py`

### Customizing Features
- **Weather APIs**: Modify API endpoints in `main.py`
- **Drift Detection**: Adjust thresholds in `scripts/drift_detection.py`
- **UI Components**: Customize Streamlit components in `main.py`

## 🐛 Troubleshooting

### Common Issues

**Streamlit Connection Error**
```bash
# Check if port 8501 is available
lsof -i :8501
# Kill process if needed
kill -9 <PID>
```

**MLflow Database Issues**
```bash
# Reset MLflow database
rm -rf mlruns/
rm mlflow.db
```

**API Rate Limiting**
- Open-Meteo: 10,000 requests/day (usually sufficient)
- OpenWeatherMap: Check your plan limits

**Model Training Failures**
```bash
# Check data quality
python -c "import pandas as pd; print(pd.read_csv('data/production_data.csv').info())"
```

### Debug Mode
Run with debug logging:
```bash
streamlit run main.py --logger.level=debug
```

## 📈 Performance Optimization

### Model Optimization
- **Hyperparameter Tuning**: Automated with MLflow
- **Feature Selection**: Automatic feature importance ranking
- **Cross-Validation**: K-fold validation for robust evaluation

### System Optimization
- **Caching**: Streamlit caching for API calls
- **Async Processing**: Non-blocking API requests
- **Memory Management**: Efficient data handling

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Add type hints where appropriate
- Include docstrings for functions
- Write clear commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Open-Meteo**: Free weather API service
- **Streamlit**: Web application framework
- **MLflow**: MLOps platform
- **XGBoost**: Gradient boosting library
- **Scikit-learn**: Machine learning library

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `mlruns/`
3. Open an issue on GitHub
4. Contact the development team

---

**WeatherMLOps** - Making weather prediction accessible and reliable with modern MLOps practices.

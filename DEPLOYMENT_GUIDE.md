# WeatherMLOps - Streamlit Cloud Deployment Guide

This guide will help you deploy your WeatherMLOps application to Streamlit Cloud successfully.

## 🚀 Quick Deployment Steps

### 1. Prepare Your Repository

1. **Push your code to GitHub**:
   ```bash
   git add .
   git commit -m "Prepare for Streamlit Cloud deployment"
   git push origin main
   ```

2. **Ensure these files are in your repository**:
   - `main_cloud.py` (or `main.py`)
   - `requirements.txt`
   - `packages.txt` (if needed)
   - `.streamlit/config.toml`
   - `data/` directory with your CSV files
   - `models/` directory with your trained models

### 2. Deploy to Streamlit Cloud

1. Go to [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository
5. Set the main file path to: `main_cloud.py` (or `main.py`)
6. Click "Deploy!"

## 🔧 Configuration Files

### Streamlit Configuration (`.streamlit/config.toml`)
```toml
[server]
headless = true
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Requirements (`requirements.txt`)
```
streamlit>=1.28.0
mlflow>=2.8.0
requests>=2.31.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
evidently>=0.3.0
pytest>=7.4.0
xgboost>=2.0.0
joblib>=1.3.0
schedule>=1.2.0
plotly>=5.17.0
altair>=5.1.0
python-dotenv>=1.0.0
psutil>=5.9.0
geopy>=2.4.0
seaborn>=0.12.0
matplotlib>=3.7.0
```

### System Dependencies (`packages.txt`)
```
python3-dev
build-essential
```

## 🐛 Common Issues and Solutions

### 1. MLflow Database Schema Error

**Error**: `Detected out-of-date database schema`

**Solution**: The cloud version (`main_cloud.py`) includes error handling for MLflow initialization. It will continue to work even if MLflow fails to initialize.

### 2. Import Errors

**Error**: `ImportError: cannot import name 'load_dotenv'`

**Solution**: 
- Ensure you're using the correct `requirements.txt`
- The cloud version doesn't rely on environment variables for basic functionality

### 3. Model File Not Found

**Error**: `Model file not found`

**Solution**: 
- Include your trained models in the `models/` directory
- Make sure the model files are committed to your repository

### 4. Data Files Not Found

**Error**: `Reference data not found`

**Solution**:
- Include your data files in the `data/` directory
- Make sure CSV files are committed to your repository

### 5. Port Already in Use

**Error**: `Port 8501 is already in use`

**Solution**: This is handled automatically by Streamlit Cloud. The platform assigns available ports.

## 📁 Required Directory Structure

```
WeatherMLOps/
├── main_cloud.py          # Main application file
├── requirements.txt       # Python dependencies
├── packages.txt          # System dependencies
├── .streamlit/
│   └── config.toml      # Streamlit configuration
├── data/
│   ├── reference_data.csv
│   └── production_data.csv
├── models/
│   └── xgboost_model.pkl
├── scripts/
│   ├── data_explorer.py
│   └── daily_monitor.py
└── README.md
```

## 🔍 Testing Your Deployment

### Local Testing
```bash
# Test the cloud version locally
streamlit run main_cloud.py --server.port 8501
```

### Cloud Testing
1. Deploy to Streamlit Cloud
2. Test all features:
   - Weather prediction
   - Data exploration
   - Model loading
   - API calls

## 🚨 Troubleshooting Checklist

- [ ] All required files are in the repository
- [ ] `requirements.txt` is up to date
- [ ] Model files are included
- [ ] Data files are included
- [ ] No hardcoded file paths
- [ ] Error handling is in place
- [ ] API calls have timeouts
- [ ] Dependencies are compatible

## 📊 Monitoring Your Deployment

### Streamlit Cloud Dashboard
- Check the "Manage app" section for logs
- Monitor resource usage
- View deployment status

### Application Logs
- Use `st.error()` and `st.warning()` for debugging
- Check browser console for JavaScript errors
- Monitor API response times

## 🔄 Updating Your Deployment

1. Make changes to your code
2. Test locally: `streamlit run main_cloud.py`
3. Commit and push to GitHub
4. Streamlit Cloud will automatically redeploy

## 💡 Best Practices

1. **Use the cloud version** (`main_cloud.py`) for deployment
2. **Include error handling** for all external API calls
3. **Test locally** before deploying
4. **Monitor resource usage** in Streamlit Cloud dashboard
5. **Use caching** for expensive operations
6. **Handle missing data** gracefully
7. **Provide user feedback** for long-running operations

## 🆘 Getting Help

If you encounter issues:

1. Check the Streamlit Cloud logs
2. Test the application locally
3. Verify all files are in the repository
4. Check the troubleshooting checklist above
5. Review the error messages in the browser console

## 📞 Support

For Streamlit Cloud specific issues:
- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-community-cloud)
- [Streamlit Community Forum](https://discuss.streamlit.io/)

For WeatherMLOps specific issues:
- Check the project README
- Review the code comments
- Test with the provided scripts 
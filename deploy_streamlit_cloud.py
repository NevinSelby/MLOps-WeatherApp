#!/usr/bin/env python3
"""
Deployment script for Streamlit Cloud
This script helps prepare the project for deployment on Streamlit Cloud
"""

import os
import shutil
import subprocess
import sys

def create_streamlit_config():
    """Create Streamlit configuration for cloud deployment"""
    config_dir = ".streamlit"
    config_file = os.path.join(config_dir, "config.toml")
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    
    config_content = """[server]
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
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print("✅ Created Streamlit configuration file")

def create_packages_txt():
    """Create packages.txt for system dependencies"""
    packages_content = """python3-dev
build-essential
"""
    
    with open("packages.txt", 'w') as f:
        f.write(packages_content)
    
    print("✅ Created packages.txt for system dependencies")

def check_requirements():
    """Check if requirements.txt exists and is valid"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    print("✅ requirements.txt found")
    return True

def create_gitignore():
    """Create .gitignore for Streamlit Cloud"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# MLflow
mlflow.db
mlruns/

# Data
*.csv
*.pkl
*.joblib

# Logs
*.log

# Environment variables
.env
"""
    
    with open(".gitignore", 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def create_readme():
    """Create README for Streamlit Cloud"""
    readme_content = """# WeatherMLOps - Streamlit Cloud Deployment

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
"""
    
    with open("README.md", 'w') as f:
        f.write(readme_content)
    
    print("✅ Created README.md for deployment")

def main():
    """Main deployment preparation function"""
    print("🚀 Preparing WeatherMLOps for Streamlit Cloud deployment...")
    print("=" * 60)
    
    # Create necessary files
    create_streamlit_config()
    create_packages_txt()
    create_gitignore()
    create_readme()
    
    # Check requirements
    if not check_requirements():
        print("❌ Please ensure requirements.txt exists")
        return
    
    print("\n✅ Deployment preparation complete!")
    print("\n📋 Next steps for Streamlit Cloud deployment:")
    print("1. Push your code to GitHub")
    print("2. Go to https://share.streamlit.io/")
    print("3. Connect your GitHub repository")
    print("4. Set the main file path to: main.py")
    print("5. Deploy!")
    
    print("\n💡 Important notes:")
    print("- The app will automatically install dependencies from requirements.txt")
    print("- MLflow database will be created automatically")
    print("- Data files should be included in your repository")
    print("- Environment variables can be set in Streamlit Cloud dashboard")

if __name__ == "__main__":
    main() 
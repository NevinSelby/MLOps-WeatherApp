#!/usr/bin/env python3
"""
Daily Monitoring Script for WeatherMLOps
This script runs daily to check for data drift and retrain the model if necessary.
"""

import os
import sys
import schedule
import time
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('daily_monitor.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def run_drift_check():
    """Run drift detection and retrain if necessary"""
    logger.info("Starting daily drift check...")
    
    try:
        # Import drift detection function
        from scripts.drift_detection import check_drift
        
        # Check for drift
        result = check_drift()
        
        if result and result.get("drift_detected", False):
            logger.warning("Significant drift detected! Starting model retraining...")
            
            # Retrain the model
            from scripts.train_model import train_model
            
            # Get default coordinates
            lat = float(os.environ.get('DEFAULT_LAT', 37.7749))
            lon = float(os.environ.get('DEFAULT_LON', -122.4194))
            
            train_model(lat, lon)
            logger.info("Model retraining completed successfully!")
            
            # Clear production data to start fresh
            production_data_path = "data/production_data.csv"
            if os.path.exists(production_data_path):
                os.remove(production_data_path)
                logger.info("Production data cleared for fresh start")
        else:
            logger.info("No significant drift detected. Model is performing well.")
            
    except Exception as e:
        logger.error(f"Error during daily drift check: {str(e)}")

def run_scheduled_monitoring():
    """Run the scheduled monitoring tasks"""
    logger.info("Daily monitoring script started")
    
    # Schedule daily drift check at 2 AM
    schedule.every().day.at("02:00").do(run_drift_check)
    
    # Also run immediately on startup
    run_drift_check()
    
    # Keep the script running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

def run_once():
    """Run the monitoring once (for manual execution)"""
    logger.info("Running one-time drift check...")
    run_drift_check()
    logger.info("One-time drift check completed")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        run_once()
    else:
        run_scheduled_monitoring() 
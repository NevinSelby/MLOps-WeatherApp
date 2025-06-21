import pandas as pd
import numpy as np
from scipy import stats
import mlflow
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "MLOPS_Weather_Prediction"
REFERENCE_DATA_PATH = "data/reference_data.csv"
PRODUCTION_DATA_PATH = "data/production_data.csv"
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", 0.3))

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def calculate_drift_score(ref_data, curr_data):
    """Calculate data drift score using statistical tests"""
    if ref_data.empty or curr_data.empty:
        return 0.0, 0
    
    drift_score = 0.0
    drifted_features = 0
    
    for column in ref_data.columns:
        if column in curr_data.columns:
            try:
                result = stats.ks_2samp(ref_data[column], curr_data[column])
                p_value_float = float(result[1])  # p-value is the second element
                if p_value_float < 0.05:  # Significance level
                    drifted_features += 1
                drift_score += (1.0 - p_value_float)
            except Exception as e:
                print(f"Error calculating drift for column {column}: {str(e)}")
                continue
    
    if len(ref_data.columns) > 0:
        drift_score /= len(ref_data.columns)
    
    return drift_score, drifted_features

def check_drift():
    """Check for data drift between reference and production data"""
    print("Checking for data drift...")
    
    if not os.path.exists(REFERENCE_DATA_PATH):
        print("Reference data not found!")
        return None
    
    if not os.path.exists(PRODUCTION_DATA_PATH):
        print("Production data not found!")
        return None

    try:
        ref_data = pd.read_csv(REFERENCE_DATA_PATH)
        curr_data = pd.read_csv(PRODUCTION_DATA_PATH)
        
        print(f"Reference data shape: {ref_data.shape}")
        print(f"Production data shape: {curr_data.shape}")
        
        # Ensure we have enough data for comparison
        if len(curr_data) < 10:
            print("Not enough production data for drift detection (minimum 10 samples)")
            return None
        
        drift_score, drifted_features = calculate_drift_score(ref_data, curr_data)
        
        drift_metrics = {
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"Drift Score: {drift_score:.4f}")
        print(f"Drifted Features: {drifted_features}")
        print(f"Drift Threshold: {DRIFT_THRESHOLD}")
        
        # Log drift metrics to MLflow
        try:
            with mlflow.start_run(run_name="drift_detection"):
                mlflow.log_metrics(drift_metrics)
                mlflow.log_param("drift_threshold", DRIFT_THRESHOLD)
                mlflow.log_param("reference_samples", len(ref_data))
                mlflow.log_param("production_samples", len(curr_data))
        except Exception as e:
            print(f"Could not log to MLflow: {str(e)}")
        
        # Determine if retraining is needed
        if drift_score > DRIFT_THRESHOLD:
            print("⚠️  SIGNIFICANT DRIFT DETECTED! Retraining recommended.")
            return {"drift_detected": True, "metrics": drift_metrics}
        else:
            print("✅ No significant drift detected.")
            return {"drift_detected": False, "metrics": drift_metrics}
            
    except Exception as e:
        print(f"Error checking drift: {str(e)}")
        return None

def save_drift_result(result):
    """Save drift detection result to file"""
    if result:
        with open('drift_result.txt', 'w') as f:
            if result["drift_detected"]:
                f.write('drift_detected')
            else:
                f.write('no_drift')
        print("Drift result saved to drift_result.txt")
    else:
        print("No drift result to save")

if __name__ == "__main__":
    result = check_drift()
    save_drift_result(result)
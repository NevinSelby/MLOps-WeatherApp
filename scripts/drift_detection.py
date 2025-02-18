import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

reference_data = pd.read_csv('data/reference_data.csv')
new_data = pd.read_csv('data/new_data.csv')

drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_data, current_data=new_data)

drift_score = drift_report.as_dict()['metrics'][0]['result']['dataset_drift']
drift_threshold = 0.1  # Set your desired threshold

with open('drift_result.txt', 'w') as f:
    f.write('drift_detected' if drift_score > drift_threshold else 'no_drift')
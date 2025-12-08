#!/usr/bin/env python3
"""
Check for data drift in recent predictions.
Usage: python check_drift.py
"""

import numpy as np
from data_drift import DataDriftDetector

# Example: Simulate some recent predictions
recent_predictions = np.array([
    [5, 140, 68, 32.0, 48],
    [7, 160, 75, 35.5, 55],
    [3, 120, 70, 28.0, 40],
    [8, 185, 80, 38.0, 60],
    [2, 100, 65, 25.0, 35]
])

feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]

# Initialize detector
detector = DataDriftDetector()

# Check for drift
drift_report = detector.detect_drift(recent_predictions, feature_names)

if drift_report:
    detector.print_drift_report(drift_report)
    
    # Exit with error code if drift detected (for CI/CD)
    if drift_report['overall_drift']:
        print("\n⚠️  Data drift detected! Review model performance.")
        exit(1)
    else:
        print("\n✅ No significant data drift detected.")
        exit(0)
else:
    print("❌ Could not check for drift. Run training first to create reference data.")
    exit(1)

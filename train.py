# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from model_versioning import ModelVersion
from data_drift import DataDriftDetector

# Load dataset from a working source (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ Columns:", df.columns.tolist())  # Debug print

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")

# Save model metadata and version
versioning = ModelVersion()
metadata = versioning.save_metadata(X_test, y_test, model)
print(f"📦 Model version: {metadata['version']}")

# Save reference data for drift detection
drift_detector = DataDriftDetector()
feature_names = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
drift_detector.save_reference_data(X_train.values, feature_names)
print("📊 Reference data saved for drift detection")

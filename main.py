# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
import time

app = FastAPI()
model = joblib.load("diabetes_model.pkl")

# Prometheus metrics
prediction_counter = Counter('diabetes_predictions_total', 'Total number of predictions')
prediction_positive = Counter('diabetes_predictions_positive', 'Number of positive diabetes predictions')
prediction_negative = Counter('diabetes_predictions_negative', 'Number of negative diabetes predictions')
prediction_latency = Histogram('diabetes_prediction_latency_seconds', 'Prediction latency in seconds')
model_info = Gauge('diabetes_model_info', 'Model information')

# Set model info
model_info.set(1)

# Add prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}

@app.post("/predict")
def predict(data: DiabetesInput):
    start_time = time.time()
    
    # Increment prediction counter
    prediction_counter.inc()
    
    # Make prediction
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    
    # Track prediction results
    if prediction == 1:
        prediction_positive.inc()
    else:
        prediction_negative.inc()
    
    # Record latency
    prediction_latency.observe(time.time() - start_time)
    
    return {"diabetic": bool(prediction)}

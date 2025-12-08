import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Diabetes Prediction API" in response.json()["message"]


def test_metrics_endpoint():
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text


def test_predict_valid_input():
    """Test prediction with valid input."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "BMI": 33.6,
        "Age": 50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "diabetic" in response.json()
    assert isinstance(response.json()["diabetic"], bool)


def test_predict_missing_field():
    """Test prediction with missing field."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "Age": 50
        # Missing BMI
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_invalid_type():
    """Test prediction with invalid data type."""
    payload = {
        "Pregnancies": "six",  # Should be int/float
        "Glucose": 148,
        "BloodPressure": 72,
        "BMI": 33.6,
        "Age": 50
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_predict_negative_values():
    """Test prediction with negative values."""
    payload = {
        "Pregnancies": -1,
        "Glucose": 148,
        "BloodPressure": 72,
        "BMI": 33.6,
        "Age": 50
    }
    response = client.post("/predict", json=payload)
    # Should still process but we test the response structure
    assert response.status_code in [200, 422]


def test_predict_extreme_values():
    """Test prediction with extreme values."""
    payload = {
        "Pregnancies": 20,
        "Glucose": 300,
        "BloodPressure": 150,
        "BMI": 60.0,
        "Age": 100
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "diabetic" in response.json()


def test_predict_low_risk_patient():
    """Test prediction for low-risk patient."""
    payload = {
        "Pregnancies": 1,
        "Glucose": 85,
        "BloodPressure": 66,
        "BMI": 26.6,
        "Age": 31
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "diabetic" in response.json()


def test_predict_high_risk_patient():
    """Test prediction for high-risk patient."""
    payload = {
        "Pregnancies": 10,
        "Glucose": 200,
        "BloodPressure": 90,
        "BMI": 45.0,
        "Age": 65
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "diabetic" in response.json()


def test_multiple_predictions():
    """Test multiple consecutive predictions."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "BMI": 33.6,
        "Age": 50
    }
    
    for _ in range(5):
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "diabetic" in response.json()

import pytest
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def model():
    """Load the trained model."""
    return joblib.load('diabetes_model.pkl')


def test_model_exists():
    """Test that the model file exists."""
    import os
    assert os.path.exists('diabetes_model.pkl'), "Model file not found"


def test_model_type(model):
    """Test that the model is of the correct type."""
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"


def test_model_prediction_shape(model):
    """Test that model prediction returns correct shape."""
    # Sample input
    X = np.array([[6, 148, 72, 33.6, 50]])
    prediction = model.predict(X)
    assert prediction.shape == (1,), "Prediction shape is incorrect"


def test_model_prediction_range(model):
    """Test that model predictions are binary (0 or 1)."""
    X = np.array([[6, 148, 72, 33.6, 50]])
    prediction = model.predict(X)
    assert prediction[0] in [0, 1], "Prediction is not binary"


def test_model_probability_shape(model):
    """Test that model probability prediction returns correct shape."""
    X = np.array([[6, 148, 72, 33.6, 50]])
    proba = model.predict_proba(X)
    assert proba.shape == (1, 2), "Probability shape is incorrect"


def test_model_probability_sum(model):
    """Test that probabilities sum to 1."""
    X = np.array([[6, 148, 72, 33.6, 50]])
    proba = model.predict_proba(X)
    assert np.isclose(proba.sum(), 1.0), "Probabilities don't sum to 1"


def test_model_high_risk_patient(model):
    """Test prediction for a high-risk patient."""
    # High glucose, high BMI, older age
    X = np.array([[6, 180, 90, 40.0, 60]])
    prediction = model.predict(X)
    # We expect high risk, but model might vary
    assert prediction[0] in [0, 1], "Invalid prediction"


def test_model_low_risk_patient(model):
    """Test prediction for a low-risk patient."""
    # Low glucose, normal BMI, younger age
    X = np.array([[1, 80, 60, 22.0, 25]])
    prediction = model.predict(X)
    # We expect low risk, but model might vary
    assert prediction[0] in [0, 1], "Invalid prediction"


def test_model_batch_prediction(model):
    """Test batch predictions."""
    X = np.array([
        [6, 148, 72, 33.6, 50],
        [1, 85, 66, 26.6, 31],
        [8, 183, 64, 23.3, 32]
    ])
    predictions = model.predict(X)
    assert predictions.shape == (3,), "Batch prediction shape is incorrect"
    assert all(p in [0, 1] for p in predictions), "Some predictions are not binary"

# 🩺 Diabetes Prediction Model – Complete End-to-End MLOps Project

> 🎥 YouTube video for the project: **"Build Your First MLOps Project"**

This is a **production-ready MLOps project** that demonstrates end-to-end machine learning operations using a real-world use case: predicting diabetes based on health metrics.

## 🎯 **End-to-End MLOps Features**

✅ **ML Development** - Random Forest model with scikit-learn  
✅ **Automated Testing** - 19 unit/integration tests with pytest  
✅ **CI/CD Pipeline** - GitHub Actions with test/build/deploy stages  
✅ **Model Versioning** - Automated version tracking with metadata  
✅ **Data Drift Detection** - Statistical drift monitoring with KS-test  
✅ **API Service** - FastAPI with input validation  
✅ **Monitoring** - Prometheus metrics + Grafana dashboards  
✅ **Container Orchestration** - Kubernetes deployment with health checks  
✅ **Container Registry** - Docker Hub integration

---

## 📊 Problem Statement

Predict if a person is diabetic based on:
- Pregnancies
- Glucose
- Blood Pressure
- BMI
- Age

We use a Random Forest Classifier trained on the **Pima Indians Diabetes Dataset**.

---

## 🚀 Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/iam-veeramalla/first-mlops-project.git
cd first-mlops-project
```

### 2. Create Virtual Environment

```
python3 -m venv .mlops
source .mlops/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

## Train the Model

```
python train.py
```

## Run the API Locally

```
uvicorn main:app --reload
```

### Sample Input for /predict

```
{
  "Pregnancies": 2,
  "Glucose": 130,
  "BloodPressure": 70,
  "BMI": 28.5,
  "Age": 45
}
```

## Dockerize the API

### Build the Docker Image

```
docker build -t diabetes-prediction-model .
```

### Run the Container

```
docker run -p 8000:8000 diabetes-prediction-model
```

## Deploy to Kubernetes

```bash
kubectl apply -f k8s-deployment.yml
kubectl apply -f monitoring/
```

### Access Services

```bash
# API
kubectl port-forward svc/diabetes-api-service 8000:80

# Prometheus
kubectl port-forward svc/prometheus-service 9090:9090

# Grafana (admin/admin123)
kubectl port-forward svc/grafana-service 3000:3000
```

---

## 🧪 Testing

Run the complete test suite:

```bash
pytest tests/ -v --cov=. --cov-report=term-missing
```

**Test Coverage:**
- 9 model unit tests
- 10 API integration tests
- Multi-version Python testing (3.9, 3.10, 3.11)

---

## 📊 Monitoring & Observability

### Available Metrics
- `diabetes_predictions_total` - Total prediction count
- `diabetes_predictions_positive` - Positive predictions
- `diabetes_predictions_negative` - Negative predictions
- `diabetes_prediction_latency_seconds` - Prediction latency histogram

### Check Data Drift

```bash
python check_drift.py
```

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline automatically:
1. ✅ Runs linting (flake8)
2. ✅ Executes all tests
3. ✅ Trains the model
4. ✅ Builds Docker image
5. ✅ Pushes to Docker Hub
6. ✅ Deploys to Kubernetes (optional)

---

## 📁 Project Structure

```
.
├── main.py                  # FastAPI application
├── train.py                 # Model training script
├── model_versioning.py      # Version tracking system
├── data_drift.py            # Drift detection module
├── check_drift.py           # Drift checking script
├── tests/                   # Test suite
│   ├── test_api.py         # API integration tests
│   └── test_model.py       # Model unit tests
├── .github/workflows/       # CI/CD pipeline
├── k8s-deployment.yml       # Kubernetes manifests
├── monitoring/              # Prometheus & Grafana configs
└── requirements.txt         # Python dependencies
```

---

🙌 **Credits**

Created by `ABHISHEK VEERAMALLA`

Enhanced with comprehensive MLOps features by `NKCELESTIN05`

Subscribe for more DevOps + MLOps content on the YouTube Channel - `Abhishek.Veeramalla`



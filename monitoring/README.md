# Monitoring Setup with Prometheus and Grafana

## Overview
This setup includes Prometheus for metrics collection and Grafana for visualization of your ML model API.

## Metrics Exposed

The application exposes the following metrics at `/metrics`:

- `diabetes_predictions_total` - Total number of predictions made
- `diabetes_predictions_positive` - Number of positive (diabetic) predictions
- `diabetes_predictions_negative` - Number of negative (non-diabetic) predictions
- `diabetes_prediction_latency_seconds` - Histogram of prediction latency
- `diabetes_model_info` - Model information gauge

## Deployment Steps

### 1. Deploy the Updated Application
```bash
kubectl apply -f k8s-deploy.yml
```

### 2. Deploy Prometheus
```bash
kubectl apply -f monitoring/prometheus-config.yml
kubectl apply -f monitoring/prometheus-deployment.yml
```

### 3. Deploy Grafana
```bash
kubectl apply -f monitoring/grafana-deployment.yml
```

### 4. Verify Deployments
```bash
kubectl get pods
kubectl get services
```

## Accessing the Services

### Access Prometheus
```bash
kubectl port-forward svc/prometheus 9090:9090
```
Visit: http://localhost:9090

### Access Grafana
```bash
kubectl port-forward svc/grafana 3000:3000
```
Visit: http://localhost:3000
- Username: `admin`
- Password: `admin`

### Access API Metrics
```bash
kubectl port-forward svc/diabetes-api-service 8000:8000
```
Visit: http://localhost:8000/metrics

## Grafana Configuration

### 1. Add Prometheus Data Source
1. Login to Grafana
2. Go to Configuration → Data Sources
3. Click "Add data source"
4. Select "Prometheus"
5. Set URL: `http://prometheus:9090`
6. Click "Save & Test"

### 2. Create Dashboard

Import the following queries:

**Total Predictions**
```promql
diabetes_predictions_total
```

**Prediction Rate (per minute)**
```promql
rate(diabetes_predictions_total[1m])
```

**Positive vs Negative Predictions**
```promql
diabetes_predictions_positive
diabetes_predictions_negative
```

**Average Prediction Latency**
```promql
rate(diabetes_prediction_latency_seconds_sum[5m]) / rate(diabetes_prediction_latency_seconds_count[5m])
```

**P95 Latency**
```promql
histogram_quantile(0.95, rate(diabetes_prediction_latency_seconds_bucket[5m]))
```

## Testing the Setup

Send some test predictions:
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "BMI": 33.6,
    "Age": 50
  }'
```

Then check metrics:
```bash
curl http://localhost:8000/metrics
```

## Cleanup

To remove monitoring stack:
```bash
kubectl delete -f monitoring/
```

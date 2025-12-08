#!/bin/bash

# Test Diabetes Prediction API
# Make sure port-forward is running first:
# kubectl port-forward svc/diabetes-api-service 8000:80 --address=0.0.0.0

API_URL="http://localhost:8000"

echo "🏥 Testing Diabetes Prediction API"
echo "=================================="
echo ""

# Test 1: Health check
echo "1️⃣ Testing health endpoint..."
curl -s $API_URL/ && echo ""
echo ""

# Test 2: Metrics endpoint
echo "2️⃣ Testing metrics endpoint..."
curl -s $API_URL/metrics | head -10
echo "... (metrics truncated)"
echo ""

# Test 3: Prediction - High risk patient
echo "3️⃣ Testing prediction (High Risk)..."
curl -s -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "BMI": 33.6,
    "Age": 50
  }'
echo ""
echo ""

# Test 4: Prediction - Low risk patient
echo "4️⃣ Testing prediction (Low Risk)..."
curl -s -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "BMI": 26.6,
    "Age": 31
  }'
echo ""
echo ""

echo "✅ API tests complete!"
echo ""
echo "📊 View metrics in Prometheus: http://localhost:9090"
echo "📈 View dashboards in Grafana: http://localhost:3000"

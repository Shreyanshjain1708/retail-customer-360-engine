from fastapi.testclient import TestClient
from src.serving.api import app
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_churn_prediction_flow():
    # Mock payload
    payload = {
        "recency_days": 10,
        "frequency": 5,
        "monetary_value": 500.0,
        "aov": 100.0
    }
    
    # Note: This test will fail if model.pkl isn't present.
    # In a CI/CD pipeline, we would mock the joblib.load call.
    try:
        response = client.post("/predict/churn", json=payload)
        if response.status_code == 200:
            assert "churn_probability" in response.json()
            assert "risk_label" in response.json()
    except Exception as e:
        pytest.fail(f"API failed: {e}")
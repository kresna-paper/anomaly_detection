"""
Unit tests for the Fraud Detection API.

Tests API endpoints, request/response models, and error handling.
"""

import os
import sys

import pytest
from datetime import datetime
from fastapi.testclient import TestClient

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from api.models import (
    TransactionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    HealthResponse
)


# Test client
client = TestClient(app)


# Sample transaction data
sample_transaction = {
    "transaction_id": "TEST001",
    "account_id": "AC001",
    "transaction_amount": 250.50,
    "transaction_date": "2024-01-15T14:30:00",
    "transaction_type": "Debit",
    "location": "New York",
    "device_id": "D001",
    "ip_address": "192.168.1.1",
    "merchant_id": "M001",
    "channel": "Online",
    "customer_age": 35,
    "customer_occupation": "Engineer",
    "transaction_duration": 45,
    "login_attempts": 1,
    "account_balance": 5000.00
}


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self):
        """Test health endpoint returns correct status."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_loaded" in data


class TestStatsEndpoint:
    """Tests for the statistics endpoint."""

    def test_stats_endpoint(self):
        """Test stats endpoint returns statistics."""
        response = client.get("/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_predictions" in data
        assert "total_anomalies" in data
        assert "anomaly_rate" in data
        assert "avg_response_time_ms" in data


class TestPredictionEndpoint:
    """Tests for the single transaction prediction endpoint."""

    def test_valid_prediction_request(self):
        """Test prediction with valid transaction data."""
        response = client.post("/api/v1/predict", json=sample_transaction)

        # May return 200 or 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "transaction_id" in data
            assert "is_anomaly" in data
            assert "anomaly_score" in data
            assert "risk_level" in data
            assert 0 <= data["anomaly_score"] <= 1

    def test_missing_required_field(self):
        """Test prediction with missing required field."""
        invalid_tx = sample_transaction.copy()
        del invalid_tx["transaction_amount"]

        response = client.post("/api/v1/predict", json=invalid_tx)
        assert response.status_code == 422  # Validation error

    def test_invalid_ip_address(self):
        """Test prediction with invalid IP address."""
        invalid_tx = sample_transaction.copy()
        invalid_tx["ip_address"] = "invalid-ip"

        response = client.post("/api/v1/predict", json=invalid_tx)
        assert response.status_code == 422

    def test_negative_amount(self):
        """Test prediction with negative amount."""
        invalid_tx = sample_transaction.copy()
        invalid_tx["transaction_amount"] = -100.00

        response = client.post("/api/v1/predict", json=invalid_tx)
        assert response.status_code == 422

    def test_underage_customer(self):
        """Test prediction with customer under 18."""
        invalid_tx = sample_transaction.copy()
        invalid_tx["customer_age"] = 16

        response = client.post("/api/v1/predict", json=invalid_tx)
        assert response.status_code == 422


class TestBatchPredictionEndpoint:
    """Tests for the batch prediction endpoint."""

    def test_valid_batch_request(self):
        """Test batch prediction with valid data."""
        batch_data = {
            "transactions": [
                sample_transaction,
                {**sample_transaction, "transaction_id": "TEST002", "transaction_amount": 100.00}
            ]
        }

        response = client.post("/api/v1/predict/batch", json=batch_data)

        # May return 200 or 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_anomalies" in data
            assert "processing_time_ms" in data
            assert len(data["predictions"]) == 2

    def test_empty_batch(self):
        """Test batch prediction with empty list."""
        batch_data = {"transactions": []}

        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 422

    def test_batch_too_many_transactions(self):
        """Test batch prediction exceeds maximum."""
        batch_data = {
            "transactions": [sample_transaction] * 101
        }

        response = client.post("/api/v1/predict/batch", json=batch_data)
        assert response.status_code == 422


class TestAccountRiskEndpoint:
    """Tests for the account risk profile endpoint."""

    def test_account_risk_profile(self):
        """Test getting account risk profile."""
        response = client.get("/api/v1/account/AC001/risk")

        # Should return 200 even for unknown accounts
        assert response.status_code == 200

        data = response.json()
        assert "account_id" in data
        assert "risk_level" in data
        assert "risk_score" in data


class TestAnomaliesEndpoint:
    """Tests for the recent anomalies endpoint."""

    def test_get_recent_anomalies(self):
        """Test getting recent anomalies."""
        response = client.get("/api/v1/anomalies?limit=10")
        assert response.status_code == 200

        data = response.json()
        assert "anomalies" in data
        assert "count" in data
        assert isinstance(data["anomalies"], list)


class TestPydanticModels:
    """Tests for Pydantic model validation."""

    def test_transaction_request_model(self):
        """Test TransactionRequest model validation."""
        tx = TransactionRequest(**sample_transaction)
        assert tx.transaction_id == "TEST001"
        assert tx.transaction_amount == 250.50
        assert tx.channel.value == "Online"

    def test_prediction_response_model(self):
        """Test PredictionResponse model validation."""
        response_data = {
            "transaction_id": "TEST001",
            "is_anomaly": False,
            "anomaly_score": 0.25,
            "risk_level": "Low",
            "key_factors": [],
            "feature_importance": []
        }

        response = PredictionResponse(**response_data)
        assert response.transaction_id == "TEST001"
        assert response.is_anomaly is False
        assert response.anomaly_score == 0.25


@pytest.fixture
def mock_inference(monkeypatch):
    """Mock the inference engine for testing."""
    class MockInference:
        def predict_single(self, transaction):
            return {
                'transaction_id': transaction['transaction_id'],
                'is_anomaly': transaction.get('login_attempts', 1) > 1,
                'anomaly_score': 0.8 if transaction.get('login_attempts', 1) > 1 else 0.2,
                'risk_level': 'High' if transaction.get('login_attempts', 1) > 1 else 'Low',
                'key_factors': ['Multiple login attempts'] if transaction.get('login_attempts', 1) > 1 else []
            }

    monkeypatch.setattr("main.inference_engine", MockInference())


class TestWithMockInference:
    """Tests with mocked inference engine."""

    def test_prediction_with_mock(self, mock_inference):
        """Test prediction returns correct mock data."""
        response = client.post("/api/v1/predict", json=sample_transaction)
        assert response.status_code == 200

        data = response.json()
        assert data["anomaly_score"] == 0.2  # Single login
        assert data["is_anomaly"] is False

    def test_high_login_attempts_detection(self, mock_inference):
        """Test detection of multiple login attempts."""
        risky_tx = {**sample_transaction, "login_attempts": 3}

        response = client.post("/api/v1/predict", json=risky_tx)
        assert response.status_code == 200

        data = response.json()
        assert data["anomaly_score"] == 0.8
        assert data["is_anomaly"] is True
        assert data["risk_level"] == "High"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

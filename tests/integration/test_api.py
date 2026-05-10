"""tests/integration/test_api.py"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# Patch model loading so tests run without a trained model file
mock_model = MagicMock()
mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])  # low fraud by default

with patch("src.api.app.get_model", return_value=mock_model):
    from src.api.app import app

client = TestClient(app)

VALID_FEATURES = {
    **{f"V{i}": float(i - 14) * 0.1 for i in range(1, 29)},
    "Amount_scaled": 0.24,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in ("healthy", "degraded")
    assert "model_loaded" in data


def test_predict_legitimate():
    # mock returns 10% fraud probability → below default 0.5 threshold
    mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
    r = client.post("/predict", json={"features": VALID_FEATURES})
    assert r.status_code == 200
    body = r.json()
    assert body["is_fraud"] is False
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_fraud():
    mock_model.predict_proba.return_value = np.array([[0.05, 0.95]])
    r = client.post("/predict", json={"features": VALID_FEATURES})
    assert r.status_code == 200
    assert r.json()["is_fraud"] is True


def test_predict_custom_threshold():
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])
    # With threshold 0.2, a 30% probability IS fraud
    r = client.post("/predict", json={"features": VALID_FEATURES, "threshold": 0.2})
    assert r.status_code == 200
    assert r.json()["is_fraud"] is True


def test_batch_predict():
    mock_model.predict_proba.return_value = np.tile([0.9, 0.1], (3, 1))
    r = client.post("/predict/batch", json={
        "transactions": [VALID_FEATURES, VALID_FEATURES, VALID_FEATURES]
    })
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert body["total_fraud_detected"] == 0


def test_batch_too_large():
    r = client.post("/predict/batch", json={
        "transactions": [VALID_FEATURES] * 501
    })
    assert r.status_code == 400


def test_missing_feature_field():
    bad = dict(VALID_FEATURES)
    del bad["V1"]
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422

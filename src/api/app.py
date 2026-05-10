"""
src/api/app.py
---------------
FastAPI application exposing fraud-detection endpoints.
Includes health check, single prediction, batch prediction,
and Prometheus metrics instrumentation.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    Counter,
    Histogram,
    Summary,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

logger = logging.getLogger("fraud_api")

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
PREDICTION_COUNTER   = Counter("predictions_total", "Total predictions", ["result"])
PREDICTION_LATENCY   = Histogram("prediction_latency_seconds", "Prediction latency")
FRAUD_SCORE_SUMMARY  = Summary("fraud_probability_summary", "Distribution of fraud probability scores")
BATCH_SIZE_HISTOGRAM = Histogram(
    "batch_size", "Batch prediction sizes", buckets=[1, 5, 10, 50, 100, 500]
)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Binary classification API for detecting fraudulent credit card transactions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/xgb_fraud_model.joblib")
_model = None

FEATURE_NAMES = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled"]


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found at {MODEL_PATH}. Run training first.")
        _model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    return _model


@app.on_event("startup")
async def startup_event():
    try:
        get_model()
        logger.info("Startup: model loaded successfully.")
    except RuntimeError as exc:
        logger.warning("Startup warning: %s", exc)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TransactionFeatures(BaseModel):
    V1:  float = Field(..., example=-1.36)
    V2:  float = Field(..., example=-0.07)
    V3:  float = Field(..., example=2.54)
    V4:  float = Field(..., example=1.38)
    V5:  float = Field(..., example=-0.34)
    V6:  float = Field(..., example=0.46)
    V7:  float = Field(..., example=0.24)
    V8:  float = Field(..., example=0.10)
    V9:  float = Field(..., example=0.36)
    V10: float = Field(..., example=0.09)
    V11: float = Field(..., example=-0.55)
    V12: float = Field(..., example=-0.62)
    V13: float = Field(..., example=-0.99)
    V14: float = Field(..., example=-0.31)
    V15: float = Field(..., example=1.47)
    V16: float = Field(..., example=-0.47)
    V17: float = Field(..., example=0.21)
    V18: float = Field(..., example=0.03)
    V19: float = Field(..., example=0.40)
    V20: float = Field(..., example=0.25)
    V21: float = Field(..., example=-0.02)
    V22: float = Field(..., example=0.28)
    V23: float = Field(..., example=-0.11)
    V24: float = Field(..., example=0.07)
    V25: float = Field(..., example=0.13)
    V26: float = Field(..., example=-0.19)
    V27: float = Field(..., example=0.13)
    V28: float = Field(..., example=-0.02)
    Amount_scaled: float = Field(..., example=0.24)

    def to_array(self) -> np.ndarray:
        return np.array([[
            self.V1, self.V2, self.V3, self.V4, self.V5,
            self.V6, self.V7, self.V8, self.V9, self.V10,
            self.V11, self.V12, self.V13, self.V14, self.V15,
            self.V16, self.V17, self.V18, self.V19, self.V20,
            self.V21, self.V22, self.V23, self.V24, self.V25,
            self.V26, self.V27, self.V28, self.Amount_scaled,
        ]])


class PredictRequest(BaseModel):
    features: TransactionFeatures
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold")


class PredictResponse(BaseModel):
    is_fraud: bool
    fraud_probability: float
    confidence: str
    threshold_used: float


class BatchPredictRequest(BaseModel):
    transactions: List[TransactionFeatures]
    threshold: float = Field(0.5, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total_fraud_detected: int
    fraud_rate: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Credit Card Fraud Detection API — visit /docs for usage."}


@app.get("/health", response_model=HealthResponse, tags=["Ops"])
async def health():
    model_loaded = MODEL_PATH.exists()
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.get("/metrics", tags=["Ops"], include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict(request: PredictRequest):
    """Predict whether a single transaction is fraudulent."""
    model = get_model()
    X = request.features.to_array()

    start = time.perf_counter()
    try:
        prob = float(model.predict_proba(X)[0, 1])
        latency = time.perf_counter() - start

        is_fraud = prob >= request.threshold
        label = "fraud" if is_fraud else "legitimate"

        # Prometheus
        PREDICTION_COUNTER.labels(result=label).inc()
        PREDICTION_LATENCY.observe(latency)
        FRAUD_SCORE_SUMMARY.observe(prob)

        if prob >= 0.8 or prob <= 0.2:
            confidence = "high"
        elif prob >= 0.6 or prob <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return PredictResponse(
            is_fraud=is_fraud,
            fraud_probability=round(prob, 6),
            confidence=confidence,
            threshold_used=request.threshold,
        )

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictRequest):
    """Predict fraud for a batch of transactions (up to 500)."""
    if len(request.transactions) > 500:
        raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 500.")

    model = get_model()
    BATCH_SIZE_HISTOGRAM.observe(len(request.transactions))

    X = np.vstack([t.to_array() for t in request.transactions])

    start = time.perf_counter()
    try:
        probs = model.predict_proba(X)[:, 1]
        latency = time.perf_counter() - start
        PREDICTION_LATENCY.observe(latency)

        results = []
        for prob in probs:
            prob = float(prob)
            is_fraud = prob >= request.threshold
            label = "fraud" if is_fraud else "legitimate"
            PREDICTION_COUNTER.labels(result=label).inc()
            FRAUD_SCORE_SUMMARY.observe(prob)

            if prob >= 0.8 or prob <= 0.2:
                confidence = "high"
            elif prob >= 0.6 or prob <= 0.4:
                confidence = "medium"
            else:
                confidence = "low"

            results.append(PredictResponse(
                is_fraud=is_fraud,
                fraud_probability=round(prob, 6),
                confidence=confidence,
                threshold_used=request.threshold,
            ))

        fraud_count = sum(r.is_fraud for r in results)
        return BatchPredictResponse(
            results=results,
            total_fraud_detected=fraud_count,
            fraud_rate=round(fraud_count / len(results), 4),
        )

    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/model/info", tags=["Model"])
async def model_info() -> Dict:
    """Return metadata about the currently loaded model."""
    import json
    metrics_path = Path("models/metrics.json")
    meta: Dict = {"model_path": str(MODEL_PATH), "model_exists": MODEL_PATH.exists()}
    if metrics_path.exists():
        with open(metrics_path) as f:
            meta["metrics"] = json.load(f)
    return meta

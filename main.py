"""FastAPI application for Fraud Detection."""

import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from api.models import (
    AccountRiskProfile,
    BatchPredictionRequest,
    BatchPredictionResponse,
    FeatureImportance,
    HealthResponse,
    PredictionResponse,
    RiskLevel,
    StatsResponse,
    TransactionRequest,
)
from config import settings
from core.inference import FraudDetectionInference
from core.llm_explainer import get_explainer
from core.behavioral_model import BehavioralFingerprintModel

# FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Real-time fraud detection using ensemble ML models",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
inference_engine: FraudDetectionInference | None = None
behavioral_model: BehavioralFingerprintModel | None = None
behavioral_historical_df = None  # Store historical data for behavioral model
llm_explainer = None
stats = {
    "total_predictions": 0,
    "total_anomalies": 0,
    "response_times": []
}


def update_stats(anomaly: bool, response_time: float):
    stats["total_predictions"] += 1
    if anomaly:
        stats["total_anomalies"] += 1
    stats["response_times"].append(response_time)
    if len(stats["response_times"]) > 1000:
        stats["response_times"] = stats["response_times"][-1000:]


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    global inference_engine, behavioral_model, behavioral_historical_df, llm_explainer

    # Load ensemble model
    inference_engine = FraudDetectionInference(
        model_path=settings.model_path,
        features_path=settings.features_path,
        historical_data_path=settings.historical_data_path
    )
    print("Ensemble model loaded successfully.")

    # Load behavioral model (optional)
    # Note: behavioral_model_path should be the directory, not the full file path
    behavioral_dir = settings.behavioral_model_path or "models/behavioral_fingerprint"
    from pathlib import Path
    behavioral_model_loaded = False

    if Path(behavioral_dir).exists():
        try:
            behavioral_model = BehavioralFingerprintModel.load_model(behavioral_dir)
            behavioral_model_loaded = True
            print(f"Behavioral model loaded successfully from {behavioral_dir}")
        except Exception as e:
            print(f"Failed to load behavioral model: {e}")
            behavioral_model = None
    else:
        print(f"Behavioral model directory not found at {behavioral_dir}, skipping...")

    # Load historical data for behavioral model predictions
    if behavioral_model_loaded:
        try:
            import pandas as pd
            behavioral_historical_df = pd.read_csv(settings.historical_data_path)
            behavioral_historical_df['TransactionDate'] = pd.to_datetime(behavioral_historical_df['TransactionDate'])
            behavioral_historical_df['PreviousTransactionDate'] = pd.to_datetime(behavioral_historical_df['PreviousTransactionDate'])
            print(f"Historical data loaded for behavioral model: {len(behavioral_historical_df)} transactions")
        except Exception as e:
            print(f"Failed to load historical data: {e}")
            behavioral_model = None
    else:
        print("Skipping historical data load (behavioral model not loaded)")

    # Initialize LLM explainer
    if settings.enable_llm_explanations and settings.project_id:
        import os
        os.environ["PROJECT_ID"] = settings.project_id
        if settings.credentials_file_path:
            os.environ["CREDENTIALS_FILE_PATH"] = settings.credentials_file_path

        llm_explainer = get_explainer()
        if llm_explainer and llm_explainer.client:
            print("LLM explainer initialized.")


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Fraud Detection API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
        "models": {
            "ensemble": "/api/v1/predict",
            "behavioral": "/api/v1/predict/behavioral"
        },
        "ensemble_loaded": inference_engine is not None,
        "behavioral_loaded": behavioral_model is not None,
        "llm_enabled": llm_explainer is not None
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        model_loaded=inference_engine is not None,
        redis_connected=False
    )


@app.get("/health/extended", tags=["System"])
async def health_check_extended():
    """Extended health check with all models status."""
    return {
        "status": "healthy",
        "version": settings.api_version,
        "models": {
            "ensemble": {"loaded": inference_engine is not None, "path": settings.model_path},
            "behavioral": {"loaded": behavioral_model is not None, "path": settings.behavioral_model_path}
        },
        "llm_enabled": llm_explainer is not None
    }


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    anomaly_rate = (
        stats["total_anomalies"] / stats["total_predictions"]
        if stats["total_predictions"] > 0 else 0
    )
    avg_response_time = (
        sum(stats["response_times"]) / len(stats["response_times"])
        if stats["response_times"] else 0
    )

    return StatsResponse(
        total_predictions=stats["total_predictions"],
        total_anomalies=stats["total_anomalies"],
        anomaly_rate=anomaly_rate,
        avg_response_time_ms=avg_response_time * 1000
    )


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionRequest) -> PredictionResponse:
    """Analyze a single transaction for fraud risk."""
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    start_time = time.time()
    tx_data = transaction.model_dump()

    # Make prediction
    result = inference_engine.predict_single(tx_data)
    key_factors = result.get("key_factors", [])

    # Build feature importance
    feature_importance = [
        FeatureImportance(
            feature=factor,
            importance=1.0,
            direction="increases"
        )
        for factor in key_factors
    ]

    # LLM explanation for anomalies
    llm_explanation = None
    if llm_explainer and result["is_anomaly"]:
        try:
            llm_explanation = llm_explainer.explain_transaction(
                transaction=tx_data,
                anomaly_score=result["anomaly_score"],
                key_factors=key_factors
            )
        except Exception as e:
            print(f"LLM explanation failed: {e}")

    response = PredictionResponse(
        transaction_id=result["transaction_id"],
        is_anomaly=result["is_anomaly"],
        anomaly_score=result["anomaly_score"],
        risk_level=RiskLevel(result["risk_level"]),
        key_factors=key_factors,
        feature_importance=feature_importance,
        llm_explanation=llm_explanation
    )

    update_stats(result["is_anomaly"], time.time() - start_time)

    return response


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Analyze multiple transactions for fraud risk."""
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    start_time = time.time()

    import asyncio
    tasks = [
        asyncio.to_thread(inference_engine.predict_single, tx.model_dump())
        for tx in request.transactions
    ]

    results = await asyncio.gather(*tasks)

    predictions = []
    anomaly_count = 0

    for result in results:
        anomaly_count += 1 if result["is_anomaly"] else 0

        predictions.append(PredictionResponse(
            transaction_id=result["transaction_id"],
            is_anomaly=result["is_anomaly"],
            anomaly_score=result["anomaly_score"],
            risk_level=RiskLevel(result["risk_level"]),
            key_factors=result.get("key_factors", []),
            feature_importance=[
                FeatureImportance(
                    feature=factor,
                    importance=1.0,
                    direction="increases"
                )
                for factor in result.get("key_factors", [])
            ]
        ))

    processing_time = (time.time() - start_time) * 1000
    stats["total_predictions"] += len(predictions)
    stats["total_anomalies"] += anomaly_count

    return BatchPredictionResponse(
        predictions=predictions,
        total_anomalies=anomaly_count,
        processing_time_ms=processing_time
    )


@app.post("/api/v1/predict/behavioral", response_model=PredictionResponse, tags=["Behavioral Model"])
async def predict_behavioral(transaction: TransactionRequest) -> PredictionResponse:
    """Analyze a transaction using Behavioral Fingerprinting model.

    This model uses:
    - Sequence-based behavioral profiling (needs historical data)
    - Per-account adaptive thresholds
    - Attention-based autoencoder

    Note: Requires historical transaction data for the account.
    """
    if behavioral_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Behavioral model not loaded."
        )

    if behavioral_historical_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Historical data not available for behavioral model."
        )

    start_time = time.time()

    import pandas as pd

    # Map snake_case API fields to camelCase CSV format
    new_tx = {
        'TransactionID': transaction.transaction_id,
        'AccountID': transaction.account_id,
        'TransactionAmount': transaction.transaction_amount,
        'TransactionDate': transaction.transaction_date,
        'TransactionType': transaction.transaction_type.value,
        'Location': transaction.location,
        'DeviceID': transaction.device_id,
        'IP Address': transaction.ip_address,
        'MerchantID': transaction.merchant_id,
        'Channel': transaction.channel.value,
        'CustomerAge': transaction.customer_age,
        'CustomerOccupation': transaction.customer_occupation.value,
        'TransactionDuration': transaction.transaction_duration,
        'LoginAttempts': transaction.login_attempts,
        'AccountBalance': transaction.account_balance,
        'PreviousTransactionDate': transaction.transaction_date  # Fallback
    }

    try:
        # Get historical transactions for this account
        account_history = behavioral_historical_df[
            behavioral_historical_df['AccountID'] == transaction.account_id
        ].copy()

        # Append new transaction
        new_tx_df = pd.DataFrame([new_tx])
        combined_df = pd.concat([account_history, new_tx_df], ignore_index=True)

        # Make prediction
        results = behavioral_model.predict(combined_df)

        if len(results) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate prediction"
            )

        # Get the result for the new transaction (last row)
        result_row = results.iloc[-1]

        # Get explanation
        explanation = behavioral_model.explain_anomaly(new_tx_df.iloc[0], combined_df)
        key_factors = explanation.get('risk_factors', [])

        # Build feature importance
        feature_importance = [
            FeatureImportance(
                feature=factor,
                importance=1.0,
                direction="increases"
            )
            for factor in key_factors
        ]

        # Map behavioral model risk level to API RiskLevel enum
        behavioral_risk = result_row['risk_level'].lower()
        risk_mapping = {
            'low': RiskLevel.LOW,
            'medium': RiskLevel.MEDIUM,
            'high': RiskLevel.HIGH,
            'critical': RiskLevel.HIGH  # critical maps to HIGH in API
        }
        api_risk_level = risk_mapping.get(behavioral_risk, RiskLevel.MEDIUM)

        response = PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_anomaly=bool(result_row['is_anomaly']),
            anomaly_score=float(result_row['anomaly_score']),
            risk_level=api_risk_level,
            key_factors=key_factors,
            feature_importance=feature_importance,
            llm_explanation=None
        )

        update_stats(bool(result_row['is_anomaly']), time.time() - start_time)
        return response

    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required data: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/api/v1/account/{account_id}/risk", response_model=AccountRiskProfile, tags=["Account"])
async def get_account_risk(account_id: str) -> AccountRiskProfile:
    """Get risk profile for an account."""
    return AccountRiskProfile(
        account_id=account_id,
        risk_level=RiskLevel.LOW,
        risk_score=0.0,
        transaction_count=0,
        anomaly_count=0,
        key_risk_factors=[],
        recommendations=["Monitor for first transactions"]
    )


@app.get("/api/v1/anomalies", tags=["Monitoring"])
async def get_recent_anomalies(limit: int = 10):
    """Get recently flagged anomalous transactions."""
    return {"anomalies": [], "count": 0}


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "message": "Invalid request data",
            "detail": exc.errors()
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "message": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

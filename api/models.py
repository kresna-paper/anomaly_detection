"""Pydantic models for Fraud Detection API."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransactionType(str, Enum):
    CREDIT = "Credit"
    DEBIT = "Debit"


class Channel(str, Enum):
    ONLINE = "Online"
    ATM = "ATM"
    BRANCH = "Branch"


class Occupation(str, Enum):
    STUDENT = "Student"
    DOCTOR = "Doctor"
    ENGINEER = "Engineer"
    RETIRED = "Retired"


class RiskLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    account_id: str = Field(..., description="Customer account identifier")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount in USD")
    transaction_date: datetime = Field(..., description="Transaction timestamp")
    transaction_type: TransactionType = Field(..., description="Credit or Debit")
    location: str = Field(..., description="Transaction location (city)")
    device_id: str = Field(..., description="Device identifier")
    ip_address: str = Field(..., description="IP address used")
    merchant_id: str = Field(..., description="Merchant identifier")
    channel: Channel = Field(..., description="Transaction channel")
    customer_age: int = Field(..., ge=18, le=120, description="Customer age")
    customer_occupation: Occupation = Field(..., description="Customer occupation")
    transaction_duration: int = Field(..., ge=0, description="Transaction duration in seconds")
    login_attempts: int = Field(..., ge=1, description="Number of login attempts")
    account_balance: float = Field(..., ge=0, description="Account balance after transaction")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
        json_schema_extra={
            "example": {
                "transaction_id": "TX00001",
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
        }
    )

    @field_validator('ip_address')
    @classmethod
    def validate_ip_address(cls, v: str) -> str:
        import ipaddress
        try:
            ipaddress.IPv4Address(v)
            return v
        except ValueError:
            raise ValueError(f"Invalid IP address: {v}")

    @field_validator('transaction_id')
    @classmethod
    def validate_transaction_id(cls, v: str) -> str:
        if not v or len(v) < 3:
            raise ValueError("Transaction ID must be at least 3 characters")
        return v


class FeatureImportance(BaseModel):
    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")
    direction: str = Field(..., description="Direction of impact")


class PredictionResponse(BaseModel):
    transaction_id: str = Field(..., description="Transaction identifier")
    is_anomaly: bool = Field(..., description="Whether transaction is flagged as anomalous")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly probability score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk category")
    key_factors: list[str] = Field(default_factory=list, description="Top contributing risk factors")
    feature_importance: list[FeatureImportance] = Field(default_factory=list, description="Detailed feature importance")
    llm_explanation: str | None = Field(None, description="LLM-generated explanation if enabled")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()}
    )


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_items=1, max_items=100, description="List of transactions")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transactions": [{
                    "transaction_id": "TX00001",
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
                }]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(..., description="List of predictions")
    total_anomalies: int = Field(..., description="Number of flagged anomalies")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    status: str = Field(..., description="API status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    redis_connected: bool = Field(..., description="Whether Redis is connected")
    timestamp: datetime = Field(default_factory=datetime.now)


class StatsResponse(BaseModel):
    total_predictions: int = Field(..., description="Total predictions made")
    total_anomalies: int = Field(..., description="Total anomalies detected")
    anomaly_rate: float = Field(..., description="Current anomaly rate")
    avg_response_time_ms: float = Field(..., description="Average response time")


class AccountRiskProfile(BaseModel):
    account_id: str = Field(..., description="Account identifier")
    risk_level: RiskLevel = Field(..., description="Overall account risk level")
    risk_score: float = Field(..., ge=0, le=1, description="Account risk score")
    transaction_count: int = Field(..., description="Number of transactions analyzed")
    anomaly_count: int = Field(..., description="Number of anomalous transactions")
    key_risk_factors: list[str] = Field(..., description="Key risk factors for this account")
    recommendations: list[str] = Field(..., description="Recommendations for fraud team")

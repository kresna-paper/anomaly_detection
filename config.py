"""Application configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # API
    api_title: str = "Fraud Detection API"
    api_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Model paths
    model_path: str = "models/fraud_detector.pkl"
    features_path: str = "models/features.txt"
    historical_data_path: str = "bank_transactions_data.csv"
    behavioral_model_path: str | None = None  # Path to behavioral fingerprint model

    # LLM (Vertex AI)
    enable_llm_explanations: bool = False
    project_id: str | None = None
    location: str = "us-central1"
    credentials_file_path: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

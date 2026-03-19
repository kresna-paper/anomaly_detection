"""Model inference module for fraud detection."""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from core.features import TransactionFeatureEngineer
from core.models import EnsembleAnomalyDetector

warnings.filterwarnings("ignore")


class FraudDetectionInference:
    """Main inference class for fraud detection."""

    def __init__(
        self,
        model_path: str = "models/fraud_detector.pkl",
        features_path: str | None = None,
        historical_data_path: str | None = None
    ):
        self.model_path = model_path
        self.model: EnsembleAnomalyDetector | None = None
        self.feature_names: list[str] = []

        # Load historical data
        historical_df = None
        if historical_data_path and os.path.exists(historical_data_path):
            print(f"Loading historical data from {historical_data_path}...")
            historical_df = pd.read_csv(historical_data_path)
            historical_df["TransactionDate"] = pd.to_datetime(historical_df["TransactionDate"])
            if "PreviousTransactionDate" in historical_df.columns:
                historical_df["PreviousTransactionDate"] = pd.to_datetime(historical_df["PreviousTransactionDate"])
            print(f"Historical data loaded: {historical_df.shape}")

        self.feature_engineer = TransactionFeatureEngineer(historical_data=historical_df)
        self._load_model(features_path)

    def _load_model(self, features_path: str | None = None) -> None:
        """Load the trained model."""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}...")
            self.model = EnsembleAnomalyDetector.load(self.model_path)
            self.feature_names = self.model.feature_names

            if features_path and os.path.exists(features_path):
                with open(features_path, "r") as f:
                    self.feature_names = [line.strip() for line in f.readlines()]

            print(f"Model loaded successfully. Features: {len(self.feature_names)}")
        else:
            print(f"Warning: Model not found at {self.model_path}")
            self.model = None

    def _normalize_column_names(self, transaction: dict) -> dict:
        """Convert snake_case API input to PascalCase."""
        column_mapping = {
            "transaction_id": "TransactionID",
            "account_id": "AccountID",
            "transaction_amount": "TransactionAmount",
            "transaction_date": "TransactionDate",
            "transaction_type": "TransactionType",
            "location": "Location",
            "device_id": "DeviceID",
            "ip_address": "IP Address",
            "merchant_id": "MerchantID",
            "channel": "Channel",
            "customer_age": "CustomerAge",
            "customer_occupation": "CustomerOccupation",
            "transaction_duration": "TransactionDuration",
            "login_attempts": "LoginAttempts",
            "account_balance": "AccountBalance",
            "previous_transaction_date": "PreviousTransactionDate",
        }

        normalized = {}
        for key, value in transaction.items():
            new_key = column_mapping.get(key, key)
            normalized[new_key] = value

        return normalized

    def predict_single(self, transaction: dict, return_features: bool = False) -> dict:
        """Make prediction on a single transaction."""
        if self.model is None:
            raise ValueError("Model not loaded")

        normalized_tx = self._normalize_column_names(transaction)
        df = pd.DataFrame([normalized_tx])

        # Convert dates
        if "TransactionDate" in df.columns:
            df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

        if "PreviousTransactionDate" not in df.columns:
            df["PreviousTransactionDate"] = df["TransactionDate"] - pd.Timedelta(hours=1)
        else:
            df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])

        # Feature engineering
        df_transformed = self.feature_engineer.transform(df)

        available_features = [col for col in self.feature_names if col in df_transformed.columns]

        if not available_features:
            available_features = df_transformed.select_dtypes(include=[np.number]).columns.tolist()

        data = df_transformed[available_features].copy()
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(data.median())

        # Make prediction
        score = self.model.predict_proba(data)[0]
        is_anomaly = score > 0.5

        if score > 0.7:
            risk_level = "High"
        elif score > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        key_factors = self._get_key_factors(normalized_tx, score, df_transformed.iloc[0])

        result = {
            "transaction_id": transaction.get("transaction_id", normalized_tx.get("TransactionID", "")),
            "is_anomaly": is_anomaly,
            "anomaly_score": float(score),
            "risk_level": risk_level,
            "key_factors": key_factors,
        }

        if return_features:
            result["features"] = data.iloc[0].to_dict()

        return result

    def predict_batch(self, transactions: list[dict]) -> list[dict]:
        """Make predictions on multiple transactions."""
        if self.model is None:
            raise ValueError("Model not loaded")

        results = []
        for tx in transactions:
            result = self.predict_single(tx)
            results.append(result)

        return results

    def _get_key_factors(self, transaction: dict, score: float, features: pd.Series) -> list[str]:
        """Extract key risk factors for a transaction."""
        factors = []

        def get_val(key, default=0):
            return transaction.get(key, transaction.get(key.lower(), default))

        # Check login attempts
        login_attempts = get_val("LoginAttempts", 1)
        if login_attempts > 1:
            factors.append(f"Multiple login attempts ({login_attempts})")

        # Check transaction amount
        amount = get_val("TransactionAmount", 0)
        if amount > 1000:
            factors.append(f"High transaction amount (${amount:.2f})")

        # Check online channel
        channel = get_val("Channel", "")
        if channel == "Online":
            factors.append("Online transaction (higher risk)")

        # Check unusual time
        tx_date = get_val("TransactionDate")
        if isinstance(tx_date, datetime):
            hour = tx_date.hour
        elif isinstance(tx_date, str):
            try:
                hour = pd.to_datetime(tx_date).hour
            except:
                hour = 12
        else:
            hour = 12
        if hour >= 23 or hour <= 5:
            factors.append("Unusual transaction time")

        # Check amount to balance ratio
        balance = get_val("AccountBalance", 1)
        if balance > 0 and amount / balance > 0.5:
            factors.append("Large amount relative to balance")

        # Engineered features
        if "Multi_Login" in features and features["Multi_Login"] == 1:
            factors.append("Multiple login attempts detected")

        if "IsUnusualHour" in features and features["IsUnusualHour"] == 1:
            factors.append("Transaction outside business hours")

        if "Account_IPCount" in features and features["Account_IPCount"] > 1:
            factors.append("Multiple IP addresses for account")

        return factors[:5]

    def get_account_risk_profile(self, account_id: str, account_transactions: list[dict]) -> dict:
        """Generate a risk profile for an account."""
        if not account_transactions:
            return {
                "account_id": account_id,
                "risk_level": "Low",
                "risk_score": 0.0,
                "transaction_count": 0,
                "anomaly_count": 0,
                "key_risk_factors": [],
                "recommendations": []
            }

        predictions = self.predict_batch(account_transactions)
        scores = [p["anomaly_score"] for p in predictions]
        anomaly_count = sum(1 for p in predictions if p["is_anomaly"])
        avg_score = np.mean(scores)

        if avg_score > 0.5 or anomaly_count >= len(predictions) * 0.2:
            risk_level = "High"
        elif avg_score > 0.3 or anomaly_count > 0:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        risk_factors = set()
        for pred in predictions:
            risk_factors.update(pred["key_factors"])

        recommendations = []
        if anomaly_count > 0:
            recommendations.append(f"Review {anomaly_count} flagged transactions")
        if avg_score > 0.4:
            recommendations.append("Monitor account activity closely")

        return {
            "account_id": account_id,
            "risk_level": risk_level,
            "risk_score": float(avg_score),
            "transaction_count": len(account_transactions),
            "anomaly_count": anomaly_count,
            "key_risk_factors": list(risk_factors)[:5],
            "recommendations": recommendations
        }

    def explain_prediction(self, transaction: dict) -> dict:
        """Generate detailed explanation for a prediction."""
        if self.model is None:
            raise ValueError("Model not loaded")

        prediction = self.predict_single(transaction, return_features=True)

        summary = f"Transaction has an anomaly score of {prediction['anomaly_score']:.3f}, "
        summary += f"indicating {prediction['risk_level'].lower()} risk. "
        summary += f"Key factors: {', '.join(prediction['key_factors'][:3])}."

        return {
            "transaction_id": transaction.get("transaction_id", ""),
            "anomaly_score": prediction["anomaly_score"],
            "shap_values": [],
            "summary": summary
        }


# Global inference instance
_inference_instance: FraudDetectionInference | None = None


def get_inference_instance() -> FraudDetectionInference:
    """Get or create the global inference instance."""
    global _inference_instance
    if _inference_instance is None:
        _inference_instance = FraudDetectionInference()
    return _inference_instance

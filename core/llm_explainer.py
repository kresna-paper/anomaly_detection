"""LLM Integration for Fraud Detection Explanations using Google Gemini via Vertex AI."""

import logging

from utils.gemini_client import initialize_gemini_client

logger = logging.getLogger(__name__)


class FraudExplainer:
    """Generate human-readable explanations for fraudulent transactions."""

    def __init__(
        self,
        project_id: str | None = None,
        location: str = "us-central1",
        credentials_file: str | None = None,
        model: str = "gemini-2.5-flash-lite"
    ):
        import os
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.location = location
        self.model = model
        self.client = None

        if self.project_id and os.getenv("CREDENTIALS_FILE_PATH"):
            try:
                self.client = initialize_gemini_client()
                logger.info(f"Gemini client initialized: {self.project_id}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")

    def explain_transaction(
        self,
        transaction: dict,
        anomaly_score: float,
        key_factors: list[str],
        shap_values: list[dict] | None = None
    ) -> str | None:
        """Generate a human-readable explanation for a flagged transaction."""
        if self.client is None:
            return None

        prompt = self._build_prompt(transaction, anomaly_score, key_factors)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"LLM explanation failed: {e}")
            return None

    def _build_prompt(
        self,
        transaction: dict,
        anomaly_score: float,
        key_factors: list[str],
        shap_values: list[dict] | None
    ) -> str:
        """Build the prompt for the LLM."""
        risk_level = "High" if anomaly_score > 0.7 else "Medium" if anomaly_score > 0.4 else "Low"

        return f"""You are a fraud detection analyst. Analyze this flagged transaction:

TRANSACTION DETAILS:
- Transaction ID: {transaction.get('transaction_id', 'N/A')}
- Amount: ${transaction.get('transaction_amount', 0):.2f}
- Type: {transaction.get('transaction_type', 'N/A')}
- Channel: {transaction.get('channel', 'N/A')}
- Location: {transaction.get('location', 'N/A')}
- Customer Age: {transaction.get('customer_age', 'N/A')}
- Occupation: {transaction.get('customer_occupation', 'N/A')}
- Account Balance: ${transaction.get('account_balance', 0):.2f}
- Login Attempts: {transaction.get('login_attempts', 1)}

ANOMALY ANALYSIS:
- Anomaly Score: {anomaly_score:.3f} (0-1 scale)
- Risk Level: {risk_level}
- Key Risk Factors: {', '.join(key_factors[:5])}

Provide a concise explanation (2-3 sentences):
1. Why this transaction was flagged
2. Primary risk factors
3. Recommendation for fraud team

Format:
**Summary:** [1-2 sentences]
**Risk Factors:** [bullets]
**Recommendation:** [action]
"""


# Singleton instance
_explainer_instance: FraudExplainer | None = None


def get_explainer(
    project_id: str | None = None,
    credentials_file: str | None = None
) -> FraudExplainer | None:
    """Get or create the global explainer instance."""
    global _explainer_instance
    if _explainer_instance is None:
        try:
            _explainer_instance = FraudExplainer(
                project_id=project_id,
                credentials_file=credentials_file
            )
        except Exception as e:
            logger.warning(f"Could not initialize explainer: {e}")
            return None
    return _explainer_instance

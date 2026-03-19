"""Ensemble Anomaly Detection Models for Bank Transaction Fraud."""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

# Optional torch for Autoencoder
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class Autoencoder(nn.Module):
    """Autoencoder neural network for anomaly detection."""

    def __init__(self, input_dim: int, encoding_dims: list = None):
        super().__init__()

        if encoding_dims is None:
            encoding_dims = [32, 16, 8]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(dim),
            ])
            prev_dim = dim

        # Decoder
        decoder_layers = []
        for dim in reversed(encoding_dims[:-1]):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(dim),
            ])
            prev_dim = dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly score based on reconstruction error."""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class EnsembleAnomalyDetector:
    """Ensemble of anomaly detection models for fraud detection."""

    def __init__(
        self,
        contamination: float = 0.05,
        iso_weight: float = 0.3,
        svm_weight: float = 0.3,
        lof_weight: float = 0.2,
        ae_weight: float = 0.2,
        random_state: int = 42
    ):
        self.contamination = contamination
        self.weights = {
            "isolation_forest": iso_weight,
            "one_class_svm": svm_weight,
            "lof": lof_weight,
            "autoencoder": ae_weight if TORCH_AVAILABLE else 0,
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        self.models = {}
        self.scaler = StandardScaler()
        self.fitted = False
        self.feature_names: list[str] = []
        self.threshold: float | None = None
        self.score_ranges: dict[str, dict[str, float]] = {}

        self._init_models()

    def _init_models(self):
        """Initialize individual models."""
        self.models["isolation_forest"] = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            n_jobs=-1
        )

        self.models["one_class_svm"] = OneClassSVM(
            nu=self.contamination,
            kernel="rbf",
            gamma="scale"
        )

        self.models["lof"] = LocalOutlierFactor(
            contamination=self.contamination,
            novelty=True,
            n_neighbors=20,
            n_jobs=-1
        )

    def fit(
        self,
        data: pd.DataFrame,
        feature_names: list[str] | None = None,
        autoencoder_epochs: int = 50,
        batch_size: int = 32
    ):
        """Fit all models in the ensemble."""
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            data_array = data.values
        else:
            data_array = data
            if feature_names:
                self.feature_names = feature_names

        data_scaled = self.scaler.fit_transform(data_array)

        # Fit sklearn models
        print("Fitting Isolation Forest...")
        self.models["isolation_forest"].fit(data_scaled)

        print("Fitting One-Class SVM...")
        self.models["one_class_svm"].fit(data_scaled)

        print("Fitting LOF...")
        self.models["lof"].fit(data_scaled)

        # Fit Autoencoder
        if TORCH_AVAILABLE and self.weights["autoencoder"] > 0:
            print("Fitting Autoencoder...")
            self._fit_autoencoder(data_scaled, epochs=autoencoder_epochs, batch_size=batch_size)

        # Store score ranges
        print("Computing score normalization parameters...")
        self.score_ranges = {}
        train_scores = self._get_model_scores(data_scaled)

        for name, scores in train_scores.items():
            self.score_ranges[name] = {
                "min": float(scores.min()),
                "max": float(scores.max())
            }

        # Compute threshold
        ensemble_score = np.zeros(len(data))
        for name, weight in self.weights.items():
            if weight > 0 and name in train_scores:
                ensemble_score += weight * train_scores[name]

        self.threshold = np.percentile(ensemble_score, 95)
        self.fitted = True
        print(f"All models fitted! Threshold: {self.threshold:.4f}")

        return self

    def _fit_autoencoder(self, data: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train the Autoencoder model."""
        input_dim = data.shape[1]
        self.models["autoencoder"] = Autoencoder(input_dim)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models["autoencoder"].to(device)

        data_tensor = torch.FloatTensor(data).to(device)
        dataset = TensorDataset(data_tensor, data_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.models["autoencoder"].parameters(), lr=0.001)

        self.models["autoencoder"].train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, _ in loader:
                optimizer.zero_grad()
                reconstructed = self.models["autoencoder"](batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

        self.models["autoencoder"].eval()
        self._ae_device = device

    def predict(self, data: pd.DataFrame, return_scores: bool = True) -> np.ndarray:
        """Make predictions on new data."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data

        data_scaled = self.scaler.transform(data_array)
        scores = self._get_model_scores(data_scaled)

        # Combine scores
        ensemble_score = np.zeros(len(data))
        for name, weight in self.weights.items():
            if weight > 0 and name in scores:
                ensemble_score += weight * scores[name]

        return ensemble_score if return_scores else np.where(ensemble_score > self.threshold, -1, 1)

    def _get_model_scores(self, data_scaled: np.ndarray) -> dict[str, np.ndarray]:
        """Get normalized anomaly scores from each model."""
        scores = {}

        # Isolation Forest
        iso_pred = self.models["isolation_forest"].score_samples(data_scaled)
        iso_scores = -iso_pred
        if "isolation_forest" in self.score_ranges:
            r = self.score_ranges["isolation_forest"]
            iso_scores = (iso_scores - r["min"]) / (r["max"] - r["min"] + 1e-6)
        else:
            iso_scores = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-6)
        scores["isolation_forest"] = np.clip(iso_scores, 0, 1)

        # One-Class SVM
        svm_score = self.models["one_class_svm"].score_samples(data_scaled)
        svm_scores = -svm_score
        if "one_class_svm" in self.score_ranges:
            r = self.score_ranges["one_class_svm"]
            svm_scores = (svm_scores - r["min"]) / (r["max"] - r["min"] + 1e-6)
        else:
            svm_scores = (svm_scores - svm_scores.min()) / (svm_scores.max() - svm_scores.min() + 1e-6)
        scores["one_class_svm"] = np.clip(svm_scores, 0, 1)

        # LOF
        lof_score = self.models["lof"].score_samples(data_scaled)
        lof_scores = -lof_score
        if "lof" in self.score_ranges:
            r = self.score_ranges["lof"]
            lof_scores = (lof_scores - r["min"]) / (r["max"] - r["min"] + 1e-6)
        else:
            lof_scores = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-6)
        scores["lof"] = np.clip(lof_scores, 0, 1)

        # Autoencoder
        if TORCH_AVAILABLE and self.models.get("autoencoder"):
            data_tensor = torch.FloatTensor(data_scaled).to(self._ae_device)
            with torch.no_grad():
                ae_scores = self.models["autoencoder"].get_anomaly_score(data_tensor).cpu().numpy()
            if "autoencoder" in self.score_ranges:
                r = self.score_ranges["autoencoder"]
                ae_scores = (ae_scores - r["min"]) / (r["max"] - r["min"] + 1e-6)
            else:
                ae_scores = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-6)
            scores["autoencoder"] = np.clip(ae_scores, 0, 1)

        return scores

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Return probability-like scores (0-1, higher = more anomalous)."""
        raw_scores = self.predict(data, return_scores=True)
        threshold = self.threshold if self.threshold is not None else np.percentile(raw_scores, 95)

        # Sigmoid scaling
        scale_factor = 0.5
        scaled = 1 / (1 + np.exp(-(raw_scores - threshold) / scale_factor))

        return np.clip(scaled, 0, 1)

    def explain_prediction(self, data: pd.DataFrame, nsamples: str | int = "auto") -> dict:
        """Explain predictions using SHAP values."""
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not available"}

        if not self.fitted:
            raise ValueError("Model must be fitted before explanation")

        model = self.models["isolation_forest"]

        if isinstance(data, pd.DataFrame):
            data_array = data.values
            feature_names = data.columns.tolist()
        else:
            data_array = data
            feature_names = self.feature_names or [f"feature_{i}" for i in range(data.shape[1])]

        data_scaled = self.scaler.transform(data_array)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_scaled)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        feature_importance = sorted(
            zip(feature_names, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "expected_value": explainer.expected_value,
        }

    def save(self, filepath: str):
        """Save the model to disk."""
        model_data = {
            "scaler": self.scaler,
            "models": {
                "isolation_forest": self.models["isolation_forest"],
                "one_class_svm": self.models["one_class_svm"],
                "lof": self.models["lof"],
            },
            "weights": self.weights,
            "contamination": self.contamination,
            "feature_names": self.feature_names,
            "fitted": self.fitted,
            "threshold": self.threshold,
            "score_ranges": self.score_ranges,
        }

        if TORCH_AVAILABLE and self.models.get("autoencoder"):
            ae_path = filepath.replace(".pkl", "_autoencoder.pt")
            torch.save(self.models["autoencoder"].state_dict(), ae_path)
            model_data["autoencoder_path"] = ae_path
            model_data["autoencoder_input_dim"] = self.models["autoencoder"].encoder[0].in_features

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "EnsembleAnomalyDetector":
        """Load a model from disk."""
        model_data = joblib.load(filepath)

        detector = cls(
            contamination=model_data["contamination"],
        )

        detector.scaler = model_data["scaler"]
        detector.models["isolation_forest"] = model_data["models"]["isolation_forest"]
        detector.models["one_class_svm"] = model_data["models"]["one_class_svm"]
        detector.models["lof"] = model_data["models"]["lof"]
        detector.weights = model_data["weights"]
        detector.feature_names = model_data["feature_names"]
        detector.fitted = model_data["fitted"]
        detector.threshold = model_data.get("threshold")
        detector.score_ranges = model_data.get("score_ranges", {})

        # Load autoencoder if exists
        if TORCH_AVAILABLE and "autoencoder_path" in model_data:
            ae_path = model_data["autoencoder_path"]
            if os.path.exists(ae_path):
                input_dim = model_data["autoencoder_input_dim"]
                detector.models["autoencoder"] = Autoencoder(input_dim)
                detector.models["autoencoder"].load_state_dict(torch.load(ae_path, weights_only=False))
                detector.models["autoencoder"].eval()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                detector.models["autoencoder"].to(device)
                detector._ae_device = device

        print(f"Model loaded from {filepath}")
        return detector


def quick_train(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_path: str = "models/fraud_detector.pkl"
):
    """Quick training function for the fraud detector."""
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    data = df[feature_cols].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.median())

    model = EnsembleAnomalyDetector(contamination=0.05)
    model.fit(data)

    predictions = model.predict(data)
    probabilities = model.predict_proba(data)

    result_df = df.copy()
    result_df["anomaly_score"] = predictions
    result_df["anomaly_probability"] = probabilities
    result_df["is_anomaly"] = (probabilities > 0.5).astype(int)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    return model, result_df

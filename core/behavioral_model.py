"""Behavioral Fingerprinting for Fraud Detection.

Advanced anomaly detection using:
- Autoencoder for behavioral compression
- Per-account adaptive thresholds
- Sequence-based feature extraction
- TorchScript/ONNX export

Original approach for Devoteam interview.
"""

import json
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Custom color palette
FRAUD_COLORS = {
    "critical": "#E63946",
    "high": "#F4A261",
    "medium": "#E9C46A",
    "low": "#2A9D8F",
    "safe": "#264653",
}


def calculate_entropy(values) -> float:
    """Calculate entropy (diversity) of categorical values."""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log(probs + 1e-10))


class BehavioralFeatureExtractor:
    """Extract behavioral fingerprints for each account."""

    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.account_profiles = {}

    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract sequence-based features from transactions."""
        features = []

        for account_id in df["AccountID"].unique():
            account_df = df[df["AccountID"] == account_id].sort_values("TransactionDate")
            recent = account_df.tail(self.sequence_length)

            feature_dict = {
                "AccountID": account_id,
                # Sequence statistics
                "amount_mean": recent["TransactionAmount"].mean(),
                "amount_std": recent["TransactionAmount"].std(),
                "amount_min": recent["TransactionAmount"].min(),
                "amount_max": recent["TransactionAmount"].max(),
                "amount_range": recent["TransactionAmount"].max() - recent["TransactionAmount"].min(),
                # Velocity
                "amount_velocity": np.abs(np.diff(recent["TransactionAmount"].values)).mean() if len(recent) > 1 else 0,
                "amount_acceleration": np.diff(np.diff(recent["TransactionAmount"].values)).std() if len(recent) > 2 else 0,
                # Temporal patterns
                "hour_entropy": calculate_entropy(recent["TransactionDate"].dt.hour.values),
                "day_entropy": calculate_entropy(recent["TransactionDate"].dt.dayofweek.values),
                "peak_hour": recent["TransactionDate"].dt.hour.mode()[0] if len(recent) > 0 else 12,
                # Diversity metrics
                "location_entropy": calculate_entropy(recent["Location"].values),
                "device_count": recent["DeviceID"].nunique(),
                "ip_count": recent["IP Address"].nunique(),
                "merchant_entropy": calculate_entropy(recent["MerchantID"].values),
                # Channel mix
                "online_ratio": (recent["Channel"] == "Online").mean(),
                "atm_ratio": (recent["Channel"] == "ATM").mean(),
                "branch_ratio": (recent["Channel"] == "Branch").mean(),
                # Login behavior
                "multi_login_ratio": (recent["LoginAttempts"] > 1).mean(),
                "max_login_attempts": recent["LoginAttempts"].max(),
                # Transaction frequency
                "tx_frequency": len(account_df) / max(1, (account_df["TransactionDate"].max() - account_df["TransactionDate"].min()).days),
                # Account balance trend
                "balance_mean": recent["AccountBalance"].mean(),
                "balance_trend": np.polyfit(range(len(recent)), recent["AccountBalance"].values, 1)[0] if len(recent) > 1 else 0,
            }

            features.append(feature_dict)

        return pd.DataFrame(features)

    def extract_realtime_features(self, transaction: dict, history: pd.DataFrame) -> dict:
        """Extract features for single transaction with historical context."""
        account_id = transaction["AccountID"]
        account_history = history[history["AccountID"] == account_id].sort_values("TransactionDate")

        return {
            "amount_vs_mean": transaction["TransactionAmount"] / account_history["TransactionAmount"].mean(),
            "amount_zscore": (transaction["TransactionAmount"] - account_history["TransactionAmount"].mean()) / account_history["TransactionAmount"].std(),
            "amount_vs_recent_mean": transaction["TransactionAmount"] / account_history.tail(5)["TransactionAmount"].mean(),
            "hour_typicality": self._get_hour_typicality(transaction["TransactionDate"].hour, account_history["TransactionDate"].dt.hour),
            "location_is_new": transaction["Location"] not in account_history["Location"].values,
            "device_is_new": transaction["DeviceID"] not in account_history["DeviceID"].values,
            "time_since_last": (transaction["TransactionDate"] - account_history["TransactionDate"].max()).total_seconds() / 60,
        }

    def _get_hour_typicality(self, hour: int, historical_hours) -> float:
        """How typical is this hour for this account?"""
        hour_dist = (historical_hours.values == hour).mean()
        return 1 - hour_dist


class AttentionAutoencoder(nn.Module):
    """Autoencoder with attention mechanism for behavioral fingerprinting."""

    def __init__(self, input_dim: int, encoding_dims: list | None = None, latent_dim: int = 8):
        super().__init__()

        if encoding_dims is None:
            encoding_dims = [32, 16]

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(prev_dim, prev_dim),
            nn.Sigmoid()
        )

        # Latent space
        self.latent = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        for dim in reversed(encoding_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        attention_weights = self.attention(encoded)
        attended = encoded * attention_weights
        latent = self.latent(attended)
        reconstructed = self.decoder(latent)
        return reconstructed, latent, attention_weights

    def get_anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate anomaly score based on reconstruction error."""
        self.eval()
        with torch.no_grad():
            reconstructed, _, _ = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class AdaptiveThresholdManager:
    """Maintains per-account anomaly thresholds."""

    def __init__(self, percentile: float = 95, min_samples: int = 5):
        self.percentile = percentile
        self.min_samples = min_samples
        self.account_thresholds: dict[str, float] = {}

    def fit_account(self, account_id: str, scores: np.ndarray) -> None:
        """Learn threshold for a specific account."""
        if len(scores) >= self.min_samples:
            self.account_thresholds[account_id] = np.percentile(scores, self.percentile)
        else:
            self.account_thresholds[account_id] = np.percentile(scores, self.percentile)

    def get_threshold(self, account_id: str) -> float:
        """Get threshold for an account."""
        return self.account_thresholds.get(account_id, 0.5)

    def is_anomaly(self, account_id: str, score: float) -> tuple[bool, str]:
        """Check if score is anomalous for this account."""
        threshold = self.get_threshold(account_id)
        is_anom = score > threshold

        if score > threshold * 1.5:
            risk_level = "critical"
        elif score > threshold * 1.2:
            risk_level = "high"
        elif score > threshold:
            risk_level = "medium"
        else:
            risk_level = "low"

        return is_anom, risk_level


class BehavioralFingerprintModel:
    """Complete behavioral fingerprinting system."""

    def __init__(
        self,
        input_dim: int = 20,
        encoding_dims: list | None = None,
        latent_dim: int = 8,
        sequence_length: int = 10,
        device: str | None = None
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feature_extractor = BehavioralFeatureExtractor(sequence_length)
        self.model = AttentionAutoencoder(input_dim, encoding_dims, latent_dim).to(self.device)
        self.threshold_manager = AdaptiveThresholdManager()
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        self.training_history = {"loss": [], "val_loss": []}

    def prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """Prepare data for training."""
        behavior_df = self.feature_extractor.extract_sequence_features(df)
        feature_cols = [col for col in behavior_df.columns if col not in ["AccountID"]]
        self.feature_names = feature_cols

        data = behavior_df[feature_cols].values

        # Handle NaN and Inf values
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        data_scaled = self.scaler.fit_transform(data)

        return data_scaled, behavior_df

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        val_split: float = 0.2,
        log_mlflow: bool = True,
        experiment_name: str = "behavioral_fingerprint"
    ):
        """Train the behavioral fingerprint model."""
        import os
        data, behavior_df = self.prepare_data(df)
        train_data, val_data = train_test_split(data, test_size=val_split, random_state=42)

        # Create models directory and set best model path
        os.makedirs("models", exist_ok=True)
        best_model_path = os.path.abspath("models/best_model.pt")

        train_dataset = TensorDataset(
            torch.FloatTensor(train_data),
            torch.FloatTensor(train_data)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(val_data),
            torch.FloatTensor(val_data)
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        criterion = nn.MSELoss()

        if log_mlflow:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_params({
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            })

        best_val_loss = float("inf")
        patience_counter = 0
        max_patience = 20

        print(f"Training on {self.device}")

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)
                # Skip batches with NaN
                if torch.isnan(batch_x).any():
                    continue
                optimizer.zero_grad()
                reconstructed, _, _ = self.model(batch_x)
                loss = criterion(reconstructed, batch_x)
                if not torch.isnan(loss):
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validate
            self.model.eval()
            val_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(self.device)
                    if torch.isnan(batch_x).any():
                        continue
                    reconstructed, _, _ = self.model(batch_x)
                    loss = criterion(reconstructed, batch_x)
                    if not torch.isnan(loss):
                        val_loss += loss.item()
                        val_batches += 1

            val_loss = val_loss / max(val_batches, 1)

            scheduler.step(val_loss)
            self.training_history["loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)

            if log_mlflow:
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, step=epoch)

            if val_loss < best_val_loss and not np.isnan(val_loss):
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

        # Load best model if it was saved
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
        self.is_fitted = True
        self._calculate_adaptive_thresholds(data, behavior_df)

        if log_mlflow:
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.end_run()

        return self

    def _calculate_adaptive_thresholds(self, data: np.ndarray, behavior_df: pd.DataFrame):
        """Calculate per-account adaptive thresholds."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            reconstructed, _, _ = self.model(data_tensor)
            errors = torch.mean((data_tensor - reconstructed) ** 2, dim=1).cpu().numpy()

        for _, row in behavior_df.iterrows():
            account_id = row["AccountID"]
            self.threshold_manager.fit_account(account_id, errors[0:1])

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions with risk levels."""
        if not self.is_fitted:
            raise ValueError("Model must be trained first")

        data, behavior_df = self.prepare_data(df)

        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.FloatTensor(data).to(self.device)
            reconstructed, latent, _ = self.model(data_tensor)
            anomaly_scores = torch.mean((data_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
            latent_codes = latent.cpu().numpy()

        results = behavior_df.copy()
        results["anomaly_score"] = anomaly_scores
        results["latent_dim_0"] = latent_codes[:, 0]

        risk_levels = []
        is_anomalies = []

        for _, row in results.iterrows():
            is_anom, risk = self.threshold_manager.is_anomaly(
                row["AccountID"],
                row["anomaly_score"]
            )
            is_anomalies.append(is_anom)
            risk_levels.append(risk)

        results["is_anomaly"] = is_anomalies
        results["risk_level"] = risk_levels

        return results

    def get_fingerprint(self, account_id: str, df: pd.DataFrame) -> dict:
        """Get behavioral fingerprint for a specific account."""
        account_data = df[df["AccountID"] == account_id]
        features = self.feature_extractor.extract_sequence_features(account_data)

        return {
            "account_id": account_id,
            "fingerprint": features.iloc[0].to_dict(),
            "transaction_count": len(account_data),
        }

    def explain_anomaly(self, transaction: pd.Series, df: pd.DataFrame) -> dict:
        """Explain why a transaction is flagged as anomalous."""
        account_id = transaction["AccountID"]
        account_history = df[df["AccountID"] == account_id]

        realtime = self.feature_extractor.extract_realtime_features(transaction, account_history)
        explanations = []

        if realtime["amount_zscore"] > 2:
            explanations.append(f"Amount is {realtime['amount_zscore']:.1f}x above account average")

        if realtime["hour_typicality"] < 0.1:
            hour = transaction["TransactionDate"].hour
            explanations.append(f"Transaction at {hour}:00 is unusual for this account")

        if realtime["location_is_new"]:
            explanations.append(f"New location: {transaction['Location']}")

        if realtime["device_is_new"]:
            explanations.append(f"New device: {transaction['DeviceID']}")

        if realtime["time_since_last"] < 5:
            explanations.append(f"Very quick transaction ({realtime['time_since_last']:.1f} mins since last)")

        if transaction["LoginAttempts"] > 1:
            explanations.append(f"Multiple login attempts: {transaction['LoginAttempts']}")

        return {
            "transaction_id": transaction.get("TransactionID", ""),
            "account_id": account_id,
            "risk_factors": explanations,
        }

    def save_model(self, path: str = "models/behavioral_fingerprint"):
        """Save model with all components."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # PyTorch model
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim
            },
            "thresholds": self.threshold_manager.account_thresholds,
            "feature_names": self.feature_names,
            "scaler": self.scaler,
        }, path / "behavioral_fingerprint.pt")

        # TorchScript
        self.model.eval()
        dummy_input = torch.randn(1, self.input_dim).to(self.device)
        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(path / "behavioral_fingerprint_trace.pt")

        # ONNX (skip if onnxscript not available)
        try:
            torch.onnx.export(
                self.model,
                dummy_input,
                path / "behavioral_fingerprint.onnx",
                input_names=["transaction_features"],
                output_names=["reconstruction", "latent", "attention"],
                dynamic_axes={
                    "transaction_features": {0: "batch_size"},
                }
            )
        except (ModuleNotFoundError, ImportError) as e:
            print(f"Skipping ONNX export: {e}")

        # Metadata
        metadata = {
            "model_type": "BehavioralFingerprint",
            "version": "1.0.0",
            "input_features": self.feature_names,
            "latent_dim": self.latent_dim,
            "trained_at": datetime.now().isoformat(),
        }

        with open(path / "model_config.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Model saved to {path}/")

    @classmethod
    def load_model(cls, path: str = "models/behavioral_fingerprint"):
        """Load saved model."""
        path = Path(path)
        checkpoint = torch.load(path / "behavioral_fingerprint.pt", weights_only=False)

        model = cls(
            input_dim=checkpoint["model_config"]["input_dim"],
            latent_dim=checkpoint["model_config"]["latent_dim"]
        )

        model.model.load_state_dict(checkpoint["model_state_dict"])
        model.feature_names = checkpoint["feature_names"]
        model.scaler = checkpoint["scaler"]
        model.threshold_manager.account_thresholds = checkpoint["thresholds"]
        model.is_fitted = True

        print(f"Model loaded from {path}/")
        return model


def train_behavioral_model(
    df: pd.DataFrame,
    save_path: str = "models/behavioral_fingerprint"
) -> BehavioralFingerprintModel:
    """Quick training function."""
    model = BehavioralFingerprintModel()
    model.train(df)
    model.save_model(save_path)
    return model


if __name__ == "__main__":
    print("Behavioral Fingerprinting Model for Fraud Detection")
    print("Original approach for Devoteam interview")

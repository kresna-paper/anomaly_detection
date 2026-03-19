"""
Unit tests for the Feature Engineering module.

Tests feature creation, transformation, and edge cases.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import TransactionFeatureEngineer


@pytest.fixture
def sample_transaction_data():
    """Create sample transaction data for testing."""
    return pd.DataFrame({
        'TransactionID': ['TX001', 'TX002', 'TX003', 'TX004', 'TX005'],
        'AccountID': ['AC001', 'AC001', 'AC002', 'AC002', 'AC003'],
        'TransactionAmount': [100.50, 1500.00, 75.25, 5000.00, 25.00],
        'TransactionDate': pd.to_datetime([
            '2024-01-15 14:30:00',
            '2024-01-15 16:00:00',
            '2024-01-15 09:15:00',
            '2024-01-15 23:45:00',
            '2024-01-15 03:00:00'
        ]),
        'TransactionType': ['Debit', 'Credit', 'Debit', 'Debit', 'Credit'],
        'Location': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
        'DeviceID': ['D001', 'D001', 'D002', 'D003', 'D001'],
        'IP Address': ['192.168.1.1', '192.168.1.1', '10.0.0.1', '10.0.0.2', '192.168.1.1'],
        'MerchantID': ['M001', 'M002', 'M001', 'M003', 'M002'],
        'Channel': ['Online', 'Branch', 'ATM', 'Online', 'Online'],
        'CustomerAge': [35, 35, 42, 28, 65],
        'CustomerOccupation': ['Engineer', 'Engineer', 'Doctor', 'Student', 'Retired'],
        'TransactionDuration': [45, 120, 30, 300, 15],
        'LoginAttempts': [1, 3, 1, 5, 1],
        'AccountBalance': [5000.00, 6500.00, 2500.00, 1000.00, 15000.00],
        'PreviousTransactionDate': pd.to_datetime([
            '2024-01-15 10:00:00',
            '2024-01-15 14:30:00',
            '2024-01-14 18:00:00',
            '2024-01-15 20:00:00',
            '2024-01-14 12:00:00'
        ]),
    })


class TestTransactionFeatureEngineer:
    """Tests for TransactionFeatureEngineer class."""

    def test_initialization(self):
        """Test engineer initialization."""
        engineer = TransactionFeatureEngineer()
        assert engineer is not None
        assert engineer.historical_data is None

    def test_initialization_with_historical_data(self, sample_transaction_data):
        """Test initialization with historical data."""
        engineer = TransactionFeatureEngineer(historical_data=sample_transaction_data)
        assert engineer.historical_data is not None
        assert len(engineer.account_stats) == 3  # AC001, AC002, AC003
        assert 'AC001' in engineer.account_stats
        assert engineer.global_stats is not None

    def test_transform_adds_features(self, sample_transaction_data):
        """Test that transform adds new features."""
        engineer = TransactionFeatureEngineer(historical_data=sample_transaction_data)
        result = engineer.transform(sample_transaction_data)

        # Original columns should be preserved
        assert 'TransactionID' in result.columns
        assert 'TransactionAmount' in result.columns

        # New features should be added
        assert 'Amount_Log' in result.columns
        assert 'Hour_Sin' in result.columns
        assert 'Hour_Cos' in result.columns
        assert 'IsWeekend' in result.columns

    def test_amount_features(self, sample_transaction_data):
        """Test amount-based features."""
        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        # Log amount
        assert result['Amount_Log'].min() > 0

        # Z-score
        assert 'Amount_ZScore_Global' in result.columns

        # High amount flags
        assert 'Amount_IsHigh' in result.columns
        assert 'Amount_IsVeryHigh' in result.columns

    def test_time_features(self, sample_transaction_data):
        """Test time-based features."""
        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        assert 'Hour' in result.columns
        assert 'DayOfWeek' in result.columns
        assert 'IsWeekend' in result.columns
        assert 'IsBusinessHours' in result.columns
        assert 'IsUnusualHour' in result.columns

        # Check unusual hour flag (TX004 at 23:45, TX005 at 03:00)
        assert result.loc[result['TransactionID'] == 'TX004', 'IsUnusualHour'].values[0] == 1
        assert result.loc[result['TransactionID'] == 'TX005', 'IsUnusualHour'].values[0] == 1

    def test_categorical_encodings(self, sample_transaction_data):
        """Test categorical encoding features."""
        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        # Channel encodings
        assert 'Channel_Online' in result.columns
        assert 'Channel_Branch' in result.columns
        assert 'Channel_ATM' in result.columns

        # Transaction type
        assert 'Type_Debit' in result.columns

        # Occupation encodings
        assert 'Occupation_Engineer' in result.columns
        assert 'Occupation_Doctor' in result.columns

    def test_behavioral_features(self, sample_transaction_data):
        """Test behavioral risk features."""
        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        assert 'Multi_Login' in result.columns
        assert 'High_Login_Attempts' in result.columns

        # Check multi-login flags
        # TX002 has 3 login attempts, TX004 has 5
        assert result.loc[result['TransactionID'] == 'TX002', 'Multi_Login'].values[0] == 1
        assert result.loc[result['TransactionID'] == 'TX004', 'Multi_Login'].values[0] == 1

    def test_risk_scores(self, sample_transaction_data):
        """Test composite risk scores."""
        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        assert 'Risk_Score_Raw' in result.columns
        assert 'Risk_Score_Normalized' in result.columns

        # Risk scores should be non-negative
        assert (result['Risk_Score_Raw'] >= 0).all()

    def test_account_history_features(self, sample_transaction_data):
        """Test account-level historical features."""
        engineer = TransactionFeatureEngineer(historical_data=sample_transaction_data)
        result = engineer.transform(sample_transaction_data)

        assert 'Account_Amount_ZScore' in result.columns
        assert 'Account_TxCount' in result.columns
        assert 'Account_DeviceCount' in result.columns

        # Check account stats
        ac001_rows = result[result['AccountID'] == 'AC001']
        assert (ac001_rows['Account_TxCount'] == 2).all()

    def test_get_feature_columns(self):
        """Test get_feature_columns method."""
        engineer = TransactionFeatureEngineer()
        features = engineer.get_feature_columns()

        assert isinstance(features, list)
        assert len(features) > 0
        assert 'Amount_Log' in features
        assert 'Hour_Sin' in features

    def test_transform_for_model(self, sample_transaction_data):
        """Test transform_for_model returns only feature columns."""
        engineer = TransactionFeatureEngineer()
        X, feature_names = engineer.transform_for_model(sample_transaction_data)

        assert isinstance(X, pd.DataFrame)
        assert len(feature_names) > 0

        # Should only contain numeric features
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col])

    def test_handles_missing_values(self, sample_transaction_data):
        """Test that transformation handles missing values."""
        # Add some NaN values
        sample_transaction_data.loc[0, 'AccountBalance'] = np.nan

        engineer = TransactionFeatureEngineer()
        result = engineer.transform(sample_transaction_data)

        # Should not raise error
        assert result is not None
        assert len(result) == len(sample_transaction_data)

    def test_single_transaction(self, sample_transaction_data):
        """Test transformation of single transaction."""
        single_tx = sample_transaction_data.iloc[[0]]

        engineer = TransactionFeatureEngineer()
        result = engineer.transform(single_tx)

        assert len(result) == 1
        assert 'Amount_Log' in result.columns


class TestFeatureEngineerEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        engineer = TransactionFeatureEngineer()
        empty_df = pd.DataFrame()

        # Should handle gracefully
        result = engineer.transform(empty_df)
        assert result is not None

    def test_missing_columns(self):
        """Test handling of missing required columns."""
        incomplete_data = pd.DataFrame({
            'TransactionID': ['TX001'],
            'TransactionAmount': [100.00]
        })

        engineer = TransactionFeatureEngineer()
        # Should not crash, but some features won't be created
        result = engineer.transform(incomplete_data)
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Feature Engineering Module for Bank Transaction Fraud Detection

This module creates behavioral, temporal, and aggregation features
that are critical for anomaly detection in financial transactions.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class TransactionFeatureEngineer:
    """
    Feature engineering for bank transaction fraud detection.

    Creates features based on:
    - Transaction behavior (amount ratios, timing)
    - Account history (aggregations, patterns)
    - Device/IP fingerprinting
    - Temporal patterns
    """

    def __init__(self, historical_data: pd.DataFrame | None = None):
        """
        Initialize the feature engineer.

        Args:
            historical_data: Optional DataFrame of historical transactions
                           for computing account-level aggregations
        """
        self.historical_data = historical_data
        self.account_stats: dict[str, dict] = {}
        self.global_stats: dict = {}

        if historical_data is not None:
            self._fit(historical_data)

    def _fit(self, df: pd.DataFrame) -> None:
        """
        Compute statistics from historical data.

        Args:
            df: Historical transaction data
        """
        # Convert dates if needed
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

        # Compute per-account statistics
        if 'AccountID' in df.columns:
            self.account_stats = self._compute_account_stats(df)

        # Compute global statistics
        self.global_stats = self._compute_global_stats(df)

    def _compute_account_stats(self, df: pd.DataFrame) -> dict[str, dict]:
        """Compute per-account historical statistics."""
        stats = {}

        for account_id, group in df.groupby('AccountID'):
            stats[account_id] = {
                'tx_count': len(group),
                'avg_amount': float(group['TransactionAmount'].mean()),
                'std_amount': float(group['TransactionAmount'].std()) if len(group) > 1 else 0,
                'max_amount': float(group['TransactionAmount'].max()),
                'min_amount': float(group['TransactionAmount'].min()),
                'avg_balance': float(group['AccountBalance'].mean()),
                'devices': group['DeviceID'].nunique(),
                'locations': group['Location'].nunique(),
                'ips': group['IP Address'].nunique(),
                'channels': group['Channel'].nunique() if 'Channel' in group.columns else 1,
                'merchant_count': group['MerchantID'].nunique(),
                'last_transaction': group['TransactionDate'].max() if 'TransactionDate' in group.columns else None,
                'multi_login_ratio': float((group['LoginAttempts'] > 1).sum() / len(group)),
            }

        return stats

    def _compute_global_stats(self, df: pd.DataFrame) -> dict:
        """Compute global statistics."""
        return {
            'avg_amount': float(df['TransactionAmount'].mean()),
            'std_amount': float(df['TransactionAmount'].std()),
            'median_amount': float(df['TransactionAmount'].median()),
            'p95_amount': float(df['TransactionAmount'].quantile(0.95)),
            'p99_amount': float(df['TransactionAmount'].quantile(0.99)),
            'avg_duration': float(df['TransactionDuration'].mean()) if 'TransactionDuration' in df.columns else 0,
            'avg_login_attempts': float(df['LoginAttempts'].mean()) if 'LoginAttempts' in df.columns else 1,
        }

    def transform(self, df: pd.DataFrame, fit_new_accounts: bool = True) -> pd.DataFrame:
        """
        Transform transaction data by adding engineered features.

        Args:
            df: Transaction data to transform
            fit_new_accounts: Whether to update stats for new accounts

        Returns:
            DataFrame with engineered features
        """
        df = df.copy()

        # Ensure datetime columns
        if 'TransactionDate' in df.columns:
            df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
        if 'PreviousTransactionDate' in df.columns:
            df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])

        # Add all feature groups
        df = self._add_amount_features(df)
        df = self._add_ratio_features(df)
        df = self._add_time_features(df)
        df = self._add_categorical_encodings(df)
        df = self._add_account_history_features(df, fit_new_accounts)
        df = self._add_behavioral_features(df)
        df = self._add_risk_scores(df)

        return df

    def _add_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction amount-based features."""
        # Log-transformed amount
        df['Amount_Log'] = np.log1p(df['TransactionAmount'])

        # Amount z-score (global)
        if self.global_stats:
            mean_amount = self.global_stats.get('avg_amount', df['TransactionAmount'].mean())
            std_amount = self.global_stats.get('std_amount', df['TransactionAmount'].std())
            df['Amount_ZScore_Global'] = (df['TransactionAmount'] - mean_amount) / (std_amount + 1e-6)

        # Amount percentiles
        df['Amount_IsHigh'] = (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.75)).astype(int)
        df['Amount_IsVeryHigh'] = (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.95)).astype(int)

        # Round amount (potential structuring indicator)
        df['Amount_IsRound'] = (df['TransactionAmount'] % 100 < 1).astype(int)

        return df

    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio-based features."""
        # Amount to balance ratio
        df['Amount_Balance_Ratio'] = df['TransactionAmount'] / (df['AccountBalance'] + df['TransactionAmount'])

        # Amount relative to typical transaction size
        if self.global_stats:
            df['Amount_vs_GlobalAvg'] = df['TransactionAmount'] / (self.global_stats.get('avg_amount', 1) + 1e-6)

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'TransactionDate' not in df.columns:
            return df

        df['Hour'] = df['TransactionDate'].dt.hour
        df['DayOfWeek'] = df['TransactionDate'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['DayOfMonth'] = df['TransactionDate'].dt.day

        # Cyclical encoding for hour
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)

        # Cyclical encoding for day of week
        df['Day_Sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['Day_Cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

        # Time since previous transaction
        if 'PreviousTransactionDate' in df.columns:
            df['Minutes_Since_Last_Tx'] = (
                (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds() / 60
            )
            # Handle negative values (data quality issue)
            df['Minutes_Since_Last_Tx'] = df['Minutes_Since_Last_Tx'].clip(lower=0)

            # Flag very quick consecutive transactions
            df['Quick_Consecutive_Tx'] = (df['Minutes_Since_Last_Tx'] < 5).astype(int)

        # Business hours
        df['IsBusinessHours'] = ((df['Hour'] >= 9) & (df['Hour'] <= 17)).astype(int)

        # Unusual hours (late night)
        df['IsUnusualHour'] = ((df['Hour'] >= 23) | (df['Hour'] <= 5)).astype(int)

        return df

    def _add_categorical_encodings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encoded categorical features."""
        # Channel encoding
        if 'Channel' in df.columns:
            for channel in ['Online', 'ATM', 'Branch']:
                df[f'Channel_{channel}'] = (df['Channel'] == channel).astype(int)

        # Transaction type encoding
        if 'TransactionType' in df.columns:
            df['Type_Debit'] = (df['TransactionType'] == 'Debit').astype(int)

        # Occupation encoding
        if 'CustomerOccupation' in df.columns:
            for occupation in ['Student', 'Doctor', 'Engineer', 'Retired']:
                df[f'Occupation_{occupation}'] = (df['CustomerOccupation'] == occupation).astype(int)

        return df

    def _add_account_history_features(self, df: pd.DataFrame, fit_new_accounts: bool) -> pd.DataFrame:
        """Add account-level historical features."""
        features = []

        for idx, row in df.iterrows():
            account_id = row.get('AccountID', '')
            stats = self.account_stats.get(account_id, {})

            # Amount deviation from account average
            if stats.get('avg_amount'):
                account_avg = stats['avg_amount']
                account_std = stats.get('std_amount', 1) + 1e-6
                amount_zscore_account = (row['TransactionAmount'] - account_avg) / account_std
            else:
                amount_zscore_account = 0

            features.append({
                'Account_AvgAmount': stats.get('avg_amount', row['TransactionAmount']),
                'Account_Amount_ZScore': amount_zscore_account,
                'Account_TxCount': stats.get('tx_count', 1),
                'Account_DeviceCount': stats.get('devices', 1),
                'Account_LocationCount': stats.get('locations', 1),
                'Account_IPCount': stats.get('ips', 1),
                'Account_MultiLoginRatio': stats.get('multi_login_ratio', 0),
            })

        feature_df = pd.DataFrame(features, index=df.index)
        df = pd.concat([df, feature_df], axis=1)

        return df

    def _add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add behavioral risk features."""
        # Multiple login attempts flag
        df['Multi_Login'] = (df['LoginAttempts'] > 1).astype(int)
        df['High_Login_Attempts'] = (df['LoginAttempts'] >= 3).astype(int)

        # Very long transaction duration
        if 'TransactionDuration' in df.columns:
            df['Long_Duration'] = (df['TransactionDuration'] > df['TransactionDuration'].quantile(0.9)).astype(int)

        # New device indicator (if historical data exists)
        if self.account_stats:
            df['New_Device'] = 0  # Would need real-time checking
            df['New_Location'] = 0

        # High risk channel (Online)
        if 'Channel' in df.columns:
            df['Is_Online'] = (df['Channel'] == 'Online').astype(int)

        return df

    def _add_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add composite risk scores."""
        risk_components = []

        for idx, row in df.iterrows():
            score = 0

            # Login attempts risk
            if row.get('LoginAttempts', 1) > 1:
                score += min(20, (row['LoginAttempts'] - 1) * 10)

            # High amount risk
            if self.global_stats:
                p95 = self.global_stats.get('p95_amount', row['TransactionAmount'])
                if row['TransactionAmount'] > p95:
                    score += 15

            # Unusual time risk
            if row.get('IsUnusualHour', 0) == 1:
                score += 10

            # Online channel risk
            if row.get('Is_Online', 0) == 1:
                score += 5

            # Quick consecutive transactions
            if row.get('Quick_Consecutive_Tx', 0) == 1:
                score += 10

            # Amount to balance ratio risk
            if row.get('Amount_Balance_Ratio', 0) > 0.5:
                score += 15

            risk_components.append(score)

        df['Risk_Score_Raw'] = risk_components
        df['Risk_Score_Normalized'] = np.array(risk_components) / 100  # Normalize to 0-1

        return df

    def get_feature_columns(self) -> list[str]:
        """Return list of engineered feature columns."""
        return [
            # Amount features
            'Amount_Log', 'Amount_ZScore_Global', 'Amount_IsHigh', 'Amount_IsVeryHigh', 'Amount_IsRound',
            'Amount_Balance_Ratio', 'Amount_vs_GlobalAvg',

            # Time features
            'Hour_Sin', 'Hour_Cos', 'Day_Sin', 'Day_Cos', 'IsWeekend',
            'IsBusinessHours', 'IsUnusualHour', 'Quick_Consecutive_Tx',

            # Categorical (one-hot)
            'Channel_Online', 'Channel_ATM', 'Channel_Branch',
            'Type_Debit',
            'Occupation_Student', 'Occupation_Doctor', 'Occupation_Engineer', 'Occupation_Retired',

            # Account history
            'Account_Amount_ZScore', 'Account_TxCount', 'Account_DeviceCount',
            'Account_LocationCount', 'Account_IPCount', 'Account_MultiLoginRatio',

            # Behavioral
            'Multi_Login', 'High_Login_Attempts', 'Long_Duration', 'Is_Online',

            # Risk scores
            'Risk_Score_Normalized',
        ]

    def transform_for_model(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """
        Transform data and return only the feature columns needed for modeling.

        Args:
            df: Transaction data

        Returns:
            Tuple of (DataFrame with features, list of feature names)
        """
        df_transformed = self.transform(df)
        feature_cols = self.get_feature_columns()

        # Only return columns that exist
        available_cols = [col for col in feature_cols if col in df_transformed.columns]

        return df_transformed[available_cols], available_cols


def create_feature_set_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-shot feature creation from a DataFrame without historical context.

    Args:
        df: Transaction DataFrame

    Returns:
        DataFrame with engineered features
    """
    engineer = TransactionFeatureEngineer()
    return engineer.transform(df)


if __name__ == "__main__":
    # Test the feature engineer
    print("Testing TransactionFeatureEngineer...")

    # Create sample data
    sample_data = pd.DataFrame({
        'TransactionID': ['TX00001', 'TX00002', 'TX00003'],
        'AccountID': ['AC001', 'AC001', 'AC002'],
        'TransactionAmount': [100.50, 1500.00, 75.25],
        'TransactionDate': ['2024-01-15 14:30:00', '2024-01-15 16:00:00', '2024-01-15 09:15:00'],
        'TransactionType': ['Debit', 'Credit', 'Debit'],
        'Location': ['New York', 'Los Angeles', 'Chicago'],
        'DeviceID': ['D001', 'D001', 'D002'],
        'IP Address': ['192.168.1.1', '192.168.1.1', '10.0.0.1'],
        'MerchantID': ['M001', 'M002', 'M001'],
        'Channel': ['Online', 'Branch', 'ATM'],
        'CustomerAge': [35, 35, 42],
        'CustomerOccupation': ['Engineer', 'Engineer', 'Doctor'],
        'TransactionDuration': [45, 120, 30],
        'LoginAttempts': [1, 3, 1],
        'AccountBalance': [5000.00, 6500.00, 2500.00],
        'PreviousTransactionDate': ['2024-01-15 10:00:00', '2024-01-15 14:30:00', '2024-01-14 18:00:00'],
    })

    # Initialize with sample data
    engineer = TransactionFeatureEngineer(sample_data)

    # Transform
    result = engineer.transform(sample_data)

    print(f"\nOriginal shape: {sample_data.shape}")
    print(f"Transformed shape: {result.shape}")
    print(f"\nEngineered features: {len(engineer.get_feature_columns())}")
    print(f"\nFeature columns:\n{engineer.get_feature_columns()}")

    print("\n" + "="*60)
    print("Sample of transformed data:")
    print("="*60)
    print(result[['TransactionID', 'Amount_ZScore_Global', 'Risk_Score_Normalized',
                  'Multi_Login', 'IsUnusualHour']].to_string())

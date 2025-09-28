"""
Preprocessing Service for Customer Churn Analysis
Handles data cleaning, feature engineering, and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class PreprocessingService:
    """Service for preprocessing customer churn data"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.fitted = False

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values in the dataset

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (cleaned_df, missing_report)
        """
        missing_report = {
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
        }

        # Create a copy to avoid modifying original
        df_clean = df.copy()

        # Handle numeric columns with median imputation
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)

        # Handle categorical columns with mode imputation
        categorical_columns = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df_clean[col].isnull().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)

        return df_clean, missing_report

    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
        """
        Encode categorical features using label encoding

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (encoded_df, encoders_dict)
        """
        df_encoded = df.copy()

        # Get categorical columns
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                # Handle unseen categories during transform
                try:
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
                except ValueError:
                    # Handle unseen labels by encoding them as the most frequent class
                    most_frequent = df_encoded[col].mode()[0] if not df_encoded[col].mode().empty else df_encoded[col].iloc[0]
                    df_encoded[col] = df_encoded[col].map(
                        lambda x: x if x in self.label_encoders[col].classes_ else most_frequent
                    )
                    df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))

        return df_encoded, self.label_encoders

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features for better model performance

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with additional interaction features
        """
        df_features = df.copy()

        # Assuming some common telecom features exist
        if 'tenure' in df_features.columns and 'monthly_charges' in df_features.columns:
            df_features['tenure_charges_ratio'] = df_features['monthly_charges'] / (df_features['tenure'] + 1)

        if 'total_charges' in df_features.columns and 'monthly_charges' in df_features.columns:
            df_features['charges_ratio'] = df_features['total_charges'] / (df_features['monthly_charges'] + 1)

        # Create polynomial features for important numerical columns
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Limit to first 3 to avoid too many features
            if col not in ['churn']:  # Don't transform target
                df_features[f'{col}_squared'] = df_features[col] ** 2

        return df_features

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler (True for training, False for test)

        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()

        # Get numerical columns (excluding target if present)
        numeric_columns = df_scaled.select_dtypes(include=[np.number]).columns
        if 'churn' in numeric_columns:
            numeric_columns = numeric_columns.drop('churn')

        if fit:
            df_scaled[numeric_columns] = self.scaler.fit_transform(df_scaled[numeric_columns])
            self.fitted = True
        else:
            if not self.fitted:
                raise ValueError("Scaler must be fitted before transform. Call with fit=True first.")
            df_scaled[numeric_columns] = self.scaler.transform(df_scaled[numeric_columns])

        return df_scaled

    def preprocess_pipeline(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete preprocessing pipeline

        Args:
            df: Input DataFrame
            fit: Whether to fit preprocessing components

        Returns:
            Tuple of (processed_df, preprocessing_report)
        """
        preprocessing_report = {}

        # Step 1: Handle missing values
        df_clean, missing_report = self.handle_missing_values(df)
        preprocessing_report['missing_values'] = missing_report

        # Step 2: Encode categorical features
        df_encoded, encoders = self.encode_categorical_features(df_clean)
        preprocessing_report['encoders'] = list(encoders.keys())

        # Step 3: Create interaction features
        df_features = self.create_interaction_features(df_encoded)
        preprocessing_report['feature_count'] = len(df_features.columns)

        # Step 4: Scale features
        df_processed = self.scale_features(df_features, fit=fit)
        preprocessing_report['scaling'] = 'StandardScaler applied'

        return df_processed, preprocessing_report
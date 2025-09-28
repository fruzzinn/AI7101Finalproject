"""
Contract: Data Preprocessing Interface
Purpose: Transform raw customer data for machine learning
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

class PreprocessingContract:
    """Contract for data preprocessing operations"""

    def handle_missing_values(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Implement robust missing value imputation

        Args:
            df: Dataset with potential missing values

        Returns:
            Tuple of (cleaned_df, imputation_strategy_dict)

        Contract Requirements:
            - Numerical columns: median/mean imputation with justification
            - Categorical columns: mode imputation or 'unknown' category
            - Document strategy used for each column
            - Preserve original data distribution characteristics
        """
        raise NotImplementedError("Must implement missing value handling")

    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        Transform categorical variables for ML algorithms

        Args:
            df: Dataset with categorical columns

        Returns:
            Tuple of (encoded_df, encoder_objects_dict)

        Contract Requirements:
            - One-hot encoding for low cardinality (< 10 unique values)
            - Target encoding for high cardinality categorical features
            - Ordinal encoding for ranked categories
            - Store encoders for consistent test set transformation
        """
        raise NotImplementedError("Must implement categorical encoding")

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Create derived features that capture customer behavior patterns

        Args:
            df: Dataset with base features

        Returns:
            Tuple of (feature_engineered_df, new_feature_names)

        Contract Requirements:
            - Create customer tenure features
            - Generate usage ratio metrics
            - Calculate temporal trend features
            - Derive customer value metrics (ARPU, CLV estimates)
            - All new features must be business-interpretable
        """
        raise NotImplementedError("Must implement feature engineering")

    def scale_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, object]:
        """
        Scale numerical features for model training

        Args:
            df: Dataset with numerical features

        Returns:
            Tuple of (scaled_df, scaler_object)

        Contract Requirements:
            - StandardScaler for normally distributed features
            - RobustScaler for features with outliers
            - MinMaxScaler for bounded features where appropriate
            - Store scaler for consistent test set transformation
        """
        raise NotImplementedError("Must implement feature scaling")
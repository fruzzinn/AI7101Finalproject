"""
Preprocessing Service Implementation
Implements PreprocessingContract for comprehensive data preprocessing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class PreprocessingService:
    """Service for comprehensive data preprocessing including missing values, encoding, and feature engineering"""

    def __init__(self):
        """Initialize the preprocessing service"""
        self.high_cardinality_threshold = 10
        self.outlier_threshold = 3.0

    def handle_missing_values(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle missing values using appropriate strategies for different data types

        Args:
            data: DataFrame with potential missing values

        Returns:
            Tuple of (cleaned_df, strategy_dict)
        """
        if data.empty:
            raise ValueError("Cannot handle missing values on empty DataFrame")

        cleaned_df = data.copy()
        strategy_dict = {}

        # Identify column types
        numerical_cols = cleaned_df.select_dtypes(include=[np.number]).columns
        categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns

        # Handle numerical missing values
        for col in numerical_cols:
            if cleaned_df[col].isnull().any():
                # Check if column is all missing
                if cleaned_df[col].isnull().all():
                    # Fill with 0 for completely missing numerical columns
                    cleaned_df[col] = 0
                    strategy_dict[col] = {'method': 'zero_fill', 'imputer': None}
                else:
                    # Use median for numerical columns (robust to outliers)
                    imputer = SimpleImputer(strategy='median')
                    cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]]).flatten()
                    strategy_dict[col] = {'method': 'median', 'imputer': imputer}

        # Handle categorical missing values
        for col in categorical_cols:
            if cleaned_df[col].isnull().any():
                # Check if column is all missing
                if cleaned_df[col].isnull().all():
                    # Fill with 'Unknown' for completely missing categorical columns
                    cleaned_df[col] = 'Unknown'
                    strategy_dict[col] = {'method': 'unknown_fill', 'imputer': None}
                else:
                    # Use mode for categorical columns
                    imputer = SimpleImputer(strategy='most_frequent')
                    cleaned_df[col] = imputer.fit_transform(cleaned_df[[col]]).flatten()
                    strategy_dict[col] = {'method': 'most_frequent', 'imputer': imputer}

        return cleaned_df, strategy_dict

    def encode_categorical_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Encode categorical features using appropriate strategies based on cardinality

        Args:
            data: DataFrame with categorical features

        Returns:
            Tuple of (encoded_df, encoder_dict)
        """
        if data.empty:
            raise ValueError("Cannot encode categorical features on empty DataFrame")

        encoded_df = data.copy()
        encoder_dict = {}

        # Identify categorical columns (exclude ID columns)
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        id_columns = [col for col in categorical_cols if 'id' in col.lower()]
        categorical_cols = [col for col in categorical_cols if col not in id_columns]

        if len(categorical_cols) == 0:
            return encoded_df, encoder_dict

        for col in categorical_cols:
            unique_count = data[col].nunique()

            if unique_count <= self.high_cardinality_threshold:
                # Low cardinality: One-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_cols = encoder.fit_transform(data[[col]])

                # Create column names
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]

                # Add encoded columns to dataframe
                encoded_cols_df = pd.DataFrame(encoded_cols, columns=feature_names, index=data.index)
                encoded_df = pd.concat([encoded_df.drop(col, axis=1), encoded_cols_df], axis=1)

                encoder_dict[col] = {'type': 'onehot', 'encoder': encoder, 'feature_names': feature_names}

            else:
                # High cardinality: Label encoding (could be improved with target encoding)
                encoder = LabelEncoder()
                encoded_df[col] = encoder.fit_transform(data[col].astype(str))
                encoder_dict[col] = {'type': 'label', 'encoder': encoder}

        # Handle ordinal features (satisfaction levels)
        if 'satisfaction_level' in encoded_df.columns:
            ordinal_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
            if all(val in ordinal_mapping for val in data['satisfaction_level'].dropna().unique()):
                encoded_df['satisfaction_level'] = data['satisfaction_level'].map(ordinal_mapping)
                encoder_dict['satisfaction_level'] = {'type': 'ordinal', 'mapping': ordinal_mapping}

        return encoded_df, encoder_dict

    def engineer_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Engineer business-interpretable features

        Args:
            data: DataFrame with base features

        Returns:
            Tuple of (featured_df, new_features_list)
        """
        featured_df = data.copy()
        new_features = []

        # Customer tenure features
        if 'tenure' in featured_df.columns:
            featured_df['tenure_years'] = featured_df['tenure'] / 12
            featured_df['tenure_category'] = pd.cut(featured_df['tenure'],
                                                   bins=[0, 12, 24, 36, float('inf')],
                                                   labels=['New', 'Developing', 'Established', 'Loyal'])
            featured_df['tenure_category'] = featured_df['tenure_category'].cat.codes
            new_features.extend(['tenure_years', 'tenure_category'])

        # Usage ratio features
        if 'total_calls' in featured_df.columns and 'plan_calls' in featured_df.columns:
            featured_df['call_usage_ratio'] = featured_df['total_calls'] / (featured_df['plan_calls'] + 1)
            new_features.append('call_usage_ratio')

        # Customer value features (ARPU - Average Revenue Per User)
        if 'monthly_charges' in featured_df.columns:
            featured_df['arpu'] = featured_df['monthly_charges']
            new_features.append('arpu')

        if 'total_charges' in featured_df.columns and 'tenure' in featured_df.columns:
            # Customer Lifetime Value estimate
            featured_df['clv_estimate'] = featured_df['total_charges'] / (featured_df['tenure'] + 1) * 12
            new_features.append('clv_estimate')

        # Charges per tenure ratio
        if 'total_charges' in featured_df.columns and 'tenure' in featured_df.columns:
            featured_df['charges_per_tenure'] = featured_df['total_charges'] / (featured_df['tenure'] + 1)
            new_features.append('charges_per_tenure')

        # Contract risk features
        if 'contract_type_Month-to-month' in featured_df.columns:
            featured_df['contract_risk_score'] = featured_df['contract_type_Month-to-month']
            new_features.append('contract_risk_score')

        # Payment method risk
        if 'payment_method_Electronic check' in featured_df.columns:
            featured_df['payment_risk_score'] = featured_df['payment_method_Electronic check']
            new_features.append('payment_risk_score')

        # Service bundling features
        service_columns = [col for col in featured_df.columns if any(service in col.lower()
                          for service in ['internet', 'phone', 'tv', 'streaming'])]
        if service_columns:
            featured_df['service_bundle_count'] = featured_df[service_columns].sum(axis=1)
            new_features.append('service_bundle_count')

        # Age group features
        if 'age' in featured_df.columns:
            featured_df['age_group'] = pd.cut(featured_df['age'],
                                            bins=[0, 30, 50, 65, float('inf')],
                                            labels=['Young', 'Middle', 'Senior', 'Elder'])
            featured_df['age_group'] = featured_df['age_group'].cat.codes
            new_features.append('age_group')

        return featured_df, new_features

    def scale_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Any]:
        """
        Scale numerical features while preserving categorical features

        Args:
            data: DataFrame with features to scale

        Returns:
            Tuple of (scaled_df, scaler_object)
        """
        if data.empty:
            raise ValueError("Cannot scale features on empty DataFrame")

        scaled_df = data.copy()

        # Identify numerical columns
        numerical_cols = scaled_df.select_dtypes(include=[np.number]).columns
        categorical_cols = scaled_df.select_dtypes(include=['object', 'category']).columns

        if len(numerical_cols) == 0:
            # No numerical columns to scale
            return scaled_df, None

        # Filter out columns with only one unique value (constants)
        valid_numerical_cols = []
        for col in numerical_cols:
            col_data = scaled_df[col].dropna()
            if len(col_data) > 0 and col_data.nunique() > 1:
                valid_numerical_cols.append(col)

        if len(valid_numerical_cols) == 0:
            # No valid columns to scale
            return scaled_df, None

        # Check for outliers to decide scaler type
        has_outliers = False
        for col in valid_numerical_cols:
            col_data = scaled_df[col].dropna()
            if len(col_data) > 1:  # Need at least 2 values for z-score
                try:
                    from scipy import stats
                    z_scores = np.abs(stats.zscore(col_data))
                    if (z_scores > self.outlier_threshold).any():
                        has_outliers = True
                        break
                except:
                    # Skip outlier detection if scipy not available or other error
                    pass

        # Use appropriate scaler based on outlier detection
        if has_outliers:
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        # Scale only valid numerical columns
        scaled_values = scaler.fit_transform(scaled_df[valid_numerical_cols])
        scaled_df[valid_numerical_cols] = scaled_values

        return scaled_df, scaler
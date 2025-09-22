"""
FeatureProcessor implementation for telecommunications churn prediction.

Educational Focus: Demonstrates comprehensive ML feature engineering patterns.
This module implements the FeatureProcessorContract for preprocessing raw customer data
into machine learning-ready features.
"""

from typing import Tuple, Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
from datetime import datetime

from ..models.processed_features import ProcessedFeatures, FeatureMetadata, FeatureType, ScalingMethod
from ..models.customer_profile import CustomerProfile


class FeatureProcessor:
    """
    Feature engineering and preprocessing operations for churn prediction.

    Educational Notes:
    - Implements contract-based architecture for maintainability
    - Supports multiple encoding and scaling strategies
    - Tracks all transformations for reproducibility
    - Handles both categorical and numerical features appropriately
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureProcessor.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.fitted_transformers: Dict[str, Any] = {}
        self.preprocessing_steps: List[str] = []

        # Set random seeds for reproducibility
        np.random.seed(self.random_state)

    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding strategy.

        Args:
            df: DataFrame with categorical features
            categorical_cols: List of categorical column names

        Returns:
            DataFrame with encoded categorical features

        Educational Notes:
        - One-hot encoding for nominal categories with low cardinality
        - Ordinal encoding for ordered categories
        - Label encoding for high cardinality nominal features
        - Handles unknown categories in test data gracefully
        """
        df_encoded = df.copy()
        encoding_info = {}

        for col in categorical_cols:
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue

            unique_values = df[col].nunique()

            # Strategy selection based on cardinality and domain knowledge
            if col in ['gender', 'partner', 'dependents', 'phone_service', 'paperless_billing']:
                # Binary categorical features - use label encoding
                encoder = LabelEncoder()
                df_encoded[col] = encoder.fit_transform(df[col].fillna('Unknown'))
                encoding_info[col] = {
                    'method': 'label',
                    'encoder': encoder,
                    'categories': encoder.classes_.tolist()
                }
                self.preprocessing_steps.append(f"Applied label encoding to {col}")

            elif col in ['contract', 'payment_method'] or unique_values <= 5:
                # Low cardinality nominal features - use one-hot encoding
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(df[[col]].fillna('Unknown'))

                # Create feature names for one-hot encoded columns
                feature_names = [f"{col}_{category}" for category in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df.index)

                # Drop original column and add encoded columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

                encoding_info[col] = {
                    'method': 'onehot',
                    'encoder': encoder,
                    'feature_names': feature_names,
                    'categories': encoder.categories_[0].tolist()
                }
                self.preprocessing_steps.append(f"Applied one-hot encoding to {col} (created {len(feature_names)} features)")

            else:
                # High cardinality features - use ordinal encoding with frequency
                value_counts = df[col].value_counts()
                ordinal_mapping = {value: idx for idx, value in enumerate(value_counts.index)}
                ordinal_mapping['Unknown'] = len(ordinal_mapping)  # Handle missing values

                df_encoded[col] = df[col].fillna('Unknown').map(ordinal_mapping)

                encoding_info[col] = {
                    'method': 'ordinal_frequency',
                    'mapping': ordinal_mapping,
                    'categories': list(ordinal_mapping.keys())
                }
                self.preprocessing_steps.append(f"Applied frequency-based ordinal encoding to {col}")

        # Store encoding information for future use
        self.fitted_transformers['categorical_encoders'] = encoding_info

        return df_encoded

    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values using specified strategies.

        Args:
            df: DataFrame with missing values
            strategy: Dict mapping column names to imputation strategies
                     ('median', 'mode', 'constant', 'drop', 'knn')

        Returns:
            DataFrame with missing values handled

        Educational Notes:
        - Different strategies for numerical vs categorical features
        - Creates missingness indicators when appropriate
        - Documents impact of imputation choices
        - KNN imputation for features with complex relationships
        """
        df_imputed = df.copy()
        imputation_info = {}

        for col, impute_strategy in strategy.items():
            if col not in df.columns:
                warnings.warn(f"Column '{col}' not found in DataFrame")
                continue

            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue

            missing_percentage = missing_count / len(df) * 100

            # Create missingness indicator if missing rate is significant (>5%)
            if missing_percentage > 5:
                df_imputed[f"{col}_was_missing"] = df[col].isnull().astype(int)
                self.preprocessing_steps.append(f"Created missingness indicator for {col} ({missing_percentage:.1f}% missing)")

            if impute_strategy == 'median':
                # For numerical features
                if df[col].dtype in ['int64', 'float64']:
                    imputer = SimpleImputer(strategy='median')
                    df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                    imputation_info[col] = {
                        'strategy': 'median',
                        'imputer': imputer,
                        'fill_value': imputer.statistics_[0]
                    }
                else:
                    # Fallback to mode for non-numerical
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                    imputation_info[col] = {
                        'strategy': 'mode_fallback',
                        'imputer': imputer,
                        'fill_value': imputer.statistics_[0]
                    }

            elif impute_strategy == 'mode':
                # For categorical features
                imputer = SimpleImputer(strategy='most_frequent')
                df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                imputation_info[col] = {
                    'strategy': 'mode',
                    'imputer': imputer,
                    'fill_value': imputer.statistics_[0]
                }

            elif impute_strategy == 'constant':
                # Fill with a constant value
                if df[col].dtype in ['int64', 'float64']:
                    fill_value = 0
                else:
                    fill_value = 'Unknown'

                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                imputation_info[col] = {
                    'strategy': 'constant',
                    'imputer': imputer,
                    'fill_value': fill_value
                }

            elif impute_strategy == 'knn':
                # KNN imputation for complex relationships
                if df[col].dtype in ['int64', 'float64']:
                    # Select numerical columns for KNN
                    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numerical_cols) >= 2:  # Need at least 2 features for KNN
                        imputer = KNNImputer(n_neighbors=5, weights='distance')
                        df_numerical = df[numerical_cols]
                        imputed_data = imputer.fit_transform(df_numerical)
                        col_idx = numerical_cols.index(col)
                        df_imputed[col] = imputed_data[:, col_idx]
                        imputation_info[col] = {
                            'strategy': 'knn',
                            'imputer': imputer,
                            'feature_set': numerical_cols
                        }
                    else:
                        # Fallback to median if not enough numerical features
                        imputer = SimpleImputer(strategy='median')
                        df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                        imputation_info[col] = {
                            'strategy': 'median_fallback',
                            'imputer': imputer,
                            'fill_value': imputer.statistics_[0]
                        }
                else:
                    # Cannot use KNN for categorical, fallback to mode
                    imputer = SimpleImputer(strategy='most_frequent')
                    df_imputed[col] = imputer.fit_transform(df[[col]]).ravel()
                    imputation_info[col] = {
                        'strategy': 'mode_fallback',
                        'imputer': imputer,
                        'fill_value': imputer.statistics_[0]
                    }

            elif impute_strategy == 'drop':
                # Drop rows with missing values in this column
                initial_rows = len(df_imputed)
                df_imputed = df_imputed.dropna(subset=[col])
                dropped_rows = initial_rows - len(df_imputed)
                imputation_info[col] = {
                    'strategy': 'drop',
                    'dropped_rows': dropped_rows
                }
                self.preprocessing_steps.append(f"Dropped {dropped_rows} rows with missing {col}")
                continue

            self.preprocessing_steps.append(
                f"Imputed {missing_count} missing values in {col} using {impute_strategy}"
            )

        # Store imputation information
        self.fitted_transformers['imputers'] = imputation_info

        return df_imputed

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data.

        Args:
            df: DataFrame with original features

        Returns:
            DataFrame with original + engineered features

        Educational Notes:
        - Domain knowledge-driven feature creation
        - Mathematical transformations and ratios
        - Binning of continuous variables
        - Interaction features for churn prediction
        """
        df_engineered = df.copy()
        engineered_features = []

        # 1. Tenure-based features
        if 'tenure' in df.columns:
            # Tenure groups (customer lifecycle stages)
            df_engineered['tenure_group'] = pd.cut(
                df['tenure'],
                bins=[0, 12, 24, 48, float('inf')],
                labels=['New (0-12)', 'Growing (12-24)', 'Mature (24-48)', 'Loyal (48+)'],
                include_lowest=True
            )
            engineered_features.append('tenure_group')

            # Tenure squared (capture non-linear relationship)
            df_engineered['tenure_squared'] = df['tenure'] ** 2
            engineered_features.append('tenure_squared')

            # Log tenure (handle skewness)
            df_engineered['log_tenure'] = np.log1p(df['tenure'])  # log(1+x) to handle 0 values
            engineered_features.append('log_tenure')

        # 2. Monthly charges analysis
        if 'monthly_charges' in df.columns:
            # Monthly charges per year of tenure
            if 'tenure' in df.columns:
                df_engineered['monthly_charges_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
                engineered_features.append('monthly_charges_per_tenure')

            # Monthly charges bins
            df_engineered['monthly_charges_bins'] = pd.cut(
                df['monthly_charges'],
                bins=[0, 35, 55, 80, float('inf')],
                labels=['Low', 'Medium', 'High', 'Premium'],
                include_lowest=True
            )
            engineered_features.append('monthly_charges_bins')

        # 3. Total charges analysis
        if 'total_charges' in df.columns:
            # Convert to numeric if it's string
            if df['total_charges'].dtype == 'object':
                df_engineered['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

            # Total charges per tenure (average monthly spend)
            if 'tenure' in df.columns:
                df_engineered['avg_monthly_spend'] = df_engineered['total_charges'] / (df['tenure'] + 1)
                engineered_features.append('avg_monthly_spend')

            # Total charges bins
            df_engineered['total_charges_bins'] = pd.qcut(
                df_engineered['total_charges'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )
            engineered_features.append('total_charges_bins')

        # 4. Service complexity features
        service_columns = [col for col in df.columns if any(service in col.lower()
                          for service in ['internet', 'phone', 'security', 'backup', 'device', 'tech', 'streaming'])]

        if service_columns:
            # Count total services
            df_engineered['total_services'] = df[service_columns].apply(
                lambda row: sum(row == 'Yes') if row.dtype == 'object' else row.sum(), axis=1
            )
            engineered_features.append('total_services')

            # Service complexity ratio
            df_engineered['service_complexity'] = df_engineered['total_services'] / len(service_columns)
            engineered_features.append('service_complexity')

        # 5. Contract and payment features
        if 'contract' in df.columns and 'payment_method' in df.columns:
            # Contract-payment interaction
            df_engineered['contract_payment_interaction'] = (
                df['contract'].astype(str) + '_' + df['payment_method'].astype(str)
            )
            engineered_features.append('contract_payment_interaction')

        # 6. Customer value indicators
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            # Customer lifetime value proxy
            df_engineered['clv_proxy'] = df['monthly_charges'] * df['tenure']
            engineered_features.append('clv_proxy')

            # High value customer indicator
            high_value_threshold = df['monthly_charges'].quantile(0.8)
            long_tenure_threshold = df['tenure'].quantile(0.7)
            df_engineered['high_value_customer'] = (
                (df['monthly_charges'] >= high_value_threshold) &
                (df['tenure'] >= long_tenure_threshold)
            ).astype(int)
            engineered_features.append('high_value_customer')

        # 7. Age-related features (if senior citizen available)
        if 'senior_citizen' in df.columns:
            # Combine with other demographics
            if 'partner' in df.columns and 'dependents' in df.columns:
                df_engineered['family_status'] = (
                    df['senior_citizen'].astype(str) + '_' +
                    df['partner'].astype(str) + '_' +
                    df['dependents'].astype(str)
                )
                engineered_features.append('family_status')

        # 8. Churn risk indicators (business rules)
        risk_indicators = []

        # Short tenure + high charges = risk
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            df_engineered['short_tenure_high_charges'] = (
                (df['tenure'] < 12) & (df['monthly_charges'] > df['monthly_charges'].median())
            ).astype(int)
            risk_indicators.append('short_tenure_high_charges')

        # Month-to-month contract = higher risk
        if 'contract' in df.columns:
            df_engineered['month_to_month_risk'] = (
                df['contract'] == 'Month-to-month'
            ).astype(int)
            risk_indicators.append('month_to_month_risk')

        # Electronic check payment = higher risk
        if 'payment_method' in df.columns:
            df_engineered['electronic_check_risk'] = (
                df['payment_method'] == 'Electronic check'
            ).astype(int)
            risk_indicators.append('electronic_check_risk')

        # Combined risk score
        if risk_indicators:
            df_engineered['churn_risk_score'] = df_engineered[risk_indicators].sum(axis=1)
            engineered_features.append('churn_risk_score')

        # Log all engineered features
        if engineered_features:
            self.preprocessing_steps.append(f"Engineered {len(engineered_features)} new features: {engineered_features}")

        return df_engineered

    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str],
                                method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
        """
        Scale numerical features for ML algorithms.

        Args:
            df: DataFrame with numerical features
            numerical_cols: List of numerical column names
            method: Scaling method ('standard', 'robust', 'minmax')

        Returns:
            Tuple of (scaled_df, fitted_scaler)

        Educational Notes:
        - StandardScaler for normal distributions
        - RobustScaler for outlier-heavy data
        - MinMaxScaler for bounded ranges
        - Preserves original feature names
        """
        df_scaled = df.copy()

        # Select appropriate scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        # Only scale numerical columns that exist
        existing_numerical_cols = [col for col in numerical_cols if col in df.columns]

        if not existing_numerical_cols:
            warnings.warn("No numerical columns found for scaling")
            return df_scaled, None

        # Handle missing values before scaling
        numerical_data = df[existing_numerical_cols].fillna(0)  # Temporary fill for scaling

        # Fit and transform
        scaled_data = scaler.fit_transform(numerical_data)

        # Replace original columns with scaled versions
        for i, col in enumerate(existing_numerical_cols):
            df_scaled[col] = scaled_data[:, i]

        # Store scaling information
        self.fitted_transformers['scaler'] = {
            'scaler': scaler,
            'method': method,
            'columns': existing_numerical_cols,
            'feature_names_out': existing_numerical_cols
        }

        self.preprocessing_steps.append(
            f"Applied {method} scaling to {len(existing_numerical_cols)} numerical features"
        )

        return df_scaled, scaler

    def process_customer_profiles(self, profiles: List[CustomerProfile],
                                target_column: str = 'churn') -> ProcessedFeatures:
        """
        Complete feature processing pipeline for customer profiles.

        Args:
            profiles: List of CustomerProfile objects
            target_column: Name of the target variable column

        Returns:
            ProcessedFeatures object with processed feature matrix

        Educational Notes:
        - End-to-end feature processing pipeline
        - Converts domain objects to ML-ready format
        - Tracks all transformations for reproducibility
        """
        # Convert profiles to DataFrame
        df = pd.DataFrame([profile.to_dict() for profile in profiles])

        # Extract customer IDs and target variable
        customer_ids = df['customer_id'].tolist()

        # Remove target variable and customer_id from features
        feature_df = df.drop(columns=['customer_id', target_column], errors='ignore')

        # Identify feature types
        categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

        self.preprocessing_steps.append(f"Starting feature processing with {len(feature_df.columns)} features")
        self.preprocessing_steps.append(f"Identified {len(categorical_cols)} categorical and {len(numerical_cols)} numerical features")

        # 1. Handle missing values
        missing_strategy = {}
        for col in categorical_cols:
            missing_strategy[col] = 'mode'
        for col in numerical_cols:
            missing_strategy[col] = 'median'

        feature_df = self.handle_missing_values(feature_df, missing_strategy)

        # 2. Engineer features
        feature_df = self.engineer_features(feature_df)

        # 3. Encode categorical features
        # Update categorical columns after feature engineering
        categorical_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            feature_df = self.encode_categorical_features(feature_df, categorical_cols)

        # 4. Scale numerical features
        # Update numerical columns after encoding
        numerical_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            feature_df, scaler = self.scale_numerical_features(feature_df, numerical_cols)

        # Create feature metadata
        feature_metadata = {}
        for col in feature_df.columns:
            if col in self.fitted_transformers.get('categorical_encoders', {}):
                encoding_info = self.fitted_transformers['categorical_encoders'][col]
                feature_metadata[col] = FeatureMetadata(
                    name=col,
                    feature_type=FeatureType.CATEGORICAL,
                    encoding_method=encoding_info['method'],
                    transformation_steps=['categorical_encoding']
                )
            elif col.endswith('_was_missing'):
                feature_metadata[col] = FeatureMetadata(
                    name=col,
                    feature_type=FeatureType.BINARY,
                    original_column=col.replace('_was_missing', ''),
                    transformation_steps=['missingness_indicator']
                )
            elif any(suffix in col for suffix in ['_bins', '_group']):
                feature_metadata[col] = FeatureMetadata(
                    name=col,
                    feature_type=FeatureType.ORDINAL,
                    transformation_steps=['feature_engineering', 'binning']
                )
            elif any(engineered in col for engineered in ['squared', 'log_', 'per_', 'avg_', 'total_', 'clv_', 'risk']):
                feature_metadata[col] = FeatureMetadata(
                    name=col,
                    feature_type=FeatureType.ENGINEERED,
                    transformation_steps=['feature_engineering']
                )
            else:
                feature_metadata[col] = FeatureMetadata(
                    name=col,
                    feature_type=FeatureType.NUMERICAL,
                    scaling_method=ScalingMethod.STANDARD if 'scaler' in self.fitted_transformers else ScalingMethod.NONE,
                    transformation_steps=['scaling'] if 'scaler' in self.fitted_transformers else []
                )

        # Create ProcessedFeatures object
        processed_features = ProcessedFeatures(
            feature_matrix=feature_df,
            feature_names=feature_df.columns.tolist(),
            customer_ids=customer_ids,
            feature_metadata=feature_metadata,
            preprocessing_steps=self.preprocessing_steps.copy(),
            scaling_parameters=self.fitted_transformers.get('scaler', {}),
            encoding_mappings=self.fitted_transformers.get('categorical_encoders', {}),
            processing_timestamp=datetime.now()
        )

        self.preprocessing_steps.append(f"Feature processing completed: {processed_features}")

        return processed_features

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted transformers.

        Args:
            df: New data to transform

        Returns:
            Transformed DataFrame

        Educational Notes:
        - Uses previously fitted transformers
        - Ensures consistent preprocessing for new data
        - Critical for production deployment
        """
        if not self.fitted_transformers:
            raise ValueError("No fitted transformers found. Must call process_customer_profiles first.")

        df_transformed = df.copy()

        # Apply imputation
        if 'imputers' in self.fitted_transformers:
            for col, imputer_info in self.fitted_transformers['imputers'].items():
                if col in df_transformed.columns and 'imputer' in imputer_info:
                    imputer = imputer_info['imputer']
                    df_transformed[col] = imputer.transform(df_transformed[[col]]).ravel()

        # Apply categorical encoding
        if 'categorical_encoders' in self.fitted_transformers:
            for col, encoder_info in self.fitted_transformers['categorical_encoders'].items():
                if col not in df_transformed.columns:
                    continue

                method = encoder_info['method']
                if method == 'label':
                    encoder = encoder_info['encoder']
                    # Handle unknown categories
                    df_transformed[col] = df_transformed[col].fillna('Unknown')
                    unknown_mask = ~df_transformed[col].isin(encoder.classes_)
                    df_transformed.loc[unknown_mask, col] = encoder.classes_[0]  # Map to first class
                    df_transformed[col] = encoder.transform(df_transformed[col])

                elif method == 'onehot':
                    encoder = encoder_info['encoder']
                    encoded_data = encoder.transform(df_transformed[[col]].fillna('Unknown'))
                    feature_names = encoder_info['feature_names']
                    encoded_df = pd.DataFrame(encoded_data, columns=feature_names, index=df_transformed.index)

                    df_transformed = df_transformed.drop(columns=[col])
                    df_transformed = pd.concat([df_transformed, encoded_df], axis=1)

                elif method == 'ordinal_frequency':
                    mapping = encoder_info['mapping']
                    df_transformed[col] = df_transformed[col].fillna('Unknown').map(mapping)
                    # Handle truly unknown categories
                    df_transformed[col] = df_transformed[col].fillna(mapping['Unknown'])

        # Apply scaling
        if 'scaler' in self.fitted_transformers:
            scaler_info = self.fitted_transformers['scaler']
            scaler = scaler_info['scaler']
            columns = scaler_info['columns']

            existing_cols = [col for col in columns if col in df_transformed.columns]
            if existing_cols:
                scaled_data = scaler.transform(df_transformed[existing_cols].fillna(0))
                for i, col in enumerate(existing_cols):
                    df_transformed[col] = scaled_data[:, i]

        return df_transformed

    def get_feature_importance_analysis(self, feature_names: List[str],
                                      importances: np.ndarray) -> Dict[str, Any]:
        """
        Analyze feature importance in context of preprocessing steps.

        Args:
            feature_names: List of feature names
            importances: Feature importance scores

        Returns:
            Analysis of feature importance with preprocessing context
        """
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Categorize features by type
        categories = {
            'original': [],
            'engineered': [],
            'encoded': [],
            'indicators': []
        }

        for feature in feature_names:
            if any(engineered in feature for engineered in ['squared', 'log_', 'per_', 'avg_', 'total_', 'clv_', 'risk']):
                categories['engineered'].append(feature)
            elif '_was_missing' in feature:
                categories['indicators'].append(feature)
            elif any(encoded in feature for encoded in ['_Yes', '_No', '_bins', '_group']):
                categories['encoded'].append(feature)
            else:
                categories['original'].append(feature)

        analysis = {
            'top_features': importance_df.head(10).to_dict('records'),
            'feature_categories': categories,
            'category_importance': {},
            'preprocessing_impact': {}
        }

        # Calculate average importance by category
        for category, features in categories.items():
            if features:
                category_importances = importance_df[importance_df['feature'].isin(features)]['importance']
                analysis['category_importance'][category] = {
                    'mean_importance': category_importances.mean(),
                    'total_importance': category_importances.sum(),
                    'feature_count': len(features)
                }

        return analysis
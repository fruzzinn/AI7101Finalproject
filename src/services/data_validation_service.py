"""
Data Validation Service Implementation
Implements DataValidationContract for comprehensive data quality assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from scipy import stats


class DataValidationService:
    """Service for validating datasets and ensuring data quality for ML workflows"""

    def __init__(self):
        """Initialize the data validation service"""
        self.outlier_threshold = 3.0  # Z-score threshold for outlier detection

    def validate_datasets(self,
                         train_data: pd.DataFrame,
                         test_data: pd.DataFrame,
                         variables_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate training and test datasets for consistency and completeness

        Args:
            train_data: Training dataset
            test_data: Test dataset
            variables_data: Variable definitions

        Returns:
            Dictionary with validation results
        """
        if train_data is None or test_data is None or variables_data is None:
            raise TypeError("Input datasets cannot be None")

        if train_data.empty or test_data.empty:
            raise ValueError("Input datasets cannot be empty")

        results = {}

        # Check for target column in training data
        results['has_target_column'] = 'churn' in train_data.columns

        # Check feature consistency between train and test
        train_features = set(train_data.columns) - {'churn'}
        test_features = set(test_data.columns)
        results['consistent_features'] = train_features == test_features

        # Check for empty columns
        train_empty_cols = train_data.isnull().all()
        test_empty_cols = test_data.isnull().all()
        results['no_empty_columns'] = not (train_empty_cols.any() or test_empty_cols.any())

        # Validate data types
        results['valid_data_types'] = self._validate_data_types(train_data, variables_data)

        return results

    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess overall data quality metrics

        Args:
            data: DataFrame to assess

        Returns:
            Dictionary with quality metrics
        """
        metrics = {}

        # Missing value rate
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        metrics['missing_rate'] = missing_cells / total_cells if total_cells > 0 else 0

        # Duplicate rate
        total_rows = len(data)
        if total_rows > 0:
            # Check for duplicate rows across all columns
            unique_rows = len(data.drop_duplicates())
            metrics['duplicate_rate'] = (total_rows - unique_rows) / total_rows
        else:
            metrics['duplicate_rate'] = 0

        # Outlier rate (for numerical columns)
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        total_numerical_values = 0
        outlier_count = 0

        for col in numerical_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                z_scores = np.abs(stats.zscore(col_data))
                outliers = z_scores > self.outlier_threshold
                outlier_count += outliers.sum()
                total_numerical_values += len(col_data)

        metrics['outlier_rate'] = outlier_count / total_numerical_values if total_numerical_values > 0 else 0

        return metrics

    def validate_target_variable(self, target: pd.Series) -> bool:
        """
        Validate target variable for binary classification

        Args:
            target: Target variable series

        Returns:
            True if valid, False otherwise
        """
        # Check for missing values
        if target.isnull().any():
            return False

        # Check for binary values only
        unique_values = target.unique()
        if not set(unique_values).issubset({0, 1}):
            return False

        # Check for both classes present
        if len(unique_values) < 2:
            return False

        return True

    def _validate_data_types(self, data: pd.DataFrame, variables_data: pd.DataFrame) -> bool:
        """
        Validate that data types are appropriate for ML

        Args:
            data: Data to validate
            variables_data: Variable type definitions

        Returns:
            True if data types are valid
        """
        try:
            # Create mapping of expected types
            var_types = dict(zip(variables_data['variable'], variables_data['type']))

            for col in data.columns:
                if col in var_types:
                    expected_type = var_types[col]
                    actual_dtype = data[col].dtype

                    # Check numeric types
                    if expected_type == 'numeric':
                        if not pd.api.types.is_numeric_dtype(actual_dtype):
                            return False

                    # Check categorical types
                    elif expected_type == 'categorical':
                        # Categorical can be string or object type
                        if not (pd.api.types.is_object_dtype(actual_dtype) or
                               pd.api.types.is_categorical_dtype(actual_dtype)):
                            # Allow numeric categories if they're small integers
                            if pd.api.types.is_numeric_dtype(actual_dtype):
                                unique_count = data[col].nunique()
                                if unique_count > 20:  # Too many unique values for categorical
                                    return False
                            else:
                                return False

                    # Check binary types
                    elif expected_type == 'binary':
                        unique_values = data[col].dropna().unique()
                        if not set(unique_values).issubset({0, 1}):
                            return False

            return True

        except Exception:
            return False
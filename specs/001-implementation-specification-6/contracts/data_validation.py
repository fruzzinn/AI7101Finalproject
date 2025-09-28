"""
Contract: Data Validation Interface
Purpose: Validate loaded datasets meet quality requirements
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd

class DataValidationContract:
    """Contract for data validation operations"""

    def validate_datasets(self,
                         train_df: pd.DataFrame,
                         test_df: pd.DataFrame,
                         variables_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate all required datasets are properly loaded and formatted

        Args:
            train_df: Training dataset with features and target
            test_df: Test dataset for final predictions
            variables_df: Data dictionary with feature descriptions

        Returns:
            Dict with validation results for each check

        Contract Requirements:
            - train_df must contain 'churn' target column
            - All datasets must have consistent feature columns
            - No completely empty columns allowed
            - Data types must be appropriate for ML processing
        """
        raise NotImplementedError("Must implement dataset validation")

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess data quality metrics for a dataset

        Args:
            df: Dataset to analyze

        Returns:
            Dict with quality metrics (missing_rate, duplicate_rate, etc.)

        Contract Requirements:
            - Missing value percentage per column
            - Duplicate row detection
            - Data type consistency checks
            - Outlier detection for numerical columns
        """
        raise NotImplementedError("Must implement data quality assessment")

    def validate_target_variable(self, target_series: pd.Series) -> bool:
        """
        Validate target variable meets churn prediction requirements

        Args:
            target_series: Target variable column

        Returns:
            True if valid, False otherwise

        Contract Requirements:
            - Binary values (0, 1) only
            - No missing values in target
            - Reasonable class distribution (not 100% one class)
        """
        raise NotImplementedError("Must implement target validation")
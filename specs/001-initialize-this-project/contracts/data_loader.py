"""
Contract: Data Loading and Validation Interface
Educational Focus: Demonstrate proper data ingestion patterns
"""

from typing import Tuple, Dict, Any
import pandas as pd
from abc import ABC, abstractmethod


class DataLoaderContract(ABC):
    """Contract for loading and validating telecommunications churn data"""

    @abstractmethod
    def load_raw_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw customer and churn data from file

        Args:
            file_path: Path to data file (CSV format expected)

        Returns:
            Tuple of (customer_features_df, churn_labels_df)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid
        """
        pass

    @abstractmethod
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return quality report

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results:
            - missing_percentages: Dict of column: missing_percentage
            - duplicate_count: Number of duplicate rows
            - invalid_values: List of columns with invalid values
            - quality_score: Overall quality score (0-1)
        """
        pass

    @abstractmethod
    def split_features_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y)

        Args:
            df: Complete dataset
            target_col: Name of target column

        Returns:
            Tuple of (features_df, target_series)

        Raises:
            KeyError: If target column doesn't exist
        """
        pass


class DataValidatorContract(ABC):
    """Contract for data validation and quality checks"""

    @abstractmethod
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Check if missing values are within acceptable threshold

        Args:
            df: DataFrame to check
            threshold: Maximum acceptable missing percentage (default 5%)

        Returns:
            True if missing values are acceptable, False otherwise
        """
        pass

    @abstractmethod
    def validate_categorical_values(self, df: pd.DataFrame, column_specs: Dict[str, list]) -> Dict[str, list]:
        """
        Validate categorical columns have expected values

        Args:
            df: DataFrame to validate
            column_specs: Dict mapping column names to list of expected values

        Returns:
            Dict of column: list of invalid values found
        """
        pass

    @abstractmethod
    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Check for business logic inconsistencies in data

        Args:
            df: DataFrame to check

        Returns:
            Dict of consistency_check: error_message
            Empty dict if all checks pass
        """
        pass
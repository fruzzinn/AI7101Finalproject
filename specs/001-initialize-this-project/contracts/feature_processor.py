"""
Contract: Feature Engineering and Preprocessing Interface
Educational Focus: Demonstrate proper ML preprocessing patterns
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureProcessorContract(ABC):
    """Contract for feature engineering and preprocessing operations"""

    @abstractmethod
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding strategy

        Args:
            df: DataFrame with categorical features
            categorical_cols: List of categorical column names

        Returns:
            DataFrame with encoded categorical features

        Educational Notes:
            - One-hot encoding for nominal categories
            - Ordinal encoding for ordered categories
            - Handle unknown categories in test data
        """
        pass

    @abstractmethod
    def handle_missing_values(self, df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
        """
        Handle missing values using specified strategies

        Args:
            df: DataFrame with missing values
            strategy: Dict mapping column names to imputation strategies
                     ('median', 'mode', 'constant', 'drop')

        Returns:
            DataFrame with missing values handled

        Educational Notes:
            - Different strategies for numerical vs categorical
            - Create missingness indicators when appropriate
            - Document impact of imputation choices
        """
        pass

    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing data

        Args:
            df: DataFrame with original features

        Returns:
            DataFrame with original + engineered features

        Educational Notes:
            - Domain knowledge-driven feature creation
            - Mathematical transformations and ratios
            - Binning of continuous variables
        """
        pass

    @abstractmethod
    def scale_numerical_features(self, df: pd.DataFrame, numerical_cols: List[str]) -> Tuple[pd.DataFrame, Any]:
        """
        Scale numerical features for ML algorithms

        Args:
            df: DataFrame with numerical features
            numerical_cols: List of numerical column names

        Returns:
            Tuple of (scaled_df, fitted_scaler)

        Educational Notes:
            - StandardScaler for normal distributions
            - RobustScaler for outlier-heavy data
            - MinMaxScaler for bounded ranges
        """
        pass


class FeatureValidatorContract(ABC):
    """Contract for validating processed features"""

    @abstractmethod
    def validate_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Validate that processed features have expected distributions

        Args:
            df: DataFrame with processed features

        Returns:
            Dict mapping column names to distribution statistics
            (mean, std, min, max, skewness, kurtosis)
        """
        pass

    @abstractmethod
    def check_feature_correlations(self, df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated features that may cause multicollinearity

        Args:
            df: DataFrame with features
            threshold: Correlation threshold for flagging

        Returns:
            List of tuples (feature1, feature2, correlation_value)
        """
        pass

    @abstractmethod
    def validate_encoding_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, str]:
        """
        Ensure consistent encoding between training and test sets

        Args:
            train_df: Training set features
            test_df: Test set features

        Returns:
            Dict of potential issues found
        """
        pass


class FeatureSelectorContract(ABC):
    """Contract for feature selection operations"""

    @abstractmethod
    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
        """
        Select top k features using univariate statistical tests

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select

        Returns:
            List of selected feature names
        """
        pass

    @abstractmethod
    def select_features_importance(self, X: pd.DataFrame, y: pd.Series,
                                  model: BaseEstimator, threshold: float) -> List[str]:
        """
        Select features based on model-derived importance scores

        Args:
            X: Feature matrix
            y: Target vector
            model: Fitted model with feature_importances_ attribute
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names
        """
        pass

    @abstractmethod
    def analyze_feature_importance(self, feature_names: List[str],
                                  importances: np.ndarray) -> Dict[str, float]:
        """
        Analyze and rank feature importance scores

        Args:
            feature_names: List of feature names
            importances: Array of importance scores

        Returns:
            Dict mapping feature names to importance scores (sorted)
        """
        pass
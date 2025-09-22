"""
Contract: Model Training and Evaluation Interface
Educational Focus: Demonstrate proper ML model development patterns
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator


class ModelTrainerContract(ABC):
    """Contract for training and evaluating ML models"""

    @abstractmethod
    def setup_cross_validation(self, X: pd.DataFrame, y: pd.Series,
                              cv_folds: int = 5, stratify: bool = True) -> BaseCrossValidator:
        """
        Set up cross-validation strategy appropriate for the data

        Args:
            X: Feature matrix
            y: Target vector
            cv_folds: Number of CV folds
            stratify: Whether to use stratified sampling

        Returns:
            Configured cross-validation object

        Educational Notes:
            - Stratified CV for imbalanced binary classification
            - Handles class distribution preservation
        """
        pass

    @abstractmethod
    def train_model(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                   hyperparams: Optional[Dict[str, Any]] = None) -> BaseEstimator:
        """
        Train a single model with given hyperparameters

        Args:
            model: Scikit-learn model instance
            X: Training features
            y: Training target
            hyperparams: Model hyperparameters to set

        Returns:
            Fitted model

        Educational Notes:
            - Proper parameter setting before training
            - Random state management for reproducibility
        """
        pass

    @abstractmethod
    def evaluate_model_cv(self, model: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                         cv: BaseCrossValidator) -> Dict[str, float]:
        """
        Evaluate model using cross-validation

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Cross-validation strategy

        Returns:
            Dict of metric_name: mean_score
            Metrics: accuracy, precision, recall, f1, roc_auc

        Educational Notes:
            - Multiple metrics for comprehensive evaluation
            - Statistical significance of results
        """
        pass

    @abstractmethod
    def hyperparameter_tuning(self, model: BaseEstimator, param_grid: Dict[str, List],
                             X: pd.DataFrame, y: pd.Series,
                             cv: BaseCrossValidator, scoring: str) -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Perform hyperparameter tuning using grid/random search

        Args:
            model: Model to tune
            param_grid: Hyperparameter search space
            X: Feature matrix
            y: Target vector
            cv: Cross-validation strategy
            scoring: Metric to optimize

        Returns:
            Tuple of (best_model, best_params)

        Educational Notes:
            - Grid search vs random search trade-offs
            - Cross-validation within hyperparameter search
        """
        pass


class ModelEvaluatorContract(ABC):
    """Contract for comprehensive model evaluation"""

    @abstractmethod
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)

        Returns:
            Dict of metric_name: value
            Includes: accuracy, precision, recall, f1, specificity, roc_auc

        Educational Notes:
            - Different metrics for different business contexts
            - Handling class imbalance in evaluation
        """
        pass

    @abstractmethod
    def generate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Generate confusion matrix and extract key values

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Tuple of (confusion_matrix, {TP, TN, FP, FN})

        Educational Notes:
            - Understanding Type I and Type II errors
            - Business cost implications of each error type
        """
        pass

    @abstractmethod
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      model_name: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Generate ROC curve data and calculate AUC

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            model_name: Name for plot title

        Returns:
            Tuple of (fpr, tpr, auc_score)

        Educational Notes:
            - ROC curve interpretation
            - AUC as threshold-independent metric
        """
        pass

    @abstractmethod
    def analyze_feature_importance(self, model: BaseEstimator,
                                  feature_names: List[str]) -> Dict[str, float]:
        """
        Extract and analyze feature importance from trained model

        Args:
            model: Trained model with feature importance
            feature_names: Names of features

        Returns:
            Dict of feature_name: importance_score (sorted)

        Educational Notes:
            - Different importance calculation methods
            - Interpretation challenges and limitations
        """
        pass


class ModelComparisonContract(ABC):
    """Contract for comparing multiple models"""

    @abstractmethod
    def compare_models(self, models: Dict[str, BaseEstimator], X: pd.DataFrame, y: pd.Series,
                      cv: BaseCrossValidator) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation

        Args:
            models: Dict of model_name: model_instance
            X: Feature matrix
            y: Target vector
            cv: Cross-validation strategy

        Returns:
            DataFrame with model comparison results

        Educational Notes:
            - Statistical testing for model comparison
            - Multiple comparison corrections
        """
        pass

    @abstractmethod
    def rank_models(self, comparison_results: pd.DataFrame,
                   primary_metric: str = 'f1') -> pd.DataFrame:
        """
        Rank models based on performance metrics

        Args:
            comparison_results: Results from compare_models
            primary_metric: Metric to use for primary ranking

        Returns:
            DataFrame with ranked models

        Educational Notes:
            - Multi-criteria decision making
            - Business metric priorities
        """
        pass

    @abstractmethod
    def select_best_model(self, comparison_results: pd.DataFrame,
                         selection_criteria: Dict[str, float]) -> str:
        """
        Select best model based on multiple criteria

        Args:
            comparison_results: Results from compare_models
            selection_criteria: Dict of metric: minimum_threshold

        Returns:
            Name of selected best model

        Educational Notes:
            - Balancing multiple objectives
            - Business constraints in model selection
        """
        pass
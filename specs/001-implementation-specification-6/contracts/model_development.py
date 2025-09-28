"""
Contract: Model Development Interface
Purpose: Train and evaluate machine learning models for churn prediction
"""

from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score

class ModelDevelopmentContract:
    """Contract for model development and evaluation operations"""

    def implement_cross_validation(self,
                                  X: pd.DataFrame,
                                  y: pd.Series,
                                  model: Any) -> Dict[str, float]:
        """
        Implement stratified cross-validation with SMOTE for imbalanced data

        Args:
            X: Feature matrix
            y: Target variable
            model: ML model to evaluate

        Returns:
            Dict with cross-validation metrics

        Contract Requirements:
            - Use StratifiedKFold with k=5
            - Integrate SMOTE within CV pipeline to prevent data leakage
            - Calculate F1-score, precision, recall, AUC-ROC
            - Report mean and standard deviation for each metric
        """
        raise NotImplementedError("Must implement cross-validation strategy")

    def train_models(self,
                    X_train: pd.DataFrame,
                    y_train: pd.Series) -> Dict[str, Any]:
        """
        Train multiple ML algorithms suited for churn prediction

        Args:
            X_train: Training feature matrix
            y_train: Training target variable

        Returns:
            Dict of trained model objects

        Contract Requirements:
            - Random Forest (baseline model)
            - Gradient Boosting (XGBoost or LightGBM)
            - Logistic Regression (interpretable model)
            - Support Vector Machine (for comparison)
            - All models must handle class imbalance appropriately
        """
        raise NotImplementedError("Must implement model training")

    def hyperparameter_tuning(self,
                             X: pd.DataFrame,
                             y: pd.Series,
                             model_type: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform systematic hyperparameter optimization

        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to tune

        Returns:
            Tuple of (best_model, best_parameters)

        Contract Requirements:
            - Use nested cross-validation to prevent overfitting
            - Grid search or random search for hyperparameter exploration
            - Optimize for F1-score (primary metric for imbalanced data)
            - Document parameter search space and reasoning
        """
        raise NotImplementedError("Must implement hyperparameter tuning")

    def evaluate_model_performance(self,
                                  model: Any,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensive model evaluation aligned with business objectives

        Args:
            model: Trained model to evaluate
            X_test: Test feature matrix
            y_test: Test target variable

        Returns:
            Dict with comprehensive performance metrics

        Contract Requirements:
            - F1-score (primary metric)
            - Precision and Recall with business interpretation
            - AUC-ROC and confusion matrix
            - Feature importance analysis
            - Prediction probability distribution analysis
        """
        raise NotImplementedError("Must implement model evaluation")

    def analyze_feature_importance(self,
                                  model: Any,
                                  feature_names: List[str]) -> Dict[str, float]:
        """
        Extract and rank feature importance for model explainability

        Args:
            model: Trained model with feature importance
            feature_names: Names of input features

        Returns:
            Dict mapping feature names to importance scores

        Contract Requirements:
            - Use model-specific importance (tree-based) or SHAP values
            - Normalize importance scores to sum to 1.0
            - Provide business interpretation for top 10 features
            - Identify potential data leakage or unexpected patterns
        """
        raise NotImplementedError("Must implement feature importance analysis")
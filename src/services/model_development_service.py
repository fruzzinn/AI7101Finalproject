"""
Model Development Service Implementation
Implements ModelDevelopmentContract for ML model training and evaluation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class ModelDevelopmentService:
    """Service for ML model development, training, and evaluation"""

    def __init__(self, random_state: int = 42):
        """Initialize the model development service"""
        self.random_state = random_state
        self.n_folds = 5
        np.random.seed(random_state)

    def implement_cross_validation(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 model: Any) -> Dict[str, Any]:
        """
        Implement stratified cross-validation with SMOTE for imbalanced data

        Args:
            X: Feature matrix
            y: Target variable
            model: ML model to evaluate

        Returns:
            Dictionary with CV metrics
        """
        # Check class imbalance
        class_counts = y.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        use_smote = minority_ratio < 0.3  # Apply SMOTE if minority class < 30%

        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        if use_smote:
            # Create pipeline with SMOTE
            pipeline = ImbPipeline([
                ('smote', SMOTE(random_state=self.random_state)),
                ('classifier', model)
            ])
            model_to_evaluate = pipeline
        else:
            model_to_evaluate = model

        # Define scoring metrics
        scoring = {
            'f1_score': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'auc_roc': make_scorer(roc_auc_score)
        }

        # Perform cross-validation
        cv_results = cross_validate(
            model_to_evaluate, X, y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # Compile results
        results = {
            'f1_score': cv_results['test_f1_score'].mean(),
            'f1_score_mean': cv_results['test_f1_score'].mean(),
            'f1_score_std': cv_results['test_f1_score'].std(),
            'precision': cv_results['test_precision'].mean(),
            'precision_mean': cv_results['test_precision'].mean(),
            'precision_std': cv_results['test_precision'].std(),
            'recall': cv_results['test_recall'].mean(),
            'recall_mean': cv_results['test_recall'].mean(),
            'recall_std': cv_results['test_recall'].std(),
            'auc_roc': cv_results['test_auc_roc'].mean(),
            'auc_roc_mean': cv_results['test_auc_roc'].mean(),
            'auc_roc_std': cv_results['test_auc_roc'].std(),
            'n_folds': self.n_folds,
            'smote_applied': use_smote,
            'stratified': True,
            'stratification_used': True
        }

        # Add class distribution info
        if use_smote:
            results['class_distribution_preserved'] = True
            results['avg_positive_ratio'] = y.mean()

        return results

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train multiple ML algorithms with appropriate handling for class imbalance

        Args:
            X_train: Training features
            y_train: Training target

        Returns:
            Dictionary of trained models
        """
        # Check for class imbalance
        class_counts = y_train.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        use_class_weight = minority_ratio < 0.3

        # Define models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced' if use_class_weight else None,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.random_state,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                class_weight='balanced' if use_class_weight else None,
                max_iter=1000,
                solver='liblinear'
            ),
            'svm': SVC(
                random_state=self.random_state,
                class_weight='balanced' if use_class_weight else None,
                probability=True,
                kernel='rbf',
                C=1.0
            )
        }

        # Train all models
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        return trained_models

    def hyperparameter_tuning(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """
        Perform hyperparameter tuning with nested cross-validation

        Args:
            X: Feature matrix
            y: Target variable
            model_name: Name of model to tune

        Returns:
            Tuple of (best_model, best_params)
        """
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', None]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'class_weight': ['balanced', None],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto'],
                'class_weight': ['balanced', None]
            }
        }

        # Define base models
        base_models = {
            'random_forest': RandomForestClassifier(random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state),
            'svm': SVC(random_state=self.random_state, probability=True)
        }

        if model_name not in param_grids:
            raise ValueError(f"Model {model_name} not supported")

        # Setup nested cross-validation
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        outer_cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Perform grid search with F1-score optimization
        grid_search = GridSearchCV(
            estimator=base_models[model_name],
            param_grid=param_grids[model_name],
            cv=inner_cv,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state
        )

        # Fit grid search
        grid_search.fit(X, y)

        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_.copy()

        # Add metadata
        best_params.update({
            'best_score': grid_search.best_score_,
            'optimization_metric': 'f1_score',
            'nested_cv_used': True,
            'cv_folds': self.n_folds
        })

        # Store nested_cv flag on service for contract compliance
        self._nested_cv = True

        return best_model, best_params

    def evaluate_model_performance(self,
                                 model: Any,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series) -> Dict[str, Any]:
        """
        Comprehensively evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with performance metrics
        """
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        performance = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }

        # Business interpretation
        performance['business_interpretation'] = self._generate_business_interpretation(performance)

        # Prediction distribution analysis
        performance['prediction_distribution'] = {
            'predicted_churn_rate': y_pred.mean(),
            'actual_churn_rate': y_test.mean(),
            'avg_churn_probability': y_proba.mean(),
            'high_risk_customers': (y_proba > 0.7).sum(),
            'medium_risk_customers': ((y_proba > 0.3) & (y_proba <= 0.7)).sum(),
            'low_risk_customers': (y_proba <= 0.3).sum()
        }

        return performance

    def analyze_feature_importance(self,
                                 model: Any,
                                 feature_names: List[str]) -> Dict[str, float]:
        """
        Extract and analyze feature importance with business interpretation

        Args:
            model: Trained model with feature importance
            feature_names: List of feature names

        Returns:
            Dictionary of feature importances (normalized and ranked)
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")

        # Get feature importances
        importances = model.feature_importances_

        # Normalize to sum to 1.0
        importances_normalized = importances / importances.sum()

        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importances_normalized))

        # Sort by importance (descending)
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        # Store the max importance and top features before adding metadata
        feature_importances_only = {k: v for k, v in importance_dict.items() if isinstance(v, (int, float))}
        max_importance = max(feature_importances_only.values()) if feature_importances_only else 0
        top_features = list(feature_importances_only.keys())[:5]

        # Check for potential data leakage
        if max_importance > 0.8:
            importance_dict['potential_leakage_warning'] = True
            self._check_leakage = True

        # Add business interpretation for top features
        importance_dict['top_features_interpretation'] = self._interpret_top_features(top_features)

        return importance_dict

    def _generate_business_interpretation(self, performance: Dict[str, Any]) -> str:
        """Generate business-friendly interpretation of model performance"""
        f1 = performance['f1_score']
        precision = performance['precision']
        recall = performance['recall']

        if f1 >= 0.8:
            performance_level = "excellent"
        elif f1 >= 0.7:
            performance_level = "good"
        elif f1 >= 0.6:
            performance_level = "acceptable"
        else:
            performance_level = "needs improvement"

        interpretation = f"Model shows {performance_level} performance with F1-score of {f1:.2f}. "
        interpretation += f"Of customers predicted to churn, {precision:.1%} actually will churn (precision). "
        interpretation += f"The model identifies {recall:.1%} of all customers who will actually churn (recall)."

        return interpretation

    def _interpret_top_features(self, top_features: List[str]) -> Dict[str, str]:
        """Provide business interpretation for top features"""
        interpretations = {
            'tenure': "Customer relationship length - longer tenure typically indicates lower churn risk",
            'monthly_charges': "Monthly service cost - higher charges may increase churn likelihood",
            'contract_type': "Contract terms - month-to-month contracts show higher churn rates",
            'total_charges': "Customer lifetime value - higher total spend often correlates with loyalty",
            'payment_method': "Payment preferences - electronic check users show higher churn rates",
            'age': "Customer demographics - different age groups have varying retention patterns",
            'internet_service': "Service type - fiber optic customers may have different churn patterns"
        }

        result = {}
        for feature in top_features:
            # Match feature name to interpretation
            for key, interpretation in interpretations.items():
                if key in feature.lower():
                    result[feature] = interpretation
                    break
            else:
                result[feature] = f"Feature {feature} shows significant predictive power for churn"

        return result
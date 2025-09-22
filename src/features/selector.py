"""
FeatureSelector implementation for telecommunications churn prediction.

Educational Focus: Demonstrates ML feature selection strategies and techniques.
This module implements various feature selection methods to identify the most
informative features for churn prediction models.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, f_classif, chi2, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
from datetime import datetime

from ..models.processed_features import ProcessedFeatures, FeatureType


class FeatureSelector:
    """
    Feature selection operations for churn prediction.

    Educational Notes:
    - Implements multiple feature selection strategies
    - Combines statistical and model-based approaches
    - Provides interpretable feature importance analysis
    - Supports both supervised and unsupervised methods
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureSelector.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.selection_history: List[Dict[str, Any]] = []
        self.fitted_selectors: Dict[str, Any] = {}

        # Set random seeds
        np.random.seed(self.random_state)

    def select_features_univariate(self, X: pd.DataFrame, y: pd.Series, k: int,
                                 score_func: str = 'auto') -> List[str]:
        """
        Select top k features using univariate statistical tests.

        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            score_func: Scoring function ('auto', 'f_classif', 'chi2', 'mutual_info')

        Returns:
            List of selected feature names

        Educational Notes:
        - Fast, model-agnostic feature selection
        - f_classif for continuous features and binary classification
        - chi2 for categorical features (requires non-negative values)
        - mutual_info_classif captures non-linear relationships
        """
        if k >= len(X.columns):
            warnings.warn(f"k ({k}) >= number of features ({len(X.columns)}). Returning all features.")
            return list(X.columns)

        # Automatically select appropriate scoring function
        if score_func == 'auto':
            # Check if we have negative values (chi2 requires non-negative)
            has_negative = (X < 0).any().any()
            if has_negative:
                score_func = 'f_classif'
            else:
                # Use chi2 for categorical-like features, f_classif for continuous
                if X.dtypes.apply(lambda x: x == 'object' or x == 'category').any():
                    score_func = 'chi2'
                else:
                    score_func = 'f_classif'

        # Map string to function
        score_functions = {
            'f_classif': f_classif,
            'chi2': chi2,
            'mutual_info': mutual_info_classif
        }

        if score_func not in score_functions:
            raise ValueError(f"Unknown score function: {score_func}")

        scoring_func = score_functions[score_func]

        # Handle missing values
        X_clean = X.fillna(0)  # Simple imputation for feature selection

        # Ensure non-negative values for chi2
        if score_func == 'chi2' and (X_clean < 0).any().any():
            # Shift to make all values non-negative
            for col in X_clean.columns:
                if (X_clean[col] < 0).any():
                    X_clean[col] = X_clean[col] - X_clean[col].min()

        try:
            # Perform feature selection
            if score_func == 'mutual_info':
                selector = SelectKBest(score_func=scoring_func, k=k)
                # Mutual info requires additional parameters
                selector.set_params(score_func__random_state=self.random_state)
            else:
                selector = SelectKBest(score_func=scoring_func, k=k)

            X_selected = selector.fit_transform(X_clean, y)
            selected_features = X.columns[selector.get_support()].tolist()

            # Store selection information
            scores = selector.scores_
            feature_scores = dict(zip(X.columns, scores))

            self.fitted_selectors[f'univariate_{score_func}'] = {
                'selector': selector,
                'method': f'univariate_{score_func}',
                'selected_features': selected_features,
                'feature_scores': feature_scores,
                'k': k
            }

            self.selection_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': f'univariate_{score_func}',
                'selected_count': len(selected_features),
                'total_features': len(X.columns),
                'selected_features': selected_features
            })

            return selected_features

        except Exception as e:
            warnings.warn(f"Univariate feature selection failed: {str(e)}")
            # Fallback: return top k features by variance
            feature_variances = X.var().sort_values(ascending=False)
            return feature_variances.head(k).index.tolist()

    def select_features_importance(self, X: pd.DataFrame, y: pd.Series,
                                 model: Optional[BaseEstimator] = None,
                                 threshold: float = 0.01) -> List[str]:
        """
        Select features based on model-derived importance scores.

        Args:
            X: Feature matrix
            y: Target vector
            model: Fitted model with feature_importances_ attribute
            threshold: Minimum importance threshold

        Returns:
            List of selected feature names

        Educational Notes:
        - Uses trained model's feature importance
        - Random Forest provides robust importance estimates
        - Threshold-based selection allows flexibility
        - Can identify complex feature interactions
        """
        if model is None:
            # Use Random Forest as default
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )

        # Handle missing values
        X_clean = X.fillna(0)

        # Fit model if not already fitted
        if not hasattr(model, 'feature_importances_'):
            model.fit(X_clean, y)

        # Check if model has feature importances
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not provide feature importances")

        importances = model.feature_importances_
        feature_importance_dict = dict(zip(X.columns, importances))

        # Select features above threshold
        selected_features = [
            feature for feature, importance in feature_importance_dict.items()
            if importance >= threshold
        ]

        if not selected_features:
            warnings.warn(f"No features meet importance threshold {threshold}. Selecting top 10.")
            # Fallback: select top 10 features
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feature for feature, _ in sorted_features[:10]]

        # Store selection information
        self.fitted_selectors['importance_based'] = {
            'model': model,
            'method': 'importance_based',
            'selected_features': selected_features,
            'feature_importances': feature_importance_dict,
            'threshold': threshold
        }

        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'method': 'importance_based',
            'selected_count': len(selected_features),
            'total_features': len(X.columns),
            'threshold': threshold,
            'selected_features': selected_features
        })

        return selected_features

    def select_features_recursive(self, X: pd.DataFrame, y: pd.Series,
                                estimator: Optional[BaseEstimator] = None,
                                n_features: Optional[int] = None,
                                step: Union[int, float] = 1,
                                cv: Optional[int] = None) -> List[str]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            X: Feature matrix
            y: Target vector
            estimator: Base estimator for RFE
            n_features: Number of features to select (None for cross-validated selection)
            step: Number of features to remove at each iteration
            cv: Number of cross-validation folds (enables RFECV)

        Returns:
            List of selected feature names

        Educational Notes:
        - Recursive elimination based on model performance
        - RFECV automatically finds optimal number of features
        - More computationally expensive but often more accurate
        - Works well with linear models and tree-based models
        """
        if estimator is None:
            # Use logistic regression as default (good for RFE)
            estimator = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='liblinear'
            )

        # Handle missing values
        X_clean = X.fillna(0)

        try:
            if cv is not None:
                # Use cross-validated RFE
                selector = RFECV(
                    estimator=estimator,
                    step=step,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    n_jobs=-1
                )
                method_name = 'recursive_cv'
            else:
                # Use regular RFE
                if n_features is None:
                    n_features = max(5, len(X.columns) // 2)  # Default to half the features

                selector = RFE(
                    estimator=estimator,
                    n_features_to_select=n_features,
                    step=step
                )
                method_name = 'recursive'

            # Fit selector
            selector.fit(X_clean, y)
            selected_features = X.columns[selector.get_support()].tolist()

            # Get feature rankings
            feature_rankings = dict(zip(X.columns, selector.ranking_))

            # Store selection information
            self.fitted_selectors[method_name] = {
                'selector': selector,
                'method': method_name,
                'selected_features': selected_features,
                'feature_rankings': feature_rankings,
                'n_features_selected': len(selected_features)
            }

            self.selection_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': method_name,
                'selected_count': len(selected_features),
                'total_features': len(X.columns),
                'selected_features': selected_features
            })

            return selected_features

        except Exception as e:
            warnings.warn(f"Recursive feature selection failed: {str(e)}")
            # Fallback to importance-based selection
            return self.select_features_importance(X, y, threshold=0.01)

    def select_features_lasso(self, X: pd.DataFrame, y: pd.Series,
                            alpha: Optional[float] = None) -> List[str]:
        """
        Select features using L1 regularization (Lasso).

        Args:
            X: Feature matrix
            y: Target vector
            alpha: Regularization strength (None for automatic selection)

        Returns:
            List of selected feature names

        Educational Notes:
        - L1 regularization automatically performs feature selection
        - Sets coefficients of irrelevant features to zero
        - Good for high-dimensional data
        - Automatically handles feature interactions
        """
        from sklearn.linear_model import LassoCV, LogisticRegressionCV

        # Handle missing values
        X_clean = X.fillna(0)

        # Scale features for Lasso
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        try:
            if alpha is None:
                # Use cross-validation to find optimal alpha
                selector = LogisticRegressionCV(
                    penalty='l1',
                    solver='liblinear',
                    cv=5,
                    random_state=self.random_state,
                    max_iter=1000
                )
            else:
                selector = LogisticRegression(
                    penalty='l1',
                    C=1/alpha,  # sklearn uses C = 1/alpha
                    solver='liblinear',
                    random_state=self.random_state,
                    max_iter=1000
                )

            # Fit selector
            selector.fit(X_scaled, y)

            # Get non-zero coefficients
            if hasattr(selector, 'coef_'):
                coefficients = selector.coef_[0] if selector.coef_.ndim > 1 else selector.coef_
            else:
                # Fallback
                coefficients = np.zeros(len(X.columns))

            selected_features = [
                feature for feature, coef in zip(X.columns, coefficients)
                if abs(coef) > 1e-6  # Small threshold for numerical stability
            ]

            if not selected_features:
                warnings.warn("Lasso selected no features. Relaxing threshold.")
                # Select features with largest absolute coefficients
                feature_coefs = list(zip(X.columns, np.abs(coefficients)))
                feature_coefs.sort(key=lambda x: x[1], reverse=True)
                selected_features = [feature for feature, _ in feature_coefs[:10]]

            # Store selection information
            feature_coefficients = dict(zip(X.columns, coefficients))

            self.fitted_selectors['lasso'] = {
                'selector': selector,
                'scaler': scaler,
                'method': 'lasso',
                'selected_features': selected_features,
                'feature_coefficients': feature_coefficients,
                'alpha': alpha
            }

            self.selection_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': 'lasso',
                'selected_count': len(selected_features),
                'total_features': len(X.columns),
                'alpha': alpha,
                'selected_features': selected_features
            })

            return selected_features

        except Exception as e:
            warnings.warn(f"Lasso feature selection failed: {str(e)}")
            # Fallback to univariate selection
            return self.select_features_univariate(X, y, k=min(10, len(X.columns)))

    def analyze_feature_importance(self, feature_names: List[str],
                                 importances: np.ndarray) -> Dict[str, float]:
        """
        Analyze and rank feature importance scores.

        Args:
            feature_names: List of feature names
            importances: Array of importance scores

        Returns:
            Dict mapping feature names to importance scores (sorted)

        Educational Notes:
        - Provides interpretable feature ranking
        - Helps understand model behavior
        - Guides feature engineering decisions
        - Essential for stakeholder communication
        """
        if len(feature_names) != len(importances):
            raise ValueError("Length of feature_names must match length of importances")

        # Create feature importance dictionary
        feature_importance_dict = dict(zip(feature_names, importances))

        # Sort by importance (descending)
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

        # Convert back to dictionary (ordered)
        importance_analysis = {
            'ranked_features': dict(sorted_features),
            'top_10_features': dict(sorted_features[:10]),
            'bottom_10_features': dict(sorted_features[-10:]),
            'statistics': {
                'mean_importance': float(np.mean(importances)),
                'std_importance': float(np.std(importances)),
                'max_importance': float(np.max(importances)),
                'min_importance': float(np.min(importances)),
                'total_importance': float(np.sum(importances))
            }
        }

        # Calculate importance concentration
        cumulative_importance = np.cumsum([imp for _, imp in sorted_features])
        total_importance = cumulative_importance[-1]

        # Find how many features account for 80% of importance
        features_for_80_percent = np.argmax(cumulative_importance >= 0.8 * total_importance) + 1
        importance_analysis['concentration'] = {
            'features_for_80_percent': int(features_for_80_percent),
            'concentration_ratio': features_for_80_percent / len(feature_names)
        }

        return importance_analysis

    def select_features_combined(self, X: pd.DataFrame, y: pd.Series,
                               methods: List[str] = None,
                               voting: str = 'majority') -> List[str]:
        """
        Select features using multiple methods and combine results.

        Args:
            X: Feature matrix
            y: Target vector
            methods: List of methods to use
            voting: How to combine results ('majority', 'union', 'intersection')

        Returns:
            List of selected feature names

        Educational Notes:
        - Combines multiple selection strategies
        - More robust than single method
        - Reduces selection bias
        - Provides consensus-based selection
        """
        if methods is None:
            methods = ['univariate', 'importance', 'lasso']

        method_selections = {}

        # Apply each method
        for method in methods:
            try:
                if method == 'univariate':
                    k = max(5, len(X.columns) // 3)  # Select 1/3 of features
                    selected = self.select_features_univariate(X, y, k=k)
                elif method == 'importance':
                    selected = self.select_features_importance(X, y, threshold=0.01)
                elif method == 'lasso':
                    selected = self.select_features_lasso(X, y)
                elif method == 'recursive':
                    n_features = max(5, len(X.columns) // 3)
                    selected = self.select_features_recursive(X, y, n_features=n_features)
                else:
                    warnings.warn(f"Unknown method: {method}")
                    continue

                method_selections[method] = set(selected)

            except Exception as e:
                warnings.warn(f"Method {method} failed: {str(e)}")
                continue

        if not method_selections:
            warnings.warn("All methods failed. Returning top 10 features by variance.")
            feature_variances = X.var().sort_values(ascending=False)
            return feature_variances.head(10).index.tolist()

        # Combine selections based on voting strategy
        all_features = set()
        for features in method_selections.values():
            all_features.update(features)

        if voting == 'union':
            # Select features chosen by any method
            combined_features = all_features
        elif voting == 'intersection':
            # Select features chosen by all methods
            combined_features = all_features.copy()
            for features in method_selections.values():
                combined_features &= features
        elif voting == 'majority':
            # Select features chosen by majority of methods
            feature_votes = {}
            for features in method_selections.values():
                for feature in features:
                    feature_votes[feature] = feature_votes.get(feature, 0) + 1

            majority_threshold = len(method_selections) / 2
            combined_features = {
                feature for feature, votes in feature_votes.items()
                if votes > majority_threshold
            }
        else:
            raise ValueError(f"Unknown voting strategy: {voting}")

        # Convert to list and ensure we have at least some features
        selected_features = list(combined_features)

        if not selected_features:
            warnings.warn("Combined selection resulted in no features. Using union instead.")
            selected_features = list(all_features)

        # If still no features, fall back to top features by variance
        if not selected_features:
            feature_variances = X.var().sort_values(ascending=False)
            selected_features = feature_variances.head(10).index.tolist()

        # Store combined selection information
        self.fitted_selectors['combined'] = {
            'method': 'combined',
            'voting': voting,
            'methods_used': list(method_selections.keys()),
            'selected_features': selected_features,
            'method_selections': {k: list(v) for k, v in method_selections.items()}
        }

        self.selection_history.append({
            'timestamp': datetime.now().isoformat(),
            'method': 'combined',
            'voting': voting,
            'methods_used': list(method_selections.keys()),
            'selected_count': len(selected_features),
            'total_features': len(X.columns),
            'selected_features': selected_features
        })

        return selected_features

    def evaluate_feature_subset(self, X: pd.DataFrame, y: pd.Series,
                              feature_subset: List[str],
                              cv: int = 5) -> Dict[str, float]:
        """
        Evaluate performance of a feature subset using cross-validation.

        Args:
            X: Full feature matrix
            y: Target vector
            feature_subset: List of features to evaluate
            cv: Number of cross-validation folds

        Returns:
            Performance metrics for the feature subset
        """
        # Check that all features exist
        missing_features = set(feature_subset) - set(X.columns)
        if missing_features:
            raise ValueError(f"Features not found in X: {missing_features}")

        X_subset = X[feature_subset].fillna(0)

        # Use multiple models for evaluation
        models = {
            'random_forest': RandomForestClassifier(n_estimators=50, random_state=self.random_state),
            'logistic_regression': LogisticRegression(random_state=self.random_state, max_iter=1000)
        }

        evaluation_results = {
            'feature_count': len(feature_subset),
            'features': feature_subset
        }

        for model_name, model in models.items():
            try:
                cv_scores = cross_val_score(
                    model, X_subset, y,
                    cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
                    scoring='roc_auc',
                    n_jobs=-1
                )

                evaluation_results[f'{model_name}_mean_auc'] = float(np.mean(cv_scores))
                evaluation_results[f'{model_name}_std_auc'] = float(np.std(cv_scores))

            except Exception as e:
                warnings.warn(f"Evaluation failed for {model_name}: {str(e)}")
                evaluation_results[f'{model_name}_mean_auc'] = 0.0
                evaluation_results[f'{model_name}_std_auc'] = 0.0

        return evaluation_results

    def get_selection_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all feature selection results.

        Returns:
            Summary of feature selection history and recommendations
        """
        if not self.selection_history:
            return {'message': 'No feature selection history available'}

        summary = {
            'total_selections': len(self.selection_history),
            'methods_used': list(set(h['method'] for h in self.selection_history)),
            'recent_selection': self.selection_history[-1],
            'selection_consistency': {}
        }

        # Analyze consistency across methods
        if len(self.selection_history) > 1:
            all_selected_features = set()
            method_features = {}

            for history in self.selection_history:
                method = history['method']
                features = set(history['selected_features'])
                method_features[method] = features
                all_selected_features.update(features)

            # Find commonly selected features
            if len(method_features) > 1:
                common_features = all_selected_features.copy()
                for features in method_features.values():
                    common_features &= features

                summary['selection_consistency'] = {
                    'total_unique_features': len(all_selected_features),
                    'commonly_selected': list(common_features),
                    'consistency_score': len(common_features) / len(all_selected_features) if all_selected_features else 0
                }

        # Recommendations based on selection history
        recommendations = []

        if len(self.selection_history) > 1:
            recent_counts = [h['selected_count'] for h in self.selection_history[-3:]]
            if max(recent_counts) - min(recent_counts) > len(self.selection_history[-1]['selected_features']) * 0.5:
                recommendations.append("Feature selection results vary significantly - consider ensemble selection")

        if 'commonly_selected' in summary.get('selection_consistency', {}):
            common_count = len(summary['selection_consistency']['commonly_selected'])
            if common_count < 5:
                recommendations.append("Few features consistently selected - may need more stable selection criteria")

        summary['recommendations'] = recommendations

        return summary
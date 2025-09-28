"""
Cross-validation Pipeline with SMOTE Integration
Provides robust cross-validation framework for imbalanced churn data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import make_scorer, confusion_matrix, classification_report
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')


class SMOTECrossValidator:
    """
    Cross-validation pipeline with integrated SMOTE for handling class imbalance
    """

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42,
                 smote_threshold: float = 0.3,
                 shuffle: bool = True):
        """
        Initialize the SMOTE cross-validator

        Args:
            n_splits: Number of CV folds
            random_state: Random seed for reproducibility
            smote_threshold: Apply SMOTE if minority class ratio < threshold
            shuffle: Whether to shuffle data before splitting
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.smote_threshold = smote_threshold
        self.shuffle = shuffle

        # Initialize components
        self.cv_splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )

        # Set random seed for reproducibility
        np.random.seed(random_state)

    def _check_class_imbalance(self, y: pd.Series) -> Tuple[bool, float]:
        """
        Check if the dataset has class imbalance

        Args:
            y: Target variable

        Returns:
            Tuple of (needs_smote, minority_ratio)
        """
        class_counts = y.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()
        needs_smote = minority_ratio < self.smote_threshold

        return needs_smote, minority_ratio

    def _should_apply_smote(self, y) -> bool:
        """
        Determine if SMOTE should be applied based on class imbalance

        Args:
            y: Target variable

        Returns:
            Boolean indicating if SMOTE should be applied
        """
        if y is None:
            raise ValueError("Target variable cannot be None")

        if hasattr(y, '__len__') and len(y) == 0:
            raise ValueError("Target variable cannot be empty")

        needs_smote, _ = self._check_class_imbalance(pd.Series(y) if not isinstance(y, pd.Series) else y)
        return needs_smote

    def _apply_smote_to_fold(self, X_train, y_train):
        """
        Apply SMOTE to a training fold

        Args:
            X_train: Training features for the fold
            y_train: Training labels for the fold

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        smote = SMOTE(random_state=self.random_state, k_neighbors=min(3, len(y_train[y_train==1])-1))
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled

    def cross_validate_with_smote(self, model, X, y, scoring_func=None):
        """
        Perform cross-validation with optional SMOTE application

        Args:
            model: Classifier to evaluate
            X: Feature matrix
            y: Target variable
            scoring_func: Scoring function to use

        Returns:
            List of scores for each fold
        """
        # Input validation
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if scoring_func is None:
            from sklearn.metrics import f1_score
            scoring_func = f1_score

        scores = []
        should_apply_smote = self._should_apply_smote(y)

        for train_idx, val_idx in self.cv_splitter.split(X, y):
            # Split data
            X_train_fold = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
            y_train_fold = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
            X_val_fold = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
            y_val_fold = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx]

            # Apply SMOTE if needed
            if should_apply_smote:
                X_train_fold, y_train_fold = self._apply_smote_to_fold(X_train_fold, y_train_fold)

            # Train and predict
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)

            # Calculate score
            score = scoring_func(y_val_fold, y_pred)
            scores.append(score)

        return scores

    def get_fold_statistics(self, X, y):
        """
        Get statistics about the cross-validation folds

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with fold statistics
        """
        should_apply_smote = self._should_apply_smote(y)

        # Convert y to pandas Series if needed for value_counts
        y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
        class_counts = y_series.value_counts().to_dict()

        stats = {
            'total_folds': self.n_splits,
            'folds_with_smote': self.n_splits if should_apply_smote else 0,
            'class_distribution': class_counts,
            'minority_class_ratio': min(class_counts.values()) / max(class_counts.values()),
            'smote_applied': should_apply_smote
        }

        return stats

    def _create_pipeline(self, estimator: BaseEstimator, use_smote: bool) -> Union[BaseEstimator, ImbPipeline]:
        """
        Create pipeline with or without SMOTE

        Args:
            estimator: Base estimator
            use_smote: Whether to include SMOTE

        Returns:
            Pipeline or estimator
        """
        if use_smote:
            smote = SMOTE(random_state=self.random_state, k_neighbors=3)
            pipeline = ImbPipeline([
                ('smote', smote),
                ('classifier', estimator)
            ])
            return pipeline
        else:
            return estimator

    def cross_validate_model(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            estimator: BaseEstimator,
                            scoring: Optional[Dict[str, Any]] = None,
                            return_train_score: bool = False) -> Dict[str, Any]:
        """
        Perform cross-validation with optional SMOTE integration

        Args:
            X: Feature matrix
            y: Target variable
            estimator: ML model to evaluate
            scoring: Custom scoring functions
            return_train_score: Whether to return training scores

        Returns:
            Dictionary with CV results and metadata
        """
        # Check for class imbalance
        needs_smote, minority_ratio = self._check_class_imbalance(y)

        # Create pipeline
        model_pipeline = self._create_pipeline(estimator, needs_smote)

        # Define default scoring if not provided
        if scoring is None:
            scoring = {
                'f1': make_scorer(f1_score),
                'precision': make_scorer(precision_score),
                'recall': make_scorer(recall_score),
                'roc_auc': make_scorer(roc_auc_score)
            }

        # Perform cross-validation
        cv_results = cross_validate(
            estimator=model_pipeline,
            X=X,
            y=y,
            cv=self.cv_splitter,
            scoring=scoring,
            return_train_score=return_train_score,
            n_jobs=-1
        )

        # Compile comprehensive results
        results = self._compile_results(cv_results, needs_smote, minority_ratio)

        return results

    def _compile_results(self,
                        cv_results: Dict[str, np.ndarray],
                        needs_smote: bool,
                        minority_ratio: float) -> Dict[str, Any]:
        """
        Compile cross-validation results with metadata

        Args:
            cv_results: Raw CV results from sklearn
            needs_smote: Whether SMOTE was applied
            minority_ratio: Minority class ratio

        Returns:
            Comprehensive results dictionary
        """
        compiled_results = {}

        # Process test scores
        for metric_key, scores in cv_results.items():
            if metric_key.startswith('test_'):
                metric_name = metric_key.replace('test_', '')
                compiled_results[f'{metric_name}_scores'] = scores
                compiled_results[f'{metric_name}_mean'] = scores.mean()
                compiled_results[f'{metric_name}_std'] = scores.std()
                compiled_results[f'{metric_name}_min'] = scores.min()
                compiled_results[f'{metric_name}_max'] = scores.max()

        # Process train scores if available
        for metric_key, scores in cv_results.items():
            if metric_key.startswith('train_'):
                metric_name = metric_key.replace('train_', '')
                compiled_results[f'train_{metric_name}_mean'] = scores.mean()
                compiled_results[f'train_{metric_name}_std'] = scores.std()

        # Add metadata
        compiled_results.update({
            'n_splits': self.n_splits,
            'smote_applied': needs_smote,
            'minority_class_ratio': minority_ratio,
            'stratified': True,
            'random_state': self.random_state,
            'cv_configuration': {
                'n_splits': self.n_splits,
                'shuffle': self.shuffle,
                'smote_threshold': self.smote_threshold
            }
        })

        # Add fit times
        if 'fit_time' in cv_results:
            compiled_results['fit_time_mean'] = cv_results['fit_time'].mean()
            compiled_results['fit_time_std'] = cv_results['fit_time'].std()

        if 'score_time' in cv_results:
            compiled_results['score_time_mean'] = cv_results['score_time'].mean()
            compiled_results['score_time_std'] = cv_results['score_time'].std()

        return compiled_results

    def detailed_cross_validation(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 estimator: BaseEstimator) -> Dict[str, Any]:
        """
        Perform detailed cross-validation with per-fold analysis

        Args:
            X: Feature matrix
            y: Target variable
            estimator: ML model to evaluate

        Returns:
            Detailed results with per-fold metrics
        """
        needs_smote, minority_ratio = self._check_class_imbalance(y)
        model_pipeline = self._create_pipeline(estimator, needs_smote)

        fold_results = []
        fold_predictions = []
        fold_probabilities = []
        fold_true_labels = []

        for fold_idx, (train_idx, val_idx) in enumerate(self.cv_splitter.split(X, y)):
            # Split data
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model_pipeline.fit(X_train_fold, y_train_fold)

            # Make predictions
            y_pred = model_pipeline.predict(X_val_fold)
            y_proba = model_pipeline.predict_proba(X_val_fold)[:, 1]

            # Calculate fold metrics
            fold_metrics = {
                'fold': fold_idx + 1,
                'f1_score': f1_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred),
                'recall': recall_score(y_val_fold, y_pred),
                'roc_auc': roc_auc_score(y_val_fold, y_proba),
                'train_samples': len(y_train_fold),
                'val_samples': len(y_val_fold),
                'val_churn_rate': y_val_fold.mean(),
                'predicted_churn_rate': y_pred.mean()
            }

            # Add confusion matrix
            cm = confusion_matrix(y_val_fold, y_pred)
            fold_metrics.update({
                'tn': cm[0][0],
                'fp': cm[0][1],
                'fn': cm[1][0],
                'tp': cm[1][1]
            })

            fold_results.append(fold_metrics)
            fold_predictions.extend(y_pred)
            fold_probabilities.extend(y_proba)
            fold_true_labels.extend(y_val_fold)

        # Compile overall results
        overall_results = {
            'fold_results': fold_results,
            'overall_metrics': self._calculate_overall_metrics(fold_results),
            'smote_applied': needs_smote,
            'minority_class_ratio': minority_ratio,
            'predictions': {
                'predictions': fold_predictions,
                'probabilities': fold_probabilities,
                'true_labels': fold_true_labels
            }
        }

        return overall_results

    def _calculate_overall_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate overall metrics from fold results

        Args:
            fold_results: List of fold-specific results

        Returns:
            Overall aggregated metrics
        """
        metrics = ['f1_score', 'precision', 'recall', 'roc_auc']
        overall_metrics = {}

        for metric in metrics:
            values = [fold[metric] for fold in fold_results]
            overall_metrics[f'{metric}_mean'] = np.mean(values)
            overall_metrics[f'{metric}_std'] = np.std(values)
            overall_metrics[f'{metric}_min'] = np.min(values)
            overall_metrics[f'{metric}_max'] = np.max(values)

        return overall_metrics

    def compare_models(self,
                      X: pd.DataFrame,
                      y: pd.Series,
                      models: Dict[str, BaseEstimator]) -> pd.DataFrame:
        """
        Compare multiple models using cross-validation

        Args:
            X: Feature matrix
            y: Target variable
            models: Dictionary of {model_name: model_instance}

        Returns:
            DataFrame with comparison results
        """
        comparison_results = []

        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")

            # Perform CV
            cv_results = self.cross_validate_model(X, y, model)

            # Extract key metrics
            result_row = {
                'model_name': model_name,
                'f1_mean': cv_results['f1_mean'],
                'f1_std': cv_results['f1_std'],
                'precision_mean': cv_results['precision_mean'],
                'precision_std': cv_results['precision_std'],
                'recall_mean': cv_results['recall_mean'],
                'recall_std': cv_results['recall_std'],
                'roc_auc_mean': cv_results['roc_auc_mean'],
                'roc_auc_std': cv_results['roc_auc_std'],
                'smote_applied': cv_results['smote_applied']
            }

            comparison_results.append(result_row)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('f1_mean', ascending=False)

        return comparison_df

    def learning_curve_analysis(self,
                               X: pd.DataFrame,
                               y: pd.Series,
                               estimator: BaseEstimator,
                               train_sizes: np.ndarray = None) -> Dict[str, Any]:
        """
        Analyze learning curves with different training set sizes

        Args:
            X: Feature matrix
            y: Target variable
            estimator: ML model to evaluate
            train_sizes: Array of training set sizes to evaluate

        Returns:
            Learning curve results
        """
        from sklearn.model_selection import learning_curve

        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        needs_smote, _ = self._check_class_imbalance(y)
        model_pipeline = self._create_pipeline(estimator, needs_smote)

        # Generate learning curve
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator=model_pipeline,
            X=X,
            y=y,
            train_sizes=train_sizes,
            cv=self.cv_splitter,
            scoring='f1',
            n_jobs=-1,
            random_state=self.random_state
        )

        # Compile results
        learning_results = {
            'train_sizes': train_sizes_abs,
            'train_scores_mean': train_scores.mean(axis=1),
            'train_scores_std': train_scores.std(axis=1),
            'val_scores_mean': val_scores.mean(axis=1),
            'val_scores_std': val_scores.std(axis=1),
            'smote_applied': needs_smote
        }

        return learning_results


def create_cv_pipeline(n_splits: int = 5,
                      random_state: int = 42,
                      smote_threshold: float = 0.3) -> SMOTECrossValidator:
    """
    Factory function to create a cross-validation pipeline

    Args:
        n_splits: Number of CV folds
        random_state: Random seed
        smote_threshold: Threshold for applying SMOTE

    Returns:
        Configured SMOTECrossValidator
    """
    return SMOTECrossValidator(
        n_splits=n_splits,
        random_state=random_state,
        smote_threshold=smote_threshold
    )


def evaluate_model_stability(cv_results: Dict[str, Any],
                           threshold: float = 0.05) -> Dict[str, bool]:
    """
    Evaluate model stability based on CV standard deviations

    Args:
        cv_results: Results from cross-validation
        threshold: Maximum acceptable standard deviation

    Returns:
        Dictionary with stability assessments
    """
    stability_assessment = {}

    metrics_to_check = ['f1', 'precision', 'recall', 'roc_auc']

    for metric in metrics_to_check:
        std_key = f'{metric}_std'
        if std_key in cv_results:
            stability_assessment[f'{metric}_stable'] = cv_results[std_key] <= threshold

    return stability_assessment
"""
MLflow Experiment Configuration and Logging Utilities
Provides comprehensive MLflow integration for experiment tracking and reproducibility
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import os
import json
from datetime import datetime
import tempfile
import pickle
import warnings
warnings.filterwarnings('ignore')


class MLflowExperimentManager:
    """Manages MLflow experiments for churn prediction model development"""

    def __init__(self, experiment_name: str = "customer_churn_prediction", tracking_uri: str = None):
        """
        Initialize MLflow experiment manager

        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking URI (defaults to local sqlite)
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        # Set tracking URI (default to local sqlite in project root)
        if tracking_uri is None:
            self.tracking_uri = "sqlite:///mlflow.db"

        mlflow.set_tracking_uri(self.tracking_uri)

        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment_id = experiment_id
            else:
                self.experiment_id = experiment.experiment_id
        except Exception as e:
            print(f"Warning: Could not set up MLflow experiment: {e}")
            self.experiment_id = None

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """
        Start a new MLflow run

        Args:
            run_name: Optional name for the run
            tags: Optional tags to add to the run

        Returns:
            Run ID
        """
        run_tags = tags or {}

        # Add default tags
        run_tags.update({
            'project': 'customer_churn_prediction',
            'framework': 'scikit-learn',
            'created_at': datetime.now().isoformat()
        })

        self.active_run = mlflow.start_run(run_name=run_name, tags=run_tags)
        return self.active_run.info.run_id

    def log_experiment_config(self, config: Dict[str, Any]):
        """
        Log experiment configuration parameters

        Args:
            config: Dictionary of configuration parameters
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log individual parameters
        for key, value in config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
            else:
                # Log complex objects as JSON strings
                mlflow.log_param(key, json.dumps(value, default=str))

    def log_data_info(self, train_data: pd.DataFrame, test_data: pd.DataFrame = None):
        """
        Log dataset information and statistics

        Args:
            train_data: Training dataset
            test_data: Optional test dataset
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log training data info
        mlflow.log_param("train_samples", len(train_data))
        mlflow.log_param("train_features", train_data.shape[1])

        if 'churn' in train_data.columns:
            churn_rate = train_data['churn'].mean()
            mlflow.log_metric("train_churn_rate", churn_rate)

        # Log test data info if provided
        if test_data is not None:
            mlflow.log_param("test_samples", len(test_data))
            if 'churn' in test_data.columns:
                test_churn_rate = test_data['churn'].mean()
                mlflow.log_metric("test_churn_rate", test_churn_rate)

        # Log data quality metrics
        missing_rate = train_data.isnull().sum().sum() / (train_data.shape[0] * train_data.shape[1])
        mlflow.log_metric("data_missing_rate", missing_rate)

    def log_preprocessing_steps(self, preprocessing_info: Dict[str, Any]):
        """
        Log preprocessing steps and transformations

        Args:
            preprocessing_info: Dictionary with preprocessing information
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log preprocessing parameters
        for step, info in preprocessing_info.items():
            mlflow.log_param(f"preprocessing_{step}", json.dumps(info, default=str))

    def log_model_training(self, model: Any, model_name: str,
                          hyperparameters: Dict[str, Any] = None):
        """
        Log trained model and its hyperparameters

        Args:
            model: Trained scikit-learn model
            model_name: Name of the model
            hyperparameters: Model hyperparameters
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=f"models/{model_name}",
            registered_model_name=f"churn_prediction_{model_name}"
        )

        # Log hyperparameters
        if hyperparameters:
            self.log_hyperparameters(hyperparameters)

        # Log model type
        mlflow.log_param("model_type", model_name)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log model hyperparameters

        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        for param, value in hyperparameters.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(f"model_{param}", value)
            else:
                mlflow.log_param(f"model_{param}", json.dumps(value, default=str))

    def log_model_performance(self, performance_metrics: Dict[str, Any]):
        """
        Log model performance metrics

        Args:
            performance_metrics: Dictionary of performance metrics
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        for metric_name, metric_value in performance_metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(f"performance_{metric_name}", metric_value)
            elif isinstance(metric_value, dict):
                # Handle nested metrics
                for sub_metric, sub_value in metric_value.items():
                    if isinstance(sub_value, (int, float)):
                        mlflow.log_metric(f"performance_{metric_name}_{sub_metric}", sub_value)

    def log_cross_validation_results(self, cv_results: Dict[str, Any]):
        """
        Log cross-validation results and metrics

        Args:
            cv_results: Dictionary with CV results from model development service
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log CV metrics
        metrics_to_log = ['f1_score', 'precision', 'recall', 'auc_roc']

        for metric in metrics_to_log:
            if f'{metric}_mean' in cv_results:
                mlflow.log_metric(f"cv_{metric}_mean", cv_results[f'{metric}_mean'])
                mlflow.log_metric(f"cv_{metric}_std", cv_results[f'{metric}_std'])

        # Log CV configuration
        mlflow.log_param("cv_folds", cv_results.get('n_folds', 5))
        mlflow.log_param("cv_stratified", cv_results.get('stratified', True))
        mlflow.log_param("cv_smote_applied", cv_results.get('smote_applied', False))

    def log_model_evaluation(self, evaluation_metrics: Dict[str, Any],
                           dataset_type: str = "test"):
        """
        Log model evaluation metrics

        Args:
            evaluation_metrics: Dictionary with evaluation metrics
            dataset_type: Type of dataset (train/test/validation)
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log evaluation metrics
        metrics_to_log = ['f1_score', 'precision', 'recall', 'auc_roc']

        for metric in metrics_to_log:
            if metric in evaluation_metrics:
                mlflow.log_metric(f"{dataset_type}_{metric}", evaluation_metrics[metric])

        # Log confusion matrix if available
        if 'confusion_matrix' in evaluation_metrics:
            cm = evaluation_metrics['confusion_matrix']
            if isinstance(cm, list):
                cm = np.array(cm)

            # Log confusion matrix elements
            mlflow.log_metric(f"{dataset_type}_tn", cm[0][0])
            mlflow.log_metric(f"{dataset_type}_fp", cm[0][1])
            mlflow.log_metric(f"{dataset_type}_fn", cm[1][0])
            mlflow.log_metric(f"{dataset_type}_tp", cm[1][1])

    def log_business_impact(self, business_metrics: Dict[str, Any]):
        """
        Log business impact analysis results

        Args:
            business_metrics: Dictionary with business impact metrics
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Log business metrics
        business_metrics_to_log = [
            'at_risk_customers_identified',
            'revenue_protection_estimate',
            'cost_reduction_estimate'
        ]

        for metric in business_metrics_to_log:
            if metric in business_metrics:
                mlflow.log_metric(f"business_{metric}", business_metrics[metric])

        # Log ROI analysis
        roi_analysis = business_metrics.get('roi_analysis', {})
        for key, value in roi_analysis.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"roi_{key}", value)

    def log_feature_importance(self, feature_importance: Dict[str, float],
                              top_n: int = 10):
        """
        Log feature importance scores

        Args:
            feature_importance: Dictionary with feature importance scores
            top_n: Number of top features to log
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Get top N features
        sorted_features = sorted(feature_importance.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]

        # Log top features
        for i, (feature, importance) in enumerate(sorted_features):
            if isinstance(importance, (int, float)):
                mlflow.log_metric(f"feature_importance_{i+1}_{feature}", importance)

        # Log feature importance as artifact
        importance_df = pd.DataFrame(sorted_features, columns=['feature', 'importance'])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            importance_df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, "feature_importance")

        os.unlink(f.name)

    def log_artifact_from_dataframe(self, df: pd.DataFrame, artifact_name: str,
                                   artifact_path: str = "data"):
        """
        Log a DataFrame as a CSV artifact

        Args:
            df: DataFrame to log
            artifact_name: Name for the artifact
            artifact_path: Path within artifacts directory
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            mlflow.log_artifact(f.name, f"{artifact_path}/{artifact_name}")

        os.unlink(f.name)

    def log_predictions(self, predictions: np.ndarray, probabilities: np.ndarray = None,
                       true_labels: np.ndarray = None, customer_ids: List[str] = None):
        """
        Log model predictions as artifacts

        Args:
            predictions: Binary predictions
            probabilities: Prediction probabilities
            true_labels: True labels (if available)
            customer_ids: Customer IDs (if available)
        """
        if not mlflow.active_run():
            raise RuntimeError("No active MLflow run. Call start_run() first.")

        # Create predictions DataFrame
        pred_data = {'predictions': predictions}

        if probabilities is not None:
            pred_data['probabilities'] = probabilities

        if true_labels is not None:
            pred_data['true_labels'] = true_labels

        if customer_ids is not None:
            pred_data['customer_id'] = customer_ids

        predictions_df = pd.DataFrame(pred_data)
        self.log_artifact_from_dataframe(predictions_df, "predictions.csv", "predictions")

    def end_run(self):
        """End the current MLflow run"""
        if mlflow.active_run():
            mlflow.end_run()

    def get_best_run(self, metric: str = "cv_f1_score_mean",
                    ascending: bool = False) -> mlflow.entities.Run:
        """
        Get the best run based on a specific metric

        Args:
            metric: Metric to optimize for
            ascending: Whether to sort in ascending order

        Returns:
            Best MLflow run
        """
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"]
        )

        if len(runs) > 0:
            return runs.iloc[0]
        else:
            return None

    def load_best_model(self, metric: str = "cv_f1_score_mean"):
        """
        Load the best model based on a specific metric

        Args:
            metric: Metric to optimize for

        Returns:
            Loaded model
        """
        best_run = self.get_best_run(metric)

        if best_run is not None:
            model_uri = f"runs:/{best_run.run_id}/models"
            return mlflow.sklearn.load_model(model_uri)
        else:
            return None


def setup_mlflow_experiment(experiment_name: str = "customer_churn_prediction",
                          tracking_uri: str = None) -> MLflowExperimentManager:
    """
    Quick setup function for MLflow experiment

    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI

    Returns:
        Configured MLflowExperimentManager
    """
    return MLflowExperimentManager(experiment_name, tracking_uri)


def log_system_info():
    """Log system and environment information"""
    if not mlflow.active_run():
        raise RuntimeError("No active MLflow run. Call start_run() first.")

    import platform
    import sys

    mlflow.log_param("python_version", sys.version)
    mlflow.log_param("platform", platform.platform())
    mlflow.log_param("processor", platform.processor())

    # Log package versions
    try:
        import sklearn
        mlflow.log_param("sklearn_version", sklearn.__version__)
    except ImportError:
        pass

    try:
        import pandas as pd
        mlflow.log_param("pandas_version", pd.__version__)
    except ImportError:
        pass

    try:
        import numpy as np
        mlflow.log_param("numpy_version", np.__version__)
    except ImportError:
        pass
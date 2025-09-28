"""
Model Evaluation Metrics and Visualization Utilities
Comprehensive evaluation framework for churn prediction models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, roc_auc_score
)
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Set style for all plots
plt.style.use('default')
sns.set_palette("husl")


class ChurnModelEvaluator:
    """
    Comprehensive model evaluation for churn prediction
    """

    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the model evaluator

        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.evaluation_results = {}

    def calculate_comprehensive_metrics(self,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities

        Returns:
            Dictionary with comprehensive metrics
        """
        metrics = {}

        # Basic classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1_score'] = f1_score(y_true, y_pred)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        metrics['tn'] = cm[0][0]
        metrics['fp'] = cm[0][1]
        metrics['fn'] = cm[1][0]
        metrics['tp'] = cm[1][1]

        # Additional derived metrics
        metrics['specificity'] = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        metrics['sensitivity'] = metrics['recall']  # Same as recall
        metrics['false_positive_rate'] = cm[0][1] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
        metrics['false_negative_rate'] = cm[1][0] / (cm[1][0] + cm[1][1]) if (cm[1][0] + cm[1][1]) > 0 else 0

        # Probability-based metrics
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_proba)

        # Business metrics
        metrics['predicted_churn_rate'] = y_pred.mean()
        metrics['actual_churn_rate'] = y_true.mean()

        return metrics

    def plot_confusion_matrix(self,
                             y_true: np.ndarray,
                             y_pred: np.ndarray,
                             normalize: bool = False,
                             title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix

        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Whether to normalize the matrix
            title: Plot title

        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])

        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()
        return fig

    def plot_roc_curve(self,
                      y_true: np.ndarray,
                      y_proba: np.ndarray,
                      title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(fpr, tpr, color='darkorange', lw=2,
               label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
               label='Random classifier')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_precision_recall_curve(self,
                                   y_true: np.ndarray,
                                   y_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot precision-recall curve

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title

        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(recall, precision, color='darkorange', lw=2,
               label=f'PR curve (AP = {avg_precision:.3f})')

        # Add baseline (random classifier performance)
        baseline = y_true.mean()
        ax.axhline(y=baseline, color='navy', linestyle='--',
                  label=f'Random classifier (AP = {baseline:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_calibration_curve(self,
                              y_true: np.ndarray,
                              y_proba: np.ndarray,
                              n_bins: int = 10,
                              title: str = "Calibration Curve") -> plt.Figure:
        """
        Plot calibration curve (reliability diagram)

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            n_bins: Number of bins for calibration
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins
        )

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(mean_predicted_value, fraction_of_positives, "s-", color='darkorange',
               label='Model', linewidth=2, markersize=8)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True)

        plt.tight_layout()
        return fig

    def plot_threshold_analysis(self,
                               y_true: np.ndarray,
                               y_proba: np.ndarray,
                               title: str = "Threshold Analysis") -> plt.Figure:
        """
        Plot metrics vs decision threshold

        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            title: Plot title

        Returns:
            Matplotlib figure
        """
        thresholds = np.linspace(0, 1, 100)
        metrics_vs_threshold = {
            'precision': [],
            'recall': [],
            'f1_score': [],
            'specificity': []
        }

        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)

            if len(np.unique(y_pred_thresh)) > 1:  # Avoid division by zero
                precision = precision_score(y_true, y_pred_thresh)
                recall = recall_score(y_true, y_pred_thresh)
                f1 = f1_score(y_true, y_pred_thresh)

                cm = confusion_matrix(y_true, y_pred_thresh)
                specificity = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0

                metrics_vs_threshold['precision'].append(precision)
                metrics_vs_threshold['recall'].append(recall)
                metrics_vs_threshold['f1_score'].append(f1)
                metrics_vs_threshold['specificity'].append(specificity)
            else:
                metrics_vs_threshold['precision'].append(0)
                metrics_vs_threshold['recall'].append(0)
                metrics_vs_threshold['f1_score'].append(0)
                metrics_vs_threshold['specificity'].append(0)

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(thresholds, metrics_vs_threshold['precision'], label='Precision', linewidth=2)
        ax.plot(thresholds, metrics_vs_threshold['recall'], label='Recall', linewidth=2)
        ax.plot(thresholds, metrics_vs_threshold['f1_score'], label='F1-Score', linewidth=2)
        ax.plot(thresholds, metrics_vs_threshold['specificity'], label='Specificity', linewidth=2)

        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        return fig

    def plot_feature_importance(self,
                               feature_importance: Dict[str, float],
                               top_n: int = 15,
                               title: str = "Feature Importance") -> plt.Figure:
        """
        Plot feature importance

        Args:
            feature_importance: Dictionary with feature importance scores
            top_n: Number of top features to display
            title: Plot title

        Returns:
            Matplotlib figure
        """
        # Filter out non-numeric values and get top N features
        numeric_importance = {k: v for k, v in feature_importance.items()
                            if isinstance(v, (int, float))}

        if not numeric_importance:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, 'No numeric feature importance available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig

        sorted_features = sorted(numeric_importance.items(),
                               key=lambda x: x[1], reverse=True)[:top_n]

        features, importances = zip(*sorted_features)

        fig, ax = plt.subplots(figsize=(10, 8))

        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='steelblue')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(title)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        return fig

    def plot_prediction_distribution(self,
                                   y_proba: np.ndarray,
                                   y_true: np.ndarray,
                                   title: str = "Prediction Probability Distribution") -> plt.Figure:
        """
        Plot distribution of prediction probabilities

        Args:
            y_proba: Prediction probabilities
            y_true: True labels
            title: Plot title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot distributions for each class
        ax.hist(y_proba[y_true == 0], bins=30, alpha=0.7, label='No Churn (Actual)',
               color='steelblue', density=True)
        ax.hist(y_proba[y_true == 1], bins=30, alpha=0.7, label='Churn (Actual)',
               color='coral', density=True)

        ax.set_xlabel('Predicted Probability of Churn')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_evaluation_report(self,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               y_proba: Optional[np.ndarray] = None,
                               feature_importance: Optional[Dict[str, float]] = None,
                               model_name: str = "Model") -> Dict[str, Any]:
        """
        Create comprehensive evaluation report

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            feature_importance: Feature importance scores
            model_name: Name of the model

        Returns:
            Dictionary with evaluation report and figures
        """
        print(f"Creating evaluation report for {model_name}...")

        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_true, y_pred, y_proba)

        # Create plots
        figures = {}

        # Confusion matrix
        figures['confusion_matrix'] = self.plot_confusion_matrix(
            y_true, y_pred, title=f"{model_name} - Confusion Matrix"
        )

        # ROC curve (if probabilities available)
        if y_proba is not None:
            figures['roc_curve'] = self.plot_roc_curve(
                y_true, y_proba, title=f"{model_name} - ROC Curve"
            )

            figures['pr_curve'] = self.plot_precision_recall_curve(
                y_true, y_proba, title=f"{model_name} - Precision-Recall Curve"
            )

            figures['calibration_curve'] = self.plot_calibration_curve(
                y_true, y_proba, title=f"{model_name} - Calibration Curve"
            )

            figures['threshold_analysis'] = self.plot_threshold_analysis(
                y_true, y_proba, title=f"{model_name} - Threshold Analysis"
            )

            figures['prediction_distribution'] = self.plot_prediction_distribution(
                y_proba, y_true, title=f"{model_name} - Prediction Distribution"
            )

        # Feature importance (if available)
        if feature_importance:
            figures['feature_importance'] = self.plot_feature_importance(
                feature_importance, title=f"{model_name} - Feature Importance"
            )

        # Classification report
        classification_rep = classification_report(y_true, y_pred, output_dict=True)

        report = {
            'model_name': model_name,
            'metrics': metrics,
            'classification_report': classification_rep,
            'figures': figures
        }

        self.evaluation_results[model_name] = report

        return report

    def compare_models(self, model_reports: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """
        Compare multiple models

        Args:
            model_reports: Dictionary of {model_name: evaluation_report}

        Returns:
            Comparison figure
        """
        metrics_to_compare = ['f1_score', 'precision', 'recall', 'roc_auc']
        comparison_data = []

        for model_name, report in model_reports.items():
            row = {'Model': model_name}
            for metric in metrics_to_compare:
                row[metric] = report['metrics'].get(metric, 0)
            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for i, metric in enumerate(metrics_to_compare):
            ax = axes[i]
            bars = ax.bar(df['Model'], df[metric], color='steelblue')
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')

        plt.tight_layout()
        return fig

    def generate_business_summary(self,
                                 metrics: Dict[str, Any],
                                 cost_per_customer: float = 25.0,
                                 revenue_per_customer: float = 900.0) -> Dict[str, Any]:
        """
        Generate business impact summary

        Args:
            metrics: Model evaluation metrics
            cost_per_customer: Cost of retention campaign per customer
            revenue_per_customer: Average revenue per customer

        Returns:
            Business impact summary
        """
        # Extract key metrics
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        predicted_churn_rate = metrics.get('predicted_churn_rate', 0)

        # Business calculations (simplified)
        customers_targeted = 1000 * predicted_churn_rate  # Assume 1000 customer base
        true_churners_caught = customers_targeted * precision
        campaign_cost = customers_targeted * cost_per_customer
        revenue_saved = true_churners_caught * revenue_per_customer * 0.3  # 30% retention success

        roi = (revenue_saved - campaign_cost) / campaign_cost * 100 if campaign_cost > 0 else 0

        business_summary = {
            'customers_targeted_for_retention': int(customers_targeted),
            'true_churners_identified': int(true_churners_caught),
            'campaign_cost': campaign_cost,
            'estimated_revenue_saved': revenue_saved,
            'roi_percentage': roi,
            'cost_per_true_positive': campaign_cost / true_churners_caught if true_churners_caught > 0 else float('inf')
        }

        return business_summary


def create_model_evaluator(figsize: Tuple[int, int] = (12, 8)) -> ChurnModelEvaluator:
    """
    Factory function to create a model evaluator

    Args:
        figsize: Default figure size for plots

    Returns:
        Configured ChurnModelEvaluator
    """
    return ChurnModelEvaluator(figsize=figsize)


def evaluate_churn_model(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None,
                        feature_importance: Optional[Dict[str, float]] = None,
                        model_name: str = "Model") -> Dict[str, Any]:
    """
    Convenience function to evaluate a churn prediction model

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        feature_importance: Feature importance scores
        model_name: Name of the model

    Returns:
        Complete evaluation report
    """
    evaluator = create_model_evaluator()
    return evaluator.create_evaluation_report(
        y_true, y_pred, y_proba, feature_importance, model_name
    )
"""
ModelPerformance entity model for churn prediction system.
Metrics and evaluations for model accuracy and business value.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelPerformance:
    """Metrics and evaluations for model accuracy and business value."""
    
    model_id: str
    experiment_id: str
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: List[float] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    training_date: Optional[datetime] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate model performance data after initialization."""
        if self.training_date is None:
            self.training_date = datetime.now()
        self._validate()
    
    def _validate(self) -> None:
        """Validate model performance data according to business rules."""
        if not self.model_id:
            raise ValueError("model_id must be non-null and non-empty")
        
        if not self.experiment_id:
            raise ValueError("experiment_id must be non-null and non-empty")
        
        if not isinstance(self.model_id, str):
            raise TypeError("model_id must be a string")
        
        if not isinstance(self.experiment_id, str):
            raise TypeError("experiment_id must be a string")
        
        # Validate accuracy metrics are between 0 and 1
        for metric_name, metric_value in self.accuracy_metrics.items():
            if not isinstance(metric_value, (int, float)):
                raise TypeError(f"Accuracy metric {metric_name} must be numeric")
            if not (0 <= metric_value <= 1):
                raise ValueError(f"Accuracy metric {metric_name} must be between 0 and 1, got {metric_value}")
        
        # Validate cross-validation scores (must have 5 values for K=5)
        if self.cross_validation_scores:
            if len(self.cross_validation_scores) != 5:
                raise ValueError(f"Cross-validation scores must have 5 values (K=5), got {len(self.cross_validation_scores)}")
            
            for i, score in enumerate(self.cross_validation_scores):
                if not isinstance(score, (int, float)):
                    raise TypeError(f"Cross-validation score {i} must be numeric")
                if not (0 <= score <= 1):
                    raise ValueError(f"Cross-validation score {i} must be between 0 and 1, got {score}")
        
        # Validate feature importance sums to 1.0
        if self.feature_importance:
            total_importance = sum(self.feature_importance.values())
            if abs(total_importance - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Feature importance must sum to 1.0, got {total_importance}")
            
            for feature_name, importance in self.feature_importance.items():
                if not isinstance(importance, (int, float)):
                    raise TypeError(f"Feature importance for {feature_name} must be numeric")
                if importance < 0:
                    raise ValueError(f"Feature importance for {feature_name} must be non-negative")
        
        # Validate confusion matrix is 2x2 for binary classification
        if self.confusion_matrix:
            if len(self.confusion_matrix) != 2:
                raise ValueError(f"Confusion matrix must be 2x2 for binary classification, got {len(self.confusion_matrix)} rows")
            
            for i, row in enumerate(self.confusion_matrix):
                if not isinstance(row, list):
                    raise TypeError(f"Confusion matrix row {i} must be a list")
                if len(row) != 2:
                    raise ValueError(f"Confusion matrix row {i} must have 2 columns, got {len(row)}")
                
                for j, value in enumerate(row):
                    if not isinstance(value, int) or value < 0:
                        raise ValueError(f"Confusion matrix value at [{i}][{j}] must be a non-negative integer")
        
        # Validate training_date
        if self.training_date is not None and not isinstance(self.training_date, datetime):
            raise TypeError("training_date must be a datetime object")
        
        # Validate hyperparameters is a dictionary
        if not isinstance(self.hyperparameters, dict):
            raise TypeError("hyperparameters must be a dictionary")
    
    def get_f1_score(self) -> Optional[float]:
        """Get F1-score (primary metric for churn prediction)."""
        return self.accuracy_metrics.get('f1_score')
    
    def get_auc_roc(self) -> Optional[float]:
        """Get AUC-ROC score."""
        return self.accuracy_metrics.get('auc_roc')
    
    def get_precision(self) -> Optional[float]:
        """Get precision score."""
        return self.accuracy_metrics.get('precision')
    
    def get_recall(self) -> Optional[float]:
        """Get recall score."""
        return self.accuracy_metrics.get('recall')
    
    def get_cv_mean_std(self) -> tuple[float, float]:
        """Get mean and standard deviation of cross-validation scores."""
        if not self.cross_validation_scores:
            return 0.0, 0.0
        return np.mean(self.cross_validation_scores), np.std(self.cross_validation_scores)
    
    def get_top_features(self, n: int = 5) -> List[tuple[str, float]]:
        """Get top N most important features."""
        if not self.feature_importance:
            return []
        
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]
    
    def calculate_accuracy_from_confusion_matrix(self) -> Optional[float]:
        """Calculate accuracy from confusion matrix."""
        if not self.confusion_matrix or len(self.confusion_matrix) != 2:
            return None
        
        tn, fp, fn, tp = (
            self.confusion_matrix[0][0],  # True Negative
            self.confusion_matrix[0][1],  # False Positive
            self.confusion_matrix[1][0],  # False Negative
            self.confusion_matrix[1][1]   # True Positive
        )
        
        total = tn + fp + fn + tp
        if total == 0:
            return 0.0
        
        return (tn + tp) / total
    
    def is_high_performing(self, f1_threshold: float = 0.8) -> bool:
        """Check if model meets performance threshold."""
        f1_score = self.get_f1_score()
        return f1_score is not None and f1_score >= f1_threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cv_mean, cv_std = self.get_cv_mean_std()
        top_features = self.get_top_features()
        
        return {
            'model_id': self.model_id,
            'experiment_id': self.experiment_id,
            'primary_metrics': {
                'f1_score': self.get_f1_score(),
                'auc_roc': self.get_auc_roc(),
                'precision': self.get_precision(),
                'recall': self.get_recall()
            },
            'cross_validation': {
                'mean_score': cv_mean,
                'std_score': cv_std,
                'scores': self.cross_validation_scores
            },
            'top_features': top_features,
            'confusion_matrix_accuracy': self.calculate_accuracy_from_confusion_matrix(),
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'is_high_performing': self.is_high_performing()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model performance to dictionary representation."""
        return {
            'model_id': self.model_id,
            'experiment_id': self.experiment_id,
            'accuracy_metrics': self.accuracy_metrics,
            'cross_validation_scores': self.cross_validation_scores,
            'feature_importance': self.feature_importance,
            'confusion_matrix': self.confusion_matrix,
            'training_date': self.training_date.isoformat() if self.training_date else None,
            'hyperparameters': self.hyperparameters
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelPerformance':
        """Create model performance from dictionary representation."""
        training_date = None
        if data.get('training_date'):
            training_date = datetime.fromisoformat(data['training_date'])
        
        return cls(
            model_id=data['model_id'],
            experiment_id=data['experiment_id'],
            accuracy_metrics=data.get('accuracy_metrics', {}),
            cross_validation_scores=data.get('cross_validation_scores', []),
            feature_importance=data.get('feature_importance', {}),
            confusion_matrix=data.get('confusion_matrix', []),
            training_date=training_date,
            hyperparameters=data.get('hyperparameters', {})
        )

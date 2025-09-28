"""
FeatureSet entity model for churn prediction system.
Processed and engineered features for machine learning.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeatureSet:
    """Processed and engineered features for machine learning."""
    
    customer_id: str
    behavioral_features: List[float] = field(default_factory=list)
    temporal_features: List[float] = field(default_factory=list)
    categorical_features: List[float] = field(default_factory=list)
    derived_features: List[float] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    preprocessing_version: str = "1.0.0"
    
    def __post_init__(self):
        """Validate feature set data after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate feature set data according to business rules."""
        if not self.customer_id:
            raise ValueError("customer_id must be non-null and non-empty")
        
        if not isinstance(self.customer_id, str):
            raise TypeError("customer_id must be a string")
        
        # Validate all feature arrays are lists
        feature_arrays = [
            ('behavioral_features', self.behavioral_features),
            ('temporal_features', self.temporal_features),
            ('categorical_features', self.categorical_features),
            ('derived_features', self.derived_features)
        ]
        
        for name, feature_array in feature_arrays:
            if not isinstance(feature_array, list):
                raise TypeError(f"{name} must be a list")
        
        # Validate feature names is a list of strings
        if not isinstance(self.feature_names, list):
            raise TypeError("feature_names must be a list")
        
        for name in self.feature_names:
            if not isinstance(name, str):
                raise TypeError("All feature names must be strings")
        
        # Validate no infinite or NaN values in features
        all_features = self.get_all_features()
        if all_features and any(not np.isfinite(val) for val in all_features):
            raise ValueError("Features must not contain infinite or NaN values")
        
        # Validate feature names match total feature count
        total_features = len(all_features)
        if len(self.feature_names) != total_features:
            raise ValueError(f"Number of feature names ({len(self.feature_names)}) must match total features ({total_features})")
        
        # Validate preprocessing version
        if not isinstance(self.preprocessing_version, str):
            raise TypeError("preprocessing_version must be a string")
    
    def get_all_features(self) -> List[float]:
        """Get all features as a single flat list."""
        all_features = []
        all_features.extend(self.behavioral_features)
        all_features.extend(self.temporal_features)
        all_features.extend(self.categorical_features)
        all_features.extend(self.derived_features)
        return all_features
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_all_features())
    
    def get_feature_dict(self) -> Dict[str, float]:
        """Get features as dictionary with feature names as keys."""
        all_features = self.get_all_features()
        if len(self.feature_names) != len(all_features):
            raise ValueError("Feature names and values must have same length")
        return dict(zip(self.feature_names, all_features))
    
    def add_behavioral_feature(self, value: float, name: str) -> None:
        """Add behavioral feature (usage and engagement metrics)."""
        if not np.isfinite(value):
            raise ValueError("Behavioral feature value must be finite")
        self.behavioral_features.append(value)
        self.feature_names.append(name)
    
    def add_temporal_feature(self, value: float, name: str) -> None:
        """Add temporal feature (time-based patterns and trends)."""
        if not np.isfinite(value):
            raise ValueError("Temporal feature value must be finite")
        self.temporal_features.append(value)
        self.feature_names.append(name)
    
    def add_categorical_feature(self, value: float, name: str) -> None:
        """Add categorical feature (encoded categorical variables)."""
        if not np.isfinite(value):
            raise ValueError("Categorical feature value must be finite")
        self.categorical_features.append(value)
        self.feature_names.append(name)
    
    def add_derived_feature(self, value: float, name: str) -> None:
        """Add derived feature (engineered metrics like CLV, ARPU, ratios)."""
        if not np.isfinite(value):
            raise ValueError("Derived feature value must be finite")
        self.derived_features.append(value)
        self.feature_names.append(name)
    
    def get_features_by_type(self, feature_type: str) -> List[float]:
        """Get features by type (behavioral, temporal, categorical, derived)."""
        if feature_type == 'behavioral':
            return self.behavioral_features.copy()
        elif feature_type == 'temporal':
            return self.temporal_features.copy()
        elif feature_type == 'categorical':
            return self.categorical_features.copy()
        elif feature_type == 'derived':
            return self.derived_features.copy()
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
    
    def is_model_ready(self) -> bool:
        """Check if feature set is ready for model input."""
        try:
            self._validate()
            return len(self.get_all_features()) > 0
        except (ValueError, TypeError):
            return False
    
    def get_preprocessing_state(self) -> str:
        """Get current preprocessing state based on feature content."""
        if not self.get_all_features():
            return "Raw"
        elif not self.categorical_features and not self.derived_features:
            return "Cleaned"
        elif not self.derived_features:
            return "Encoded"
        elif self.is_model_ready():
            return "Model-Ready"
        else:
            return "Engineered"
    
    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array(self.get_all_features())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feature set to dictionary representation."""
        return {
            'customer_id': self.customer_id,
            'behavioral_features': self.behavioral_features,
            'temporal_features': self.temporal_features,
            'categorical_features': self.categorical_features,
            'derived_features': self.derived_features,
            'feature_names': self.feature_names,
            'preprocessing_version': self.preprocessing_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSet':
        """Create feature set from dictionary representation."""
        return cls(
            customer_id=data['customer_id'],
            behavioral_features=data.get('behavioral_features', []),
            temporal_features=data.get('temporal_features', []),
            categorical_features=data.get('categorical_features', []),
            derived_features=data.get('derived_features', []),
            feature_names=data.get('feature_names', []),
            preprocessing_version=data.get('preprocessing_version', "1.0.0")
        )

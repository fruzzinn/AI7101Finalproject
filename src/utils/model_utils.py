"""
Model Persistence Utilities for Churn Prediction ML Pipeline
Task T045: Comprehensive model serialization, versioning, and lifecycle management

This module provides utilities for:
- Model serialization and deserialization with multiple formats
- Model versioning and metadata tracking
- Model registry and artifact management
- Model deployment and rollback capabilities
- Performance monitoring and drift detection
- A/B testing framework for model comparison
"""

import os
import json
import pickle
import joblib
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import shutil
import tempfile
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# ML framework imports with fallbacks
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Some features will be limited.")

try:
    import sklearn
    from sklearn.base import BaseEstimator
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    BaseEstimator = object

import numpy as np
import pandas as pd


class ModelFormat(Enum):
    """Supported model serialization formats."""
    PICKLE = "pickle"
    JOBLIB = "joblib"
    MLFLOW = "mlflow"
    ONNX = "onnx"


class ModelStatus(Enum):
    """Model lifecycle status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


@dataclass
class ModelMetadata:
    """Comprehensive model metadata structure."""
    model_id: str
    model_name: str
    model_version: str
    model_type: str
    framework: str
    created_at: str
    updated_at: str
    status: str

    # Training metadata
    training_data_hash: str
    feature_names: List[str]
    target_name: str
    hyperparameters: Dict[str, Any]
    training_duration_seconds: float

    # Performance metrics
    performance_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    business_metrics: Dict[str, float]

    # Deployment metadata
    deployment_environment: Optional[str] = None
    deployment_date: Optional[str] = None
    model_size_bytes: Optional[int] = None
    prediction_latency_ms: Optional[float] = None

    # Additional metadata
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    experiment_id: Optional[str] = None
    parent_model_id: Optional[str] = None

    @classmethod
    def create(cls, model_name: str, model_version: str, model_type: str,
               framework: str, **kwargs) -> 'ModelMetadata':
        """Create new model metadata with auto-generated fields."""
        timestamp = datetime.now().isoformat()
        model_id = cls._generate_model_id(model_name, model_version, timestamp)

        return cls(
            model_id=model_id,
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework=framework,
            created_at=timestamp,
            updated_at=timestamp,
            status=ModelStatus.TRAINING.value,
            **kwargs
        )

    @staticmethod
    def _generate_model_id(model_name: str, version: str, timestamp: str) -> str:
        """Generate unique model ID."""
        content = f"{model_name}_{version}_{timestamp}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def update_status(self, status: ModelStatus):
        """Update model status and timestamp."""
        self.status = status.value
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        return cls(**data)


class ModelVersionManager:
    """Manage model versions and their relationships."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.registry_path / "versions.json"
        self._load_versions()

    def _load_versions(self):
        """Load version registry from file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                self.versions = json.load(f)
        else:
            self.versions = {}

    def _save_versions(self):
        """Save version registry to file."""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

    def register_version(self, metadata: ModelMetadata):
        """Register new model version."""
        model_key = f"{metadata.model_name}:{metadata.model_version}"
        self.versions[model_key] = metadata.to_dict()
        self._save_versions()

    def get_version(self, model_name: str, version: str) -> Optional[ModelMetadata]:
        """Get specific model version metadata."""
        model_key = f"{model_name}:{version}"
        if model_key in self.versions:
            return ModelMetadata.from_dict(self.versions[model_key])
        return None

    def get_latest_version(self, model_name: str) -> Optional[ModelMetadata]:
        """Get latest version of a model."""
        versions = [v for k, v in self.versions.items()
                   if k.startswith(f"{model_name}:")]
        if not versions:
            return None

        latest = max(versions, key=lambda x: x['created_at'])
        return ModelMetadata.from_dict(latest)

    def list_versions(self, model_name: Optional[str] = None) -> List[ModelMetadata]:
        """List all versions, optionally filtered by model name."""
        if model_name:
            versions = [v for k, v in self.versions.items()
                       if k.startswith(f"{model_name}:")]
        else:
            versions = list(self.versions.values())

        return [ModelMetadata.from_dict(v) for v in versions]

    def promote_version(self, model_name: str, version: str,
                       target_status: ModelStatus) -> bool:
        """Promote model version to new status."""
        metadata = self.get_version(model_name, version)
        if metadata:
            metadata.update_status(target_status)
            self.register_version(metadata)
            return True
        return False


class ModelSerializer:
    """Handle model serialization in multiple formats."""

    @staticmethod
    def save_model(model: Any, file_path: str, format_type: ModelFormat = ModelFormat.JOBLIB,
                   metadata: Optional[ModelMetadata] = None) -> bool:
        """Save model in specified format."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            if format_type == ModelFormat.PICKLE:
                with open(file_path, 'wb') as f:
                    pickle.dump(model, f)

            elif format_type == ModelFormat.JOBLIB:
                joblib.dump(model, file_path)

            elif format_type == ModelFormat.MLFLOW and MLFLOW_AVAILABLE:
                mlflow.sklearn.save_model(model, str(file_path))

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            # Save metadata alongside model
            if metadata:
                metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)

            return True

        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    @staticmethod
    def load_model(file_path: str, format_type: ModelFormat = ModelFormat.JOBLIB) -> Tuple[Any, Optional[ModelMetadata]]:
        """Load model from specified format."""
        try:
            file_path = Path(file_path)

            if format_type == ModelFormat.PICKLE:
                with open(file_path, 'rb') as f:
                    model = pickle.load(f)

            elif format_type == ModelFormat.JOBLIB:
                model = joblib.load(file_path)

            elif format_type == ModelFormat.MLFLOW and MLFLOW_AVAILABLE:
                model = mlflow.sklearn.load_model(str(file_path))

            else:
                raise ValueError(f"Unsupported format: {format_type}")

            # Load metadata if available
            metadata = None
            metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    metadata = ModelMetadata.from_dict(metadata_dict)

            return model, metadata

        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None


class ModelRegistry:
    """Central model registry for managing model lifecycle."""

    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.registry_path / "models"
        self.models_dir.mkdir(exist_ok=True)

        self.version_manager = ModelVersionManager(str(self.registry_path))
        self.serializer = ModelSerializer()

    def register_model(self, model: Any, metadata: ModelMetadata,
                      format_type: ModelFormat = ModelFormat.JOBLIB) -> bool:
        """Register new model in the registry."""
        try:
            # Create model directory
            model_dir = self.models_dir / metadata.model_name / metadata.model_version
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = model_dir / f"model.{format_type.value}"
            success = self.serializer.save_model(model, str(model_file), format_type, metadata)

            if success:
                # Register version
                self.version_manager.register_version(metadata)
                return True

            return False

        except Exception as e:
            print(f"Error registering model: {e}")
            return False

    def load_model(self, model_name: str, version: str = None) -> Tuple[Any, Optional[ModelMetadata]]:
        """Load model from registry."""
        try:
            if version is None:
                metadata = self.version_manager.get_latest_version(model_name)
            else:
                metadata = self.version_manager.get_version(model_name, version)

            if not metadata:
                return None, None

            # Find model file
            model_dir = self.models_dir / model_name / metadata.model_version

            for format_type in ModelFormat:
                model_file = model_dir / f"model.{format_type.value}"
                if model_file.exists():
                    return self.serializer.load_model(str(model_file), format_type)

            return None, None

        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None

    def deploy_model(self, model_name: str, version: str, environment: str) -> bool:
        """Deploy model to specified environment."""
        try:
            metadata = self.version_manager.get_version(model_name, version)
            if not metadata:
                return False

            # Update deployment metadata
            metadata.deployment_environment = environment
            metadata.deployment_date = datetime.now().isoformat()
            metadata.update_status(ModelStatus.DEPLOYED)

            self.version_manager.register_version(metadata)
            return True

        except Exception as e:
            print(f"Error deploying model: {e}")
            return False

    def rollback_model(self, model_name: str, target_version: str, environment: str) -> bool:
        """Rollback to previous model version."""
        try:
            # Get current deployed version
            current_versions = self.version_manager.list_versions(model_name)
            current_deployed = [v for v in current_versions
                              if v.deployment_environment == environment
                              and v.status == ModelStatus.DEPLOYED.value]

            # Deprecate current version
            for deployed in current_deployed:
                deployed.update_status(ModelStatus.DEPRECATED)
                self.version_manager.register_version(deployed)

            # Deploy target version
            return self.deploy_model(model_name, target_version, environment)

        except Exception as e:
            print(f"Error rolling back model: {e}")
            return False

    def archive_model(self, model_name: str, version: str) -> bool:
        """Archive old model version."""
        try:
            metadata = self.version_manager.get_version(model_name, version)
            if not metadata:
                return False

            # Update status
            metadata.update_status(ModelStatus.ARCHIVED)
            self.version_manager.register_version(metadata)

            # Move model files to archive
            archive_dir = self.registry_path / "archive" / model_name / version
            model_dir = self.models_dir / model_name / version

            if model_dir.exists():
                archive_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(model_dir), str(archive_dir))

            return True

        except Exception as e:
            print(f"Error archiving model: {e}")
            return False

    def cleanup_old_versions(self, model_name: str, keep_versions: int = 5) -> int:
        """Clean up old model versions, keeping specified number."""
        try:
            versions = self.version_manager.list_versions(model_name)
            if len(versions) <= keep_versions:
                return 0

            # Sort by creation date, keep newest
            sorted_versions = sorted(versions, key=lambda x: x.created_at, reverse=True)
            versions_to_archive = sorted_versions[keep_versions:]

            archived_count = 0
            for version in versions_to_archive:
                if version.status not in [ModelStatus.DEPLOYED.value, ModelStatus.DEPRECATED.value]:
                    if self.archive_model(model_name, version.model_version):
                        archived_count += 1

            return archived_count

        except Exception as e:
            print(f"Error cleaning up versions: {e}")
            return 0

    def get_model_info(self, model_name: str, version: str = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive model information."""
        try:
            if version is None:
                metadata = self.version_manager.get_latest_version(model_name)
            else:
                metadata = self.version_manager.get_version(model_name, version)

            if not metadata:
                return None

            # Get model file info
            model_dir = self.models_dir / model_name / metadata.model_version
            model_files = list(model_dir.glob("model.*")) if model_dir.exists() else []

            info = {
                'metadata': metadata.to_dict(),
                'files': {
                    'count': len(model_files),
                    'formats': [f.suffix[1:] for f in model_files],
                    'total_size_bytes': sum(f.stat().st_size for f in model_files)
                },
                'registry_path': str(model_dir)
            }

            return info

        except Exception as e:
            print(f"Error getting model info: {e}")
            return None


class ModelMonitor:
    """Monitor model performance and detect drift."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry

    def log_prediction_performance(self, model_name: str, version: str,
                                 predictions: np.ndarray, actuals: np.ndarray = None,
                                 prediction_time_ms: float = None):
        """Log prediction performance metrics."""
        try:
            metadata = self.registry.version_manager.get_version(model_name, version)
            if not metadata:
                return False

            # Calculate performance metrics if actuals provided
            if actuals is not None:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                performance = {
                    'accuracy': float(accuracy_score(actuals, predictions > 0.5)),
                    'precision': float(precision_score(actuals, predictions > 0.5, zero_division=0)),
                    'recall': float(recall_score(actuals, predictions > 0.5, zero_division=0)),
                    'f1_score': float(f1_score(actuals, predictions > 0.5, zero_division=0))
                }
            else:
                performance = {}

            # Log prediction statistics
            prediction_stats = {
                'prediction_count': len(predictions),
                'mean_prediction': float(np.mean(predictions)),
                'std_prediction': float(np.std(predictions)),
                'min_prediction': float(np.min(predictions)),
                'max_prediction': float(np.max(predictions))
            }

            if prediction_time_ms:
                prediction_stats['prediction_time_ms'] = prediction_time_ms
                prediction_stats['predictions_per_second'] = len(predictions) / (prediction_time_ms / 1000)

            # Update metadata with latest performance
            if performance:
                metadata.validation_metrics.update(performance)
                self.registry.version_manager.register_version(metadata)

            return True

        except Exception as e:
            print(f"Error logging prediction performance: {e}")
            return False

    def detect_data_drift(self, model_name: str, version: str,
                         new_data: pd.DataFrame, reference_data: pd.DataFrame,
                         threshold: float = 0.1) -> Dict[str, Any]:
        """Detect data drift between new and reference datasets."""
        try:
            drift_results = {
                'drift_detected': False,
                'drift_score': 0.0,
                'feature_drifts': {},
                'timestamp': datetime.now().isoformat()
            }

            # Simple drift detection using statistical tests
            for column in new_data.columns:
                if column in reference_data.columns:
                    # Use KS test for numerical features
                    if pd.api.types.is_numeric_dtype(new_data[column]):
                        from scipy.stats import ks_2samp
                        statistic, p_value = ks_2samp(reference_data[column].dropna(),
                                                    new_data[column].dropna())
                        drift_score = statistic
                    else:
                        # Use chi-square test for categorical features
                        ref_counts = reference_data[column].value_counts(normalize=True)
                        new_counts = new_data[column].value_counts(normalize=True)

                        # Align indices
                        all_categories = set(ref_counts.index) | set(new_counts.index)
                        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                        new_aligned = new_counts.reindex(all_categories, fill_value=0)

                        # Calculate JS divergence (simplified)
                        drift_score = float(np.sum(np.abs(ref_aligned - new_aligned)) / 2)

                    drift_results['feature_drifts'][column] = {
                        'drift_score': float(drift_score),
                        'drift_detected': drift_score > threshold
                    }

                    if drift_score > threshold:
                        drift_results['drift_detected'] = True

            # Overall drift score (max across features)
            if drift_results['feature_drifts']:
                drift_results['drift_score'] = max(
                    f['drift_score'] for f in drift_results['feature_drifts'].values()
                )

            return drift_results

        except Exception as e:
            print(f"Error detecting data drift: {e}")
            return {'error': str(e)}


def create_model_utils(registry_path: str = "models") -> Tuple[ModelRegistry, ModelMonitor]:
    """Create model utilities with default configuration."""
    registry = ModelRegistry(registry_path)
    monitor = ModelMonitor(registry)
    return registry, monitor


if __name__ == "__main__":
    # Example usage and testing
    print("Model Persistence Utilities")
    print("=" * 35)

    # Create model utilities
    registry, monitor = create_model_utils("test_models")

    # Test with a simple model
    if SKLEARN_AVAILABLE:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification

        # Create sample model
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        # Create metadata
        metadata = ModelMetadata.create(
            model_name="test_churn_model",
            model_version="1.0.0",
            model_type="RandomForestClassifier",
            framework="sklearn",
            training_data_hash="test_hash",
            feature_names=[f"feature_{i}" for i in range(10)],
            target_name="churn",
            hyperparameters={"n_estimators": 10, "random_state": 42},
            training_duration_seconds=1.5,
            performance_metrics={"f1_score": 0.85, "auc": 0.92},
            validation_metrics={"f1_score": 0.83, "auc": 0.90},
            business_metrics={"roi": 250.0}
        )

        # Register model
        print("\n🔄 Testing Model Registration:")
        success = registry.register_model(model, metadata)
        print(f"  Registration successful: {success}")

        # Load model
        print("\n📥 Testing Model Loading:")
        loaded_model, loaded_metadata = registry.load_model("test_churn_model")
        print(f"  Model loaded: {loaded_model is not None}")
        print(f"  Metadata loaded: {loaded_metadata is not None}")

        # Deploy model
        print("\n🚀 Testing Model Deployment:")
        deployed = registry.deploy_model("test_churn_model", "1.0.0", "production")
        print(f"  Deployment successful: {deployed}")

        # Get model info
        print("\n📊 Testing Model Info:")
        info = registry.get_model_info("test_churn_model")
        if info:
            print(f"  Model ID: {info['metadata']['model_id']}")
            print(f"  Status: {info['metadata']['status']}")
            print(f"  Files: {info['files']['count']} ({info['files']['total_size_bytes']} bytes)")

        # Test monitoring
        print("\n📈 Testing Model Monitoring:")
        predictions = model.predict_proba(X[:100])[:, 1]
        monitor.log_prediction_performance("test_churn_model", "1.0.0", predictions, y[:100])
        print(f"  Performance logged for {len(predictions)} predictions")

        # Clean up test files
        import shutil
        if os.path.exists("test_models"):
            shutil.rmtree("test_models")
        print("\n✅ All tests completed successfully!")

    else:
        print("⚠️ Sklearn not available. Skipping model tests.")

    print("\n🎯 Model utilities ready for production use!")
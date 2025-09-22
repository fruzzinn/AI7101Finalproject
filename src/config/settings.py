"""
Configuration Management System for Churn Prediction ML Pipeline
Task T043: Centralized configuration management with environment-specific settings

This module provides a comprehensive configuration management system that supports:
- Environment-specific configurations (development, testing, production)
- Model hyperparameters and training configurations
- Data pipeline settings and paths
- Business rules and thresholds
- Feature engineering parameters
- Monitoring and alerting configurations
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from enum import Enum


class Environment(Enum):
    """Environment types for configuration management."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str
    port: int
    database: str
    username: str
    password: str
    connection_timeout: int = 30
    pool_size: int = 10

    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class ModelConfig:
    """Model training and inference configuration."""
    model_name: str
    model_version: str
    target_column: str
    feature_columns: List[str]
    categorical_features: List[str]
    numerical_features: List[str]

    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5

    # Model-specific hyperparameters
    hyperparameters: Dict[str, Any] = None

    # Performance thresholds
    min_f1_score: float = 0.80
    min_precision: float = 0.75
    min_recall: float = 0.75
    min_auc_score: float = 0.85


@dataclass
class DataConfig:
    """Data pipeline configuration."""
    # Data paths
    raw_data_path: str
    processed_data_path: str
    model_artifacts_path: str
    logs_path: str

    # Data quality thresholds
    max_missing_percentage: float = 0.10
    min_data_freshness_hours: int = 24
    outlier_detection_threshold: float = 3.0

    # Feature engineering
    feature_selection_threshold: float = 0.05
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01


@dataclass
class BusinessConfig:
    """Business rules and domain-specific configuration."""
    # Customer lifetime value parameters
    avg_monthly_revenue: float = 50.0
    customer_acquisition_cost: float = 150.0
    discount_rate: float = 0.08

    # Churn prediction thresholds
    high_risk_threshold: float = 0.75
    medium_risk_threshold: float = 0.50
    low_risk_threshold: float = 0.25

    # Intervention strategies
    retention_campaign_cost: float = 25.0
    max_intervention_attempts: int = 3
    intervention_cooldown_days: int = 30

    # ROI calculations
    min_acceptable_roi: float = 200.0
    roi_calculation_period_months: int = 12


@dataclass
class MonitoringConfig:
    """Model monitoring and alerting configuration."""
    # Performance monitoring
    performance_check_frequency_hours: int = 24
    data_drift_threshold: float = 0.1
    model_drift_threshold: float = 0.05

    # Alerting thresholds
    alert_f1_drop_threshold: float = 0.05
    alert_prediction_volume_change: float = 0.20
    alert_error_rate_threshold: float = 0.05

    # Retraining triggers
    retrain_performance_threshold: float = 0.10
    retrain_data_drift_threshold: float = 0.15
    max_days_without_retrain: int = 90


class ConfigurationManager:
    """Central configuration management system."""

    def __init__(self, environment: Environment = Environment.DEVELOPMENT, config_file: Optional[str] = None):
        self.environment = environment
        self.config_file = config_file
        self._config = self._load_configuration()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from file or environment variables."""
        config = {}

        # Load from configuration file if provided
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    config = yaml.safe_load(f)
                elif self.config_file.endswith('.json'):
                    config = json.load(f)

        # Override with environment-specific settings
        env_config = self._get_environment_config()
        config.update(env_config)

        # Override with environment variables
        env_vars = self._get_environment_variables()
        config.update(env_vars)

        return config

    def _get_environment_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        configs = {
            Environment.DEVELOPMENT: {
                'database': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'churn_dev',
                    'username': 'dev_user',
                    'password': 'dev_password'
                },
                'data_paths': {
                    'raw_data_path': 'data/raw/',
                    'processed_data_path': 'data/processed/',
                    'model_artifacts_path': 'models/',
                    'logs_path': 'logs/'
                },
                'model': {
                    'model_name': 'churn_predictor_dev',
                    'model_version': '1.0.0-dev'
                },
                'monitoring': {
                    'performance_check_frequency_hours': 1,
                    'alert_enabled': False
                }
            },
            Environment.TESTING: {
                'database': {
                    'host': 'test-db.internal',
                    'port': 5432,
                    'database': 'churn_test',
                    'username': 'test_user',
                    'password': 'test_password'
                },
                'data_paths': {
                    'raw_data_path': '/tmp/test_data/raw/',
                    'processed_data_path': '/tmp/test_data/processed/',
                    'model_artifacts_path': '/tmp/test_models/',
                    'logs_path': '/tmp/test_logs/'
                },
                'model': {
                    'model_name': 'churn_predictor_test',
                    'model_version': '1.0.0-test'
                }
            },
            Environment.STAGING: {
                'database': {
                    'host': 'staging-db.internal',
                    'port': 5432,
                    'database': 'churn_staging',
                    'username': 'staging_user',
                    'password': os.getenv('STAGING_DB_PASSWORD')
                },
                'data_paths': {
                    'raw_data_path': '/app/data/raw/',
                    'processed_data_path': '/app/data/processed/',
                    'model_artifacts_path': '/app/models/',
                    'logs_path': '/app/logs/'
                },
                'model': {
                    'model_name': 'churn_predictor_staging',
                    'model_version': '1.0.0-staging'
                }
            },
            Environment.PRODUCTION: {
                'database': {
                    'host': os.getenv('PROD_DB_HOST'),
                    'port': int(os.getenv('PROD_DB_PORT', 5432)),
                    'database': os.getenv('PROD_DB_NAME'),
                    'username': os.getenv('PROD_DB_USER'),
                    'password': os.getenv('PROD_DB_PASSWORD')
                },
                'data_paths': {
                    'raw_data_path': os.getenv('PROD_RAW_DATA_PATH', '/data/raw/'),
                    'processed_data_path': os.getenv('PROD_PROCESSED_DATA_PATH', '/data/processed/'),
                    'model_artifacts_path': os.getenv('PROD_MODEL_PATH', '/models/'),
                    'logs_path': os.getenv('PROD_LOGS_PATH', '/logs/')
                },
                'model': {
                    'model_name': 'churn_predictor_prod',
                    'model_version': os.getenv('MODEL_VERSION', '1.0.0')
                },
                'monitoring': {
                    'performance_check_frequency_hours': 6,
                    'alert_enabled': True
                }
            }
        }

        return configs.get(self.environment, {})

    def _get_environment_variables(self) -> Dict[str, Any]:
        """Extract configuration from environment variables."""
        env_vars = {}

        # Model configuration from environment
        if os.getenv('MODEL_NAME'):
            env_vars.setdefault('model', {})['model_name'] = os.getenv('MODEL_NAME')

        if os.getenv('TARGET_COLUMN'):
            env_vars.setdefault('model', {})['target_column'] = os.getenv('TARGET_COLUMN')

        # Business configuration from environment
        if os.getenv('HIGH_RISK_THRESHOLD'):
            env_vars.setdefault('business', {})['high_risk_threshold'] = float(os.getenv('HIGH_RISK_THRESHOLD'))

        if os.getenv('RETENTION_CAMPAIGN_COST'):
            env_vars.setdefault('business', {})['retention_campaign_cost'] = float(os.getenv('RETENTION_CAMPAIGN_COST'))

        return env_vars

    @property
    def database(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self._config.get('database', {})
        return DatabaseConfig(**db_config)

    @property
    def model(self) -> ModelConfig:
        """Get model configuration."""
        model_config = self._config.get('model', {})

        # Set defaults
        defaults = {
            'model_name': 'churn_predictor',
            'model_version': '1.0.0',
            'target_column': 'churn',
            'feature_columns': [],
            'categorical_features': ['gender', 'senior_citizen', 'partner', 'dependents',
                                   'phone_service', 'multiple_lines', 'internet_service',
                                   'online_security', 'online_backup', 'device_protection',
                                   'tech_support', 'streaming_tv', 'streaming_movies',
                                   'contract', 'paperless_billing', 'payment_method'],
            'numerical_features': ['tenure', 'monthly_charges', 'total_charges']
        }

        # Merge with provided config
        for key, value in defaults.items():
            model_config.setdefault(key, value)

        return ModelConfig(**model_config)

    @property
    def data(self) -> DataConfig:
        """Get data configuration."""
        data_config = self._config.get('data_paths', {})

        # Set defaults based on environment
        defaults = {
            'raw_data_path': 'data/raw/',
            'processed_data_path': 'data/processed/',
            'model_artifacts_path': 'models/',
            'logs_path': 'logs/'
        }

        for key, value in defaults.items():
            data_config.setdefault(key, value)

        return DataConfig(**data_config)

    @property
    def business(self) -> BusinessConfig:
        """Get business configuration."""
        business_config = self._config.get('business', {})
        return BusinessConfig(**business_config)

    @property
    def monitoring(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_config = self._config.get('monitoring', {})
        return MonitoringConfig(**monitoring_config)

    def get_model_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """Get hyperparameters for specific model type."""
        hyperparameters = {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_samples_split': 20,
                'min_samples_leaf': 20,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }

        # Override with custom hyperparameters if provided
        custom_params = self._config.get('hyperparameters', {}).get(model_type, {})
        if custom_params:
            hyperparameters[model_type].update(custom_params)

        return hyperparameters.get(model_type, {})

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of validation errors."""
        errors = []

        # Validate database configuration
        try:
            db_config = self.database
            if not all([db_config.host, db_config.database, db_config.username]):
                errors.append("Database configuration incomplete: missing host, database, or username")
        except Exception as e:
            errors.append(f"Database configuration error: {str(e)}")

        # Validate model configuration
        try:
            model_config = self.model
            if not model_config.target_column:
                errors.append("Model configuration error: target_column not specified")
            if not model_config.feature_columns:
                errors.append("Model configuration warning: feature_columns not specified")
        except Exception as e:
            errors.append(f"Model configuration error: {str(e)}")

        # Validate data paths
        try:
            data_config = self.data
            for path_name in ['raw_data_path', 'processed_data_path', 'model_artifacts_path', 'logs_path']:
                path = getattr(data_config, path_name)
                if self.environment == Environment.PRODUCTION:
                    if not os.path.exists(os.path.dirname(path)):
                        errors.append(f"Data path does not exist: {path}")
        except Exception as e:
            errors.append(f"Data configuration error: {str(e)}")

        return errors

    def create_directories(self) -> None:
        """Create necessary directories based on configuration."""
        data_config = self.data

        directories = [
            data_config.raw_data_path,
            data_config.processed_data_path,
            data_config.model_artifacts_path,
            data_config.logs_path
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def export_config(self, file_path: str) -> None:
        """Export current configuration to file."""
        with open(file_path, 'w') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                yaml.dump(self._config, f, default_flow_style=False)
            elif file_path.endswith('.json'):
                json.dump(self._config, f, indent=2)

    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigurationManager(environment={self.environment.value}, config_keys={list(self._config.keys())})"


# Global configuration instance
_config_manager = None

def get_config(environment: Environment = None, config_file: str = None) -> ConfigurationManager:
    """Get global configuration manager instance."""
    global _config_manager

    if _config_manager is None or environment is not None or config_file is not None:
        env = environment or Environment(os.getenv('ENVIRONMENT', 'development'))
        _config_manager = ConfigurationManager(environment=env, config_file=config_file)

    return _config_manager


def initialize_config(environment: str = None, config_file: str = None) -> ConfigurationManager:
    """Initialize configuration manager with specific environment."""
    env = Environment(environment) if environment else Environment.DEVELOPMENT
    config_manager = ConfigurationManager(environment=env, config_file=config_file)

    # Validate configuration
    errors = config_manager.validate_configuration()
    if errors:
        print("Configuration validation warnings/errors:")
        for error in errors:
            print(f"  - {error}")

    # Create necessary directories
    config_manager.create_directories()

    return config_manager


if __name__ == "__main__":
    # Example usage and testing
    print("Configuration Management System")
    print("=" * 40)

    # Test different environments
    for env in Environment:
        print(f"\n{env.value.upper()} Environment:")
        config = ConfigurationManager(environment=env)

        print(f"  Model: {config.model.model_name} v{config.model.model_version}")
        print(f"  Data Path: {config.data.processed_data_path}")
        print(f"  High Risk Threshold: {config.business.high_risk_threshold}")
        print(f"  Monitoring Frequency: {config.monitoring.performance_check_frequency_hours}h")

        # Validate configuration
        errors = config.validate_configuration()
        if errors:
            print(f"  Validation Errors: {len(errors)}")
        else:
            print("  ✅ Configuration Valid")

    # Test hyperparameter retrieval
    print(f"\nModel Hyperparameters:")
    config = get_config()
    for model_type in ['logistic_regression', 'random_forest', 'xgboost']:
        params = config.get_model_hyperparameters(model_type)
        print(f"  {model_type}: {len(params)} parameters")
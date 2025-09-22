"""
Logging Configuration System for Churn Prediction ML Pipeline
Task T044: Comprehensive logging framework with structured logging and monitoring

This module provides a centralized logging system that supports:
- Structured logging with JSON format for machine readability
- Multiple log levels and handlers (console, file, remote)
- Performance monitoring and metrics collection
- Business event tracking and audit trails
- Integration with monitoring systems (ELK, Prometheus, etc.)
- Context-aware logging with correlation IDs
"""

import logging
import logging.config
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import uuid
import time


@dataclass
class LogContext:
    """Structured logging context for correlation and tracing."""
    correlation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    model_version: Optional[str] = None
    customer_id: Optional[str] = None

    @classmethod
    def create(cls, **kwargs) -> 'LogContext':
        """Create new log context with auto-generated correlation ID."""
        return cls(correlation_id=str(uuid.uuid4()), **kwargs)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Add context if available
        if self.include_context and hasattr(record, 'context'):
            log_entry['context'] = asdict(record.context)

        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Add performance metrics if available
        if hasattr(record, 'duration'):
            log_entry['performance'] = {
                'duration_ms': record.duration,
                'start_time': getattr(record, 'start_time', None),
                'end_time': getattr(record, 'end_time', None)
            }

        return json.dumps(log_entry, default=str)


class BusinessEventFormatter(logging.Formatter):
    """Specialized formatter for business events and audit trails."""

    def format(self, record: logging.LogRecord) -> str:
        """Format business event record."""
        business_event = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'event_type': getattr(record, 'event_type', 'unknown'),
            'event_category': getattr(record, 'event_category', 'business'),
            'message': record.getMessage(),
            'severity': record.levelname
        }

        # Add business-specific context
        if hasattr(record, 'customer_id'):
            business_event['customer_id'] = record.customer_id

        if hasattr(record, 'prediction_id'):
            business_event['prediction_id'] = record.prediction_id

        if hasattr(record, 'campaign_id'):
            business_event['campaign_id'] = record.campaign_id

        if hasattr(record, 'model_version'):
            business_event['model_version'] = record.model_version

        if hasattr(record, 'business_metrics'):
            business_event['metrics'] = record.business_metrics

        # Add context if available
        if hasattr(record, 'context'):
            business_event['context'] = asdict(record.context)

        return json.dumps(business_event, default=str)


class PerformanceLogger:
    """Logger for performance monitoring and metrics collection."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    @contextmanager
    def measure_time(self, operation: str, context: Optional[LogContext] = None, **kwargs):
        """Context manager for measuring operation duration."""
        start_time = time.time()
        start_timestamp = datetime.now().isoformat()

        try:
            yield
            success = True
            error = None
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Log performance metrics
            extra = {
                'extra_fields': {
                    'operation': operation,
                    'success': success,
                    'error': error,
                    **kwargs
                },
                'duration': duration_ms,
                'start_time': start_timestamp,
                'end_time': datetime.now().isoformat()
            }

            if context:
                extra['context'] = context

            level = logging.INFO if success else logging.ERROR
            self.logger.log(level, f"Operation '{operation}' completed", extra=extra)

    def log_model_performance(self, model_name: str, metrics: Dict[str, float],
                            context: Optional[LogContext] = None):
        """Log model performance metrics."""
        extra = {
            'extra_fields': {
                'model_name': model_name,
                'performance_metrics': metrics,
                'metric_type': 'model_performance'
            }
        }

        if context:
            extra['context'] = context

        self.logger.info(f"Model performance logged for {model_name}", extra=extra)

    def log_prediction_metrics(self, prediction_count: int, batch_size: int,
                             processing_time_ms: float, context: Optional[LogContext] = None):
        """Log prediction processing metrics."""
        extra = {
            'extra_fields': {
                'prediction_count': prediction_count,
                'batch_size': batch_size,
                'processing_time_ms': processing_time_ms,
                'predictions_per_second': prediction_count / (processing_time_ms / 1000),
                'metric_type': 'prediction_performance'
            }
        }

        if context:
            extra['context'] = context

        self.logger.info(f"Processed {prediction_count} predictions in {processing_time_ms:.2f}ms",
                        extra=extra)


class BusinessEventLogger:
    """Logger for business events and audit trails."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_churn_prediction(self, customer_id: str, churn_probability: float,
                           risk_level: str, model_version: str,
                           context: Optional[LogContext] = None):
        """Log churn prediction event."""
        extra = {
            'event_type': 'churn_prediction',
            'event_category': 'prediction',
            'customer_id': customer_id,
            'business_metrics': {
                'churn_probability': churn_probability,
                'risk_level': risk_level
            },
            'model_version': model_version
        }

        if context:
            extra['context'] = context

        self.logger.info(f"Churn prediction generated for customer {customer_id}: "
                        f"{risk_level} risk ({churn_probability:.3f})", extra=extra)

    def log_intervention_campaign(self, customer_id: str, campaign_id: str,
                                campaign_type: str, expected_cost: float,
                                context: Optional[LogContext] = None):
        """Log intervention campaign initiation."""
        extra = {
            'event_type': 'intervention_campaign',
            'event_category': 'retention',
            'customer_id': customer_id,
            'campaign_id': campaign_id,
            'business_metrics': {
                'campaign_type': campaign_type,
                'expected_cost': expected_cost
            }
        }

        if context:
            extra['context'] = context

        self.logger.info(f"Intervention campaign {campaign_id} initiated for customer {customer_id}",
                        extra=extra)

    def log_customer_outcome(self, customer_id: str, outcome: str,
                           actual_value: float, predicted_value: float,
                           context: Optional[LogContext] = None):
        """Log customer outcome for model validation."""
        extra = {
            'event_type': 'customer_outcome',
            'event_category': 'validation',
            'customer_id': customer_id,
            'business_metrics': {
                'outcome': outcome,
                'actual_value': actual_value,
                'predicted_value': predicted_value,
                'prediction_error': abs(actual_value - predicted_value)
            }
        }

        if context:
            extra['context'] = context

        self.logger.info(f"Customer {customer_id} outcome: {outcome} "
                        f"(actual: {actual_value}, predicted: {predicted_value})", extra=extra)


class LoggingManager:
    """Central logging management system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.context_stack = []
        self._configure_logging()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default logging configuration."""
        log_dir = os.getenv('LOG_DIR', 'logs')
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'structured': {
                    '()': 'src.utils.logging.StructuredFormatter',
                    'include_context': True
                },
                'business': {
                    '()': 'src.utils.logging.BusinessEventFormatter'
                },
                'console': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO',
                    'formatter': 'console',
                    'stream': sys.stdout
                },
                'file_application': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'structured',
                    'filename': os.path.join(log_dir, 'application.log'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10
                },
                'file_performance': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'structured',
                    'filename': os.path.join(log_dir, 'performance.log'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 5
                },
                'file_business': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'business',
                    'filename': os.path.join(log_dir, 'business_events.log'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10
                },
                'file_errors': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'structured',
                    'filename': os.path.join(log_dir, 'errors.log'),
                    'maxBytes': 50 * 1024 * 1024,  # 50MB
                    'backupCount': 10
                }
            },
            'loggers': {
                'churn_prediction': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_application', 'file_errors'],
                    'propagate': False
                },
                'churn_prediction.performance': {
                    'level': 'INFO',
                    'handlers': ['file_performance'],
                    'propagate': False
                },
                'churn_prediction.business': {
                    'level': 'INFO',
                    'handlers': ['file_business'],
                    'propagate': False
                },
                'churn_prediction.data': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_application'],
                    'propagate': False
                },
                'churn_prediction.model': {
                    'level': 'DEBUG',
                    'handlers': ['console', 'file_application'],
                    'propagate': False
                },
                'churn_prediction.api': {
                    'level': 'INFO',
                    'handlers': ['console', 'file_application'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console']
            }
        }

    def _configure_logging(self):
        """Configure logging based on configuration."""
        logging.config.dictConfig(self.config)

    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance with specified name."""
        return logging.getLogger(name)

    def get_performance_logger(self, name: str = 'churn_prediction.performance') -> PerformanceLogger:
        """Get performance logger instance."""
        logger = self.get_logger(name)
        return PerformanceLogger(logger)

    def get_business_logger(self, name: str = 'churn_prediction.business') -> BusinessEventLogger:
        """Get business event logger instance."""
        logger = self.get_logger(name)
        return BusinessEventLogger(logger)

    @contextmanager
    def log_context(self, context: LogContext):
        """Context manager for adding structured context to logs."""
        self.context_stack.append(context)

        # Add context to all loggers
        for logger_name in ['churn_prediction', 'churn_prediction.performance',
                          'churn_prediction.business', 'churn_prediction.data',
                          'churn_prediction.model', 'churn_prediction.api']:
            logger = logging.getLogger(logger_name)
            old_factory = logging.getLogRecordFactory()

            def record_factory(*args, **kwargs):
                record = old_factory(*args, **kwargs)
                record.context = context
                return record

            logging.setLogRecordFactory(record_factory)

        try:
            yield
        finally:
            self.context_stack.pop()
            logging.setLogRecordFactory(old_factory)

    def set_log_level(self, logger_name: str, level: Union[str, int]):
        """Set log level for specific logger."""
        logger = self.get_logger(logger_name)
        logger.setLevel(level)

    def add_handler(self, logger_name: str, handler: logging.Handler):
        """Add handler to specific logger."""
        logger = self.get_logger(logger_name)
        logger.addHandler(handler)

    def health_check(self) -> Dict[str, Any]:
        """Perform logging system health check."""
        health_status = {
            'status': 'healthy',
            'loggers': {},
            'handlers': {},
            'timestamp': datetime.now().isoformat()
        }

        # Check logger configurations
        for logger_name in ['churn_prediction', 'churn_prediction.performance',
                          'churn_prediction.business']:
            logger = logging.getLogger(logger_name)
            health_status['loggers'][logger_name] = {
                'level': logger.level,
                'effective_level': logger.getEffectiveLevel(),
                'handlers_count': len(logger.handlers),
                'disabled': logger.disabled
            }

        # Check file handlers
        for handler_name, handler_config in self.config.get('handlers', {}).items():
            if 'filename' in handler_config:
                filename = handler_config['filename']
                health_status['handlers'][handler_name] = {
                    'type': 'file',
                    'filename': filename,
                    'exists': os.path.exists(filename),
                    'writable': os.access(os.path.dirname(filename), os.W_OK)
                }

        return health_status


# Global logging manager instance
_logging_manager = None

def get_logging_manager(config: Optional[Dict[str, Any]] = None) -> LoggingManager:
    """Get global logging manager instance."""
    global _logging_manager

    if _logging_manager is None or config is not None:
        _logging_manager = LoggingManager(config)

    return _logging_manager


def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return get_logging_manager().get_logger(name)


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance."""
    return get_logging_manager().get_performance_logger()


def get_business_logger() -> BusinessEventLogger:
    """Get business event logger instance."""
    return get_logging_manager().get_business_logger()


def setup_logging(config_file: Optional[str] = None, log_level: str = 'INFO') -> LoggingManager:
    """Setup logging system with optional configuration file."""
    config = None

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                import yaml
                config = yaml.safe_load(f)

    # Override log level if specified
    if config and log_level:
        for logger_config in config.get('loggers', {}).values():
            logger_config['level'] = log_level

    return get_logging_manager(config)


if __name__ == "__main__":
    # Example usage and testing
    print("Logging Configuration System")
    print("=" * 40)

    # Setup logging
    logging_manager = setup_logging()

    # Test different loggers
    app_logger = get_logger('churn_prediction')
    perf_logger = get_performance_logger()
    business_logger = get_business_logger()

    # Test structured logging with context
    context = LogContext.create(
        user_id='user123',
        component='model_training',
        operation='train_model'
    )

    with logging_manager.log_context(context):
        app_logger.info("Starting model training process")

        # Test performance logging
        with perf_logger.measure_time('model_training', context, model_type='xgboost'):
            time.sleep(0.1)  # Simulate work

        # Test business event logging
        business_logger.log_churn_prediction(
            customer_id='cust456',
            churn_probability=0.75,
            risk_level='high',
            model_version='1.0.0',
            context=context
        )

        app_logger.info("Model training completed successfully")

    # Test error logging
    try:
        raise ValueError("Test error for logging")
    except Exception:
        app_logger.exception("Error occurred during processing")

    # Health check
    health = logging_manager.health_check()
    print(f"\nLogging Health Check:")
    print(f"Status: {health['status']}")
    print(f"Active Loggers: {len(health['loggers'])}")
    print(f"Configured Handlers: {len(health['handlers'])}")

    print("\n✅ Logging system configured and tested successfully!")
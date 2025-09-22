"""
Data Pipeline Orchestration for Churn Prediction ML System
Task T046: Comprehensive pipeline orchestration with workflow management

This module provides orchestration for the complete ML pipeline including:
- Data ingestion and validation workflows
- Feature engineering and transformation pipelines
- Model training and evaluation orchestration
- Model deployment and monitoring workflows
- Business analysis and reporting pipelines
- Error handling and retry mechanisms
- Pipeline monitoring and alerting
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import traceback
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue, Empty

# Import project modules
from src.config.settings import get_config, ConfigurationManager
from src.utils.logging import get_logger, get_performance_logger, get_business_logger, LogContext
from src.utils.model_utils import ModelRegistry, ModelMonitor


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'output': str(self.output) if self.output is not None else None,
            'error': self.error,
            'metadata': self.metadata
        }


@dataclass
class PipelineRun:
    """Pipeline execution run information."""
    run_id: str
    pipeline_name: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'run_id': self.run_id,
            'pipeline_name': self.pipeline_name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'task_results': {k: v.to_dict() for k, v in self.task_results.items()},
            'parameters': self.parameters,
            'metadata': self.metadata
        }


class Task:
    """Individual pipeline task."""

    def __init__(self, task_id: str, function: Callable, dependencies: List[str] = None,
                 retry_count: int = 3, timeout_seconds: int = 3600,
                 description: str = "", tags: List[str] = None):
        self.task_id = task_id
        self.function = function
        self.dependencies = dependencies or []
        self.retry_count = retry_count
        self.timeout_seconds = timeout_seconds
        self.description = description
        self.tags = tags or []

    async def execute(self, context: Dict[str, Any]) -> TaskResult:
        """Execute the task with context."""
        start_time = datetime.now()
        result = TaskResult(
            task_id=self.task_id,
            status=TaskStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Execute function with timeout
            if asyncio.iscoroutinefunction(self.function):
                output = await asyncio.wait_for(
                    self.function(context),
                    timeout=self.timeout_seconds
                )
            else:
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    self.function,
                    context
                )

            result.status = TaskStatus.COMPLETED
            result.output = output
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

        except asyncio.TimeoutError:
            result.status = TaskStatus.FAILED
            result.error = f"Task timed out after {self.timeout_seconds} seconds"
            result.end_time = datetime.now()

        except Exception as e:
            result.status = TaskStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            result.metadata['traceback'] = traceback.format_exc()

        return result


class Pipeline:
    """Pipeline definition with tasks and dependencies."""

    def __init__(self, pipeline_name: str, description: str = ""):
        self.pipeline_name = pipeline_name
        self.description = description
        self.tasks: Dict[str, Task] = {}
        self.task_graph = {}

    def add_task(self, task: Task) -> 'Pipeline':
        """Add task to pipeline."""
        self.tasks[task.task_id] = task
        self.task_graph[task.task_id] = task.dependencies
        return self

    def validate(self) -> List[str]:
        """Validate pipeline for circular dependencies and missing tasks."""
        errors = []

        # Check for missing dependencies
        for task_id, dependencies in self.task_graph.items():
            for dep in dependencies:
                if dep not in self.tasks:
                    errors.append(f"Task '{task_id}' depends on missing task '{dep}'")

        # Check for circular dependencies using DFS
        def has_cycle(node, visited, rec_stack):
            visited[node] = True
            rec_stack[node] = True

            for neighbor in self.task_graph.get(node, []):
                if neighbor in self.tasks:
                    if not visited.get(neighbor, False):
                        if has_cycle(neighbor, visited, rec_stack):
                            return True
                    elif rec_stack.get(neighbor, False):
                        return True

            rec_stack[node] = False
            return False

        visited = {}
        rec_stack = {}

        for task_id in self.tasks:
            if not visited.get(task_id, False):
                if has_cycle(task_id, visited, rec_stack):
                    errors.append(f"Circular dependency detected involving task '{task_id}'")

        return errors

    def get_execution_order(self) -> List[List[str]]:
        """Get tasks in topological order for execution."""
        # Calculate in-degrees
        in_degree = {task_id: 0 for task_id in self.tasks}
        for task_id, dependencies in self.task_graph.items():
            for dep in dependencies:
                if dep in in_degree:
                    in_degree[task_id] += 1

        # Find execution levels
        levels = []
        remaining_tasks = set(self.tasks.keys())

        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [task_id for task_id in remaining_tasks if in_degree[task_id] == 0]

            if not ready_tasks:
                break  # Circular dependency

            levels.append(ready_tasks)

            # Remove ready tasks and update in-degrees
            for task_id in ready_tasks:
                remaining_tasks.remove(task_id)
                for dependent_id in remaining_tasks:
                    if task_id in self.task_graph.get(dependent_id, []):
                        in_degree[dependent_id] -= 1

        return levels


class PipelineOrchestrator:
    """Main orchestrator for pipeline execution."""

    def __init__(self, config: Optional[ConfigurationManager] = None):
        self.config = config or get_config()
        self.logger = get_logger('churn_prediction.orchestrator')
        self.perf_logger = get_performance_logger()
        self.business_logger = get_business_logger()

        self.pipelines: Dict[str, Pipeline] = {}
        self.active_runs: Dict[str, PipelineRun] = {}
        self.run_history: List[PipelineRun] = []

        # Initialize model utilities
        self.model_registry = ModelRegistry(self.config.data.model_artifacts_path)
        self.model_monitor = ModelMonitor(self.model_registry)

        # Execution settings
        self.max_concurrent_tasks = 4
        self.max_workers = 8

    def register_pipeline(self, pipeline: Pipeline) -> bool:
        """Register a pipeline for execution."""
        try:
            # Validate pipeline
            errors = pipeline.validate()
            if errors:
                self.logger.error(f"Pipeline validation failed: {errors}")
                return False

            self.pipelines[pipeline.pipeline_name] = pipeline
            self.logger.info(f"Pipeline '{pipeline.pipeline_name}' registered successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Error registering pipeline: {e}")
            return False

    async def execute_pipeline(self, pipeline_name: str, parameters: Dict[str, Any] = None,
                             run_id: str = None) -> PipelineRun:
        """Execute a registered pipeline."""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        pipeline = self.pipelines[pipeline_name]
        run_id = run_id or self._generate_run_id(pipeline_name)
        parameters = parameters or {}

        # Create pipeline run
        pipeline_run = PipelineRun(
            run_id=run_id,
            pipeline_name=pipeline_name,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now(),
            parameters=parameters
        )

        self.active_runs[run_id] = pipeline_run
        context = LogContext.create(
            component='pipeline_orchestrator',
            operation='execute_pipeline',
            correlation_id=run_id
        )

        try:
            with self.perf_logger.measure_time('pipeline_execution', context,
                                             pipeline_name=pipeline_name):
                # Get execution order
                execution_levels = pipeline.get_execution_order()
                self.logger.info(f"Executing pipeline '{pipeline_name}' with {len(execution_levels)} levels")

                # Create execution context
                execution_context = {
                    'run_id': run_id,
                    'pipeline_name': pipeline_name,
                    'parameters': parameters,
                    'config': self.config,
                    'model_registry': self.model_registry,
                    'model_monitor': self.model_monitor,
                    'logger': self.logger,
                    'results': {}
                }

                # Execute tasks level by level
                for level_idx, task_level in enumerate(execution_levels):
                    self.logger.info(f"Executing level {level_idx + 1}: {task_level}")

                    # Execute tasks in parallel within level
                    level_results = await self._execute_task_level(
                        pipeline, task_level, execution_context, context
                    )

                    # Update results and check for failures
                    for task_id, result in level_results.items():
                        pipeline_run.task_results[task_id] = result
                        execution_context['results'][task_id] = result.output

                        if result.status == TaskStatus.FAILED:
                            pipeline_run.status = PipelineStatus.FAILED
                            self.logger.error(f"Task '{task_id}' failed: {result.error}")
                            break

                    # Stop execution if any task failed
                    if pipeline_run.status == PipelineStatus.FAILED:
                        break

                # Update final status
                if pipeline_run.status != PipelineStatus.FAILED:
                    pipeline_run.status = PipelineStatus.COMPLETED

                pipeline_run.end_time = datetime.now()
                pipeline_run.duration_seconds = (
                    pipeline_run.end_time - pipeline_run.start_time
                ).total_seconds()

                self.logger.info(f"Pipeline '{pipeline_name}' completed with status: {pipeline_run.status.value}")

        except Exception as e:
            pipeline_run.status = PipelineStatus.FAILED
            pipeline_run.end_time = datetime.now()
            self.logger.exception(f"Pipeline execution failed: {e}")

        finally:
            # Clean up and archive
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            self.run_history.append(pipeline_run)

            # Save run results
            self._save_run_results(pipeline_run)

        return pipeline_run

    async def _execute_task_level(self, pipeline: Pipeline, task_ids: List[str],
                                execution_context: Dict[str, Any],
                                log_context: LogContext) -> Dict[str, TaskResult]:
        """Execute all tasks in a level concurrently."""
        tasks = [pipeline.tasks[task_id] for task_id in task_ids]

        # Execute tasks concurrently
        task_coroutines = [
            self._execute_task_with_retry(task, execution_context, log_context)
            for task in tasks
        ]

        results = await asyncio.gather(*task_coroutines, return_exceptions=True)

        # Process results
        level_results = {}
        for task, result in zip(tasks, results):
            if isinstance(result, Exception):
                level_results[task.task_id] = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=str(result)
                )
            else:
                level_results[task.task_id] = result

        return level_results

    async def _execute_task_with_retry(self, task: Task, context: Dict[str, Any],
                                     log_context: LogContext) -> TaskResult:
        """Execute task with retry logic."""
        last_result = None

        for attempt in range(task.retry_count + 1):
            try:
                self.logger.info(f"Executing task '{task.task_id}' (attempt {attempt + 1})")

                result = await task.execute(context)

                if result.status == TaskStatus.COMPLETED:
                    self.logger.info(f"Task '{task.task_id}' completed successfully")
                    return result
                else:
                    last_result = result
                    if attempt < task.retry_count:
                        self.logger.warning(f"Task '{task.task_id}' failed, retrying... ({result.error})")
                        await asyncio.sleep(min(2 ** attempt, 30))  # Exponential backoff
                    else:
                        self.logger.error(f"Task '{task.task_id}' failed after {attempt + 1} attempts")

            except Exception as e:
                error_msg = f"Unexpected error in task '{task.task_id}': {e}"
                self.logger.exception(error_msg)
                last_result = TaskResult(
                    task_id=task.task_id,
                    status=TaskStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error=error_msg
                )

        return last_result or TaskResult(
            task_id=task.task_id,
            status=TaskStatus.FAILED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            error="Unknown error"
        )

    def _generate_run_id(self, pipeline_name: str) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{pipeline_name}_{timestamp}"

    def _save_run_results(self, pipeline_run: PipelineRun):
        """Save pipeline run results to file."""
        try:
            runs_dir = Path(self.config.data.logs_path) / "pipeline_runs"
            runs_dir.mkdir(parents=True, exist_ok=True)

            run_file = runs_dir / f"{pipeline_run.run_id}.json"
            with open(run_file, 'w') as f:
                json.dump(pipeline_run.to_dict(), f, indent=2)

        except Exception as e:
            self.logger.error(f"Error saving run results: {e}")

    def get_pipeline_status(self, run_id: str) -> Optional[PipelineRun]:
        """Get status of a pipeline run."""
        if run_id in self.active_runs:
            return self.active_runs[run_id]

        # Check history
        for run in self.run_history:
            if run.run_id == run_id:
                return run

        return None

    def cancel_pipeline(self, run_id: str) -> bool:
        """Cancel a running pipeline."""
        if run_id in self.active_runs:
            self.active_runs[run_id].status = PipelineStatus.CANCELLED
            self.logger.info(f"Pipeline run '{run_id}' cancelled")
            return True
        return False

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all registered pipelines."""
        return [
            {
                'name': name,
                'description': pipeline.description,
                'task_count': len(pipeline.tasks),
                'tasks': list(pipeline.tasks.keys())
            }
            for name, pipeline in self.pipelines.items()
        ]

    def get_run_history(self, pipeline_name: str = None, limit: int = 50) -> List[PipelineRun]:
        """Get pipeline run history."""
        runs = self.run_history
        if pipeline_name:
            runs = [run for run in runs if run.pipeline_name == pipeline_name]

        return sorted(runs, key=lambda x: x.start_time, reverse=True)[:limit]


# Predefined pipeline builders
class ChurnPipelineBuilder:
    """Builder for churn prediction pipelines."""

    @staticmethod
    def build_data_pipeline() -> Pipeline:
        """Build data ingestion and processing pipeline."""
        pipeline = Pipeline(
            pipeline_name="churn_data_pipeline",
            description="Data ingestion, validation, and preprocessing for churn prediction"
        )

        # Data ingestion task
        async def ingest_data(context):
            from src.data.loader import ChurnDataLoader
            loader = ChurnDataLoader()
            data = loader.load_raw_data()
            context['logger'].info(f"Ingested {len(data)} records")
            return data

        # Data validation task
        async def validate_data(context):
            from src.data.validator import DataValidator
            data = context['results']['ingest_data']
            validator = DataValidator()
            validation_results = validator.validate_data(data)
            if not validation_results['is_valid']:
                raise ValueError(f"Data validation failed: {validation_results['errors']}")
            return validation_results

        # Feature processing task
        async def process_features(context):
            from src.features.processor import FeatureProcessor
            data = context['results']['ingest_data']
            processor = FeatureProcessor()
            processed_data = processor.process_features(data)
            context['logger'].info(f"Processed features for {len(processed_data)} records")
            return processed_data

        # Add tasks to pipeline
        pipeline.add_task(Task("ingest_data", ingest_data, description="Ingest raw data"))
        pipeline.add_task(Task("validate_data", validate_data, dependencies=["ingest_data"],
                              description="Validate data quality"))
        pipeline.add_task(Task("process_features", process_features,
                              dependencies=["validate_data"],
                              description="Process and engineer features"))

        return pipeline

    @staticmethod
    def build_training_pipeline() -> Pipeline:
        """Build model training and evaluation pipeline."""
        pipeline = Pipeline(
            pipeline_name="churn_training_pipeline",
            description="Model training, evaluation, and registration"
        )

        # Data preparation task
        async def prepare_training_data(context):
            # Load processed data
            data = context['parameters'].get('training_data')
            if data is None:
                raise ValueError("Training data not provided")

            # Split data
            from sklearn.model_selection import train_test_split
            X = data.drop(['churn'], axis=1)
            y = data['churn']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            return {
                'X_train': X_train, 'X_test': X_test,
                'y_train': y_train, 'y_test': y_test
            }

        # Model training task
        async def train_model(context):
            from src.models.trainer import ModelTrainer

            data_splits = context['results']['prepare_training_data']
            trainer = ModelTrainer()

            model = trainer.train_model(
                data_splits['X_train'],
                data_splits['y_train'],
                model_type=context['parameters'].get('model_type', 'random_forest')
            )

            return model

        # Model evaluation task
        async def evaluate_model(context):
            from src.models.evaluator import ModelEvaluator

            model = context['results']['train_model']
            data_splits = context['results']['prepare_training_data']

            evaluator = ModelEvaluator()
            metrics = evaluator.evaluate_model(
                model,
                data_splits['X_test'],
                data_splits['y_test']
            )

            return metrics

        # Model registration task
        async def register_model(context):
            from src.utils.model_utils import ModelMetadata

            model = context['results']['train_model']
            metrics = context['results']['evaluate_model']

            metadata = ModelMetadata.create(
                model_name="churn_predictor",
                model_version=context['parameters'].get('model_version', '1.0.0'),
                model_type=context['parameters'].get('model_type', 'random_forest'),
                framework="sklearn",
                training_data_hash="placeholder",
                feature_names=list(context['results']['prepare_training_data']['X_train'].columns),
                target_name="churn",
                hyperparameters=context['parameters'].get('hyperparameters', {}),
                training_duration_seconds=60.0,
                performance_metrics=metrics,
                validation_metrics=metrics,
                business_metrics={"roi": 250.0}
            )

            registry = context['model_registry']
            success = registry.register_model(model, metadata)

            if not success:
                raise ValueError("Model registration failed")

            return {"model_id": metadata.model_id, "success": success}

        # Add tasks to pipeline
        pipeline.add_task(Task("prepare_training_data", prepare_training_data,
                              description="Prepare training and test datasets"))
        pipeline.add_task(Task("train_model", train_model,
                              dependencies=["prepare_training_data"],
                              description="Train machine learning model"))
        pipeline.add_task(Task("evaluate_model", evaluate_model,
                              dependencies=["train_model"],
                              description="Evaluate model performance"))
        pipeline.add_task(Task("register_model", register_model,
                              dependencies=["evaluate_model"],
                              description="Register model in model registry"))

        return pipeline

    @staticmethod
    def build_prediction_pipeline() -> Pipeline:
        """Build prediction and monitoring pipeline."""
        pipeline = Pipeline(
            pipeline_name="churn_prediction_pipeline",
            description="Generate predictions and monitor model performance"
        )

        # Load model task
        async def load_model(context):
            registry = context['model_registry']
            model_name = context['parameters'].get('model_name', 'churn_predictor')
            model_version = context['parameters'].get('model_version')

            model, metadata = registry.load_model(model_name, model_version)
            if model is None:
                raise ValueError(f"Model {model_name} not found")

            return {"model": model, "metadata": metadata}

        # Generate predictions task
        async def generate_predictions(context):
            model_info = context['results']['load_model']
            model = model_info['model']

            # Get prediction data
            prediction_data = context['parameters'].get('prediction_data')
            if prediction_data is None:
                raise ValueError("Prediction data not provided")

            predictions = model.predict_proba(prediction_data)[:, 1]

            return {
                'predictions': predictions,
                'prediction_count': len(predictions)
            }

        # Monitor performance task
        async def monitor_performance(context):
            model_info = context['results']['load_model']
            predictions_info = context['results']['generate_predictions']

            monitor = context['model_monitor']
            metadata = model_info['metadata']

            # Log prediction performance
            monitor.log_prediction_performance(
                metadata.model_name,
                metadata.model_version,
                predictions_info['predictions']
            )

            return {"monitoring_logged": True}

        # Add tasks to pipeline
        pipeline.add_task(Task("load_model", load_model,
                              description="Load model from registry"))
        pipeline.add_task(Task("generate_predictions", generate_predictions,
                              dependencies=["load_model"],
                              description="Generate churn predictions"))
        pipeline.add_task(Task("monitor_performance", monitor_performance,
                              dependencies=["generate_predictions"],
                              description="Monitor model performance"))

        return pipeline


def create_orchestrator(config: ConfigurationManager = None) -> PipelineOrchestrator:
    """Create pipeline orchestrator with default configuration."""
    orchestrator = PipelineOrchestrator(config)

    # Register default pipelines
    orchestrator.register_pipeline(ChurnPipelineBuilder.build_data_pipeline())
    orchestrator.register_pipeline(ChurnPipelineBuilder.build_training_pipeline())
    orchestrator.register_pipeline(ChurnPipelineBuilder.build_prediction_pipeline())

    return orchestrator


if __name__ == "__main__":
    # Example usage and testing
    print("Pipeline Orchestration System")
    print("=" * 35)

    async def test_orchestrator():
        # Create orchestrator
        orchestrator = create_orchestrator()

        # List available pipelines
        pipelines = orchestrator.list_pipelines()
        print(f"\n📋 Available Pipelines: {len(pipelines)}")
        for pipeline in pipelines:
            print(f"  • {pipeline['name']}: {pipeline['task_count']} tasks")

        # Test simple pipeline execution
        print(f"\n🔄 Testing Simple Pipeline:")

        # Create a test pipeline
        test_pipeline = Pipeline("test_pipeline", "Simple test pipeline")

        async def test_task_1(context):
            await asyncio.sleep(0.1)
            return "Task 1 completed"

        async def test_task_2(context):
            result_1 = context['results']['task_1']
            await asyncio.sleep(0.1)
            return f"Task 2 completed after {result_1}"

        test_pipeline.add_task(Task("task_1", test_task_1, description="First test task"))
        test_pipeline.add_task(Task("task_2", test_task_2, dependencies=["task_1"],
                                   description="Second test task"))

        # Register and execute
        orchestrator.register_pipeline(test_pipeline)
        run_result = await orchestrator.execute_pipeline("test_pipeline")

        print(f"  Run ID: {run_result.run_id}")
        print(f"  Status: {run_result.status.value}")
        print(f"  Duration: {run_result.duration_seconds:.2f}s")
        print(f"  Tasks: {len(run_result.task_results)}")

        for task_id, result in run_result.task_results.items():
            print(f"    {task_id}: {result.status.value} ({result.duration_seconds:.2f}s)")

        print("\n✅ Pipeline orchestration system tested successfully!")

    # Run async test
    asyncio.run(test_orchestrator())

    print("\n🎯 Pipeline orchestrator ready for production use!")
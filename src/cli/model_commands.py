"""
CLI Commands for Model Training in Churn Prediction ML Pipeline
Task T048: Command-line interface for model operations

This module provides CLI commands for:
- Model training with various algorithms
- Hyperparameter optimization and tuning
- Model evaluation and comparison
- Model deployment and versioning
- Model monitoring and performance tracking
- A/B testing and model rollback
"""

import click
import sys
import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
import joblib
import numpy as np

# Import project modules
from src.config.settings import get_config, Environment, initialize_config
from src.utils.logging import setup_logging, get_logger, LogContext
from src.utils.model_utils import ModelRegistry, ModelMonitor, ModelMetadata, ModelStatus
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator
from src.models.comparison import ModelComparison
from src.models.tuning import HyperparameterTuner
from src.pipeline.orchestrator import create_orchestrator


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              default='development', help='Environment configuration')
@click.option('--log-level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def model(ctx, config, environment, log_level, verbose):
    """Model training and management commands for churn prediction pipeline."""

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Initialize configuration
    ctx.obj['config'] = initialize_config(environment, config)

    # Setup logging
    setup_logging(log_level=log_level)
    ctx.obj['logger'] = get_logger('churn_prediction.cli.model')

    # Initialize model utilities
    config_obj = ctx.obj['config']
    ctx.obj['model_registry'] = ModelRegistry(config_obj.data.model_artifacts_path)
    ctx.obj['model_monitor'] = ModelMonitor(ctx.obj['model_registry'])

    # Set verbose mode
    ctx.obj['verbose'] = verbose

    if verbose:
        click.echo(f"🔧 Environment: {environment}")
        click.echo(f"📋 Config: {config or 'default'}")
        click.echo(f"📊 Log Level: {log_level}")
        click.echo(f"🗄️ Model Registry: {config_obj.data.model_artifacts_path}")


@model.command()
@click.option('--data-path', '-d', required=True, help='Training data file path')
@click.option('--model-type', '-t', default='random_forest',
              type=click.Choice(['logistic_regression', 'random_forest', 'gradient_boosting',
                               'xgboost', 'lightgbm', 'neural_network', 'ensemble']),
              help='Model type to train')
@click.option('--model-name', '-n', default='churn_predictor', help='Model name')
@click.option('--model-version', '-v', help='Model version (auto-generated if not specified)')
@click.option('--test-size', type=float, default=0.2, help='Test set size (0.0-1.0)')
@click.option('--cv-folds', type=int, default=5, help='Cross-validation folds')
@click.option('--hyperparams', help='Hyperparameters as JSON string')
@click.option('--save-model', is_flag=True, help='Save trained model to registry')
@click.option('--output-path', '-o', help='Output path for model artifacts')
@click.pass_context
def train(ctx, data_path, model_type, model_name, model_version, test_size, cv_folds,
          hyperparams, save_model, output_path):
    """Train a machine learning model for churn prediction."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🚀 Starting model training: {model_type}")

        # Load training data
        if not Path(data_path).exists():
            click.echo(f"❌ Error: Data file {data_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(data_path)
        click.echo(f"📊 Loaded {len(data)} records for training")

        # Validate required columns
        required_columns = ['churn']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            click.echo(f"❌ Error: Missing required columns: {missing_columns}", err=True)
            sys.exit(1)

        # Prepare features and target
        target_column = 'churn'
        feature_columns = [col for col in data.columns if col != target_column]

        X = data[feature_columns]
        y = data[target_column]

        click.echo(f"🎯 Target: {target_column}")
        click.echo(f"📋 Features: {len(feature_columns)} columns")
        click.echo(f"⚖️ Class distribution: {y.value_counts().to_dict()}")

        # Parse hyperparameters
        model_hyperparams = {}
        if hyperparams:
            try:
                model_hyperparams = json.loads(hyperparams)
                if verbose:
                    click.echo(f"🔧 Custom hyperparameters: {model_hyperparams}")
            except json.JSONDecodeError as e:
                click.echo(f"❌ Error parsing hyperparameters: {e}", err=True)
                sys.exit(1)

        # Initialize trainer
        trainer = ModelTrainer()

        # Train model
        with click.progressbar(length=100, label='Training model') as bar:
            def progress_callback(progress):
                bar.update(progress - bar.pos)

            training_results = trainer.train_model(
                X, y,
                model_type=model_type,
                test_size=test_size,
                cv_folds=cv_folds,
                hyperparameters=model_hyperparams,
                progress_callback=progress_callback
            )

        model = training_results['model']
        metrics = training_results['metrics']
        training_time = training_results.get('training_time', 0.0)

        # Display results
        click.echo("\n✅ Training completed successfully!")
        click.echo(f"⏱️ Training time: {training_time:.2f} seconds")
        click.echo("\n📊 Model Performance:")

        for metric_name, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {metric_name}: {value:.4f}")
            else:
                click.echo(f"  {metric_name}: {value}")

        # Generate model version if not specified
        if not model_version:
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create model metadata
        metadata = ModelMetadata.create(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework="sklearn",
            training_data_hash=str(hash(data_path)),
            feature_names=feature_columns,
            target_name=target_column,
            hyperparameters=model_hyperparams,
            training_duration_seconds=training_time,
            performance_metrics=metrics,
            validation_metrics=metrics,
            business_metrics={"training_records": len(data)}
        )

        # Save model to registry if requested
        if save_model:
            click.echo(f"\n💾 Saving model to registry...")
            success = model_registry.register_model(model, metadata)

            if success:
                click.echo(f"✅ Model registered successfully")
                click.echo(f"🆔 Model ID: {metadata.model_id}")
                click.echo(f"📦 Version: {model_version}")
            else:
                click.echo(f"❌ Error: Failed to register model", err=True)

        # Save model artifacts if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save model
            model_file = output_path / f"{model_name}_v{model_version}.joblib"
            joblib.dump(model, model_file)

            # Save metadata
            metadata_file = output_path / f"{model_name}_v{model_version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)

            # Save metrics
            metrics_file = output_path / f"{model_name}_v{model_version}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

            click.echo(f"📁 Model artifacts saved to {output_path}")

        logger.info(f"Model training completed: {model_name} v{model_version}")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--data-path', '-d', required=True, help='Training data file path')
@click.option('--model-type', '-t', default='random_forest',
              type=click.Choice(['logistic_regression', 'random_forest', 'gradient_boosting',
                               'xgboost', 'lightgbm', 'neural_network']),
              help='Model type to optimize')
@click.option('--optimization-method', '-m', default='grid_search',
              type=click.Choice(['grid_search', 'random_search', 'bayesian']),
              help='Optimization method')
@click.option('--n-trials', type=int, default=50, help='Number of optimization trials')
@click.option('--cv-folds', type=int, default=5, help='Cross-validation folds')
@click.option('--metric', default='f1_score',
              type=click.Choice(['accuracy', 'precision', 'recall', 'f1_score', 'auc']),
              help='Optimization metric')
@click.option('--save-best', is_flag=True, help='Save best model to registry')
@click.option('--output-path', '-o', help='Output path for optimization results')
@click.pass_context
def optimize(ctx, data_path, model_type, optimization_method, n_trials, cv_folds,
             metric, save_best, output_path):
    """Optimize model hyperparameters using various methods."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🔍 Starting hyperparameter optimization: {model_type}")

        # Load training data
        if not Path(data_path).exists():
            click.echo(f"❌ Error: Data file {data_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(data_path)
        click.echo(f"📊 Loaded {len(data)} records for optimization")

        # Prepare features and target
        target_column = 'churn'
        feature_columns = [col for col in data.columns if col != target_column]

        X = data[feature_columns]
        y = data[target_column]

        # Initialize tuner
        tuner = HyperparameterTuner()

        # Run optimization
        click.echo(f"🔧 Optimization method: {optimization_method}")
        click.echo(f"🎯 Target metric: {metric}")
        click.echo(f"🔢 Number of trials: {n_trials}")

        with click.progressbar(length=n_trials, label='Optimizing hyperparameters') as bar:
            def progress_callback(trial_num, best_score):
                bar.update(1)
                if verbose and trial_num % 10 == 0:
                    click.echo(f"  Trial {trial_num}: Best {metric} = {best_score:.4f}")

            if optimization_method == 'grid_search':
                results = tuner.grid_search_optimize(
                    X, y, model_type, metric, cv_folds, progress_callback
                )
            elif optimization_method == 'random_search':
                results = tuner.random_search_optimize(
                    X, y, model_type, metric, cv_folds, n_trials, progress_callback
                )
            elif optimization_method == 'bayesian':
                results = tuner.bayesian_optimize(
                    X, y, model_type, metric, cv_folds, n_trials, progress_callback
                )

        # Display results
        click.echo("\n✅ Optimization completed!")
        click.echo(f"🏆 Best {metric}: {results['best_score']:.4f}")
        click.echo(f"⏱️ Optimization time: {results.get('optimization_time', 0):.2f} seconds")

        click.echo("\n🔧 Best Hyperparameters:")
        for param, value in results['best_params'].items():
            click.echo(f"  {param}: {value}")

        # Train final model with best parameters
        if save_best:
            click.echo(f"\n🚀 Training final model with best parameters...")

            trainer = ModelTrainer()
            training_results = trainer.train_model(
                X, y,
                model_type=model_type,
                hyperparameters=results['best_params']
            )

            model = training_results['model']
            metrics = training_results['metrics']

            # Create model metadata
            model_version = datetime.now().strftime("%Y%m%d_%H%M%S") + "_optimized"
            metadata = ModelMetadata.create(
                model_name=f"churn_predictor_{model_type}",
                model_version=model_version,
                model_type=model_type,
                framework="sklearn",
                training_data_hash=str(hash(data_path)),
                feature_names=feature_columns,
                target_name=target_column,
                hyperparameters=results['best_params'],
                training_duration_seconds=results.get('optimization_time', 0),
                performance_metrics=metrics,
                validation_metrics=metrics,
                business_metrics={
                    "optimization_method": optimization_method,
                    "optimization_trials": n_trials,
                    "best_score": results['best_score']
                }
            )

            # Save to registry
            success = model_registry.register_model(model, metadata)
            if success:
                click.echo(f"✅ Optimized model registered: {metadata.model_id}")
            else:
                click.echo(f"❌ Error: Failed to register optimized model", err=True)

        # Save optimization results
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            results_file = output_path / f"optimization_results_{model_type}_{optimization_method}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            click.echo(f"📄 Optimization results saved to {results_file}")

        logger.info(f"Hyperparameter optimization completed: {results['best_score']:.4f} {metric}")

    except Exception as e:
        logger.error(f"Hyperparameter optimization failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--model-name', '-n', required=True, help='Model name to evaluate')
@click.option('--model-version', '-v', help='Model version (latest if not specified)')
@click.option('--test-data', '-d', required=True, help='Test data file path')
@click.option('--metrics', '-m', multiple=True,
              default=['accuracy', 'precision', 'recall', 'f1_score', 'auc'],
              help='Evaluation metrics')
@click.option('--output-path', '-o', help='Output path for evaluation report')
@click.option('--generate-plots', is_flag=True, help='Generate evaluation plots')
@click.pass_context
def evaluate(ctx, model_name, model_version, test_data, metrics, output_path, generate_plots):
    """Evaluate a trained model on test data."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📊 Starting model evaluation: {model_name}")

        # Load model from registry
        model, metadata = model_registry.load_model(model_name, model_version)
        if model is None:
            click.echo(f"❌ Error: Model {model_name} not found", err=True)
            sys.exit(1)

        click.echo(f"✅ Loaded model: {metadata.model_name} v{metadata.model_version}")

        # Load test data
        if not Path(test_data).exists():
            click.echo(f"❌ Error: Test data file {test_data} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(test_data)
        click.echo(f"📊 Loaded {len(data)} test records")

        # Prepare features and target
        target_column = metadata.target_name
        feature_columns = metadata.feature_names

        # Validate features
        missing_features = set(feature_columns) - set(data.columns)
        if missing_features:
            click.echo(f"❌ Error: Missing features in test data: {missing_features}", err=True)
            sys.exit(1)

        X_test = data[feature_columns]
        y_test = data[target_column] if target_column in data.columns else None

        if y_test is None:
            click.echo(f"⚠️ Warning: No target column '{target_column}' found in test data")
            click.echo(f"🔮 Generating predictions only...")

        # Initialize evaluator
        evaluator = ModelEvaluator()

        # Evaluate model
        if y_test is not None:
            evaluation_results = evaluator.evaluate_model(
                model, X_test, y_test, metrics=list(metrics)
            )

            # Display results
            click.echo("\n📊 Evaluation Results:")
            for metric_name, value in evaluation_results['metrics'].items():
                if isinstance(value, float):
                    click.echo(f"  {metric_name}: {value:.4f}")
                else:
                    click.echo(f"  {metric_name}: {value}")

            # Display confusion matrix if available
            if 'confusion_matrix' in evaluation_results:
                cm = evaluation_results['confusion_matrix']
                click.echo(f"\n📋 Confusion Matrix:")
                click.echo(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
                click.echo(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")

        else:
            # Just generate predictions
            predictions = model.predict_proba(X_test)[:, 1]
            evaluation_results = {
                'predictions': predictions.tolist(),
                'prediction_statistics': {
                    'mean': float(predictions.mean()),
                    'std': float(predictions.std()),
                    'min': float(predictions.min()),
                    'max': float(predictions.max())
                }
            }

            click.echo(f"\n🔮 Generated {len(predictions)} predictions")
            click.echo(f"📊 Prediction Statistics:")
            for stat, value in evaluation_results['prediction_statistics'].items():
                click.echo(f"  {stat}: {value:.4f}")

        # Generate plots if requested
        if generate_plots and y_test is not None:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Model Evaluation: {model_name} v{metadata.model_version}')

                # ROC Curve
                from sklearn.metrics import roc_curve, auc
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                axes[0, 0].plot([0, 1], [0, 1], 'k--')
                axes[0, 0].set_xlabel('False Positive Rate')
                axes[0, 0].set_ylabel('True Positive Rate')
                axes[0, 0].set_title('ROC Curve')
                axes[0, 0].legend()

                # Prediction distribution
                axes[0, 1].hist(y_pred_proba, bins=30, alpha=0.7)
                axes[0, 1].set_xlabel('Prediction Probability')
                axes[0, 1].set_ylabel('Frequency')
                axes[0, 1].set_title('Prediction Distribution')

                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False).head(10)

                    axes[1, 0].barh(importance_df['feature'], importance_df['importance'])
                    axes[1, 0].set_xlabel('Importance')
                    axes[1, 0].set_title('Top 10 Feature Importances')

                # Confusion matrix heatmap
                if 'confusion_matrix' in evaluation_results:
                    cm = evaluation_results['confusion_matrix']
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
                    axes[1, 1].set_xlabel('Predicted')
                    axes[1, 1].set_ylabel('Actual')
                    axes[1, 1].set_title('Confusion Matrix')

                plt.tight_layout()

                # Save plot
                if output_path:
                    plot_path = Path(output_path).parent / f"evaluation_plots_{model_name}_{metadata.model_version}.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    click.echo(f"📊 Evaluation plots saved to {plot_path}")

            except ImportError:
                click.echo(f"⚠️ Warning: Matplotlib not available, skipping plots")

        # Save evaluation report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'model_version': metadata.model_version,
                'test_data_path': str(test_data),
                'test_records': len(data),
                'evaluation_results': evaluation_results,
                'model_metadata': metadata.to_dict()
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            click.echo(f"📄 Evaluation report saved to {output_path}")

        logger.info(f"Model evaluation completed: {model_name} v{metadata.model_version}")

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--models', '-m', multiple=True, required=True,
              help='Model specifications in format "name:version" (e.g., "model1:1.0.0")')
@click.option('--test-data', '-d', required=True, help='Test data file path')
@click.option('--output-path', '-o', help='Output path for comparison report')
@click.option('--generate-plots', is_flag=True, help='Generate comparison plots')
@click.pass_context
def compare(ctx, models, test_data, output_path, generate_plots):
    """Compare multiple models on the same test data."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"⚖️ Starting model comparison for {len(models)} models")

        # Parse model specifications
        model_specs = []
        for model_spec in models:
            if ':' in model_spec:
                name, version = model_spec.split(':', 1)
            else:
                name, version = model_spec, None

            model_specs.append((name, version))

        # Load models
        loaded_models = {}
        for name, version in model_specs:
            model, metadata = model_registry.load_model(name, version)
            if model is None:
                click.echo(f"❌ Error: Model {name} not found", err=True)
                continue

            key = f"{name}:{metadata.model_version}"
            loaded_models[key] = {'model': model, 'metadata': metadata}
            click.echo(f"✅ Loaded model: {key}")

        if not loaded_models:
            click.echo(f"❌ Error: No models could be loaded", err=True)
            sys.exit(1)

        # Load test data
        if not Path(test_data).exists():
            click.echo(f"❌ Error: Test data file {test_data} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(test_data)
        click.echo(f"📊 Loaded {len(data)} test records")

        # Initialize comparison
        comparison = ModelComparison()

        # Compare models
        click.echo(f"\n🔄 Comparing {len(loaded_models)} models...")

        comparison_results = comparison.compare_models(
            {k: v['model'] for k, v in loaded_models.items()},
            data,
            target_column='churn'
        )

        # Display comparison results
        click.echo(f"\n📊 Model Comparison Results:")

        # Performance comparison
        if 'performance_comparison' in comparison_results:
            perf_df = comparison_results['performance_comparison']
            click.echo(f"\n🏆 Performance Metrics:")
            print(perf_df.round(4).to_string())

        # Statistical significance
        if 'statistical_tests' in comparison_results:
            stats = comparison_results['statistical_tests']
            click.echo(f"\n📈 Statistical Significance Tests:")
            for test_name, result in stats.items():
                click.echo(f"  {test_name}: p-value = {result.get('p_value', 'N/A')}")

        # Best model recommendation
        if 'recommendations' in comparison_results:
            rec = comparison_results['recommendations']
            click.echo(f"\n🥇 Best Model: {rec.get('best_model', 'N/A')}")
            click.echo(f"📋 Reasoning: {rec.get('reasoning', 'N/A')}")

        # Generate comparison plots
        if generate_plots:
            try:
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Model Comparison Results')

                # Performance comparison bar chart
                if 'performance_comparison' in comparison_results:
                    perf_df = comparison_results['performance_comparison']
                    perf_df[['f1_score', 'auc_score']].plot(kind='bar', ax=axes[0, 0])
                    axes[0, 0].set_title('F1 Score and AUC Comparison')
                    axes[0, 0].set_ylabel('Score')
                    axes[0, 0].legend()

                # ROC curves comparison
                if 'roc_curves' in comparison_results:
                    for model_name, roc_data in comparison_results['roc_curves'].items():
                        axes[0, 1].plot(roc_data['fpr'], roc_data['tpr'],
                                       label=f"{model_name} (AUC={roc_data['auc']:.3f})")
                    axes[0, 1].plot([0, 1], [0, 1], 'k--')
                    axes[0, 1].set_xlabel('False Positive Rate')
                    axes[0, 1].set_ylabel('True Positive Rate')
                    axes[0, 1].set_title('ROC Curves Comparison')
                    axes[0, 1].legend()

                # Prediction distributions
                if 'predictions' in comparison_results:
                    for model_name, predictions in comparison_results['predictions'].items():
                        axes[1, 0].hist(predictions, alpha=0.5, label=model_name, bins=20)
                    axes[1, 0].set_xlabel('Prediction Probability')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].set_title('Prediction Distributions')
                    axes[1, 0].legend()

                # Performance radar chart
                # (Implementation would require additional plotting code)

                plt.tight_layout()

                # Save plot
                if output_path:
                    plot_path = Path(output_path).parent / f"model_comparison_plots.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    click.echo(f"📊 Comparison plots saved to {plot_path}")

            except ImportError:
                click.echo(f"⚠️ Warning: Matplotlib not available, skipping plots")

        # Save comparison report
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            report = {
                'timestamp': datetime.now().isoformat(),
                'models_compared': list(loaded_models.keys()),
                'test_data_path': str(test_data),
                'test_records': len(data),
                'comparison_results': comparison_results,
                'model_metadata': {k: v['metadata'].to_dict() for k, v in loaded_models.items()}
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            click.echo(f"📄 Comparison report saved to {output_path}")

        logger.info(f"Model comparison completed for {len(loaded_models)} models")

    except Exception as e:
        logger.error(f"Model comparison failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--model-name', '-n', required=True, help='Model name to deploy')
@click.option('--model-version', '-v', help='Model version (latest if not specified)')
@click.option('--environment', '-e', required=True,
              type=click.Choice(['staging', 'production']),
              help='Deployment environment')
@click.option('--force', is_flag=True, help='Force deployment without validation')
@click.pass_context
def deploy(ctx, model_name, model_version, environment, force):
    """Deploy a model to specified environment."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🚀 Starting model deployment: {model_name} to {environment}")

        # Load model from registry
        model, metadata = model_registry.load_model(model_name, model_version)
        if model is None:
            click.echo(f"❌ Error: Model {model_name} not found", err=True)
            sys.exit(1)

        click.echo(f"✅ Found model: {metadata.model_name} v{metadata.model_version}")

        # Validate model for deployment
        if not force:
            click.echo(f"🔍 Validating model for {environment} deployment...")

            # Check model performance
            min_f1_score = 0.80 if environment == 'production' else 0.70
            current_f1 = metadata.performance_metrics.get('f1_score', 0.0)

            if current_f1 < min_f1_score:
                click.echo(f"❌ Error: Model F1 score ({current_f1:.3f}) below minimum ({min_f1_score})", err=True)
                click.echo(f"Use --force to override validation")
                sys.exit(1)

            # Check model status
            if metadata.status not in [ModelStatus.VALIDATED.value, ModelStatus.DEPLOYED.value]:
                click.echo(f"❌ Error: Model status '{metadata.status}' not suitable for deployment", err=True)
                click.echo(f"Use --force to override validation")
                sys.exit(1)

            click.echo(f"✅ Model validation passed")

        # Deploy model
        success = model_registry.deploy_model(model_name, metadata.model_version, environment)

        if success:
            click.echo(f"🎉 Model deployed successfully!")
            click.echo(f"🌐 Environment: {environment}")
            click.echo(f"📦 Model: {model_name} v{metadata.model_version}")
            click.echo(f"🆔 Model ID: {metadata.model_id}")

            # Log deployment event
            logger.info(f"Model deployed: {model_name} v{metadata.model_version} to {environment}")

        else:
            click.echo(f"❌ Error: Model deployment failed", err=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--model-name', '-n', help='Filter by model name')
@click.option('--environment', '-e', help='Filter by deployment environment')
@click.option('--status', '-s', help='Filter by model status')
@click.option('--limit', '-l', type=int, default=10, help='Maximum number of models to list')
@click.pass_context
def list_models(ctx, model_name, environment, status, limit):
    """List models in the registry with filtering options."""

    config = ctx.obj['config']
    model_registry = ctx.obj['model_registry']
    verbose = ctx.obj['verbose']

    try:
        # Get all model versions
        all_versions = model_registry.version_manager.list_versions(model_name)

        # Apply filters
        filtered_versions = all_versions
        if environment:
            filtered_versions = [v for v in filtered_versions
                               if v.deployment_environment == environment]
        if status:
            filtered_versions = [v for v in filtered_versions if v.status == status]

        # Sort by creation date (newest first)
        filtered_versions.sort(key=lambda x: x.created_at, reverse=True)

        # Limit results
        filtered_versions = filtered_versions[:limit]

        # Display results
        if not filtered_versions:
            click.echo(f"📭 No models found matching the criteria")
            return

        click.echo(f"📋 Found {len(filtered_versions)} models:")
        click.echo(f"{'Name':<20} {'Version':<15} {'Status':<12} {'Environment':<12} {'F1 Score':<10} {'Created'}")
        click.echo("-" * 100)

        for metadata in filtered_versions:
            f1_score = metadata.performance_metrics.get('f1_score', 0.0)
            created_date = metadata.created_at[:10]  # Just the date part

            click.echo(f"{metadata.model_name:<20} "
                      f"{metadata.model_version:<15} "
                      f"{metadata.status:<12} "
                      f"{metadata.deployment_environment or 'N/A':<12} "
                      f"{f1_score:<10.3f} "
                      f"{created_date}")

            if verbose:
                click.echo(f"    ID: {metadata.model_id}")
                click.echo(f"    Type: {metadata.model_type}")
                click.echo(f"    Metrics: {metadata.performance_metrics}")
                click.echo()

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@model.command()
@click.option('--pipeline', '-p', default='churn_training_pipeline',
              help='Training pipeline name to execute')
@click.option('--parameters', help='Pipeline parameters as JSON string')
@click.option('--wait', is_flag=True, help='Wait for pipeline completion')
@click.pass_context
def run_training_pipeline(ctx, pipeline, parameters, wait):
    """Execute model training pipeline."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    async def execute_pipeline():
        try:
            if verbose:
                click.echo(f"🚀 Starting training pipeline: {pipeline}")

            # Parse parameters
            pipeline_params = {}
            if parameters:
                try:
                    pipeline_params = json.loads(parameters)
                except json.JSONDecodeError as e:
                    click.echo(f"❌ Error parsing parameters: {e}", err=True)
                    return

            # Create orchestrator
            orchestrator = create_orchestrator(config)

            # Execute pipeline
            click.echo(f"⚙️ Executing training pipeline...")
            pipeline_run = await orchestrator.execute_pipeline(pipeline, pipeline_params)

            click.echo(f"🎯 Training pipeline completed")
            click.echo(f"Run ID: {pipeline_run.run_id}")
            click.echo(f"Status: {pipeline_run.status.value}")
            click.echo(f"Duration: {pipeline_run.duration_seconds:.2f}s")

            # Display task results
            if verbose:
                click.echo("\n📋 Task Results:")
                for task_id, result in pipeline_run.task_results.items():
                    status_icon = "✅" if result.status.value == "completed" else "❌"
                    click.echo(f"  {status_icon} {task_id}: {result.status.value}")

                    if result.error:
                        click.echo(f"    Error: {result.error}")

            logger.info(f"Training pipeline completed: {pipeline_run.run_id}")

            # Exit with error code if pipeline failed
            if pipeline_run.status.value == 'failed':
                sys.exit(1)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)

    # Run async pipeline execution
    asyncio.run(execute_pipeline())


if __name__ == '__main__':
    model()
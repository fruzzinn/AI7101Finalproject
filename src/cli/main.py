"""
Main CLI Entry Point for Churn Prediction ML Pipeline
Task T050: Unified command-line interface for all system operations

This module provides the main CLI entry point that orchestrates:
- Data processing commands
- Model training and management commands
- Business analysis and reporting commands
- System administration and monitoring
- Pipeline orchestration and automation
"""

import click
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# Import project modules
from src.config.settings import get_config, Environment, initialize_config
from src.utils.logging import setup_logging, get_logger
from src.cli.data_commands import data
from src.cli.model_commands import model
from src.cli.business_commands import business


@click.group()
@click.version_option(version='1.0.0', prog_name='ChurnPredictor')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              default='development', help='Environment configuration')
@click.option('--log-level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def cli(ctx, config, environment, log_level, verbose):
    """
    🎯 ChurnPredictor ML Pipeline CLI

    A comprehensive command-line interface for churn prediction machine learning operations.

    COMMANDS:
    ∙ data      - Data ingestion, validation, and processing
    ∙ model     - Model training, evaluation, and deployment
    ∙ business  - Business analysis, ROI calculation, and reporting
    ∙ system    - System administration and monitoring
    ∙ pipeline  - End-to-end pipeline orchestration

    EXAMPLES:
    ∙ churn data ingest --source csv --input-path data.csv
    ∙ churn model train --data-path processed_data.csv --model-type random_forest
    ∙ churn business analyze-clv --customer-data customers.csv
    ∙ churn pipeline run --pipeline-name churn_training_pipeline

    For more information on specific commands, use:
    churn <command> --help
    """

    # Ensure context object exists
    ctx.ensure_object(dict)

    try:
        # Initialize configuration
        ctx.obj['config'] = initialize_config(environment, config)

        # Setup logging
        setup_logging(log_level=log_level)
        ctx.obj['logger'] = get_logger('churn_prediction.cli')

        # Store common settings
        ctx.obj['environment'] = environment
        ctx.obj['verbose'] = verbose

        if verbose:
            click.echo(f"🔧 ChurnPredictor CLI v1.0.0")
            click.echo(f"🌍 Environment: {environment}")
            click.echo(f"📋 Config: {config or 'default'}")
            click.echo(f"📊 Log Level: {log_level}")
            click.echo(f"📅 Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        click.echo(f"❌ Initialization error: {e}", err=True)
        sys.exit(1)


# Add command groups
cli.add_command(data)
cli.add_command(model)
cli.add_command(business)


@cli.group()
@click.pass_context
def system(ctx):
    """System administration and monitoring commands."""
    pass


@system.command()
@click.option('--component', '-c',
              type=click.Choice(['all', 'config', 'database', 'models', 'logging']),
              default='all', help='Component to check')
@click.option('--output-format', '-f',
              type=click.Choice(['text', 'json']),
              default='text', help='Output format')
@click.pass_context
def health_check(ctx, component, output_format):
    """Perform system health checks."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🏥 Performing health check for: {component}")

        health_results = {
            'timestamp': datetime.now().isoformat(),
            'environment': ctx.obj['environment'],
            'overall_status': 'healthy',
            'components': {}
        }

        # Configuration health check
        if component in ['all', 'config']:
            config_errors = config.validate_configuration()
            health_results['components']['configuration'] = {
                'status': 'healthy' if not config_errors else 'warning',
                'errors': config_errors,
                'details': {
                    'environment': config.environment.value,
                    'data_paths_exist': all(
                        os.path.exists(getattr(config.data, attr))
                        for attr in ['raw_data_path', 'processed_data_path', 'model_artifacts_path', 'logs_path']
                    )
                }
            }

        # Database health check
        if component in ['all', 'database']:
            try:
                db_config = config.database
                # Simple connection test (placeholder)
                health_results['components']['database'] = {
                    'status': 'healthy',
                    'details': {
                        'host': db_config.host,
                        'port': db_config.port,
                        'database': db_config.database
                    }
                }
            except Exception as e:
                health_results['components']['database'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Model registry health check
        if component in ['all', 'models']:
            try:
                from src.utils.model_utils import ModelRegistry
                registry = ModelRegistry(config.data.model_artifacts_path)

                # Count models
                all_versions = registry.version_manager.list_versions()
                deployed_models = [v for v in all_versions if v.status == 'deployed']

                health_results['components']['models'] = {
                    'status': 'healthy',
                    'details': {
                        'total_models': len(all_versions),
                        'deployed_models': len(deployed_models),
                        'registry_path': config.data.model_artifacts_path
                    }
                }
            except Exception as e:
                health_results['components']['models'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Logging health check
        if component in ['all', 'logging']:
            try:
                from src.utils.logging import get_logging_manager
                logging_manager = get_logging_manager()
                logging_health = logging_manager.health_check()

                health_results['components']['logging'] = {
                    'status': logging_health.get('status', 'unknown'),
                    'details': logging_health
                }
            except Exception as e:
                health_results['components']['logging'] = {
                    'status': 'error',
                    'error': str(e)
                }

        # Determine overall status
        component_statuses = [comp['status'] for comp in health_results['components'].values()]
        if 'error' in component_statuses:
            health_results['overall_status'] = 'error'
        elif 'warning' in component_statuses:
            health_results['overall_status'] = 'warning'

        # Output results
        if output_format == 'json':
            click.echo(json.dumps(health_results, indent=2))
        else:
            # Text format
            overall_icon = "✅" if health_results['overall_status'] == 'healthy' else "⚠️" if health_results['overall_status'] == 'warning' else "❌"
            click.echo(f"{overall_icon} Overall System Health: {health_results['overall_status'].upper()}")
            click.echo(f"📅 Check Time: {health_results['timestamp']}")
            click.echo(f"🌍 Environment: {health_results['environment']}")

            click.echo(f"\n📋 Component Status:")
            for comp_name, comp_data in health_results['components'].items():
                status_icon = "✅" if comp_data['status'] == 'healthy' else "⚠️" if comp_data['status'] == 'warning' else "❌"
                click.echo(f"  {status_icon} {comp_name.title()}: {comp_data['status'].upper()}")

                if comp_data['status'] != 'healthy' and 'error' in comp_data:
                    click.echo(f"    Error: {comp_data['error']}")
                elif comp_data['status'] != 'healthy' and 'errors' in comp_data:
                    for error in comp_data['errors']:
                        click.echo(f"    • {error}")

                if verbose and 'details' in comp_data:
                    for key, value in comp_data['details'].items():
                        click.echo(f"    {key}: {value}")

        logger.info(f"Health check completed: {health_results['overall_status']}")

        # Exit with error code if system is unhealthy
        if health_results['overall_status'] == 'error':
            sys.exit(1)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        click.echo(f"❌ Health check error: {e}", err=True)
        sys.exit(1)


@system.command()
@click.option('--component', '-c',
              type=click.Choice(['all', 'directories', 'config', 'logging']),
              default='all', help='Component to initialize')
@click.option('--force', is_flag=True, help='Force initialization, overwriting existing')
@click.pass_context
def init(ctx, component, force):
    """Initialize system components and directories."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🚀 Initializing system component: {component}")

        initialization_results = []

        # Initialize directories
        if component in ['all', 'directories']:
            click.echo(f"📁 Creating directory structure...")

            directories = [
                config.data.raw_data_path,
                config.data.processed_data_path,
                config.data.model_artifacts_path,
                config.data.logs_path,
                'data/interim',
                'data/external',
                'reports',
                'references'
            ]

            for directory in directories:
                dir_path = Path(directory)
                if not dir_path.exists() or force:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    initialization_results.append(f"✅ Created directory: {directory}")
                    if verbose:
                        click.echo(f"  Created: {directory}")
                elif verbose:
                    click.echo(f"  Exists: {directory}")

        # Initialize configuration files
        if component in ['all', 'config']:
            click.echo(f"⚙️ Setting up configuration...")

            # Create sample configuration
            config_dir = Path('config')
            config_dir.mkdir(exist_ok=True)

            sample_config_path = config_dir / 'sample_config.yaml'
            if not sample_config_path.exists() or force:
                sample_config = {
                    'model': {
                        'target_column': 'churn',
                        'test_size': 0.2,
                        'cv_folds': 5
                    },
                    'business': {
                        'avg_monthly_revenue': 50.0,
                        'customer_acquisition_cost': 150.0,
                        'high_risk_threshold': 0.7
                    },
                    'hyperparameters': {
                        'random_forest': {
                            'n_estimators': 100,
                            'max_depth': 10,
                            'random_state': 42
                        }
                    }
                }

                import yaml
                with open(sample_config_path, 'w') as f:
                    yaml.dump(sample_config, f, default_flow_style=False)

                initialization_results.append(f"✅ Created sample config: {sample_config_path}")
                if verbose:
                    click.echo(f"  Created: {sample_config_path}")

        # Initialize logging
        if component in ['all', 'logging']:
            click.echo(f"📊 Setting up logging...")

            # Create logs directory structure
            logs_dir = Path(config.data.logs_path)
            log_subdirs = ['pipeline_runs', 'model_training', 'business_analysis']

            for subdir in log_subdirs:
                subdir_path = logs_dir / subdir
                if not subdir_path.exists() or force:
                    subdir_path.mkdir(parents=True, exist_ok=True)
                    initialization_results.append(f"✅ Created log directory: {subdir_path}")
                    if verbose:
                        click.echo(f"  Created: {subdir_path}")

        # Summary
        click.echo(f"\n🎉 Initialization completed!")
        click.echo(f"📊 Operations performed: {len(initialization_results)}")

        if verbose:
            for result in initialization_results:
                click.echo(f"  {result}")

        logger.info(f"System initialization completed: {component}")

    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        click.echo(f"❌ Initialization error: {e}", err=True)
        sys.exit(1)


@system.command()
@click.option('--days', '-d', type=int, default=30, help='Number of days of logs to show')
@click.option('--level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Minimum log level to show')
@click.option('--component', '-c', help='Filter by component name')
@click.option('--tail', '-t', type=int, help='Show last N log entries')
@click.pass_context
def logs(ctx, days, level, component, tail):
    """View system logs and activity."""

    config = ctx.obj['config']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📋 Retrieving system logs (last {days} days)")

        logs_dir = Path(config.data.logs_path)

        if not logs_dir.exists():
            click.echo(f"📭 No logs directory found at {logs_dir}")
            return

        # Find log files
        log_files = []
        for log_file in logs_dir.glob('**/*.log'):
            # Check if file is recent enough
            file_age = (datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)).days
            if file_age <= days:
                log_files.append(log_file)

        if not log_files:
            click.echo(f"📭 No recent log files found")
            return

        click.echo(f"📄 Found {len(log_files)} log files:")

        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True):
            file_size = log_file.stat().st_size / 1024  # KB
            modified_time = datetime.fromtimestamp(log_file.stat().st_mtime)

            click.echo(f"  📁 {log_file.name}: {file_size:.1f}KB, "
                      f"modified {modified_time.strftime('%Y-%m-%d %H:%M')}")

            # Show recent entries if requested
            if tail and log_file.suffix == '.log':
                try:
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        recent_lines = lines[-tail:] if len(lines) > tail else lines

                        if recent_lines:
                            click.echo(f"    Recent entries:")
                            for line in recent_lines:
                                line = line.strip()
                                if line:
                                    # Simple filtering
                                    if level and level not in line:
                                        continue
                                    if component and component.lower() not in line.lower():
                                        continue

                                    # Truncate long lines
                                    if len(line) > 120:
                                        line = line[:117] + "..."

                                    click.echo(f"      {line}")

                except Exception as e:
                    click.echo(f"    ⚠️ Error reading file: {e}")

        # Show pipeline run history
        pipeline_runs_dir = logs_dir / 'pipeline_runs'
        if pipeline_runs_dir.exists():
            run_files = list(pipeline_runs_dir.glob('*.json'))
            if run_files:
                click.echo(f"\n🔄 Recent Pipeline Runs ({len(run_files)}):")

                for run_file in sorted(run_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                    try:
                        with open(run_file, 'r') as f:
                            run_data = json.load(f)

                        run_id = run_data.get('run_id', 'unknown')
                        pipeline_name = run_data.get('pipeline_name', 'unknown')
                        status = run_data.get('status', 'unknown')
                        duration = run_data.get('duration_seconds', 0)

                        status_icon = "✅" if status == 'completed' else "❌" if status == 'failed' else "🔄"

                        click.echo(f"  {status_icon} {run_id}: {pipeline_name} "
                                  f"({status}, {duration:.1f}s)")

                    except Exception:
                        click.echo(f"  ⚠️ {run_file.name}: Error reading run data")

    except Exception as e:
        click.echo(f"❌ Error retrieving logs: {e}", err=True)
        sys.exit(1)


@cli.group()
@click.pass_context
def pipeline(ctx):
    """End-to-end pipeline orchestration commands."""
    pass


@pipeline.command()
@click.option('--pipeline-name', '-p', required=True, help='Pipeline name to execute')
@click.option('--config-file', '-c', help='Pipeline configuration file')
@click.option('--parameters', help='Pipeline parameters as JSON string')
@click.option('--dry-run', is_flag=True, help='Validate pipeline without execution')
@click.option('--wait', is_flag=True, help='Wait for pipeline completion')
@click.option('--output-path', '-o', help='Output path for pipeline results')
@click.pass_context
def run(ctx, pipeline_name, config_file, parameters, dry_run, wait, output_path):
    """Execute end-to-end machine learning pipeline."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    async def execute_pipeline():
        try:
            if verbose:
                click.echo(f"🚀 Executing pipeline: {pipeline_name}")

            # Parse parameters
            pipeline_params = {}
            if parameters:
                try:
                    pipeline_params = json.loads(parameters)
                except json.JSONDecodeError as e:
                    click.echo(f"❌ Error parsing parameters: {e}", err=True)
                    return

            # Load additional config if provided
            if config_file and Path(config_file).exists():
                with open(config_file, 'r') as f:
                    if config_file.endswith('.json'):
                        file_config = json.load(f)
                    else:
                        import yaml
                        file_config = yaml.safe_load(f)

                    pipeline_params.update(file_config)
                    if verbose:
                        click.echo(f"📋 Loaded config from {config_file}")

            # Create orchestrator
            from src.pipeline.orchestrator import create_orchestrator
            orchestrator = create_orchestrator(config)

            # List available pipelines
            available_pipelines = orchestrator.list_pipelines()
            pipeline_names = [p['name'] for p in available_pipelines]

            if pipeline_name not in pipeline_names:
                click.echo(f"❌ Error: Pipeline '{pipeline_name}' not found", err=True)
                click.echo(f"Available pipelines: {', '.join(pipeline_names)}")
                return

            # Dry run validation
            if dry_run:
                click.echo(f"🔍 Validating pipeline: {pipeline_name}")
                pipeline_obj = orchestrator.pipelines[pipeline_name]
                validation_errors = pipeline_obj.validate()

                if validation_errors:
                    click.echo(f"❌ Pipeline validation failed:")
                    for error in validation_errors:
                        click.echo(f"  • {error}")
                    return
                else:
                    click.echo(f"✅ Pipeline validation passed")
                    execution_order = pipeline_obj.get_execution_order()
                    click.echo(f"📊 Execution plan: {len(execution_order)} levels")
                    for i, level in enumerate(execution_order):
                        click.echo(f"  Level {i+1}: {', '.join(level)}")
                    return

            # Execute pipeline
            click.echo(f"⚙️ Starting pipeline execution...")
            click.echo(f"📝 Parameters: {pipeline_params}")

            import asyncio
            pipeline_run = await orchestrator.execute_pipeline(pipeline_name, pipeline_params)

            # Display results
            click.echo(f"\n🎯 Pipeline execution completed!")
            click.echo(f"🆔 Run ID: {pipeline_run.run_id}")
            click.echo(f"📊 Status: {pipeline_run.status.value}")
            click.echo(f"⏱️ Duration: {pipeline_run.duration_seconds:.2f} seconds")
            click.echo(f"📋 Tasks: {len(pipeline_run.task_results)}")

            # Task summary
            completed_tasks = sum(1 for r in pipeline_run.task_results.values() if r.status.value == 'completed')
            failed_tasks = sum(1 for r in pipeline_run.task_results.values() if r.status.value == 'failed')

            click.echo(f"✅ Completed: {completed_tasks}")
            click.echo(f"❌ Failed: {failed_tasks}")

            # Detailed task results
            if verbose or failed_tasks > 0:
                click.echo(f"\n📋 Task Results:")
                for task_id, result in pipeline_run.task_results.items():
                    status_icon = "✅" if result.status.value == "completed" else "❌"
                    click.echo(f"  {status_icon} {task_id}: {result.status.value} "
                              f"({result.duration_seconds:.2f}s)")

                    if result.error:
                        click.echo(f"    Error: {result.error}")

            # Save results if output path specified
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w') as f:
                    json.dump(pipeline_run.to_dict(), f, indent=2, default=str)

                click.echo(f"📄 Pipeline results saved to {output_path}")

            logger.info(f"Pipeline execution completed: {pipeline_run.run_id}")

            # Exit with error code if pipeline failed
            if pipeline_run.status.value == 'failed':
                sys.exit(1)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)

    # Run async pipeline execution
    import asyncio
    asyncio.run(execute_pipeline())


@pipeline.command()
@click.pass_context
def list_pipelines(ctx):
    """List available pipelines."""

    try:
        from src.pipeline.orchestrator import create_orchestrator
        orchestrator = create_orchestrator(ctx.obj['config'])

        pipelines = orchestrator.list_pipelines()

        if not pipelines:
            click.echo(f"📭 No pipelines available")
            return

        click.echo(f"📋 Available Pipelines ({len(pipelines)}):")

        for pipeline in pipelines:
            click.echo(f"\n🔗 {pipeline['name']}")
            click.echo(f"  📝 Description: {pipeline['description']}")
            click.echo(f"  📊 Tasks: {pipeline['task_count']}")
            click.echo(f"  📋 Task List: {', '.join(pipeline['tasks'])}")

    except Exception as e:
        click.echo(f"❌ Error listing pipelines: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information and system details."""

    try:
        config = ctx.obj['config']

        click.echo(f"🎯 ChurnPredictor ML Pipeline")
        click.echo(f"📦 Version: 1.0.0")
        click.echo(f"🐍 Python: {sys.version.split()[0]}")
        click.echo(f"🌍 Environment: {ctx.obj['environment']}")
        click.echo(f"📁 Working Directory: {os.getcwd()}")
        click.echo(f"🗂️ Model Registry: {config.data.model_artifacts_path}")
        click.echo(f"📊 Log Directory: {config.data.logs_path}")
        click.echo(f"📅 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Package versions
        try:
            import pandas as pd
            import numpy as np
            import sklearn

            click.echo(f"\n📚 Key Dependencies:")
            click.echo(f"  • pandas: {pd.__version__}")
            click.echo(f"  • numpy: {np.__version__}")
            click.echo(f"  • scikit-learn: {sklearn.__version__}")

            try:
                import xgboost
                click.echo(f"  • xgboost: {xgboost.__version__}")
            except ImportError:
                pass

            try:
                import lightgbm
                click.echo(f"  • lightgbm: {lightgbm.__version__}")
            except ImportError:
                pass

        except ImportError as e:
            click.echo(f"⚠️ Some dependencies not available: {e}")

    except Exception as e:
        click.echo(f"❌ Error getting version info: {e}", err=True)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo(f"\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
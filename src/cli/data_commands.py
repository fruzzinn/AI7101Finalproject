"""
CLI Commands for Data Processing in Churn Prediction ML Pipeline
Task T047: Command-line interface for data operations

This module provides CLI commands for:
- Data ingestion from various sources
- Data validation and quality checks
- Feature engineering and preprocessing
- Data export and transformation
- Data pipeline execution and monitoring
"""

import click
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import json

# Import project modules
from src.config.settings import get_config, Environment, initialize_config
from src.utils.logging import setup_logging, get_logger, LogContext
from src.data.loader import ChurnDataLoader
from src.data.validator import DataValidator
from src.data.quality import DataQualityChecker
from src.features.processor import FeatureProcessor
from src.features.validator import FeatureValidator
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
def data(ctx, config, environment, log_level, verbose):
    """Data processing commands for churn prediction pipeline."""

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Initialize configuration
    ctx.obj['config'] = initialize_config(environment, config)

    # Setup logging
    setup_logging(log_level=log_level)
    ctx.obj['logger'] = get_logger('churn_prediction.cli.data')

    # Set verbose mode
    ctx.obj['verbose'] = verbose

    if verbose:
        click.echo(f"🔧 Environment: {environment}")
        click.echo(f"📋 Config: {config or 'default'}")
        click.echo(f"📊 Log Level: {log_level}")


@data.command()
@click.option('--source', '-s', default='csv',
              type=click.Choice(['csv', 'database', 'api']),
              help='Data source type')
@click.option('--input-path', '-i', help='Input file path for CSV source')
@click.option('--output-path', '-o', help='Output path for processed data')
@click.option('--validate', is_flag=True, help='Validate data after ingestion')
@click.pass_context
def ingest(ctx, source, input_path, output_path, validate):
    """Ingest raw data from various sources."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🔄 Starting data ingestion from {source}")

        # Initialize data loader
        loader = ChurnDataLoader()

        # Load data based on source
        if source == 'csv':
            if not input_path:
                input_path = click.prompt('Enter CSV file path')

            if not Path(input_path).exists():
                click.echo(f"❌ Error: File {input_path} not found", err=True)
                sys.exit(1)

            data = loader.load_csv_data(input_path)

        elif source == 'database':
            data = loader.load_database_data()

        elif source == 'api':
            data = loader.load_api_data()

        # Display data summary
        click.echo(f"✅ Successfully ingested {len(data)} records")
        click.echo(f"📊 Columns: {len(data.columns)}")
        click.echo(f"🗃️ Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if verbose:
            click.echo("\n📋 Data Summary:")
            click.echo(data.info())
            click.echo("\n📈 Data Description:")
            click.echo(data.describe())

        # Validate data if requested
        if validate:
            click.echo("\n🔍 Validating data...")
            validator = DataValidator()
            validation_results = validator.validate_data(data)

            if validation_results['is_valid']:
                click.echo("✅ Data validation passed")
            else:
                click.echo("⚠️ Data validation issues found:")
                for error in validation_results['errors']:
                    click.echo(f"  • {error}")

        # Save data if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == '.csv':
                data.to_csv(output_path, index=False)
            elif output_path.suffix in ['.pkl', '.pickle']:
                data.to_pickle(output_path)
            elif output_path.suffix == '.parquet':
                data.to_parquet(output_path)
            else:
                click.echo(f"⚠️ Unknown output format: {output_path.suffix}")
                data.to_csv(output_path.with_suffix('.csv'), index=False)

            click.echo(f"💾 Data saved to {output_path}")

        logger.info(f"Data ingestion completed: {len(data)} records from {source}")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--input-path', '-i', required=True, help='Input data file path')
@click.option('--output-path', '-o', help='Output validation report path')
@click.option('--checks', '-k', multiple=True,
              type=click.Choice(['completeness', 'uniqueness', 'consistency', 'validity']),
              default=['completeness', 'uniqueness', 'consistency', 'validity'],
              help='Validation checks to perform')
@click.option('--threshold', '-t', type=float, default=0.1,
              help='Quality threshold (0.0-1.0)')
@click.pass_context
def validate(ctx, input_path, output_path, checks, threshold):
    """Validate data quality and generate reports."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🔍 Starting data validation for {input_path}")

        # Load data
        if not Path(input_path).exists():
            click.echo(f"❌ Error: File {input_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(input_path)
        click.echo(f"📊 Loaded {len(data)} records for validation")

        # Initialize validators
        validator = DataValidator()
        quality_checker = DataQualityChecker()

        # Perform validation
        validation_results = validator.validate_data(data)

        # Perform quality checks
        quality_results = {}
        if 'completeness' in checks:
            quality_results['completeness'] = quality_checker.check_completeness(data)

        if 'uniqueness' in checks:
            quality_results['uniqueness'] = quality_checker.check_uniqueness(data)

        if 'consistency' in checks:
            quality_results['consistency'] = quality_checker.check_consistency(data)

        if 'validity' in checks:
            quality_results['validity'] = quality_checker.check_validity(data)

        # Display results
        click.echo("\n📋 Validation Results:")
        click.echo(f"Overall Status: {'✅ PASSED' if validation_results['is_valid'] else '❌ FAILED'}")

        if not validation_results['is_valid']:
            click.echo("\n⚠️ Validation Issues:")
            for error in validation_results['errors']:
                click.echo(f"  • {error}")

        click.echo("\n📊 Quality Check Results:")
        for check_name, result in quality_results.items():
            status = "✅ PASSED" if result.get('passed', False) else "❌ FAILED"
            score = result.get('score', 0.0)
            click.echo(f"  {check_name.title()}: {status} (Score: {score:.3f})")

            if verbose and 'details' in result:
                for detail in result['details'][:5]:  # Show first 5 details
                    click.echo(f"    - {detail}")

        # Calculate overall quality score
        overall_score = sum(r.get('score', 0.0) for r in quality_results.values()) / len(quality_results)
        quality_passed = overall_score >= threshold

        click.echo(f"\n🎯 Overall Quality Score: {overall_score:.3f}")
        click.echo(f"Quality Threshold: {threshold}")
        click.echo(f"Status: {'✅ PASSED' if quality_passed else '❌ FAILED'}")

        # Save validation report
        report = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_path),
            'record_count': len(data),
            'validation_results': validation_results,
            'quality_results': quality_results,
            'overall_score': overall_score,
            'threshold': threshold,
            'passed': quality_passed
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            click.echo(f"📄 Validation report saved to {output_path}")

        logger.info(f"Data validation completed: {overall_score:.3f} score")

        # Exit with error code if validation failed
        if not quality_passed:
            sys.exit(1)

    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--input-path', '-i', required=True, help='Input data file path')
@click.option('--output-path', '-o', required=True, help='Output processed data path')
@click.option('--features', '-f', multiple=True,
              help='Specific features to process (default: all)')
@click.option('--skip-validation', is_flag=True, help='Skip feature validation')
@click.option('--config-file', help='Feature processing configuration file')
@click.pass_context
def process_features(ctx, input_path, output_path, features, skip_validation, config_file):
    """Process and engineer features for model training."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"⚙️ Starting feature processing for {input_path}")

        # Load data
        if not Path(input_path).exists():
            click.echo(f"❌ Error: File {input_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(input_path)
        click.echo(f"📊 Loaded {len(data)} records for feature processing")

        # Initialize feature processor
        processor = FeatureProcessor()

        # Load configuration if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                feature_config = json.load(f)
            processor.configure(feature_config)
            if verbose:
                click.echo(f"🔧 Loaded feature configuration from {config_file}")

        # Process features
        if features:
            # Process only specified features
            click.echo(f"🎯 Processing specific features: {list(features)}")
            processed_data = processor.process_selected_features(data, list(features))
        else:
            # Process all features
            click.echo("🔄 Processing all features...")
            processed_data = processor.process_features(data)

        # Display processing summary
        original_cols = len(data.columns)
        processed_cols = len(processed_data.columns)
        click.echo(f"✅ Feature processing completed")
        click.echo(f"📊 Columns: {original_cols} → {processed_cols}")
        click.echo(f"🗃️ Memory usage: {processed_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if verbose:
            click.echo("\n📋 New Features Created:")
            new_features = set(processed_data.columns) - set(data.columns)
            for feature in sorted(new_features):
                click.echo(f"  • {feature}")

        # Validate features if not skipped
        if not skip_validation:
            click.echo("\n🔍 Validating processed features...")
            validator = FeatureValidator()
            validation_results = validator.validate_features(processed_data)

            if validation_results['is_valid']:
                click.echo("✅ Feature validation passed")
            else:
                click.echo("⚠️ Feature validation issues found:")
                for error in validation_results['errors']:
                    click.echo(f"  • {error}")

        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.csv':
            processed_data.to_csv(output_path, index=False)
        elif output_path.suffix in ['.pkl', '.pickle']:
            processed_data.to_pickle(output_path)
        elif output_path.suffix == '.parquet':
            processed_data.to_parquet(output_path)

        click.echo(f"💾 Processed data saved to {output_path}")

        # Save feature metadata
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        feature_metadata = {
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_path),
            'output_file': str(output_path),
            'original_features': list(data.columns),
            'processed_features': list(processed_data.columns),
            'new_features': list(set(processed_data.columns) - set(data.columns)),
            'processing_config': getattr(processor, 'config', {}),
            'record_count': len(processed_data)
        }

        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)

        click.echo(f"📄 Feature metadata saved to {metadata_path}")

        logger.info(f"Feature processing completed: {original_cols} → {processed_cols} features")

    except Exception as e:
        logger.error(f"Feature processing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--input-path', '-i', required=True, help='Input data file path')
@click.option('--output-format', '-f', default='csv',
              type=click.Choice(['csv', 'json', 'parquet', 'excel']),
              help='Output format')
@click.option('--output-path', '-o', help='Output file path')
@click.option('--filter', 'filter_expr', help='Filter expression (e.g., "age > 30")')
@click.option('--sample', type=int, help='Sample N random records')
@click.option('--columns', '-c', multiple=True, help='Specific columns to export')
@click.pass_context
def export(ctx, input_path, output_format, output_path, filter_expr, sample, columns):
    """Export data in various formats with filtering options."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📤 Starting data export from {input_path}")

        # Load data
        if not Path(input_path).exists():
            click.echo(f"❌ Error: File {input_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(input_path)
        original_count = len(data)
        click.echo(f"📊 Loaded {original_count} records")

        # Apply filter if specified
        if filter_expr:
            try:
                data = data.query(filter_expr)
                click.echo(f"🔍 Applied filter '{filter_expr}': {len(data)} records remaining")
            except Exception as e:
                click.echo(f"❌ Error applying filter: {e}", err=True)
                sys.exit(1)

        # Select specific columns if specified
        if columns:
            missing_cols = set(columns) - set(data.columns)
            if missing_cols:
                click.echo(f"⚠️ Warning: Missing columns {missing_cols}")

            available_cols = [col for col in columns if col in data.columns]
            data = data[available_cols]
            click.echo(f"📋 Selected {len(available_cols)} columns")

        # Sample data if specified
        if sample:
            if sample > len(data):
                click.echo(f"⚠️ Warning: Sample size ({sample}) larger than data size ({len(data)})")
                sample = len(data)

            data = data.sample(n=sample, random_state=42)
            click.echo(f"🎲 Sampled {sample} random records")

        # Generate output path if not specified
        if not output_path:
            input_path_obj = Path(input_path)
            output_path = input_path_obj.parent / f"{input_path_obj.stem}_export.{output_format}"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export data based on format
        if output_format == 'csv':
            data.to_csv(output_path, index=False)
        elif output_format == 'json':
            data.to_json(output_path, orient='records', indent=2)
        elif output_format == 'parquet':
            data.to_parquet(output_path)
        elif output_format == 'excel':
            data.to_excel(output_path, index=False)

        click.echo(f"✅ Export completed")
        click.echo(f"📁 Output: {output_path}")
        click.echo(f"📊 Records: {len(data)}")
        click.echo(f"📋 Columns: {len(data.columns)}")
        click.echo(f"📏 File size: {output_path.stat().st_size / 1024**2:.2f} MB")

        logger.info(f"Data export completed: {len(data)} records to {output_format}")

    except Exception as e:
        logger.error(f"Data export failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@data.command()
@click.option('--pipeline', '-p', default='churn_data_pipeline',
              help='Pipeline name to execute')
@click.option('--parameters', help='Pipeline parameters as JSON string')
@click.option('--wait', is_flag=True, help='Wait for pipeline completion')
@click.option('--monitor', is_flag=True, help='Monitor pipeline execution')
@click.pass_context
def run_pipeline(ctx, pipeline, parameters, wait, monitor):
    """Execute data processing pipeline."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    async def execute_pipeline():
        try:
            if verbose:
                click.echo(f"🚀 Starting pipeline execution: {pipeline}")

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

            # List available pipelines
            available_pipelines = orchestrator.list_pipelines()
            pipeline_names = [p['name'] for p in available_pipelines]

            if pipeline not in pipeline_names:
                click.echo(f"❌ Error: Pipeline '{pipeline}' not found", err=True)
                click.echo(f"Available pipelines: {', '.join(pipeline_names)}")
                return

            # Execute pipeline
            click.echo(f"⚙️ Executing pipeline with parameters: {pipeline_params}")

            pipeline_run = await orchestrator.execute_pipeline(pipeline, pipeline_params)

            click.echo(f"🎯 Pipeline execution completed")
            click.echo(f"Run ID: {pipeline_run.run_id}")
            click.echo(f"Status: {pipeline_run.status.value}")
            click.echo(f"Duration: {pipeline_run.duration_seconds:.2f}s")

            # Display task results
            if verbose or monitor:
                click.echo("\n📋 Task Results:")
                for task_id, result in pipeline_run.task_results.items():
                    status_icon = "✅" if result.status.value == "completed" else "❌"
                    click.echo(f"  {status_icon} {task_id}: {result.status.value} ({result.duration_seconds:.2f}s)")

                    if result.error:
                        click.echo(f"    Error: {result.error}")

            logger.info(f"Pipeline execution completed: {pipeline_run.run_id}")

            # Exit with error code if pipeline failed
            if pipeline_run.status.value == 'failed':
                sys.exit(1)

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)

    # Run async pipeline execution
    asyncio.run(execute_pipeline())


@data.command()
@click.option('--input-path', '-i', required=True, help='Input data file path')
@click.option('--report-path', '-r', help='Output report path')
@click.pass_context
def profile(ctx, input_path, report_path):
    """Generate comprehensive data profile report."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📊 Starting data profiling for {input_path}")

        # Load data
        if not Path(input_path).exists():
            click.echo(f"❌ Error: File {input_path} not found", err=True)
            sys.exit(1)

        data = pd.read_csv(input_path)
        click.echo(f"📊 Loaded {len(data)} records for profiling")

        # Generate comprehensive profile
        profile_report = {
            'timestamp': datetime.now().isoformat(),
            'file_path': str(input_path),
            'basic_info': {
                'rows': len(data),
                'columns': len(data.columns),
                'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024**2,
                'data_types': data.dtypes.value_counts().to_dict()
            },
            'missing_data': {
                'total_missing': data.isnull().sum().sum(),
                'missing_percentage': (data.isnull().sum() / len(data)).mean() * 100,
                'columns_with_missing': data.isnull().sum()[data.isnull().sum() > 0].to_dict()
            },
            'column_profiles': {}
        }

        # Profile each column
        for column in data.columns:
            col_profile = {
                'type': str(data[column].dtype),
                'missing_count': int(data[column].isnull().sum()),
                'missing_percentage': float(data[column].isnull().sum() / len(data) * 100),
                'unique_count': int(data[column].nunique()),
                'unique_percentage': float(data[column].nunique() / len(data) * 100)
            }

            if pd.api.types.is_numeric_dtype(data[column]):
                col_profile.update({
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'mean': float(data[column].mean()),
                    'median': float(data[column].median()),
                    'std': float(data[column].std())
                })
            else:
                value_counts = data[column].value_counts().head(10)
                col_profile['top_values'] = value_counts.to_dict()

            profile_report['column_profiles'][column] = col_profile

        # Display summary
        click.echo("\n📋 Data Profile Summary:")
        click.echo(f"Rows: {profile_report['basic_info']['rows']:,}")
        click.echo(f"Columns: {profile_report['basic_info']['columns']}")
        click.echo(f"Memory Usage: {profile_report['basic_info']['memory_usage_mb']:.2f} MB")
        click.echo(f"Missing Data: {profile_report['missing_data']['missing_percentage']:.2f}%")

        if verbose:
            click.echo("\n📊 Column Summary:")
            for column, profile in profile_report['column_profiles'].items():
                click.echo(f"  {column}: {profile['type']} "
                          f"({profile['unique_count']} unique, "
                          f"{profile['missing_percentage']:.1f}% missing)")

        # Save report if path specified
        if not report_path:
            report_path = Path(input_path).parent / f"{Path(input_path).stem}_profile.json"
        else:
            report_path = Path(report_path)

        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(profile_report, f, indent=2, default=str)

        click.echo(f"📄 Profile report saved to {report_path}")

        logger.info(f"Data profiling completed for {len(data)} records")

    except Exception as e:
        logger.error(f"Data profiling failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    data()
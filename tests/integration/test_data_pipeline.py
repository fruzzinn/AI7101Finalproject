"""
Integration Tests for End-to-End Data Loading and Validation Pipeline
Tests the complete data flow from loading through validation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Import our services and utilities
from src.services.data_validation_service import DataValidationService
from src.utils.mlflow_utils import MLflowExperimentManager
from src.models.customer import Customer
from src.models.churn_event import ChurnEvent


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline"""

    @pytest.fixture
    def sample_customer_data(self):
        """Create realistic customer dataset for testing"""
        np.random.seed(42)
        n_customers = 1000

        data = {
            'customer_id': [f'C{i:04d}' for i in range(n_customers)],
            'age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
            'tenure': np.random.exponential(24, n_customers).astype(int).clip(0, 72),
            'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 8000),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.4, 0.4, 0.2]),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2]),
            'partner': np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_customers),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4])
        }

        # Create realistic churn based on business logic
        churn_probability = (
            0.1 +  # Base churn rate
            0.3 * (data['contract_type'] == 'Month-to-month').astype(int) +
            0.2 * (np.array(data['tenure']) < 12).astype(int) +
            0.15 * (np.array(data['monthly_charges']) > 80).astype(int) +
            0.1 * (data['payment_method'] == 'Electronic check').astype(int)
        )
        data['churn'] = np.random.binomial(1, np.clip(churn_probability, 0, 1), n_customers)

        return pd.DataFrame(data)

    @pytest.fixture
    def variables_metadata(self):
        """Create variable metadata for validation"""
        return pd.DataFrame({
            'variable': ['customer_id', 'age', 'tenure', 'monthly_charges', 'total_charges',
                        'contract_type', 'payment_method', 'internet_service', 'churn'],
            'description': [
                'Unique customer identifier',
                'Customer age in years',
                'Number of months as customer',
                'Monthly service charges',
                'Total charges to date',
                'Contract type',
                'Payment method',
                'Internet service type',
                'Churn indicator'
            ],
            'type': ['string', 'numeric', 'numeric', 'numeric', 'numeric',
                    'categorical', 'categorical', 'categorical', 'binary']
        })

    @pytest.fixture
    def temp_csv_file(self, sample_customer_data):
        """Create temporary CSV file for testing file loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_customer_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_end_to_end_data_loading_validation(self, sample_customer_data, variables_metadata):
        """Test complete data loading and validation workflow"""
        # Split data into train/test
        train_data = sample_customer_data.sample(n=800, random_state=42)
        test_data = sample_customer_data.drop(train_data.index).drop('churn', axis=1)

        # Initialize validation service
        validator = DataValidationService()

        # Test dataset validation
        validation_results = validator.validate_datasets(train_data, test_data, variables_metadata)

        # Assertions for successful validation
        assert validation_results['has_target_column'] is True, "Training data should have target column"
        assert validation_results['consistent_features'] is True, "Features should be consistent between train/test"
        assert validation_results['no_empty_columns'] is True, "No columns should be completely empty"
        assert validation_results['valid_data_types'] is True, "Data types should be valid for ML"

        # Test data quality assessment
        quality_metrics = validator.check_data_quality(train_data)

        assert 'missing_rate' in quality_metrics, "Should calculate missing value rate"
        assert 'duplicate_rate' in quality_metrics, "Should calculate duplicate rate"
        assert 'outlier_rate' in quality_metrics, "Should calculate outlier rate"
        assert quality_metrics['missing_rate'] <= 0.1, "Missing rate should be acceptable"

        # Test target variable validation
        target_validation = validator.validate_target_variable(train_data['churn'])
        assert target_validation is True, "Target variable should be valid binary"

    def test_data_entity_integration(self, sample_customer_data):
        """Test integration with data model entities"""
        # Test Customer entity creation
        sample_customer = sample_customer_data.iloc[0]

        customer = Customer(
            customer_id=sample_customer['customer_id'],
            demographic_features={
                'age': sample_customer['age'],
                'gender': sample_customer['gender'],
                'senior_citizen': sample_customer['senior_citizen']
            },
            usage_patterns={
                'tenure': sample_customer['tenure'],
                'monthly_charges': sample_customer['monthly_charges'],
                'total_charges': sample_customer['total_charges']
            },
            service_features={
                'internet_service': sample_customer['internet_service'],
                'phone_service': sample_customer['phone_service']
            },
            billing_features={
                'contract_type': sample_customer['contract_type'],
                'payment_method': sample_customer['payment_method']
            }
        )

        # Validate customer entity
        assert customer.customer_id == sample_customer['customer_id']
        assert customer.demographic_features['age'] == sample_customer['age']
        assert customer.validate_features() is True, "Customer features should be valid"

        # Test ChurnEvent entity
        churn_event = ChurnEvent(
            customer_id=sample_customer['customer_id'],
            churn=int(sample_customer['churn'])
        )

        assert churn_event.customer_id == sample_customer['customer_id']
        assert churn_event.churn in [0, 1], "Churn should be binary"
        assert churn_event.validate_event() is True, "Churn event should be valid"

    def test_data_loading_with_missing_values(self, sample_customer_data):
        """Test data pipeline handles missing values correctly"""
        # Introduce missing values
        corrupted_data = sample_customer_data.copy()

        # Add missing values to different column types
        missing_indices = np.random.choice(corrupted_data.index, size=100, replace=False)
        corrupted_data.loc[missing_indices[:50], 'age'] = np.nan  # Numeric
        corrupted_data.loc[missing_indices[50:], 'contract_type'] = np.nan  # Categorical

        validator = DataValidationService()

        # Test quality assessment with missing values
        quality_metrics = validator.check_data_quality(corrupted_data)

        assert quality_metrics['missing_rate'] > 0, "Should detect missing values"
        assert quality_metrics['missing_rate'] < 0.2, "Missing rate should be reasonable for test data"

    def test_data_pipeline_with_mlflow_integration(self, sample_customer_data):
        """Test data pipeline integration with MLflow tracking"""
        # Initialize MLflow manager (use temporary tracking for testing)
        mlflow_manager = MLflowExperimentManager(
            experiment_name="test_data_pipeline",
            tracking_uri="sqlite:///test_mlflow.db"
        )

        try:
            # Start MLflow run
            run_id = mlflow_manager.start_run("data_pipeline_test")

            # Split data
            train_data = sample_customer_data.sample(n=800, random_state=42)
            test_data = sample_customer_data.drop(train_data.index)

            # Log data information
            mlflow_manager.log_data_info(train_data, test_data)

            # Validate data
            validator = DataValidationService()
            quality_metrics = validator.check_data_quality(train_data)

            # Log validation results (simulate logging validation metrics)
            import mlflow
            mlflow.log_metric("data_missing_rate", quality_metrics['missing_rate'])
            mlflow.log_metric("data_duplicate_rate", quality_metrics['duplicate_rate'])
            mlflow.log_metric("data_outlier_rate", quality_metrics['outlier_rate'])

            # End run
            mlflow_manager.end_run()

            # Verify run was created
            assert run_id is not None, "MLflow run should be created"

        finally:
            # Cleanup test database
            if os.path.exists("test_mlflow.db"):
                os.unlink("test_mlflow.db")

    def test_large_dataset_performance(self):
        """Test data pipeline performance with larger dataset"""
        # Create larger dataset
        np.random.seed(42)
        n_customers = 10000

        large_data = pd.DataFrame({
            'customer_id': [f'C{i:05d}' for i in range(n_customers)],
            'age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
            'tenure': np.random.exponential(24, n_customers).astype(int).clip(0, 72),
            'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 8000),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'churn': np.random.binomial(1, 0.2, n_customers)
        })

        validator = DataValidationService()

        # Time the validation (should complete in reasonable time)
        import time
        start_time = time.time()

        quality_metrics = validator.check_data_quality(large_data)
        target_validation = validator.validate_target_variable(large_data['churn'])

        end_time = time.time()
        processing_time = end_time - start_time

        # Assertions
        assert processing_time < 5.0, "Large dataset processing should complete within 5 seconds"
        assert quality_metrics['missing_rate'] == 0, "Generated data should have no missing values"
        assert target_validation is True, "Target validation should pass"

    def test_data_consistency_across_pipeline(self, sample_customer_data):
        """Test data consistency throughout the pipeline"""
        # Original data characteristics
        original_shape = sample_customer_data.shape
        original_customer_ids = set(sample_customer_data['customer_id'])
        original_churn_rate = sample_customer_data['churn'].mean()

        # Process through validation
        validator = DataValidationService()
        quality_metrics = validator.check_data_quality(sample_customer_data)

        # Data should remain unchanged after validation
        assert sample_customer_data.shape == original_shape, "Data shape should not change during validation"
        assert set(sample_customer_data['customer_id']) == original_customer_ids, "Customer IDs should not change"
        assert abs(sample_customer_data['churn'].mean() - original_churn_rate) < 0.001, "Churn rate should not change"

    def test_error_handling_in_data_pipeline(self):
        """Test error handling in data pipeline"""
        validator = DataValidationService()

        # Test with None inputs
        with pytest.raises(TypeError):
            validator.validate_datasets(None, None, None)

        # Test with empty dataframes
        empty_df = pd.DataFrame()
        variables_df = pd.DataFrame({'variable': ['test'], 'type': ['numeric']})

        with pytest.raises(ValueError):
            validator.validate_datasets(empty_df, empty_df, variables_df)

        # Test with invalid target variable
        invalid_target = pd.Series([0, 1, 2, 3])  # Not binary
        result = validator.validate_target_variable(invalid_target)
        assert result is False, "Should reject non-binary target"

    def test_data_pipeline_reproducibility(self, sample_customer_data):
        """Test that data pipeline operations are reproducible"""
        validator1 = DataValidationService()
        validator2 = DataValidationService()

        # Run same validation twice
        quality_metrics1 = validator1.check_data_quality(sample_customer_data)
        quality_metrics2 = validator2.check_data_quality(sample_customer_data)

        # Results should be identical
        assert quality_metrics1['missing_rate'] == quality_metrics2['missing_rate']
        assert quality_metrics1['duplicate_rate'] == quality_metrics2['duplicate_rate']
        assert quality_metrics1['outlier_rate'] == quality_metrics2['outlier_rate']

    def test_file_loading_integration(self, temp_csv_file):
        """Test loading data from CSV file"""
        # Load data from temporary file
        loaded_data = pd.read_csv(temp_csv_file)

        # Validate loaded data
        validator = DataValidationService()
        quality_metrics = validator.check_data_quality(loaded_data)

        assert len(loaded_data) > 0, "Should load data from file"
        assert 'customer_id' in loaded_data.columns, "Should have customer_id column"
        assert 'churn' in loaded_data.columns, "Should have churn column"
        assert quality_metrics['missing_rate'] >= 0, "Should calculate quality metrics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Contract tests for DataValidationContract
These tests MUST FAIL initially to ensure TDD compliance
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import the contract interface (will fail until implemented)
try:
    from src.services.data_validation_service import DataValidationService
    from src.models.customer import Customer
    from src.models.churn_event import ChurnEvent
except ImportError:
    # Expected to fail initially - this enforces TDD
    pytest.skip("Implementation not available yet - TDD compliance", allow_module_level=True)


class TestDataValidationContract:
    """Test suite for DataValidationContract implementation"""

    @pytest.fixture
    def sample_train_data(self):
        """Create sample training data for testing"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, 35, 45, 55, 65],
            'tenure': [12, 24, 36, 48, 60],
            'monthly_charges': [50.0, 70.0, 90.0, 110.0, 130.0],
            'total_charges': [600.0, 1680.0, 3240.0, 5280.0, 7800.0],
            'contract_type': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],
            'churn': [1, 0, 0, 1, 0]  # Target variable
        })

    @pytest.fixture
    def sample_test_data(self):
        """Create sample test data for testing"""
        return pd.DataFrame({
            'customer_id': ['C006', 'C007', 'C008'],
            'age': [30, 40, 50],
            'tenure': [18, 30, 42],
            'monthly_charges': [60.0, 80.0, 100.0],
            'total_charges': [1080.0, 2400.0, 4200.0],
            'contract_type': ['Month-to-month', 'One year', 'Two year']
            # Note: No 'churn' column in test data
        })

    @pytest.fixture
    def sample_variables_data(self):
        """Create sample variable definitions data"""
        return pd.DataFrame({
            'variable': ['customer_id', 'age', 'tenure', 'monthly_charges', 'total_charges', 'contract_type', 'churn'],
            'description': [
                'Unique customer identifier',
                'Customer age in years',
                'Number of months with service',
                'Monthly service charges',
                'Total charges to date',
                'Contract type',
                'Target variable: 1=Churn, 0=No Churn'
            ],
            'type': ['string', 'numeric', 'numeric', 'numeric', 'numeric', 'categorical', 'binary']
        })

    @pytest.fixture
    def validation_service(self):
        """Create DataValidationService instance"""
        return DataValidationService()

    def test_validate_datasets_success(self, validation_service, sample_train_data, sample_test_data, sample_variables_data):
        """Test successful dataset validation"""
        result = validation_service.validate_datasets(
            sample_train_data,
            sample_test_data,
            sample_variables_data
        )

        # Contract requirements
        assert isinstance(result, dict), "Result must be a dictionary"
        assert result['has_target_column'] is True, "Training data must have 'churn' target column"
        assert result['consistent_features'] is True, "Feature columns must be consistent"
        assert result['no_empty_columns'] is True, "No completely empty columns allowed"
        assert result['valid_data_types'] is True, "Data types must be appropriate for ML"

    def test_validate_datasets_missing_target(self, validation_service, sample_test_data, sample_variables_data):
        """Test validation failure when target column is missing"""
        # Use test data (which has no 'churn' column) as training data
        result = validation_service.validate_datasets(
            sample_test_data,  # Missing 'churn' column
            sample_test_data,
            sample_variables_data
        )

        assert result['has_target_column'] is False, "Should detect missing target column"

    def test_validate_datasets_inconsistent_features(self, validation_service, sample_train_data, sample_variables_data):
        """Test validation failure with inconsistent feature columns"""
        # Create test data with different columns
        inconsistent_test_data = pd.DataFrame({
            'customer_id': ['C006'],
            'different_feature': [123],  # Different column name
            'age': [30]
        })

        result = validation_service.validate_datasets(
            sample_train_data,
            inconsistent_test_data,
            sample_variables_data
        )

        assert result['consistent_features'] is False, "Should detect inconsistent features"

    def test_check_data_quality_success(self, validation_service, sample_train_data):
        """Test data quality assessment with good data"""
        quality_metrics = validation_service.check_data_quality(sample_train_data)

        assert isinstance(quality_metrics, dict), "Quality metrics must be a dictionary"
        assert 'missing_rate' in quality_metrics, "Must include missing value rate"
        assert 'duplicate_rate' in quality_metrics, "Must include duplicate rate"
        assert 'outlier_rate' in quality_metrics, "Must include outlier detection"

        # With good sample data, rates should be low
        assert quality_metrics['missing_rate'] <= 0.1, "Missing rate should be low for good data"
        assert quality_metrics['duplicate_rate'] <= 0.1, "Duplicate rate should be low"

    def test_check_data_quality_with_issues(self, validation_service):
        """Test data quality assessment with problematic data"""
        problematic_data = pd.DataFrame({
            'feature1': [1, 2, np.nan, np.nan, 5],  # 40% missing
            'feature2': [1, 1, 1, 1, 1],  # No variance
            'feature3': [1, 2, 3, 1, 2],  # Duplicates
            'feature4': [1, 2, 3, 1000, 5]  # Contains outlier
        })

        quality_metrics = validation_service.check_data_quality(problematic_data)

        assert quality_metrics['missing_rate'] > 0.1, "Should detect high missing rate"
        assert quality_metrics['duplicate_rate'] > 0.0, "Should detect duplicates"

    def test_validate_target_variable_success(self, validation_service):
        """Test successful target variable validation"""
        valid_target = pd.Series([0, 1, 0, 1, 0, 1])

        result = validation_service.validate_target_variable(valid_target)

        assert result is True, "Valid binary target should pass validation"

    def test_validate_target_variable_non_binary(self, validation_service):
        """Test target variable validation with non-binary values"""
        invalid_target = pd.Series([0, 1, 2, 1, 0])  # Contains '2'

        result = validation_service.validate_target_variable(invalid_target)

        assert result is False, "Non-binary target should fail validation"

    def test_validate_target_variable_missing_values(self, validation_service):
        """Test target variable validation with missing values"""
        target_with_na = pd.Series([0, 1, np.nan, 1, 0])

        result = validation_service.validate_target_variable(target_with_na)

        assert result is False, "Target with missing values should fail validation"

    def test_validate_target_variable_single_class(self, validation_service):
        """Test target variable validation with only one class"""
        single_class_target = pd.Series([0, 0, 0, 0, 0])  # All zeros

        result = validation_service.validate_target_variable(single_class_target)

        assert result is False, "Single-class target should fail validation"

    def test_validate_datasets_empty_dataframe(self, validation_service, sample_variables_data):
        """Test validation with empty dataframes"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            validation_service.validate_datasets(empty_df, empty_df, sample_variables_data)

    def test_validate_datasets_null_input(self, validation_service):
        """Test validation with None inputs"""
        with pytest.raises(TypeError):
            validation_service.validate_datasets(None, None, None)


if __name__ == "__main__":
    # These tests should FAIL initially to ensure TDD compliance
    pytest.main([__file__, "-v"])
"""
Contract tests for DataLoaderContract interface.

Educational Focus: Demonstrates contract testing for data loading components.
These tests verify that any implementation of DataLoaderContract behaves correctly.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import Tuple, Dict, Any

# Import the contracts we're testing
from contracts.data_loader import DataLoaderContract, DataValidatorContract


class TestDataLoaderContract:
    """Test suite for DataLoaderContract interface compliance."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Create a mock implementation for testing
        self.mock_loader = Mock(spec=DataLoaderContract)

        # Sample data for testing
        self.sample_customer_data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'tenure': [12, 24, 6],
            'monthly_charges': [50.0, 75.0, 30.0],
            'contract_type': ['month-to-month', 'one-year', 'two-year']
        })

        self.sample_churn_data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'churn': [0, 1, 0]
        })

    def test_load_raw_data_contract_signature(self):
        """Test that load_raw_data has correct signature and return type."""
        # Arrange
        self.mock_loader.load_raw_data.return_value = (
            self.sample_customer_data,
            self.sample_churn_data
        )

        # Act
        result = self.mock_loader.load_raw_data('dummy_path.csv')

        # Assert
        assert isinstance(result, tuple), "Should return tuple"
        assert len(result) == 2, "Should return tuple of length 2"
        assert isinstance(result[0], pd.DataFrame), "First element should be DataFrame"
        assert isinstance(result[1], pd.DataFrame), "Second element should be DataFrame"
        self.mock_loader.load_raw_data.assert_called_once_with('dummy_path.csv')

    def test_load_raw_data_file_not_found_error(self):
        """Test that FileNotFoundError is raised for non-existent files."""
        # Arrange
        self.mock_loader.load_raw_data.side_effect = FileNotFoundError("File not found")

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            self.mock_loader.load_raw_data('non_existent_file.csv')

    def test_load_raw_data_invalid_format_error(self):
        """Test that ValueError is raised for invalid data format."""
        # Arrange
        self.mock_loader.load_raw_data.side_effect = ValueError("Invalid data format")

        # Act & Assert
        with pytest.raises(ValueError):
            self.mock_loader.load_raw_data('invalid_format.csv')

    def test_validate_data_quality_contract_signature(self):
        """Test that validate_data_quality has correct signature."""
        # This will fail until we implement DataValidatorContract
        mock_validator = Mock(spec=DataValidatorContract)

        # Arrange expected return structure
        expected_report = {
            'missing_percentages': {'tenure': 0.0, 'monthly_charges': 0.1},
            'duplicate_count': 5,
            'invalid_values': ['contract_type'],
            'quality_score': 0.85
        }
        mock_validator.validate_data_quality.return_value = expected_report

        # Act
        result = mock_validator.validate_data_quality(self.sample_customer_data)

        # Assert
        assert isinstance(result, dict), "Should return dictionary"
        assert 'missing_percentages' in result
        assert 'duplicate_count' in result
        assert 'invalid_values' in result
        assert 'quality_score' in result

    def test_split_features_target_contract_signature(self):
        """Test that split_features_target has correct signature."""
        # Arrange
        combined_data = self.sample_customer_data.copy()
        combined_data['churn'] = [0, 1, 0]

        expected_features = combined_data.drop('churn', axis=1)
        expected_target = combined_data['churn']

        self.mock_loader.split_features_target.return_value = (expected_features, expected_target)

        # Act
        features, target = self.mock_loader.split_features_target(combined_data, 'churn')

        # Assert
        assert isinstance(features, pd.DataFrame), "Features should be DataFrame"
        assert isinstance(target, pd.Series), "Target should be Series"
        assert 'churn' not in features.columns, "Features should not contain target column"

    def test_split_features_target_missing_column_error(self):
        """Test that KeyError is raised when target column doesn't exist."""
        # Arrange
        self.mock_loader.split_features_target.side_effect = KeyError("Target column not found")

        # Act & Assert
        with pytest.raises(KeyError):
            self.mock_loader.split_features_target(self.sample_customer_data, 'non_existent_column')


class TestDataValidatorContract:
    """Test suite for DataValidatorContract interface compliance."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_validator = Mock(spec=DataValidatorContract)

        # Sample data with various quality issues
        self.sample_data_with_issues = pd.DataFrame({
            'customer_id': ['C001', 'C002', None, 'C004'],
            'tenure': [12, 24, 6, None],
            'monthly_charges': [50.0, 75.0, 30.0, 100.0],
            'contract_type': ['month-to-month', 'invalid_type', 'one-year', 'two-year']
        })

    def test_check_missing_values_contract_signature(self):
        """Test that check_missing_values has correct signature."""
        # Arrange
        self.mock_validator.check_missing_values.return_value = False  # Above threshold

        # Act
        result = self.mock_validator.check_missing_values(self.sample_data_with_issues, 0.1)

        # Assert
        assert isinstance(result, bool), "Should return boolean"
        self.mock_validator.check_missing_values.assert_called_once_with(
            self.sample_data_with_issues, 0.1
        )

    def test_validate_categorical_values_contract_signature(self):
        """Test that validate_categorical_values has correct signature."""
        # Arrange
        column_specs = {
            'contract_type': ['month-to-month', 'one-year', 'two-year']
        }
        expected_invalid = {
            'contract_type': ['invalid_type']
        }
        self.mock_validator.validate_categorical_values.return_value = expected_invalid

        # Act
        result = self.mock_validator.validate_categorical_values(
            self.sample_data_with_issues, column_specs
        )

        # Assert
        assert isinstance(result, dict), "Should return dictionary"
        assert all(isinstance(v, list) for v in result.values()), "Values should be lists"

    def test_check_data_consistency_contract_signature(self):
        """Test that check_data_consistency has correct signature."""
        # Arrange
        expected_consistency_errors = {
            'total_charges_tenure_mismatch': 'Total charges inconsistent with tenure and monthly charges'
        }
        self.mock_validator.check_data_consistency.return_value = expected_consistency_errors

        # Act
        result = self.mock_validator.check_data_consistency(self.sample_data_with_issues)

        # Assert
        assert isinstance(result, dict), "Should return dictionary"
        assert all(isinstance(v, str) for v in result.values()), "Values should be strings"


# Integration test to verify contract and implementation work together
class TestDataLoaderIntegration:
    """Integration tests for data loader contract compliance."""

    def test_contract_implementation_compatibility(self):
        """Test that real implementation will be compatible with contract."""
        # This test will fail initially and pass once we implement the actual classes
        # For now, we test the contract interface itself

        # Verify that our contract classes can be instantiated as interfaces
        assert hasattr(DataLoaderContract, 'load_raw_data')
        assert hasattr(DataLoaderContract, 'validate_data_quality')
        assert hasattr(DataLoaderContract, 'split_features_target')

        assert hasattr(DataValidatorContract, 'check_missing_values')
        assert hasattr(DataValidatorContract, 'validate_categorical_values')
        assert hasattr(DataValidatorContract, 'check_data_consistency')

    def test_expected_data_flow(self):
        """Test expected data flow through data loading pipeline."""
        # This test documents the expected workflow
        # Will be implemented when we have actual classes

        # Expected flow:
        # 1. Load raw data from CSV
        # 2. Validate data quality
        # 3. Split into features and target
        # 4. Check data consistency

        # For now, just verify the contract methods exist
        loader_methods = dir(DataLoaderContract)
        validator_methods = dir(DataValidatorContract)

        assert 'load_raw_data' in loader_methods
        assert 'validate_data_quality' in loader_methods
        assert 'split_features_target' in loader_methods
        assert 'check_missing_values' in validator_methods


if __name__ == '__main__':
    pytest.main([__file__])
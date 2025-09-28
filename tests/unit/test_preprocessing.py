"""
Unit Tests for Preprocessing Functions
Tests individual preprocessing functions in isolation for correctness and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Import preprocessing modules to test
from src.services.preprocessing_service import PreprocessingService
from src.preprocessing.feature_engineering import ChurnFeatureEngineer


class TestPreprocessingService:
    """Unit tests for PreprocessingService"""

    @pytest.fixture
    def preprocessing_service(self):
        """Create preprocessing service instance"""
        return PreprocessingService()

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, np.nan, 35, 45, np.nan],
            'income': [50000, 60000, np.nan, 80000, 70000],
            'gender': ['M', 'F', np.nan, 'M', 'F'],
            'city': ['NYC', 'LA', 'Chicago', np.nan, 'Boston']
        })

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample data with categorical variables"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'contract_type': ['Month-to-month', 'One year', 'Two year', 'Month-to-month'],
            'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Credit card'],
            'internet_service': ['DSL', 'Fiber optic', 'No', 'DSL'],
            'age': [25, 35, 45, 55]
        })

    @pytest.fixture
    def sample_numerical_data(self):
        """Create sample data with numerical variables"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'age': [25, 35, 45, 55],
            'tenure': [12, 24, 36, 48],
            'monthly_charges': [50.0, 75.5, 89.0, 45.25],
            'total_charges': [600, 1812, 3204, 2172]
        })

    def test_handle_missing_values_numerical(self, preprocessing_service, sample_data_with_missing):
        """Test handling missing values in numerical columns"""
        result_data, strategies = preprocessing_service.handle_missing_values(sample_data_with_missing)

        # Check no missing values remain
        assert result_data.isnull().sum().sum() == 0, "No missing values should remain"

        # Check strategies are documented
        assert len(strategies) > 0, "Should document missing value strategies"
        assert 'age' in strategies, "Should document strategy for age column"
        assert 'gender' in strategies, "Should document strategy for gender column"

        # Check numerical values are reasonable
        assert result_data['age'].min() >= 0, "Age should be non-negative"
        assert result_data['income'].min() >= 0, "Income should be non-negative"

    def test_handle_missing_values_categorical(self, preprocessing_service):
        """Test handling missing values in categorical columns"""
        data = pd.DataFrame({
            'category1': ['A', 'B', np.nan, 'A', 'B'],
            'category2': ['X', np.nan, 'Y', 'X', np.nan]
        })

        result_data, strategies = preprocessing_service.handle_missing_values(data)

        assert result_data.isnull().sum().sum() == 0, "No missing values should remain"
        assert 'category1' in strategies, "Should document strategy for category1"
        assert 'category2' in strategies, "Should document strategy for category2"

    def test_handle_missing_values_empty_data(self, preprocessing_service):
        """Test handling missing values with empty DataFrame"""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError):
            preprocessing_service.handle_missing_values(empty_data)

    def test_handle_missing_values_all_missing_column(self, preprocessing_service):
        """Test handling column with all missing values"""
        data = pd.DataFrame({
            'normal_col': [1, 2, 3, 4],
            'all_missing': [np.nan, np.nan, np.nan, np.nan]
        })

        result_data, strategies = preprocessing_service.handle_missing_values(data)

        # Should handle gracefully (drop column or fill with default)
        assert 'all_missing' in strategies, "Should document strategy for all-missing column"

    def test_encode_categorical_features(self, preprocessing_service, sample_categorical_data):
        """Test categorical feature encoding"""
        result_data, encoders = preprocessing_service.encode_categorical_features(sample_categorical_data)

        # Check that categorical columns are encoded
        original_categorical = sample_categorical_data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in original_categorical if col != 'customer_id']

        assert len(encoders) >= len(categorical_cols), "Should create encoders for categorical features"

        # Check that customer_id is preserved
        assert 'customer_id' in result_data.columns, "Should preserve customer_id"
        assert result_data['customer_id'].dtype == 'object', "Customer_id should remain as object"

        # Check that numerical columns are preserved
        numerical_cols = sample_categorical_data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            assert col in result_data.columns, f"Should preserve numerical column {col}"

    def test_encode_categorical_features_no_categorical(self, preprocessing_service, sample_numerical_data):
        """Test categorical encoding with no categorical features"""
        result_data, encoders = preprocessing_service.encode_categorical_features(sample_numerical_data)

        # Should handle gracefully
        assert len(encoders) == 0, "Should not create encoders when no categorical features"
        pd.testing.assert_frame_equal(result_data, sample_numerical_data, check_dtype=False)

    def test_encode_categorical_features_high_cardinality(self, preprocessing_service):
        """Test categorical encoding with high cardinality features"""
        # Create data with high cardinality categorical
        data = pd.DataFrame({
            'high_cardinality': [f'category_{i}' for i in range(100)],
            'normal_feature': np.random.randn(100)
        })

        result_data, encoders = preprocessing_service.encode_categorical_features(data)

        # Should handle high cardinality appropriately
        assert 'high_cardinality' in encoders, "Should create encoder for high cardinality feature"

    def test_scale_features(self, preprocessing_service, sample_numerical_data):
        """Test feature scaling"""
        result_data, scaler = preprocessing_service.scale_features(sample_numerical_data)

        # Check scaler is created
        assert scaler is not None, "Should create scaler"

        # Check numerical features are scaled
        numerical_cols = sample_numerical_data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'customer_id']

        if len(numerical_cols) > 0:
            for col in numerical_cols:
                if col in result_data.columns:
                    # Check that values are reasonably scaled (mean close to 0, std close to 1)
                    col_mean = result_data[col].mean()
                    col_std = result_data[col].std()
                    assert abs(col_mean) < 2, f"Scaled feature {col} should have mean close to 0"
                    assert 0.5 < col_std < 2, f"Scaled feature {col} should have std close to 1"

        # Check customer_id is preserved if present
        if 'customer_id' in sample_numerical_data.columns:
            assert 'customer_id' in result_data.columns, "Should preserve customer_id"

    def test_scale_features_no_numerical(self, preprocessing_service):
        """Test feature scaling with no numerical features"""
        data = pd.DataFrame({
            'category1': ['A', 'B', 'C'],
            'category2': ['X', 'Y', 'Z']
        })

        result_data, scaler = preprocessing_service.scale_features(data)

        # Should handle gracefully
        assert scaler is not None, "Should create scaler even with no numerical features"
        pd.testing.assert_frame_equal(result_data, data, check_dtype=False)

    def test_scale_features_single_value_column(self, preprocessing_service):
        """Test feature scaling with single-value column"""
        data = pd.DataFrame({
            'constant_col': [5, 5, 5, 5],
            'normal_col': [1, 2, 3, 4]
        })

        result_data, scaler = preprocessing_service.scale_features(data)

        # Should handle constant columns gracefully
        assert scaler is not None, "Should create scaler"
        assert result_data.shape == data.shape, "Should preserve data shape"

    def test_preprocessing_pipeline_integration(self, preprocessing_service, sample_data_with_missing):
        """Test complete preprocessing pipeline"""
        # Step 1: Handle missing values
        cleaned_data, missing_strategies = preprocessing_service.handle_missing_values(sample_data_with_missing)

        # Step 2: Encode categorical features
        encoded_data, encoders = preprocessing_service.encode_categorical_features(cleaned_data)

        # Step 3: Scale features
        scaled_data, scaler = preprocessing_service.scale_features(encoded_data)

        # Verify pipeline completion
        assert scaled_data.isnull().sum().sum() == 0, "Final data should have no missing values"
        assert len(missing_strategies) > 0, "Should document missing value strategies"
        assert scaler is not None, "Should create scaler"

        # Verify data integrity
        assert len(scaled_data) == len(sample_data_with_missing), "Should preserve number of rows"


class TestChurnFeatureEngineer:
    """Unit tests for ChurnFeatureEngineer"""

    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return ChurnFeatureEngineer(random_state=42)

    @pytest.fixture
    def sample_churn_data(self):
        """Create sample churn dataset"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, 35, 45, 65, 30],
            'tenure': [12, 24, 36, 48, 6],
            'monthly_charges': [50.0, 75.5, 89.0, 45.25, 95.0],
            'total_charges': [600, 1812, 3204, 2172, 570],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'senior_citizen': [0, 0, 0, 1, 0],
            'contract_type': ['Month-to-month', 'One year', 'Two year', 'One year', 'Month-to-month'],
            'internet_service': ['DSL', 'Fiber optic', 'DSL', 'No', 'Fiber optic'],
            'streaming_tv': ['Yes', 'No', 'Yes', 'No internet service', 'Yes'],
            'streaming_movies': ['No', 'Yes', 'Yes', 'No internet service', 'No'],
            'tech_support': ['No', 'Yes', 'No', 'No internet service', 'No']
        })

    def test_create_demographic_features(self, feature_engineer, sample_churn_data):
        """Test demographic feature creation"""
        result = feature_engineer.create_demographic_features(sample_churn_data.copy())

        # Check that new demographic features are created
        expected_features = ['age_group', 'age_squared']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check age groups are reasonable
        assert result['age_group'].isin(['Young', 'Middle-aged', 'Senior', 'Elder']).all(), "Age groups should be valid"

        # Check age squared calculation
        assert (result['age_squared'] == result['age'] ** 2).all(), "Age squared should be correctly calculated"

    def test_create_tenure_features(self, feature_engineer, sample_churn_data):
        """Test tenure feature creation"""
        result = feature_engineer.create_tenure_features(sample_churn_data.copy())

        # Check that new tenure features are created
        expected_features = ['tenure_years', 'tenure_group', 'tenure_squared']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check tenure years calculation
        assert (result['tenure_years'] == result['tenure'] / 12).all(), "Tenure years should be correctly calculated"

        # Check tenure groups are reasonable
        assert result['tenure_group'].isin(['New', 'Established', 'Loyal']).all(), "Tenure groups should be valid"

    def test_create_financial_features(self, feature_engineer, sample_churn_data):
        """Test financial feature creation"""
        result = feature_engineer.create_financial_features(sample_churn_data.copy())

        # Check that new financial features are created
        expected_features = ['charges_per_tenure', 'high_value_customer']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check charges per tenure calculation
        expected_charges_per_tenure = result['total_charges'] / (result['tenure'] + 1)  # +1 to avoid division by zero
        assert np.allclose(result['charges_per_tenure'], expected_charges_per_tenure, rtol=1e-5), \
            "Charges per tenure should be correctly calculated"

        # Check high value customer is binary
        assert result['high_value_customer'].isin([0, 1]).all(), "High value customer should be binary"

    def test_create_service_features(self, feature_engineer, sample_churn_data):
        """Test service feature creation"""
        result = feature_engineer.create_service_features(sample_churn_data.copy())

        # Check that new service features are created
        expected_features = ['fiber_optic_user', 'no_internet_services']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check fiber optic user is binary
        assert result['fiber_optic_user'].isin([0, 1]).all(), "Fiber optic user should be binary"

        # Check no internet services calculation
        fiber_users = result['internet_service'] == 'Fiber optic'
        assert (result['fiber_optic_user'] == fiber_users.astype(int)).all(), \
            "Fiber optic user should match internet service"

    def test_create_behavioral_features(self, feature_engineer, sample_churn_data):
        """Test behavioral feature creation"""
        result = feature_engineer.create_behavioral_features(sample_churn_data.copy())

        # Check that new behavioral features are created
        expected_features = ['streaming_services_count', 'has_streaming_services']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check streaming services count is reasonable
        assert result['streaming_services_count'].min() >= 0, "Streaming count should be non-negative"
        assert result['streaming_services_count'].max() <= 2, "Max streaming count should be 2"

        # Check has streaming services is binary
        assert result['has_streaming_services'].isin([0, 1]).all(), "Has streaming services should be binary"

    def test_create_interaction_features(self, feature_engineer, sample_churn_data):
        """Test interaction feature creation"""
        result = feature_engineer.create_interaction_features(sample_churn_data.copy())

        # Check that new interaction features are created
        expected_features = ['age_tenure_interaction', 'charges_contract_interaction']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check age-tenure interaction
        expected_interaction = result['age'] * result['tenure']
        assert (result['age_tenure_interaction'] == expected_interaction).all(), \
            "Age-tenure interaction should be correctly calculated"

    def test_create_risk_features(self, feature_engineer, sample_churn_data):
        """Test risk feature creation"""
        result = feature_engineer.create_risk_features(sample_churn_data.copy())

        # Check that new risk features are created
        expected_features = ['risk_score', 'high_risk_customer']
        for feature in expected_features:
            assert feature in result.columns, f"Should create {feature}"

        # Check risk score is reasonable
        assert result['risk_score'].min() >= 0, "Risk score should be non-negative"

        # Check high risk customer is binary
        assert result['high_risk_customer'].isin([0, 1]).all(), "High risk customer should be binary"

    def test_apply_feature_engineering_complete(self, feature_engineer, sample_churn_data):
        """Test complete feature engineering pipeline"""
        result = feature_engineer.apply_feature_engineering(sample_churn_data)

        # Check that result has more features than original
        assert result.shape[1] > sample_churn_data.shape[1], "Should add new features"

        # Check that original features are preserved
        for col in sample_churn_data.columns:
            assert col in result.columns, f"Should preserve original column {col}"

        # Check no missing values introduced
        original_missing = sample_churn_data.isnull().sum().sum()
        result_missing = result.isnull().sum().sum()
        assert result_missing == original_missing, "Should not introduce new missing values"

    def test_get_feature_summary(self, feature_engineer, sample_churn_data):
        """Test feature summary generation"""
        # Apply feature engineering first
        feature_engineer.apply_feature_engineering(sample_churn_data)

        # Get summary
        summary = feature_engineer.get_feature_summary()

        # Check summary structure
        assert 'total_engineered_features' in summary, "Should include total engineered features"
        assert 'feature_categories' in summary, "Should include feature categories"

        # Check feature categories
        expected_categories = ['demographic', 'tenure', 'financial', 'service', 'behavioral', 'interaction', 'risk']
        for category in expected_categories:
            assert category in summary['feature_categories'], f"Should include {category} category"

        # Check that totals are reasonable
        assert summary['total_engineered_features'] > 0, "Should have engineered features"

    def test_feature_engineering_edge_cases(self, feature_engineer):
        """Test feature engineering with edge cases"""
        # Test with minimal data
        minimal_data = pd.DataFrame({
            'age': [25],
            'tenure': [0],  # Zero tenure
            'monthly_charges': [0],  # Zero charges
            'total_charges': [0]
        })

        result = feature_engineer.apply_feature_engineering(minimal_data)

        # Should handle gracefully without errors
        assert len(result) == 1, "Should preserve single row"
        assert result.shape[1] > minimal_data.shape[1], "Should add features even with edge cases"

    def test_feature_engineering_reproducibility(self, sample_churn_data):
        """Test that feature engineering is reproducible"""
        engineer1 = ChurnFeatureEngineer(random_state=42)
        engineer2 = ChurnFeatureEngineer(random_state=42)

        result1 = engineer1.apply_feature_engineering(sample_churn_data.copy())
        result2 = engineer2.apply_feature_engineering(sample_churn_data.copy())

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)

    def test_feature_engineering_missing_columns(self, feature_engineer):
        """Test feature engineering with missing expected columns"""
        incomplete_data = pd.DataFrame({
            'age': [25, 35, 45],
            'tenure': [12, 24, 36]
            # Missing many expected columns
        })

        # Should handle gracefully or raise informative error
        try:
            result = feature_engineer.apply_feature_engineering(incomplete_data)
            # If successful, check basic properties
            assert len(result) == len(incomplete_data), "Should preserve number of rows"
        except KeyError as e:
            # If it raises KeyError, that's acceptable for missing required columns
            assert "column" in str(e).lower(), "Error should mention missing column"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
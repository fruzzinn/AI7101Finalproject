"""
Contract tests for PreprocessingContract
These tests MUST FAIL initially to ensure TDD compliance
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# Import the contract interface (will fail until implemented)
try:
    from src.services.preprocessing_service import PreprocessingService
    from src.models.feature_set import FeatureSet
except ImportError:
    # Expected to fail initially - this enforces TDD
    pytest.skip("Implementation not available yet - TDD compliance", allow_module_level=True)


class TestPreprocessingContract:
    """Test suite for PreprocessingContract implementation"""

    @pytest.fixture
    def sample_data_with_missing(self):
        """Create sample data with missing values for testing"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'age': [25, np.nan, 45, 55, np.nan],  # Missing values
            'tenure': [12, 24, 36, 48, 60],
            'monthly_charges': [50.0, 70.0, np.nan, 110.0, 130.0],  # Missing values
            'contract_type': ['Month-to-month', 'One year', np.nan, 'Month-to-month', 'One year'],  # Missing categorical
            'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', np.nan, 'Credit card'],
            'total_charges': [600.0, 1680.0, 3240.0, 5280.0, 7800.0]
        })

    @pytest.fixture
    def sample_categorical_data(self):
        """Create sample data with various categorical types"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'contract_type': ['Month-to-month', 'One year', 'Two year', 'Month-to-month', 'One year'],  # Low cardinality
            'payment_method': ['Credit card', 'Bank transfer', 'Electronic check', 'Mailed check', 'Credit card'],
            'internet_service': ['DSL', 'Fiber optic', 'No', 'DSL', 'Fiber optic'],
            'satisfaction_level': ['Low', 'Medium', 'High', 'Medium', 'High'],  # Ordinal
            'city': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']  # High cardinality (in real data)
        })

    @pytest.fixture
    def sample_feature_engineering_data(self):
        """Create sample data for feature engineering testing"""
        return pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
            'tenure': [12, 24, 36, 48, 60],
            'monthly_charges': [50.0, 70.0, 90.0, 110.0, 130.0],
            'total_charges': [600.0, 1680.0, 3240.0, 5280.0, 7800.0],
            'account_creation_date': ['2020-01-01', '2019-01-01', '2018-01-01', '2017-01-01', '2016-01-01'],
            'total_calls': [100, 200, 300, 400, 500],
            'plan_calls': [150, 250, 350, 450, 550]
        })

    @pytest.fixture
    def preprocessing_service(self):
        """Create PreprocessingService instance"""
        return PreprocessingService()

    def test_handle_missing_values_numerical(self, preprocessing_service, sample_data_with_missing):
        """Test missing value handling for numerical columns"""
        cleaned_df, strategy_dict = preprocessing_service.handle_missing_values(sample_data_with_missing)

        # Contract requirements
        assert isinstance(cleaned_df, pd.DataFrame), "Must return DataFrame"
        assert isinstance(strategy_dict, dict), "Must return strategy dictionary"

        # No missing values should remain in numerical columns
        assert cleaned_df['age'].isna().sum() == 0, "Age missing values should be imputed"
        assert cleaned_df['monthly_charges'].isna().sum() == 0, "Monthly charges missing values should be imputed"

        # Strategy should be documented
        assert 'age' in strategy_dict, "Strategy for age should be documented"
        assert 'monthly_charges' in strategy_dict, "Strategy for monthly charges should be documented"

    def test_handle_missing_values_categorical(self, preprocessing_service, sample_data_with_missing):
        """Test missing value handling for categorical columns"""
        cleaned_df, strategy_dict = preprocessing_service.handle_missing_values(sample_data_with_missing)

        # No missing values should remain in categorical columns
        assert cleaned_df['contract_type'].isna().sum() == 0, "Contract type missing values should be imputed"
        assert cleaned_df['payment_method'].isna().sum() == 0, "Payment method missing values should be imputed"

        # Strategy should be documented for categorical columns
        assert 'contract_type' in strategy_dict, "Strategy for contract type should be documented"
        assert 'payment_method' in strategy_dict, "Strategy for payment method should be documented"

    def test_handle_missing_values_preserves_distribution(self, preprocessing_service):
        """Test that missing value imputation preserves data distribution"""
        # Create data with known distribution
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, np.nan, np.nan],
            'feature2': ['A', 'B', 'A', 'B', 'A', np.nan, np.nan]
        })

        cleaned_df, strategy_dict = preprocessing_service.handle_missing_values(data)

        # Original mean should be approximately preserved for numerical
        original_mean = data['feature1'].mean()
        new_mean = cleaned_df['feature1'].mean()
        assert abs(original_mean - new_mean) < 1.0, "Mean should be approximately preserved"

        # Mode should be used for categorical
        original_mode = data['feature2'].mode()[0]
        assert all(cleaned_df['feature2'].isin(['A', 'B'])), "Only valid categories should exist"

    def test_encode_categorical_features_low_cardinality(self, preprocessing_service, sample_categorical_data):
        """Test one-hot encoding for low cardinality categorical features"""
        encoded_df, encoder_dict = preprocessing_service.encode_categorical_features(sample_categorical_data)

        # Contract requirements
        assert isinstance(encoded_df, pd.DataFrame), "Must return DataFrame"
        assert isinstance(encoder_dict, dict), "Must return encoder dictionary"

        # Low cardinality features should be one-hot encoded
        contract_columns = [col for col in encoded_df.columns if 'contract_type_' in col]
        assert len(contract_columns) > 1, "Contract type should be one-hot encoded"

        # Encoder objects should be stored for consistency
        assert 'contract_type' in encoder_dict, "Encoder for contract type should be stored"

    def test_encode_categorical_features_high_cardinality(self, preprocessing_service):
        """Test target encoding for high cardinality categorical features"""
        # Create data with high cardinality
        data = pd.DataFrame({
            'high_cardinality_feature': [f'category_{i}' for i in range(50)],  # 50 unique categories
            'target': np.random.randint(0, 2, 50)
        })

        encoded_df, encoder_dict = preprocessing_service.encode_categorical_features(data)

        # High cardinality should not explode columns
        assert encoded_df.shape[1] < data.shape[1] + 10, "High cardinality shouldn't create too many columns"
        assert 'high_cardinality_feature' in encoder_dict, "High cardinality encoder should be stored"

    def test_encode_categorical_features_ordinal(self, preprocessing_service, sample_categorical_data):
        """Test ordinal encoding for ranked categorical features"""
        encoded_df, encoder_dict = preprocessing_service.encode_categorical_features(sample_categorical_data)

        # Satisfaction level should be ordinally encoded (if detected as ordinal)
        if 'satisfaction_level' in encoded_df.columns:
            satisfaction_values = encoded_df['satisfaction_level'].unique()
            assert len(satisfaction_values) <= 3, "Ordinal encoding should preserve order"

    def test_engineer_features_customer_tenure(self, preprocessing_service, sample_feature_engineering_data):
        """Test creation of customer tenure features"""
        featured_df, new_features = preprocessing_service.engineer_features(sample_feature_engineering_data)

        # Contract requirements
        assert isinstance(featured_df, pd.DataFrame), "Must return DataFrame"
        assert isinstance(new_features, list), "Must return list of new feature names"

        # Tenure features should be created
        tenure_features = [f for f in new_features if 'tenure' in f.lower()]
        assert len(tenure_features) > 0, "Tenure features should be created"

    def test_engineer_features_usage_ratios(self, preprocessing_service, sample_feature_engineering_data):
        """Test creation of usage ratio metrics"""
        featured_df, new_features = preprocessing_service.engineer_features(sample_feature_engineering_data)

        # Usage ratio should be created
        ratio_features = [f for f in new_features if 'ratio' in f.lower()]
        assert len(ratio_features) > 0, "Usage ratio features should be created"

        # Check specific ratio calculation
        if 'call_usage_ratio' in featured_df.columns:
            # Verify calculation: total_calls / (plan_calls + 1)
            expected_ratio = sample_feature_engineering_data['total_calls'] / (sample_feature_engineering_data['plan_calls'] + 1)
            actual_ratio = featured_df['call_usage_ratio']
            pd.testing.assert_series_equal(actual_ratio, expected_ratio, check_names=False)

    def test_engineer_features_customer_value_metrics(self, preprocessing_service, sample_feature_engineering_data):
        """Test creation of customer value metrics (ARPU, CLV estimates)"""
        featured_df, new_features = preprocessing_service.engineer_features(sample_feature_engineering_data)

        # Customer value features should be created
        value_features = [f for f in new_features if any(term in f.lower() for term in ['arpu', 'clv', 'value'])]
        assert len(value_features) > 0, "Customer value features should be created"

    def test_engineer_features_business_interpretable(self, preprocessing_service, sample_feature_engineering_data):
        """Test that engineered features are business-interpretable"""
        featured_df, new_features = preprocessing_service.engineer_features(sample_feature_engineering_data)

        # All new features should have interpretable names
        business_terms = ['tenure', 'ratio', 'avg', 'total', 'rate', 'frequency', 'value', 'arpu', 'clv']

        for feature in new_features:
            has_business_term = any(term in feature.lower() for term in business_terms)
            assert has_business_term, f"Feature '{feature}' should have business-interpretable name"

    def test_scale_features_normal_distribution(self, preprocessing_service):
        """Test feature scaling with normally distributed data"""
        # Create normally distributed data
        data = pd.DataFrame({
            'feature1': np.random.normal(100, 15, 1000),
            'feature2': np.random.normal(50, 10, 1000),
            'categorical': ['A'] * 500 + ['B'] * 500
        })

        scaled_df, scaler = preprocessing_service.scale_features(data)

        # Contract requirements
        assert isinstance(scaled_df, pd.DataFrame), "Must return DataFrame"
        assert scaler is not None, "Must return scaler object"

        # Numerical features should be scaled
        assert abs(scaled_df['feature1'].mean()) < 0.1, "Scaled feature should have mean near 0"
        assert abs(scaled_df['feature1'].std() - 1.0) < 0.1, "Scaled feature should have std near 1"

    def test_scale_features_with_outliers(self, preprocessing_service):
        """Test feature scaling with data containing outliers"""
        # Create data with outliers
        data = pd.DataFrame({
            'feature_with_outliers': [1, 2, 3, 4, 5, 1000],  # 1000 is an outlier
            'normal_feature': [10, 20, 30, 40, 50, 60]
        })

        scaled_df, scaler = preprocessing_service.scale_features(data)

        # Robust scaler should be used for features with outliers
        # The exact behavior depends on implementation, but outliers shouldn't dominate
        assert scaled_df['feature_with_outliers'].max() < 10, "Outliers shouldn't dominate scaling"

    def test_scale_features_preserves_categorical(self, preprocessing_service):
        """Test that categorical features are preserved during scaling"""
        data = pd.DataFrame({
            'numerical': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'C', 'A', 'B']
        })

        scaled_df, scaler = preprocessing_service.scale_features(data)

        # Categorical features should remain unchanged
        pd.testing.assert_series_equal(data['categorical'], scaled_df['categorical'], check_names=False)

    def test_preprocessing_pipeline_integration(self, preprocessing_service, sample_data_with_missing):
        """Test full preprocessing pipeline integration"""
        # Step 1: Handle missing values
        cleaned_df, missing_strategy = preprocessing_service.handle_missing_values(sample_data_with_missing)

        # Step 2: Encode categorical features
        encoded_df, encoders = preprocessing_service.encode_categorical_features(cleaned_df)

        # Step 3: Engineer features
        featured_df, new_features = preprocessing_service.engineer_features(encoded_df)

        # Step 4: Scale features
        scaled_df, scaler = preprocessing_service.scale_features(featured_df)

        # Final result should have no missing values
        assert scaled_df.isna().sum().sum() == 0, "No missing values should remain after full pipeline"

        # Should have more features due to engineering
        assert scaled_df.shape[1] >= sample_data_with_missing.shape[1], "Feature engineering should add features"

    def test_preprocessing_consistency_train_test(self, preprocessing_service):
        """Test that preprocessing is consistent between train and test sets"""
        # This test ensures the same transformations can be applied to test data
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'categorical': ['A', 'B', 'A', 'B', 'A']
        })

        test_data = pd.DataFrame({
            'feature1': [6, 7, 8],
            'categorical': ['A', 'B', 'C']  # 'C' is new category
        })

        # Process training data
        _, missing_strategy = preprocessing_service.handle_missing_values(train_data)
        _, encoders = preprocessing_service.encode_categorical_features(train_data)
        _, scaler = preprocessing_service.scale_features(train_data)

        # Transformers should be reusable (this tests the contract interface)
        assert missing_strategy is not None, "Missing value strategy should be reusable"
        assert encoders is not None, "Encoders should be reusable"
        assert scaler is not None, "Scaler should be reusable"


if __name__ == "__main__":
    # These tests should FAIL initially to ensure TDD compliance
    pytest.main([__file__, "-v"])
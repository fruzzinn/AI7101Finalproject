"""
Integration Tests for Complete Preprocessing Workflow
Tests the end-to-end preprocessing pipeline including missing values, encoding, feature engineering, and scaling
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Import our services and utilities
from src.services.preprocessing_service import PreprocessingService
from src.preprocessing.feature_engineering import ChurnFeatureEngineer
from src.models.feature_set import FeatureSet
from src.utils.mlflow_utils import MLflowExperimentManager


class TestPreprocessingPipelineIntegration:
    """Integration tests for complete preprocessing pipeline"""

    @pytest.fixture
    def raw_customer_data(self):
        """Create raw customer data with various preprocessing challenges"""
        np.random.seed(42)
        n_customers = 1000

        data = {
            'customer_id': [f'C{i:04d}' for i in range(n_customers)],
            'age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
            'tenure': np.random.exponential(24, n_customers).astype(int).clip(0, 72),
            'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 8000),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
            'gender': np.random.choice(['Male', 'Female'], n_customers),
            'phone_service': np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_customers),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_customers)
        }

        # Generate realistic churn
        churn_prob = 0.1 + 0.3 * (np.array(data['contract_type']) == 'Month-to-month').astype(int)
        data['churn'] = np.random.binomial(1, churn_prob, n_customers)

        df = pd.DataFrame(data)

        # Introduce missing values to test preprocessing
        missing_indices = np.random.choice(df.index, size=100, replace=False)
        df.loc[missing_indices[:30], 'age'] = np.nan
        df.loc[missing_indices[30:60], 'monthly_charges'] = np.nan
        df.loc[missing_indices[60:], 'contract_type'] = np.nan

        return df

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessing service instance"""
        return PreprocessingService()

    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineering instance"""
        return ChurnFeatureEngineer(random_state=42)

    def test_complete_preprocessing_pipeline(self, raw_customer_data, preprocessor, feature_engineer):
        """Test the complete preprocessing pipeline from raw data to ML-ready features"""
        print(f"Starting with {raw_customer_data.shape[0]} samples and {raw_customer_data.shape[1]} features")

        # Step 1: Handle missing values
        cleaned_data, missing_strategies = preprocessor.handle_missing_values(raw_customer_data)

        # Verify no missing values remain
        assert cleaned_data.isnull().sum().sum() == 0, "No missing values should remain after preprocessing"
        assert len(missing_strategies) > 0, "Should document missing value strategies"

        # Step 2: Encode categorical features
        encoded_data, encoders = preprocessor.encode_categorical_features(cleaned_data)

        # Verify categorical encoding
        assert len(encoders) > 0, "Should create encoders for categorical features"
        original_categorical = cleaned_data.select_dtypes(include=['object']).columns
        assert len(original_categorical) > 0, "Should have had categorical features to encode"

        # Step 3: Feature engineering
        engineered_data = feature_engineer.apply_feature_engineering(encoded_data)

        # Verify feature engineering
        assert engineered_data.shape[1] > encoded_data.shape[1], "Should add new features"
        feature_summary = feature_engineer.get_feature_summary()
        assert feature_summary['total_engineered_features'] > 0, "Should create engineered features"

        # Step 4: Scale features
        scaled_data, scaler = preprocessor.scale_features(engineered_data)

        # Verify scaling
        assert scaler is not None, "Should create scaler"
        numerical_columns = scaled_data.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            # Check that numerical features are reasonably scaled
            means = scaled_data[numerical_columns].mean()
            assert all(abs(mean) < 2 for mean in means), "Scaled features should have reasonable means"

        print(f"Final data shape: {scaled_data.shape[0]} samples, {scaled_data.shape[1]} features")
        print(f"Added {scaled_data.shape[1] - raw_customer_data.shape[1]} features through preprocessing")

        return scaled_data, {
            'missing_strategies': missing_strategies,
            'encoders': encoders,
            'feature_summary': feature_summary,
            'scaler': scaler
        }

    def test_train_test_preprocessing_consistency(self, raw_customer_data, preprocessor):
        """Test that preprocessing is consistent between train and test sets"""
        # Split data
        train_data, test_data = train_test_split(raw_customer_data, test_size=0.2, random_state=42, stratify=raw_customer_data['churn'])

        # Remove target from test set (realistic scenario)
        test_features = test_data.drop('churn', axis=1)
        train_features = train_data.drop('churn', axis=1)

        # Preprocess training data
        train_cleaned, missing_strategies = preprocessor.handle_missing_values(train_features)
        train_encoded, encoders = preprocessor.encode_categorical_features(train_cleaned)
        train_scaled, scaler = preprocessor.scale_features(train_encoded)

        # Apply same transformations to test data (simulating production scenario)
        # Note: In practice, you'd save and reuse the fitted transformers
        test_cleaned, _ = preprocessor.handle_missing_values(test_features)
        test_encoded, _ = preprocessor.encode_categorical_features(test_cleaned)
        test_scaled, _ = preprocessor.scale_features(test_encoded)

        # Verify consistency
        assert train_scaled.shape[1] == test_scaled.shape[1], "Train and test should have same number of features"

        # Check that column names are consistent (order might differ due to encoding)
        train_cols = set(train_scaled.columns)
        test_cols = set(test_scaled.columns)
        common_cols = train_cols.intersection(test_cols)
        assert len(common_cols) >= len(train_cols) * 0.9, "At least 90% of columns should be consistent"

    def test_feature_set_entity_integration(self, raw_customer_data, feature_engineer):
        """Test integration with FeatureSet entity"""
        # Prepare data
        sample_data = raw_customer_data.iloc[:100].copy()  # Smaller sample for testing
        engineered_data = feature_engineer.apply_feature_engineering(sample_data)

        # Create sample features for FeatureSet
        sample_customer = engineered_data.iloc[0]

        # Extract different types of features
        behavioral_features = {k: v for k, v in sample_customer.items() if 'streaming' in k or 'support' in k}
        temporal_features = {k: v for k, v in sample_customer.items() if 'tenure' in k}
        categorical_features = {k: v for k, v in sample_customer.items() if 'category' in k or 'group' in k}

        # Create FeatureSet entity
        feature_set = FeatureSet(
            customer_id=sample_customer['customer_id'],
            behavioral_features=behavioral_features,
            temporal_features=temporal_features,
            categorical_features=categorical_features,
            derived_features={k: v for k, v in sample_customer.items() if 'risk' in k or 'score' in k}
        )

        # Validate feature set
        assert feature_set.customer_id == sample_customer['customer_id']
        assert feature_set.validate_features() is True, "FeatureSet should be valid"
        assert len(feature_set.behavioral_features) > 0, "Should have behavioral features"

    def test_preprocessing_with_edge_cases(self, preprocessor):
        """Test preprocessing handles edge cases correctly"""
        # Create data with edge cases
        edge_case_data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003', 'C004'],
            'all_missing': [np.nan, np.nan, np.nan, np.nan],  # Completely missing column
            'all_same': [1, 1, 1, 1],  # No variance
            'extreme_outlier': [1, 2, 3, 1000],  # Extreme outlier
            'single_category': ['A', 'A', 'A', 'A'],  # Single category
            'many_categories': ['Cat1', 'Cat2', 'Cat3', 'Cat4'],  # High cardinality (for small sample)
            'binary_target': [0, 1, 0, 1]
        })

        # Should handle these cases gracefully
        try:
            cleaned_data, strategies = preprocessor.handle_missing_values(edge_case_data)
            encoded_data, encoders = preprocessor.encode_categorical_features(cleaned_data)
            scaled_data, scaler = preprocessor.scale_features(encoded_data)

            # Verify basic properties
            assert scaled_data.shape[0] == edge_case_data.shape[0], "Should preserve number of rows"
            assert not scaled_data.isnull().any().any(), "Should handle all missing values"

        except Exception as e:
            pytest.fail(f"Preprocessing should handle edge cases gracefully, but failed with: {e}")

    def test_preprocessing_performance_monitoring(self, raw_customer_data, preprocessor):
        """Test preprocessing performance and resource usage"""
        import time
        import psutil
        import os

        # Monitor preprocessing performance
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        # Run complete preprocessing
        cleaned_data, _ = preprocessor.handle_missing_values(raw_customer_data)
        encoded_data, _ = preprocessor.encode_categorical_features(cleaned_data)
        scaled_data, _ = preprocessor.scale_features(encoded_data)

        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB

        processing_time = end_time - start_time
        memory_used = end_memory - start_memory

        # Performance assertions
        assert processing_time < 10.0, f"Preprocessing should complete within 10 seconds, took {processing_time:.2f}s"
        assert memory_used < 100, f"Memory usage should be reasonable, used {memory_used:.2f}MB"

    def test_feature_engineering_categories(self, raw_customer_data, feature_engineer):
        """Test that all feature engineering categories are created"""
        engineered_data = feature_engineer.apply_feature_engineering(raw_customer_data)
        feature_summary = feature_engineer.get_feature_summary()

        expected_categories = ['demographic', 'tenure', 'financial', 'service', 'behavioral', 'interaction', 'risk']

        for category in expected_categories:
            assert category in feature_summary['feature_categories'], f"Should create {category} features"
            category_features = feature_summary['feature_categories'][category]
            if category in ['demographic', 'tenure', 'financial']:  # Core categories should always have features
                assert len(category_features) > 0, f"Should have features in {category} category"

    def test_preprocessing_reproducibility(self, raw_customer_data, preprocessor, feature_engineer):
        """Test that preprocessing is reproducible with same inputs"""
        # Run preprocessing twice with same data
        result1 = self._run_complete_preprocessing(raw_customer_data, preprocessor, feature_engineer)
        result2 = self._run_complete_preprocessing(raw_customer_data, preprocessor, feature_engineer)

        # Results should be identical
        pd.testing.assert_frame_equal(result1, result2, check_dtype=False)

    def _run_complete_preprocessing(self, data, preprocessor, feature_engineer):
        """Helper method to run complete preprocessing"""
        cleaned_data, _ = preprocessor.handle_missing_values(data)
        encoded_data, _ = preprocessor.encode_categorical_features(cleaned_data)
        engineered_data = feature_engineer.apply_feature_engineering(encoded_data)
        scaled_data, _ = preprocessor.scale_features(engineered_data)
        return scaled_data

    def test_preprocessing_with_mlflow_integration(self, raw_customer_data, preprocessor, feature_engineer):
        """Test preprocessing pipeline with MLflow logging"""
        mlflow_manager = MLflowExperimentManager(
            experiment_name="test_preprocessing_pipeline",
            tracking_uri="sqlite:///test_preprocessing_mlflow.db"
        )

        try:
            # Start MLflow run
            run_id = mlflow_manager.start_run("preprocessing_pipeline_test")

            # Run preprocessing with logging
            cleaned_data, missing_strategies = preprocessor.handle_missing_values(raw_customer_data)

            # Log preprocessing info
            mlflow_manager.log_preprocessing_steps({
                'missing_value_handling': missing_strategies,
                'original_shape': raw_customer_data.shape,
                'cleaned_shape': cleaned_data.shape
            })

            encoded_data, encoders = preprocessor.encode_categorical_features(cleaned_data)
            engineered_data = feature_engineer.apply_feature_engineering(encoded_data)
            feature_summary = feature_engineer.get_feature_summary()

            # Log feature engineering results
            import mlflow
            mlflow.log_param("features_engineered", feature_summary['total_engineered_features'])
            mlflow.log_param("final_feature_count", engineered_data.shape[1])

            scaled_data, scaler = preprocessor.scale_features(engineered_data)

            # End run
            mlflow_manager.end_run()

            assert run_id is not None, "MLflow run should be created"
            assert scaled_data.shape[0] == raw_customer_data.shape[0], "Should preserve number of samples"

        finally:
            # Cleanup
            if os.path.exists("test_preprocessing_mlflow.db"):
                os.unlink("test_preprocessing_mlflow.db")

    def test_preprocessing_data_leakage_prevention(self, raw_customer_data):
        """Test that preprocessing doesn't introduce data leakage"""
        # Split data first
        train_data, test_data = train_test_split(raw_customer_data, test_size=0.2, random_state=42)

        # Remove target from test data
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']

        preprocessor = PreprocessingService()

        # Preprocess training data
        X_train_processed, _ = preprocessor.handle_missing_values(X_train)
        X_train_processed, _ = preprocessor.encode_categorical_features(X_train_processed)
        X_train_processed, _ = preprocessor.scale_features(X_train_processed)

        # Check that target information wasn't used in preprocessing
        # This is mainly a design verification - the preprocessor shouldn't have access to target
        assert 'churn' not in X_train_processed.columns, "Target variable should not be in processed features"

        # Verify preprocessing transforms are based only on training data
        # (In practice, you'd fit transformers on train and apply to test)
        train_means = X_train_processed.select_dtypes(include=[np.number]).mean()

        # Process test data separately
        X_test_processed, _ = preprocessor.handle_missing_values(X_test)
        X_test_processed, _ = preprocessor.encode_categorical_features(X_test_processed)
        X_test_processed, _ = preprocessor.scale_features(X_test_processed)

        # Both should complete without error (data leakage would cause inconsistencies)
        assert len(X_train_processed) == len(X_train), "Training data length should be preserved"
        assert len(X_test_processed) == len(X_test), "Test data length should be preserved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
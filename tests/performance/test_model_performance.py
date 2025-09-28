"""
Performance Validation Tests
Tests that validate model performance meets the 85% accuracy target and other performance criteria
"""

import pytest
import pandas as pd
import numpy as np
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import our services
from src.services.model_service import HighPerformanceModelService
from src.services.preprocessing_service import PreprocessingService
from src.preprocessing.feature_engineering import ChurnFeatureEngineer
from src.utils.cv_pipeline import SMOTECrossValidator
from src.evaluation.metrics import ChurnModelEvaluator


class TestModelPerformanceTargets:
    """Test model performance against defined targets"""

    @pytest.fixture
    def large_realistic_dataset(self):
        """Create large realistic dataset for performance testing"""
        np.random.seed(42)
        n_customers = 5000

        # Create realistic customer data with proper correlations
        data = {}

        # Demographics
        data['customer_id'] = [f'C{i:05d}' for i in range(n_customers)]
        data['age'] = np.random.normal(45, 15, n_customers).astype(int).clip(18, 80)
        data['senior_citizen'] = (data['age'] >= 65).astype(int)
        data['gender'] = np.random.choice(['Male', 'Female'], n_customers)

        # Account information
        data['tenure'] = np.random.exponential(24, n_customers).astype(int).clip(1, 72)
        data['partner'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.5, 0.5])
        data['dependents'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7])

        # Services
        data['phone_service'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.9, 0.1])
        data['multiple_lines'] = np.where(
            np.array(data['phone_service']) == 'Yes',
            np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'No phone service'
        )

        data['internet_service'] = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers, p=[0.35, 0.45, 0.2])

        # Internet-dependent services
        internet_mask = np.array(data['internet_service']) != 'No'
        data['online_security'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'No internet service'
        )
        data['online_backup'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.35, 0.65]),
            'No internet service'
        )
        data['device_protection'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'No internet service'
        )
        data['tech_support'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
            'No internet service'
        )
        data['streaming_tv'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'No internet service'
        )
        data['streaming_movies'] = np.where(
            internet_mask,
            np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
            'No internet service'
        )

        # Contract and billing
        data['contract_type'] = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                               n_customers, p=[0.55, 0.25, 0.2])
        data['paperless_billing'] = np.random.choice(['Yes', 'No'], n_customers, p=[0.6, 0.4])
        data['payment_method'] = np.random.choice([
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n_customers, p=[0.35, 0.2, 0.25, 0.2])

        # Financial data with realistic correlations
        base_monthly = 45 + 25 * (data['internet_service'] == 'Fiber optic').astype(int)
        base_monthly += 10 * (data['streaming_tv'] == 'Yes').astype(int)
        base_monthly += 10 * (data['streaming_movies'] == 'Yes').astype(int)
        data['monthly_charges'] = base_monthly + np.random.normal(0, 10, n_customers)
        data['monthly_charges'] = np.clip(data['monthly_charges'], 20, 120)

        data['total_charges'] = data['monthly_charges'] * data['tenure'] + np.random.normal(0, 200, n_customers)
        data['total_charges'] = np.clip(data['total_charges'], 0, 8000)

        # Create realistic churn with business logic
        churn_probability = (
            0.05 +  # Base churn rate
            0.35 * (np.array(data['contract_type']) == 'Month-to-month').astype(int) +
            0.25 * (np.array(data['tenure']) <= 6).astype(int) +
            0.20 * (np.array(data['monthly_charges']) > 80).astype(int) +
            0.15 * (np.array(data['payment_method']) == 'Electronic check').astype(int) +
            0.10 * (np.array(data['senior_citizen']) == 1).astype(int) -
            0.15 * (np.array(data['contract_type']) == 'Two year').astype(int) -
            0.10 * (np.array(data['tech_support']) == 'Yes').astype(int)
        )
        churn_probability = np.clip(churn_probability, 0.02, 0.8)
        data['churn'] = np.random.binomial(1, churn_probability, n_customers)

        return pd.DataFrame(data)

    @pytest.fixture
    def model_service(self):
        """Create model service instance"""
        return HighPerformanceModelService()

    @pytest.fixture
    def preprocessing_service(self):
        """Create preprocessing service instance"""
        return PreprocessingService()

    @pytest.fixture
    def feature_engineer(self):
        """Create feature engineer instance"""
        return ChurnFeatureEngineer(random_state=42)

    def test_accuracy_target_85_percent(self, large_realistic_dataset, model_service,
                                      preprocessing_service, feature_engineer):
        """Test that the model achieves at least 85% accuracy on realistic data"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)

        # Preprocess data
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        # Apply same preprocessing to test data
        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns between train and test
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train models
        trained_models = model_service.train_models(X_train_final, y_train)

        # Test each model for accuracy target
        best_accuracy = 0
        best_model_name = ""

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_final)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name

            print(f"{model_name} accuracy: {accuracy:.4f}")

        # Assert that at least one model achieves 85% accuracy
        assert best_accuracy >= 0.85, f"Best model ({best_model_name}) accuracy {best_accuracy:.4f} < 85% target"

    def test_precision_recall_balance(self, large_realistic_dataset, model_service,
                                    preprocessing_service, feature_engineer):
        """Test that models achieve balanced precision and recall (both > 70%)"""
        # Prepare data (similar to accuracy test)
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train and evaluate models
        trained_models = model_service.train_models(X_train_final, y_train)

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_final)

            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

            print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}")

            # Both precision and recall should be > 70%
            assert precision >= 0.70, f"{model_name} precision {precision:.4f} < 70%"
            assert recall >= 0.70, f"{model_name} recall {recall:.4f} < 70%"

    def test_f1_score_target(self, large_realistic_dataset, model_service,
                           preprocessing_service, feature_engineer):
        """Test that F1-score is at least 80%"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train and evaluate
        trained_models = model_service.train_models(X_train_final, y_train)

        best_f1 = 0
        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_final)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1

            print(f"{model_name} F1-score: {f1:.4f}")

        assert best_f1 >= 0.80, f"Best F1-score {best_f1:.4f} < 80% target"

    def test_roc_auc_target(self, large_realistic_dataset, model_service,
                          preprocessing_service, feature_engineer):
        """Test that ROC-AUC is at least 85%"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                          random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train and evaluate
        trained_models = model_service.train_models(X_train_final, y_train)

        best_auc = 0
        for model_name, model in trained_models.items():
            try:
                y_proba = model.predict_proba(X_test_final)[:, 1]
                auc = roc_auc_score(y_test, y_proba)

                if auc > best_auc:
                    best_auc = auc

                print(f"{model_name} ROC-AUC: {auc:.4f}")
            except AttributeError:
                # Some models might not have predict_proba
                continue

        assert best_auc >= 0.85, f"Best ROC-AUC {best_auc:.4f} < 85% target"


class TestModelPerformanceConstraints:
    """Test model performance meets operational constraints"""

    def test_training_time_constraint(self, large_realistic_dataset, model_service,
                                    preprocessing_service, feature_engineer):
        """Test that model training completes within reasonable time (< 300 seconds)"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        # Time the training
        start_time = time.time()
        trained_models = model_service.train_models(X_train_final, y_train)
        end_time = time.time()

        training_time = end_time - start_time
        print(f"Total training time: {training_time:.2f} seconds")

        assert training_time < 300, f"Training time {training_time:.2f}s exceeds 300s limit"
        assert len(trained_models) > 0, "Should train at least one model"

    def test_prediction_speed_constraint(self, large_realistic_dataset, model_service,
                                       preprocessing_service, feature_engineer):
        """Test that predictions are fast enough for real-time use (< 100ms for 1000 predictions)"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train models
        trained_models = model_service.train_models(X_train_final, y_train)

        # Test prediction speed
        test_sample = X_test_final.head(1000)  # 1000 predictions

        for model_name, model in trained_models.items():
            start_time = time.time()
            _ = model.predict(test_sample)
            end_time = time.time()

            prediction_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"{model_name} prediction time for 1000 samples: {prediction_time:.2f}ms")

            assert prediction_time < 100, f"{model_name} prediction time {prediction_time:.2f}ms > 100ms limit"

    def test_memory_usage_constraint(self, large_realistic_dataset, model_service,
                                   preprocessing_service, feature_engineer):
        """Test that model training uses reasonable memory (< 2GB additional)"""
        # Measure initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        # Train models
        trained_models = model_service.train_models(X_train_final, y_train)

        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Memory increase during training: {memory_increase:.2f}MB")

        assert memory_increase < 2048, f"Memory increase {memory_increase:.2f}MB exceeds 2GB limit"
        assert len(trained_models) > 0, "Should successfully train models"

    def test_cross_validation_performance_consistency(self, large_realistic_dataset):
        """Test that cross-validation performance is consistent across folds (std < 5%)"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1).select_dtypes(include=[np.number])
        y = large_realistic_dataset['churn']

        # Use simpler preprocessing for CV test
        X_filled = X.fillna(X.median())

        # Perform cross-validation
        cv = SMOTECrossValidator(n_splits=5, random_state=42)
        model = RandomForestClassifier(n_estimators=50, random_state=42)  # Smaller for speed

        scores = cv.cross_validate_with_smote(model, X_filled, y, scoring_func=accuracy_score)

        # Check consistency
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv_coefficient = std_score / mean_score

        print(f"CV scores: {scores}")
        print(f"Mean: {mean_score:.4f}, Std: {std_score:.4f}, CV: {cv_coefficient:.4f}")

        assert cv_coefficient < 0.05, f"CV coefficient {cv_coefficient:.4f} > 5% indicates inconsistent performance"
        assert mean_score > 0.75, f"Mean CV score {mean_score:.4f} < 75% minimum"

    def test_feature_importance_stability(self, large_realistic_dataset, model_service,
                                        preprocessing_service, feature_engineer):
        """Test that feature importance is stable across different runs"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        # Train models multiple times
        importance_runs = []
        for run in range(3):
            # Re-initialize with different random state for model
            model_service_run = HighPerformanceModelService()
            trained_models = model_service_run.train_models(X_train_final, y_train)

            # Get feature importance from Random Forest (most models have this)
            if 'random_forest' in trained_models:
                rf_model = trained_models['random_forest']
                importance_runs.append(rf_model.feature_importances_)

        if len(importance_runs) >= 2:
            # Check correlation between importance runs
            from scipy.stats import pearsonr
            corr, _ = pearsonr(importance_runs[0], importance_runs[1])

            print(f"Feature importance correlation across runs: {corr:.4f}")
            assert corr > 0.8, f"Feature importance correlation {corr:.4f} < 0.8 indicates instability"


class TestModelRobustness:
    """Test model robustness under various conditions"""

    def test_performance_with_missing_data(self, large_realistic_dataset, model_service,
                                         preprocessing_service, feature_engineer):
        """Test model performance degrades gracefully with missing data"""
        # Prepare clean data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        # Train on clean data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Preprocess clean training data
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        # Train models
        trained_models = model_service.train_models(X_train_final, y_train)

        # Test with different levels of missing data
        missing_levels = [0.1, 0.2, 0.3]

        for missing_level in missing_levels:
            # Introduce missing values to test set
            X_test_missing = X_test.copy()
            n_missing = int(len(X_test_missing) * missing_level)
            missing_indices = np.random.choice(X_test_missing.index, n_missing, replace=False)

            for idx in missing_indices:
                # Randomly select columns to make missing
                cols_to_miss = np.random.choice(X_test_missing.columns,
                                              size=np.random.randint(1, 4), replace=False)
                X_test_missing.loc[idx, cols_to_miss] = np.nan

            # Preprocess test data with missing values
            X_test_clean, _ = preprocessing_service.handle_missing_values(X_test_missing)
            X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
            X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
            X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

            # Align columns
            common_columns = X_train_final.columns.intersection(X_test_final.columns)
            X_test_aligned = X_test_final[common_columns]

            # Test each model
            for model_name, model in trained_models.items():
                y_pred = model.predict(X_test_aligned)
                accuracy = accuracy_score(y_test, y_pred)

                print(f"{model_name} accuracy with {missing_level*100}% missing: {accuracy:.4f}")

                # Performance should not drop below 70% even with 30% missing data
                min_expected = 0.70 if missing_level <= 0.3 else 0.60
                assert accuracy >= min_expected, \
                    f"{model_name} accuracy {accuracy:.4f} < {min_expected} with {missing_level*100}% missing data"

    def test_performance_with_outliers(self, large_realistic_dataset, model_service,
                                     preprocessing_service, feature_engineer):
        """Test model performance with outliers in data"""
        # Prepare data
        X = large_realistic_dataset.drop(['churn', 'customer_id'], axis=1)
        y = large_realistic_dataset['churn']

        # Introduce outliers to numerical columns
        X_with_outliers = X.copy()
        numerical_cols = X_with_outliers.select_dtypes(include=[np.number]).columns

        for col in numerical_cols:
            # Introduce extreme outliers (5% of data)
            outlier_indices = np.random.choice(X_with_outliers.index,
                                             size=int(0.05 * len(X_with_outliers)), replace=False)
            # Make outliers 10x the max value
            max_val = X_with_outliers[col].max()
            X_with_outliers.loc[outlier_indices, col] = max_val * 10

        # Train and test with outliers
        X_train, X_test, y_train, y_test = train_test_split(X_with_outliers, y,
                                                          test_size=0.2, random_state=42, stratify=y)

        # Preprocess
        X_train_clean, _ = preprocessing_service.handle_missing_values(X_train)
        X_train_encoded, _ = preprocessing_service.encode_categorical_features(X_train_clean)
        X_train_engineered = feature_engineer.apply_feature_engineering(X_train_encoded)
        X_train_final, _ = preprocessing_service.scale_features(X_train_engineered)

        X_test_clean, _ = preprocessing_service.handle_missing_values(X_test)
        X_test_encoded, _ = preprocessing_service.encode_categorical_features(X_test_clean)
        X_test_engineered = feature_engineer.apply_feature_engineering(X_test_encoded)
        X_test_final, _ = preprocessing_service.scale_features(X_test_engineered)

        # Align columns
        common_columns = X_train_final.columns.intersection(X_test_final.columns)
        X_train_final = X_train_final[common_columns]
        X_test_final = X_test_final[common_columns]

        # Train and evaluate
        trained_models = model_service.train_models(X_train_final, y_train)

        for model_name, model in trained_models.items():
            y_pred = model.predict(X_test_final)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"{model_name} accuracy with outliers: {accuracy:.4f}")

            # Should maintain reasonable performance even with outliers
            assert accuracy >= 0.70, f"{model_name} accuracy {accuracy:.4f} < 70% with outliers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
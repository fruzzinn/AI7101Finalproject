"""
Unit Tests for Utility Functions
Tests individual utility functions in isolation for correctness and robustness
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import mlflow

# Import utility modules to test
from src.utils.mlflow_utils import MLflowExperimentManager
from src.utils.cv_pipeline import SMOTECrossValidator


class TestMLflowUtils:
    """Unit tests for MLflow utility functions"""

    @pytest.fixture
    def temp_mlflow_db(self):
        """Create temporary MLflow database for testing"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        try:
            yield f"sqlite:///{db_path}"
        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

    @pytest.fixture
    def mlflow_manager(self, temp_mlflow_db):
        """Create MLflow manager with temporary database"""
        return MLflowExperimentManager(
            experiment_name="test_experiment",
            tracking_uri=temp_mlflow_db
        )

    def test_mlflow_manager_initialization(self, temp_mlflow_db):
        """Test MLflow manager initialization"""
        manager = MLflowExperimentManager(
            experiment_name="test_init",
            tracking_uri=temp_mlflow_db
        )

        assert manager.experiment_name == "test_init"
        assert manager.tracking_uri == temp_mlflow_db

    def test_mlflow_manager_default_tracking_uri(self):
        """Test MLflow manager with default tracking URI"""
        manager = MLflowExperimentManager(experiment_name="test_default")
        assert manager.tracking_uri == "sqlite:///mlflow.db"

    def test_start_and_end_run(self, mlflow_manager):
        """Test starting and ending MLflow runs"""
        run_id = mlflow_manager.start_run("test_run")
        assert run_id is not None

        mlflow_manager.end_run()
        # Verify run is ended by checking active run
        assert mlflow.active_run() is None

    def test_log_data_info(self, mlflow_manager):
        """Test logging data information"""
        # Create sample data
        train_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })
        test_data = train_data.drop('target', axis=1)

        # Start run and log data info
        mlflow_manager.start_run("test_data_logging")
        mlflow_manager.log_data_info(train_data, test_data)
        mlflow_manager.end_run()

        # Verify no errors occurred (successful execution)
        assert True

    def test_log_preprocessing_steps(self, mlflow_manager):
        """Test logging preprocessing steps"""
        preprocessing_info = {
            'missing_value_handling': {'strategy': 'median', 'columns': ['age', 'income']},
            'encoding': {'categorical_columns': ['gender', 'region']},
            'scaling': {'method': 'standard', 'features': ['age', 'income']}
        }

        mlflow_manager.start_run("test_preprocessing_logging")
        mlflow_manager.log_preprocessing_steps(preprocessing_info)
        mlflow_manager.end_run()

        # Verify no errors occurred
        assert True

    def test_log_model_performance(self, mlflow_manager):
        """Test logging model performance metrics"""
        performance_metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85,
            'roc_auc': 0.91
        }

        mlflow_manager.start_run("test_performance_logging")
        mlflow_manager.log_model_performance(performance_metrics)
        mlflow_manager.end_run()

        # Verify no errors occurred
        assert True

    def test_log_feature_importance(self, mlflow_manager):
        """Test logging feature importance"""
        feature_importance = {
            'tenure': 0.35,
            'monthly_charges': 0.28,
            'contract_type': 0.22,
            'total_charges': 0.15
        }

        mlflow_manager.start_run("test_feature_importance_logging")
        mlflow_manager.log_feature_importance(feature_importance)
        mlflow_manager.end_run()

        # Verify no errors occurred
        assert True

    def test_log_hyperparameters(self, mlflow_manager):
        """Test logging hyperparameters"""
        hyperparameters = {
            'max_depth': 6,
            'n_estimators': 100,
            'learning_rate': 0.1,
            'random_state': 42
        }

        mlflow_manager.start_run("test_hyperparameter_logging")
        mlflow_manager.log_hyperparameters(hyperparameters)
        mlflow_manager.end_run()

        # Verify no errors occurred
        assert True

    def test_error_handling_invalid_experiment(self):
        """Test error handling with invalid experiment setup"""
        with pytest.raises(Exception):
            manager = MLflowExperimentManager(
                experiment_name="",  # Invalid empty name
                tracking_uri="invalid://uri"
            )

    def test_run_context_management(self, mlflow_manager):
        """Test proper run context management"""
        # Verify no active run initially
        assert mlflow.active_run() is None

        # Start run
        run_id = mlflow_manager.start_run("context_test")
        assert mlflow.active_run() is not None

        # End run
        mlflow_manager.end_run()
        assert mlflow.active_run() is None


class TestSMOTECrossValidator:
    """Unit tests for SMOTE Cross Validator"""

    @pytest.fixture
    def sample_imbalanced_data(self):
        """Create sample imbalanced dataset"""
        np.random.seed(42)
        n_majority = 900
        n_minority = 100

        # Majority class
        X_majority = np.random.normal(0, 1, (n_majority, 4))
        y_majority = np.zeros(n_majority)

        # Minority class
        X_minority = np.random.normal(2, 1, (n_minority, 4))
        y_minority = np.ones(n_minority)

        X = np.vstack([X_majority, X_minority])
        y = np.hstack([y_majority, y_minority])

        return X, y

    @pytest.fixture
    def sample_balanced_data(self):
        """Create sample balanced dataset"""
        np.random.seed(42)
        n_per_class = 500

        X_class0 = np.random.normal(0, 1, (n_per_class, 4))
        y_class0 = np.zeros(n_per_class)

        X_class1 = np.random.normal(1, 1, (n_per_class, 4))
        y_class1 = np.ones(n_per_class)

        X = np.vstack([X_class0, X_class1])
        y = np.hstack([y_class0, y_class1])

        return X, y

    def test_smote_cv_initialization(self):
        """Test SMOTE CV initialization with different parameters"""
        # Default initialization
        cv = SMOTECrossValidator()
        assert cv.n_splits == 5
        assert cv.random_state == 42
        assert cv.smote_threshold == 0.3

        # Custom initialization
        cv_custom = SMOTECrossValidator(
            n_splits=3,
            random_state=123,
            smote_threshold=0.2
        )
        assert cv_custom.n_splits == 3
        assert cv_custom.random_state == 123
        assert cv_custom.smote_threshold == 0.2

    def test_should_apply_smote_imbalanced_data(self, sample_imbalanced_data):
        """Test SMOTE application decision on imbalanced data"""
        X, y = sample_imbalanced_data
        cv = SMOTECrossValidator(smote_threshold=0.3)

        should_apply = cv._should_apply_smote(y)
        assert should_apply is True, "Should apply SMOTE to imbalanced data"

    def test_should_apply_smote_balanced_data(self, sample_balanced_data):
        """Test SMOTE application decision on balanced data"""
        X, y = sample_balanced_data
        cv = SMOTECrossValidator(smote_threshold=0.3)

        should_apply = cv._should_apply_smote(y)
        assert should_apply is False, "Should not apply SMOTE to balanced data"

    def test_apply_smote_to_fold(self, sample_imbalanced_data):
        """Test SMOTE application to individual fold"""
        X, y = sample_imbalanced_data
        cv = SMOTECrossValidator()

        # Get first fold
        train_idx, test_idx = next(cv.cv_splitter.split(X, y))
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]

        # Apply SMOTE
        X_resampled, y_resampled = cv._apply_smote_to_fold(X_train_fold, y_train_fold)

        # Check that minority class was upsampled
        original_minority_count = np.sum(y_train_fold == 1)
        resampled_minority_count = np.sum(y_resampled == 1)

        assert resampled_minority_count > original_minority_count, "SMOTE should increase minority class samples"
        assert len(X_resampled) == len(y_resampled), "X and y should have same length after SMOTE"

    def test_cross_validate_with_smote(self, sample_imbalanced_data):
        """Test complete cross-validation with SMOTE"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X, y = sample_imbalanced_data
        cv = SMOTECrossValidator(n_splits=3)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        scores = cv.cross_validate_with_smote(model, X, y, scoring_func=accuracy_score)

        assert len(scores) == 3, "Should return score for each fold"
        assert all(0 <= score <= 1 for score in scores), "Accuracy scores should be between 0 and 1"

    def test_cross_validate_without_smote_balanced_data(self, sample_balanced_data):
        """Test cross-validation without SMOTE on balanced data"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X, y = sample_balanced_data
        cv = SMOTECrossValidator(n_splits=3, smote_threshold=0.3)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        scores = cv.cross_validate_with_smote(model, X, y, scoring_func=accuracy_score)

        assert len(scores) == 3, "Should return score for each fold"
        assert all(0 <= score <= 1 for score in scores), "Accuracy scores should be between 0 and 1"

    def test_get_fold_statistics(self, sample_imbalanced_data):
        """Test getting fold statistics"""
        X, y = sample_imbalanced_data
        cv = SMOTECrossValidator(n_splits=3)

        stats = cv.get_fold_statistics(X, y)

        assert 'total_folds' in stats
        assert 'folds_with_smote' in stats
        assert 'class_distribution' in stats
        assert stats['total_folds'] == 3
        assert stats['folds_with_smote'] > 0, "Should apply SMOTE to imbalanced data"

    def test_error_handling_invalid_data(self):
        """Test error handling with invalid data"""
        cv = SMOTECrossValidator()

        # Test with None data
        with pytest.raises(ValueError):
            cv._should_apply_smote(None)

        # Test with empty array
        with pytest.raises(ValueError):
            cv._should_apply_smote(np.array([]))

        # Test with single class
        single_class_y = np.ones(100)
        with pytest.raises(ValueError):
            cv._apply_smote_to_fold(np.random.randn(100, 4), single_class_y)

    def test_reproducibility(self, sample_imbalanced_data):
        """Test that results are reproducible with same random state"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        X, y = sample_imbalanced_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Run twice with same random state
        cv1 = SMOTECrossValidator(n_splits=3, random_state=42)
        cv2 = SMOTECrossValidator(n_splits=3, random_state=42)

        scores1 = cv1.cross_validate_with_smote(model, X, y, scoring_func=accuracy_score)
        scores2 = cv2.cross_validate_with_smote(model, X, y, scoring_func=accuracy_score)

        np.testing.assert_array_almost_equal(scores1, scores2, decimal=5)

    def test_custom_scoring_function(self, sample_imbalanced_data):
        """Test with custom scoring function"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import precision_score

        X, y = sample_imbalanced_data
        cv = SMOTECrossValidator(n_splits=3)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        def custom_scorer(y_true, y_pred):
            return precision_score(y_true, y_pred, average='weighted', zero_division=0)

        scores = cv.cross_validate_with_smote(model, X, y, scoring_func=custom_scorer)

        assert len(scores) == 3, "Should return score for each fold"
        assert all(0 <= score <= 1 for score in scores), "Precision scores should be between 0 and 1"


class TestUtilityHelperFunctions:
    """Unit tests for utility helper functions"""

    def test_data_type_validation(self):
        """Test data type validation utilities"""
        # This would test any data validation utilities
        # Currently focusing on testing existing utility classes
        pass

    def test_configuration_management(self):
        """Test configuration management utilities"""
        # This would test configuration loading/saving utilities
        # Currently focusing on testing existing utility classes
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
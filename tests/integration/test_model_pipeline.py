"""
Integration Tests for Model Training and Evaluation Pipeline
Tests the complete ML pipeline from preprocessed data through model training to evaluation
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Import our services and utilities
from src.services.model_development_service import ModelDevelopmentService
from src.services.preprocessing_service import PreprocessingService
from src.utils.cv_pipeline import SMOTECrossValidator
from src.evaluation.metrics import ChurnModelEvaluator
from src.utils.mlflow_utils import MLflowExperimentManager
from src.models.model_performance import ModelPerformance


class TestModelPipelineIntegration:
    """Integration tests for complete model training and evaluation pipeline"""

    @pytest.fixture
    def preprocessed_data(self):
        """Create preprocessed dataset ready for ML"""
        np.random.seed(42)
        n_samples = 1000

        # Generate realistic features after preprocessing
        data = {
            'customer_id': [f'C{i:04d}' for i in range(n_samples)],
            'age_normalized': np.random.normal(0, 1, n_samples),
            'tenure_normalized': np.random.normal(0, 1, n_samples),
            'monthly_charges_normalized': np.random.normal(0, 1, n_samples),
            'total_charges_normalized': np.random.normal(0, 1, n_samples),
            'tenure_years': np.random.exponential(2, n_samples),
            'clv_estimate': np.random.normal(1500, 500, n_samples),
            'service_bundle_count': np.random.poisson(3, n_samples),
            'composite_risk_score': np.random.beta(2, 5, n_samples),

            # One-hot encoded categorical features
            'contract_type_Month-to-month': np.random.binomial(1, 0.5, n_samples),
            'contract_type_One year': np.random.binomial(1, 0.3, n_samples),
            'contract_type_Two year': np.random.binomial(1, 0.2, n_samples),
            'payment_method_Electronic check': np.random.binomial(1, 0.25, n_samples),
            'payment_method_Credit card': np.random.binomial(1, 0.25, n_samples),
            'internet_service_Fiber optic': np.random.binomial(1, 0.4, n_samples),
            'internet_service_DSL': np.random.binomial(1, 0.4, n_samples),

            # Engineered features
            'early_adopter_score': np.random.gamma(2, 0.5, n_samples),
            'payment_risk_score': np.random.binomial(1, 0.3, n_samples),
            'tenure_price_interaction': np.random.normal(0, 1, n_samples)
        }

        # Create realistic churn based on feature relationships
        churn_logits = (
            -1.0 +  # Base log-odds
            1.5 * data['contract_type_Month-to-month'] +
            1.2 * data['payment_method_Electronic check'] +
            -0.8 * data['tenure_normalized'] +
            0.5 * data['composite_risk_score'] +
            0.3 * data['monthly_charges_normalized']
        )

        churn_probs = 1 / (1 + np.exp(-np.array(churn_logits)))
        data['churn'] = np.random.binomial(1, churn_probs, n_samples)

        df = pd.DataFrame(data)
        return df

    @pytest.fixture
    def model_service(self):
        """Create model development service"""
        return ModelDevelopmentService(random_state=42)

    @pytest.fixture
    def cv_pipeline(self):
        """Create cross-validation pipeline"""
        return SMOTECrossValidator(n_splits=3, random_state=42)  # Reduced folds for faster testing

    @pytest.fixture
    def evaluator(self):
        """Create model evaluator"""
        return ChurnModelEvaluator()

    def test_complete_model_training_pipeline(self, preprocessed_data, model_service):
        """Test complete model training from data to trained models"""
        # Prepare data
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']

        print(f"Training with {X.shape[0]} samples and {X.shape[1]} features")
        print(f"Churn rate: {y.mean():.3f}")

        # Train multiple models
        trained_models = model_service.train_models(X, y)

        # Verify models were trained
        assert len(trained_models) >= 4, "Should train at least 4 different algorithms"
        expected_models = ['random_forest', 'gradient_boosting', 'logistic_regression', 'svm']

        for model_name in expected_models:
            assert model_name in trained_models, f"Should train {model_name}"
            model = trained_models[model_name]
            assert hasattr(model, 'predict'), f"{model_name} should have predict method"
            assert hasattr(model, 'predict_proba'), f"{model_name} should have predict_proba method"

        # Test predictions
        sample_predictions = trained_models['random_forest'].predict(X[:10])
        assert len(sample_predictions) == 10, "Should generate predictions"
        assert all(pred in [0, 1] for pred in sample_predictions), "Predictions should be binary"

        return trained_models

    def test_cross_validation_integration(self, preprocessed_data, model_service, cv_pipeline):
        """Test cross-validation integration with model training"""
        # Prepare data
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']

        # Test with a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        # Perform cross-validation using both services
        # Method 1: Using model development service
        cv_results_service = model_service.implement_cross_validation(X, y, model)

        # Method 2: Using CV pipeline
        cv_results_pipeline = cv_pipeline.cross_validate_model(X, y, model)

        # Verify both methods work
        assert 'f1_score_mean' in cv_results_service, "Service CV should return F1 score"
        assert 'f1_mean' in cv_results_pipeline, "Pipeline CV should return F1 score"

        # Both should detect SMOTE usage
        assert 'smote_applied' in cv_results_service, "Should document SMOTE usage"
        assert 'smote_applied' in cv_results_pipeline, "Should document SMOTE usage"

        # Performance should be reasonable
        assert cv_results_service['f1_score_mean'] > 0.3, "F1 score should be reasonable"
        assert cv_results_pipeline['f1_mean'] > 0.3, "F1 score should be reasonable"

    def test_hyperparameter_tuning_integration(self, preprocessed_data, model_service):
        """Test hyperparameter tuning integration"""
        # Use smaller dataset for faster tuning
        sample_data = preprocessed_data.sample(n=300, random_state=42)
        X = sample_data.drop(['customer_id', 'churn'], axis=1)
        y = sample_data['churn']

        # Test hyperparameter tuning
        best_model, best_params = model_service.hyperparameter_tuning(X, y, 'random_forest')

        # Verify tuning results
        assert best_model is not None, "Should return best model"
        assert isinstance(best_params, dict), "Should return parameters dictionary"
        assert 'best_score' in best_params, "Should include best score"
        assert 'optimization_metric' in best_params, "Should document optimization metric"

        # Test predictions with tuned model
        predictions = best_model.predict(X)
        probabilities = best_model.predict_proba(X)[:, 1]

        assert len(predictions) == len(X), "Should predict for all samples"
        assert len(probabilities) == len(X), "Should return probabilities for all samples"
        assert all(0 <= p <= 1 for p in probabilities), "Probabilities should be between 0 and 1"

    def test_model_evaluation_integration(self, preprocessed_data, model_service, evaluator):
        """Test model evaluation integration"""
        # Split data
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train a model
        models = model_service.train_models(X_train, y_train)
        model = models['random_forest']

        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate using model service
        service_metrics = model_service.evaluate_model_performance(model, X_test, y_test)

        # Evaluate using evaluator
        evaluator_metrics = evaluator.calculate_comprehensive_metrics(y_test, y_pred, y_proba)

        # Verify both evaluation methods
        common_metrics = ['f1_score', 'precision', 'recall', 'roc_auc']
        for metric in common_metrics:
            assert metric in service_metrics, f"Service should calculate {metric}"
            assert metric in evaluator_metrics, f"Evaluator should calculate {metric}"

            # Metrics should be similar (small differences due to implementation details)
            diff = abs(service_metrics[metric] - evaluator_metrics[metric])
            assert diff < 0.01, f"{metric} should be consistent between evaluation methods"

        # Test comprehensive evaluation report
        report = evaluator.create_evaluation_report(y_test, y_pred, y_proba, model_name="RandomForest")

        assert 'metrics' in report, "Report should include metrics"
        assert 'figures' in report, "Report should include figures"
        assert report['metrics']['f1_score'] > 0, "F1 score should be positive"

    def test_feature_importance_integration(self, preprocessed_data, model_service):
        """Test feature importance analysis integration"""
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']

        # Train a model with feature importance
        models = model_service.train_models(X, y)
        rf_model = models['random_forest']

        # Analyze feature importance
        feature_importance = model_service.analyze_feature_importance(rf_model, X.columns.tolist())

        # Verify feature importance analysis
        assert isinstance(feature_importance, dict), "Should return feature importance dictionary"
        assert len(feature_importance) >= len(X.columns), "Should include all features (plus metadata)"

        # Check that importances are normalized
        numeric_importance = {k: v for k, v in feature_importance.items() if isinstance(v, (int, float))}
        total_importance = sum(numeric_importance.values())
        assert abs(total_importance - 1.0) < 0.01, "Feature importances should sum to 1.0"

        # Most important feature should be reasonable
        sorted_features = sorted(numeric_importance.items(), key=lambda x: x[1], reverse=True)
        top_feature, top_importance = sorted_features[0]
        assert top_importance > 0.05, "Top feature should have significant importance"

    def test_model_performance_entity_integration(self, preprocessed_data, model_service):
        """Test integration with ModelPerformance entity"""
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train and evaluate model
        models = model_service.train_models(X_train, y_train)
        model = models['random_forest']
        performance_metrics = model_service.evaluate_model_performance(model, X_test, y_test)

        # Create ModelPerformance entity
        model_performance = ModelPerformance(
            model_id=f"rf_model_{np.random.randint(1000, 9999)}",
            f1_score=performance_metrics['f1_score'],
            precision=performance_metrics['precision'],
            recall=performance_metrics['recall'],
            auc_roc=performance_metrics['auc_roc'],
            confusion_matrix=performance_metrics['confusion_matrix']
        )

        # Validate entity
        assert model_performance.validate_metrics() is True, "ModelPerformance should be valid"
        assert model_performance.f1_score > 0, "F1 score should be positive"
        assert 0 <= model_performance.auc_roc <= 1, "AUC-ROC should be between 0 and 1"

    def test_pipeline_with_mlflow_integration(self, preprocessed_data, model_service):
        """Test complete model pipeline with MLflow tracking"""
        mlflow_manager = MLflowExperimentManager(
            experiment_name="test_model_pipeline",
            tracking_uri="sqlite:///test_model_mlflow.db"
        )

        try:
            # Start MLflow run
            run_id = mlflow_manager.start_run("model_pipeline_test")

            # Prepare data
            X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
            y = preprocessed_data['churn']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Log data information
            mlflow_manager.log_data_info(pd.concat([X_train, y_train], axis=1), pd.concat([X_test, y_test], axis=1))

            # Train models
            models = model_service.train_models(X_train, y_train)

            # Evaluate best model
            best_model = models['random_forest']
            mlflow_manager.log_model_training(best_model, "random_forest")

            # Cross-validation
            cv_results = model_service.implement_cross_validation(X_train, y_train, best_model)
            mlflow_manager.log_cross_validation_results(cv_results)

            # Final evaluation
            evaluation_metrics = model_service.evaluate_model_performance(best_model, X_test, y_test)
            mlflow_manager.log_model_evaluation(evaluation_metrics, "test")

            # Feature importance
            feature_importance = model_service.analyze_feature_importance(best_model, X.columns.tolist())
            mlflow_manager.log_feature_importance(feature_importance)

            # End run
            mlflow_manager.end_run()

            assert run_id is not None, "MLflow run should be created"

        finally:
            # Cleanup
            if os.path.exists("test_model_mlflow.db"):
                os.unlink("test_model_mlflow.db")

    def test_model_comparison_integration(self, preprocessed_data, model_service, evaluator):
        """Test model comparison across different algorithms"""
        X = preprocessed_data.drop(['customer_id', 'churn'], axis=1)
        y = preprocessed_data['churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train multiple models
        models = model_service.train_models(X_train, y_train)

        # Evaluate all models
        model_reports = {}
        for model_name, model in models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            report = evaluator.create_evaluation_report(
                y_test, y_pred, y_proba, model_name=model_name
            )
            model_reports[model_name] = report

        # Create comparison
        comparison_fig = evaluator.compare_models(model_reports)

        # Verify comparison
        assert len(model_reports) >= 4, "Should evaluate multiple models"
        for model_name, report in model_reports.items():
            assert 'metrics' in report, f"Report for {model_name} should include metrics"
            assert report['metrics']['f1_score'] > 0, f"F1 score for {model_name} should be positive"

        # Find best model
        best_model_name = max(model_reports.keys(),
                            key=lambda x: model_reports[x]['metrics']['f1_score'])
        best_f1 = model_reports[best_model_name]['metrics']['f1_score']

        print(f"Best model: {best_model_name} with F1 score: {best_f1:.3f}")
        assert best_f1 > 0.3, "Best model should have reasonable performance"

    def test_pipeline_performance_and_scalability(self, model_service):
        """Test pipeline performance with different data sizes"""
        import time

        data_sizes = [100, 500, 1000]
        performance_metrics = {}

        for size in data_sizes:
            # Generate data of specific size
            np.random.seed(42)
            X = pd.DataFrame(np.random.randn(size, 10), columns=[f'feature_{i}' for i in range(10)])
            y = np.random.binomial(1, 0.3, size)

            # Time the training
            start_time = time.time()
            models = model_service.train_models(X, y)
            training_time = time.time() - start_time

            # Time cross-validation
            start_time = time.time()
            cv_results = model_service.implement_cross_validation(X, y, models['random_forest'])
            cv_time = time.time() - start_time

            performance_metrics[size] = {
                'training_time': training_time,
                'cv_time': cv_time,
                'total_time': training_time + cv_time
            }

        # Verify performance scales reasonably
        for size in data_sizes:
            assert performance_metrics[size]['total_time'] < 30, f"Size {size} should complete within 30 seconds"

        # Performance should scale roughly linearly (not exponentially)
        ratio = performance_metrics[1000]['total_time'] / performance_metrics[100]['total_time']
        assert ratio < 50, "Performance should scale reasonably with data size"

    def test_error_handling_in_model_pipeline(self, model_service):
        """Test error handling throughout the model pipeline"""
        # Test with invalid data
        invalid_X = pd.DataFrame([[np.nan, np.inf, -np.inf]], columns=['a', 'b', 'c'])
        invalid_y = pd.Series([1])

        # Should handle gracefully or raise informative errors
        try:
            models = model_service.train_models(invalid_X, invalid_y)
            # If it succeeds, verify basic functionality
            assert len(models) > 0, "Should train at least some models"
        except (ValueError, RuntimeError) as e:
            # Expected for invalid data
            assert "invalid" in str(e).lower() or "nan" in str(e).lower()

        # Test with mismatched dimensions
        X_mismatch = pd.DataFrame(np.random.randn(10, 5))
        y_mismatch = pd.Series(np.random.binomial(1, 0.5, 15))  # Wrong length

        with pytest.raises((ValueError, IndexError)):
            model_service.train_models(X_mismatch, y_mismatch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
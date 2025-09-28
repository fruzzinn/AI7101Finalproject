"""
Integration Tests for Business Impact Analysis Pipeline
Tests the complete business analysis workflow from model predictions to actionable insights
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os

# Import our services and utilities
from src.services.business_analysis_service import BusinessAnalysisService
from src.services.model_development_service import ModelDevelopmentService
from src.evaluation.metrics import ChurnModelEvaluator
from src.utils.mlflow_utils import MLflowExperimentManager
from src.models.business_impact import BusinessImpact
from src.models.retention_strategy import RetentionStrategy, StrategyType, UrgencyLevel


class TestBusinessAnalysisPipelineIntegration:
    """Integration tests for complete business analysis pipeline"""

    @pytest.fixture
    def customer_data_with_predictions(self):
        """Create customer data with model predictions and business context"""
        np.random.seed(42)
        n_customers = 1000

        # Customer demographics and service data
        data = {
            'customer_id': [f'C{i:04d}' for i in range(n_customers)],
            'age': np.random.normal(45, 15, n_customers).astype(int).clip(18, 80),
            'tenure': np.random.exponential(24, n_customers).astype(int).clip(0, 72),
            'monthly_charges': np.random.normal(65, 20, n_customers).clip(20, 120),
            'total_charges': np.random.normal(1500, 800, n_customers).clip(0, 8000),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_customers),
            'customer_segment': np.random.choice(['High-value', 'Medium-value', 'Low-value'], n_customers, p=[0.2, 0.5, 0.3]),
            'support_tickets': np.random.poisson(2, n_customers),
            'satisfaction_score': np.random.normal(7, 2, n_customers).clip(1, 10)
        }

        df = pd.DataFrame(data)

        # Generate realistic churn probabilities based on customer characteristics
        churn_logits = (
            -2.0 +  # Base log-odds
            1.5 * (df['contract_type'] == 'Month-to-month').astype(int) +
            1.2 * (df['payment_method'] == 'Electronic check').astype(int) +
            -0.02 * df['tenure'] +
            0.3 * (df['satisfaction_score'] < 5).astype(int) +
            0.1 * df['support_tickets']
        )

        df['churn_probability'] = 1 / (1 + np.exp(-churn_logits))
        df['churn_prediction'] = (df['churn_probability'] > 0.5).astype(int)
        df['actual_churn'] = np.random.binomial(1, df['churn_probability'], n_customers)

        # Customer values for business impact calculation
        df['annual_revenue'] = df['monthly_charges'] * 12
        df['lifetime_value'] = df['annual_revenue'] * np.random.uniform(1.5, 4.0, n_customers)

        return df

    @pytest.fixture
    def sample_feature_importance(self):
        """Create sample feature importance for retention strategy generation"""
        return {
            'tenure': 0.25,
            'monthly_charges': 0.20,
            'contract_type_Month-to-month': 0.15,
            'payment_method_Electronic check': 0.12,
            'total_charges': 0.10,
            'age': 0.08,
            'satisfaction_score': 0.06,
            'support_tickets': 0.04
        }

    @pytest.fixture
    def business_service(self):
        """Create business analysis service"""
        return BusinessAnalysisService()

    @pytest.fixture
    def model_service(self):
        """Create model development service"""
        return ModelDevelopmentService(random_state=42)

    @pytest.fixture
    def evaluator(self):
        """Create model evaluator"""
        return ChurnModelEvaluator()

    def test_complete_business_analysis_pipeline(self, customer_data_with_predictions, business_service, sample_feature_importance):
        """Test the complete business analysis pipeline from predictions to actionable insights"""
        data = customer_data_with_predictions

        # Step 1: Calculate business impact
        impact_metrics = business_service.calculate_business_impact(
            predictions=data['churn_prediction'].values,
            probabilities=data['churn_probability'].values,
            customer_values=data['lifetime_value']
        )

        # Verify business impact calculation
        assert 'at_risk_customers_identified' in impact_metrics, "Should identify at-risk customers"
        assert 'revenue_protection_estimate' in impact_metrics, "Should estimate revenue protection"
        assert 'roi_analysis' in impact_metrics, "Should include ROI analysis"

        at_risk_count = impact_metrics['at_risk_customers_identified']
        expected_at_risk = data['churn_prediction'].sum()
        assert at_risk_count == expected_at_risk, "At-risk count should match predictions"

        # Step 2: Generate retention strategies
        high_risk_customers = data[data['churn_probability'] > 0.7].head(100)  # Top 100 high-risk customers
        strategies = business_service.generate_retention_strategies(
            sample_feature_importance,
            high_risk_customers
        )

        # Verify retention strategies
        assert len(strategies) > 0, "Should generate retention strategies"
        for strategy in strategies:
            assert 'strategy_id' in strategy, "Strategy should have ID"
            assert 'target_segment' in strategy, "Strategy should have target segment"
            assert 'recommended_actions' in strategy, "Strategy should have actions"
            assert 'priority_score' in strategy, "Strategy should have priority"
            assert 'expected_success_rate' in strategy, "Strategy should have success rate"

        # Step 3: Create executive summary
        key_insights = [
            "Customer tenure is the strongest predictor of churn",
            "Month-to-month contracts show 3x higher churn rates",
            "Electronic check users have elevated churn risk"
        ]

        sample_model_performance = {
            'f1_score': 0.75,
            'precision': 0.78,
            'recall': 0.72,
            'auc_roc': 0.85
        }

        executive_summary = business_service.create_executive_summary(
            sample_model_performance,
            impact_metrics,
            key_insights
        )

        # Verify executive summary
        assert 'business_impact_summary' in executive_summary, "Should include business impact summary"
        assert 'recommendations' in executive_summary, "Should include recommendations"
        assert 'next_steps' in executive_summary, "Should include next steps"

        return {
            'impact_metrics': impact_metrics,
            'strategies': strategies,
            'executive_summary': executive_summary
        }

    def test_end_to_end_ml_to_business_pipeline(self, customer_data_with_predictions, model_service, business_service):
        """Test complete pipeline from ML model training to business recommendations"""
        data = customer_data_with_predictions

        # Prepare ML data
        feature_columns = ['age', 'tenure', 'monthly_charges', 'total_charges', 'support_tickets', 'satisfaction_score']
        X = data[feature_columns]
        y = data['actual_churn']

        # Train models
        models = model_service.train_models(X, y)
        best_model = models['random_forest']

        # Make predictions
        predictions = best_model.predict(X)
        probabilities = best_model.predict_proba(X)[:, 1]

        # Evaluate model
        evaluation_metrics = model_service.evaluate_model_performance(best_model, X, y)

        # Analyze feature importance
        feature_importance = model_service.analyze_feature_importance(best_model, feature_columns)

        # Calculate business impact with actual model predictions
        business_impact = business_service.calculate_business_impact(
            predictions=predictions,
            probabilities=probabilities,
            customer_values=data['lifetime_value']
        )

        # Generate retention strategies based on actual feature importance
        high_risk_customers = data[probabilities > 0.7]
        if len(high_risk_customers) == 0:
            high_risk_customers = data.nlargest(50, 'churn_probability')  # Fallback

        strategies = business_service.generate_retention_strategies(
            feature_importance,
            high_risk_customers
        )

        # Create comprehensive analysis
        executive_summary = business_service.create_executive_summary(
            evaluation_metrics,
            business_impact,
            ["Data-driven insights from actual model predictions"]
        )

        # Verify end-to-end pipeline
        assert evaluation_metrics['f1_score'] > 0, "Model should have positive F1 score"
        assert business_impact['at_risk_customers_identified'] > 0, "Should identify some at-risk customers"
        assert len(strategies) > 0, "Should generate strategies"
        assert len(executive_summary['recommendations']) > 0, "Should provide recommendations"

        print(f"Model F1 Score: {evaluation_metrics['f1_score']:.3f}")
        print(f"At-risk customers: {business_impact['at_risk_customers_identified']}")
        print(f"Estimated ROI: {business_impact['roi_analysis']['roi_percentage']:.1f}%")

    def test_business_impact_entity_integration(self, customer_data_with_predictions, business_service):
        """Test integration with BusinessImpact entity"""
        data = customer_data_with_predictions

        # Calculate business impact
        impact_metrics = business_service.calculate_business_impact(
            predictions=data['churn_prediction'].values,
            probabilities=data['churn_probability'].values,
            customer_values=data['lifetime_value']
        )

        # Create BusinessImpact entity
        business_impact = BusinessImpact(
            model_id=f"analysis_{np.random.randint(1000, 9999)}",
            predicted_churners=impact_metrics['at_risk_customers_identified'],
            retention_rate_improvement=0.3,  # 30% retention rate assumption
            revenue_protection=impact_metrics['revenue_protection_estimate'],
            cost_reduction=impact_metrics['cost_reduction_estimate'],
            roi_analysis=impact_metrics['roi_analysis']
        )

        # Validate entity
        assert business_impact.model_id is not None, "BusinessImpact should be valid"
        assert business_impact.revenue_protection > 0, "Revenue protection should be positive"
        assert business_impact.get_total_value() >= 0, "Total value should be non-negative"

    def test_retention_strategy_entity_integration(self, business_service, sample_feature_importance):
        """Test integration with RetentionStrategy entity"""
        # Create sample high-risk customer data
        high_risk_data = pd.DataFrame({
            'customer_id': ['C001', 'C002', 'C003'],
            'churn_probability': [0.85, 0.90, 0.75],
            'monthly_charges': [120, 80, 95],
            'contract_type': ['Month-to-month', 'Month-to-month', 'One year'],
            'customer_segment': ['High-value', 'Low-value', 'Medium-value']
        })

        # Generate strategies
        strategies = business_service.generate_retention_strategies(
            sample_feature_importance,
            high_risk_data
        )

        # Create RetentionStrategy entities
        for strategy_data in strategies:
            retention_strategy = RetentionStrategy(
                strategy_id=strategy_data['strategy_id'],
                target_segment={'segment_name': strategy_data['target_segment']},
                recommended_actions=strategy_data['recommended_actions'],
                priority_score=strategy_data['priority_score'],
                expected_success_rate=strategy_data['expected_success_rate'],
                resource_requirements=strategy_data['resource_requirements'],
                strategy_type=StrategyType.DISCOUNT,  # Use valid enum value
                urgency_level=UrgencyLevel.HIGH
            )

            # Validate entity
            assert retention_strategy.strategy_id is not None, "RetentionStrategy should be valid"
            assert retention_strategy.expected_success_rate > 0, "Success rate should be positive"
            assert retention_strategy.get_cost_per_customer() >= 0, "Cost per customer should be non-negative"

    def test_business_analysis_with_mlflow_integration(self, customer_data_with_predictions, business_service):
        """Test business analysis with MLflow tracking"""
        mlflow_manager = MLflowExperimentManager(
            experiment_name="test_business_analysis",
            tracking_uri="sqlite:///test_business_mlflow.db"
        )

        try:
            # Start MLflow run
            run_id = mlflow_manager.start_run("business_analysis_test")

            data = customer_data_with_predictions

            # Calculate business impact
            impact_metrics = business_service.calculate_business_impact(
                predictions=data['churn_prediction'].values,
                probabilities=data['churn_probability'].values,
                customer_values=data['lifetime_value']
            )

            # Log business impact to MLflow
            mlflow_manager.log_business_impact(impact_metrics)

            # Log predictions
            mlflow_manager.log_predictions(
                predictions=data['churn_prediction'].values,
                probabilities=data['churn_probability'].values,
                customer_ids=data['customer_id'].tolist()
            )

            # Additional business metrics
            import mlflow
            mlflow.log_metric("total_customers_analyzed", len(data))
            mlflow.log_metric("churn_rate_predicted", data['churn_prediction'].mean())
            mlflow.log_metric("avg_customer_value", data['lifetime_value'].mean())

            # End run
            mlflow_manager.end_run()

            assert run_id is not None, "MLflow run should be created"

        finally:
            # Cleanup
            if os.path.exists("test_business_mlflow.db"):
                os.unlink("test_business_mlflow.db")

    def test_model_limitations_assessment(self, business_service):
        """Test model limitations and uncertainty assessment"""
        sample_performance = {
            'f1_score': 0.72,
            'precision': 0.75,
            'recall': 0.69,
            'auc_roc': 0.81
        }

        data_quality_issues = [
            "15% missing demographic data",
            "Limited 12-month historical data",
            "Seasonal patterns not captured"
        ]

        limitations = business_service.assess_model_limitations(
            sample_performance,
            data_quality_issues
        )

        # Verify limitations assessment
        assert 'model_failure_scenarios' in limitations, "Should identify failure scenarios"
        assert 'prediction_uncertainty' in limitations, "Should quantify uncertainty"
        assert 'data_limitations' in limitations, "Should document data limitations"
        assert 'monitoring_recommendations' in limitations, "Should provide monitoring guidance"

        assert len(limitations['model_failure_scenarios']) > 0, "Should identify specific failure scenarios"
        assert len(limitations['monitoring_recommendations']) > 0, "Should provide monitoring recommendations"

    def test_monitoring_framework_generation(self, business_service):
        """Test monitoring framework for ongoing model performance"""
        baseline_metrics = {
            'f1_score': 0.75,
            'precision': 0.78,
            'recall': 0.72,
            'auc_roc': 0.85
        }

        monitoring_framework = business_service.generate_monitoring_framework(baseline_metrics)

        # Verify monitoring framework
        assert 'performance_thresholds' in monitoring_framework, "Should define performance thresholds"
        assert 'business_kpis' in monitoring_framework, "Should define business KPIs"
        assert 'retraining_triggers' in monitoring_framework, "Should define retraining triggers"
        assert 'drift_detection' in monitoring_framework, "Should include drift detection"

        # Check threshold settings
        thresholds = monitoring_framework['performance_thresholds']
        for metric, baseline_value in baseline_metrics.items():
            assert metric in thresholds, f"Should set threshold for {metric}"
            assert thresholds[metric] < baseline_value, f"Threshold for {metric} should be below baseline"

    def test_customer_segmentation_analysis(self, customer_data_with_predictions, business_service):
        """Test business analysis across different customer segments"""
        data = customer_data_with_predictions

        # Analyze by customer segment
        segment_analysis = {}

        for segment in data['customer_segment'].unique():
            segment_data = data[data['customer_segment'] == segment]

            segment_impact = business_service.calculate_business_impact(
                predictions=segment_data['churn_prediction'].values,
                probabilities=segment_data['churn_probability'].values,
                customer_values=segment_data['lifetime_value']
            )

            segment_analysis[segment] = {
                'customer_count': len(segment_data),
                'churn_rate': segment_data['churn_prediction'].mean(),
                'avg_customer_value': segment_data['lifetime_value'].mean(),
                'revenue_at_risk': segment_impact['revenue_protection_estimate'] / 0.3,
                'business_impact': segment_impact
            }

        # Verify segment analysis
        assert len(segment_analysis) > 0, "Should analyze multiple segments"

        for segment, analysis in segment_analysis.items():
            assert analysis['customer_count'] > 0, f"Segment {segment} should have customers"
            assert 0 <= analysis['churn_rate'] <= 1, f"Churn rate for {segment} should be between 0 and 1"
            assert analysis['avg_customer_value'] > 0, f"Average customer value for {segment} should be positive"

        # High-value customers should have higher average customer value
        if 'High-value' in segment_analysis and 'Low-value' in segment_analysis:
            high_value_avg = segment_analysis['High-value']['avg_customer_value']
            low_value_avg = segment_analysis['Low-value']['avg_customer_value']
            assert high_value_avg > low_value_avg, "High-value segment should have higher average customer value"

    def test_business_summary_generation(self, customer_data_with_predictions, business_service, evaluator):
        """Test comprehensive business summary generation"""
        data = customer_data_with_predictions

        # Calculate business impact
        impact_metrics = business_service.calculate_business_impact(
            predictions=data['churn_prediction'].values,
            probabilities=data['churn_probability'].values,
            customer_values=data['lifetime_value']
        )

        # Generate business summary with custom assumptions
        business_summary = evaluator.generate_business_summary(
            metrics={'precision': 0.75, 'recall': 0.70, 'predicted_churn_rate': data['churn_prediction'].mean()},
            cost_per_customer=30.0,
            revenue_per_customer=1200.0
        )

        # Verify business summary
        assert 'customers_targeted_for_retention' in business_summary, "Should calculate targeted customers"
        assert 'roi_percentage' in business_summary, "Should calculate ROI"
        assert 'estimated_revenue_saved' in business_summary, "Should estimate revenue saved"

        # ROI should be reasonable
        assert business_summary['roi_percentage'] > -100, "ROI should not be extremely negative"
        assert business_summary['estimated_revenue_saved'] >= 0, "Revenue saved should be non-negative"

    def test_integration_with_evaluation_pipeline(self, customer_data_with_predictions, business_service, evaluator):
        """Test integration between business analysis and model evaluation"""
        data = customer_data_with_predictions

        # Create evaluation report
        evaluation_report = evaluator.create_evaluation_report(
            y_true=data['actual_churn'].values,
            y_pred=data['churn_prediction'].values,
            y_proba=data['churn_probability'].values,
            model_name="IntegrationTest"
        )

        # Use evaluation metrics for business analysis
        business_impact = business_service.calculate_business_impact(
            predictions=data['churn_prediction'].values,
            probabilities=data['churn_probability'].values,
            customer_values=data['lifetime_value']
        )

        # Create integrated summary
        integrated_summary = business_service.create_executive_summary(
            evaluation_report['metrics'],
            business_impact,
            ["Integrated ML and business analysis"]
        )

        # Verify integration
        assert 'business_impact_summary' in integrated_summary, "Should integrate business impact"
        assert 'financial_projection' in integrated_summary, "Should include financial projections"

        # Business summary should reference model performance
        business_text = integrated_summary['business_impact_summary']
        assert any(term in business_text.lower() for term in ['accuracy', 'predict', 'model']), "Should reference model performance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
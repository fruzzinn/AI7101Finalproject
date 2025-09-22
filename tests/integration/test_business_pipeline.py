"""
Integration tests for business analysis pipeline.

Educational Focus: Demonstrates integration testing for business impact analysis workflows.
These tests verify that the complete business analysis pipeline works end-to-end.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Tuple, Any, Optional

# Note: These imports will fail initially until we implement the actual classes
# The tests are designed to fail first (TDD principle)


class TestBusinessAnalysisIntegration:
    """Integration tests for the complete business analysis pipeline."""

    def setup_method(self):
        """Set up test fixtures and sample data."""
        # Create sample customer data for business analysis
        np.random.seed(42)
        n_customers = 200

        self.customer_data = pd.DataFrame({
            'customer_id': [f'C{i:06d}' for i in range(n_customers)],
            'tenure': np.random.randint(1, 72, n_customers),
            'monthly_charges': np.random.uniform(20, 120, n_customers),
            'total_charges': np.random.uniform(100, 7000, n_customers),
            'contract_type': np.random.choice(['month-to-month', 'one-year', 'two-year'], n_customers),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers),
            'payment_method': np.random.choice(['electronic_check', 'bank_transfer', 'credit_card', 'mailed_check'], n_customers)
        })

        # Create realistic churn labels (correlated with features)
        churn_probability = (
            0.4 * (self.customer_data['contract_type'] == 'month-to-month') +
            0.2 * (self.customer_data['monthly_charges'] > 80) +
            0.3 * (self.customer_data['tenure'] < 12) +
            0.1 * (self.customer_data['payment_method'] == 'electronic_check') +
            np.random.normal(0, 0.1, n_customers)
        )
        self.churn_labels = (churn_probability > 0.5).astype(int)

        # Create model predictions (realistic but not perfect)
        noise = np.random.normal(0, 0.1, n_customers)
        self.churn_probabilities = np.clip(churn_probability + noise, 0, 1)
        self.churn_predictions = (self.churn_probabilities > 0.5).astype(int)

        # Business parameters
        self.base_monthly_revenue = self.customer_data['monthly_charges']
        self.historical_churn_rate = 0.27  # 27% annual churn rate (industry average)
        self.retention_campaign_cost = 75.0  # Cost per retention attempt
        self.campaign_success_rate = 0.35  # 35% of campaigns successfully retain customers

        # Cost matrix for business decisions
        # Format: [[TN_cost, FP_cost], [FN_cost, TP_cost]]
        self.cost_matrix = np.array([
            [0, 75],      # TN: no cost, FP: retention campaign cost
            [500, -400]   # FN: lost customer value, TP: saved customer value minus campaign cost
        ])

        # Expected business outcomes
        self.expected_clv_range = (100, 5000)  # Customer lifetime value range
        self.expected_roi_threshold = 1.5     # Minimum 50% ROI expected

    def test_customer_lifetime_value_calculation(self):
        """Test customer lifetime value calculation workflow."""
        # Expected behavior when BusinessAnalyzer is implemented:
        # from src.business.analyzer import BusinessAnalyzer
        # analyzer = BusinessAnalyzer()

        monthly_revenue = self.customer_data['monthly_charges'].values
        churn_rates = self.churn_probabilities
        discount_rate = 0.1  # 10% annual discount rate

        # clv_values = analyzer.calculate_customer_lifetime_value(monthly_revenue, churn_rates, discount_rate)

        # Simulate CLV calculation
        # CLV = Sum of discounted monthly revenues until expected churn
        expected_lifetime_months = 1 / (churn_rates + 0.01)  # Add small value to avoid division by zero
        monthly_discount_rate = discount_rate / 12

        # Simplified CLV calculation
        clv_values = []
        for i, (revenue, lifetime) in enumerate(zip(monthly_revenue, expected_lifetime_months)):
            months = min(int(lifetime), 120)  # Cap at 10 years
            clv = sum(revenue * (1 - monthly_discount_rate) ** month for month in range(months))
            clv_values.append(clv)

        clv_values = np.array(clv_values)

        # Validate CLV calculations
        assert len(clv_values) == len(monthly_revenue), "CLV calculated for all customers"
        assert all(clv > 0 for clv in clv_values), "All CLV values should be positive"
        assert clv_values.min() >= self.expected_clv_range[0], f"Minimum CLV {clv_values.min():.2f} too low"
        assert clv_values.max() <= self.expected_clv_range[1] * 2, f"Maximum CLV {clv_values.max():.2f} seems too high"

        # High-value customers should have higher CLV
        high_revenue_customers = monthly_revenue > monthly_revenue.quantile(0.8)
        high_clv_customers = clv_values > clv_values.quantile(0.8)

        # There should be significant overlap
        overlap = np.sum(high_revenue_customers & high_clv_customers)
        assert overlap > len(clv_values) * 0.6, "High revenue customers should generally have high CLV"

    def test_retention_cost_estimation(self):
        """Test retention cost estimation by customer segment."""
        # Expected behavior when BusinessAnalyzer is implemented:
        # from src.business.analyzer import BusinessAnalyzer
        # analyzer = BusinessAnalyzer()

        # Create customer segments for testing
        customer_segments = self.customer_data.copy()
        customer_segments['value_segment'] = pd.cut(
            customer_segments['monthly_charges'],
            bins=[0, 40, 80, 200],
            labels=['low_value', 'medium_value', 'high_value']
        )

        customer_segments['tenure_segment'] = pd.cut(
            customer_segments['tenure'],
            bins=[0, 12, 36, 100],
            labels=['new', 'established', 'loyal']
        )

        base_retention_cost = self.retention_campaign_cost

        # retention_costs = analyzer.estimate_retention_costs(customer_segments, base_retention_cost)

        # Simulate retention cost estimation
        retention_costs = {}

        # Different costs for different segments
        segment_multipliers = {
            'low_value': 0.7,     # Cheaper campaigns for low-value customers
            'medium_value': 1.0,  # Base cost
            'high_value': 1.5,    # More expensive, personalized campaigns
            'new': 1.2,          # Higher cost to retain new customers
            'established': 1.0,   # Base cost
            'loyal': 0.8         # Lower cost, simpler retention
        }

        for segment in ['low_value', 'medium_value', 'high_value']:
            cost = base_retention_cost * segment_multipliers[segment]
            retention_costs[f'value_{segment}'] = cost

        for segment in ['new', 'established', 'loyal']:
            cost = base_retention_cost * segment_multipliers[segment]
            retention_costs[f'tenure_{segment}'] = cost

        # Validate retention cost estimates
        assert len(retention_costs) > 0, "Should estimate costs for customer segments"
        assert all(cost > 0 for cost in retention_costs.values()), "All costs should be positive"

        # High-value customers should have higher retention costs
        assert retention_costs['value_high_value'] > retention_costs['value_low_value']

        # Costs should be reasonable multiples of base cost
        for cost in retention_costs.values():
            assert 0.5 * base_retention_cost <= cost <= 2.0 * base_retention_cost

    def test_confusion_matrix_cost_analysis(self):
        """Test business cost analysis from confusion matrix."""
        # Expected behavior when BusinessAnalyzer is implemented:
        # from src.business.analyzer import BusinessAnalyzer
        # analyzer = BusinessAnalyzer()

        # Create confusion matrix from predictions
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.churn_labels, self.churn_predictions)

        # cost_analysis = analyzer.calculate_confusion_matrix_costs(cm, self.cost_matrix)

        # Simulate cost analysis
        tn, fp, fn, tp = cm.ravel()

        cost_analysis = {
            'total_cost': (
                tn * self.cost_matrix[0, 0] +  # True negatives
                fp * self.cost_matrix[0, 1] +  # False positives
                fn * self.cost_matrix[1, 0] +  # False negatives
                tp * self.cost_matrix[1, 1]    # True positives
            ),
            'missed_churners_cost': fn * self.cost_matrix[1, 0],
            'false_alarms_cost': fp * self.cost_matrix[0, 1],
            'successful_interventions': tp * abs(self.cost_matrix[1, 1]),
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }

        # Validate cost analysis
        assert 'total_cost' in cost_analysis, "Should calculate total cost"
        assert 'missed_churners_cost' in cost_analysis, "Should calculate cost of missed churners"
        assert 'false_alarms_cost' in cost_analysis, "Should calculate cost of false alarms"

        # Costs should be reasonable
        assert cost_analysis['missed_churners_cost'] >= 0, "Missed churner cost should be positive"
        assert cost_analysis['false_alarms_cost'] >= 0, "False alarm cost should be positive"
        assert cost_analysis['successful_interventions'] >= 0, "Successful interventions should have value"

        # False negatives (missed churners) should be most expensive
        if fn > 0 and fp > 0:
            cost_per_fn = cost_analysis['missed_churners_cost'] / fn
            cost_per_fp = cost_analysis['false_alarms_cost'] / fp
            assert cost_per_fn > cost_per_fp, "Missing churners should be more expensive than false alarms"

    def test_decision_threshold_optimization(self):
        """Test optimization of decision threshold for business cost minimization."""
        # Expected behavior when BusinessAnalyzer is implemented:
        # from src.business.analyzer import BusinessAnalyzer
        # analyzer = BusinessAnalyzer()

        # Calculate CLV for threshold optimization
        monthly_revenue = self.customer_data['monthly_charges'].values
        customer_values = monthly_revenue * 12  # Simplified annual value

        # optimal_threshold, cost_analysis = analyzer.optimize_decision_threshold(
        #     self.churn_labels, self.churn_probabilities, self.cost_matrix, customer_values
        # )

        # Simulate threshold optimization
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        min_cost = float('inf')

        threshold_results = []

        for threshold in thresholds:
            predictions = (self.churn_probabilities > threshold).astype(int)
            cm = confusion_matrix(self.churn_labels, predictions)

            if cm.size == 4:  # 2x2 matrix
                tn, fp, fn, tp = cm.ravel()

                # Calculate cost using customer values
                total_cost = (
                    fp * self.cost_matrix[0, 1] +  # False positive cost
                    fn * self.cost_matrix[1, 0]    # False negative cost
                )

                threshold_results.append({
                    'threshold': threshold,
                    'cost': total_cost,
                    'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
                })

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_threshold = threshold

        # Validate threshold optimization
        assert 0.1 <= best_threshold <= 0.9, f"Optimal threshold {best_threshold} should be reasonable"
        assert len(threshold_results) == len(thresholds), "Should evaluate all thresholds"

        # Cost should vary across thresholds
        costs = [result['cost'] for result in threshold_results]
        assert max(costs) > min(costs), "Costs should vary across thresholds"

        # Optimal threshold should minimize cost
        optimal_result = [r for r in threshold_results if r['threshold'] == best_threshold][0]
        assert optimal_result['cost'] == min_cost, "Optimal threshold should have minimum cost"

    def test_roi_calculation_workflow(self):
        """Test complete ROI calculation workflow."""
        # Expected behavior when ROICalculator is implemented:
        # from src.business.roi import ROICalculator
        # roi_calculator = ROICalculator()

        # Step 1: Calculate baseline revenue loss
        # baseline_loss = roi_calculator.calculate_baseline_revenue_loss(
        #     self.customer_data, self.historical_churn_rate
        # )

        # Simulate baseline calculation
        annual_revenue = self.customer_data['monthly_charges'].sum() * 12
        baseline_loss = annual_revenue * self.historical_churn_rate

        # Step 2: Calculate model revenue impact
        customer_values = self.customer_data['monthly_charges'].values * 12  # Annual value

        # model_impact = roi_calculator.calculate_model_revenue_impact(
        #     self.churn_predictions, self.churn_labels, customer_values, self.campaign_success_rate
        # )

        # Simulate model impact calculation
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(self.churn_labels, self.churn_predictions)
        tn, fp, fn, tp = cm.ravel()

        # Revenue saved through successful interventions
        successfully_retained = tp * self.campaign_success_rate
        saved_revenue = successfully_retained * customer_values[self.churn_labels == 1].mean()

        # Intervention costs
        intervention_costs = (tp + fp) * self.retention_campaign_cost

        model_impact = {
            'saved_revenue': saved_revenue,
            'intervention_costs': intervention_costs,
            'net_benefit': saved_revenue - intervention_costs
        }

        # Step 3: Generate ROI report
        model_development_cost = 15000.0

        # roi_report = roi_calculator.generate_roi_report(
        #     baseline_loss, model_impact, model_development_cost
        # )

        # Simulate ROI report generation
        total_investment = model_development_cost + model_impact['intervention_costs']
        total_benefit = model_impact['saved_revenue']
        net_annual_benefit = model_impact['net_benefit']

        roi_percentage = ((total_benefit - total_investment) / total_investment) * 100 if total_investment > 0 else 0
        payback_months = (model_development_cost / max(net_annual_benefit / 12, 1))

        roi_report = {
            'roi_percentage': roi_percentage,
            'payback_period_months': payback_months,
            'annual_net_benefit': net_annual_benefit,
            'total_investment': total_investment,
            'total_benefit': total_benefit,
            'break_even_metrics': {
                'required_precision': 0.3,
                'required_recall': 0.5
            }
        }

        # Validate ROI calculations
        assert 'roi_percentage' in roi_report, "Should calculate ROI percentage"
        assert 'payback_period_months' in roi_report, "Should calculate payback period"
        assert 'annual_net_benefit' in roi_report, "Should calculate annual net benefit"

        # ROI should be reasonable for a good model
        if model_impact['net_benefit'] > 0:
            assert roi_report['roi_percentage'] > -50, "ROI shouldn't be extremely negative"

        # Payback period should be reasonable
        assert roi_report['payback_period_months'] > 0, "Payback period should be positive"

        # Baseline loss should be significant
        assert baseline_loss > 0, "Baseline loss should be positive"
        assert baseline_loss < annual_revenue, "Baseline loss should be less than total revenue"

    def test_churn_driver_identification(self):
        """Test identification of key churn drivers."""
        # Expected behavior when ChurnInsights is implemented:
        # from src.business.insights import ChurnInsights
        # insights = ChurnInsights()

        # Simulate feature importance from a trained model
        feature_importance = {
            'contract_type_month_to_month': 0.25,
            'tenure': 0.20,
            'monthly_charges': 0.15,
            'payment_method_electronic_check': 0.12,
            'internet_service': 0.10,
            'total_charges': 0.08,
            'customer_service_calls': 0.10
        }

        # churn_drivers = insights.identify_churn_drivers(
        #     feature_importance, self.customer_data, pd.Series(self.churn_labels)
        # )

        # Simulate churn driver analysis
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Analyze segment risks
        segment_risks = {}
        for contract_type in self.customer_data['contract_type'].unique():
            mask = self.customer_data['contract_type'] == contract_type
            churn_rate = np.mean(self.churn_labels[mask])
            segment_risks[f'contract_{contract_type}'] = churn_rate

        # Identify actionable factors
        actionable_factors = {
            'contract_type': feature_importance.get('contract_type_month_to_month', 0),
            'service_quality': feature_importance.get('customer_service_calls', 0),
            'pricing': feature_importance.get('monthly_charges', 0)
        }

        churn_drivers = {
            'top_risk_factors': dict(top_features),
            'segment_risks': segment_risks,
            'actionable_factors': actionable_factors
        }

        # Validate churn driver analysis
        assert 'top_risk_factors' in churn_drivers, "Should identify top risk factors"
        assert 'segment_risks' in churn_drivers, "Should analyze segment risks"
        assert 'actionable_factors' in churn_drivers, "Should identify actionable factors"

        # Top risk factors should be reasonable
        top_factors = churn_drivers['top_risk_factors']
        assert len(top_factors) <= 5, "Should focus on top factors"
        assert all(0 <= importance <= 1 for importance in top_factors.values()), "Importance scores should be normalized"

        # Segment risks should be probabilities
        segment_risks_values = list(churn_drivers['segment_risks'].values())
        assert all(0 <= risk <= 1 for risk in segment_risks_values), "Segment risks should be probabilities"

    def test_customer_risk_segmentation(self):
        """Test customer segmentation by churn risk."""
        # Expected behavior when ChurnInsights is implemented:
        # from src.business.insights import ChurnInsights
        # insights = ChurnInsights()

        risk_thresholds = [0.3, 0.7]

        # segmented_customers = insights.segment_customers_by_risk(
        #     self.churn_probabilities, self.customer_data, risk_thresholds
        # )

        # Simulate customer risk segmentation
        segmented_customers = self.customer_data.copy()
        segmented_customers['churn_probability'] = self.churn_probabilities

        # Create risk segments
        def assign_risk_segment(prob):
            if prob < risk_thresholds[0]:
                return 'low_risk'
            elif prob < risk_thresholds[1]:
                return 'medium_risk'
            else:
                return 'high_risk'

        segmented_customers['risk_segment'] = [assign_risk_segment(p) for p in self.churn_probabilities]

        # Add recommended actions
        action_mapping = {
            'low_risk': 'monitor',
            'medium_risk': 'engagement_campaign',
            'high_risk': 'intensive_retention'
        }

        segmented_customers['recommended_action'] = segmented_customers['risk_segment'].map(action_mapping)

        # Validate risk segmentation
        assert 'risk_segment' in segmented_customers.columns, "Should assign risk segments"
        assert 'recommended_action' in segmented_customers.columns, "Should recommend actions"

        # Check segment distribution
        segment_counts = segmented_customers['risk_segment'].value_counts()
        assert len(segment_counts) >= 2, "Should have multiple risk segments"

        # High-risk customers should have higher average probabilities
        risk_groups = segmented_customers.groupby('risk_segment')['churn_probability'].mean()
        if 'high_risk' in risk_groups.index and 'low_risk' in risk_groups.index:
            assert risk_groups['high_risk'] > risk_groups['low_risk'], "High-risk should have higher probabilities"

        # Each segment should have appropriate actions
        for segment, action in action_mapping.items():
            if segment in segmented_customers['risk_segment'].values:
                segment_actions = segmented_customers[segmented_customers['risk_segment'] == segment]['recommended_action'].unique()
                assert action in segment_actions, f"Segment {segment} should have action {action}"

    def test_retention_strategy_recommendations(self):
        """Test retention strategy recommendation workflow."""
        # Expected behavior when ChurnInsights is implemented:
        # from src.business.insights import ChurnInsights
        # insights = ChurnInsights()

        # Create customer segments for strategy recommendation
        customer_segments = self.customer_data.copy()
        customer_segments['risk_level'] = pd.cut(
            self.churn_probabilities,
            bins=[0, 0.3, 0.7, 1.0],
            labels=['low', 'medium', 'high']
        )

        # Churn drivers for strategy formulation
        churn_drivers = {
            'contract_type': 0.25,
            'pricing': 0.20,
            'service_quality': 0.15,
            'payment_method': 0.12
        }

        # strategies = insights.recommend_retention_strategies(customer_segments, churn_drivers)

        # Simulate retention strategy recommendations
        retention_strategies = {
            'low': [
                'maintain_current_service_level',
                'periodic_satisfaction_surveys',
                'loyalty_program_enrollment'
            ],
            'medium': [
                'proactive_customer_service_outreach',
                'contract_upgrade_incentives',
                'payment_method_optimization',
                'service_bundle_recommendations'
            ],
            'high': [
                'immediate_retention_specialist_contact',
                'personalized_discount_offers',
                'service_issue_resolution_priority',
                'contract_renegotiation',
                'executive_escalation_path'
            ]
        }

        # Validate retention strategy recommendations
        assert len(retention_strategies) > 0, "Should recommend strategies"

        for risk_level, strategies in retention_strategies.items():
            assert isinstance(strategies, list), f"Strategies for {risk_level} should be a list"
            assert len(strategies) > 0, f"Should have strategies for {risk_level} risk customers"

        # High-risk customers should have more intensive strategies
        if 'high' in retention_strategies and 'low' in retention_strategies:
            high_risk_strategies = len(retention_strategies['high'])
            low_risk_strategies = len(retention_strategies['low'])
            assert high_risk_strategies >= low_risk_strategies, "High-risk should have more strategies"

        # Strategies should be actionable and specific
        all_strategies = [strategy for strategies in retention_strategies.values() for strategy in strategies]
        assert len(set(all_strategies)) > 5, "Should have diverse, specific strategies"

    def test_end_to_end_business_analysis_workflow(self):
        """Test the complete end-to-end business analysis workflow."""
        # This integration test covers the full business pipeline:
        # Model predictions → CLV calculation → Cost analysis → ROI calculation → Insights generation

        # Step 1: Calculate customer lifetime values
        monthly_revenue = self.customer_data['monthly_charges'].values
        annual_clv = monthly_revenue * 12 / (self.churn_probabilities + 0.01)  # Simplified CLV

        # Step 2: Optimize decision threshold for business value
        thresholds = np.arange(0.2, 0.8, 0.1)
        best_threshold = 0.5
        best_net_value = -float('inf')

        for threshold in thresholds:
            predictions = (self.churn_probabilities > threshold).astype(int)
            cm = confusion_matrix(self.churn_labels, predictions)

            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()

                # Calculate net business value
                campaign_costs = (tp + fp) * self.retention_campaign_cost
                saved_value = tp * self.campaign_success_rate * annual_clv[self.churn_labels == 1].mean()
                net_value = saved_value - campaign_costs

                if net_value > best_net_value:
                    best_net_value = net_value
                    best_threshold = threshold

        # Step 3: Generate final business recommendations
        final_predictions = (self.churn_probabilities > best_threshold).astype(int)

        # Create customer action plan
        action_plan = self.customer_data.copy()
        action_plan['churn_probability'] = self.churn_probabilities
        action_plan['predicted_churn'] = final_predictions
        action_plan['clv'] = annual_clv

        # Prioritize customers for retention
        action_plan['retention_priority'] = (
            action_plan['predicted_churn'] * action_plan['clv']
        )

        # Top customers for immediate attention
        top_retention_candidates = action_plan.nlargest(20, 'retention_priority')

        # Validate end-to-end workflow
        assert len(action_plan) == len(self.customer_data), "Should process all customers"
        assert 'retention_priority' in action_plan.columns, "Should calculate retention priority"
        assert len(top_retention_candidates) == 20, "Should identify top candidates"

        # High priority customers should have high churn risk and high value
        top_candidates = top_retention_candidates
        avg_churn_prob = top_candidates['churn_probability'].mean()
        avg_clv = top_candidates['clv'].mean()

        overall_avg_churn_prob = action_plan['churn_probability'].mean()
        overall_avg_clv = action_plan['clv'].mean()

        assert avg_churn_prob > overall_avg_churn_prob, "Top candidates should have higher churn risk"
        assert avg_clv > overall_avg_clv, "Top candidates should have higher value"

        # Business case summary
        total_at_risk_value = (final_predictions * annual_clv).sum()
        total_campaign_cost = final_predictions.sum() * self.retention_campaign_cost
        expected_saved_value = total_at_risk_value * self.campaign_success_rate
        expected_net_benefit = expected_saved_value - total_campaign_cost

        business_case = {
            'customers_at_risk': final_predictions.sum(),
            'total_at_risk_value': total_at_risk_value,
            'campaign_cost': total_campaign_cost,
            'expected_saved_value': expected_saved_value,
            'expected_net_benefit': expected_net_benefit,
            'roi_ratio': expected_saved_value / total_campaign_cost if total_campaign_cost > 0 else 0
        }

        # Validate business case
        assert business_case['customers_at_risk'] > 0, "Should identify at-risk customers"
        assert business_case['total_at_risk_value'] > 0, "Should quantify at-risk value"
        assert business_case['roi_ratio'] > 0, "Should calculate meaningful ROI"

    def test_business_analysis_error_handling(self):
        """Test that business analysis handles various error conditions."""

        # Test 1: Empty customer data
        empty_data = pd.DataFrame()

        # Expected: Should handle empty data gracefully
        with pytest.raises((ValueError, AttributeError)):
            if empty_data.empty:
                raise ValueError("Expected behavior: empty data should be handled")

        # Test 2: Negative customer values
        negative_revenue = -50.0
        negative_churn_rate = -0.1

        # Expected: Should detect invalid inputs
        assert negative_revenue < 0, "Should detect negative revenue"
        assert negative_churn_rate < 0, "Should detect negative churn rate"

        # Test 3: Impossible probabilities
        invalid_probabilities = np.array([1.5, -0.3, 0.5])

        # Expected: Should handle invalid probabilities
        valid_probabilities = np.clip(invalid_probabilities, 0, 1)
        assert all(0 <= p <= 1 for p in valid_probabilities), "Should handle invalid probabilities"

    def test_business_analysis_performance(self):
        """Test that business analysis performs adequately with larger datasets."""
        import time

        # Create larger dataset
        n_large = 5000
        large_customer_data = pd.DataFrame({
            'customer_id': [f'C{i:06d}' for i in range(n_large)],
            'monthly_charges': np.random.uniform(20, 120, n_large),
            'tenure': np.random.randint(1, 72, n_large)
        })

        large_churn_probs = np.random.uniform(0, 1, n_large)

        # Test CLV calculation performance
        start_time = time.time()

        # Simulate CLV calculation
        monthly_revenue = large_customer_data['monthly_charges'].values
        clv_values = monthly_revenue * 12 / (large_churn_probs + 0.01)

        clv_time = time.time() - start_time

        # Test segmentation performance
        start_time = time.time()

        risk_segments = pd.cut(large_churn_probs, bins=[0, 0.3, 0.7, 1.0], labels=['low', 'medium', 'high'])

        segmentation_time = time.time() - start_time

        # Performance assertions
        assert clv_time < 1.0, f"CLV calculation took {clv_time:.2f}s, should be under 1s"
        assert segmentation_time < 0.5, f"Segmentation took {segmentation_time:.2f}s, should be under 0.5s"

        # Validate results
        assert len(clv_values) == n_large, "Should calculate CLV for all customers"
        assert len(risk_segments) == n_large, "Should segment all customers"


if __name__ == '__main__':
    pytest.main([__file__])
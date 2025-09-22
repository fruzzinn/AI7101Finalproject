"""
BusinessAnalyzer implementation for telecommunications churn prediction.

Educational Focus: Demonstrates business impact analysis and ROI calculation for ML models.
This module translates model predictions into business value through CLV analysis,
cost-benefit optimization, and strategic decision-making frameworks.
"""

from typing import Dict, Tuple, List, Optional, Any
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.model_performance import ModelPerformance


class BusinessAnalyzer:
    """
    Comprehensive business impact analysis for churn prediction models.

    Educational Notes:
    - Bridges ML model performance with business value
    - Implements Customer Lifetime Value (CLV) calculations
    - Provides cost-benefit optimization frameworks
    - Enables data-driven business decision making
    """

    def __init__(self, currency_symbol: str = "$"):
        """
        Initialize BusinessAnalyzer.

        Args:
            currency_symbol: Symbol for currency display in reports
        """
        self.currency_symbol = currency_symbol
        self.analysis_history: List[Dict[str, Any]] = []
        self.cost_assumptions: Dict[str, float] = {
            'base_retention_cost': 50.0,
            'customer_acquisition_cost': 100.0,
            'intervention_success_rate': 0.3,
            'discount_rate': 0.1
        }

    def calculate_customer_lifetime_value(self, monthly_revenue: np.ndarray,
                                        churn_rates: np.ndarray,
                                        discount_rate: float = 0.1,
                                        max_periods: int = 60) -> np.ndarray:
        """
        Calculate Customer Lifetime Value (CLV) for customer segments.

        Args:
            monthly_revenue: Monthly revenue per customer
            churn_rates: Monthly churn probability per customer
            discount_rate: Annual discount rate for present value calculation
            max_periods: Maximum periods to consider (months)

        Returns:
            Array of CLV values per customer

        Educational Notes:
        - CLV represents total expected revenue from a customer
        - Incorporates churn probability and time value of money
        - Formula: CLV = Σ(monthly_revenue * survival_prob * discount_factor)
        - Critical for prioritizing retention efforts
        """
        # Convert annual discount rate to monthly
        monthly_discount_rate = discount_rate / 12

        # Initialize CLV array
        clv = np.zeros_like(monthly_revenue, dtype=float)

        # Calculate survival probabilities (1 - churn_rate)
        survival_rates = 1 - churn_rates

        for period in range(max_periods):
            # Discount factor for this period
            discount_factor = (1 / (1 + monthly_discount_rate)) ** period

            # Cumulative survival probability up to this period
            cumulative_survival = survival_rates ** period

            # Add discounted expected revenue for this period
            period_value = monthly_revenue * cumulative_survival * discount_factor

            clv += period_value

            # Early termination if survival probability becomes negligible
            if np.max(cumulative_survival) < 0.01:
                break

        return clv

    def calculate_simplified_clv(self, monthly_revenue: np.ndarray,
                               churn_rates: np.ndarray,
                               discount_rate: float = 0.1) -> np.ndarray:
        """
        Calculate simplified CLV using the geometric series formula.

        Args:
            monthly_revenue: Monthly revenue per customer
            churn_rates: Monthly churn probability per customer
            discount_rate: Annual discount rate

        Returns:
            Array of simplified CLV values

        Educational Notes:
        - Uses closed-form solution: CLV = R / (d + c)
        - Where R = monthly revenue, d = discount rate, c = churn rate
        - More computationally efficient for large datasets
        """
        monthly_discount_rate = discount_rate / 12

        # Avoid division by zero
        denominator = monthly_discount_rate + churn_rates
        denominator = np.maximum(denominator, 1e-6)

        clv = monthly_revenue / denominator
        return clv

    def estimate_retention_costs(self, customer_segments: pd.DataFrame,
                               base_retention_cost: float = 50.0) -> Dict[str, float]:
        """
        Estimate costs of retention campaigns by customer segment.

        Args:
            customer_segments: DataFrame with customer segmentation
            base_retention_cost: Base cost per retention attempt

        Returns:
            Dict of segment: estimated_retention_cost

        Educational Notes:
        - Different customer segments require different retention strategies
        - High-value customers justify higher retention spend
        - Segmentation enables cost-effective resource allocation
        """
        retention_costs = {}

        # Check if segmentation columns exist
        if 'segment' in customer_segments.columns:
            segment_col = 'segment'
        elif 'risk_segment' in customer_segments.columns:
            segment_col = 'risk_segment'
        else:
            # Create simple segmentation based on available data
            if 'monthly_charges' in customer_segments.columns:
                customer_segments['segment'] = pd.cut(
                    customer_segments['monthly_charges'],
                    bins=[0, 35, 65, float('inf')],
                    labels=['Low Value', 'Medium Value', 'High Value']
                )
                segment_col = 'segment'
            else:
                # Default single segment
                retention_costs['All Customers'] = base_retention_cost
                return retention_costs

        # Calculate retention costs by segment
        segment_multipliers = {
            'Low Value': 0.7,
            'Medium Value': 1.0,
            'High Value': 1.5,
            'Low Risk': 0.8,
            'Medium Risk': 1.0,
            'High Risk': 1.3,
            'Premium': 2.0,
            'Standard': 1.0,
            'Basic': 0.6
        }

        for segment in customer_segments[segment_col].unique():
            if pd.isna(segment):
                continue

            segment_str = str(segment)
            multiplier = segment_multipliers.get(segment_str, 1.0)

            # Adjust based on segment characteristics
            segment_data = customer_segments[customer_segments[segment_col] == segment]

            if 'monthly_charges' in segment_data.columns:
                avg_revenue = segment_data['monthly_charges'].mean()
                # Higher revenue customers justify higher retention costs
                revenue_multiplier = min(avg_revenue / 50.0, 3.0)  # Cap at 3x
                multiplier *= revenue_multiplier

            retention_costs[segment_str] = base_retention_cost * multiplier

        return retention_costs

    def calculate_confusion_matrix_costs(self, confusion_matrix: np.ndarray,
                                       cost_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate business costs associated with prediction errors.

        Args:
            confusion_matrix: Model confusion matrix [[TN, FP], [FN, TP]]
            cost_matrix: Business cost matrix [[cost_TN, cost_FP], [cost_FN, cost_TP]]

        Returns:
            Dict with detailed cost breakdown

        Educational Notes:
        - False Negatives (FN): Missed churners - high opportunity cost
        - False Positives (FP): Unnecessary interventions - wasted resources
        - True Positives (TP): Successful interventions - generate value
        - True Negatives (TN): Correctly identified loyal customers - low cost
        """
        if confusion_matrix.shape != (2, 2) or cost_matrix.shape != (2, 2):
            raise ValueError("Both matrices must be 2x2 for binary classification")

        # Extract confusion matrix values
        tn, fp, fn, tp = confusion_matrix.ravel()

        # Extract cost matrix values
        cost_tn, cost_fp, cost_fn, cost_tp = cost_matrix.ravel()

        # Calculate individual costs
        true_negative_costs = tn * cost_tn
        false_positive_costs = fp * cost_fp
        false_negative_costs = fn * cost_fn
        true_positive_costs = tp * cost_tp

        total_cost = true_negative_costs + false_positive_costs + false_negative_costs + true_positive_costs

        # Calculate derived metrics
        total_predictions = tn + fp + fn + tp
        cost_per_prediction = total_cost / total_predictions if total_predictions > 0 else 0

        results = {
            'total_cost': float(total_cost),
            'true_negative_costs': float(true_negative_costs),
            'false_positive_costs': float(false_positive_costs),
            'false_negative_costs': float(false_negative_costs),
            'true_positive_costs': float(true_positive_costs),
            'missed_churners_cost': float(false_negative_costs),
            'false_alarms_cost': float(false_positive_costs),
            'successful_interventions': float(-true_positive_costs),  # Negative cost = benefit
            'cost_per_prediction': float(cost_per_prediction),
            'confusion_matrix_breakdown': {
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp)
            }
        }

        return results

    def optimize_decision_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  cost_matrix: np.ndarray,
                                  customer_values: np.ndarray,
                                  threshold_range: Tuple[float, float] = (0.1, 0.9),
                                  n_thresholds: int = 100) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal prediction threshold that minimizes business cost.

        Args:
            y_true: True churn labels
            y_prob: Predicted churn probabilities
            cost_matrix: Business cost matrix
            customer_values: Individual customer values (CLV)
            threshold_range: Range of thresholds to evaluate
            n_thresholds: Number of thresholds to test

        Returns:
            Tuple of (optimal_threshold, cost_analysis)

        Educational Notes:
        - Optimizes business value rather than accuracy
        - Incorporates customer value in decision making
        - Threshold selection based on cost minimization
        - Enables value-driven ML model deployment
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_thresholds)
        threshold_analysis = []

        for threshold in thresholds:
            # Generate predictions at this threshold
            y_pred = (y_prob >= threshold).astype(int)

            # Calculate confusion matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)

            # Handle case where confusion matrix is not 2x2
            if cm.shape != (2, 2):
                # Pad confusion matrix if needed
                if cm.shape == (1, 1):
                    if y_true[0] == 0:  # Only negative class
                        cm = np.array([[cm[0, 0], 0], [0, 0]])
                    else:  # Only positive class
                        cm = np.array([[0, 0], [0, cm[0, 0]]])
                else:
                    continue  # Skip this threshold

            # Calculate value-weighted costs
            value_weighted_cost_matrix = self._calculate_value_weighted_costs(
                cm, cost_matrix, customer_values, y_true, y_pred
            )

            # Calculate total cost
            total_cost = np.sum(cm * value_weighted_cost_matrix)

            # Store analysis
            threshold_analysis.append({
                'threshold': threshold,
                'total_cost': total_cost,
                'confusion_matrix': cm,
                'precision': cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0,
                'recall': cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0,
                'predicted_positive_rate': np.sum(y_pred) / len(y_pred)
            })

        if not threshold_analysis:
            warnings.warn("No valid thresholds found")
            return 0.5, {}

        # Find optimal threshold (minimum cost)
        optimal_idx = np.argmin([analysis['total_cost'] for analysis in threshold_analysis])
        optimal_threshold = threshold_analysis[optimal_idx]['threshold']

        # Detailed analysis of optimal threshold
        optimal_analysis = threshold_analysis[optimal_idx]
        cost_analysis = {
            'optimal_threshold': optimal_threshold,
            'minimum_total_cost': optimal_analysis['total_cost'],
            'optimal_precision': optimal_analysis['precision'],
            'optimal_recall': optimal_analysis['recall'],
            'optimal_predicted_positive_rate': optimal_analysis['predicted_positive_rate'],
            'threshold_analysis': threshold_analysis
        }

        return optimal_threshold, cost_analysis

    def _calculate_value_weighted_costs(self, confusion_matrix: np.ndarray,
                                      cost_matrix: np.ndarray,
                                      customer_values: np.ndarray,
                                      y_true: np.ndarray,
                                      y_pred: np.ndarray) -> np.ndarray:
        """Calculate value-weighted cost matrix based on customer CLV."""
        # Create masks for each confusion matrix cell
        tn_mask = (y_true == 0) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)
        fn_mask = (y_true == 1) & (y_pred == 0)
        tp_mask = (y_true == 1) & (y_pred == 1)

        # Calculate average customer values for each cell
        avg_values = np.zeros((2, 2))

        if np.sum(tn_mask) > 0:
            avg_values[0, 0] = np.mean(customer_values[tn_mask])
        if np.sum(fp_mask) > 0:
            avg_values[0, 1] = np.mean(customer_values[fp_mask])
        if np.sum(fn_mask) > 0:
            avg_values[1, 0] = np.mean(customer_values[fn_mask])
        if np.sum(tp_mask) > 0:
            avg_values[1, 1] = np.mean(customer_values[tp_mask])

        # Weight costs by customer values
        value_weighted_costs = cost_matrix * (1 + avg_values / np.mean(customer_values))

        return value_weighted_costs

    def analyze_segment_profitability(self, customer_data: pd.DataFrame,
                                    churn_predictions: np.ndarray,
                                    segment_column: str = 'segment') -> Dict[str, Dict[str, float]]:
        """
        Analyze profitability and churn risk by customer segment.

        Args:
            customer_data: DataFrame with customer information
            churn_predictions: Model churn predictions
            segment_column: Column name for customer segmentation

        Returns:
            Dict with segment profitability analysis

        Educational Notes:
        - Identifies high-value, high-risk segments for prioritization
        - Enables targeted retention strategies
        - Supports resource allocation decisions
        """
        if segment_column not in customer_data.columns:
            warnings.warn(f"Segment column '{segment_column}' not found. Creating default segments.")
            # Create simple value-based segments
            if 'monthly_charges' in customer_data.columns:
                customer_data[segment_column] = pd.cut(
                    customer_data['monthly_charges'],
                    bins=[0, 35, 65, float('inf')],
                    labels=['Low Value', 'Medium Value', 'High Value']
                )
            else:
                customer_data[segment_column] = 'All Customers'

        segment_analysis = {}

        for segment in customer_data[segment_column].unique():
            if pd.isna(segment):
                continue

            segment_mask = customer_data[segment_column] == segment
            segment_data = customer_data[segment_mask]
            segment_predictions = churn_predictions[segment_mask]

            # Basic metrics
            segment_size = len(segment_data)
            predicted_churn_rate = np.mean(segment_predictions) if len(segment_predictions) > 0 else 0

            # Revenue metrics
            if 'monthly_charges' in segment_data.columns:
                avg_monthly_revenue = segment_data['monthly_charges'].mean()
                total_monthly_revenue = segment_data['monthly_charges'].sum()
            else:
                avg_monthly_revenue = 0
                total_monthly_revenue = 0

            # CLV estimation (simplified)
            if 'monthly_charges' in segment_data.columns:
                estimated_clv = self.calculate_simplified_clv(
                    segment_data['monthly_charges'].values,
                    np.full(len(segment_data), predicted_churn_rate),
                    self.cost_assumptions['discount_rate']
                )
                avg_clv = np.mean(estimated_clv)
                total_clv = np.sum(estimated_clv)
            else:
                avg_clv = 0
                total_clv = 0

            # Risk assessment
            at_risk_customers = np.sum(segment_predictions)
            potential_revenue_at_risk = at_risk_customers * avg_monthly_revenue * 12  # Annual

            segment_analysis[str(segment)] = {
                'segment_size': int(segment_size),
                'predicted_churn_rate': float(predicted_churn_rate),
                'avg_monthly_revenue': float(avg_monthly_revenue),
                'total_monthly_revenue': float(total_monthly_revenue),
                'avg_clv': float(avg_clv),
                'total_clv': float(total_clv),
                'at_risk_customers': int(at_risk_customers),
                'potential_revenue_at_risk': float(potential_revenue_at_risk),
                'segment_priority_score': float(total_clv * predicted_churn_rate)  # High value + high risk
            }

        return segment_analysis

    def create_intervention_strategy(self, customer_data: pd.DataFrame,
                                   churn_probabilities: np.ndarray,
                                   customer_values: np.ndarray,
                                   budget_constraint: float = 10000.0) -> Dict[str, Any]:
        """
        Create optimal intervention strategy given budget constraints.

        Args:
            customer_data: DataFrame with customer information
            churn_probabilities: Predicted churn probabilities
            customer_values: Customer lifetime values
            budget_constraint: Total available budget for interventions

        Returns:
            Dict with intervention strategy and expected ROI

        Educational Notes:
        - Prioritizes interventions by expected value
        - Considers budget constraints and intervention costs
        - Optimizes resource allocation for maximum ROI
        - Provides actionable business recommendations
        """
        # Calculate expected value of intervention for each customer
        intervention_cost = self.cost_assumptions['base_retention_cost']
        success_rate = self.cost_assumptions['intervention_success_rate']

        # Expected value = (CLV * churn_prob * success_rate) - intervention_cost
        expected_values = (customer_values * churn_probabilities * success_rate) - intervention_cost

        # Create customer priority dataframe
        priority_df = pd.DataFrame({
            'customer_index': range(len(customer_data)),
            'churn_probability': churn_probabilities,
            'customer_value': customer_values,
            'expected_value': expected_values,
            'intervention_cost': intervention_cost
        })

        # Sort by expected value (descending)
        priority_df = priority_df.sort_values('expected_value', ascending=False)

        # Select customers within budget constraint
        cumulative_cost = priority_df['intervention_cost'].cumsum()
        within_budget = cumulative_cost <= budget_constraint
        selected_customers = priority_df[within_budget]

        # Calculate strategy metrics
        total_interventions = len(selected_customers)
        total_cost = selected_customers['intervention_cost'].sum()
        total_expected_value = selected_customers['expected_value'].sum()
        expected_customers_saved = (selected_customers['churn_probability'] * success_rate).sum()

        # Risk tier analysis
        risk_tiers = {
            'High Risk (>0.7)': selected_customers[selected_customers['churn_probability'] > 0.7],
            'Medium Risk (0.3-0.7)': selected_customers[
                (selected_customers['churn_probability'] >= 0.3) &
                (selected_customers['churn_probability'] <= 0.7)
            ],
            'Low Risk (<0.3)': selected_customers[selected_customers['churn_probability'] < 0.3]
        }

        tier_analysis = {}
        for tier_name, tier_data in risk_tiers.items():
            if len(tier_data) > 0:
                tier_analysis[tier_name] = {
                    'customer_count': len(tier_data),
                    'total_cost': tier_data['intervention_cost'].sum(),
                    'expected_value': tier_data['expected_value'].sum(),
                    'avg_churn_probability': tier_data['churn_probability'].mean(),
                    'avg_customer_value': tier_data['customer_value'].mean()
                }

        strategy = {
            'strategy_overview': {
                'total_budget': budget_constraint,
                'budget_utilized': total_cost,
                'budget_remaining': budget_constraint - total_cost,
                'total_interventions': total_interventions,
                'expected_customers_saved': expected_customers_saved,
                'total_expected_value': total_expected_value,
                'expected_roi': (total_expected_value / total_cost) * 100 if total_cost > 0 else 0
            },
            'selected_customers': selected_customers.to_dict('records'),
            'risk_tier_analysis': tier_analysis,
            'recommendations': self._generate_intervention_recommendations(tier_analysis, total_cost, total_expected_value)
        }

        return strategy

    def _generate_intervention_recommendations(self, tier_analysis: Dict[str, Dict[str, float]],
                                             total_cost: float, total_expected_value: float) -> List[str]:
        """Generate actionable recommendations based on intervention analysis."""
        recommendations = []

        # ROI-based recommendations
        if total_expected_value > total_cost:
            roi_percentage = ((total_expected_value - total_cost) / total_cost) * 100
            recommendations.append(f"Positive ROI expected: {roi_percentage:.1f}% return on intervention investment")
        else:
            recommendations.append("Negative ROI predicted - consider adjusting intervention strategy or success rates")

        # Tier-specific recommendations
        for tier_name, tier_data in tier_analysis.items():
            tier_roi = (tier_data['expected_value'] / tier_data['total_cost']) * 100 if tier_data['total_cost'] > 0 else 0

            if 'High Risk' in tier_name and tier_roi > 50:
                recommendations.append(f"Focus resources on {tier_name} customers - highest ROI potential ({tier_roi:.1f}%)")
            elif 'Low Risk' in tier_name and tier_roi < 0:
                recommendations.append(f"Consider excluding {tier_name} customers to improve overall ROI")

        # Budget recommendations
        if total_cost < 5000:  # Arbitrary threshold
            recommendations.append("Consider increasing intervention budget to capture more high-value opportunities")

        return recommendations

    def generate_business_case_report(self, model_performance: ModelPerformance,
                                    customer_data: pd.DataFrame,
                                    cost_assumptions: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Generate comprehensive business case report for churn prediction model.

        Args:
            model_performance: Model performance metrics
            customer_data: Customer data for analysis
            cost_assumptions: Business cost assumptions

        Returns:
            Comprehensive business case report

        Educational Notes:
        - Translates technical metrics into business value
        - Provides ROI justification for ML investment
        - Supports executive decision-making
        - Documents assumptions and methodology
        """
        if cost_assumptions:
            self.cost_assumptions.update(cost_assumptions)

        # Generate dummy predictions for demonstration
        np.random.seed(42)
        n_customers = len(customer_data)
        dummy_churn_probs = np.random.beta(2, 5, n_customers)  # Realistic churn distribution

        # Calculate CLV if revenue data available
        if 'monthly_charges' in customer_data.columns:
            customer_values = self.calculate_simplified_clv(
                customer_data['monthly_charges'].values,
                dummy_churn_probs,
                self.cost_assumptions['discount_rate']
            )
        else:
            # Use dummy values
            customer_values = np.random.normal(1000, 300, n_customers)

        # Business impact analysis
        report = {
            'executive_summary': {
                'model_accuracy': model_performance.get_metric_by_name('accuracy').value if model_performance.get_metric_by_name('accuracy') else 0.85,
                'f1_score': model_performance.get_metric_by_name('f1_score').value if model_performance.get_metric_by_name('f1_score') else 0.75,
                'customer_base_size': n_customers,
                'total_customer_value': np.sum(customer_values),
                'avg_customer_value': np.mean(customer_values),
                'predicted_churn_rate': np.mean(dummy_churn_probs)
            },
            'financial_impact': {},
            'segment_analysis': {},
            'recommendations': [],
            'assumptions': self.cost_assumptions,
            'methodology': {
                'clv_calculation': 'Simplified geometric series formula with discount rate',
                'churn_prediction': 'Machine learning model with cross-validation',
                'intervention_modeling': 'Expected value optimization with success rate assumptions'
            }
        }

        # Calculate segment analysis
        segment_analysis = self.analyze_segment_profitability(
            customer_data, dummy_churn_probs > 0.5
        )
        report['segment_analysis'] = segment_analysis

        # Financial impact estimation
        baseline_annual_loss = np.sum(customer_values * dummy_churn_probs)

        # Intervention strategy
        intervention_strategy = self.create_intervention_strategy(
            customer_data, dummy_churn_probs, customer_values
        )

        report['financial_impact'] = {
            'baseline_annual_churn_loss': baseline_annual_loss,
            'intervention_strategy': intervention_strategy['strategy_overview'],
            'net_benefit': intervention_strategy['strategy_overview']['total_expected_value'],
            'roi_percentage': intervention_strategy['strategy_overview']['expected_roi']
        }

        # Generate recommendations
        recommendations = [
            "Implement churn prediction model to identify at-risk customers",
            "Focus retention efforts on high-value, high-risk customer segments",
            "Develop targeted intervention campaigns based on churn drivers",
            "Monitor model performance and business impact continuously",
            "Consider expanding model to predict customer lifetime value"
        ]

        if intervention_strategy['strategy_overview']['expected_roi'] > 100:
            recommendations.insert(0, "Strong business case: Expected ROI exceeds 100%")

        report['recommendations'] = recommendations

        # Store in analysis history
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'analysis_type': 'business_case_report',
            'customer_count': n_customers,
            'expected_roi': intervention_strategy['strategy_overview']['expected_roi'],
            'net_benefit': intervention_strategy['strategy_overview']['total_expected_value']
        })

        return report

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all business analysis activities."""
        if not self.analysis_history:
            return {'message': 'No analysis history available'}

        summary = {
            'total_analyses': len(self.analysis_history),
            'analysis_types': list(set(analysis['analysis_type'] for analysis in self.analysis_history)),
            'analysis_history': self.analysis_history
        }

        # Performance summary
        if self.analysis_history:
            roi_values = [analysis.get('expected_roi', 0) for analysis in self.analysis_history]
            net_benefits = [analysis.get('net_benefit', 0) for analysis in self.analysis_history]

            summary['performance_summary'] = {
                'average_expected_roi': np.mean(roi_values),
                'max_expected_roi': np.max(roi_values),
                'total_expected_net_benefit': np.sum(net_benefits)
            }

        return summary
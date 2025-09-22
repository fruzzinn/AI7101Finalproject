"""
Contract: Business Impact Analysis Interface
Educational Focus: Demonstrate ML model business value assessment
"""

from typing import Dict, Tuple, List, Optional
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BusinessAnalyzerContract(ABC):
    """Contract for analyzing business impact of churn prediction models"""

    @abstractmethod
    def calculate_customer_lifetime_value(self, monthly_revenue: np.ndarray,
                                        churn_rates: np.ndarray,
                                        discount_rate: float = 0.1) -> np.ndarray:
        """
        Calculate Customer Lifetime Value (CLV) for customer segments

        Args:
            monthly_revenue: Monthly revenue per customer
            churn_rates: Monthly churn probability per customer
            discount_rate: Discount rate for future value calculation

        Returns:
            Array of CLV values per customer

        Educational Notes:
            - CLV = Sum of discounted future revenues until churn
            - Incorporates churn probability in value calculation
        """
        pass

    @abstractmethod
    def estimate_retention_costs(self, customer_segments: pd.DataFrame,
                               base_retention_cost: float = 50.0) -> Dict[str, float]:
        """
        Estimate costs of retention campaigns by customer segment

        Args:
            customer_segments: DataFrame with customer segmentation
            base_retention_cost: Base cost per retention attempt

        Returns:
            Dict of segment: estimated_retention_cost

        Educational Notes:
            - Different retention costs for different customer types
            - Campaign effectiveness varies by segment
        """
        pass

    @abstractmethod
    def calculate_confusion_matrix_costs(self, confusion_matrix: np.ndarray,
                                       cost_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate business costs associated with prediction errors

        Args:
            confusion_matrix: Model confusion matrix [[TN, FP], [FN, TP]]
            cost_matrix: Business cost matrix [[cost_TN, cost_FP], [cost_FN, cost_TP]]

        Returns:
            Dict with cost breakdown:
            - total_cost: Total business cost
            - missed_churners_cost: Cost of undetected churners (FN)
            - false_alarms_cost: Cost of unnecessary interventions (FP)
            - successful_interventions: Value from prevented churn (TP)

        Educational Notes:
            - False negatives: Lost customers (high cost)
            - False positives: Wasted retention spend (medium cost)
        """
        pass

    @abstractmethod
    def optimize_decision_threshold(self, y_true: np.ndarray, y_prob: np.ndarray,
                                  cost_matrix: np.ndarray,
                                  customer_values: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal prediction threshold that minimizes business cost

        Args:
            y_true: True churn labels
            y_prob: Predicted churn probabilities
            cost_matrix: Business cost matrix
            customer_values: Individual customer values (CLV)

        Returns:
            Tuple of (optimal_threshold, cost_analysis)

        Educational Notes:
            - Threshold optimization beyond accuracy maximization
            - Customer value-weighted decision making
        """
        pass


class ROICalculatorContract(ABC):
    """Contract for calculating return on investment of churn prediction"""

    @abstractmethod
    def calculate_baseline_revenue_loss(self, customer_data: pd.DataFrame,
                                      observed_churn_rate: float) -> float:
        """
        Calculate revenue loss without churn prediction (baseline)

        Args:
            customer_data: DataFrame with customer information
            observed_churn_rate: Historical churn rate

        Returns:
            Estimated annual revenue loss without intervention

        Educational Notes:
            - Baseline scenario: no churn prediction
            - Natural churn rate from historical data
        """
        pass

    @abstractmethod
    def calculate_model_revenue_impact(self, model_predictions: np.ndarray,
                                     true_labels: np.ndarray,
                                     customer_values: np.ndarray,
                                     intervention_success_rate: float = 0.3) -> Dict[str, float]:
        """
        Calculate revenue impact of using churn prediction model

        Args:
            model_predictions: Binary model predictions
            true_labels: True churn labels
            customer_values: CLV for each customer
            intervention_success_rate: Success rate of retention campaigns

        Returns:
            Dict with revenue impact analysis:
            - saved_revenue: Revenue saved through successful interventions
            - intervention_costs: Total cost of retention campaigns
            - net_benefit: Saved revenue minus intervention costs

        Educational Notes:
            - Not all identified churners can be retained
            - Intervention costs must be subtracted from benefits
        """
        pass

    @abstractmethod
    def generate_roi_report(self, baseline_loss: float, model_impact: Dict[str, float],
                          model_development_cost: float = 10000.0) -> Dict[str, Any]:
        """
        Generate comprehensive ROI report for churn prediction project

        Args:
            baseline_loss: Revenue loss without model
            model_impact: Revenue impact with model
            model_development_cost: Cost to develop and deploy model

        Returns:
            Dict with complete ROI analysis:
            - roi_percentage: Return on investment percentage
            - payback_period_months: Time to recover development costs
            - annual_net_benefit: Net annual benefit from model
            - break_even_metrics: Required performance for break-even

        Educational Notes:
            - ROI calculation includes development costs
            - Payback period for business case justification
        """
        pass


class ChurnInsightsContract(ABC):
    """Contract for generating business insights from churn analysis"""

    @abstractmethod
    def identify_churn_drivers(self, feature_importance: Dict[str, float],
                             customer_data: pd.DataFrame,
                             churn_labels: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Identify key drivers of customer churn for business action

        Args:
            feature_importance: Model feature importance scores
            customer_data: Customer features DataFrame
            churn_labels: Churn target variable

        Returns:
            Dict of insights:
            - top_risk_factors: Most important churn predictors
            - segment_risks: Churn rates by customer segment
            - actionable_factors: Controllable factors affecting churn

        Educational Notes:
            - Distinguish correlation from causation
            - Focus on actionable business levers
        """
        pass

    @abstractmethod
    def segment_customers_by_risk(self, churn_probabilities: np.ndarray,
                                customer_data: pd.DataFrame,
                                risk_thresholds: List[float] = [0.3, 0.7]) -> pd.DataFrame:
        """
        Segment customers into risk categories for targeted interventions

        Args:
            churn_probabilities: Model-predicted churn probabilities
            customer_data: Customer features DataFrame
            risk_thresholds: Probability thresholds for risk categories

        Returns:
            DataFrame with risk segments and recommended actions

        Educational Notes:
            - Different retention strategies for different risk levels
            - Resource allocation based on predicted risk
        """
        pass

    @abstractmethod
    def recommend_retention_strategies(self, customer_segments: pd.DataFrame,
                                     churn_drivers: Dict[str, float]) -> Dict[str, List[str]]:
        """
        Recommend retention strategies based on churn analysis

        Args:
            customer_segments: Customer risk segmentation results
            churn_drivers: Key factors driving churn

        Returns:
            Dict of segment: list of recommended retention strategies

        Educational Notes:
            - Personalized retention based on risk factors
            - Cost-effective intervention targeting
        """
        pass
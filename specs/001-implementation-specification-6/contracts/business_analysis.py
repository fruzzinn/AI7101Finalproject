"""
Contract: Business Impact Analysis Interface
Purpose: Quantify business value and generate actionable insights
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

class BusinessAnalysisContract:
    """Contract for business impact analysis operations"""

    def calculate_business_impact(self,
                                 predictions: np.ndarray,
                                 probabilities: np.ndarray,
                                 customer_values: Optional[pd.Series] = None) -> Dict[str, float]:
        """
        Quantify potential business impact of churn predictions

        Args:
            predictions: Binary churn predictions
            probabilities: Prediction probabilities
            customer_values: Customer revenue/value data (if available)

        Returns:
            Dict with business impact metrics

        Contract Requirements:
            - Calculate number of at-risk customers identified
            - Estimate revenue protection potential
            - Quantify cost reduction from targeted retention
            - Provide ROI analysis for retention campaigns
            - Include confidence intervals for estimates
        """
        raise NotImplementedError("Must implement business impact calculation")

    def generate_retention_strategies(self,
                                     feature_importance: Dict[str, float],
                                     high_risk_customers: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Create actionable retention recommendations based on model insights

        Args:
            feature_importance: Key factors driving churn
            high_risk_customers: Customers predicted to churn

        Returns:
            List of retention strategy dictionaries

        Contract Requirements:
            - Identify customer segments for targeted interventions
            - Recommend specific actions based on churn drivers
            - Prioritize strategies by expected impact and cost
            - Provide implementation timeline and resource requirements
        """
        raise NotImplementedError("Must implement retention strategy generation")

    def create_executive_summary(self,
                                model_performance: Dict[str, float],
                                business_impact: Dict[str, float],
                                key_insights: List[str]) -> Dict[str, Any]:
        """
        Generate executive summary suitable for non-technical stakeholders

        Args:
            model_performance: Technical model metrics
            business_impact: Business value calculations
            key_insights: Main findings from analysis

        Returns:
            Dict with executive summary components

        Contract Requirements:
            - Translate technical metrics to business language
            - Highlight key findings and recommendations
            - Provide clear next steps and implementation guidance
            - Include visual summary elements for presentation
        """
        raise NotImplementedError("Must implement executive summary creation")

    def assess_model_limitations(self,
                                performance_metrics: Dict[str, float],
                                data_quality_issues: List[str]) -> Dict[str, Any]:
        """
        Honest assessment of model limitations and uncertainty

        Args:
            performance_metrics: Model evaluation results
            data_quality_issues: Known data limitations

        Returns:
            Dict with limitation analysis

        Contract Requirements:
            - Identify scenarios where model may fail
            - Quantify prediction uncertainty
            - Document data quality constraints
            - Recommend model monitoring and retraining schedule
        """
        raise NotImplementedError("Must implement limitation assessment")

    def generate_monitoring_framework(self,
                                     baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Create framework for ongoing model performance monitoring

        Args:
            baseline_metrics: Initial model performance benchmarks

        Returns:
            Dict with monitoring framework specification

        Contract Requirements:
            - Define performance degradation thresholds
            - Specify data drift detection methods
            - Outline model retraining triggers and schedule
            - Establish business KPI tracking for model impact
        """
        raise NotImplementedError("Must implement monitoring framework")
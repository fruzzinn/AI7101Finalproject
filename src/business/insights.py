"""
Churn Insights Generator

This module generates actionable business insights from churn prediction models,
providing interpretable explanations and strategic recommendations for customer retention.
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class InsightType(Enum):
    """Types of business insights."""
    FEATURE_IMPORTANCE = "feature_importance"
    SEGMENT_ANALYSIS = "segment_analysis"
    TEMPORAL_PATTERNS = "temporal_patterns"
    PREDICTIVE_FACTORS = "predictive_factors"
    INTERVENTION_OPPORTUNITIES = "intervention_opportunities"
    RISK_ASSESSMENT = "risk_assessment"


class RiskLevel(Enum):
    """Customer risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChurnInsight:
    """Single actionable insight about churn patterns."""
    insight_type: InsightType
    title: str
    description: str
    evidence: Dict[str, Any]
    impact_score: float  # 0-1 scale
    confidence: float  # 0-1 scale
    actionable_recommendations: List[str]
    affected_customers: int
    potential_revenue_impact: float
    priority: str = field(default="medium")


@dataclass
class CustomerSegmentInsight:
    """Insights specific to a customer segment."""
    segment_name: str
    segment_size: int
    churn_rate: float
    risk_level: RiskLevel
    key_churn_drivers: List[Tuple[str, float]]  # (feature, importance)
    segment_characteristics: Dict[str, Any]
    recommended_interventions: List[str]
    expected_intervention_success_rate: float
    revenue_at_risk: float


@dataclass
class FeatureInsight:
    """Insights about feature importance and impact."""
    feature_name: str
    importance_score: float
    business_interpretation: str
    typical_range: Tuple[float, float]
    churn_correlation: float
    actionability: str  # "high", "medium", "low"
    intervention_strategies: List[str]


class ChurnInsightsGenerator:
    """
    Generates comprehensive business insights from churn prediction models
    and customer data to support strategic decision-making.
    """

    def __init__(self, model_explainer: Optional[Any] = None):
        """
        Initialize insights generator.

        Args:
            model_explainer: Optional model explainer (SHAP, LIME, etc.)
        """
        self.model_explainer = model_explainer
        self.feature_interpretations = self._initialize_feature_interpretations()
        self.insights_history: List[ChurnInsight] = []

    def _initialize_feature_interpretations(self) -> Dict[str, Dict[str, str]]:
        """Initialize business interpretations for common features."""
        return {
            'tenure_months': {
                'business_meaning': 'Customer relationship length',
                'actionability': 'low',
                'intervention_focus': 'Early lifecycle engagement'
            },
            'monthly_charges': {
                'business_meaning': 'Monthly revenue per customer',
                'actionability': 'high',
                'intervention_focus': 'Pricing and value perception'
            },
            'total_charges': {
                'business_meaning': 'Cumulative customer value',
                'actionability': 'medium',
                'intervention_focus': 'Loyalty programs'
            },
            'contract_type': {
                'business_meaning': 'Commitment level',
                'actionability': 'high',
                'intervention_focus': 'Contract optimization'
            },
            'payment_method': {
                'business_meaning': 'Payment convenience and automation',
                'actionability': 'high',
                'intervention_focus': 'Payment method migration'
            },
            'internet_service': {
                'business_meaning': 'Core service type',
                'actionability': 'medium',
                'intervention_focus': 'Service upgrade/downgrade'
            },
            'support_tickets': {
                'business_meaning': 'Service satisfaction indicator',
                'actionability': 'high',
                'intervention_focus': 'Proactive support'
            },
            'late_payments': {
                'business_meaning': 'Payment reliability',
                'actionability': 'high',
                'intervention_focus': 'Payment assistance programs'
            }
        }

    def generate_comprehensive_insights(self,
                                      model: Any,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      feature_importance: Optional[Dict[str, float]] = None,
                                      customer_data: Optional[pd.DataFrame] = None) -> List[ChurnInsight]:
        """
        Generate comprehensive business insights from model and data.

        Args:
            model: Trained churn prediction model
            X: Feature matrix
            y: Target variable
            feature_importance: Optional feature importance scores
            customer_data: Optional additional customer information

        Returns:
            List of actionable insights
        """
        insights = []

        try:
            # Feature importance insights
            if feature_importance or hasattr(model, 'feature_importances_'):
                insights.extend(self._generate_feature_insights(
                    feature_importance or dict(zip(X.columns, model.feature_importances_)),
                    X, y
                ))

            # Segment analysis insights
            if customer_data is not None:
                insights.extend(self._generate_segment_insights(X, y, customer_data))

            # Temporal pattern insights
            insights.extend(self._generate_temporal_insights(X, y))

            # Risk assessment insights
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
                insights.extend(self._generate_risk_insights(X, probabilities))

            # Intervention opportunity insights
            insights.extend(self._generate_intervention_insights(X, y, model))

        except Exception as e:
            logger.error(f"Error generating insights: {e}")

        # Sort by impact and confidence
        insights.sort(key=lambda x: x.impact_score * x.confidence, reverse=True)
        self.insights_history.extend(insights)

        return insights

    def _generate_feature_insights(self,
                                 feature_importance: Dict[str, float],
                                 X: pd.DataFrame,
                                 y: pd.Series) -> List[ChurnInsight]:
        """Generate insights about feature importance and business impact."""
        insights = []

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:5]

        for feature_name, importance in top_features:
            if feature_name not in X.columns:
                continue

            feature_values = X[feature_name]
            churn_correlation = feature_values.corr(y)

            # Calculate business impact
            high_values = feature_values > feature_values.quantile(0.75)
            low_values = feature_values < feature_values.quantile(0.25)
            high_churn_rate = y[high_values].mean() if high_values.sum() > 0 else 0
            low_churn_rate = y[low_values].mean() if low_values.sum() > 0 else 0

            # Get business interpretation
            interpretation = self.feature_interpretations.get(feature_name, {})
            business_meaning = interpretation.get('business_meaning', f'Feature: {feature_name}')
            actionability = interpretation.get('actionability', 'medium')

            # Generate recommendations
            recommendations = self._generate_feature_recommendations(
                feature_name, churn_correlation, actionability
            )

            # Calculate affected customers
            affected_customers = len(X) if abs(churn_correlation) > 0.1 else 0

            insight = ChurnInsight(
                insight_type=InsightType.FEATURE_IMPORTANCE,
                title=f"High Impact Factor: {business_meaning}",
                description=f"{feature_name} has {importance:.1%} importance in predicting churn. "
                           f"Correlation with churn: {churn_correlation:.3f}",
                evidence={
                    'importance_score': importance,
                    'correlation': churn_correlation,
                    'high_value_churn_rate': high_churn_rate,
                    'low_value_churn_rate': low_churn_rate,
                    'feature_range': (feature_values.min(), feature_values.max())
                },
                impact_score=importance,
                confidence=min(0.9, abs(churn_correlation) + 0.3),
                actionable_recommendations=recommendations,
                affected_customers=affected_customers,
                potential_revenue_impact=0.0,  # Calculate based on business context
                priority="high" if importance > 0.15 else "medium"
            )
            insights.append(insight)

        return insights

    def _generate_segment_insights(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 customer_data: pd.DataFrame) -> List[ChurnInsight]:
        """Generate insights about customer segments."""
        insights = []

        # Define key segmentation features
        segment_features = ['contract_type', 'payment_method', 'internet_service']
        available_features = [f for f in segment_features if f in customer_data.columns]

        for feature in available_features:
            segments = customer_data[feature].value_counts()

            for segment_value, count in segments.items():
                if count < 50:  # Skip small segments
                    continue

                # Get segment data
                segment_mask = customer_data[feature] == segment_value
                segment_churn_rate = y[segment_mask].mean()
                overall_churn_rate = y.mean()

                # Calculate significance
                if abs(segment_churn_rate - overall_churn_rate) < 0.05:
                    continue

                # Determine risk level
                if segment_churn_rate > overall_churn_rate * 1.5:
                    risk_level = RiskLevel.HIGH
                elif segment_churn_rate > overall_churn_rate * 1.2:
                    risk_level = RiskLevel.MEDIUM
                else:
                    risk_level = RiskLevel.LOW

                # Generate recommendations
                recommendations = self._generate_segment_recommendations(
                    feature, segment_value, segment_churn_rate, overall_churn_rate
                )

                insight = ChurnInsight(
                    insight_type=InsightType.SEGMENT_ANALYSIS,
                    title=f"High-Risk Segment: {feature} = {segment_value}",
                    description=f"Customers with {feature} '{segment_value}' have {segment_churn_rate:.1%} "
                               f"churn rate vs {overall_churn_rate:.1%} overall",
                    evidence={
                        'segment_size': count,
                        'segment_churn_rate': segment_churn_rate,
                        'overall_churn_rate': overall_churn_rate,
                        'relative_risk': segment_churn_rate / overall_churn_rate,
                        'risk_level': risk_level.value
                    },
                    impact_score=min(1.0, abs(segment_churn_rate - overall_churn_rate) * 2),
                    confidence=0.8 if count > 100 else 0.6,
                    actionable_recommendations=recommendations,
                    affected_customers=count,
                    potential_revenue_impact=0.0,
                    priority="high" if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL] else "medium"
                )
                insights.append(insight)

        return insights

    def _generate_temporal_insights(self,
                                  X: pd.DataFrame,
                                  y: pd.Series) -> List[ChurnInsight]:
        """Generate insights about temporal patterns."""
        insights = []

        # Look for tenure-related patterns
        if 'tenure_months' in X.columns:
            tenure = X['tenure_months']

            # Early churn risk
            early_customers = tenure <= 6
            if early_customers.sum() > 0:
                early_churn_rate = y[early_customers].mean()
                overall_churn_rate = y.mean()

                if early_churn_rate > overall_churn_rate * 1.3:
                    insight = ChurnInsight(
                        insight_type=InsightType.TEMPORAL_PATTERNS,
                        title="High Early Churn Risk",
                        description=f"Customers with ≤6 months tenure have {early_churn_rate:.1%} "
                                   f"churn rate vs {overall_churn_rate:.1%} overall",
                        evidence={
                            'early_customers': early_customers.sum(),
                            'early_churn_rate': early_churn_rate,
                            'relative_risk': early_churn_rate / overall_churn_rate
                        },
                        impact_score=0.8,
                        confidence=0.9,
                        actionable_recommendations=[
                            "Implement enhanced onboarding program",
                            "Provide early customer success check-ins",
                            "Offer first-time customer incentives",
                            "Monitor satisfaction closely in first 6 months"
                        ],
                        affected_customers=early_customers.sum(),
                        potential_revenue_impact=0.0,
                        priority="high"
                    )
                    insights.append(insight)

        return insights

    def _generate_risk_insights(self,
                              X: pd.DataFrame,
                              churn_probabilities: np.ndarray) -> List[ChurnInsight]:
        """Generate insights about customer risk levels."""
        insights = []

        # Define risk thresholds
        high_risk_threshold = 0.7
        medium_risk_threshold = 0.4

        high_risk_customers = churn_probabilities >= high_risk_threshold
        medium_risk_customers = (churn_probabilities >= medium_risk_threshold) & (churn_probabilities < high_risk_threshold)

        if high_risk_customers.sum() > 0:
            insight = ChurnInsight(
                insight_type=InsightType.RISK_ASSESSMENT,
                title="Critical Risk Customer Alert",
                description=f"{high_risk_customers.sum()} customers have ≥{high_risk_threshold:.0%} "
                           f"probability of churning",
                evidence={
                    'high_risk_count': high_risk_customers.sum(),
                    'average_risk_score': churn_probabilities[high_risk_customers].mean(),
                    'risk_threshold': high_risk_threshold
                },
                impact_score=0.9,
                confidence=0.85,
                actionable_recommendations=[
                    "Immediate outreach to high-risk customers",
                    "Offer retention incentives",
                    "Schedule customer success calls",
                    "Investigate specific pain points"
                ],
                affected_customers=high_risk_customers.sum(),
                potential_revenue_impact=0.0,
                priority="critical"
            )
            insights.append(insight)

        return insights

    def _generate_intervention_insights(self,
                                      X: pd.DataFrame,
                                      y: pd.Series,
                                      model: Any) -> List[ChurnInsight]:
        """Generate insights about intervention opportunities."""
        insights = []

        # Identify actionable features
        actionable_features = [
            'monthly_charges', 'contract_type', 'payment_method',
            'support_tickets', 'paperless_billing'
        ]

        available_actionable = [f for f in actionable_features if f in X.columns]

        for feature in available_actionable:
            # Analyze feature impact on churn
            feature_values = X[feature]
            correlation = feature_values.corr(y)

            if abs(correlation) > 0.1:  # Significant correlation
                recommendations = self._generate_intervention_recommendations(feature, correlation)

                insight = ChurnInsight(
                    insight_type=InsightType.INTERVENTION_OPPORTUNITIES,
                    title=f"Intervention Opportunity: {feature}",
                    description=f"Optimizing {feature} could significantly impact churn rates",
                    evidence={
                        'correlation': correlation,
                        'feature_importance': abs(correlation),
                        'potential_impact': 'high' if abs(correlation) > 0.2 else 'medium'
                    },
                    impact_score=abs(correlation),
                    confidence=0.7,
                    actionable_recommendations=recommendations,
                    affected_customers=len(X),
                    potential_revenue_impact=0.0,
                    priority="high" if abs(correlation) > 0.2 else "medium"
                )
                insights.append(insight)

        return insights

    def _generate_feature_recommendations(self,
                                        feature_name: str,
                                        correlation: float,
                                        actionability: str) -> List[str]:
        """Generate recommendations for specific features."""
        recommendations = []

        feature_strategies = {
            'monthly_charges': [
                "Review pricing strategy for high-churn segments",
                "Implement value-based pricing",
                "Offer targeted discounts to at-risk customers",
                "Create tiered service offerings"
            ],
            'contract_type': [
                "Incentivize longer-term contracts",
                "Offer contract conversion bonuses",
                "Improve month-to-month customer experience",
                "Create hybrid contract options"
            ],
            'payment_method': [
                "Migrate customers to automatic payments",
                "Offer payment method optimization consultations",
                "Provide payment convenience incentives",
                "Implement payment reminder systems"
            ],
            'support_tickets': [
                "Implement proactive customer support",
                "Improve first-call resolution rates",
                "Create self-service options",
                "Monitor customer satisfaction post-support"
            ]
        }

        if feature_name in feature_strategies:
            recommendations = feature_strategies[feature_name]
        else:
            recommendations = [
                f"Monitor {feature_name} as a churn indicator",
                f"Investigate {feature_name} impact on customer satisfaction",
                f"Consider interventions targeting {feature_name}"
            ]

        return recommendations

    def _generate_segment_recommendations(self,
                                        feature: str,
                                        segment_value: str,
                                        segment_churn_rate: float,
                                        overall_churn_rate: float) -> List[str]:
        """Generate recommendations for specific customer segments."""
        recommendations = []

        if segment_churn_rate > overall_churn_rate:
            recommendations.extend([
                f"Develop targeted retention program for {feature} = {segment_value}",
                f"Investigate why {segment_value} customers churn more",
                f"Create specialized customer success approach",
                f"Consider {segment_value}-specific pricing or incentives"
            ])
        else:
            recommendations.extend([
                f"Study success factors of {segment_value} customers",
                f"Apply {segment_value} best practices to other segments",
                f"Use {segment_value} customers as retention model"
            ])

        return recommendations

    def _generate_intervention_recommendations(self,
                                             feature: str,
                                             correlation: float) -> List[str]:
        """Generate intervention recommendations for actionable features."""
        recommendations = []

        if correlation > 0:  # Positive correlation with churn
            recommendations.append(f"Reduce {feature} values where possible")
            recommendations.append(f"Implement {feature} optimization program")
        else:  # Negative correlation with churn
            recommendations.append(f"Increase {feature} values for at-risk customers")
            recommendations.append(f"Promote {feature} as retention strategy")

        recommendations.extend([
            f"Monitor {feature} changes and churn impact",
            f"A/B test {feature} interventions",
            f"Create {feature}-based customer segments"
        ])

        return recommendations

    def generate_customer_risk_profile(self,
                                     customer_id: str,
                                     customer_features: pd.Series,
                                     model: Any,
                                     feature_importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate detailed risk profile for individual customer.

        Args:
            customer_id: Unique customer identifier
            customer_features: Customer's feature values
            model: Trained churn prediction model
            feature_importance: Feature importance scores

        Returns:
            Comprehensive customer risk profile
        """
        # Predict churn probability
        features_array = customer_features.values.reshape(1, -1)
        churn_probability = model.predict_proba(features_array)[0, 1]

        # Determine risk level
        if churn_probability >= 0.8:
            risk_level = RiskLevel.CRITICAL
        elif churn_probability >= 0.6:
            risk_level = RiskLevel.HIGH
        elif churn_probability >= 0.3:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Identify top risk factors
        risk_factors = []
        for feature, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            if feature in customer_features.index:
                risk_factors.append({
                    'feature': feature,
                    'value': customer_features[feature],
                    'importance': importance,
                    'interpretation': self.feature_interpretations.get(feature, {}).get('business_meaning', feature)
                })

        # Generate personalized recommendations
        recommendations = self._generate_personalized_recommendations(
            customer_features, risk_factors, risk_level
        )

        return {
            'customer_id': customer_id,
            'churn_probability': churn_probability,
            'risk_level': risk_level.value,
            'top_risk_factors': risk_factors,
            'personalized_recommendations': recommendations,
            'priority_score': churn_probability * sum(rf['importance'] for rf in risk_factors[:3]),
            'intervention_urgency': 'immediate' if risk_level == RiskLevel.CRITICAL else
                                   'within_week' if risk_level == RiskLevel.HIGH else
                                   'within_month' if risk_level == RiskLevel.MEDIUM else 'monitor',
            'estimated_lifetime_value': 0.0,  # Calculate based on business context
            'retention_cost_threshold': 0.0   # Calculate based on CLV
        }

    def _generate_personalized_recommendations(self,
                                             customer_features: pd.Series,
                                             risk_factors: List[Dict],
                                             risk_level: RiskLevel) -> List[str]:
        """Generate personalized recommendations for individual customer."""
        recommendations = []

        # High-level recommendations based on risk level
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "Schedule immediate customer success call",
                "Offer significant retention incentive",
                "Escalate to senior customer success manager"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Proactive outreach within 48 hours",
                "Offer personalized retention offer",
                "Schedule customer satisfaction survey"
            ])

        # Feature-specific recommendations
        for risk_factor in risk_factors[:3]:  # Top 3 risk factors
            feature = risk_factor['feature']
            if feature in self.feature_interpretations:
                interpretation = self.feature_interpretations[feature]
                if interpretation.get('actionability') == 'high':
                    recommendations.append(f"Address {interpretation['intervention_focus']}")

        return recommendations

    def export_insights_report(self,
                             insights: List[ChurnInsight],
                             output_path: Path,
                             format_type: str = 'markdown') -> str:
        """
        Export insights as formatted report.

        Args:
            insights: List of insights to export
            output_path: Path for output file
            format_type: Format type ('markdown', 'json', 'html')

        Returns:
            Report content as string
        """
        if format_type == 'markdown':
            return self._export_markdown_report(insights, output_path)
        elif format_type == 'json':
            return self._export_json_report(insights, output_path)
        elif format_type == 'html':
            return self._export_html_report(insights, output_path)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")

    def _export_markdown_report(self, insights: List[ChurnInsight], output_path: Path) -> str:
        """Export insights as markdown report."""
        report_lines = [
            "# Churn Prediction Insights Report",
            "",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Insights: {len(insights)}",
            "",
            "## Executive Summary",
            ""
        ]

        # Summary statistics
        critical_insights = [i for i in insights if i.priority == 'critical']
        high_priority = [i for i in insights if i.priority == 'high']
        total_affected = sum(i.affected_customers for i in insights)

        report_lines.extend([
            f"- **Critical Issues**: {len(critical_insights)}",
            f"- **High Priority Items**: {len(high_priority)}",
            f"- **Total Customers Affected**: {total_affected:,}",
            "",
            "## Detailed Insights",
            ""
        ])

        # Group insights by type
        insights_by_type = {}
        for insight in insights:
            insight_type = insight.insight_type.value
            if insight_type not in insights_by_type:
                insights_by_type[insight_type] = []
            insights_by_type[insight_type].append(insight)

        for insight_type, type_insights in insights_by_type.items():
            report_lines.extend([
                f"### {insight_type.replace('_', ' ').title()}",
                ""
            ])

            for insight in type_insights:
                report_lines.extend([
                    f"#### {insight.title}",
                    f"**Priority**: {insight.priority.title()}",
                    f"**Impact Score**: {insight.impact_score:.2f}",
                    f"**Confidence**: {insight.confidence:.2f}",
                    f"**Affected Customers**: {insight.affected_customers:,}",
                    "",
                    insight.description,
                    "",
                    "**Recommendations**:",
                    ""
                ])

                for i, rec in enumerate(insight.actionable_recommendations, 1):
                    report_lines.append(f"{i}. {rec}")

                report_lines.extend(["", "---", ""])

        report_content = "\n".join(report_lines)

        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"Markdown report exported to {output_path}")
        return report_content

    def _export_json_report(self, insights: List[ChurnInsight], output_path: Path) -> str:
        """Export insights as JSON report."""
        insights_data = []

        for insight in insights:
            insight_data = {
                'insight_type': insight.insight_type.value,
                'title': insight.title,
                'description': insight.description,
                'evidence': insight.evidence,
                'impact_score': insight.impact_score,
                'confidence': insight.confidence,
                'actionable_recommendations': insight.actionable_recommendations,
                'affected_customers': insight.affected_customers,
                'potential_revenue_impact': insight.potential_revenue_impact,
                'priority': insight.priority
            }
            insights_data.append(insight_data)

        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_insights': len(insights),
                'insights_by_priority': {
                    'critical': len([i for i in insights if i.priority == 'critical']),
                    'high': len([i for i in insights if i.priority == 'high']),
                    'medium': len([i for i in insights if i.priority == 'medium']),
                    'low': len([i for i in insights if i.priority == 'low'])
                }
            },
            'insights': insights_data
        }

        report_content = json.dumps(report_data, indent=2)

        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"JSON report exported to {output_path}")
        return report_content

    def _export_html_report(self, insights: List[ChurnInsight], output_path: Path) -> str:
        """Export insights as HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Churn Prediction Insights Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .insight { border: 1px solid #ddd; margin: 20px 0; padding: 20px; border-radius: 5px; }
                .critical { border-left: 5px solid #ff4444; }
                .high { border-left: 5px solid #ff8800; }
                .medium { border-left: 5px solid #ffaa00; }
                .low { border-left: 5px solid #00aa00; }
                .metric { display: inline-block; margin: 5px 15px 5px 0; }
                .recommendations { margin-top: 15px; }
                .recommendations ul { margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>Churn Prediction Insights Report</h1>
            <p>Generated on: {timestamp}</p>
            <p>Total Insights: {total_insights}</p>

            {insights_html}
        </body>
        </html>
        """

        insights_html = ""
        for insight in insights:
            insight_html = f"""
            <div class="insight {insight.priority}">
                <h3>{insight.title}</h3>
                <div class="metrics">
                    <span class="metric"><strong>Priority:</strong> {insight.priority.title()}</span>
                    <span class="metric"><strong>Impact:</strong> {insight.impact_score:.2f}</span>
                    <span class="metric"><strong>Confidence:</strong> {insight.confidence:.2f}</span>
                    <span class="metric"><strong>Affected Customers:</strong> {insight.affected_customers:,}</span>
                </div>
                <p>{insight.description}</p>
                <div class="recommendations">
                    <strong>Recommendations:</strong>
                    <ul>
                        {''.join(f'<li>{rec}</li>' for rec in insight.actionable_recommendations)}
                    </ul>
                </div>
            </div>
            """
            insights_html += insight_html

        report_content = html_template.format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_insights=len(insights),
            insights_html=insights_html
        )

        with open(output_path, 'w') as f:
            f.write(report_content)

        logger.info(f"HTML report exported to {output_path}")
        return report_content
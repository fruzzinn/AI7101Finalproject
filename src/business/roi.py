"""
ROI Calculator for Churn Prevention Interventions

This module calculates return on investment for various churn prevention strategies,
providing financial justification for ML-driven customer retention programs.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of churn prevention interventions."""
    DISCOUNT = "discount"
    LOYALTY_PROGRAM = "loyalty_program"
    PERSONAL_OUTREACH = "personal_outreach"
    PRODUCT_UPGRADE = "product_upgrade"
    SUPPORT_ENHANCEMENT = "support_enhancement"
    RETENTION_BONUS = "retention_bonus"


@dataclass
class InterventionCost:
    """Cost structure for a specific intervention."""
    fixed_cost: float
    variable_cost_per_customer: float
    implementation_cost: float
    ongoing_monthly_cost: float
    success_rate: float
    duration_months: int


@dataclass
class ROIScenario:
    """ROI calculation scenario with assumptions."""
    scenario_name: str
    customer_segment: str
    intervention_type: InterventionType
    baseline_churn_rate: float
    intervention_churn_rate: float
    customers_targeted: int
    average_monthly_revenue: float
    average_customer_lifetime_months: int
    discount_rate: float
    roi_percentage: float
    net_present_value: float
    payback_period_months: float
    confidence_interval: Tuple[float, float]


class ROICalculator:
    """
    Calculates ROI for churn prevention interventions with comprehensive
    financial modeling and scenario analysis.
    """

    def __init__(self, discount_rate: float = 0.12):
        """
        Initialize ROI calculator.

        Args:
            discount_rate: Annual discount rate for NPV calculations
        """
        self.discount_rate = discount_rate
        self.intervention_costs = self._initialize_intervention_costs()

    def _initialize_intervention_costs(self) -> Dict[InterventionType, InterventionCost]:
        """Initialize default cost structures for different interventions."""
        return {
            InterventionType.DISCOUNT: InterventionCost(
                fixed_cost=5000,
                variable_cost_per_customer=25,
                implementation_cost=10000,
                ongoing_monthly_cost=2000,
                success_rate=0.65,
                duration_months=6
            ),
            InterventionType.LOYALTY_PROGRAM: InterventionCost(
                fixed_cost=15000,
                variable_cost_per_customer=10,
                implementation_cost=25000,
                ongoing_monthly_cost=5000,
                success_rate=0.45,
                duration_months=12
            ),
            InterventionType.PERSONAL_OUTREACH: InterventionCost(
                fixed_cost=8000,
                variable_cost_per_customer=50,
                implementation_cost=12000,
                ongoing_monthly_cost=3000,
                success_rate=0.75,
                duration_months=3
            ),
            InterventionType.PRODUCT_UPGRADE: InterventionCost(
                fixed_cost=20000,
                variable_cost_per_customer=100,
                implementation_cost=30000,
                ongoing_monthly_cost=8000,
                success_rate=0.55,
                duration_months=24
            ),
            InterventionType.SUPPORT_ENHANCEMENT: InterventionCost(
                fixed_cost=12000,
                variable_cost_per_customer=15,
                implementation_cost=18000,
                ongoing_monthly_cost=4000,
                success_rate=0.40,
                duration_months=12
            ),
            InterventionType.RETENTION_BONUS: InterventionCost(
                fixed_cost=3000,
                variable_cost_per_customer=75,
                implementation_cost=5000,
                ongoing_monthly_cost=1000,
                success_rate=0.70,
                duration_months=1
            )
        }

    def calculate_customer_lifetime_value(self,
                                        monthly_revenue: float,
                                        churn_rate: float,
                                        discount_rate: Optional[float] = None) -> float:
        """
        Calculate customer lifetime value based on retention.

        Args:
            monthly_revenue: Average monthly revenue per customer
            churn_rate: Monthly churn rate
            discount_rate: Monthly discount rate (defaults to annual/12)

        Returns:
            Customer lifetime value
        """
        if discount_rate is None:
            discount_rate = self.discount_rate / 12

        if churn_rate >= 1.0:
            return monthly_revenue

        # CLV = sum of discounted future revenues
        # Using geometric series formula for infinite horizon
        retention_rate = 1 - churn_rate
        effective_discount = discount_rate + churn_rate

        if effective_discount >= 1.0:
            return monthly_revenue

        clv = monthly_revenue / effective_discount
        return clv

    def calculate_intervention_cost(self,
                                  intervention_type: InterventionType,
                                  customers_targeted: int,
                                  duration_months: Optional[int] = None) -> float:
        """
        Calculate total cost of intervention program.

        Args:
            intervention_type: Type of intervention
            customers_targeted: Number of customers to target
            duration_months: Override default duration

        Returns:
            Total intervention cost
        """
        cost_structure = self.intervention_costs[intervention_type]

        duration = duration_months or cost_structure.duration_months

        total_cost = (
            cost_structure.fixed_cost +
            cost_structure.implementation_cost +
            (cost_structure.variable_cost_per_customer * customers_targeted) +
            (cost_structure.ongoing_monthly_cost * duration)
        )

        return total_cost

    def calculate_roi_simple(self,
                           intervention_type: InterventionType,
                           customers_targeted: int,
                           baseline_churn_rate: float,
                           monthly_revenue_per_customer: float,
                           time_horizon_months: int = 24) -> Dict[str, float]:
        """
        Calculate simple ROI for intervention.

        Args:
            intervention_type: Type of intervention
            customers_targeted: Number of customers targeted
            baseline_churn_rate: Baseline monthly churn rate
            monthly_revenue_per_customer: Average monthly revenue
            time_horizon_months: Analysis time horizon

        Returns:
            Dictionary with ROI metrics
        """
        cost_structure = self.intervention_costs[intervention_type]

        # Calculate intervention effect
        intervention_churn_rate = baseline_churn_rate * (1 - cost_structure.success_rate)

        # Calculate costs
        intervention_cost = self.calculate_intervention_cost(
            intervention_type, customers_targeted
        )

        # Calculate benefits (retained revenue)
        baseline_clv = self.calculate_customer_lifetime_value(
            monthly_revenue_per_customer, baseline_churn_rate
        )
        intervention_clv = self.calculate_customer_lifetime_value(
            monthly_revenue_per_customer, intervention_churn_rate
        )

        # Revenue lift per customer
        revenue_lift_per_customer = intervention_clv - baseline_clv
        total_revenue_lift = revenue_lift_per_customer * customers_targeted

        # ROI calculation
        roi_percentage = ((total_revenue_lift - intervention_cost) / intervention_cost) * 100

        return {
            'roi_percentage': roi_percentage,
            'total_cost': intervention_cost,
            'total_benefit': total_revenue_lift,
            'net_benefit': total_revenue_lift - intervention_cost,
            'benefit_cost_ratio': total_revenue_lift / intervention_cost if intervention_cost > 0 else 0,
            'cost_per_customer': intervention_cost / customers_targeted,
            'benefit_per_customer': revenue_lift_per_customer,
            'baseline_clv': baseline_clv,
            'intervention_clv': intervention_clv,
            'churn_reduction': baseline_churn_rate - intervention_churn_rate
        }

    def calculate_roi_with_uncertainty(self,
                                     intervention_type: InterventionType,
                                     customers_targeted: int,
                                     baseline_churn_rate: float,
                                     monthly_revenue_per_customer: float,
                                     uncertainty_params: Dict[str, Tuple[float, float]],
                                     n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Calculate ROI with Monte Carlo simulation for uncertainty.

        Args:
            intervention_type: Type of intervention
            customers_targeted: Number of customers targeted
            baseline_churn_rate: Baseline monthly churn rate
            monthly_revenue_per_customer: Average monthly revenue
            uncertainty_params: Parameter ranges for simulation
            n_simulations: Number of Monte Carlo simulations

        Returns:
            ROI statistics with confidence intervals
        """
        roi_results = []

        for _ in range(n_simulations):
            # Sample uncertain parameters
            sampled_params = {}
            for param, (min_val, max_val) in uncertainty_params.items():
                sampled_params[param] = np.random.uniform(min_val, max_val)

            # Use sampled parameters
            sim_churn_rate = sampled_params.get('churn_rate', baseline_churn_rate)
            sim_revenue = sampled_params.get('monthly_revenue', monthly_revenue_per_customer)
            sim_success_rate = sampled_params.get('success_rate',
                                                 self.intervention_costs[intervention_type].success_rate)

            # Temporarily modify success rate
            original_success_rate = self.intervention_costs[intervention_type].success_rate
            self.intervention_costs[intervention_type].success_rate = sim_success_rate

            # Calculate ROI for this simulation
            roi_metrics = self.calculate_roi_simple(
                intervention_type, customers_targeted, sim_churn_rate, sim_revenue
            )
            roi_results.append(roi_metrics['roi_percentage'])

            # Restore original success rate
            self.intervention_costs[intervention_type].success_rate = original_success_rate

        roi_array = np.array(roi_results)

        return {
            'mean_roi': np.mean(roi_array),
            'median_roi': np.median(roi_array),
            'std_roi': np.std(roi_array),
            'confidence_interval_95': (np.percentile(roi_array, 2.5), np.percentile(roi_array, 97.5)),
            'confidence_interval_90': (np.percentile(roi_array, 5), np.percentile(roi_array, 95)),
            'probability_positive_roi': np.mean(roi_array > 0),
            'probability_roi_above_threshold': lambda threshold: np.mean(roi_array > threshold),
            'value_at_risk_5': np.percentile(roi_array, 5),
            'roi_distribution': roi_array
        }

    def calculate_payback_period(self,
                               intervention_type: InterventionType,
                               customers_targeted: int,
                               baseline_churn_rate: float,
                               monthly_revenue_per_customer: float) -> Dict[str, float]:
        """
        Calculate payback period for intervention investment.

        Args:
            intervention_type: Type of intervention
            customers_targeted: Number of customers targeted
            baseline_churn_rate: Baseline monthly churn rate
            monthly_revenue_per_customer: Average monthly revenue

        Returns:
            Payback analysis metrics
        """
        cost_structure = self.intervention_costs[intervention_type]
        intervention_churn_rate = baseline_churn_rate * (1 - cost_structure.success_rate)

        # Monthly revenue lift
        baseline_monthly_loss = customers_targeted * baseline_churn_rate * monthly_revenue_per_customer
        intervention_monthly_loss = customers_targeted * intervention_churn_rate * monthly_revenue_per_customer
        monthly_revenue_saved = baseline_monthly_loss - intervention_monthly_loss

        # Total investment
        total_investment = self.calculate_intervention_cost(intervention_type, customers_targeted)

        # Payback period
        if monthly_revenue_saved <= 0:
            payback_months = float('inf')
        else:
            payback_months = total_investment / monthly_revenue_saved

        return {
            'payback_period_months': payback_months,
            'monthly_revenue_saved': monthly_revenue_saved,
            'total_investment': total_investment,
            'break_even_customers': total_investment / (monthly_revenue_per_customer * (baseline_churn_rate - intervention_churn_rate)) if (baseline_churn_rate - intervention_churn_rate) > 0 else float('inf')
        }

    def compare_interventions(self,
                            customers_targeted: int,
                            baseline_churn_rate: float,
                            monthly_revenue_per_customer: float,
                            budget_constraint: Optional[float] = None) -> pd.DataFrame:
        """
        Compare ROI across all intervention types.

        Args:
            customers_targeted: Number of customers to target
            baseline_churn_rate: Baseline monthly churn rate
            monthly_revenue_per_customer: Average monthly revenue
            budget_constraint: Maximum budget available

        Returns:
            DataFrame comparing interventions
        """
        comparison_data = []

        for intervention_type in InterventionType:
            try:
                # Calculate cost
                cost = self.calculate_intervention_cost(intervention_type, customers_targeted)

                # Skip if over budget
                if budget_constraint and cost > budget_constraint:
                    continue

                # Calculate ROI
                roi_metrics = self.calculate_roi_simple(
                    intervention_type, customers_targeted,
                    baseline_churn_rate, monthly_revenue_per_customer
                )

                # Calculate payback
                payback_metrics = self.calculate_payback_period(
                    intervention_type, customers_targeted,
                    baseline_churn_rate, monthly_revenue_per_customer
                )

                comparison_data.append({
                    'intervention_type': intervention_type.value,
                    'total_cost': cost,
                    'roi_percentage': roi_metrics['roi_percentage'],
                    'net_benefit': roi_metrics['net_benefit'],
                    'benefit_cost_ratio': roi_metrics['benefit_cost_ratio'],
                    'payback_months': payback_metrics['payback_period_months'],
                    'monthly_revenue_saved': payback_metrics['monthly_revenue_saved'],
                    'success_rate': self.intervention_costs[intervention_type].success_rate,
                    'churn_reduction': roi_metrics['churn_reduction']
                })

            except Exception as e:
                logger.error(f"Error calculating ROI for {intervention_type.value}: {e}")
                continue

        df = pd.DataFrame(comparison_data)
        if not df.empty:
            df = df.sort_values('roi_percentage', ascending=False)

        return df

    def optimize_intervention_mix(self,
                                customers_by_segment: Dict[str, int],
                                churn_rates_by_segment: Dict[str, float],
                                revenue_by_segment: Dict[str, float],
                                total_budget: float) -> Dict[str, Any]:
        """
        Optimize mix of interventions across customer segments.

        Args:
            customers_by_segment: Customer counts by segment
            churn_rates_by_segment: Churn rates by segment
            revenue_by_segment: Monthly revenue by segment
            total_budget: Total available budget

        Returns:
            Optimal intervention allocation
        """
        from scipy.optimize import linprog

        # Generate all possible intervention-segment combinations
        options = []
        costs = []
        benefits = []

        for segment, customers in customers_by_segment.items():
            churn_rate = churn_rates_by_segment[segment]
            revenue = revenue_by_segment[segment]

            for intervention_type in InterventionType:
                # Calculate metrics for this combination
                cost = self.calculate_intervention_cost(intervention_type, customers)
                roi_metrics = self.calculate_roi_simple(
                    intervention_type, customers, churn_rate, revenue
                )

                options.append({
                    'segment': segment,
                    'intervention': intervention_type.value,
                    'customers': customers,
                    'cost': cost,
                    'benefit': roi_metrics['total_benefit']
                })
                costs.append(cost)
                benefits.append(roi_metrics['total_benefit'])

        # Set up linear programming problem (maximize benefit subject to budget)
        c = [-b for b in benefits]  # Negative because linprog minimizes
        A_ub = [costs]  # Budget constraint
        b_ub = [total_budget]
        bounds = [(0, 1) for _ in range(len(options))]  # Binary variables

        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

            if result.success:
                # Extract optimal solution
                optimal_allocation = []
                total_cost = 0
                total_benefit = 0

                for i, x in enumerate(result.x):
                    if x > 0.5:  # Binary threshold
                        option = options[i]
                        optimal_allocation.append(option)
                        total_cost += option['cost']
                        total_benefit += option['benefit']

                return {
                    'optimal_allocation': optimal_allocation,
                    'total_cost': total_cost,
                    'total_benefit': total_benefit,
                    'roi_percentage': ((total_benefit - total_cost) / total_cost) * 100 if total_cost > 0 else 0,
                    'budget_utilization': total_cost / total_budget,
                    'optimization_successful': True
                }
            else:
                logger.warning("Optimization failed, using greedy approach")

        except ImportError:
            logger.warning("SciPy not available, using greedy approach")

        # Fallback: greedy algorithm
        return self._greedy_optimization(options, total_budget)

    def _greedy_optimization(self, options: List[Dict], budget: float) -> Dict[str, Any]:
        """Greedy fallback for intervention optimization."""
        # Sort by benefit/cost ratio
        sorted_options = sorted(options,
                              key=lambda x: x['benefit'] / x['cost'] if x['cost'] > 0 else 0,
                              reverse=True)

        selected = []
        remaining_budget = budget
        total_benefit = 0

        for option in sorted_options:
            if option['cost'] <= remaining_budget:
                selected.append(option)
                remaining_budget -= option['cost']
                total_benefit += option['benefit']

        total_cost = budget - remaining_budget

        return {
            'optimal_allocation': selected,
            'total_cost': total_cost,
            'total_benefit': total_benefit,
            'roi_percentage': ((total_benefit - total_cost) / total_cost) * 100 if total_cost > 0 else 0,
            'budget_utilization': total_cost / budget,
            'optimization_successful': True
        }

    def generate_roi_report(self,
                          scenarios: List[ROIScenario],
                          output_path: Optional[Path] = None) -> str:
        """
        Generate comprehensive ROI analysis report.

        Args:
            scenarios: List of ROI scenarios to analyze
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        report_lines = [
            "# ROI Analysis Report for Churn Prevention Interventions",
            "",
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Discount Rate: {self.discount_rate:.1%}",
            "",
            "## Executive Summary",
            ""
        ]

        if scenarios:
            best_roi = max(scenarios, key=lambda x: x.roi_percentage)
            total_customers = sum(s.customers_targeted for s in scenarios)
            avg_roi = np.mean([s.roi_percentage for s in scenarios])

            report_lines.extend([
                f"- **Best ROI Scenario**: {best_roi.scenario_name} ({best_roi.roi_percentage:.1f}%)",
                f"- **Average ROI**: {avg_roi:.1f}%",
                f"- **Total Customers Analyzed**: {total_customers:,}",
                f"- **Total Investment**: ${sum(s.net_present_value + s.roi_percentage/100 * s.net_present_value for s in scenarios):,.0f}",
                ""
            ])

        # Detailed scenario analysis
        report_lines.extend([
            "## Scenario Analysis",
            ""
        ])

        for scenario in scenarios:
            report_lines.extend([
                f"### {scenario.scenario_name}",
                f"- **Customer Segment**: {scenario.customer_segment}",
                f"- **Intervention Type**: {scenario.intervention_type.value}",
                f"- **Customers Targeted**: {scenario.customers_targeted:,}",
                f"- **ROI**: {scenario.roi_percentage:.1f}%",
                f"- **NPV**: ${scenario.net_present_value:,.0f}",
                f"- **Payback Period**: {scenario.payback_period_months:.1f} months",
                f"- **Churn Reduction**: {(scenario.baseline_churn_rate - scenario.intervention_churn_rate)*100:.1f} percentage points",
                f"- **Confidence Interval**: ({scenario.confidence_interval[0]:.1f}%, {scenario.confidence_interval[1]:.1f}%)",
                ""
            ])

        # Recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])

        if scenarios:
            positive_roi_scenarios = [s for s in scenarios if s.roi_percentage > 0]
            if positive_roi_scenarios:
                report_lines.extend([
                    "### Recommended Interventions:",
                    ""
                ])
                for scenario in sorted(positive_roi_scenarios, key=lambda x: x.roi_percentage, reverse=True)[:3]:
                    report_lines.append(f"1. **{scenario.scenario_name}**: {scenario.roi_percentage:.1f}% ROI, {scenario.payback_period_months:.1f} month payback")
            else:
                report_lines.append("- No scenarios show positive ROI under current assumptions")
                report_lines.append("- Consider revising intervention strategies or targeting criteria")

        report_content = "\n".join(report_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"ROI report saved to {output_path}")

        return report_content

    def update_intervention_costs(self,
                                intervention_type: InterventionType,
                                cost_updates: Dict[str, float]) -> None:
        """
        Update cost parameters for specific intervention type.

        Args:
            intervention_type: Intervention to update
            cost_updates: Dictionary of cost parameter updates
        """
        cost_structure = self.intervention_costs[intervention_type]

        for param, value in cost_updates.items():
            if hasattr(cost_structure, param):
                setattr(cost_structure, param, value)
                logger.info(f"Updated {intervention_type.value} {param} to {value}")
            else:
                logger.warning(f"Unknown cost parameter: {param}")
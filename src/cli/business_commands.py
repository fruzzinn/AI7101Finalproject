"""
CLI Commands for Business Analysis in Churn Prediction ML Pipeline
Task T049: Command-line interface for business operations

This module provides CLI commands for:
- Customer lifetime value (CLV) analysis
- ROI calculations and scenario planning
- Churn risk assessment and intervention recommendations
- Business impact reporting and dashboard generation
- Campaign optimization and A/B testing
- Financial modeling and forecasting
"""

import click
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Import project modules
from src.config.settings import get_config, Environment, initialize_config
from src.utils.logging import setup_logging, get_logger, LogContext
from src.utils.model_utils import ModelRegistry, ModelMonitor
from src.business.analyzer import BusinessAnalyzer
from src.business.roi import ROICalculator
from src.business.insights import ChurnInsights


@click.group()
@click.option('--config', '-c', help='Configuration file path')
@click.option('--environment', '-e',
              type=click.Choice(['development', 'testing', 'staging', 'production']),
              default='development', help='Environment configuration')
@click.option('--log-level', '-l',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def business(ctx, config, environment, log_level, verbose):
    """Business analysis commands for churn prediction pipeline."""

    # Ensure context object exists
    ctx.ensure_object(dict)

    # Initialize configuration
    ctx.obj['config'] = initialize_config(environment, config)

    # Setup logging
    setup_logging(log_level=log_level)
    ctx.obj['logger'] = get_logger('churn_prediction.cli.business')

    # Initialize business utilities
    config_obj = ctx.obj['config']
    ctx.obj['model_registry'] = ModelRegistry(config_obj.data.model_artifacts_path)
    ctx.obj['business_analyzer'] = BusinessAnalyzer()
    ctx.obj['roi_calculator'] = ROICalculator()

    # Set verbose mode
    ctx.obj['verbose'] = verbose

    if verbose:
        click.echo(f"🔧 Environment: {environment}")
        click.echo(f"📋 Config: {config or 'default'}")
        click.echo(f"📊 Log Level: {log_level}")


@business.command()
@click.option('--customer-data', '-d', required=True, help='Customer data file path')
@click.option('--predictions', '-p', help='Churn predictions file path')
@click.option('--model-name', '-m', help='Model name for generating predictions')
@click.option('--output-path', '-o', help='Output path for CLV analysis')
@click.option('--time-horizon', '-t', type=int, default=24, help='Analysis time horizon in months')
@click.option('--discount-rate', type=float, default=0.08, help='Annual discount rate for NPV calculations')
@click.pass_context
def analyze_clv(ctx, customer_data, predictions, model_name, output_path, time_horizon, discount_rate):
    """Analyze Customer Lifetime Value (CLV) with churn predictions."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    business_analyzer = ctx.obj['business_analyzer']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"💰 Starting CLV analysis for {time_horizon} months horizon")

        # Load customer data
        if not Path(customer_data).exists():
            click.echo(f"❌ Error: Customer data file {customer_data} not found", err=True)
            sys.exit(1)

        customer_df = pd.read_csv(customer_data)
        click.echo(f"👥 Loaded {len(customer_df)} customer records")

        # Load or generate churn predictions
        if predictions:
            if not Path(predictions).exists():
                click.echo(f"❌ Error: Predictions file {predictions} not found", err=True)
                sys.exit(1)
            predictions_df = pd.read_csv(predictions)
            churn_probabilities = predictions_df.get('churn_probability', predictions_df.iloc[:, -1])
        elif model_name:
            # Generate predictions using specified model
            model_registry = ctx.obj['model_registry']
            model, metadata = model_registry.load_model(model_name)

            if model is None:
                click.echo(f"❌ Error: Model {model_name} not found", err=True)
                sys.exit(1)

            feature_columns = metadata.feature_names
            missing_features = set(feature_columns) - set(customer_df.columns)
            if missing_features:
                click.echo(f"❌ Error: Missing features: {missing_features}", err=True)
                sys.exit(1)

            X = customer_df[feature_columns]
            churn_probabilities = model.predict_proba(X)[:, 1]
            click.echo(f"🔮 Generated predictions using model {model_name}")
        else:
            click.echo(f"❌ Error: Either --predictions or --model-name must be specified", err=True)
            sys.exit(1)

        # Perform CLV analysis
        click.echo(f"🔄 Analyzing CLV with discount rate {discount_rate:.1%}...")

        clv_results = business_analyzer.analyze_customer_lifetime_value(
            customer_df,
            churn_probabilities,
            time_horizon_months=time_horizon,
            discount_rate=discount_rate
        )

        # Display summary results
        click.echo(f"\n💰 CLV Analysis Results:")
        click.echo(f"📊 Total Customers: {clv_results['total_customers']:,}")
        click.echo(f"💵 Average CLV: ${clv_results['average_clv']:,.2f}")
        click.echo(f"📈 Total CLV: ${clv_results['total_clv']:,.2f}")
        click.echo(f"⚠️ At-Risk CLV: ${clv_results['at_risk_clv']:,.2f}")
        click.echo(f"🎯 Retention Opportunity: ${clv_results['retention_opportunity']:,.2f}")

        # Display segment analysis
        if 'segment_analysis' in clv_results:
            click.echo(f"\n📊 CLV by Risk Segments:")
            segment_df = clv_results['segment_analysis']
            for _, row in segment_df.iterrows():
                risk_level = row['risk_level']
                customer_count = int(row['customer_count'])
                avg_clv = row['average_clv']
                total_clv = row['total_clv']

                click.echo(f"  {risk_level.title():<10}: {customer_count:>6,} customers, "
                          f"Avg CLV: ${avg_clv:>8,.2f}, Total: ${total_clv:>12,.2f}")

        # Top value customers at risk
        if 'top_at_risk_customers' in clv_results:
            click.echo(f"\n🚨 Top 10 High-Value At-Risk Customers:")
            top_customers = clv_results['top_at_risk_customers'].head(10)

            for idx, (_, customer) in enumerate(top_customers.iterrows(), 1):
                customer_id = customer.get('customer_id', f'Customer_{idx}')
                clv = customer['clv']
                churn_prob = customer['churn_probability']

                click.echo(f"  {idx:2d}. {customer_id}: CLV=${clv:,.2f}, "
                          f"Churn Risk={churn_prob:.1%}")

        # Save detailed results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save comprehensive results
            with open(output_path, 'w') as f:
                json.dump(clv_results, f, indent=2, default=str)

            # Save customer-level CLV data
            customer_clv_path = output_path.parent / f"{output_path.stem}_customer_clv.csv"
            if 'customer_clv_details' in clv_results:
                customer_clv_df = clv_results['customer_clv_details']
                customer_clv_df.to_csv(customer_clv_path, index=False)
                click.echo(f"📄 Customer CLV details saved to {customer_clv_path}")

            click.echo(f"📄 CLV analysis results saved to {output_path}")

        logger.info(f"CLV analysis completed: ${clv_results['total_clv']:,.2f} total CLV")

    except Exception as e:
        logger.error(f"CLV analysis failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@business.command()
@click.option('--customer-data', '-d', required=True, help='Customer data file path')
@click.option('--predictions', '-p', help='Churn predictions file path')
@click.option('--intervention-cost', type=float, default=25.0, help='Cost per intervention campaign')
@click.option('--success-rate', type=float, default=0.3, help='Expected intervention success rate')
@click.option('--scenarios', '-s', multiple=True,
              default=['optimistic', 'realistic', 'pessimistic'],
              help='ROI scenarios to analyze')
@click.option('--output-path', '-o', help='Output path for ROI analysis')
@click.option('--monte-carlo', is_flag=True, help='Run Monte Carlo simulation')
@click.option('--simulation-runs', type=int, default=1000, help='Number of Monte Carlo runs')
@click.pass_context
def calculate_roi(ctx, customer_data, predictions, intervention_cost, success_rate,
                 scenarios, output_path, monte_carlo, simulation_runs):
    """Calculate ROI for churn prevention interventions."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    roi_calculator = ctx.obj['roi_calculator']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📈 Starting ROI calculation for churn prevention")

        # Load customer data
        if not Path(customer_data).exists():
            click.echo(f"❌ Error: Customer data file {customer_data} not found", err=True)
            sys.exit(1)

        customer_df = pd.read_csv(customer_data)
        click.echo(f"👥 Loaded {len(customer_df)} customer records")

        # Load predictions
        if predictions and Path(predictions).exists():
            predictions_df = pd.read_csv(predictions)
            churn_probabilities = predictions_df.get('churn_probability', predictions_df.iloc[:, -1])
        else:
            # Use dummy predictions for demonstration
            click.echo(f"⚠️ Warning: Using simulated churn probabilities")
            np.random.seed(42)
            churn_probabilities = np.random.beta(2, 5, len(customer_df))

        # Set up ROI parameters
        roi_params = {
            'intervention_cost': intervention_cost,
            'success_rate': success_rate,
            'avg_monthly_revenue': config.business.avg_monthly_revenue,
            'customer_acquisition_cost': config.business.customer_acquisition_cost
        }

        click.echo(f"💰 ROI Parameters:")
        for param, value in roi_params.items():
            click.echo(f"  {param.replace('_', ' ').title()}: ${value:.2f}")

        # Calculate ROI for different scenarios
        roi_results = {}

        for scenario in scenarios:
            click.echo(f"\n🔄 Calculating ROI for {scenario} scenario...")

            # Adjust parameters based on scenario
            scenario_params = roi_params.copy()
            if scenario == 'optimistic':
                scenario_params['success_rate'] *= 1.5
                scenario_params['intervention_cost'] *= 0.8
            elif scenario == 'pessimistic':
                scenario_params['success_rate'] *= 0.7
                scenario_params['intervention_cost'] *= 1.2

            roi_result = roi_calculator.calculate_intervention_roi(
                customer_df,
                churn_probabilities,
                **scenario_params
            )

            roi_results[scenario] = roi_result

            # Display scenario results
            click.echo(f"  💵 Investment: ${roi_result['total_investment']:,.2f}")
            click.echo(f"  💰 Revenue Protected: ${roi_result['revenue_protected']:,.2f}")
            click.echo(f"  📈 Net Benefit: ${roi_result['net_benefit']:,.2f}")
            click.echo(f"  🎯 ROI: {roi_result['roi_percentage']:.1f}%")
            click.echo(f"  👥 Customers Targeted: {roi_result['customers_targeted']:,}")

        # Run Monte Carlo simulation if requested
        if monte_carlo:
            click.echo(f"\n🎲 Running Monte Carlo simulation ({simulation_runs:,} runs)...")

            mc_results = roi_calculator.monte_carlo_roi_simulation(
                customer_df,
                churn_probabilities,
                n_simulations=simulation_runs,
                base_params=roi_params
            )

            click.echo(f"📊 Monte Carlo Results:")
            click.echo(f"  Mean ROI: {mc_results['mean_roi']:.1f}%")
            click.echo(f"  Median ROI: {mc_results['median_roi']:.1f}%")
            click.echo(f"  ROI Standard Deviation: {mc_results['roi_std']:.1f}%")
            click.echo(f"  95% Confidence Interval: [{mc_results['roi_ci_lower']:.1f}%, {mc_results['roi_ci_upper']:.1f}%]")
            click.echo(f"  Probability of Positive ROI: {mc_results['prob_positive_roi']:.1%}")

            roi_results['monte_carlo'] = mc_results

        # Portfolio optimization
        click.echo(f"\n🎯 Optimizing intervention portfolio...")
        portfolio_results = roi_calculator.optimize_intervention_portfolio(
            customer_df,
            churn_probabilities,
            budget_constraint=roi_params['intervention_cost'] * len(customer_df) * 0.1  # 10% of max budget
        )

        click.echo(f"📊 Portfolio Optimization Results:")
        click.echo(f"  Optimal Budget Allocation: ${portfolio_results['optimal_budget']:,.2f}")
        click.echo(f"  Expected ROI: {portfolio_results['expected_roi']:.1f}%")
        click.echo(f"  Customers to Target: {portfolio_results['customers_to_target']:,}")
        click.echo(f"  Expected Revenue Protection: ${portfolio_results['expected_revenue']:,.2f}")

        roi_results['portfolio_optimization'] = portfolio_results

        # Investment recommendations
        click.echo(f"\n💡 Investment Recommendations:")

        best_scenario = max(roi_results.keys() - {'monte_carlo', 'portfolio_optimization'},
                          key=lambda s: roi_results[s]['roi_percentage'])
        best_roi = roi_results[best_scenario]['roi_percentage']

        click.echo(f"  🏆 Best Scenario: {best_scenario} ({best_roi:.1f}% ROI)")

        if best_roi > 200:
            click.echo(f"  ✅ Strong Investment Case: ROI exceeds 200%")
        elif best_roi > 100:
            click.echo(f"  ⚠️ Moderate Investment Case: ROI above 100%")
        else:
            click.echo(f"  ❌ Weak Investment Case: ROI below 100%")

        # Risk assessment
        if monte_carlo and mc_results['prob_positive_roi'] > 0.8:
            click.echo(f"  🛡️ Low Risk: 80%+ probability of positive ROI")
        elif monte_carlo and mc_results['prob_positive_roi'] > 0.6:
            click.echo(f"  ⚖️ Medium Risk: 60-80% probability of positive ROI")
        elif monte_carlo:
            click.echo(f"  ⚠️ High Risk: <60% probability of positive ROI")

        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'parameters': roi_params,
                'scenario_results': roi_results,
                'recommendations': {
                    'best_scenario': best_scenario,
                    'investment_decision': 'INVEST' if best_roi > 150 else 'CONSIDER' if best_roi > 100 else 'AVOID',
                    'risk_level': 'LOW' if monte_carlo and mc_results['prob_positive_roi'] > 0.8 else 'MEDIUM'
                }
            }

            with open(output_path, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)

            click.echo(f"📄 ROI analysis results saved to {output_path}")

        logger.info(f"ROI calculation completed: {best_roi:.1f}% best case ROI")

    except Exception as e:
        logger.error(f"ROI calculation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@business.command()
@click.option('--customer-data', '-d', required=True, help='Customer data file path')
@click.option('--predictions', '-p', required=True, help='Churn predictions file path')
@click.option('--risk-threshold', type=float, default=0.5, help='High-risk threshold for interventions')
@click.option('--budget', '-b', type=float, help='Total intervention budget')
@click.option('--campaign-types', '-c', multiple=True,
              default=['discount', 'retention_call', 'loyalty_program'],
              help='Available campaign types')
@click.option('--output-path', '-o', help='Output path for intervention plan')
@click.pass_context
def plan_interventions(ctx, customer_data, predictions, risk_threshold, budget,
                      campaign_types, output_path):
    """Plan targeted intervention campaigns for at-risk customers."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    business_analyzer = ctx.obj['business_analyzer']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🎯 Planning intervention campaigns for high-risk customers")

        # Load customer data
        if not Path(customer_data).exists():
            click.echo(f"❌ Error: Customer data file {customer_data} not found", err=True)
            sys.exit(1)

        customer_df = pd.read_csv(customer_data)

        # Load predictions
        if not Path(predictions).exists():
            click.echo(f"❌ Error: Predictions file {predictions} not found", err=True)
            sys.exit(1)

        predictions_df = pd.read_csv(predictions)

        # Merge customer data with predictions
        if 'customer_id' in customer_df.columns and 'customer_id' in predictions_df.columns:
            merged_df = customer_df.merge(predictions_df, on='customer_id', how='inner')
        else:
            # Assume same order if no customer_id
            merged_df = customer_df.copy()
            merged_df['churn_probability'] = predictions_df.get('churn_probability', predictions_df.iloc[:, -1])

        click.echo(f"👥 Loaded {len(merged_df)} customers with predictions")

        # Identify high-risk customers
        high_risk_customers = merged_df[merged_df['churn_probability'] >= risk_threshold]
        click.echo(f"⚠️ High-risk customers (>={risk_threshold:.1%}): {len(high_risk_customers):,}")

        if len(high_risk_customers) == 0:
            click.echo(f"ℹ️ No customers above risk threshold. Consider lowering threshold.")
            return

        # Define campaign characteristics
        campaign_configs = {
            'discount': {
                'cost_per_customer': 15.0,
                'success_rate': 0.35,
                'applicable_segments': ['price_sensitive', 'discount_seekers']
            },
            'retention_call': {
                'cost_per_customer': 5.0,
                'success_rate': 0.25,
                'applicable_segments': ['high_tenure', 'service_oriented']
            },
            'loyalty_program': {
                'cost_per_customer': 30.0,
                'success_rate': 0.45,
                'applicable_segments': ['high_value', 'engaged']
            },
            'premium_support': {
                'cost_per_customer': 50.0,
                'success_rate': 0.60,
                'applicable_segments': ['enterprise', 'high_value']
            }
        }

        # Customer segmentation for campaign targeting
        click.echo(f"🔄 Segmenting customers for targeted campaigns...")

        # Simple segmentation based on available features
        high_risk_customers = high_risk_customers.copy()

        # Revenue-based segmentation
        if 'monthly_charges' in high_risk_customers.columns:
            revenue_median = high_risk_customers['monthly_charges'].median()
            high_risk_customers['value_segment'] = high_risk_customers['monthly_charges'].apply(
                lambda x: 'high_value' if x > revenue_median else 'standard_value'
            )
        else:
            high_risk_customers['value_segment'] = 'standard_value'

        # Tenure-based segmentation
        if 'tenure' in high_risk_customers.columns:
            tenure_median = high_risk_customers['tenure'].median()
            high_risk_customers['tenure_segment'] = high_risk_customers['tenure'].apply(
                lambda x: 'high_tenure' if x > tenure_median else 'low_tenure'
            )
        else:
            high_risk_customers['tenure_segment'] = 'low_tenure'

        # Plan interventions
        intervention_plan = []
        total_cost = 0.0
        customers_targeted = 0

        for campaign_type in campaign_types:
            if campaign_type not in campaign_configs:
                click.echo(f"⚠️ Warning: Unknown campaign type '{campaign_type}', skipping")
                continue

            config_data = campaign_configs[campaign_type]

            # Select eligible customers for this campaign
            eligible_customers = high_risk_customers.copy()

            # Apply budget constraint
            cost_per_customer = config_data['cost_per_customer']
            if budget:
                remaining_budget = budget - total_cost
                max_customers_affordable = int(remaining_budget / cost_per_customer)

                if max_customers_affordable <= 0:
                    click.echo(f"💰 Budget exhausted, skipping {campaign_type}")
                    continue

                # Prioritize by churn probability and customer value
                if 'monthly_charges' in eligible_customers.columns:
                    eligible_customers['priority_score'] = (
                        eligible_customers['churn_probability'] * 0.7 +
                        (eligible_customers['monthly_charges'] / eligible_customers['monthly_charges'].max()) * 0.3
                    )
                else:
                    eligible_customers['priority_score'] = eligible_customers['churn_probability']

                eligible_customers = eligible_customers.nlargest(max_customers_affordable, 'priority_score')

            # Calculate campaign metrics
            campaign_customers = len(eligible_customers)
            campaign_cost = campaign_customers * cost_per_customer
            expected_saves = int(campaign_customers * config_data['success_rate'])

            if 'monthly_charges' in eligible_customers.columns:
                avg_monthly_revenue = eligible_customers['monthly_charges'].mean()
            else:
                avg_monthly_revenue = config.business.avg_monthly_revenue

            revenue_protected = expected_saves * avg_monthly_revenue * 12  # Annual revenue
            campaign_roi = ((revenue_protected - campaign_cost) / campaign_cost * 100) if campaign_cost > 0 else 0

            intervention_plan.append({
                'campaign_type': campaign_type,
                'customers_targeted': campaign_customers,
                'cost_per_customer': cost_per_customer,
                'total_cost': campaign_cost,
                'expected_success_rate': config_data['success_rate'],
                'expected_saves': expected_saves,
                'revenue_protected': revenue_protected,
                'roi_percentage': campaign_roi,
                'avg_monthly_revenue': avg_monthly_revenue
            })

            total_cost += campaign_cost
            customers_targeted += campaign_customers

        # Display intervention plan
        click.echo(f"\n🎯 Intervention Plan Summary:")
        click.echo(f"📊 Total Customers at Risk: {len(high_risk_customers):,}")
        click.echo(f"👥 Customers Targeted: {customers_targeted:,}")
        click.echo(f"💰 Total Investment: ${total_cost:,.2f}")

        if budget:
            click.echo(f"💳 Budget Utilization: {(total_cost/budget)*100:.1f}%")

        click.echo(f"\n📋 Campaign Details:")
        total_expected_saves = 0
        total_revenue_protected = 0

        for campaign in intervention_plan:
            total_expected_saves += campaign['expected_saves']
            total_revenue_protected += campaign['revenue_protected']

            click.echo(f"\n  🎪 {campaign['campaign_type'].title()}:")
            click.echo(f"    Customers: {campaign['customers_targeted']:,}")
            click.echo(f"    Cost: ${campaign['total_cost']:,.2f}")
            click.echo(f"    Expected Saves: {campaign['expected_saves']}")
            click.echo(f"    Revenue Protected: ${campaign['revenue_protected']:,.2f}")
            click.echo(f"    ROI: {campaign['roi_percentage']:.1f}%")

        # Overall impact
        overall_roi = ((total_revenue_protected - total_cost) / total_cost * 100) if total_cost > 0 else 0

        click.echo(f"\n📈 Overall Impact:")
        click.echo(f"  Expected Customer Saves: {total_expected_saves:,}")
        click.echo(f"  Total Revenue Protected: ${total_revenue_protected:,.2f}")
        click.echo(f"  Overall ROI: {overall_roi:.1f}%")
        click.echo(f"  Payback Period: {(total_cost / (total_revenue_protected / 12)):.1f} months")

        # Recommendations
        click.echo(f"\n💡 Recommendations:")

        best_campaign = max(intervention_plan, key=lambda x: x['roi_percentage']) if intervention_plan else None
        if best_campaign:
            click.echo(f"  🏆 Best ROI Campaign: {best_campaign['campaign_type']} ({best_campaign['roi_percentage']:.1f}%)")

        if overall_roi > 200:
            click.echo(f"  ✅ Highly Recommended: Execute full intervention plan")
        elif overall_roi > 100:
            click.echo(f"  ⚠️ Proceed with Caution: Focus on highest ROI campaigns")
        else:
            click.echo(f"  ❌ Not Recommended: ROI below 100%")

        # Save intervention plan
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            plan_results = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_at_risk_customers': len(high_risk_customers),
                    'customers_targeted': customers_targeted,
                    'total_investment': total_cost,
                    'expected_saves': total_expected_saves,
                    'total_revenue_protected': total_revenue_protected,
                    'overall_roi': overall_roi,
                    'risk_threshold': risk_threshold,
                    'budget_constraint': budget
                },
                'campaigns': intervention_plan,
                'customer_segments': {
                    'high_value_customers': len(high_risk_customers[high_risk_customers['value_segment'] == 'high_value']),
                    'high_tenure_customers': len(high_risk_customers[high_risk_customers['tenure_segment'] == 'high_tenure'])
                }
            }

            with open(output_path, 'w') as f:
                json.dump(plan_results, f, indent=2, default=str)

            # Save customer-level intervention assignments
            customer_plan_path = output_path.parent / f"{output_path.stem}_customer_assignments.csv"
            high_risk_customers.to_csv(customer_plan_path, index=False)

            click.echo(f"📄 Intervention plan saved to {output_path}")
            click.echo(f"📄 Customer assignments saved to {customer_plan_path}")

        logger.info(f"Intervention planning completed: {customers_targeted} customers targeted, {overall_roi:.1f}% ROI")

    except Exception as e:
        logger.error(f"Intervention planning failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@business.command()
@click.option('--data-sources', '-d', multiple=True, required=True,
              help='Data source paths (customer data, predictions, etc.)')
@click.option('--report-type', '-t', default='comprehensive',
              type=click.Choice(['summary', 'detailed', 'comprehensive', 'executive']),
              help='Type of business report to generate')
@click.option('--output-path', '-o', required=True, help='Output path for business report')
@click.option('--time-period', '-p', default='monthly',
              type=click.Choice(['daily', 'weekly', 'monthly', 'quarterly']),
              help='Reporting time period')
@click.option('--include-charts', is_flag=True, help='Include visualizations in report')
@click.pass_context
def generate_report(ctx, data_sources, report_type, output_path, time_period, include_charts):
    """Generate comprehensive business reports and dashboards."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    business_analyzer = ctx.obj['business_analyzer']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"📊 Generating {report_type} business report")

        # Load and combine data sources
        combined_data = {}
        for i, source_path in enumerate(data_sources):
            if not Path(source_path).exists():
                click.echo(f"❌ Error: Data source {source_path} not found", err=True)
                continue

            source_name = f"source_{i}" if len(data_sources) > 1 else "main_data"
            combined_data[source_name] = pd.read_csv(source_path)
            click.echo(f"📂 Loaded {len(combined_data[source_name])} records from {source_path}")

        if not combined_data:
            click.echo(f"❌ Error: No valid data sources found", err=True)
            sys.exit(1)

        # Generate report based on type
        report_data = {
            'metadata': {
                'report_type': report_type,
                'generation_timestamp': datetime.now().isoformat(),
                'time_period': time_period,
                'data_sources': list(data_sources)
            }
        }

        main_data = list(combined_data.values())[0]

        if report_type in ['summary', 'comprehensive', 'executive']:
            # High-level metrics
            click.echo(f"🔄 Calculating summary metrics...")

            summary_metrics = {
                'total_customers': len(main_data),
                'reporting_period': time_period,
                'data_freshness': datetime.now().isoformat()
            }

            # Churn analysis if predictions available
            if 'churn_probability' in main_data.columns:
                high_risk_count = len(main_data[main_data['churn_probability'] >= 0.5])
                summary_metrics.update({
                    'high_risk_customers': high_risk_count,
                    'churn_risk_rate': high_risk_count / len(main_data),
                    'avg_churn_probability': main_data['churn_probability'].mean()
                })

            # Revenue metrics if available
            if 'monthly_charges' in main_data.columns:
                total_revenue = main_data['monthly_charges'].sum()
                at_risk_revenue = main_data[main_data.get('churn_probability', 0) >= 0.5]['monthly_charges'].sum()

                summary_metrics.update({
                    'total_monthly_revenue': total_revenue,
                    'at_risk_revenue': at_risk_revenue,
                    'revenue_at_risk_percentage': at_risk_revenue / total_revenue if total_revenue > 0 else 0
                })

            report_data['summary_metrics'] = summary_metrics

        if report_type in ['detailed', 'comprehensive']:
            # Detailed analysis
            click.echo(f"🔄 Performing detailed analysis...")

            detailed_analysis = {}

            # Customer segmentation
            if 'monthly_charges' in main_data.columns:
                revenue_quartiles = main_data['monthly_charges'].quantile([0.25, 0.5, 0.75])
                detailed_analysis['revenue_segments'] = {
                    'low_value': len(main_data[main_data['monthly_charges'] <= revenue_quartiles[0.25]]),
                    'medium_value': len(main_data[(main_data['monthly_charges'] > revenue_quartiles[0.25]) &
                                                (main_data['monthly_charges'] <= revenue_quartiles[0.75])]),
                    'high_value': len(main_data[main_data['monthly_charges'] > revenue_quartiles[0.75]])
                }

            # Tenure analysis
            if 'tenure' in main_data.columns:
                detailed_analysis['tenure_analysis'] = {
                    'avg_tenure_months': main_data['tenure'].mean(),
                    'median_tenure_months': main_data['tenure'].median(),
                    'new_customers_6m': len(main_data[main_data['tenure'] <= 6]),
                    'loyal_customers_24m': len(main_data[main_data['tenure'] >= 24])
                }

            # Churn risk distribution
            if 'churn_probability' in main_data.columns:
                risk_distribution = {
                    'low_risk': len(main_data[main_data['churn_probability'] < 0.3]),
                    'medium_risk': len(main_data[(main_data['churn_probability'] >= 0.3) &
                                               (main_data['churn_probability'] < 0.7)]),
                    'high_risk': len(main_data[main_data['churn_probability'] >= 0.7])
                }
                detailed_analysis['risk_distribution'] = risk_distribution

            report_data['detailed_analysis'] = detailed_analysis

        if report_type == 'comprehensive':
            # Advanced analytics
            click.echo(f"🔄 Performing advanced analytics...")

            advanced_analytics = {}

            # Trend analysis (if temporal data available)
            if 'date' in main_data.columns or 'created_date' in main_data.columns:
                # Placeholder for trend analysis
                advanced_analytics['trends'] = {
                    'note': 'Trend analysis requires time-series data setup'
                }

            # Correlation analysis
            if 'churn_probability' in main_data.columns:
                numeric_columns = main_data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 1:
                    correlations = main_data[numeric_columns].corr()['churn_probability'].abs().sort_values(ascending=False)
                    advanced_analytics['feature_correlations'] = correlations.head(10).to_dict()

            # CLV estimation
            if 'monthly_charges' in main_data.columns and 'tenure' in main_data.columns:
                estimated_clv = main_data['monthly_charges'] * main_data['tenure']
                advanced_analytics['clv_analysis'] = {
                    'avg_estimated_clv': estimated_clv.mean(),
                    'total_estimated_clv': estimated_clv.sum(),
                    'clv_distribution_quartiles': estimated_clv.quantile([0.25, 0.5, 0.75]).to_dict()
                }

            report_data['advanced_analytics'] = advanced_analytics

        # Executive summary for executive reports
        if report_type == 'executive':
            click.echo(f"🔄 Creating executive summary...")

            executive_summary = {
                'key_findings': [],
                'recommendations': [],
                'risk_assessment': 'Medium',  # Default
                'investment_priorities': []
            }

            # Generate key findings
            if 'summary_metrics' in report_data:
                metrics = report_data['summary_metrics']

                if 'churn_risk_rate' in metrics:
                    risk_rate = metrics['churn_risk_rate']
                    executive_summary['key_findings'].append(
                        f"{risk_rate:.1%} of customers are at high risk of churn"
                    )

                    if risk_rate > 0.2:
                        executive_summary['risk_assessment'] = 'High'
                        executive_summary['recommendations'].append(
                            "Immediate intervention required for high-risk customers"
                        )
                    elif risk_rate > 0.1:
                        executive_summary['risk_assessment'] = 'Medium'
                        executive_summary['recommendations'].append(
                            "Implement targeted retention campaigns"
                        )

                if 'revenue_at_risk_percentage' in metrics:
                    revenue_risk = metrics['revenue_at_risk_percentage']
                    executive_summary['key_findings'].append(
                        f"{revenue_risk:.1%} of monthly revenue is at risk"
                    )

            report_data['executive_summary'] = executive_summary

        # Generate visualizations if requested
        if include_charts:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns

                click.echo(f"📊 Generating visualizations...")

                # Create output directory for charts
                output_dir = Path(output_path).parent / f"{Path(output_path).stem}_charts"
                output_dir.mkdir(exist_ok=True)

                chart_files = []

                # Churn risk distribution chart
                if 'churn_probability' in main_data.columns:
                    plt.figure(figsize=(10, 6))
                    plt.hist(main_data['churn_probability'], bins=30, alpha=0.7, edgecolor='black')
                    plt.xlabel('Churn Probability')
                    plt.ylabel('Number of Customers')
                    plt.title('Distribution of Churn Risk Scores')
                    plt.grid(True, alpha=0.3)

                    chart_path = output_dir / 'churn_risk_distribution.png'
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files.append(str(chart_path))

                # Revenue vs Risk scatter plot
                if 'monthly_charges' in main_data.columns and 'churn_probability' in main_data.columns:
                    plt.figure(figsize=(10, 6))
                    plt.scatter(main_data['monthly_charges'], main_data['churn_probability'], alpha=0.6)
                    plt.xlabel('Monthly Charges ($)')
                    plt.ylabel('Churn Probability')
                    plt.title('Customer Value vs Churn Risk')
                    plt.grid(True, alpha=0.3)

                    chart_path = output_dir / 'value_vs_risk.png'
                    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    chart_files.append(str(chart_path))

                report_data['visualizations'] = {
                    'chart_directory': str(output_dir),
                    'chart_files': chart_files
                }

                click.echo(f"📊 Generated {len(chart_files)} visualization(s)")

            except ImportError:
                click.echo(f"⚠️ Warning: Matplotlib not available, skipping charts")

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        click.echo(f"✅ Business report generated successfully")
        click.echo(f"📄 Report saved to {output_path}")
        click.echo(f"📊 Report type: {report_type}")
        click.echo(f"📈 Data sources: {len(data_sources)}")

        # Display key metrics
        if 'summary_metrics' in report_data:
            metrics = report_data['summary_metrics']
            click.echo(f"\n📊 Key Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    if 'percentage' in key or 'rate' in key:
                        click.echo(f"  {key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        click.echo(f"  {key.replace('_', ' ').title()}: {value:,.2f}")
                else:
                    click.echo(f"  {key.replace('_', ' ').title()}: {value}")

        logger.info(f"Business report generated: {report_type} report with {len(data_sources)} data sources")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@business.command()
@click.option('--baseline-data', '-b', required=True, help='Baseline performance data')
@click.option('--test-data', '-t', required=True, help='A/B test performance data')
@click.option('--metric', '-m', default='conversion_rate',
              help='Primary metric for comparison')
@click.option('--confidence-level', type=float, default=0.95,
              help='Statistical confidence level')
@click.option('--output-path', '-o', help='Output path for A/B test results')
@click.pass_context
def ab_test_analysis(ctx, baseline_data, test_data, metric, confidence_level, output_path):
    """Analyze A/B test results for intervention campaigns."""

    config = ctx.obj['config']
    logger = ctx.obj['logger']
    verbose = ctx.obj['verbose']

    try:
        if verbose:
            click.echo(f"🧪 Starting A/B test analysis for {metric}")

        # Load test data
        if not Path(baseline_data).exists():
            click.echo(f"❌ Error: Baseline data file {baseline_data} not found", err=True)
            sys.exit(1)

        if not Path(test_data).exists():
            click.echo(f"❌ Error: Test data file {test_data} not found", err=True)
            sys.exit(1)

        baseline_df = pd.read_csv(baseline_data)
        test_df = pd.read_csv(test_data)

        click.echo(f"📊 Baseline group: {len(baseline_df)} observations")
        click.echo(f"📊 Test group: {len(test_df)} observations")

        # Validate metric exists
        if metric not in baseline_df.columns or metric not in test_df.columns:
            click.echo(f"❌ Error: Metric '{metric}' not found in data", err=True)
            sys.exit(1)

        # Calculate basic statistics
        baseline_mean = baseline_df[metric].mean()
        test_mean = test_df[metric].mean()

        baseline_std = baseline_df[metric].std()
        test_std = test_df[metric].std()

        # Effect size
        absolute_effect = test_mean - baseline_mean
        relative_effect = (absolute_effect / baseline_mean) * 100 if baseline_mean != 0 else 0

        click.echo(f"\n📊 Descriptive Statistics:")
        click.echo(f"  Baseline {metric}: {baseline_mean:.4f} ± {baseline_std:.4f}")
        click.echo(f"  Test {metric}: {test_mean:.4f} ± {test_std:.4f}")
        click.echo(f"  Absolute Effect: {absolute_effect:.4f}")
        click.echo(f"  Relative Effect: {relative_effect:+.2f}%")

        # Statistical significance test
        from scipy import stats

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(test_df[metric], baseline_df[metric])

        # Calculate confidence interval for difference
        n1, n2 = len(baseline_df), len(test_df)
        pooled_std = np.sqrt(((n1-1)*baseline_std**2 + (n2-1)*test_std**2) / (n1+n2-2))
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)

        alpha = 1 - confidence_level
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)

        ci_lower = absolute_effect - t_critical * se_diff
        ci_upper = absolute_effect + t_critical * se_diff

        # Results
        is_significant = p_value < alpha

        click.echo(f"\n🔬 Statistical Analysis:")
        click.echo(f"  T-statistic: {t_stat:.4f}")
        click.echo(f"  P-value: {p_value:.6f}")
        click.echo(f"  Significant at {confidence_level:.0%} level: {'Yes' if is_significant else 'No'}")
        click.echo(f"  {confidence_level:.0%} Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

        # Effect size interpretation
        if pooled_std > 0:
            cohens_d = absolute_effect / pooled_std
            click.echo(f"  Cohen's d (effect size): {cohens_d:.3f}")

            if abs(cohens_d) < 0.2:
                effect_interpretation = "Small"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "Medium"
            else:
                effect_interpretation = "Large"

            click.echo(f"  Effect size interpretation: {effect_interpretation}")

        # Business impact assessment
        click.echo(f"\n💼 Business Impact:")

        if metric in ['conversion_rate', 'retention_rate', 'success_rate']:
            # Calculate impact on customer outcomes
            if is_significant and relative_effect > 0:
                click.echo(f"  ✅ Test intervention shows {relative_effect:.1f}% improvement")

                # Estimate customer impact
                total_customers = 10000  # Example customer base
                customers_impacted = int(total_customers * (absolute_effect))
                click.echo(f"  📈 Estimated additional customers retained: {customers_impacted:,}")

                # Revenue impact
                avg_monthly_revenue = config.business.avg_monthly_revenue
                annual_revenue_impact = customers_impacted * avg_monthly_revenue * 12
                click.echo(f"  💰 Estimated annual revenue impact: ${annual_revenue_impact:,.2f}")

            elif is_significant and relative_effect < 0:
                click.echo(f"  ❌ Test intervention shows {abs(relative_effect):.1f}% decrease")
                click.echo(f"  🚫 Recommendation: Do not implement this intervention")
            else:
                click.echo(f"  ⚖️ No significant difference detected")
                click.echo(f"  🔄 Recommendation: Continue current approach or test alternatives")

        # Sample size and power analysis
        from math import sqrt, log

        # Post-hoc power calculation
        try:
            from statsmodels.stats.power import ttest_power
            observed_power = ttest_power(cohens_d, n1, alpha, alternative='two-sided')
            click.echo(f"\n⚡ Power Analysis:")
            click.echo(f"  Observed statistical power: {observed_power:.3f}")

            if observed_power < 0.8:
                click.echo(f"  ⚠️ Low power detected. Consider larger sample size for future tests.")
        except ImportError:
            click.echo(f"\n⚡ Power Analysis: (statsmodels required for detailed power analysis)")

        # Recommendations
        click.echo(f"\n💡 Recommendations:")

        if is_significant and relative_effect > 5:  # >5% improvement
            click.echo(f"  ✅ Strong Recommendation: Implement test intervention")
            click.echo(f"  📊 Evidence: Statistically significant with meaningful business impact")
        elif is_significant and relative_effect > 0:
            click.echo(f"  ⚠️ Moderate Recommendation: Consider implementing with monitoring")
            click.echo(f"  📊 Evidence: Statistically significant but modest business impact")
        elif not is_significant and abs(relative_effect) > 2:
            click.echo(f"  🔄 Inconclusive: Increase sample size and retest")
            click.echo(f"  📊 Evidence: Effect size suggests potential impact but lacks significance")
        else:
            click.echo(f"  ❌ No Action: No evidence of meaningful improvement")
            click.echo(f"  📊 Evidence: No significant effect or very small effect size")

        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            ab_results = {
                'timestamp': datetime.now().isoformat(),
                'test_configuration': {
                    'baseline_data': str(baseline_data),
                    'test_data': str(test_data),
                    'metric': metric,
                    'confidence_level': confidence_level
                },
                'sample_sizes': {
                    'baseline_n': len(baseline_df),
                    'test_n': len(test_df)
                },
                'descriptive_statistics': {
                    'baseline_mean': baseline_mean,
                    'baseline_std': baseline_std,
                    'test_mean': test_mean,
                    'test_std': test_std
                },
                'effect_analysis': {
                    'absolute_effect': absolute_effect,
                    'relative_effect_percent': relative_effect,
                    'cohens_d': cohens_d if 'cohens_d' in locals() else None
                },
                'statistical_test': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'is_significant': is_significant,
                    'confidence_interval': [ci_lower, ci_upper]
                },
                'business_impact': {
                    'recommendation': 'IMPLEMENT' if is_significant and relative_effect > 5 else
                                    'CONSIDER' if is_significant and relative_effect > 0 else
                                    'RETEST' if not is_significant and abs(relative_effect) > 2 else 'REJECT'
                }
            }

            with open(output_path, 'w') as f:
                json.dump(ab_results, f, indent=2, default=str)

            click.echo(f"📄 A/B test results saved to {output_path}")

        logger.info(f"A/B test analysis completed: {relative_effect:+.2f}% effect, p={p_value:.6f}")

    except Exception as e:
        logger.error(f"A/B test analysis failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    business()
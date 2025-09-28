"""
Explainable AI and Model Interpretability Module
Professional-grade model interpretation for customer churn prediction

Requirements:
pip install shap lime scikit-learn pandas numpy matplotlib seaborn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Explainable AI libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("⚠️  SHAP not installed. Run: pip install shap")
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    print("⚠️  LIME not installed. Run: pip install lime")
    LIME_AVAILABLE = False


class ExplainableChurnModel:
    """
    Professional explainable AI implementation for churn prediction

    Provides model interpretability through:
    1. SHAP (SHapley Additive exPlanations) values
    2. LIME (Local Interpretable Model-agnostic Explanations)
    3. Feature importance analysis
    4. Customer segment analysis
    5. Bias detection and mitigation
    """

    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.shap_explainer = None
        self.lime_explainer = None
        self.X_train = None
        self.y_train = None

    def fit_explainers(self, X_train, y_train, X_test=None):
        """
        Fit explainability models on training data
        """
        self.X_train = X_train
        self.y_train = y_train

        print("🔧 Fitting Explainable AI Components...")

        # Fit SHAP explainer
        if SHAP_AVAILABLE and self.model is not None:
            print("  ✅ Initializing SHAP explainer...")
            if hasattr(self.model, 'predict_proba'):
                # For tree-based models, use TreeExplainer for efficiency
                if hasattr(self.model, 'estimators_'):
                    self.shap_explainer = shap.TreeExplainer(self.model)
                else:
                    # For other models, use KernelExplainer
                    self.shap_explainer = shap.KernelExplainer(
                        self.model.predict_proba,
                        shap.sample(X_train, 100)  # Use sample for efficiency
                    )

        # Fit LIME explainer
        if LIME_AVAILABLE:
            print("  ✅ Initializing LIME explainer...")
            self.lime_explainer = LimeTabularExplainer(
                X_train.values if hasattr(X_train, 'values') else X_train,
                feature_names=self.feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification',
                discretize_continuous=True
            )

        print("✅ Explainability components ready")

    def analyze_global_importance(self, save_plots=True):
        """
        Global model interpretability analysis
        """
        print("🌍 Global Model Interpretability Analysis")
        print("=" * 50)

        results = {}

        # 1. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names or [f'Feature_{i}' for i in range(len(self.model.feature_importances_))],
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            results['feature_importance'] = importance_df

            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df.head(15), y='feature', x='importance', palette='viridis')
            plt.title('Global Feature Importance (Model-based)', fontsize=16, fontweight='bold')
            plt.xlabel('Importance Score')
            plt.ylabel('Features')
            plt.tight_layout()
            if save_plots:
                plt.savefig('feature_importance_global.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"📊 Top 10 Most Important Features:")
            for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
                print(f"   {i:2d}. {row['feature']:20}: {row['importance']:.4f}")

        # 2. SHAP Global Analysis
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            print(f"\n🔍 SHAP Global Analysis...")

            # Calculate SHAP values for a sample (for efficiency)
            sample_size = min(500, len(self.X_train))
            sample_indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train.iloc[sample_indices] if hasattr(self.X_train, 'iloc') else self.X_train[sample_indices]

            try:
                shap_values = self.shap_explainer.shap_values(X_sample)

                # For binary classification, get positive class SHAP values
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Churn class

                results['shap_values'] = shap_values
                results['shap_sample'] = X_sample

                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample,
                                feature_names=self.feature_names,
                                show=False, max_display=15)
                plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold')
                if save_plots:
                    plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
                plt.show()

                # Feature importance plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample,
                                feature_names=self.feature_names,
                                plot_type="bar", show=False, max_display=15)
                plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16, fontweight='bold')
                if save_plots:
                    plt.savefig('shap_importance_bar.png', dpi=300, bbox_inches='tight')
                plt.show()

                print(f"✅ SHAP analysis complete for {sample_size} samples")

            except Exception as e:
                print(f"⚠️  SHAP analysis failed: {str(e)}")

        return results

    def explain_individual_prediction(self, customer_data, customer_id="Unknown"):
        """
        Explain individual customer churn prediction
        """
        print(f"👤 Individual Customer Analysis: {customer_id}")
        print("=" * 50)

        # Ensure customer_data is in correct format
        if hasattr(customer_data, 'values'):
            customer_array = customer_data.values.reshape(1, -1)
        else:
            customer_array = np.array(customer_data).reshape(1, -1)

        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            churn_probability = self.model.predict_proba(customer_array)[0][1]
            prediction = "CHURN RISK" if churn_probability > 0.5 else "SAFE"
        else:
            prediction = self.model.predict(customer_array)[0]
            churn_probability = prediction

        print(f"🎯 Prediction: {prediction}")
        print(f"📊 Churn Probability: {churn_probability:.1%}")

        explanations = {}

        # SHAP individual explanation
        if SHAP_AVAILABLE and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(customer_array)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Churn class

                explanations['shap'] = shap_values[0]

                # Create SHAP waterfall plot
                if hasattr(shap, 'waterfall_plot'):
                    plt.figure(figsize=(12, 8))
                    shap.waterfall_plot(shap.Explanation(
                        values=shap_values[0],
                        base_values=self.shap_explainer.expected_value[1] if isinstance(self.shap_explainer.expected_value, list) else self.shap_explainer.expected_value,
                        data=customer_array[0],
                        feature_names=self.feature_names
                    ))
                    plt.title(f'SHAP Explanation for Customer {customer_id}', fontsize=16, fontweight='bold')
                    plt.tight_layout()
                    plt.show()

                # Feature contribution analysis
                feature_contributions = list(zip(
                    self.feature_names or [f'Feature_{i}' for i in range(len(shap_values[0]))],
                    shap_values[0]
                ))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                print(f"\n🔍 Top 10 Contributing Factors (SHAP):")
                for i, (feature, contribution) in enumerate(feature_contributions[:10], 1):
                    direction = "↗️ INCREASES" if contribution > 0 else "↘️ DECREASES"
                    print(f"   {i:2d}. {feature:20}: {contribution:+.4f} {direction} churn risk")

            except Exception as e:
                print(f"⚠️  SHAP individual analysis failed: {str(e)}")

        # LIME individual explanation
        if LIME_AVAILABLE and self.lime_explainer is not None:
            try:
                lime_explanation = self.lime_explainer.explain_instance(
                    customer_array[0],
                    self.model.predict_proba,
                    num_features=10
                )

                explanations['lime'] = lime_explanation

                # Show LIME explanation
                lime_explanation.show_in_notebook(show_table=True)

                # Get LIME feature contributions
                lime_features = lime_explanation.as_list()
                print(f"\n🔍 Top Contributing Factors (LIME):")
                for i, (feature_condition, contribution) in enumerate(lime_features, 1):
                    direction = "↗️ INCREASES" if contribution > 0 else "↘️ DECREASES"
                    print(f"   {i:2d}. {feature_condition}: {contribution:+.4f} {direction} churn risk")

            except Exception as e:
                print(f"⚠️  LIME individual analysis failed: {str(e)}")

        return explanations

    def analyze_customer_segments(self, X_data, y_data=None, segment_features=None):
        """
        Analyze model behavior across different customer segments
        """
        print("👥 Customer Segment Analysis")
        print("=" * 50)

        if segment_features is None:
            segment_features = self.feature_names[:3] if self.feature_names else [0, 1, 2]

        segment_analysis = {}

        # Create segments based on feature quantiles
        for feature in segment_features:
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature) if self.feature_names else 0
            else:
                feature_idx = feature

            feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature_{feature_idx}"

            # Get feature values
            if hasattr(X_data, 'iloc'):
                feature_values = X_data.iloc[:, feature_idx]
            else:
                feature_values = X_data[:, feature_idx]

            # Create quartile-based segments
            quartiles = np.percentile(feature_values, [25, 50, 75])

            segments = {
                'Low': feature_values <= quartiles[0],
                'Medium-Low': (feature_values > quartiles[0]) & (feature_values <= quartiles[1]),
                'Medium-High': (feature_values > quartiles[1]) & (feature_values <= quartiles[2]),
                'High': feature_values > quartiles[2]
            }

            segment_results = {}

            for segment_name, segment_mask in segments.items():
                if np.sum(segment_mask) > 0:
                    segment_X = X_data[segment_mask] if not hasattr(X_data, 'iloc') else X_data.iloc[segment_mask]

                    # Get predictions for segment
                    if hasattr(self.model, 'predict_proba'):
                        segment_predictions = self.model.predict_proba(segment_X)[:, 1]
                    else:
                        segment_predictions = self.model.predict(segment_X)

                    segment_results[segment_name] = {
                        'size': np.sum(segment_mask),
                        'avg_churn_prob': np.mean(segment_predictions),
                        'churn_rate': np.mean(segment_predictions > 0.5) if hasattr(self.model, 'predict_proba') else np.mean(segment_predictions)
                    }

            segment_analysis[feature_name] = segment_results

            # Print segment analysis
            print(f"\n📊 Segments by {feature_name}:")
            for segment_name, stats in segment_results.items():
                print(f"   {segment_name:12}: {stats['size']:>4} customers, "
                      f"{stats['avg_churn_prob']:>5.1%} avg churn prob, "
                      f"{stats['churn_rate']:>5.1%} predicted churn rate")

        return segment_analysis

    def detect_model_bias(self, X_data, sensitive_features=None):
        """
        Detect potential bias in model predictions across sensitive attributes
        """
        print("⚖️  Model Bias Detection Analysis")
        print("=" * 50)

        if sensitive_features is None:
            # Use first few features as proxy for sensitive attributes
            sensitive_features = self.feature_names[:2] if self.feature_names else [0, 1]

        bias_analysis = {}

        for feature in sensitive_features:
            if isinstance(feature, str):
                feature_idx = self.feature_names.index(feature) if self.feature_names else 0
            else:
                feature_idx = feature

            feature_name = self.feature_names[feature_idx] if self.feature_names else f"Feature_{feature_idx}"

            # Get feature values
            if hasattr(X_data, 'iloc'):
                feature_values = X_data.iloc[:, feature_idx]
            else:
                feature_values = X_data[:, feature_idx]

            # Create binary groups (above/below median)
            median_value = np.median(feature_values)
            group_high = feature_values > median_value
            group_low = feature_values <= median_value

            # Get predictions for each group
            X_high = X_data[group_high] if not hasattr(X_data, 'iloc') else X_data.iloc[group_high]
            X_low = X_data[group_low] if not hasattr(X_data, 'iloc') else X_data.iloc[group_low]

            if len(X_high) > 0 and len(X_low) > 0:
                if hasattr(self.model, 'predict_proba'):
                    pred_high = self.model.predict_proba(X_high)[:, 1]
                    pred_low = self.model.predict_proba(X_low)[:, 1]
                else:
                    pred_high = self.model.predict(X_high)
                    pred_low = self.model.predict(X_low)

                # Calculate bias metrics
                avg_pred_high = np.mean(pred_high)
                avg_pred_low = np.mean(pred_low)
                bias_difference = avg_pred_high - avg_pred_low
                bias_ratio = avg_pred_high / avg_pred_low if avg_pred_low > 0 else np.inf

                bias_analysis[feature_name] = {
                    'high_group_avg': avg_pred_high,
                    'low_group_avg': avg_pred_low,
                    'difference': bias_difference,
                    'ratio': bias_ratio,
                    'high_group_size': len(X_high),
                    'low_group_size': len(X_low)
                }

                print(f"\n📊 Bias Analysis for {feature_name}:")
                print(f"   High Group (>{median_value:.2f}): {avg_pred_high:.1%} avg churn prob ({len(X_high)} customers)")
                print(f"   Low Group  (≤{median_value:.2f}): {avg_pred_low:.1%} avg churn prob ({len(X_low)} customers)")
                print(f"   Difference: {bias_difference:+.1%}")
                print(f"   Ratio: {bias_ratio:.2f}")

                # Flag potential bias
                if abs(bias_difference) > 0.1 or bias_ratio > 1.5 or bias_ratio < 0.67:
                    print(f"   ⚠️  POTENTIAL BIAS DETECTED")
                else:
                    print(f"   ✅ No significant bias detected")

        return bias_analysis

    def generate_business_insights(self, importance_analysis, segment_analysis):
        """
        Generate business-focused insights from explainability analysis
        """
        print("💼 Business Insights & Recommendations")
        print("=" * 50)

        insights = []

        # Feature importance insights
        if 'feature_importance' in importance_analysis:
            top_features = importance_analysis['feature_importance'].head(5)
            insights.append("🔍 KEY CHURN DRIVERS:")
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                insights.append(f"   {i}. {row['feature']} - High impact on churn decisions")

        # Segment insights
        if segment_analysis:
            insights.append("\n👥 HIGH-RISK CUSTOMER SEGMENTS:")
            for feature, segments in segment_analysis.items():
                max_risk_segment = max(segments.items(), key=lambda x: x[1]['churn_rate'])
                insights.append(f"   • {feature}: {max_risk_segment[0]} segment ({max_risk_segment[1]['churn_rate']:.1%} churn rate)")

        # Actionable recommendations
        insights.append("\n💡 ACTIONABLE RECOMMENDATIONS:")
        insights.append("   1. Focus retention efforts on high-risk segments identified above")
        insights.append("   2. Monitor key churn drivers for early warning signals")
        insights.append("   3. Develop targeted interventions for each risk segment")
        insights.append("   4. Implement real-time scoring for customer interactions")
        insights.append("   5. Regular model monitoring to detect performance drift")

        for insight in insights:
            print(insight)

        return insights


def demonstrate_explainable_ai():
    """
    Demonstration of explainable AI capabilities
    """
    print("🚀 Explainable AI Demonstration")
    print("=" * 60)

    # Generate sample data for demonstration
    np.random.seed(42)
    n_samples = 1000

    # Create realistic customer features
    data = pd.DataFrame({
        'tenure': np.random.exponential(12, n_samples).clip(1, 60),
        'monthly_charges': np.random.normal(80, 25, n_samples).clip(30, 150),
        'support_calls': np.random.poisson(1.5, n_samples).clip(0, 10),
        'contract_monthly': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'payment_electronic': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'service_quality': np.random.normal(7, 2, n_samples).clip(1, 10)
    })

    # Create realistic churn target
    churn_score = (
        -0.1 * data['tenure'] +
        0.02 * data['monthly_charges'] +
        0.5 * data['support_calls'] +
        2.0 * data['contract_monthly'] +
        1.5 * data['payment_electronic'] +
        -0.3 * data['service_quality'] +
        np.random.normal(0, 1, n_samples)
    )

    churn_prob = 1 / (1 + np.exp(-churn_score))
    y = np.random.binomial(1, churn_prob)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, y, test_size=0.3, random_state=42, stratify=y
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Initialize explainable AI
    explainer = ExplainableChurnModel(
        model=model,
        feature_names=list(data.columns)
    )

    # Fit explainers
    explainer.fit_explainers(X_train, y_train, X_test)

    # Global analysis
    importance_results = explainer.analyze_global_importance(save_plots=False)

    # Individual customer analysis
    sample_customer = X_test.iloc[0]
    explainer.explain_individual_prediction(sample_customer, customer_id="DEMO_001")

    # Segment analysis
    segment_results = explainer.analyze_customer_segments(X_test, y_test)

    # Bias detection
    bias_results = explainer.detect_model_bias(X_test)

    # Business insights
    explainer.generate_business_insights(importance_results, segment_results)

    print("\n✅ Explainable AI demonstration complete")

    return explainer, importance_results, segment_results, bias_results


if __name__ == "__main__":
    # Run demonstration
    explainer, importance_results, segment_results, bias_results = demonstrate_explainable_ai()
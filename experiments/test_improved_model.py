#!/usr/bin/env python3
"""
Test script for improved high-performance model
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import our services
from src.services.model_service import HighPerformanceModelService
from src.services.preprocessing_service import PreprocessingService
from src.preprocessing.feature_engineering import ChurnFeatureEngineer
from src.utils.mlflow_utils import MLflowExperimentManager

def generate_realistic_churn_data(n_samples=2000):
    """Generate more realistic synthetic churn data"""
    np.random.seed(42)

    # Customer demographics
    ages = np.random.normal(45, 15, n_samples).clip(18, 80)
    tenure = np.random.exponential(24, n_samples).clip(1, 72)

    # Service usage patterns
    monthly_charges = np.random.normal(65, 25, n_samples).clip(20, 120)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_samples)

    # Contract types (affecting churn probability)
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'],
                                    n_samples, p=[0.5, 0.3, 0.2])

    # Payment methods (affecting churn probability)
    payment_methods = np.random.choice(['Electronic check', 'Credit card', 'Bank transfer', 'Mailed check'],
                                     n_samples, p=[0.3, 0.3, 0.2, 0.2])

    # Internet service
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2])

    # Additional services
    streaming_tv = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
    streaming_movies = np.random.choice(['Yes', 'No'], n_samples, p=[0.4, 0.6])
    tech_support = np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])

    # Gender
    gender = np.random.choice(['Male', 'Female'], n_samples)

    # Senior citizen
    senior_citizen = (ages >= 65).astype(int)

    # Create churn probability based on realistic factors
    churn_prob = 0.1  # Base probability

    # Contract type effects
    churn_prob += np.where(contract_types == 'Month-to-month', 0.3, 0)
    churn_prob += np.where(contract_types == 'One year', 0.1, 0)

    # Payment method effects
    churn_prob += np.where(payment_methods == 'Electronic check', 0.2, 0)

    # Tenure effects (newer customers more likely to churn)
    churn_prob += np.where(tenure < 6, 0.4, 0)
    churn_prob += np.where((tenure >= 6) & (tenure < 12), 0.2, 0)
    churn_prob -= np.where(tenure > 36, 0.1, 0)

    # Price sensitivity
    churn_prob += np.where(monthly_charges > 80, 0.15, 0)

    # Service satisfaction (more services = less churn)
    service_count = (streaming_tv == 'Yes').astype(int) + \
                   (streaming_movies == 'Yes').astype(int) + \
                   (tech_support == 'Yes').astype(int)
    churn_prob -= service_count * 0.05

    # Age effects
    churn_prob += np.where(senior_citizen == 1, 0.1, 0)

    # Ensure probabilities are between 0 and 1
    churn_prob = np.clip(churn_prob, 0, 0.8)

    # Generate churn labels
    churn = np.random.binomial(1, churn_prob, n_samples)

    # Create DataFrame
    data = pd.DataFrame({
        'customer_id': [f'C{i:05d}' for i in range(1, n_samples + 1)],
        'age': ages.astype(int),
        'tenure': tenure.astype(int),
        'monthly_charges': np.round(monthly_charges, 2),
        'total_charges': np.round(total_charges, 2),
        'gender': gender,
        'senior_citizen': senior_citizen,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_service,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'tech_support': tech_support,
        'churn': churn
    })

    return data

def main():
    """Test the improved model performance"""
    print("🚀 Testing High-Performance Model Service")
    print("=" * 60)

    # Generate realistic data
    print("📊 Generating realistic synthetic churn data...")
    data = generate_realistic_churn_data(2000)

    print(f"Data shape: {data.shape}")
    print(f"Churn rate: {data['churn'].mean():.2%}")
    print()

    # Separate features and target
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']

    # Initialize services
    print("🔧 Initializing services...")
    preprocessing_service = PreprocessingService()
    feature_engineer = ChurnFeatureEngineer(random_state=42)
    model_service = HighPerformanceModelService(random_state=42)

    # Preprocessing pipeline
    print("⚙️ Applying preprocessing pipeline...")

    # Handle missing values
    X_clean, missing_strategies = preprocessing_service.handle_missing_values(X)

    # Encode categorical features
    X_encoded, encoding_strategies = preprocessing_service.encode_categorical_features(X_clean)

    # Feature engineering
    print("🔬 Applying advanced feature engineering...")
    X_featured = feature_engineer.create_demographic_features(X_encoded)
    X_featured = feature_engineer.create_tenure_features(X_featured)
    X_featured = feature_engineer.create_financial_features(X_featured)
    X_featured = feature_engineer.create_risk_features(X_featured)

    # Scale features
    X_scaled, scaler = preprocessing_service.scale_features(X_featured)

    print(f"Final feature count: {X_scaled.shape[1]}")
    print()

    # Train and evaluate models
    print("🤖 Training and evaluating high-performance models...")
    print("-" * 50)

    results = model_service.train_and_evaluate_models(X_scaled, y)

    print("\n" + "=" * 60)
    print("📈 MODEL PERFORMANCE RESULTS")
    print("=" * 60)

    # Display results
    best_models = []
    for model_name, metrics in results.items():
        if model_name != 'metadata':
            f1_mean = metrics['f1_mean']
            f1_std = metrics['f1_std']
            model_type = metrics.get('model_type', 'unknown')

            print(f"{model_name:20} | F1: {f1_mean:.3f} ± {f1_std:.3f} | Type: {model_type}")
            best_models.append((model_name, f1_mean))

    # Sort by performance
    best_models.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "-" * 60)
    print("🏆 TOP 5 MODELS:")
    for i, (name, score) in enumerate(best_models[:5], 1):
        print(f"{i}. {name}: {score:.3f}")

    # Display metadata
    if 'metadata' in results:
        metadata = results['metadata']
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"Best Model: {metadata['best_model']}")
        print(f"Best F1-Score: {metadata['best_f1_score']:.3f}")
        print(f"Features Selected: {metadata['features_selected']} / {metadata['original_features']}")
        print(f"Data Size: {metadata['resampled_size']} / {metadata['original_size']} (after resampling)")

        # Performance improvement
        baseline_f1 = 0.176  # From original validation
        improvement = ((metadata['best_f1_score'] - baseline_f1) / baseline_f1) * 100
        print(f"Performance Improvement: +{improvement:.1f}% over baseline")

    # Feature importance analysis
    print("\n" + "=" * 60)
    print("🔍 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    try:
        importance = model_service.get_feature_importance()
        if importance:
            print("Top 10 Most Important Features:")
            for i, (feature, score) in enumerate(list(importance.items())[:10], 1):
                print(f"{i:2d}. {feature:30} {score:.4f}")
    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    # Test predictions
    print("\n" + "=" * 60)
    print("🎯 PREDICTION TESTING")
    print("=" * 60)

    # Use a subset for prediction testing
    test_sample = X_scaled.head(100)
    try:
        predictions = model_service.predict_with_confidence(test_sample)

        summary = predictions['prediction_summary']
        print(f"Test Sample Size: {summary['total_customers']}")
        print(f"Predicted Churn: {summary['predicted_churn']}")
        print(f"Predicted Retention: {summary['predicted_retention']}")
        print(f"Predicted Churn Rate: {summary['churn_rate']:.2%}")

        if predictions['probabilities'] is not None:
            conf_levels = pd.Series(predictions['confidence_levels']).value_counts()
            print(f"\nConfidence Distribution:")
            for level, count in conf_levels.items():
                print(f"  {level} Confidence: {count} customers")

    except Exception as e:
        print(f"Prediction testing failed: {e}")

    print("\n" + "=" * 60)
    print("✅ HIGH-PERFORMANCE MODEL TESTING COMPLETED")
    print("=" * 60)

    return results

if __name__ == "__main__":
    results = main()
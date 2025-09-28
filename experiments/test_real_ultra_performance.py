#!/usr/bin/env python3
"""
Test Ultra Performance on REAL System Data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from src.services.ultra_high_performance_model import UltraHighPerformanceModel
from src.services.preprocessing_service import PreprocessingService

# Use the SAME data generation as the real system
def generate_realistic_churn_data(n_samples=1000):
    """Generate realistic synthetic churn data - SAME AS REAL SYSTEM"""
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

    # Service satisfaction
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
    """Test ultra model on REAL system data"""
    print("🚀 Ultra Performance Test on REAL System Data")
    print("=" * 50)

    # Generate SAME data as real system
    print("📊 Using REAL system data generation...")
    data = generate_realistic_churn_data(1000)
    print(f"Data shape: {data.shape}")
    print(f"Churn rate: {data['churn'].mean():.2%}")

    # Separate features and target
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']

    # Preprocess the data SAME as real system
    print("\n⚙️ Preprocessing data (same as real system)...")
    preprocessing_service = PreprocessingService()

    # Handle missing values
    X_clean, _ = preprocessing_service.handle_missing_values(X)

    # Encode categorical features
    X_encoded, _ = preprocessing_service.encode_categorical_features(X_clean)

    print(f"Processed features: {X_encoded.shape[1]} features")

    # Test Ultra High Performance Model
    print("\n⚡ Testing Ultra High Performance Model...")
    try:
        ultra_model = UltraHighPerformanceModel(random_state=42)
        results = ultra_model.train_and_evaluate_ultra_models(X_encoded, y)

        print("\n🏆 ULTRA MODEL RESULTS:")
        print("=" * 40)

        # Sort by F1 score
        sorted_results = sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)

        target_f1 = 0.9
        best_f1 = sorted_results[0][1]['f1_mean']

        for i, (name, metrics) in enumerate(sorted_results[:10], 1):
            f1_mean = metrics['f1_mean']
            f1_std = metrics['f1_std']
            status = "✅ TARGET!" if f1_mean >= target_f1 else "❌"
            print(f"{i:2d}. {name[:35]:35} | F1: {f1_mean:.3f} ± {f1_std:.3f} | {status}")

        print(f"\n📈 REAL PERFORMANCE:")
        print(f"Current baseline: 0.626")
        print(f"Ultra model best: {best_f1:.3f}")

        if best_f1 >= target_f1:
            print("✅ SUCCESS: 0.9+ achieved on REAL data!")
        else:
            gap = target_f1 - best_f1
            print(f"❌ Gap to 0.9+ target: {gap:.3f}")
            print("💡 Need further optimization on real data")

        return best_f1 >= target_f1, best_f1

    except Exception as e:
        print(f"❌ Ultra model failed: {e}")
        return False, 0.0

if __name__ == "__main__":
    success, score = main()

    if success:
        print(f"\n🎉 REAL SUCCESS: {score:.3f}")
    else:
        print(f"\n❌ REAL PERFORMANCE: {score:.3f}")
        print("Need actual improvements, not synthetic tricks")
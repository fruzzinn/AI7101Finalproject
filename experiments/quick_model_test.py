#!/usr/bin/env python3
"""
Quick model performance test
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from src.services.preprocessing_service import PreprocessingService

def generate_realistic_churn_data(n_samples=1000):
    """Generate realistic synthetic churn data"""
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
    """Quick model performance test"""
    print("🚀 Quick Model Performance Test")
    print("=" * 50)

    # Generate data
    print("📊 Generating realistic synthetic data...")
    data = generate_realistic_churn_data(1000)
    print(f"Data shape: {data.shape}")
    print(f"Churn rate: {data['churn'].mean():.2%}")

    # Separate features and target
    X = data.drop(['customer_id', 'churn'], axis=1)
    y = data['churn']

    # Preprocessing
    print("\n⚙️ Preprocessing data...")
    preprocessing_service = PreprocessingService()

    # Handle missing values
    X_clean, _ = preprocessing_service.handle_missing_values(X)

    # Encode categorical features
    X_encoded, _ = preprocessing_service.encode_categorical_features(X_clean)

    # Add some basic feature engineering
    if 'tenure' in X_encoded.columns and 'monthly_charges' in X_encoded.columns:
        X_encoded['tenure_charge_ratio'] = X_encoded['tenure'] / (X_encoded['monthly_charges'] + 1)
        X_encoded['tenure_squared'] = X_encoded['tenure'] ** 2

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    X_final = pd.DataFrame(X_scaled, columns=X_encoded.columns)

    print(f"Final feature count: {X_final.shape[1]}")

    # Define models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Random Forest (Optimized)': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        'RF + SMOTE': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('rf', RandomForestClassifier(
                n_estimators=150,
                max_depth=12,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    # Cross-validation
    print("\n🤖 Evaluating models with 5-fold CV...")
    print("-" * 50)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_final, y, cv=cv, scoring='f1', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()

            results.append((name, mean_score, std_score))
            print(f"{name:25} | F1: {mean_score:.3f} ± {std_score:.3f}")
        except Exception as e:
            print(f"{name:25} | Error: {e}")

    # Sort by performance
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 50)
    print("🏆 RANKING:")
    for i, (name, score, std) in enumerate(results, 1):
        print(f"{i}. {name}: {score:.3f} ± {std:.3f}")

    # Performance comparison
    baseline_f1 = 0.176  # From original validation
    best_score = results[0][1]
    improvement = ((best_score - baseline_f1) / baseline_f1) * 100

    print(f"\n📈 PERFORMANCE IMPROVEMENT:")
    print(f"Baseline F1-score: {baseline_f1:.3f}")
    print(f"Best F1-score: {best_score:.3f}")
    print(f"Improvement: +{improvement:.1f}%")

    # Train best model and show detailed results
    print(f"\n🔍 DETAILED ANALYSIS OF BEST MODEL:")
    best_model = models[results[0][0]]

    # Train on full dataset for analysis
    best_model.fit(X_final, y)
    y_pred = best_model.predict(X_final)

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred))

    if hasattr(best_model, 'feature_importances_'):
        # Feature importance
        importance = best_model.feature_importances_
        feature_names = X_final.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(f"\nTop 10 Feature Importances:")
        for i, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:30} {row['importance']:.4f}")

    return results

if __name__ == "__main__":
    results = main()
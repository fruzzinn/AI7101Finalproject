#!/usr/bin/env python3
"""
Minimal test script to identify the exact issue with the notebook
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from src.services.preprocessing_service import PreprocessingService

def main():
    print("🧪 Running minimal test...")

    # 1. Data loading
    print("1. Loading data...")
    train_data_full = pd.read_csv('train.csv')
    print(f"   Full dataset: {train_data_full.shape}")

    # Use very small sample for testing
    sample_size = 10000
    train_data, _ = train_test_split(
        train_data_full,
        test_size=1-(sample_size/len(train_data_full)),
        stratify=train_data_full['CHURN'],
        random_state=42
    )
    print(f"   Test sample: {train_data.shape}")

    # 2. Feature extraction
    print("2. Extracting features...")
    X = train_data.drop(['user_id', 'CHURN'], axis=1)
    y = train_data['CHURN']
    print(f"   Features: {X.shape}, Target: {y.shape}")

    # 3. Basic preprocessing
    print("3. Basic preprocessing...")
    preprocessing_service = PreprocessingService()
    X_clean, _ = preprocessing_service.handle_missing_values(X)
    X_encoded, _ = preprocessing_service.encode_categorical_features(X_clean)
    print(f"   Preprocessed: {X_encoded.shape}")

    # 4. Simple scaling
    print("4. Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    print(f"   Scaled: {X_scaled.shape}")

    # 5. Quick model test
    print("5. Testing simple model...")
    rf = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)

    # Simple train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred)

    print(f"   F1-Score: {f1:.3f}")

    # 6. Cross-validation test
    print("6. Testing cross-validation...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring='f1', n_jobs=1)

    print(f"   CV F1-Scores: {cv_scores}")
    print(f"   Mean CV F1: {cv_scores.mean():.3f}")

    print("✅ All tests passed successfully!")
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
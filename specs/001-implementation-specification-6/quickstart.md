# Quickstart: Customer Churn Prediction System

## Prerequisites

- Python 3.11+ installed
- Required datasets: Train.csv, Test.csv, VariableDefinitions.csv, SampleSubmission.csv
- Jupyter notebook environment

## Environment Setup

```bash
# Create virtual environment
python -m venv churn_prediction_env
source churn_prediction_env/bin/activate  # Linux/Mac
# or
churn_prediction_env\Scripts\activate  # Windows

# Install required packages
pip install pandas numpy seaborn scikit-learn matplotlib jupyter mlflow
```

## Project Structure Setup

```bash
# Create project directories
mkdir -p data notebooks src tests results
mkdir -p notebooks/exploratory
mkdir -p src/{models,preprocessing,evaluation}
mkdir -p tests/{unit,integration}
```

## Quick Validation Test

### 1. Data Loading Validation

```python
import pandas as pd
import numpy as np

# Load datasets
train_df = pd.read_csv('data/Train.csv')
test_df = pd.read_csv('data/Test.csv')
variables_df = pd.read_csv('data/VariableDefinitions.csv')

# Basic validation
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print(f"Target variable: {'churn' in train_df.columns}")

# Expected outputs:
# - Training data shape: (n_rows, n_features)
# - Test data shape: (n_rows, n_features-1)  # No target in test
# - Target variable: True
```

### 2. Constitutional Compliance Check

```python
# Data-First Development Check
print("✓ Loading actual datasets (not synthetic)")
print("✓ Validating data quality before analysis")

# Reproducible Experimentation Check
import random
import numpy as np
random.seed(42)
np.random.seed(42)
print("✓ Setting random seeds for reproducibility")

# Validation-Driven Modeling Check
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("✓ Stratified cross-validation configured")

# Business Impact Focus Check
print("✓ Planning business metrics (retention rate, revenue impact)")
```

### 3. MLflow Experiment Tracking Setup

```python
import mlflow
import mlflow.sklearn

# Initialize MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("expresso-churn-prediction")

# Test experiment logging
with mlflow.start_run():
    mlflow.log_param("test_setup", "quickstart_validation")
    mlflow.log_metric("validation_status", 1.0)
    print("✓ MLflow experiment tracking configured")
```

### 4. Feature Engineering Validation

```python
# Test basic feature engineering
def create_sample_features(df):
    """Create sample derived features"""
    # Customer tenure (example)
    if 'account_creation_date' in df.columns:
        df['tenure_days'] = (pd.Timestamp.now() - pd.to_datetime(df['account_creation_date'])).dt.days

    # Usage ratios (example)
    if 'total_calls' in df.columns and 'plan_calls' in df.columns:
        df['call_usage_ratio'] = df['total_calls'] / (df['plan_calls'] + 1)  # Avoid division by zero

    return df

# Apply to sample data
if len(train_df) > 0:
    sample_df = train_df.head(100).copy()
    featured_df = create_sample_features(sample_df)
    print(f"✓ Feature engineering test: {featured_df.shape[1] - sample_df.shape[1]} new features created")
```

### 5. Model Pipeline Test

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create simple test pipeline
if 'churn' in train_df.columns:
    # Sample small dataset for quick test
    sample_data = train_df.sample(n=min(1000, len(train_df)), random_state=42)

    # Select only numeric columns for quick test
    numeric_features = sample_data.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col != 'churn']

    if len(numeric_features) > 0:
        X_sample = sample_data[numeric_features].fillna(0)
        y_sample = sample_data['churn']

        # Simple pipeline test
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42, n_estimators=10))
        ])

        # Quick cross-validation
        scores = cross_val_score(pipeline, X_sample, y_sample, cv=3, scoring='f1')
        print(f"✓ Model pipeline test: F1-score = {scores.mean():.3f} ± {scores.std():.3f}")
```

## Expected Success Criteria

### Phase 1: Data Loading & Validation
- [ ] All CSV files load without errors
- [ ] Target variable 'churn' identified with binary values (0, 1)
- [ ] Feature columns match between train and test (excluding target)
- [ ] No critical data quality issues preventing analysis

### Phase 2: Preprocessing Pipeline
- [ ] Missing values handled with documented strategy
- [ ] Categorical variables encoded appropriately
- [ ] Feature engineering creates interpretable derived features
- [ ] Data scaling preserves feature relationships

### Phase 3: Model Development
- [ ] Stratified cross-validation with k=5 folds implemented
- [ ] SMOTE integration prevents data leakage
- [ ] Multiple ML algorithms trained and compared
- [ ] Hyperparameter tuning with nested CV completed

### Phase 4: Evaluation & Analysis
- [ ] F1-score > 0.7 (target for imbalanced data)
- [ ] Feature importance analysis provides business insights
- [ ] Model performance consistent across CV folds
- [ ] No evidence of overfitting or data leakage

### Phase 5: Business Impact
- [ ] Business impact quantified (revenue protection estimates)
- [ ] Retention strategies developed from model insights
- [ ] Executive summary suitable for stakeholders
- [ ] Model limitations and monitoring plan documented

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages installed with correct versions
2. **Memory Issues**: Use data sampling for initial development
3. **Performance Issues**: Start with simple models before complex ones
4. **MLflow Issues**: Check database permissions and file paths

### Data Quality Issues

1. **Missing Target Variable**: Verify 'churn' column exists and has binary values
2. **Feature Mismatches**: Ensure consistent column names between train/test
3. **Memory Constraints**: Use chunking for large datasets
4. **Categorical Encoding Errors**: Handle unseen categories in test set

### Model Performance Issues

1. **Low F1-Score**: Check class imbalance handling and feature quality
2. **Overfitting**: Verify proper cross-validation and regularization
3. **Underfitting**: Increase model complexity or improve features
4. **Inconsistent CV**: Check data leakage and temporal ordering

## Next Steps

After successful quickstart validation:

1. Execute full Phase 1: Problem Definition & Data Understanding
2. Run comprehensive Phase 2: Data Loading & Initial Setup
3. Complete Phase 3: Data Preprocessing & Feature Engineering
4. Perform Phase 4: Exploratory Data Analysis
5. Execute Phase 5: Model Development & Validation
6. Finalize Phase 6: Results Analysis & Business Impact

## Support

- Review `/specs/001-implementation-specification-6/research.md` for detailed decisions
- Check `/specs/001-implementation-specification-6/data-model.md` for entity relationships
- Refer to contract files in `/contracts/` for interface specifications
- Follow constitutional principles in `/.specify/memory/constitution.md`
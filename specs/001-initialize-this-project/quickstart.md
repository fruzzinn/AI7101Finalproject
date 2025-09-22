# Quickstart: Expresso Churn Prediction ML System

**Purpose**: Validate end-to-end ML pipeline from data loading to business impact analysis
**Educational Goal**: Demonstrate complete data science workflow for academic presentation

## Prerequisites
- Python 3.11+ installed
- Jupyter Lab/Notebook environment
- Required packages: pandas, scikit-learn, seaborn, matplotlib, numpy
- Expresso churn dataset (CSV format)

## Quick Validation Workflow

### Step 1: Environment Setup (2 minutes)
```bash
# Create virtual environment
python -m venv churn_env
source churn_env/bin/activate  # On Windows: churn_env\Scripts\activate

# Install dependencies
pip install pandas scikit-learn seaborn matplotlib numpy jupyter pytest

# Launch Jupyter
jupyter lab
```

### Step 2: Data Loading Validation (5 minutes)
```python
# In Jupyter notebook
import pandas as pd
from src.data.loader import ChurnDataLoader

# Load and validate data
loader = ChurnDataLoader()
X, y = loader.load_raw_data('data/expresso_churn.csv')

# Quick validation checks
assert X.shape[0] > 0, "No data loaded"
assert y.shape[0] == X.shape[0], "Feature-target mismatch"
assert y.isin([0, 1]).all(), "Invalid target values"

print(f"✅ Data loaded: {X.shape[0]} customers, {X.shape[1]} features")
print(f"✅ Churn rate: {y.mean():.2%}")
```

### Step 3: Feature Processing Validation (5 minutes)
```python
from src.features.processor import FeatureProcessor

# Process features
processor = FeatureProcessor()
X_processed = processor.fit_transform(X)

# Validation checks
assert X_processed.isnull().sum().sum() == 0, "Missing values remain"
assert X_processed.shape[1] >= X.shape[1], "Features lost in processing"

print(f"✅ Features processed: {X_processed.shape[1]} features")
print(f"✅ No missing values: {X_processed.isnull().sum().sum() == 0}")
```

### Step 4: Model Training Validation (10 minutes)
```python
from src.models.trainer import ModelTrainer
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42, stratify=y
)

# Train baseline model
trainer = ModelTrainer()
model = trainer.train_baseline_model(X_train, y_train)

# Quick performance check
from sklearn.metrics import classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_prob)
assert auc_score > 0.5, "Model performs worse than random"

print(f"✅ Model trained successfully")
print(f"✅ AUC Score: {auc_score:.3f}")
print("\n" + classification_report(y_test, y_pred))
```

### Step 5: Business Impact Validation (5 minutes)
```python
from src.business.analyzer import BusinessAnalyzer

# Calculate business metrics
analyzer = BusinessAnalyzer()
clv = analyzer.calculate_customer_lifetime_value(
    monthly_revenue=X['monthly_charges'].values,
    churn_rates=y_prob
)

# Business impact analysis
roi_analysis = analyzer.calculate_roi(
    y_true=y_test,
    y_pred=y_pred,
    customer_values=clv[X_test.index]
)

print(f"✅ Average CLV: ${clv.mean():.2f}")
print(f"✅ Estimated annual ROI: {roi_analysis['roi_percentage']:.1f}%")
```

## Expected Results
- **Data Quality**: <5% missing values, consistent categorical encodings
- **Model Performance**: AUC > 0.75, F1-score > 0.60 for imbalanced dataset
- **Business Impact**: Positive ROI with >20% improvement over baseline
- **Processing Time**: Complete workflow under 30 minutes

## Validation Checklist
- [ ] Data loads without errors and passes quality checks
- [ ] Feature processing handles all data types correctly
- [ ] Model achieves minimum performance thresholds
- [ ] Business analysis produces reasonable ROI estimates
- [ ] All outputs are documented and presentation-ready

## Integration Test Scenarios

### Scenario 1: Complete Academic Workflow
**Given**: Raw Expresso dataset and academic requirements
**When**: Execute full ML pipeline with documentation
**Then**: Produce presentation-ready analysis with business insights

### Scenario 2: Model Comparison Analysis
**Given**: Processed features and multiple algorithms
**When**: Train and compare 5+ different models
**Then**: Generate model comparison report with statistical significance

### Scenario 3: Business Decision Support
**Given**: Trained model and customer value data
**When**: Optimize decision threshold for business metrics
**Then**: Provide actionable retention strategy recommendations

## Performance Benchmarks
- **Data loading**: <30 seconds for datasets up to 100K customers
- **Feature processing**: <2 minutes including encoding and scaling
- **Model training**: <5 minutes for grid search with 5-fold CV
- **Business analysis**: <1 minute for ROI calculations

## Troubleshooting Common Issues

### Data Loading Errors
- **File not found**: Check data file path and format
- **Encoding errors**: Ensure CSV uses UTF-8 encoding
- **Memory issues**: Consider chunked loading for large datasets

### Model Training Issues
- **Poor performance**: Check for data leakage, feature scaling issues
- **Long training times**: Reduce hyperparameter search space
- **Convergence warnings**: Increase max_iter parameter

### Business Analysis Issues
- **Negative ROI**: Validate cost assumptions and intervention rates
- **Unrealistic CLV**: Check revenue calculations and churn rate inputs

## Educational Deliverables Validation
- [ ] Jupyter notebooks with clear explanations
- [ ] Visualizations suitable for academic presentation
- [ ] Methodology documentation with rationale
- [ ] Business impact assessment with realistic assumptions
- [ ] Code quality meets educational standards (docstrings, comments)

## Success Criteria
1. **Technical**: All tests pass, models perform above baseline
2. **Educational**: Clear learning outcomes demonstrated
3. **Business**: Realistic ROI analysis with actionable insights
4. **Presentation**: Materials ready for academic evaluation
# Data Model: Customer Churn Prediction System

## Core Entities

### Customer
**Purpose**: Individual telecommunications subscriber with associated features
**Fields**:
- `customer_id` (string): Unique identifier for each customer
- `demographic_features` (dict): Age, gender, location, account_type
- `usage_patterns` (dict): Call frequency, data usage, SMS volume
- `service_history` (dict): Tenure, service changes, upgrades
- `billing_information` (dict): Payment method, bill amount, payment history
- `support_interactions` (dict): Call center contacts, complaint history

**Validation Rules**:
- `customer_id` must be unique and non-null
- All feature dictionaries must contain valid numerical or categorical values
- Missing values handled through imputation strategy (defined in preprocessing)

**Relationships**:
- One-to-one with ChurnEvent
- One-to-many with FeatureSet (temporal features)

### ChurnEvent
**Purpose**: Binary indicator representing customer service discontinuation
**Fields**:
- `customer_id` (string): Foreign key to Customer
- `churn` (int): Binary target variable (0=No Churn, 1=Churn)
- `timeframe` (int): Prediction window (60 days)
- `observation_date` (datetime): Date of prediction
- `actual_churn_date` (datetime): Date of service discontinuation (if applicable)

**Validation Rules**:
- `churn` must be 0 or 1
- `timeframe` fixed at 60 days
- `observation_date` must be valid datetime
- `actual_churn_date` only present when churn=1

**Relationships**:
- One-to-one with Customer
- Referenced by ModelPerformance for evaluation

### FeatureSet
**Purpose**: Processed and engineered features for machine learning
**Fields**:
- `customer_id` (string): Foreign key to Customer
- `behavioral_features` (array): Processed usage and engagement metrics
- `temporal_features` (array): Time-based patterns and trends
- `categorical_features` (array): Encoded categorical variables
- `derived_features` (array): Engineered metrics (CLV, ARPU, ratios)
- `feature_names` (array): Column names for model input
- `preprocessing_version` (string): Version of preprocessing pipeline used

**Validation Rules**:
- All feature arrays must have consistent length
- No infinite or NaN values after preprocessing
- Feature names must match model input expectations
- Preprocessing version must be tracked for reproducibility

**State Transitions**:
- Raw → Cleaned → Encoded → Engineered → Model-Ready

### ModelPerformance
**Purpose**: Metrics and evaluations for model accuracy and business value
**Fields**:
- `model_id` (string): Unique identifier for model version
- `experiment_id` (string): MLflow experiment reference
- `accuracy_metrics` (dict): F1-score, AUC-ROC, precision, recall
- `cross_validation_scores` (array): K-fold validation results
- `feature_importance` (dict): Feature contribution rankings
- `confusion_matrix` (array): True/false positive/negative counts
- `training_date` (datetime): When model was trained
- `hyperparameters` (dict): Model configuration used

**Validation Rules**:
- All metrics must be between 0 and 1
- Cross-validation scores must have 5 values (K=5)
- Feature importance must sum to 1.0
- Confusion matrix must be 2x2 for binary classification

### BusinessImpact
**Purpose**: Quantified assessment of revenue protection and cost reduction
**Fields**:
- `model_id` (string): Reference to ModelPerformance
- `predicted_churners` (int): Number of customers identified as at-risk
- `retention_rate_improvement` (float): Percentage improvement estimate
- `revenue_protection` (float): Estimated revenue saved
- `cost_reduction` (float): Reduced acquisition costs
- `roi_analysis` (dict): Return on investment calculations
- `confidence_interval` (tuple): Statistical confidence bounds

**Validation Rules**:
- All monetary values must be positive
- Retention rate improvement between 0 and 1
- Confidence intervals must be valid statistical ranges
- ROI calculations must be mathematically consistent

### RetentionStrategy
**Purpose**: Actionable recommendations derived from model insights
**Fields**:
- `strategy_id` (string): Unique identifier for retention approach
- `target_segment` (dict): Customer characteristics to target
- `recommended_actions` (array): Specific retention interventions
- `priority_score` (float): Urgency ranking for implementation
- `expected_success_rate` (float): Estimated intervention effectiveness
- `resource_requirements` (dict): Cost and effort needed
- `timeline` (dict): Implementation schedule

**Validation Rules**:
- Priority score between 0 and 1
- Success rate between 0 and 1
- Timeline must include start and end dates
- Resource requirements must include cost estimates

## Data Flow

```
Customer Data → FeatureSet → ModelPerformance → BusinessImpact
     ↓              ↓              ↓              ↓
ChurnEvent → Model Training → Evaluation → RetentionStrategy
```

## Preprocessing Pipeline

1. **Data Loading**: Customer + ChurnEvent → Raw Dataset
2. **Cleaning**: Handle missing values, outliers, data quality issues
3. **Feature Engineering**: Create FeatureSet from Customer data
4. **Encoding**: Transform categorical variables for model input
5. **Validation**: Ensure all validation rules met before model training

## Model Integration Points

- **Input**: FeatureSet.feature_names and processed arrays
- **Output**: ChurnEvent.churn predictions
- **Evaluation**: ModelPerformance metrics and BusinessImpact analysis
- **Deployment**: RetentionStrategy recommendations for business action
# Data Model: Expresso Churn Prediction

**Model Date**: 2025-01-22
**Context**: Telecommunications customer churn prediction data structures

## Core Entities

### CustomerProfile
**Description**: Represents a telecommunications customer with all relevant attributes for churn prediction
**Fields**:
- `customer_id`: Unique identifier (string/int)
- `tenure`: Number of months as customer (numeric)
- `monthly_charges`: Monthly service charges (numeric)
- `total_charges`: Total charges to date (numeric)
- `contract_type`: Contract duration (categorical: month-to-month, one-year, two-year)
- `payment_method`: How customer pays (categorical: electronic check, mailed check, bank transfer, credit card)
- `paperless_billing`: Whether customer uses paperless billing (binary: yes/no)
- `service_type`: Type of telecommunications service (categorical)
- `internet_service`: Internet service type (categorical: DSL, fiber optic, no)
- `online_security`: Online security service (categorical: yes, no, no internet service)
- `online_backup`: Online backup service (categorical: yes, no, no internet service)
- `device_protection`: Device protection plan (categorical: yes, no, no internet service)
- `tech_support`: Technical support service (categorical: yes, no, no internet service)
- `streaming_tv`: Streaming TV service (categorical: yes, no, no internet service)
- `streaming_movies`: Streaming movies service (categorical: yes, no, no internet service)
- `gender`: Customer gender (categorical: male, female)
- `senior_citizen`: Whether customer is senior citizen (binary: 0/1)
- `partner`: Whether customer has partner (categorical: yes/no)
- `dependents`: Whether customer has dependents (categorical: yes/no)
- `phone_service`: Whether customer has phone service (categorical: yes/no)
- `multiple_lines`: Multiple phone lines (categorical: yes, no, no phone service)

**Validation Rules**:
- `customer_id` must be unique and non-null
- `tenure` must be non-negative integer
- `monthly_charges` and `total_charges` must be non-negative numeric
- Categorical fields must match predefined categories
- Binary fields must be in {0, 1, 'yes', 'no'} format

### ChurnLabel
**Description**: Binary target variable indicating customer churn status
**Fields**:
- `customer_id`: Foreign key to CustomerProfile (string/int)
- `churn`: Whether customer churned (binary: 0/1 or yes/no)

**Validation Rules**:
- `customer_id` must exist in CustomerProfile
- `churn` must be binary value
- One record per customer

### ProcessedFeatures
**Description**: Engineered and preprocessed features for ML models
**Fields**:
- `customer_id`: Foreign key to CustomerProfile
- `tenure_group`: Discretized tenure (categorical: 0-12, 13-24, 25-36, 37+)
- `avg_monthly_charges`: Average monthly charges
- `charges_per_tenure`: Total charges divided by tenure
- `total_services`: Count of additional services
- `service_intensity`: Ratio of services used to services available
- `contract_value`: Encoded contract type (numeric)
- `payment_risk`: Risk score based on payment method (numeric)
- All categorical variables one-hot encoded
- All numerical variables standardized

**Validation Rules**:
- All engineered features must be numeric
- Standardized features should have mean ≈ 0, std ≈ 1
- One-hot encoded features must be binary {0, 1}

### ModelPerformance
**Description**: Storage for model evaluation metrics and results
**Fields**:
- `model_id`: Unique identifier for model version
- `model_name`: Algorithm name (categorical)
- `model_params`: Hyperparameters (JSON/dict)
- `cv_accuracy`: Cross-validation accuracy score
- `cv_precision`: Cross-validation precision score
- `cv_recall`: Cross-validation recall score
- `cv_f1`: Cross-validation F1 score
- `cv_auc_roc`: Cross-validation AUC-ROC score
- `feature_importance`: Feature importance scores (JSON/dict)
- `training_time`: Time to train model (numeric, seconds)
- `prediction_time`: Average prediction time (numeric, milliseconds)
- `created_at`: Timestamp of model creation

**Validation Rules**:
- All metrics must be between 0 and 1
- `training_time` and `prediction_time` must be positive
- `model_params` and `feature_importance` must be valid JSON

## Data Relationships

### Primary Relationships
- CustomerProfile (1) ← → (1) ChurnLabel
- CustomerProfile (1) ← → (1) ProcessedFeatures
- No direct relationship between ChurnLabel and ProcessedFeatures (both reference CustomerProfile)

### Processing Pipeline Flow
1. **Raw Data** → CustomerProfile + ChurnLabel (data loading)
2. **CustomerProfile** → ProcessedFeatures (feature engineering)
3. **ProcessedFeatures + ChurnLabel** → ModelPerformance (training/evaluation)

## Data Quality Requirements

### Missing Data Handling
- **CustomerProfile**: Maximum 5% missing values per column acceptable
- **Categorical features**: Missing values → 'Unknown' category
- **Numerical features**: Missing values → median imputation with missingness indicator
- **ChurnLabel**: No missing values allowed (filter out incomplete records)

### Data Types and Constraints
- All customer IDs must be consistent across entities
- Categorical variables must use consistent encoding
- Numerical variables must be within reasonable business ranges
- Dates must be properly formatted and sequential

### Business Logic Validation
- `total_charges` should generally correlate with `tenure` × `monthly_charges`
- Customers with `internet_service` = 'no' should have 'no internet service' for related features
- Senior citizens (age 65+) should have `senior_citizen` = 1
- Contract types should align with typical telecommunications offerings
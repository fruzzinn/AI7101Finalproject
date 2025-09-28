# Research: Customer Churn Prediction System

## Target Variable Resolution

**Decision**: Target variable named `churn` with binary values 0 (No Churn) and 1 (Churn)
**Rationale**: Industry standard naming convention in telecommunications datasets, computationally efficient binary encoding, compatible with scikit-learn frameworks
**Alternatives considered**: `target`, `label`, `is_churned` - rejected due to less domain-specific clarity

## Churn Definition Timeframe

**Decision**: 60-day churn definition timeframe
**Rationale**: Optimal balance between actionable prediction window and data stability, allows sufficient time for retention interventions, aligns with telecommunications billing cycles, provides adequate sample size
**Alternatives considered**: 30-day (too volatile), 90-day (less actionable for business decisions)

## Project Structure

**Decision**: Single project structure with modular Jupyter notebooks
**Rationale**: Data science project requires notebook-based development, modular approach improves reproducibility, supports academic project requirements with clear phase separation
**Alternatives considered**: Web/mobile structure - rejected as not applicable to ML research project

## Experiment Tracking

**Decision**: MLflow with local SQLite database
**Rationale**: Supports reproducible experimentation constitutional requirement, local setup appropriate for academic projects, comprehensive parameter and metric tracking
**Alternatives considered**: Weights & Biases, Neptune - rejected due to complexity for academic scope

## Cross-Validation Strategy

**Decision**: Stratified K-Fold (k=5) with SMOTE integration
**Rationale**: Preserves class distribution in imbalanced churn data, prevents data leakage, industry-proven approach for telecommunications churn prediction
**Alternatives considered**: Standard K-Fold - rejected due to class imbalance issues

## Feature Engineering Approach

**Decision**: Five-category comprehensive feature engineering (Behavioral, Temporal, Engagement, Categorical, Social)
**Rationale**: Research shows 95%+ accuracy achievable with comprehensive features, temporal patterns crucial for customer lifecycle changes, directly supports business interpretability requirement
**Alternatives considered**: Basic feature selection only - rejected as insufficient for business impact constitutional requirement

## Technology Stack

**Decision**: Python 3.11+ with pandas, numpy, seaborn, scikit-learn, matplotlib, jupyter, MLflow
**Rationale**: Meets all specified requirements from implementation specification, seaborn mandatory for visualizations, MLflow supports reproducible experimentation
**Alternatives considered**: R or other languages - rejected due to seaborn requirement specifying Python ecosystem

## Performance Metrics

**Decision**: F1-Score as primary metric, supplemented by AUC-ROC, Precision, Recall, and Confusion Matrix
**Rationale**: F1-Score optimal for imbalanced datasets, aligns with business objectives for customer retention, supports model evaluation constitutional requirement
**Alternatives considered**: Accuracy only - rejected due to class imbalance issues in churn data
"""
DataValidator implementation for telecommunications churn prediction.

Educational Focus: Demonstrates comprehensive data validation patterns and quality assurance.
This module implements the DataValidatorContract for validating data quality and consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple
import logging
from datetime import datetime

# Import our contract
from specs.contracts.data_loader import DataValidatorContract


class DataValidator(DataValidatorContract):
    """
    Implementation of DataValidatorContract for telecommunications churn data validation.

    Educational Notes:
    - Demonstrates systematic data quality assessment
    - Implements business logic validation
    - Provides actionable quality reports
    - Supports data-driven decision making
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the data validator.

        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)

        # Define business rules for telecommunications data
        self._setup_validation_rules()

    def _setup_validation_rules(self):
        """Set up validation rules specific to telecommunications churn data."""

        # Expected categorical values
        self.categorical_specifications = {
            'gender': ['Male', 'Female'],
            'senior_citizen': [0, 1],
            'partner': ['Yes', 'No'],
            'dependents': ['Yes', 'No'],
            'phone_service': ['Yes', 'No'],
            'multiple_lines': ['Yes', 'No', 'No phone service'],
            'internet_service': ['DSL', 'Fiber optic', 'No'],
            'online_security': ['Yes', 'No', 'No internet service'],
            'online_backup': ['Yes', 'No', 'No internet service'],
            'device_protection': ['Yes', 'No', 'No internet service'],
            'tech_support': ['Yes', 'No', 'No internet service'],
            'streaming_tv': ['Yes', 'No', 'No internet service'],
            'streaming_movies': ['Yes', 'No', 'No internet service'],
            'contract_type': ['month-to-month', 'one-year', 'two-year'],
            'paperless_billing': ['Yes', 'No'],
            'payment_method': [
                'electronic_check', 'mailed_check', 'bank_transfer', 'credit_card'
            ],
            'churn': [0, 1]
        }

        # Numeric column ranges (min, max)
        self.numeric_ranges = {
            'tenure': (0, 200),  # 0 to ~16 years seems reasonable
            'monthly_charges': (0, 500),  # $0 to $500 per month
            'total_charges': (0, 50000),  # $0 to $50k total
            'senior_citizen': (0, 1),  # Binary 0/1
            'churn': (0, 1)  # Binary 0/1
        }

        # Business logic rules
        self.business_rules = {
            'internet_service_dependency': {
                'description': 'Internet-related services require internet service',
                'condition': 'internet_service == "No"',
                'expected_values': ['No', 'No internet service']
            },
            'phone_service_dependency': {
                'description': 'Multiple lines require phone service',
                'condition': 'phone_service == "No"',
                'expected_value': 'No phone service'
            },
            'charges_consistency': {
                'description': 'Total charges should be consistent with tenure and monthly charges',
                'tolerance': 0.5  # 50% tolerance for real-world variance
            }
        }

    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Check if missing values are within acceptable threshold.

        Args:
            df: DataFrame to check
            threshold: Maximum acceptable missing percentage (default 5%)

        Returns:
            True if missing values are acceptable, False otherwise

        Educational Notes:
        - Missing data can severely impact ML model performance
        - Different strategies exist for handling missing values
        - Threshold should be adjusted based on domain requirements
        """
        self.logger.info(f"Checking missing values with threshold {threshold*100:.1f}%")

        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False

        # Calculate missing percentages for each column
        missing_percentages = df.isnull().sum() / len(df)

        # Check if any column exceeds threshold
        problematic_columns = missing_percentages[missing_percentages > threshold]

        if len(problematic_columns) > 0:
            self.logger.warning(f"Columns exceeding missing value threshold:")
            for col, pct in problematic_columns.items():
                self.logger.warning(f"  {col}: {pct*100:.1f}% missing")
            return False

        # Overall missing value check
        overall_missing = df.isnull().sum().sum() / (len(df) * len(df.columns))

        if overall_missing > threshold:
            self.logger.warning(f"Overall missing value rate {overall_missing*100:.1f}% exceeds threshold")
            return False

        self.logger.info("Missing values check passed")
        return True

    def validate_categorical_values(self, df: pd.DataFrame,
                                  column_specs: Dict[str, List] = None) -> Dict[str, List]:
        """
        Validate categorical columns have expected values.

        Args:
            df: DataFrame to validate
            column_specs: Dict mapping column names to list of expected values
                         If None, uses default specifications

        Returns:
            Dict of column: list of invalid values found

        Educational Notes:
        - Categorical validation prevents data quality issues
        - Unexpected values often indicate data entry errors
        - Important for maintaining model consistency
        """
        if column_specs is None:
            column_specs = self.categorical_specifications

        self.logger.info(f"Validating categorical values for {len(column_specs)} columns")

        invalid_values = {}

        for column, expected_values in column_specs.items():
            if column not in df.columns:
                self.logger.warning(f"Column '{column}' not found in DataFrame")
                continue

            # Get unique values, excluding NaN
            actual_values = set(df[column].dropna().unique())
            expected_set = set(expected_values)

            # Find invalid values
            invalid = actual_values - expected_set

            if invalid:
                invalid_values[column] = list(invalid)
                self.logger.warning(f"Invalid values in {column}: {invalid}")

                # Log frequency of invalid values
                for invalid_val in invalid:
                    count = (df[column] == invalid_val).sum()
                    pct = (count / len(df)) * 100
                    self.logger.warning(f"  '{invalid_val}': {count} occurrences ({pct:.1f}%)")

        if not invalid_values:
            self.logger.info("All categorical values are valid")

        return invalid_values

    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Check for business logic inconsistencies in telecommunications data.

        Args:
            df: DataFrame to check

        Returns:
            Dict of consistency_check: error_message
            Empty dict if all checks pass

        Educational Notes:
        - Business logic validation ensures domain-specific correctness
        - Identifies logical inconsistencies that could impact model performance
        - Provides actionable feedback for data quality improvement
        """
        self.logger.info("Performing business logic consistency checks")

        consistency_errors = {}

        # Check 1: Internet service dependencies
        consistency_errors.update(self._check_internet_service_consistency(df))

        # Check 2: Phone service dependencies
        consistency_errors.update(self._check_phone_service_consistency(df))

        # Check 3: Financial data consistency
        consistency_errors.update(self._check_financial_consistency(df))

        # Check 4: Numeric range validation
        consistency_errors.update(self._check_numeric_ranges(df))

        # Check 5: Duplicate customer IDs
        consistency_errors.update(self._check_duplicate_customers(df))

        # Check 6: Data type consistency
        consistency_errors.update(self._check_data_types(df))

        if not consistency_errors:
            self.logger.info("All consistency checks passed")
        else:
            self.logger.warning(f"Found {len(consistency_errors)} consistency issues")

        return consistency_errors

    def _check_internet_service_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check consistency of internet service related fields."""
        errors = {}

        if 'internet_service' not in df.columns:
            return errors

        # Internet-related service columns
        internet_columns = [
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]

        no_internet_customers = df[df['internet_service'] == 'No']

        for col in internet_columns:
            if col in df.columns:
                # For customers with no internet, service should be 'No' or 'No internet service'
                invalid_mask = (
                    (df['internet_service'] == 'No') &
                    (~df[col].isin(['No', 'No internet service']))
                )

                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    errors[f"internet_service_{col}_consistency"] = (
                        f"{invalid_count} customers have no internet service but "
                        f"have {col} values other than 'No' or 'No internet service'"
                    )

        return errors

    def _check_phone_service_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check consistency of phone service related fields."""
        errors = {}

        if 'phone_service' not in df.columns or 'multiple_lines' not in df.columns:
            return errors

        # Customers with no phone service should have 'No phone service' for multiple lines
        invalid_mask = (
            (df['phone_service'] == 'No') &
            (df['multiple_lines'] != 'No phone service')
        )

        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            errors['phone_service_multiple_lines_consistency'] = (
                f"{invalid_count} customers have no phone service but "
                f"multiple_lines is not 'No phone service'"
            )

        return errors

    def _check_financial_consistency(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check consistency of financial data."""
        errors = {}

        required_cols = ['tenure', 'monthly_charges', 'total_charges']
        if not all(col in df.columns for col in required_cols):
            return errors

        # Check for reasonable relationship between tenure, monthly charges, and total charges
        # Allow for significant variance to account for discounts, promotions, etc.

        # Calculate expected range for total charges
        expected_min = df['tenure'] * df['monthly_charges'] * 0.3  # 70% discount tolerance
        expected_max = df['tenure'] * df['monthly_charges'] * 2.0  # 100% increase tolerance

        # Find inconsistent rows (excluding zero tenure customers)
        non_zero_tenure = df['tenure'] > 0
        inconsistent_mask = (
            non_zero_tenure &
            ((df['total_charges'] < expected_min) | (df['total_charges'] > expected_max))
        )

        inconsistent_count = inconsistent_mask.sum()
        if inconsistent_count > 0:
            percentage = (inconsistent_count / len(df)) * 100
            if self.strict_mode or percentage > 10:  # Only report if >10% or in strict mode
                errors['financial_consistency'] = (
                    f"{inconsistent_count} customers ({percentage:.1f}%) have "
                    f"total_charges inconsistent with tenure and monthly_charges"
                )

        # Check for negative values
        for col in ['monthly_charges', 'total_charges']:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    errors[f'{col}_negative_values'] = (
                        f"{negative_count} customers have negative {col}"
                    )

        return errors

    def _check_numeric_ranges(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check numeric columns are within expected ranges."""
        errors = {}

        for column, (min_val, max_val) in self.numeric_ranges.items():
            if column not in df.columns:
                continue

            # Check for values outside expected range
            out_of_range = (df[column] < min_val) | (df[column] > max_val)
            out_of_range_count = out_of_range.sum()

            if out_of_range_count > 0:
                actual_min = df[column].min()
                actual_max = df[column].max()
                errors[f'{column}_range_validation'] = (
                    f"{out_of_range_count} values in {column} outside expected range "
                    f"[{min_val}, {max_val}]. Actual range: [{actual_min}, {actual_max}]"
                )

        return errors

    def _check_duplicate_customers(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check for duplicate customer records."""
        errors = {}

        if 'customer_id' not in df.columns:
            return errors

        duplicate_count = df['customer_id'].duplicated().sum()
        if duplicate_count > 0:
            errors['duplicate_customers'] = (
                f"Found {duplicate_count} duplicate customer IDs"
            )

        return errors

    def _check_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Check data types are appropriate for each column."""
        errors = {}

        # Expected data types
        expected_types = {
            'customer_id': 'object',
            'tenure': ['int64', 'int32', 'float64'],
            'monthly_charges': ['float64', 'float32'],
            'total_charges': ['float64', 'float32'],
            'senior_citizen': ['int64', 'int32'],
            'churn': ['int64', 'int32']
        }

        for column, expected in expected_types.items():
            if column not in df.columns:
                continue

            actual_type = str(df[column].dtype)

            if isinstance(expected, list):
                if actual_type not in expected:
                    errors[f'{column}_data_type'] = (
                        f"Column {column} has type {actual_type}, "
                        f"expected one of {expected}"
                    )
            else:
                if actual_type != expected:
                    errors[f'{column}_data_type'] = (
                        f"Column {column} has type {actual_type}, "
                        f"expected {expected}"
                    )

        return errors

    def validate_sample_distribution(self, df: pd.DataFrame,
                                   target_column: str = 'churn') -> Dict[str, Any]:
        """
        Validate sample distribution and class balance.

        Args:
            df: DataFrame to validate
            target_column: Name of target column

        Returns:
            Dict with distribution analysis

        Educational Notes:
        - Class imbalance significantly affects ML model performance
        - Understanding distribution helps choose appropriate algorithms
        - Informs sampling and evaluation strategies
        """
        self.logger.info(f"Analyzing sample distribution for {target_column}")

        distribution_report = {
            'total_samples': len(df),
            'class_distribution': {},
            'class_balance_ratio': None,
            'imbalance_severity': 'unknown',
            'recommendations': []
        }

        if target_column not in df.columns:
            distribution_report['error'] = f"Target column '{target_column}' not found"
            return distribution_report

        # Calculate class distribution
        class_counts = df[target_column].value_counts().sort_index()
        class_percentages = df[target_column].value_counts(normalize=True).sort_index() * 100

        for class_val in class_counts.index:
            distribution_report['class_distribution'][f'class_{class_val}'] = {
                'count': int(class_counts[class_val]),
                'percentage': float(class_percentages[class_val])
            }

        # Calculate imbalance ratio
        if len(class_counts) == 2:
            majority_count = class_counts.max()
            minority_count = class_counts.min()
            imbalance_ratio = majority_count / minority_count
            distribution_report['class_balance_ratio'] = float(imbalance_ratio)

            # Categorize imbalance severity
            if imbalance_ratio <= 1.5:
                distribution_report['imbalance_severity'] = 'balanced'
            elif imbalance_ratio <= 3.0:
                distribution_report['imbalance_severity'] = 'moderate'
            elif imbalance_ratio <= 10.0:
                distribution_report['imbalance_severity'] = 'high'
            else:
                distribution_report['imbalance_severity'] = 'extreme'

            # Add recommendations based on imbalance
            if imbalance_ratio > 3.0:
                distribution_report['recommendations'].extend([
                    "Consider using stratified sampling for train/test split",
                    "Evaluate models using balanced metrics (F1, balanced accuracy)",
                    "Consider resampling techniques (SMOTE, undersampling)"
                ])

            if imbalance_ratio > 10.0:
                distribution_report['recommendations'].extend([
                    "Consider cost-sensitive learning algorithms",
                    "Use precision-recall curves instead of ROC curves",
                    "Consider ensemble methods designed for imbalanced data"
                ])

        return distribution_report

    def generate_comprehensive_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data validation report.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with complete validation results

        Educational Notes:
        - Provides holistic view of data quality
        - Combines multiple validation dimensions
        - Supports data-driven quality decisions
        """
        self.logger.info("Generating comprehensive validation report")

        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_overview': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_values': {},
            'categorical_validation': {},
            'consistency_checks': {},
            'distribution_analysis': {},
            'overall_quality': {},
            'recommendations': []
        }

        # Missing values check
        missing_acceptable = self.check_missing_values(df)
        report['missing_values'] = {
            'passes_threshold': missing_acceptable,
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
        }

        # Categorical validation
        invalid_categoricals = self.validate_categorical_values(df)
        report['categorical_validation'] = {
            'invalid_values_found': len(invalid_categoricals) > 0,
            'invalid_values': invalid_categoricals,
            'columns_validated': list(self.categorical_specifications.keys())
        }

        # Consistency checks
        consistency_errors = self.check_data_consistency(df)
        report['consistency_checks'] = {
            'passes_consistency': len(consistency_errors) == 0,
            'errors': consistency_errors,
            'error_count': len(consistency_errors)
        }

        # Distribution analysis (if churn column exists)
        if 'churn' in df.columns:
            report['distribution_analysis'] = self.validate_sample_distribution(df)

        # Calculate overall quality score
        quality_score = self._calculate_overall_quality_score(report)
        report['overall_quality'] = {
            'score': quality_score,
            'grade': self._get_quality_grade(quality_score),
            'ready_for_ml': quality_score >= 0.7
        }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)

        self.logger.info(f"Validation complete. Overall quality score: {quality_score:.2f}")

        return report

    def _calculate_overall_quality_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall quality score from validation results."""
        score = 1.0

        # Missing values penalty
        if not report['missing_values']['passes_threshold']:
            score -= 0.2

        # Categorical validation penalty
        if report['categorical_validation']['invalid_values_found']:
            invalid_count = len(report['categorical_validation']['invalid_values'])
            score -= min(invalid_count * 0.1, 0.3)

        # Consistency penalty
        error_count = report['consistency_checks']['error_count']
        if error_count > 0:
            score -= min(error_count * 0.05, 0.3)

        # Distribution penalty (extreme imbalance)
        if 'distribution_analysis' in report:
            dist = report['distribution_analysis']
            if 'imbalance_severity' in dist and dist['imbalance_severity'] == 'extreme':
                score -= 0.1

        return max(score, 0.0)

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Missing values recommendations
        if not report['missing_values']['passes_threshold']:
            recommendations.append("Address missing values through imputation or removal")

        # Categorical recommendations
        if report['categorical_validation']['invalid_values_found']:
            recommendations.append("Clean or recode invalid categorical values")

        # Consistency recommendations
        if report['consistency_checks']['error_count'] > 0:
            recommendations.append("Resolve business logic inconsistencies")

        # Distribution recommendations
        if 'distribution_analysis' in report:
            recommendations.extend(
                report['distribution_analysis'].get('recommendations', [])
            )

        # Overall quality recommendations
        quality_score = report['overall_quality']['score']
        if quality_score < 0.7:
            recommendations.append("Consider data quality improvement before ML training")

        return recommendations
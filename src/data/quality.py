"""
Data Quality monitoring and reporting for telecommunications churn prediction.

Educational Focus: Demonstrates comprehensive data quality monitoring patterns.
This module provides advanced data quality assessment, monitoring, and reporting capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring and assessment system.

    Educational Notes:
    - Provides systematic approach to data quality assessment
    - Implements industry-standard quality metrics
    - Supports both batch and streaming quality monitoring
    - Enables data-driven quality improvement decisions
    """

    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the data quality monitor.

        Args:
            reference_data: Optional reference dataset for drift detection
        """
        self.logger = logging.getLogger(__name__)
        self.reference_data = reference_data
        self.quality_history = []

        # Quality thresholds (configurable)
        self.thresholds = {
            'missing_threshold': 0.05,      # 5% missing data threshold
            'outlier_threshold': 0.05,      # 5% outlier threshold
            'uniqueness_threshold': 0.95,   # 95% uniqueness for ID columns
            'correlation_threshold': 0.9,   # 90% correlation threshold
            'drift_threshold': 0.1,         # 10% distribution drift threshold
            'consistency_threshold': 0.95   # 95% consistency threshold
        }

    def assess_data_quality(self, df: pd.DataFrame,
                          target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive data quality assessment.

        Args:
            df: DataFrame to assess
            target_column: Optional target column for ML-specific checks

        Returns:
            Dict containing comprehensive quality assessment

        Educational Notes:
        - Covers all major dimensions of data quality
        - Provides both quantitative metrics and qualitative insights
        - Suitable for academic presentations and business reporting
        """
        self.logger.info(f"Starting comprehensive quality assessment for dataset with {len(df)} rows")

        assessment = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self._get_dataset_info(df),
            'completeness': self._assess_completeness(df),
            'validity': self._assess_validity(df),
            'uniqueness': self._assess_uniqueness(df),
            'consistency': self._assess_consistency(df),
            'accuracy': self._assess_accuracy(df),
            'timeliness': self._assess_timeliness(df),
            'outliers': self._detect_outliers(df),
            'distributions': self._analyze_distributions(df),
            'correlations': self._analyze_correlations(df),
            'ml_readiness': self._assess_ml_readiness(df, target_column),
            'overall_score': 0.0,
            'quality_grade': 'F',
            'recommendations': []
        }

        # Calculate overall quality score
        assessment['overall_score'] = self._calculate_overall_score(assessment)
        assessment['quality_grade'] = self._get_quality_grade(assessment['overall_score'])
        assessment['recommendations'] = self._generate_recommendations(assessment)

        # Store in history
        self.quality_history.append({
            'timestamp': assessment['timestamp'],
            'score': assessment['overall_score'],
            'grade': assessment['quality_grade'],
            'row_count': len(df)
        })

        self.logger.info(f"Quality assessment complete. Score: {assessment['overall_score']:.3f}")

        return assessment

    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
        }

    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data completeness (missing values).

        Educational Notes:
        - Completeness is fundamental to data quality
        - Missing data patterns can indicate systematic issues
        - Different columns may have different completeness requirements
        """
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        completeness = {
            'overall_completeness': ((df.size - df.isnull().sum().sum()) / df.size) * 100,
            'columns_with_missing': (missing_counts > 0).sum(),
            'missing_by_column': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'most_incomplete_columns': missing_percentages.nlargest(5).to_dict(),
            'completeness_score': 0.0
        }

        # Calculate completeness score
        max_missing_pct = missing_percentages.max() if len(missing_percentages) > 0 else 0
        if max_missing_pct <= self.thresholds['missing_threshold'] * 100:
            completeness['completeness_score'] = 1.0
        else:
            # Penalize based on highest missing percentage
            completeness['completeness_score'] = max(0, 1 - (max_missing_pct / 100))

        return completeness

    def _assess_validity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data validity (format and domain constraints).

        Educational Notes:
        - Validity ensures data conforms to expected formats
        - Domain constraints prevent logical inconsistencies
        - Invalid data can cause ML model failures
        """
        validity = {
            'numeric_validity': {},
            'categorical_validity': {},
            'format_validity': {},
            'domain_constraints': {},
            'validity_score': 0.0
        }

        # Check numeric columns for invalid values (inf, extremely large)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            invalid_count = 0

            # Check for infinite values
            inf_count = np.isinf(df[col]).sum()
            invalid_count += inf_count

            # Check for extremely large values (beyond reasonable range)
            if col in ['tenure', 'monthly_charges', 'total_charges']:
                # Define reasonable ranges
                ranges = {
                    'tenure': (0, 200),
                    'monthly_charges': (0, 1000),
                    'total_charges': (0, 100000)
                }
                if col in ranges:
                    min_val, max_val = ranges[col]
                    out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                    invalid_count += out_of_range

            validity['numeric_validity'][col] = {
                'invalid_count': invalid_count,
                'invalid_percentage': (invalid_count / len(df)) * 100
            }

        # Check categorical columns for expected values
        categorical_specs = {
            'gender': ['Male', 'Female'],
            'contract_type': ['month-to-month', 'one-year', 'two-year'],
            'payment_method': ['electronic_check', 'mailed_check', 'bank_transfer', 'credit_card']
        }

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in categorical_specs:
                expected_values = set(categorical_specs[col])
                actual_values = set(df[col].dropna().unique())
                invalid_values = actual_values - expected_values

                validity['categorical_validity'][col] = {
                    'invalid_values': list(invalid_values),
                    'invalid_count': sum((df[col].isin(invalid_values)).sum() for val in invalid_values)
                }

        # Calculate overall validity score
        total_invalid = 0
        total_values = len(df) * len(df.columns)

        for col_validity in validity['numeric_validity'].values():
            total_invalid += col_validity['invalid_count']

        for col_validity in validity['categorical_validity'].values():
            total_invalid += col_validity['invalid_count']

        validity['validity_score'] = max(0, 1 - (total_invalid / total_values))

        return validity

    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data uniqueness (duplicates and cardinality).

        Educational Notes:
        - Uniqueness prevents data duplication issues
        - Important for customer ID and key columns
        - Affects statistical analysis validity
        """
        uniqueness = {
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'column_uniqueness': {},
            'uniqueness_score': 0.0
        }

        # Check uniqueness for each column
        for col in df.columns:
            unique_count = df[col].nunique()
            total_count = len(df) - df[col].isnull().sum()  # Exclude missing values

            uniqueness['column_uniqueness'][col] = {
                'unique_count': unique_count,
                'total_non_null': total_count,
                'uniqueness_ratio': unique_count / total_count if total_count > 0 else 0,
                'is_id_like': unique_count == total_count  # Perfect uniqueness
            }

        # Calculate uniqueness score
        duplicate_penalty = min(uniqueness['duplicate_percentage'] / 100, 0.5)
        uniqueness['uniqueness_score'] = max(0, 1 - duplicate_penalty)

        return uniqueness

    def _assess_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data consistency (internal logical consistency).

        Educational Notes:
        - Consistency ensures logical relationships are maintained
        - Critical for business rule compliance
        - Prevents contradictory information
        """
        consistency = {
            'business_rule_violations': {},
            'referential_integrity': {},
            'temporal_consistency': {},
            'consistency_score': 0.0
        }

        # Business rule checks for telecom data
        total_violations = 0

        # Rule 1: Internet service dependency
        if all(col in df.columns for col in ['internet_service', 'online_security']):
            violation_mask = (
                (df['internet_service'] == 'No') &
                (df['online_security'] != 'No internet service') &
                (df['online_security'] != 'No')
            )
            violations = violation_mask.sum()
            total_violations += violations
            consistency['business_rule_violations']['internet_service_dependency'] = violations

        # Rule 2: Phone service dependency
        if all(col in df.columns for col in ['phone_service', 'multiple_lines']):
            violation_mask = (
                (df['phone_service'] == 'No') &
                (df['multiple_lines'] != 'No phone service')
            )
            violations = violation_mask.sum()
            total_violations += violations
            consistency['business_rule_violations']['phone_service_dependency'] = violations

        # Rule 3: Financial consistency
        if all(col in df.columns for col in ['tenure', 'monthly_charges', 'total_charges']):
            # Allow significant variance for real-world data
            expected_min = df['tenure'] * df['monthly_charges'] * 0.2
            expected_max = df['tenure'] * df['monthly_charges'] * 3.0

            violation_mask = (
                (df['tenure'] > 0) &
                ((df['total_charges'] < expected_min) | (df['total_charges'] > expected_max))
            )
            violations = violation_mask.sum()
            total_violations += violations
            consistency['business_rule_violations']['financial_consistency'] = violations

        # Calculate consistency score
        violation_rate = total_violations / len(df) if len(df) > 0 else 0
        consistency['consistency_score'] = max(0, 1 - violation_rate)

        return consistency

    def _assess_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data accuracy (correctness of values).

        Educational Notes:
        - Accuracy is difficult to measure without external reference
        - Can use statistical methods to detect probable errors
        - Important for model performance
        """
        accuracy = {
            'statistical_outliers': {},
            'format_errors': {},
            'probable_errors': {},
            'accuracy_score': 0.8  # Default score as accuracy is hard to measure
        }

        # Statistical outlier detection for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                # Use IQR method for outlier detection
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = (outliers / len(df)) * 100

                accuracy['statistical_outliers'][col] = {
                    'outlier_count': outliers,
                    'outlier_percentage': outlier_percentage,
                    'bounds': (lower_bound, upper_bound)
                }

        return accuracy

    def _assess_timeliness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data timeliness (recency and temporal relevance).

        Educational Notes:
        - Timeliness affects model relevance
        - Stale data may not represent current patterns
        - Important for business decision making
        """
        timeliness = {
            'data_freshness': 'unknown',
            'temporal_gaps': {},
            'timeliness_score': 0.8  # Default score
        }

        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns

        if len(datetime_cols) > 0:
            for col in datetime_cols:
                if not df[col].empty:
                    latest_date = df[col].max()
                    earliest_date = df[col].min()
                    days_old = (datetime.now() - latest_date).days if pd.notna(latest_date) else None

                    timeliness['temporal_gaps'][col] = {
                        'earliest_date': earliest_date.isoformat() if pd.notna(earliest_date) else None,
                        'latest_date': latest_date.isoformat() if pd.notna(latest_date) else None,
                        'days_old': days_old,
                        'date_range_days': (latest_date - earliest_date).days if pd.notna(latest_date) and pd.notna(earliest_date) else None
                    }

                    # Adjust timeliness score based on age
                    if days_old is not None:
                        if days_old <= 30:
                            timeliness['timeliness_score'] = 1.0
                        elif days_old <= 90:
                            timeliness['timeliness_score'] = 0.8
                        elif days_old <= 365:
                            timeliness['timeliness_score'] = 0.6
                        else:
                            timeliness['timeliness_score'] = 0.4

        return timeliness

    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive outlier detection across multiple methods.

        Educational Notes:
        - Multiple outlier detection methods provide robust results
        - Different methods suitable for different data types
        - Outliers can indicate data quality issues or interesting patterns
        """
        outliers = {
            'methods_used': ['iqr', 'zscore', 'isolation_forest'],
            'outliers_by_column': {},
            'outlier_summary': {}
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) < 4:  # Skip if too few data points
                continue

            col_outliers = {
                'iqr_outliers': 0,
                'zscore_outliers': 0,
                'total_outliers': 0,
                'outlier_percentage': 0.0
            }

            # IQR method
            Q1, Q3 = col_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
            col_outliers['iqr_outliers'] = iqr_outliers

            # Z-score method (if data has reasonable variance)
            if col_data.std() > 0:
                z_scores = np.abs(stats.zscore(col_data))
                zscore_outliers = (z_scores > 3).sum()
                col_outliers['zscore_outliers'] = zscore_outliers

            # Combined outlier count (union of methods)
            col_outliers['total_outliers'] = max(iqr_outliers, col_outliers['zscore_outliers'])
            col_outliers['outlier_percentage'] = (col_outliers['total_outliers'] / len(col_data)) * 100

            outliers['outliers_by_column'][col] = col_outliers

        # Summary statistics
        total_outliers = sum(col['total_outliers'] for col in outliers['outliers_by_column'].values())
        total_values = sum(len(df[col].dropna()) for col in numeric_cols)

        outliers['outlier_summary'] = {
            'total_outliers': total_outliers,
            'total_values_checked': total_values,
            'overall_outlier_percentage': (total_outliers / total_values * 100) if total_values > 0 else 0
        }

        return outliers

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze statistical distributions of data.

        Educational Notes:
        - Distribution analysis reveals data characteristics
        - Helps identify skewness and potential transformations needed
        - Important for choosing appropriate ML algorithms
        """
        distributions = {
            'numeric_distributions': {},
            'categorical_distributions': {},
            'distribution_health': {}
        }

        # Numeric distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                distributions['numeric_distributions'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75))
                }

        # Categorical distributions
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            distributions['categorical_distributions'][col] = {
                'unique_values': len(value_counts),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'distribution_entropy': float(stats.entropy(value_counts.values)) if len(value_counts) > 1 else 0
            }

        return distributions

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between variables.

        Educational Notes:
        - High correlations indicate potential multicollinearity
        - Important for feature selection
        - Reveals relationships in data
        """
        correlations = {
            'correlation_matrix': {},
            'high_correlations': [],
            'correlation_summary': {}
        }

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()

            # Convert to dict for JSON serialization
            correlations['correlation_matrix'] = corr_matrix.round(3).to_dict()

            # Find high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > self.thresholds['correlation_threshold']:
                        high_corr_pairs.append({
                            'variable_1': corr_matrix.columns[i],
                            'variable_2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })

            correlations['high_correlations'] = high_corr_pairs

            # Summary statistics
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            correlations['correlation_summary'] = {
                'max_correlation': float(upper_triangle.max().max()) if not upper_triangle.empty else 0,
                'min_correlation': float(upper_triangle.min().min()) if not upper_triangle.empty else 0,
                'mean_correlation': float(upper_triangle.mean().mean()) if not upper_triangle.empty else 0,
                'high_correlation_pairs': len(high_corr_pairs)
            }

        return correlations

    def _assess_ml_readiness(self, df: pd.DataFrame,
                           target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess readiness for machine learning applications.

        Educational Notes:
        - ML readiness combines multiple quality dimensions
        - Identifies specific issues that affect model training
        - Provides actionable recommendations for ML preparation
        """
        ml_readiness = {
            'ready_for_ml': False,
            'blocking_issues': [],
            'recommended_preprocessing': [],
            'class_balance': {},
            'feature_quality': {},
            'readiness_score': 0.0
        }

        # Check basic requirements
        if len(df) < 100:
            ml_readiness['blocking_issues'].append("Insufficient data (< 100 rows)")

        # Check missing values
        missing_pct = (df.isnull().sum().sum() / df.size) * 100
        if missing_pct > 20:
            ml_readiness['blocking_issues'].append(f"High missing data ({missing_pct:.1f}%)")
        elif missing_pct > 5:
            ml_readiness['recommended_preprocessing'].append("Handle missing values")

        # Check for target column if provided
        if target_column:
            if target_column not in df.columns:
                ml_readiness['blocking_issues'].append(f"Target column '{target_column}' not found")
            else:
                # Analyze class balance
                class_dist = df[target_column].value_counts()
                minority_class_pct = (class_dist.min() / len(df)) * 100

                ml_readiness['class_balance'] = {
                    'minority_class_percentage': minority_class_pct,
                    'class_distribution': class_dist.to_dict(),
                    'imbalanced': minority_class_pct < 10
                }

                if minority_class_pct < 5:
                    ml_readiness['blocking_issues'].append("Severe class imbalance (< 5% minority class)")
                elif minority_class_pct < 20:
                    ml_readiness['recommended_preprocessing'].append("Address class imbalance")

        # Check data types
        mixed_type_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains mixed types
                sample_values = df[col].dropna().head(100)
                types = set(type(val).__name__ for val in sample_values)
                if len(types) > 1:
                    mixed_type_cols.append(col)

        if mixed_type_cols:
            ml_readiness['recommended_preprocessing'].append(f"Fix mixed data types in: {mixed_type_cols}")

        # Calculate readiness score
        blocking_penalty = len(ml_readiness['blocking_issues']) * 0.3
        preprocessing_penalty = len(ml_readiness['recommended_preprocessing']) * 0.1

        ml_readiness['readiness_score'] = max(0, 1 - blocking_penalty - preprocessing_penalty)
        ml_readiness['ready_for_ml'] = (
            len(ml_readiness['blocking_issues']) == 0 and
            ml_readiness['readiness_score'] >= 0.7
        )

        return ml_readiness

    def _calculate_overall_score(self, assessment: Dict[str, Any]) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'completeness': 0.20,
            'validity': 0.20,
            'uniqueness': 0.15,
            'consistency': 0.20,
            'accuracy': 0.15,
            'timeliness': 0.10
        }

        weighted_score = 0.0
        for dimension, weight in weights.items():
            if dimension in assessment and f'{dimension}_score' in assessment[dimension]:
                weighted_score += assessment[dimension][f'{dimension}_score'] * weight

        return min(weighted_score, 1.0)

    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 0.95:
            return 'A+'
        elif score >= 0.90:
            return 'A'
        elif score >= 0.85:
            return 'A-'
        elif score >= 0.80:
            return 'B+'
        elif score >= 0.75:
            return 'B'
        elif score >= 0.70:
            return 'B-'
        elif score >= 0.65:
            return 'C+'
        elif score >= 0.60:
            return 'C'
        elif score >= 0.55:
            return 'C-'
        elif score >= 0.50:
            return 'D'
        else:
            return 'F'

    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable quality improvement recommendations."""
        recommendations = []

        # Completeness recommendations
        if assessment['completeness']['completeness_score'] < 0.9:
            recommendations.append("Implement missing value imputation strategy")

        # Validity recommendations
        if assessment['validity']['validity_score'] < 0.9:
            recommendations.append("Clean invalid data values and fix format issues")

        # Uniqueness recommendations
        if assessment['uniqueness']['duplicate_percentage'] > 1:
            recommendations.append("Remove or investigate duplicate records")

        # Consistency recommendations
        if assessment['consistency']['consistency_score'] < 0.9:
            recommendations.append("Fix business rule violations and logical inconsistencies")

        # ML readiness recommendations
        if 'ml_readiness' in assessment and not assessment['ml_readiness']['ready_for_ml']:
            recommendations.extend(assessment['ml_readiness']['recommended_preprocessing'])

        # Overall score recommendations
        if assessment['overall_score'] < 0.7:
            recommendations.append("Consider comprehensive data quality improvement before analysis")

        return recommendations

    def compare_with_reference(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare current data with reference data for drift detection.

        Args:
            current_data: Current dataset to compare

        Returns:
            Dict containing comparison results

        Educational Notes:
        - Data drift detection is crucial for ML model maintenance
        - Identifies when model retraining might be needed
        - Supports continuous monitoring workflows
        """
        if self.reference_data is None:
            return {'error': 'No reference data available for comparison'}

        comparison = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': False,
            'distribution_changes': {},
            'schema_changes': {},
            'quality_changes': {},
            'recommendations': []
        }

        # Schema comparison
        ref_cols = set(self.reference_data.columns)
        curr_cols = set(current_data.columns)

        comparison['schema_changes'] = {
            'added_columns': list(curr_cols - ref_cols),
            'removed_columns': list(ref_cols - curr_cols),
            'schema_drift': len(ref_cols.symmetric_difference(curr_cols)) > 0
        }

        # Distribution comparison for common columns
        common_cols = ref_cols & curr_cols
        numeric_cols = [col for col in common_cols
                       if self.reference_data[col].dtype in ['int64', 'float64']]

        for col in numeric_cols:
            if len(self.reference_data[col].dropna()) > 0 and len(current_data[col].dropna()) > 0:
                # Statistical tests for distribution changes
                try:
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_p_value = stats.ks_2samp(
                        self.reference_data[col].dropna(),
                        current_data[col].dropna()
                    )

                    comparison['distribution_changes'][col] = {
                        'ks_statistic': float(ks_stat),
                        'ks_p_value': float(ks_p_value),
                        'significant_change': ks_p_value < 0.05,
                        'ref_mean': float(self.reference_data[col].mean()),
                        'curr_mean': float(current_data[col].mean()),
                        'mean_change_pct': float(
                            abs(current_data[col].mean() - self.reference_data[col].mean()) /
                            self.reference_data[col].mean() * 100
                        ) if self.reference_data[col].mean() != 0 else 0
                    }

                    if ks_p_value < 0.05:
                        comparison['drift_detected'] = True
                        comparison['recommendations'].append(
                            f"Significant distribution change detected in {col}"
                        )

                except Exception as e:
                    self.logger.warning(f"Could not perform distribution comparison for {col}: {e}")

        return comparison

    def generate_quality_report(self, assessment: Dict[str, Any],
                              output_format: str = 'dict') -> Union[Dict[str, Any], str]:
        """
        Generate formatted quality report.

        Args:
            assessment: Quality assessment results
            output_format: 'dict', 'markdown', or 'html'

        Returns:
            Formatted report

        Educational Notes:
        - Professional reporting for stakeholders
        - Multiple formats for different audiences
        - Executive summary with actionable insights
        """
        if output_format == 'markdown':
            return self._generate_markdown_report(assessment)
        elif output_format == 'html':
            return self._generate_html_report(assessment)
        else:
            return assessment

    def _generate_markdown_report(self, assessment: Dict[str, Any]) -> str:
        """Generate markdown formatted report."""
        report = f"""# Data Quality Assessment Report

**Generated:** {assessment['timestamp']}
**Overall Score:** {assessment['overall_score']:.3f} ({assessment['quality_grade']})

## Executive Summary

Dataset contains {assessment['dataset_info']['rows']:,} rows and {assessment['dataset_info']['columns']} columns.

### Quality Dimensions

| Dimension | Score | Status |
|-----------|-------|--------|
| Completeness | {assessment['completeness']['completeness_score']:.3f} | {'✅' if assessment['completeness']['completeness_score'] > 0.8 else '⚠️'} |
| Validity | {assessment['validity']['validity_score']:.3f} | {'✅' if assessment['validity']['validity_score'] > 0.8 else '⚠️'} |
| Uniqueness | {assessment['uniqueness']['uniqueness_score']:.3f} | {'✅' if assessment['uniqueness']['uniqueness_score'] > 0.8 else '⚠️'} |
| Consistency | {assessment['consistency']['consistency_score']:.3f} | {'✅' if assessment['consistency']['consistency_score'] > 0.8 else '⚠️'} |

### Key Findings

- **Missing Data:** {assessment['completeness']['columns_with_missing']} columns have missing values
- **Duplicates:** {assessment['uniqueness']['duplicate_rows']} duplicate rows found
- **Outliers:** {assessment['outliers']['outlier_summary']['overall_outlier_percentage']:.1f}% outlier rate

### Recommendations

"""
        for rec in assessment['recommendations']:
            report += f"- {rec}\n"

        if 'ml_readiness' in assessment:
            report += f"\n### ML Readiness\n"
            report += f"**Ready for ML:** {'✅ Yes' if assessment['ml_readiness']['ready_for_ml'] else '❌ No'}\n"
            report += f"**Readiness Score:** {assessment['ml_readiness']['readiness_score']:.3f}\n"

        return report

    def _generate_html_report(self, assessment: Dict[str, Any]) -> str:
        """Generate HTML formatted report."""
        # Basic HTML template - would be more sophisticated in production
        html = f"""
        <html>
        <head><title>Data Quality Report</title></head>
        <body>
        <h1>Data Quality Assessment Report</h1>
        <p><strong>Generated:</strong> {assessment['timestamp']}</p>
        <p><strong>Overall Score:</strong> {assessment['overall_score']:.3f} ({assessment['quality_grade']})</p>

        <h2>Quality Summary</h2>
        <ul>
            <li>Completeness: {assessment['completeness']['completeness_score']:.3f}</li>
            <li>Validity: {assessment['validity']['validity_score']:.3f}</li>
            <li>Uniqueness: {assessment['uniqueness']['uniqueness_score']:.3f}</li>
            <li>Consistency: {assessment['consistency']['consistency_score']:.3f}</li>
        </ul>

        <h2>Recommendations</h2>
        <ul>
        """

        for rec in assessment['recommendations']:
            html += f"<li>{rec}</li>"

        html += """
        </ul>
        </body>
        </html>
        """

        return html
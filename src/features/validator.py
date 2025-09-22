"""
FeatureValidator implementation for telecommunications churn prediction.

Educational Focus: Demonstrates ML feature validation and quality assurance patterns.
This module implements validation checks for processed features to ensure data quality
and consistency across training and test sets.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
import warnings
from datetime import datetime

from ..models.processed_features import ProcessedFeatures, FeatureType


class FeatureValidator:
    """
    Feature validation and quality assurance for churn prediction features.

    Educational Notes:
    - Validates feature distributions and statistical properties
    - Detects data drift between training and test sets
    - Identifies multicollinearity and feature quality issues
    - Provides comprehensive validation reports for ML pipelines
    """

    def __init__(self, tolerance: float = 0.1):
        """
        Initialize FeatureValidator.

        Args:
            tolerance: Tolerance level for validation checks (0.0 to 1.0)
        """
        self.tolerance = tolerance
        self.validation_history: List[Dict[str, Any]] = []

    def validate_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Validate that processed features have expected distributions.

        Args:
            df: DataFrame with processed features

        Returns:
            Dict mapping column names to distribution statistics
            (mean, std, min, max, skewness, kurtosis)

        Educational Notes:
        - Identifies potential data quality issues
        - Detects extreme outliers and distribution anomalies
        - Helps validate feature engineering steps
        - Essential for model performance monitoring
        """
        distribution_stats = {}

        for column in df.columns:
            feature_data = df[column].dropna()

            if len(feature_data) == 0:
                warnings.warn(f"Feature '{column}' has no non-null values")
                continue

            stats_dict = {
                'count': len(feature_data),
                'missing_count': df[column].isnull().sum(),
                'missing_percentage': df[column].isnull().sum() / len(df) * 100
            }

            # Check if feature is numerical
            if pd.api.types.is_numeric_dtype(feature_data):
                # Basic statistics
                stats_dict.update({
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'median': float(feature_data.median()),
                    'q25': float(feature_data.quantile(0.25)),
                    'q75': float(feature_data.quantile(0.75))
                })

                # Advanced distribution metrics
                try:
                    stats_dict.update({
                        'skewness': float(stats.skew(feature_data)),
                        'kurtosis': float(stats.kurtosis(feature_data)),
                        'variance': float(feature_data.var())
                    })
                except Exception as e:
                    warnings.warn(f"Could not calculate distribution metrics for {column}: {e}")
                    stats_dict.update({
                        'skewness': np.nan,
                        'kurtosis': np.nan,
                        'variance': np.nan
                    })

                # Outlier detection using IQR method
                q1, q3 = feature_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = feature_data[(feature_data < lower_bound) | (feature_data > upper_bound)]

                stats_dict.update({
                    'outlier_count': len(outliers),
                    'outlier_percentage': len(outliers) / len(feature_data) * 100,
                    'iqr': float(iqr),
                    'lower_outlier_bound': float(lower_bound),
                    'upper_outlier_bound': float(upper_bound)
                })

                # Check for infinite values
                inf_count = np.isinf(feature_data).sum()
                stats_dict.update({
                    'infinite_count': int(inf_count),
                    'infinite_percentage': inf_count / len(feature_data) * 100
                })

                # Distribution shape assessment
                if stats_dict['std'] > 0:
                    # Coefficient of variation
                    stats_dict['coefficient_of_variation'] = abs(stats_dict['std'] / stats_dict['mean']) if stats_dict['mean'] != 0 else np.inf

                    # Assess normality (Shapiro-Wilk test for small samples)
                    if len(feature_data) <= 5000:
                        try:
                            shapiro_stat, shapiro_p = stats.shapiro(feature_data.sample(min(5000, len(feature_data))))
                            stats_dict.update({
                                'shapiro_stat': float(shapiro_stat),
                                'shapiro_p_value': float(shapiro_p),
                                'is_normal': shapiro_p > 0.05
                            })
                        except:
                            stats_dict.update({
                                'shapiro_stat': np.nan,
                                'shapiro_p_value': np.nan,
                                'is_normal': False
                            })
                else:
                    stats_dict['coefficient_of_variation'] = 0.0

            else:
                # Categorical feature statistics
                value_counts = feature_data.value_counts()
                unique_count = feature_data.nunique()

                stats_dict.update({
                    'unique_count': unique_count,
                    'most_frequent_value': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                    'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    'most_frequent_percentage': value_counts.iloc[0] / len(feature_data) * 100 if len(value_counts) > 0 else 0
                })

                # Calculate entropy (measure of diversity)
                probabilities = value_counts / len(feature_data)
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)
                stats_dict['entropy'] = float(entropy)

                # Check for high cardinality
                cardinality_ratio = unique_count / len(feature_data)
                stats_dict.update({
                    'cardinality_ratio': float(cardinality_ratio),
                    'high_cardinality': cardinality_ratio > 0.5
                })

            # Flag potential issues
            issues = []

            # High missing value rate
            if stats_dict['missing_percentage'] > 20:
                issues.append(f"High missing values: {stats_dict['missing_percentage']:.1f}%")

            # For numerical features
            if pd.api.types.is_numeric_dtype(feature_data):
                # Very low variance
                if 'std' in stats_dict and stats_dict['std'] < 1e-6:
                    issues.append("Very low variance (near constant)")

                # High outlier rate
                if 'outlier_percentage' in stats_dict and stats_dict['outlier_percentage'] > 10:
                    issues.append(f"High outlier rate: {stats_dict['outlier_percentage']:.1f}%")

                # Infinite values
                if 'infinite_count' in stats_dict and stats_dict['infinite_count'] > 0:
                    issues.append(f"Contains {stats_dict['infinite_count']} infinite values")

                # Extreme skewness
                if 'skewness' in stats_dict and abs(stats_dict['skewness']) > 3:
                    issues.append(f"Extreme skewness: {stats_dict['skewness']:.2f}")

            # For categorical features
            else:
                # Very high cardinality
                if 'high_cardinality' in stats_dict and stats_dict['high_cardinality']:
                    issues.append(f"High cardinality: {stats_dict['unique_count']} unique values")

                # Highly imbalanced
                if 'most_frequent_percentage' in stats_dict and stats_dict['most_frequent_percentage'] > 95:
                    issues.append(f"Highly imbalanced: {stats_dict['most_frequent_percentage']:.1f}% in single category")

            stats_dict['validation_issues'] = issues

            distribution_stats[column] = stats_dict

        return distribution_stats

    def check_feature_correlations(self, df: pd.DataFrame, threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Identify highly correlated features that may cause multicollinearity.

        Args:
            df: DataFrame with features
            threshold: Correlation threshold for flagging

        Returns:
            List of tuples (feature1, feature2, correlation_value)

        Educational Notes:
        - Identifies redundant features
        - Helps prevent multicollinearity in linear models
        - Essential for feature selection decisions
        - Can indicate data leakage issues
        """
        # Select only numerical features for correlation analysis
        numerical_df = df.select_dtypes(include=[np.number])

        if numerical_df.empty:
            warnings.warn("No numerical features found for correlation analysis")
            return []

        # Calculate correlation matrix
        correlation_matrix = numerical_df.corr()

        # Find highly correlated pairs
        high_correlations = []

        # Iterate through upper triangle of correlation matrix
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                correlation = correlation_matrix.iloc[i, j]

                # Skip NaN correlations
                if pd.isna(correlation):
                    continue

                # Check if correlation exceeds threshold
                if abs(correlation) >= threshold:
                    high_correlations.append((feature1, feature2, float(correlation)))

        # Sort by absolute correlation value (descending)
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        return high_correlations

    def validate_encoding_consistency(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, str]:
        """
        Ensure consistent encoding between training and test sets.

        Args:
            train_df: Training set features
            test_df: Test set features

        Returns:
            Dict of potential issues found

        Educational Notes:
        - Prevents data leakage and inconsistencies
        - Ensures model can handle test data properly
        - Critical for production deployment
        - Validates preprocessing pipeline integrity
        """
        issues = {}

        # Check feature name consistency
        train_features = set(train_df.columns)
        test_features = set(test_df.columns)

        missing_in_test = train_features - test_features
        extra_in_test = test_features - train_features

        if missing_in_test:
            issues['missing_features_in_test'] = f"Features missing in test set: {list(missing_in_test)}"

        if extra_in_test:
            issues['extra_features_in_test'] = f"Extra features in test set: {list(extra_in_test)}"

        # Check common features
        common_features = train_features & test_features

        for feature in common_features:
            train_feature = train_df[feature]
            test_feature = test_df[feature]

            # Check data type consistency
            if train_feature.dtype != test_feature.dtype:
                issues[f'{feature}_dtype_mismatch'] = (
                    f"Data type mismatch for {feature}: "
                    f"train={train_feature.dtype}, test={test_feature.dtype}"
                )

            # For numerical features
            if pd.api.types.is_numeric_dtype(train_feature):
                # Check for extreme value differences
                train_range = train_feature.max() - train_feature.min()
                test_range = test_feature.max() - test_feature.min()

                if train_range > 0 and test_range > 0:
                    range_ratio = test_range / train_range
                    if range_ratio > 2 or range_ratio < 0.5:
                        issues[f'{feature}_range_difference'] = (
                            f"Significant range difference for {feature}: "
                            f"train_range={train_range:.2f}, test_range={test_range:.2f}"
                        )

                # Statistical distribution comparison (Kolmogorov-Smirnov test)
                try:
                    train_clean = train_feature.dropna()
                    test_clean = test_feature.dropna()

                    if len(train_clean) > 10 and len(test_clean) > 10:
                        ks_stat, ks_p = ks_2samp(train_clean, test_clean)
                        if ks_p < 0.05:  # Significant difference
                            issues[f'{feature}_distribution_shift'] = (
                                f"Significant distribution shift for {feature}: "
                                f"KS statistic={ks_stat:.3f}, p-value={ks_p:.3f}"
                            )
                except:
                    # Skip if statistical test fails
                    pass

            # For categorical features
            else:
                train_categories = set(train_feature.dropna().unique())
                test_categories = set(test_feature.dropna().unique())

                # Check for new categories in test set
                new_categories = test_categories - train_categories
                if new_categories:
                    issues[f'{feature}_new_categories'] = (
                        f"New categories in test set for {feature}: {list(new_categories)}"
                    )

                # Check for missing categories
                missing_categories = train_categories - test_categories
                if len(missing_categories) / len(train_categories) > 0.1:  # More than 10% missing
                    issues[f'{feature}_missing_categories'] = (
                        f"Many categories missing in test set for {feature}: "
                        f"{len(missing_categories)}/{len(train_categories)} missing"
                    )

                # Check category distribution shifts (Chi-square test for small number of categories)
                if len(train_categories) <= 20 and len(test_categories) <= 20:
                    try:
                        # Create contingency table
                        train_counts = train_feature.value_counts().reindex(train_categories | test_categories, fill_value=0)
                        test_counts = test_feature.value_counts().reindex(train_categories | test_categories, fill_value=0)

                        contingency_table = np.array([train_counts.values, test_counts.values])

                        # Perform chi-square test if we have enough data
                        if contingency_table.sum() > 50 and (contingency_table > 5).all():
                            chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
                            if chi2_p < 0.05:
                                issues[f'{feature}_category_distribution_shift'] = (
                                    f"Significant category distribution shift for {feature}: "
                                    f"Chi-square statistic={chi2_stat:.3f}, p-value={chi2_p:.3f}"
                                )
                    except:
                        # Skip if chi-square test fails
                        pass

        return issues

    def validate_processed_features(self, features: ProcessedFeatures) -> Dict[str, Any]:
        """
        Comprehensive validation of ProcessedFeatures object.

        Args:
            features: ProcessedFeatures object to validate

        Returns:
            Comprehensive validation report

        Educational Notes:
        - End-to-end feature quality assessment
        - Combines multiple validation checks
        - Provides actionable recommendations
        - Essential for ML pipeline quality control
        """
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(features.feature_names),
            'customer_count': len(features.customer_ids),
            'validation_passed': True,
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }

        # 1. Basic integrity checks
        try:
            features.validate()
            validation_report['basic_validation'] = 'PASSED'
        except ValueError as e:
            validation_report['basic_validation'] = 'FAILED'
            validation_report['critical_issues'].append(f"Basic validation failed: {str(e)}")
            validation_report['validation_passed'] = False

        # 2. Distribution validation
        distribution_stats = self.validate_feature_distributions(features.feature_matrix)
        validation_report['distribution_analysis'] = distribution_stats

        # Count features with issues
        features_with_issues = [
            feature for feature, stats in distribution_stats.items()
            if stats.get('validation_issues', [])
        ]

        if features_with_issues:
            validation_report['warnings'].append(
                f"{len(features_with_issues)} features have distribution issues"
            )

        # 3. Correlation analysis
        high_correlations = self.check_feature_correlations(features.feature_matrix)
        validation_report['correlation_analysis'] = {
            'high_correlation_pairs': len(high_correlations),
            'pairs': high_correlations[:10]  # Top 10 most correlated pairs
        }

        if len(high_correlations) > 0:
            validation_report['warnings'].append(
                f"Found {len(high_correlations)} highly correlated feature pairs"
            )

        # 4. Feature type distribution
        feature_types = {}
        for metadata in features.feature_metadata.values():
            feature_type = metadata.feature_type.value
            feature_types[feature_type] = feature_types.get(feature_type, 0) + 1

        validation_report['feature_type_distribution'] = feature_types

        # 5. Missing value analysis
        missing_analysis = {}
        total_missing = 0
        for feature in features.feature_names:
            missing_count = features.feature_matrix[feature].isnull().sum()
            if missing_count > 0:
                missing_analysis[feature] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(features.customer_ids) * 100)
                }
                total_missing += missing_count

        validation_report['missing_value_analysis'] = {
            'total_missing_values': total_missing,
            'features_with_missing': len(missing_analysis),
            'detailed_analysis': missing_analysis
        }

        # 6. Recommendations based on findings
        recommendations = []

        # High correlation recommendations
        if len(high_correlations) > 5:
            recommendations.append(
                "Consider feature selection to remove highly correlated features"
            )

        # Missing value recommendations
        if len(missing_analysis) > len(features.feature_names) * 0.3:
            recommendations.append(
                "High proportion of features have missing values - review imputation strategy"
            )

        # Feature engineering recommendations
        engineered_count = len(features.get_engineered_features())
        if engineered_count / len(features.feature_names) < 0.1:
            recommendations.append(
                "Consider additional feature engineering to improve model performance"
            )

        # Scaling recommendations
        numerical_features = features.get_numerical_features()
        if numerical_features:
            # Check if features appear to need scaling
            for feature in numerical_features[:5]:  # Check first 5 numerical features
                feature_data = features.feature_matrix[feature]
                if feature_data.std() > 10 or abs(feature_data.mean()) > 10:
                    recommendations.append(
                        "Some numerical features may benefit from scaling/normalization"
                    )
                    break

        validation_report['recommendations'] = recommendations

        # 7. Overall assessment
        critical_issue_count = len(validation_report['critical_issues'])
        warning_count = len(validation_report['warnings'])

        if critical_issue_count > 0:
            validation_report['overall_assessment'] = 'CRITICAL'
            validation_report['validation_passed'] = False
        elif warning_count > 3:
            validation_report['overall_assessment'] = 'NEEDS_ATTENTION'
        else:
            validation_report['overall_assessment'] = 'GOOD'

        # Store validation in history
        self.validation_history.append(validation_report)

        return validation_report

    def compare_feature_sets(self, features1: ProcessedFeatures, features2: ProcessedFeatures,
                           labels: Tuple[str, str] = ('Set 1', 'Set 2')) -> Dict[str, Any]:
        """
        Compare two feature sets for consistency and drift.

        Args:
            features1: First feature set (e.g., training)
            features2: Second feature set (e.g., test)
            labels: Labels for the two sets

        Returns:
            Comparison report

        Educational Notes:
        - Detects data drift between datasets
        - Validates train/test consistency
        - Essential for model monitoring
        - Helps identify preprocessing issues
        """
        comparison_report = {
            'timestamp': datetime.now().isoformat(),
            'set1_label': labels[0],
            'set2_label': labels[1],
            'set1_shape': features1.feature_matrix.shape,
            'set2_shape': features2.feature_matrix.shape,
            'comparison_passed': True,
            'issues': []
        }

        # Basic consistency checks
        consistency_issues = self.validate_encoding_consistency(
            features1.feature_matrix,
            features2.feature_matrix
        )

        if consistency_issues:
            comparison_report['issues'].extend(list(consistency_issues.values()))
            comparison_report['comparison_passed'] = False

        comparison_report['encoding_consistency'] = consistency_issues

        # Feature-by-feature comparison
        common_features = set(features1.feature_names) & set(features2.feature_names)
        feature_comparisons = {}

        for feature in common_features:
            data1 = features1.feature_matrix[feature]
            data2 = features2.feature_matrix[feature]

            feature_comparison = {
                'feature_type': features1.feature_metadata.get(feature, {}).get('feature_type', 'unknown'),
                'set1_stats': {},
                'set2_stats': {},
                'drift_detected': False
            }

            if pd.api.types.is_numeric_dtype(data1):
                # Numerical feature comparison
                feature_comparison['set1_stats'] = {
                    'mean': float(data1.mean()),
                    'std': float(data1.std()),
                    'median': float(data1.median()),
                    'min': float(data1.min()),
                    'max': float(data1.max())
                }

                feature_comparison['set2_stats'] = {
                    'mean': float(data2.mean()),
                    'std': float(data2.std()),
                    'median': float(data2.median()),
                    'min': float(data2.min()),
                    'max': float(data2.max())
                }

                # Statistical test for distribution difference
                try:
                    ks_stat, ks_p = ks_2samp(data1.dropna(), data2.dropna())
                    feature_comparison['ks_test'] = {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_p),
                        'significant': ks_p < 0.05
                    }
                    if ks_p < 0.05:
                        feature_comparison['drift_detected'] = True
                except:
                    feature_comparison['ks_test'] = None

            else:
                # Categorical feature comparison
                counts1 = data1.value_counts(normalize=True)
                counts2 = data2.value_counts(normalize=True)

                feature_comparison['set1_stats'] = {
                    'unique_count': data1.nunique(),
                    'most_frequent': str(counts1.index[0]) if len(counts1) > 0 else None,
                    'most_frequent_freq': float(counts1.iloc[0]) if len(counts1) > 0 else 0
                }

                feature_comparison['set2_stats'] = {
                    'unique_count': data2.nunique(),
                    'most_frequent': str(counts2.index[0]) if len(counts2) > 0 else None,
                    'most_frequent_freq': float(counts2.iloc[0]) if len(counts2) > 0 else 0
                }

                # Check for significant category distribution changes
                all_categories = set(counts1.index) | set(counts2.index)
                if len(all_categories) <= 20:  # Only for low cardinality features
                    max_freq_diff = 0
                    for category in all_categories:
                        freq1 = counts1.get(category, 0)
                        freq2 = counts2.get(category, 0)
                        max_freq_diff = max(max_freq_diff, abs(freq1 - freq2))

                    feature_comparison['max_frequency_difference'] = float(max_freq_diff)
                    if max_freq_diff > 0.1:  # 10% difference threshold
                        feature_comparison['drift_detected'] = True

            feature_comparisons[feature] = feature_comparison

        comparison_report['feature_comparisons'] = feature_comparisons

        # Summary statistics
        drift_count = sum(1 for comp in feature_comparisons.values() if comp['drift_detected'])
        comparison_report['drift_summary'] = {
            'features_with_drift': drift_count,
            'total_features_compared': len(common_features),
            'drift_percentage': drift_count / len(common_features) * 100 if common_features else 0
        }

        if drift_count > len(common_features) * 0.2:  # More than 20% of features show drift
            comparison_report['issues'].append(
                f"Significant data drift detected in {drift_count} features"
            )
            comparison_report['comparison_passed'] = False

        return comparison_report

    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get summary of all validation history.

        Returns:
            Summary of validation results across time
        """
        if not self.validation_history:
            return {'message': 'No validation history available'}

        summary = {
            'total_validations': len(self.validation_history),
            'recent_validation': self.validation_history[-1],
            'validation_trends': {}
        }

        # Analyze trends if we have multiple validations
        if len(self.validation_history) > 1:
            recent_assessments = [v['overall_assessment'] for v in self.validation_history[-5:]]
            summary['validation_trends'] = {
                'recent_assessments': recent_assessments,
                'improving': recent_assessments[-1] == 'GOOD' and any(a != 'GOOD' for a in recent_assessments[:-1]),
                'degrading': recent_assessments[-1] != 'GOOD' and recent_assessments[0] == 'GOOD'
            }

        return summary
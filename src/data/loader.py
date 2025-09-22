"""
ChurnDataLoader implementation for telecommunications churn prediction.

Educational Focus: Demonstrates professional data loading patterns with validation.
This module implements the DataLoaderContract for loading and processing churn data.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from datetime import datetime
import logging

# Import our contract and data models
from specs.contracts.data_loader import DataLoaderContract
from src.models.customer_profile import CustomerProfile, profiles_to_dataframe
from src.models.churn_label import ChurnLabel, labels_to_dataframe


class ChurnDataLoader(DataLoaderContract):
    """
    Implementation of DataLoaderContract for telecommunications churn data.

    Educational Notes:
    - Demonstrates proper separation of concerns
    - Implements comprehensive data validation
    - Provides clear error handling and logging
    - Supports multiple data formats and sources
    """

    def __init__(self, encoding: str = 'utf-8', delimiter: str = ','):
        """
        Initialize the data loader.

        Args:
            encoding: File encoding (default: utf-8)
            delimiter: CSV delimiter (default: comma)
        """
        self.encoding = encoding
        self.delimiter = delimiter
        self.logger = logging.getLogger(__name__)

        # Define expected columns for validation
        self.required_columns = {
            'customer_id', 'tenure', 'monthly_charges', 'total_charges',
            'contract_type', 'payment_method', 'churn'
        }

        # Define expected categorical values for validation
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

    def load_raw_data(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load raw customer and churn data from CSV file.

        Args:
            file_path: Path to CSV file containing customer data

        Returns:
            Tuple of (customer_features_df, churn_labels_df)

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data format is invalid

        Educational Notes:
        - Demonstrates proper file handling with validation
        - Separates features from target variable
        - Provides detailed error messages for debugging
        """
        self.logger.info(f"Loading data from: {file_path}")

        # Validate file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Validate file extension
        if not file_path.lower().endswith('.csv'):
            raise ValueError(f"Unsupported file format. Expected CSV, got: {file_path}")

        try:
            # Load CSV with error handling
            raw_data = pd.read_csv(
                file_path,
                encoding=self.encoding,
                delimiter=self.delimiter
            )

            self.logger.info(f"Loaded {len(raw_data)} rows and {len(raw_data.columns)} columns")

        except pd.errors.EmptyDataError:
            raise ValueError(f"CSV file is empty: {file_path}")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error reading file: {e}")

        # Validate basic structure
        if raw_data.empty:
            raise ValueError("Loaded data is empty")

        # Check for required columns
        missing_columns = self.required_columns - set(raw_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate churn column exists for splitting
        if 'churn' not in raw_data.columns:
            raise ValueError("Target column 'churn' not found in data")

        # Perform initial data cleaning
        cleaned_data = self._clean_raw_data(raw_data)

        # Split into features and target
        features_df, target_series = self.split_features_target(cleaned_data, 'churn')

        # Convert target series to DataFrame for consistent return type
        target_df = pd.DataFrame({'customer_id': features_df['customer_id'], 'churn': target_series})

        self.logger.info(f"Split data into {len(features_df)} feature rows and {len(target_df)} target rows")

        return features_df, target_df

    def _clean_raw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning and standardization.

        Args:
            df: Raw DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame

        Educational Notes:
        - Demonstrates common data cleaning patterns
        - Handles missing values appropriately
        - Standardizes data types and formats
        """
        cleaned_df = df.copy()

        # Handle total_charges column (often has string values)
        if 'total_charges' in cleaned_df.columns:
            # Convert non-numeric total_charges to numeric
            cleaned_df['total_charges'] = pd.to_numeric(
                cleaned_df['total_charges'],
                errors='coerce'
            )

        # Standardize categorical values
        for column, expected_values in self.categorical_specifications.items():
            if column in cleaned_df.columns:
                # Check for unexpected values and log warnings
                unexpected_values = set(cleaned_df[column].dropna().unique()) - set(expected_values)
                if unexpected_values:
                    self.logger.warning(f"Unexpected values in {column}: {unexpected_values}")

        # Convert churn to integer if it's not already
        if 'churn' in cleaned_df.columns:
            cleaned_df['churn'] = cleaned_df['churn'].astype(int)

        # Ensure customer_id is string
        if 'customer_id' in cleaned_df.columns:
            cleaned_df['customer_id'] = cleaned_df['customer_id'].astype(str)

        return cleaned_df

    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and return comprehensive quality report.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results including:
            - missing_percentages: Dict of column: missing_percentage
            - duplicate_count: Number of duplicate rows
            - invalid_values: List of columns with invalid values
            - quality_score: Overall quality score (0-1)

        Educational Notes:
        - Demonstrates systematic data quality assessment
        - Provides quantitative quality metrics
        - Identifies specific data issues for remediation
        """
        self.logger.info("Performing data quality validation")

        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_percentages': {},
            'duplicate_count': 0,
            'invalid_values': {},
            'data_type_issues': {},
            'quality_score': 0.0,
            'recommendations': []
        }

        # Check missing values
        missing_counts = df.isnull().sum()
        for column in df.columns:
            missing_pct = (missing_counts[column] / len(df)) * 100
            quality_report['missing_percentages'][column] = missing_pct

            if missing_pct > 20:
                quality_report['recommendations'].append(
                    f"High missing values in {column}: {missing_pct:.1f}%"
                )

        # Check for duplicate rows
        quality_report['duplicate_count'] = df.duplicated().sum()
        if quality_report['duplicate_count'] > 0:
            quality_report['recommendations'].append(
                f"Found {quality_report['duplicate_count']} duplicate rows"
            )

        # Check for invalid categorical values
        for column, expected_values in self.categorical_specifications.items():
            if column in df.columns:
                actual_values = set(df[column].dropna().unique())
                invalid_values = actual_values - set(expected_values)
                if invalid_values:
                    quality_report['invalid_values'][column] = list(invalid_values)
                    quality_report['recommendations'].append(
                        f"Invalid values in {column}: {invalid_values}"
                    )

        # Check numeric columns for reasonable ranges
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            if column in ['tenure', 'monthly_charges', 'total_charges']:
                negative_count = (df[column] < 0).sum()
                if negative_count > 0:
                    quality_report['invalid_values'][f"{column}_negative"] = negative_count
                    quality_report['recommendations'].append(
                        f"Found {negative_count} negative values in {column}"
                    )

                # Check for extremely large values
                q99 = df[column].quantile(0.99)
                outlier_threshold = q99 * 10  # 10x the 99th percentile
                extreme_count = (df[column] > outlier_threshold).sum()
                if extreme_count > 0:
                    quality_report['invalid_values'][f"{column}_extreme"] = extreme_count

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_report)
        quality_report['quality_score'] = quality_score

        # Add overall assessment
        if quality_score >= 0.8:
            quality_report['overall_assessment'] = "Good data quality"
        elif quality_score >= 0.6:
            quality_report['overall_assessment'] = "Acceptable data quality with minor issues"
        else:
            quality_report['overall_assessment'] = "Poor data quality - significant issues found"

        self.logger.info(f"Data quality validation complete. Score: {quality_score:.2f}")

        return quality_report

    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score (0-1).

        Args:
            quality_report: Quality report dictionary

        Returns:
            float: Quality score between 0 and 1

        Educational Notes:
        - Demonstrates quantitative quality assessment
        - Combines multiple quality dimensions
        - Provides actionable quality metric
        """
        score = 1.0

        # Penalize for missing values
        avg_missing = np.mean(list(quality_report['missing_percentages'].values()))
        missing_penalty = min(avg_missing / 20, 0.3)  # Max 30% penalty for missing values
        score -= missing_penalty

        # Penalize for duplicates
        duplicate_rate = quality_report['duplicate_count'] / quality_report['total_rows']
        duplicate_penalty = min(duplicate_rate * 2, 0.2)  # Max 20% penalty for duplicates
        score -= duplicate_penalty

        # Penalize for invalid values
        invalid_columns = len(quality_report['invalid_values'])
        total_columns = quality_report['total_columns']
        invalid_penalty = min((invalid_columns / total_columns) * 0.5, 0.3)  # Max 30% penalty
        score -= invalid_penalty

        return max(score, 0.0)  # Ensure score doesn't go below 0

    def split_features_target(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split DataFrame into features (X) and target (y).

        Args:
            df: Complete dataset
            target_col: Name of target column

        Returns:
            Tuple of (features_df, target_series)

        Raises:
            KeyError: If target column doesn't exist

        Educational Notes:
        - Standard ML preprocessing step
        - Ensures proper separation of features and target
        - Maintains data integrity and indexing
        """
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        # Create features DataFrame (everything except target)
        features_df = df.drop(columns=[target_col]).copy()

        # Create target Series
        target_series = df[target_col].copy()

        # Ensure indices are aligned
        if not features_df.index.equals(target_series.index):
            self.logger.warning("Features and target indices are not aligned")

        self.logger.info(f"Split complete: {len(features_df.columns)} features, {len(target_series)} targets")

        return features_df, target_series

    def load_from_dataframes(self, features_df: pd.DataFrame,
                           target_df: pd.DataFrame) -> Tuple[List[CustomerProfile], List[ChurnLabel]]:
        """
        Convert DataFrames to domain model objects.

        Args:
            features_df: DataFrame containing customer features
            target_df: DataFrame containing churn labels

        Returns:
            Tuple of (customer_profiles, churn_labels)

        Educational Notes:
        - Demonstrates conversion between data formats
        - Utilizes domain models for type safety
        - Provides validation through model constructors
        """
        self.logger.info("Converting DataFrames to domain objects")

        customer_profiles = []
        churn_labels = []

        # Align DataFrames by customer_id if needed
        if 'customer_id' in features_df.columns and 'customer_id' in target_df.columns:
            # Merge on customer_id to ensure alignment
            combined_df = features_df.merge(target_df, on='customer_id', how='inner')

            # Split back into features and targets
            feature_columns = [col for col in features_df.columns if col != 'customer_id']
            features_aligned = combined_df[['customer_id'] + feature_columns]
            target_aligned = combined_df[['customer_id', 'churn']]
        else:
            features_aligned = features_df
            target_aligned = target_df

        # Convert to CustomerProfile objects
        for _, row in features_aligned.iterrows():
            try:
                # Handle missing columns with defaults
                profile_data = self._prepare_customer_profile_data(row)
                profile = CustomerProfile.from_series(pd.Series(profile_data))
                customer_profiles.append(profile)
            except Exception as e:
                self.logger.warning(f"Failed to create CustomerProfile for row {row.name}: {e}")
                continue

        # Convert to ChurnLabel objects
        for _, row in target_aligned.iterrows():
            try:
                label = ChurnLabel(
                    customer_id=str(row['customer_id']),
                    churn=int(row['churn']),
                    observation_date=datetime.now()
                )
                churn_labels.append(label)
            except Exception as e:
                self.logger.warning(f"Failed to create ChurnLabel for row {row.name}: {e}")
                continue

        self.logger.info(f"Created {len(customer_profiles)} customer profiles and {len(churn_labels)} churn labels")

        return customer_profiles, churn_labels

    def _prepare_customer_profile_data(self, row: pd.Series) -> Dict[str, Any]:
        """
        Prepare customer profile data with defaults for missing fields.

        Args:
            row: Pandas Series containing customer data

        Returns:
            Dict[str, Any]: Complete customer profile data

        Educational Notes:
        - Handles missing data gracefully
        - Provides sensible defaults for required fields
        - Demonstrates defensive programming
        """
        # Define default values for required CustomerProfile fields
        defaults = {
            'gender': 'Unknown',
            'senior_citizen': 0,
            'partner': 'No',
            'dependents': 'No',
            'tenure': 0,
            'phone_service': 'No',
            'multiple_lines': 'No phone service',
            'internet_service': 'No',
            'online_security': 'No internet service',
            'online_backup': 'No internet service',
            'device_protection': 'No internet service',
            'tech_support': 'No internet service',
            'streaming_tv': 'No internet service',
            'streaming_movies': 'No internet service',
            'contract_type': 'month-to-month',
            'paperless_billing': 'No',
            'payment_method': 'electronic_check',
            'monthly_charges': 0.0,
            'total_charges': 0.0
        }

        # Start with defaults and update with actual values
        profile_data = defaults.copy()

        # Update with actual row data, handling NaN values
        for key in defaults.keys():
            if key in row and pd.notna(row[key]):
                profile_data[key] = row[key]

        # Ensure customer_id is always included
        if 'customer_id' in row:
            profile_data['customer_id'] = str(row['customer_id'])
        else:
            raise ValueError("customer_id is required but not found in row")

        return profile_data

    def save_processed_data(self, features_df: pd.DataFrame, target_df: pd.DataFrame,
                          output_dir: str, file_prefix: str = "processed") -> Dict[str, str]:
        """
        Save processed data to files for later use.

        Args:
            features_df: Processed features DataFrame
            target_df: Target labels DataFrame
            output_dir: Directory to save files
            file_prefix: Prefix for output filenames

        Returns:
            Dict[str, str]: Dictionary of saved file paths

        Educational Notes:
        - Demonstrates data persistence patterns
        - Enables reproducible ML workflows
        - Provides versioning and metadata
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = {}

        # Save features
        features_path = os.path.join(output_dir, f"{file_prefix}_features_{timestamp}.csv")
        features_df.to_csv(features_path, index=False)
        saved_files['features'] = features_path

        # Save targets
        target_path = os.path.join(output_dir, f"{file_prefix}_targets_{timestamp}.csv")
        target_df.to_csv(target_path, index=False)
        saved_files['targets'] = target_path

        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'features_shape': features_df.shape,
            'targets_shape': target_df.shape,
            'feature_columns': list(features_df.columns),
            'target_columns': list(target_df.columns)
        }

        metadata_path = os.path.join(output_dir, f"{file_prefix}_metadata_{timestamp}.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_files['metadata'] = metadata_path

        self.logger.info(f"Saved processed data to {output_dir}")

        return saved_files

    def get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data summary.

        Args:
            df: DataFrame to summarize

        Returns:
            Dict[str, Any]: Data summary with statistics and insights

        Educational Notes:
        - Provides quick data overview
        - Identifies potential issues
        - Supports data exploration phase
        """
        summary = {
            'basic_info': {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'missing_data': {
                'total_missing': df.isnull().sum().sum(),
                'columns_with_missing': (df.isnull().sum() > 0).sum(),
                'missing_by_column': df.isnull().sum().to_dict()
            },
            'data_types': df.dtypes.value_counts().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }

        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = {
                'count': len(numeric_cols),
                'columns': list(numeric_cols),
                'statistics': df[numeric_cols].describe().to_dict()
            }

        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_summary = {}
            for col in categorical_cols:
                unique_count = df[col].nunique()
                cat_summary[col] = {
                    'unique_values': unique_count,
                    'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    'top_values': df[col].value_counts().head(5).to_dict()
                }

            summary['categorical_summary'] = {
                'count': len(categorical_cols),
                'columns': list(categorical_cols),
                'details': cat_summary
            }

        return summary
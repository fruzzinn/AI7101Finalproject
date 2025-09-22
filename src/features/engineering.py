"""
Feature engineering utilities for telecommunications churn prediction.

Educational Focus: Demonstrates advanced feature engineering patterns and utilities.
This module provides utility functions and classes to support complex feature engineering
workflows for churn prediction models.
"""

from typing import List, Dict, Any, Tuple, Optional, Callable, Union
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import warnings
from functools import wraps

from ..models.processed_features import ProcessedFeatures, FeatureMetadata, FeatureType


def feature_engineering_logger(func):
    """
    Decorator to log feature engineering operations.

    Educational Notes:
    - Tracks all feature transformations
    - Enables reproducibility and debugging
    - Provides audit trail for model interpretation
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Log successful operation
            if hasattr(args[0], 'engineering_log'):
                args[0].engineering_log.append({
                    'function': func.__name__,
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': duration,
                    'status': 'success',
                    'args_summary': str(kwargs) if kwargs else 'default_params'
                })

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Log failed operation
            if hasattr(args[0], 'engineering_log'):
                args[0].engineering_log.append({
                    'function': func.__name__,
                    'timestamp': start_time.isoformat(),
                    'duration_seconds': duration,
                    'status': 'failed',
                    'error': str(e)
                })

            raise e

    return wrapper


class FeatureEngineeringUtils:
    """
    Utility class for advanced feature engineering operations.

    Educational Notes:
    - Provides reusable feature engineering patterns
    - Supports both automated and custom transformations
    - Maintains transformation history for reproducibility
    - Enables complex feature interaction discovery
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize FeatureEngineeringUtils.

        Args:
            random_state: Random seed for reproducible results
        """
        self.random_state = random_state
        self.engineering_log: List[Dict[str, Any]] = []
        self.fitted_transformers: Dict[str, Any] = {}

        # Set random seeds
        np.random.seed(self.random_state)

    @feature_engineering_logger
    def create_polynomial_features(self, df: pd.DataFrame, columns: List[str],
                                 degree: int = 2, include_bias: bool = False) -> pd.DataFrame:
        """
        Create polynomial features from specified columns.

        Args:
            df: Input DataFrame
            columns: Columns to create polynomial features from
            degree: Polynomial degree
            include_bias: Whether to include bias term

        Returns:
            DataFrame with polynomial features added

        Educational Notes:
        - Captures non-linear relationships
        - Useful for tree-based models and linear models
        - Can lead to feature explosion - use carefully
        - Helps model complex interactions between features
        """
        df_poly = df.copy()

        # Select only specified numerical columns that exist
        available_columns = [col for col in columns if col in df.columns]
        numerical_columns = [col for col in available_columns
                           if pd.api.types.is_numeric_dtype(df[col])]

        if not numerical_columns:
            warnings.warn("No numerical columns available for polynomial features")
            return df_poly

        # Handle missing values
        X = df[numerical_columns].fillna(0)

        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias,
                                interaction_only=False)
        poly_features = poly.fit_transform(X)

        # Get feature names
        feature_names = poly.get_feature_names_out(numerical_columns)

        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)

        # Remove original features (they're included in polynomial features)
        poly_df = poly_df.drop(columns=numerical_columns, errors='ignore')

        # Add polynomial features to original DataFrame
        df_poly = pd.concat([df_poly, poly_df], axis=1)

        # Store transformer
        self.fitted_transformers['polynomial'] = {
            'transformer': poly,
            'input_columns': numerical_columns,
            'output_columns': list(poly_df.columns),
            'degree': degree
        }

        return df_poly

    @feature_engineering_logger
    def create_interaction_features(self, df: pd.DataFrame, column_pairs: List[Tuple[str, str]],
                                  operations: List[str] = None) -> pd.DataFrame:
        """
        Create interaction features between specified column pairs.

        Args:
            df: Input DataFrame
            column_pairs: List of column pairs to create interactions for
            operations: List of operations ('multiply', 'add', 'subtract', 'divide', 'ratio')

        Returns:
            DataFrame with interaction features added

        Educational Notes:
        - Captures relationships between features
        - Domain knowledge can guide interaction selection
        - Multiplication often most useful for churn prediction
        - Ratios can reveal important business metrics
        """
        if operations is None:
            operations = ['multiply', 'add', 'ratio']

        df_interactions = df.copy()

        for col1, col2 in column_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                warnings.warn(f"Column pair ({col1}, {col2}) not found in DataFrame")
                continue

            # Ensure columns are numerical
            if not (pd.api.types.is_numeric_dtype(df[col1]) and
                   pd.api.types.is_numeric_dtype(df[col2])):
                continue

            for operation in operations:
                feature_name = f"{col1}_{operation}_{col2}"

                try:
                    if operation == 'multiply':
                        df_interactions[feature_name] = df[col1] * df[col2]
                    elif operation == 'add':
                        df_interactions[feature_name] = df[col1] + df[col2]
                    elif operation == 'subtract':
                        df_interactions[feature_name] = df[col1] - df[col2]
                    elif operation == 'divide':
                        # Handle division by zero
                        df_interactions[feature_name] = df[col1] / (df[col2] + 1e-6)
                    elif operation == 'ratio':
                        # Safe ratio calculation
                        df_interactions[feature_name] = df[col1] / (df[col1] + df[col2] + 1e-6)
                    else:
                        warnings.warn(f"Unknown operation: {operation}")
                        continue

                    # Handle infinite values
                    df_interactions[feature_name] = df_interactions[feature_name].replace(
                        [np.inf, -np.inf], np.nan
                    )

                except Exception as e:
                    warnings.warn(f"Failed to create interaction {feature_name}: {str(e)}")
                    continue

        return df_interactions

    @feature_engineering_logger
    def create_time_based_features(self, df: pd.DataFrame, date_column: str,
                                 reference_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Create time-based features from date column.

        Args:
            df: Input DataFrame
            date_column: Name of date column
            reference_date: Reference date for calculating differences

        Returns:
            DataFrame with time-based features added

        Educational Notes:
        - Extracts cyclical patterns from dates
        - Useful for seasonal churn patterns
        - Creates both linear and cyclical representations
        - Handles different date formats automatically
        """
        df_time = df.copy()

        if date_column not in df.columns:
            warnings.warn(f"Date column '{date_column}' not found")
            return df_time

        # Convert to datetime if not already
        try:
            df_time[date_column] = pd.to_datetime(df[date_column])
        except:
            warnings.warn(f"Could not convert '{date_column}' to datetime")
            return df_time

        if reference_date is None:
            reference_date = datetime.now()

        # Extract time components
        df_time[f'{date_column}_year'] = df_time[date_column].dt.year
        df_time[f'{date_column}_month'] = df_time[date_column].dt.month
        df_time[f'{date_column}_day'] = df_time[date_column].dt.day
        df_time[f'{date_column}_weekday'] = df_time[date_column].dt.weekday
        df_time[f'{date_column}_quarter'] = df_time[date_column].dt.quarter

        # Create cyclical features for seasonal patterns
        df_time[f'{date_column}_month_sin'] = np.sin(2 * np.pi * df_time[f'{date_column}_month'] / 12)
        df_time[f'{date_column}_month_cos'] = np.cos(2 * np.pi * df_time[f'{date_column}_month'] / 12)
        df_time[f'{date_column}_weekday_sin'] = np.sin(2 * np.pi * df_time[f'{date_column}_weekday'] / 7)
        df_time[f'{date_column}_weekday_cos'] = np.cos(2 * np.pi * df_time[f'{date_column}_weekday'] / 7)

        # Time since reference date
        df_time[f'{date_column}_days_since_ref'] = (reference_date - df_time[date_column]).dt.days

        # Boolean features
        df_time[f'{date_column}_is_weekend'] = (df_time[f'{date_column}_weekday'] >= 5).astype(int)
        df_time[f'{date_column}_is_month_start'] = df_time[date_column].dt.is_month_start.astype(int)
        df_time[f'{date_column}_is_month_end'] = df_time[date_column].dt.is_month_end.astype(int)

        return df_time

    @feature_engineering_logger
    def create_aggregation_features(self, df: pd.DataFrame, group_column: str,
                                  agg_columns: List[str],
                                  agg_functions: List[str] = None) -> pd.DataFrame:
        """
        Create aggregation features by grouping.

        Args:
            df: Input DataFrame
            group_column: Column to group by
            agg_columns: Columns to aggregate
            agg_functions: Aggregation functions to apply

        Returns:
            DataFrame with aggregation features added

        Educational Notes:
        - Creates customer segment-based features
        - Useful for behavioral pattern analysis
        - Can reveal hidden customer segments
        - Helps capture population-level statistics
        """
        if agg_functions is None:
            agg_functions = ['mean', 'std', 'min', 'max', 'count']

        df_agg = df.copy()

        if group_column not in df.columns:
            warnings.warn(f"Group column '{group_column}' not found")
            return df_agg

        # Filter to existing numerical columns
        existing_agg_columns = [col for col in agg_columns
                              if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if not existing_agg_columns:
            warnings.warn("No valid aggregation columns found")
            return df_agg

        try:
            # Create aggregations
            agg_dict = {}
            for col in existing_agg_columns:
                for func in agg_functions:
                    if func in ['mean', 'std', 'min', 'max', 'sum', 'count']:
                        agg_dict[f'{col}_{func}_by_{group_column}'] = (col, func)

            # Perform groupby aggregation
            grouped = df.groupby(group_column)[existing_agg_columns].agg(agg_functions)

            # Flatten column names
            grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

            # Add suffix to distinguish aggregation features
            grouped = grouped.add_suffix(f'_by_{group_column}')

            # Merge back to original DataFrame
            df_agg = df_agg.merge(grouped, left_on=group_column, right_index=True, how='left')

        except Exception as e:
            warnings.warn(f"Aggregation failed: {str(e)}")

        return df_agg

    @feature_engineering_logger
    def create_clustering_features(self, df: pd.DataFrame, columns: List[str],
                                 n_clusters: int = 5, cluster_prefix: str = 'cluster') -> pd.DataFrame:
        """
        Create clustering-based features.

        Args:
            df: Input DataFrame
            columns: Columns to use for clustering
            n_clusters: Number of clusters
            cluster_prefix: Prefix for cluster feature names

        Returns:
            DataFrame with clustering features added

        Educational Notes:
        - Discovers hidden customer segments
        - Provides non-linear feature combinations
        - K-means assumes spherical clusters
        - Cluster membership can be powerful feature
        """
        df_cluster = df.copy()

        # Select numerical columns that exist
        existing_columns = [col for col in columns
                          if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if len(existing_columns) < 2:
            warnings.warn("Need at least 2 numerical columns for clustering")
            return df_cluster

        # Prepare data for clustering
        X = df[existing_columns].fillna(0)

        # Standardize features for clustering
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)

            # Add cluster labels
            df_cluster[f'{cluster_prefix}_id'] = cluster_labels

            # Add distances to cluster centers
            distances = kmeans.transform(X_scaled)
            for i in range(n_clusters):
                df_cluster[f'{cluster_prefix}_dist_to_{i}'] = distances[:, i]

            # Add distance to assigned cluster center
            assigned_distances = distances[np.arange(len(distances)), cluster_labels]
            df_cluster[f'{cluster_prefix}_assigned_distance'] = assigned_distances

            # Store clustering information
            self.fitted_transformers['clustering'] = {
                'kmeans': kmeans,
                'scaler': scaler,
                'input_columns': existing_columns,
                'n_clusters': n_clusters
            }

        except Exception as e:
            warnings.warn(f"Clustering failed: {str(e)}")

        return df_cluster

    @feature_engineering_logger
    def create_statistical_features(self, df: pd.DataFrame, columns: List[str],
                                  window_size: Optional[int] = None) -> pd.DataFrame:
        """
        Create statistical features from numerical columns.

        Args:
            df: Input DataFrame
            columns: Columns to create statistical features from
            window_size: Rolling window size (None for global statistics)

        Returns:
            DataFrame with statistical features added

        Educational Notes:
        - Captures distribution characteristics
        - Rolling statistics reveal trends
        - Useful for time series and sequential data
        - Helps identify outliers and anomalies
        """
        df_stats = df.copy()

        # Select numerical columns that exist
        numerical_columns = [col for col in columns
                           if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if not numerical_columns:
            warnings.warn("No numerical columns found for statistical features")
            return df_stats

        for col in numerical_columns:
            data = df[col]

            # Basic statistical features
            df_stats[f'{col}_zscore'] = (data - data.mean()) / (data.std() + 1e-6)
            df_stats[f'{col}_iqr'] = data.quantile(0.75) - data.quantile(0.25)
            df_stats[f'{col}_outlier_iqr'] = ((data < (data.quantile(0.25) - 1.5 * df_stats[f'{col}_iqr'])) |
                                            (data > (data.quantile(0.75) + 1.5 * df_stats[f'{col}_iqr']))).astype(int)

            # Percentile-based features
            df_stats[f'{col}_percentile_rank'] = data.rank(pct=True)
            df_stats[f'{col}_is_top_10pct'] = (data >= data.quantile(0.9)).astype(int)
            df_stats[f'{col}_is_bottom_10pct'] = (data <= data.quantile(0.1)).astype(int)

            # Distribution shape features
            try:
                from scipy import stats
                df_stats[f'{col}_skewness'] = stats.skew(data.dropna())
                df_stats[f'{col}_kurtosis'] = stats.kurtosis(data.dropna())
            except:
                df_stats[f'{col}_skewness'] = 0
                df_stats[f'{col}_kurtosis'] = 0

            # Rolling statistics if window size specified
            if window_size is not None and window_size > 1:
                df_stats[f'{col}_rolling_mean_{window_size}'] = data.rolling(window=window_size).mean()
                df_stats[f'{col}_rolling_std_{window_size}'] = data.rolling(window=window_size).std()
                df_stats[f'{col}_rolling_min_{window_size}'] = data.rolling(window=window_size).min()
                df_stats[f'{col}_rolling_max_{window_size}'] = data.rolling(window=window_size).max()

                # Trend features
                df_stats[f'{col}_trend_{window_size}'] = (data - data.rolling(window=window_size).mean()) / (data.rolling(window=window_size).std() + 1e-6)

        return df_stats

    @feature_engineering_logger
    def create_binning_features(self, df: pd.DataFrame, columns: List[str],
                              n_bins: int = 5, strategy: str = 'quantile') -> pd.DataFrame:
        """
        Create binned versions of numerical features.

        Args:
            df: Input DataFrame
            columns: Columns to create bins for
            n_bins: Number of bins
            strategy: Binning strategy ('quantile', 'uniform', 'kmeans')

        Returns:
            DataFrame with binned features added

        Educational Notes:
        - Converts continuous to categorical features
        - Can help linear models capture non-linearities
        - Quantile binning ensures balanced bins
        - K-means binning adapts to data distribution
        """
        df_binned = df.copy()

        # Select numerical columns that exist
        numerical_columns = [col for col in columns
                           if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if not numerical_columns:
            warnings.warn("No numerical columns found for binning")
            return df_binned

        for col in numerical_columns:
            data = df[col].dropna()

            if len(data.unique()) < n_bins:
                warnings.warn(f"Column {col} has fewer unique values than requested bins")
                continue

            try:
                if strategy == 'quantile':
                    # Equal-frequency binning
                    binned, bin_edges = pd.qcut(data, q=n_bins, retbins=True, duplicates='drop')
                    df_binned[f'{col}_binned_quantile'] = pd.qcut(df[col], q=n_bins,
                                                                labels=False, duplicates='drop')

                elif strategy == 'uniform':
                    # Equal-width binning
                    binned, bin_edges = pd.cut(data, bins=n_bins, retbins=True)
                    df_binned[f'{col}_binned_uniform'] = pd.cut(df[col], bins=n_bins, labels=False)

                elif strategy == 'kmeans':
                    # K-means binning
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_bins, random_state=self.random_state, n_init=10)
                    cluster_labels = kmeans.fit_predict(data.values.reshape(-1, 1))

                    # Map back to full dataset
                    bin_mapping = pd.Series(cluster_labels, index=data.index)
                    df_binned[f'{col}_binned_kmeans'] = df[col].map(lambda x: bin_mapping.get(x, -1) if pd.notna(x) else -1)

                else:
                    warnings.warn(f"Unknown binning strategy: {strategy}")
                    continue

                # Create binary indicators for each bin
                bin_column = f'{col}_binned_{strategy}'
                if bin_column in df_binned.columns:
                    for bin_val in range(n_bins):
                        df_binned[f'{col}_bin_{bin_val}_{strategy}'] = (df_binned[bin_column] == bin_val).astype(int)

            except Exception as e:
                warnings.warn(f"Binning failed for {col}: {str(e)}")
                continue

        return df_binned

    @feature_engineering_logger
    def create_pca_features(self, df: pd.DataFrame, columns: List[str],
                          n_components: Optional[int] = None,
                          explained_variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Create PCA-based features for dimensionality reduction.

        Args:
            df: Input DataFrame
            columns: Columns to apply PCA to
            n_components: Number of components (None for automatic)
            explained_variance_threshold: Variance threshold for automatic component selection

        Returns:
            DataFrame with PCA features added

        Educational Notes:
        - Reduces dimensionality while preserving variance
        - Creates orthogonal features
        - Useful for high-dimensional data
        - Can help with multicollinearity
        """
        df_pca = df.copy()

        # Select numerical columns that exist
        numerical_columns = [col for col in columns
                           if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

        if len(numerical_columns) < 2:
            warnings.warn("Need at least 2 numerical columns for PCA")
            return df_pca

        # Prepare data
        X = df[numerical_columns].fillna(0)

        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        try:
            # Determine number of components
            if n_components is None:
                # Use explained variance threshold
                pca_temp = PCA()
                pca_temp.fit(X_scaled)
                cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
                n_components = np.argmax(cumsum_variance >= explained_variance_threshold) + 1
                n_components = min(n_components, len(numerical_columns))

            # Apply PCA
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca_features = pca.fit_transform(X_scaled)

            # Create PCA feature columns
            pca_columns = [f'pca_component_{i}' for i in range(n_components)]
            pca_df = pd.DataFrame(pca_features, columns=pca_columns, index=df.index)

            # Add PCA features to original DataFrame
            df_pca = pd.concat([df_pca, pca_df], axis=1)

            # Store PCA information
            self.fitted_transformers['pca'] = {
                'pca': pca,
                'scaler': scaler,
                'input_columns': numerical_columns,
                'output_columns': pca_columns,
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(np.sum(pca.explained_variance_ratio_))
            }

        except Exception as e:
            warnings.warn(f"PCA failed: {str(e)}")

        return df_pca

    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformations to new data.

        Args:
            df: New data to transform

        Returns:
            Transformed DataFrame

        Educational Notes:
        - Ensures consistent preprocessing for new data
        - Uses previously fitted parameters
        - Critical for production deployment
        """
        df_transformed = df.copy()

        # Apply fitted transformations in order
        for transform_name, transform_info in self.fitted_transformers.items():
            try:
                if transform_name == 'polynomial':
                    poly = transform_info['transformer']
                    input_cols = transform_info['input_columns']

                    if all(col in df_transformed.columns for col in input_cols):
                        X = df_transformed[input_cols].fillna(0)
                        poly_features = poly.transform(X)
                        feature_names = poly.get_feature_names_out(input_cols)
                        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
                        poly_df = poly_df.drop(columns=input_cols, errors='ignore')
                        df_transformed = pd.concat([df_transformed, poly_df], axis=1)

                elif transform_name == 'clustering':
                    kmeans = transform_info['kmeans']
                    scaler = transform_info['scaler']
                    input_cols = transform_info['input_columns']

                    if all(col in df_transformed.columns for col in input_cols):
                        X = df_transformed[input_cols].fillna(0)
                        X_scaled = scaler.transform(X)
                        cluster_labels = kmeans.predict(X_scaled)
                        distances = kmeans.transform(X_scaled)

                        df_transformed['cluster_id'] = cluster_labels
                        for i in range(transform_info['n_clusters']):
                            df_transformed[f'cluster_dist_to_{i}'] = distances[:, i]

                elif transform_name == 'pca':
                    pca = transform_info['pca']
                    scaler = transform_info['scaler']
                    input_cols = transform_info['input_columns']
                    output_cols = transform_info['output_columns']

                    if all(col in df_transformed.columns for col in input_cols):
                        X = df_transformed[input_cols].fillna(0)
                        X_scaled = scaler.transform(X)
                        pca_features = pca.transform(X_scaled)
                        pca_df = pd.DataFrame(pca_features, columns=output_cols, index=df.index)
                        df_transformed = pd.concat([df_transformed, pca_df], axis=1)

            except Exception as e:
                warnings.warn(f"Failed to apply transformation {transform_name}: {str(e)}")
                continue

        return df_transformed

    def get_engineering_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of feature engineering operations.

        Returns:
            Summary of all feature engineering activities
        """
        summary = {
            'total_operations': len(self.engineering_log),
            'successful_operations': len([op for op in self.engineering_log if op['status'] == 'success']),
            'failed_operations': len([op for op in self.engineering_log if op['status'] == 'failed']),
            'fitted_transformers': list(self.fitted_transformers.keys()),
            'operation_history': self.engineering_log
        }

        # Calculate total engineering time
        total_time = sum(op.get('duration_seconds', 0) for op in self.engineering_log)
        summary['total_engineering_time_seconds'] = total_time

        # Most used operations
        operation_counts = {}
        for op in self.engineering_log:
            func_name = op['function']
            operation_counts[func_name] = operation_counts.get(func_name, 0) + 1

        summary['operation_usage'] = operation_counts

        return summary


# Standalone utility functions

def detect_feature_types(df: pd.DataFrame) -> Dict[str, FeatureType]:
    """
    Automatically detect feature types in a DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary mapping column names to FeatureType

    Educational Notes:
    - Automates feature type detection
    - Useful for initial data exploration
    - Can guide preprocessing decisions
    """
    feature_types = {}

    for column in df.columns:
        data = df[column]

        if pd.api.types.is_numeric_dtype(data):
            # Check if binary
            unique_values = data.dropna().unique()
            if len(unique_values) == 2 and all(val in [0, 1] for val in unique_values):
                feature_types[column] = FeatureType.BINARY
            else:
                feature_types[column] = FeatureType.NUMERICAL
        else:
            # Categorical
            unique_count = data.nunique()
            if unique_count == 2:
                feature_types[column] = FeatureType.BINARY
            elif unique_count <= 10:
                feature_types[column] = FeatureType.CATEGORICAL
            else:
                feature_types[column] = FeatureType.CATEGORICAL

    return feature_types


def suggest_feature_engineering(df: pd.DataFrame, target_column: str = None) -> Dict[str, List[str]]:
    """
    Suggest feature engineering opportunities based on data characteristics.

    Args:
        df: Input DataFrame
        target_column: Name of target column

    Returns:
        Dictionary of suggestions by category

    Educational Notes:
    - Provides automated feature engineering recommendations
    - Based on data characteristics and domain knowledge
    - Helps data scientists identify opportunities
    """
    suggestions = {
        'polynomial_features': [],
        'interaction_features': [],
        'binning_candidates': [],
        'clustering_candidates': [],
        'statistical_features': [],
        'time_features': []
    }

    numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Remove target column from analysis
    if target_column and target_column in numerical_columns:
        numerical_columns.remove(target_column)
    if target_column and target_column in categorical_columns:
        categorical_columns.remove(target_column)

    # Polynomial feature suggestions
    for col in numerical_columns:
        if df[col].std() > 0:  # Has variance
            correlation_with_target = 0
            if target_column and target_column in df.columns:
                try:
                    correlation_with_target = abs(df[col].corr(df[target_column]))
                except:
                    correlation_with_target = 0

            if correlation_with_target > 0.3:  # Moderately correlated with target
                suggestions['polynomial_features'].append(col)

    # Interaction feature suggestions
    if len(numerical_columns) >= 2:
        # Suggest interactions between related features
        for i, col1 in enumerate(numerical_columns):
            for col2 in numerical_columns[i+1:]:
                # Check if columns might be related (similar names or correlation)
                if any(word in col1.lower() and word in col2.lower()
                      for word in ['charge', 'service', 'time', 'count']):
                    suggestions['interaction_features'].append((col1, col2))

    # Binning suggestions
    for col in numerical_columns:
        unique_count = df[col].nunique()
        if unique_count > 10:  # High cardinality numerical feature
            suggestions['binning_candidates'].append(col)

    # Clustering suggestions
    if len(numerical_columns) >= 3:
        suggestions['clustering_candidates'] = numerical_columns[:5]  # Top 5 numerical features

    # Statistical features
    for col in numerical_columns:
        if df[col].std() > 0 and len(df[col].dropna()) > 100:  # Sufficient data
            suggestions['statistical_features'].append(col)

    # Time feature suggestions
    for col in df.columns:
        if any(time_word in col.lower() for time_word in ['date', 'time', 'timestamp']):
            suggestions['time_features'].append(col)

    return suggestions
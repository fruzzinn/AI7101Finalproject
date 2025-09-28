"""
Feature Engineering Pipeline
Comprehensive feature engineering for customer churn prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class ChurnFeatureEngineer:
    """
    Comprehensive feature engineering pipeline for customer churn prediction
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize the feature engineering pipeline

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.fitted_transformers = {}
        self.feature_names = []
        self.engineered_features = []

        # Set random seed
        np.random.seed(random_state)

    def create_demographic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic-based features

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with demographic features
        """
        df = data.copy()
        features_added = []

        # Age-based features
        if 'age' in df.columns:
            # Age groups
            df['age_group'] = pd.cut(df['age'],
                                   bins=[0, 30, 45, 60, 100],
                                   labels=['Young', 'Middle-aged', 'Senior', 'Elder'])

            # Age squared (non-linear relationship)
            df['age_squared'] = df['age'] ** 2

            # Age standardized within age group
            df['age_within_group'] = df.groupby('age_group', observed=True)['age'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

            # Convert age_group to numeric if it's categorical
            if df['age_group'].dtype.name == 'category':
                df['age_group'] = df['age_group'].cat.codes

            features_added.extend(['age_group', 'age_squared', 'age_within_group'])

        # Gender-based features (if available)
        if 'gender' in df.columns:
            # Gender encoding
            df['is_male'] = (df['gender'].str.lower() == 'male').astype(int)
            features_added.append('is_male')

        self.engineered_features.extend(features_added)
        return df

    def create_tenure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based features

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with tenure features
        """
        df = data.copy()
        features_added = []

        if 'tenure' in df.columns:
            # Convert tenure to years
            df['tenure_years'] = df['tenure'] / 12

            # Tenure categories
            df['tenure_category'] = pd.cut(df['tenure'],
                                         bins=[0, 6, 12, 24, 36, float('inf')],
                                         labels=['Very_new', 'New', 'Developing', 'Established', 'Loyal'])

            # Convert to numeric
            if df['tenure_category'].dtype.name == 'category':
                df['tenure_category'] = df['tenure_category'].cat.codes

            # Tenure milestones
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
            df['is_loyal_customer'] = (df['tenure'] >= 36).astype(int)
            df['tenure_squared'] = df['tenure'] ** 2
            df['tenure_log'] = np.log1p(df['tenure'])

            # Tenure within service type groups (if applicable)
            if 'internet_service' in df.columns:
                df['tenure_within_service'] = df.groupby('internet_service')['tenure'].transform(
                    lambda x: (x - x.mean()) / x.std()
                )
                features_added.append('tenure_within_service')

            features_added.extend([
                'tenure_years', 'tenure_category', 'is_new_customer',
                'is_loyal_customer', 'tenure_squared', 'tenure_log'
            ])

        self.engineered_features.extend(features_added)
        return df

    def create_financial_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial and pricing-based features

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with financial features
        """
        df = data.copy()
        features_added = []

        # Monthly charges features
        if 'monthly_charges' in df.columns:
            # Price categories
            df['price_category'] = pd.cut(df['monthly_charges'],
                                        bins=[0, 35, 65, 95, float('inf')],
                                        labels=['Budget', 'Standard', 'Premium', 'Enterprise'])
            # Convert to numeric
            if df['price_category'].dtype.name == 'category':
                df['price_category'] = df['price_category'].cat.codes

            # Price per unit features
            if 'tenure' in df.columns:
                df['monthly_charges_per_tenure'] = df['monthly_charges'] / (df['tenure'] + 1)
                features_added.append('monthly_charges_per_tenure')

            # Price standardization
            df['monthly_charges_normalized'] = (df['monthly_charges'] - df['monthly_charges'].mean()) / df['monthly_charges'].std()

            features_added.extend(['price_category', 'monthly_charges_normalized'])

        # Total charges features
        if 'total_charges' in df.columns:
            # Handle potential string formatting issues
            df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')

            # Average monthly spending
            if 'tenure' in df.columns:
                df['avg_monthly_spend'] = df['total_charges'] / (df['tenure'] + 1)
                features_added.append('avg_monthly_spend')

            # Total charges categories
            df['total_charges_category'] = pd.qcut(df['total_charges'].fillna(0),
                                                 q=4,
                                                 labels=['Low_spend', 'Medium_spend', 'High_spend', 'Premium_spend'],
                                                 duplicates='drop')
            # Convert to numeric
            if df['total_charges_category'].dtype.name == 'category':
                df['total_charges_category'] = df['total_charges_category'].cat.codes

            # Customer lifetime value estimate
            if 'monthly_charges' in df.columns and 'tenure' in df.columns:
                df['clv_estimate'] = df['total_charges'] + (df['monthly_charges'] * 12)  # Current + 1 year projection
                df['payment_consistency'] = df['total_charges'] / (df['monthly_charges'] * df['tenure'] + 1)
                df['charges_per_tenure'] = df['total_charges'] / (df['tenure'] + 1)
                features_added.extend(['clv_estimate', 'payment_consistency', 'charges_per_tenure'])

        # High value customer identification
        if 'total_charges' in df.columns:
            threshold = df['total_charges'].quantile(0.75)  # Top 25%
            df['high_value_customer'] = (df['total_charges'] >= threshold).astype(int)
            features_added.append('high_value_customer')

            features_added.append('total_charges_category')

        # Payment method risk scoring
        if 'payment_method' in df.columns:
            # High-risk payment methods
            high_risk_methods = ['Electronic check', 'Mailed check']
            df['payment_risk_score'] = df['payment_method'].isin(high_risk_methods).astype(int)

            # Automatic payment methods
            automatic_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
            df['automatic_payment'] = df['payment_method'].isin(automatic_methods).astype(int)

            features_added.extend(['payment_risk_score', 'automatic_payment'])

        self.engineered_features.extend(features_added)
        return df

    def create_service_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create service usage and bundling features

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with service features
        """
        df = data.copy()
        features_added = []

        # Service bundling score
        service_columns = [col for col in df.columns if any(service in col.lower()
                          for service in ['phone', 'internet', 'online', 'backup', 'protection', 'support', 'streaming'])]

        if service_columns:
            # Convert Yes/No to 1/0 for service columns
            for col in service_columns:
                if df[col].dtype == 'object':
                    df[col + '_binary'] = (df[col].str.lower() == 'yes').astype(int)
                    service_columns.append(col + '_binary')

            # Service bundle count
            binary_service_cols = [col for col in service_columns if col.endswith('_binary')]
            if binary_service_cols:
                df['service_bundle_count'] = df[binary_service_cols].sum(axis=1)
                features_added.append('service_bundle_count')

            # Service diversity score
            if len(binary_service_cols) > 0:
                df['service_diversity'] = df['service_bundle_count'] / len(binary_service_cols)
                features_added.append('service_diversity')

        # Contract type features
        if 'contract_type' in df.columns:
            # Contract risk (month-to-month is higher risk)
            df['contract_risk'] = (df['contract_type'] == 'Month-to-month').astype(int)

            # Contract length mapping
            contract_mapping = {'Month-to-month': 1, 'One year': 12, 'Two year': 24}
            df['contract_length_months'] = df['contract_type'].map(contract_mapping)

            features_added.extend(['contract_risk', 'contract_length_months'])

        # Internet service features
        if 'internet_service' in df.columns:
            # Fiber optic vs others
            df['has_fiber'] = (df['internet_service'] == 'Fiber optic').astype(int)
            df['has_internet'] = (df['internet_service'] != 'No').astype(int)

            features_added.extend(['has_fiber', 'has_internet'])

        self.engineered_features.extend(features_added)
        return df

    def create_behavioral_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create behavioral and interaction features

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with behavioral features
        """
        df = data.copy()
        features_added = []

        # Customer support interaction features
        if 'tech_support' in df.columns:
            df['uses_tech_support'] = (df['tech_support'].str.lower() == 'yes').astype(int)
            features_added.append('uses_tech_support')

        # Device protection and online security
        protection_cols = [col for col in df.columns if 'protection' in col.lower() or 'security' in col.lower()]
        if protection_cols:
            for col in protection_cols:
                if df[col].dtype == 'object':
                    df[f'{col}_binary'] = (df[col].str.lower() == 'yes').astype(int)
                    features_added.append(f'{col}_binary')

        # Streaming services adoption
        streaming_cols = [col for col in df.columns if 'streaming' in col.lower()]
        if streaming_cols:
            streaming_binary_cols = []
            for col in streaming_cols:
                if df[col].dtype == 'object':
                    binary_col = f'{col}_binary'
                    df[binary_col] = (df[col].str.lower() == 'yes').astype(int)
                    streaming_binary_cols.append(binary_col)
                    features_added.append(binary_col)

            # Streaming adoption score
            if streaming_binary_cols:
                df['streaming_adoption'] = df[streaming_binary_cols].sum(axis=1)
                features_added.append('streaming_adoption')

        # Early adoption indicators (new services with short tenure)
        if 'tenure' in df.columns and 'service_bundle_count' in df.columns:
            df['early_adopter_score'] = df['service_bundle_count'] / (df['tenure'] + 1)
            features_added.append('early_adopter_score')

        self.engineered_features.extend(features_added)
        return df

    def create_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with interaction features
        """
        df = data.copy()
        features_added = []

        # Tenure-Price interactions
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            df['tenure_price_interaction'] = df['tenure'] * df['monthly_charges']
            df['tenure_price_ratio'] = df['tenure'] / (df['monthly_charges'] + 1)
            features_added.extend(['tenure_price_interaction', 'tenure_price_ratio'])

        # Age-Tenure interactions
        if 'age' in df.columns and 'tenure' in df.columns:
            df['age_tenure_interaction'] = df['age'] * df['tenure']
            df['age_tenure_ratio'] = df['age'] / (df['tenure'] + 1)
            features_added.extend(['age_tenure_interaction', 'age_tenure_ratio'])

        # Service-Price interactions
        if 'service_bundle_count' in df.columns and 'monthly_charges' in df.columns:
            df['service_price_efficiency'] = df['service_bundle_count'] / (df['monthly_charges'] + 1)
            features_added.append('service_price_efficiency')

        # Contract-Price interactions
        if 'contract_length_months' in df.columns and 'monthly_charges' in df.columns:
            df['contract_price_commitment'] = df['contract_length_months'] * df['monthly_charges']
            features_added.append('contract_price_commitment')

        self.engineered_features.extend(features_added)
        return df

    def create_risk_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk-based features for churn prediction

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with risk features
        """
        df = data.copy()
        features_added = []

        # Contract risk - month-to-month contracts are higher risk
        if 'contract' in df.columns:
            if df['contract'].dtype == 'object':
                df['contract_risk'] = (df['contract'].str.contains('Month', case=False, na=False)).astype(int)
            else:
                # For encoded data, look for month-to-month encoded columns
                contract_month_cols = [col for col in df.columns if 'contract' in col.lower() and 'month' in col.lower()]
                if contract_month_cols:
                    df['contract_risk'] = df[contract_month_cols[0]]
                else:
                    df['contract_risk'] = 0
            features_added.append('contract_risk')

        # Payment method risk - electronic check is higher risk
        if 'payment_method' in df.columns:
            if df['payment_method'].dtype == 'object':
                df['payment_risk'] = (df['payment_method'].str.contains('Electronic check', case=False, na=False)).astype(int)
            else:
                # For encoded data, look for electronic check encoded columns
                payment_electronic_cols = [col for col in df.columns if 'payment' in col.lower() and 'electronic' in col.lower()]
                if payment_electronic_cols:
                    df['payment_risk'] = df[payment_electronic_cols[0]]
                else:
                    df['payment_risk'] = 0
            features_added.append('payment_risk')

        # Tenure risk - new customers are higher risk
        if 'tenure' in df.columns:
            df['tenure_risk_score'] = np.where(df['tenure'] <= 12, 1, 0)
            features_added.append('tenure_risk_score')

        # Service risk - paperless billing can indicate risk
        if 'paperless_billing' in df.columns:
            if df['paperless_billing'].dtype == 'object':
                df['paperless_risk'] = (df['paperless_billing'].str.contains('Yes', case=False, na=False)).astype(int)
            else:
                # For encoded data, assume 1 means "Yes"
                df['paperless_risk'] = df['paperless_billing']
            features_added.append('paperless_risk')

        # Multiple service risk - customers with fewer services are at higher risk
        service_cols = [col for col in df.columns if any(service in col.lower()
                       for service in ['internet', 'phone', 'tv', 'streaming', 'security', 'backup', 'protection'])]
        if service_cols:
            # Check if columns are string type or encoded
            if df[service_cols[0]].dtype == 'object':
                # String data - use string methods
                df['service_count'] = df[service_cols].apply(lambda x: (x.str.contains('Yes', case=False, na=False)).sum(), axis=1)
            else:
                # Encoded data - assume binary encoding where 1 means "Yes"
                df['service_count'] = df[service_cols].sum(axis=1)

            df['low_service_risk'] = (df['service_count'] <= 2).astype(int)
            features_added.extend(['service_count', 'low_service_risk'])

        # Create composite risk score
        risk_components = []
        if 'contract_risk' in df.columns:
            risk_components.append('contract_risk')
        if 'payment_risk' in df.columns:
            risk_components.append('payment_risk')
        if 'tenure_risk_score' in df.columns:
            risk_components.append('tenure_risk_score')
        if 'low_service_risk' in df.columns:
            risk_components.append('low_service_risk')

        if risk_components:
            df['risk_score'] = df[risk_components].mean(axis=1)
            df['high_risk_customer'] = (df['risk_score'] > 0.5).astype(int)
            features_added.extend(['risk_score', 'high_risk_customer'])

        self.engineered_features.extend(features_added)
        return df

    def create_risk_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk scores

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with risk scores
        """
        df = data.copy()
        features_added = []

        risk_factors = []

        # Tenure risk (higher risk for newer customers)
        if 'tenure' in df.columns:
            df['tenure_risk'] = 1 / (df['tenure'] + 1)
            risk_factors.append('tenure_risk')

        # Contract risk
        if 'contract_risk' in df.columns:
            risk_factors.append('contract_risk')

        # Payment risk
        if 'payment_risk_score' in df.columns:
            risk_factors.append('payment_risk_score')

        # Price sensitivity risk
        if 'monthly_charges' in df.columns:
            df['price_sensitivity_risk'] = (df['monthly_charges'] - df['monthly_charges'].median()) / df['monthly_charges'].std()
            risk_factors.append('price_sensitivity_risk')

        # Service adoption risk (low adoption = higher risk)
        if 'service_bundle_count' in df.columns:
            max_services = df['service_bundle_count'].max()
            df['service_adoption_risk'] = 1 - (df['service_bundle_count'] / max_services)
            risk_factors.append('service_adoption_risk')

        # Composite risk score
        if risk_factors:
            # Normalize each risk factor to 0-1 scale
            for factor in risk_factors:
                min_val = df[factor].min()
                max_val = df[factor].max()
                if max_val > min_val:
                    df[f'{factor}_normalized'] = (df[factor] - min_val) / (max_val - min_val)
                else:
                    df[f'{factor}_normalized'] = 0

            # Calculate composite risk score
            normalized_factors = [f'{factor}_normalized' for factor in risk_factors]
            df['composite_risk_score'] = df[normalized_factors].mean(axis=1)

            features_added.extend([f'{factor}_normalized' for factor in risk_factors])
            features_added.append('composite_risk_score')

        self.engineered_features.extend(features_added)
        return df

    def apply_feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply complete feature engineering pipeline

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with all engineered features
        """
        print("Starting feature engineering pipeline...")

        # Reset engineered features list
        self.engineered_features = []

        # Apply all feature engineering steps
        df = data.copy()

        print("Creating demographic features...")
        df = self.create_demographic_features(df)

        print("Creating tenure features...")
        df = self.create_tenure_features(df)

        print("Creating financial features...")
        df = self.create_financial_features(df)

        print("Creating service features...")
        df = self.create_service_features(df)

        print("Creating behavioral features...")
        df = self.create_behavioral_features(df)

        print("Creating interaction features...")
        df = self.create_interaction_features(df)

        print("Creating risk scores...")
        df = self.create_risk_scores(df)

        print(f"Feature engineering complete. Added {len(self.engineered_features)} new features.")

        return df

    def get_feature_summary(self) -> Dict[str, Any]:
        """
        Get summary of engineered features

        Returns:
            Dictionary with feature engineering summary
        """
        return {
            'total_engineered_features': len(self.engineered_features),
            'engineered_features': self.engineered_features,
            'feature_categories': {
                'demographic': [f for f in self.engineered_features if any(term in f for term in ['age', 'gender'])],
                'tenure': [f for f in self.engineered_features if 'tenure' in f],
                'financial': [f for f in self.engineered_features if any(term in f for term in ['charge', 'price', 'payment', 'clv'])],
                'service': [f for f in self.engineered_features if any(term in f for term in ['service', 'bundle', 'contract'])],
                'behavioral': [f for f in self.engineered_features if any(term in f for term in ['support', 'streaming', 'adoption'])],
                'interaction': [f for f in self.engineered_features if 'interaction' in f or 'ratio' in f],
                'risk': [f for f in self.engineered_features if 'risk' in f]
            }
        }


def create_feature_engineer(random_state: int = 42) -> ChurnFeatureEngineer:
    """
    Factory function to create a feature engineer

    Args:
        random_state: Random seed for reproducibility

    Returns:
        Configured ChurnFeatureEngineer
    """
    return ChurnFeatureEngineer(random_state=random_state)


def engineer_features_for_churn(data: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to apply feature engineering

    Args:
        data: Input DataFrame
        random_state: Random seed

    Returns:
        Tuple of (engineered_data, feature_summary)
    """
    engineer = create_feature_engineer(random_state)
    engineered_data = engineer.apply_feature_engineering(data)
    summary = engineer.get_feature_summary()

    return engineered_data, summary
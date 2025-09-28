"""
High-Performance Model Service
Advanced ML pipeline with optimized algorithms for churn prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class HighPerformanceModelService:
    """
    High-performance model service with advanced ML techniques
    """

    def __init__(self, random_state: int = 42):
        """Initialize the high-performance model service"""
        self.random_state = random_state
        self.n_folds = 5
        self.best_models = {}
        self.feature_selector = None
        np.random.seed(random_state)

    def create_advanced_models(self) -> Dict[str, Any]:
        """
        Create advanced ML models with optimized hyperparameters

        Returns:
            Dictionary of advanced models
        """
        models = {}

        # Random Forest with optimized parameters
        models['rf_optimized'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Extra Trees (more randomized than RF)
        models['extra_trees'] = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        # Gradient Boosting with optimized parameters
        models['gb_optimized'] = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=self.random_state
        )

        # XGBoost if available
        if XGBOOST_AVAILABLE:
            models['xgboost'] = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=-1
            )

        # LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = LGBMClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                verbose=-1,
                n_jobs=-1
            )

        # Logistic Regression with regularization
        models['logistic_l1'] = LogisticRegression(
            penalty='l1',
            C=0.1,
            solver='liblinear',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )

        models['logistic_l2'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000
        )

        return models

    def create_ensemble_models(self, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create ensemble models for improved performance

        Args:
            base_models: Dictionary of base models

        Returns:
            Dictionary of ensemble models
        """
        ensembles = {}

        # Select best performing base models for ensembles
        model_list = list(base_models.values())[:3]  # Top 3 models
        model_names = list(base_models.keys())[:3]

        if len(model_list) >= 3:
            # Voting Classifier (Hard voting)
            ensembles['voting_hard'] = VotingClassifier(
                estimators=[(name, model) for name, model in zip(model_names, model_list)],
                voting='hard'
            )

            # Voting Classifier (Soft voting)
            ensembles['voting_soft'] = VotingClassifier(
                estimators=[(name, model) for name, model in zip(model_names, model_list)],
                voting='soft'
            )

            # Stacking Classifier
            ensembles['stacking'] = StackingClassifier(
                estimators=[(name, model) for name, model in zip(model_names, model_list)],
                final_estimator=LogisticRegression(
                    class_weight='balanced',
                    random_state=self.random_state
                ),
                cv=3
            )

        return ensembles

    def advanced_feature_engineering(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Apply advanced feature engineering techniques

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Enhanced feature matrix
        """
        X_enhanced = X.copy()

        # Numerical features for polynomial features
        numerical_cols = X_enhanced.select_dtypes(include=[np.number]).columns

        # Create interaction features for top numerical features
        if len(numerical_cols) >= 2:
            top_numerical = numerical_cols[:4]  # Limit to prevent feature explosion

            for i, col1 in enumerate(top_numerical):
                for col2 in top_numerical[i+1:]:
                    # Interaction terms
                    X_enhanced[f'{col1}_x_{col2}'] = X_enhanced[col1] * X_enhanced[col2]

                    # Ratio features (avoid division by zero)
                    X_enhanced[f'{col1}_div_{col2}'] = X_enhanced[col1] / (X_enhanced[col2] + 1e-8)

        # Polynomial features for key numerical columns (degree 2)
        if 'tenure' in X_enhanced.columns:
            X_enhanced['tenure_squared'] = X_enhanced['tenure'] ** 2
            X_enhanced['tenure_cubed'] = X_enhanced['tenure'] ** 3

        if 'monthly_charges' in X_enhanced.columns:
            X_enhanced['monthly_charges_squared'] = X_enhanced['monthly_charges'] ** 2

        # Binning for numerical features
        if 'tenure' in X_enhanced.columns:
            X_enhanced['tenure_binned'] = pd.cut(
                X_enhanced['tenure'],
                bins=5,
                labels=False,
                duplicates='drop'
            )

        if 'monthly_charges' in X_enhanced.columns:
            X_enhanced['charges_binned'] = pd.cut(
                X_enhanced['monthly_charges'],
                bins=5,
                labels=False,
                duplicates='drop'
            )

        # Log transformation for skewed features
        for col in numerical_cols:
            if X_enhanced[col].min() > 0:  # Ensure positive values
                skewness = X_enhanced[col].skew()
                if abs(skewness) > 1:  # Apply log transform if highly skewed
                    X_enhanced[f'{col}_log'] = np.log1p(X_enhanced[col])

        return X_enhanced

    def feature_selection(self, X: pd.DataFrame, y: pd.Series, method: str = 'rfecv') -> pd.DataFrame:
        """
        Perform feature selection to reduce overfitting and improve performance

        Args:
            X: Feature matrix
            y: Target variable
            method: Feature selection method ('rfecv', 'selectk', 'both')

        Returns:
            Feature matrix with selected features
        """
        if method == 'selectk' or method == 'both':
            # Statistical feature selection
            selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]//2))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            if method == 'selectk':
                self.feature_selector = selector
                return X_result

        if method == 'rfecv' or method == 'both':
            # Recursive feature elimination with cross-validation
            base_estimator = RandomForestClassifier(
                n_estimators=50,
                random_state=self.random_state,
                class_weight='balanced'
            )

            selector = RFECV(
                estimator=base_estimator,
                step=1,
                cv=3,
                scoring='f1',
                min_features_to_select=5
            )

            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X_result = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

            self.feature_selector = selector
            return X_result

        return X

    def advanced_sampling(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote_tomek') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply advanced sampling techniques for class imbalance

        Args:
            X: Feature matrix
            y: Target variable
            method: Sampling method ('smote', 'adasyn', 'smote_tomek')

        Returns:
            Resampled feature matrix and target variable
        """
        # Check if resampling is needed
        class_counts = y.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()

        if minority_ratio >= 0.3:  # Relatively balanced
            return X, y

        # Apply selected resampling method
        if method == 'smote':
            sampler = SMOTE(
                sampling_strategy='auto',
                random_state=self.random_state,
                k_neighbors=5
            )
        elif method == 'adasyn':
            sampler = ADASYN(
                sampling_strategy='auto',
                random_state=self.random_state,
                n_neighbors=5
            )
        elif method == 'smote_tomek':
            sampler = SMOTETomek(
                sampling_strategy='auto',
                random_state=self.random_state,
                smote=SMOTE(random_state=self.random_state, k_neighbors=5),
                tomek=TomekLinks(sampling_strategy='majority')
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except Exception as e:
            print(f"Warning: Resampling failed with {method}, using original data: {e}")
            return X, y

    def train_and_evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train and evaluate all models with comprehensive metrics

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary with model performance results
        """
        # Apply advanced feature engineering
        print("Applying advanced feature engineering...")
        X_enhanced = self.advanced_feature_engineering(X, y)

        # Feature selection
        print("Performing feature selection...")
        X_selected = self.feature_selection(X_enhanced, y, method='rfecv')

        # Advanced sampling
        print("Applying advanced sampling...")
        X_resampled, y_resampled = self.advanced_sampling(X_selected, y, method='smote_tomek')

        # Create models
        print("Creating advanced models...")
        base_models = self.create_advanced_models()

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        results = {}

        print("Evaluating base models...")
        # Evaluate base models
        for name, model in base_models.items():
            try:
                scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1', n_jobs=-1)
                results[name] = {
                    'f1_mean': scores.mean(),
                    'f1_std': scores.std(),
                    'f1_scores': scores.tolist(),
                    'model_type': 'base'
                }
                print(f"{name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")
                continue

        # Train models on full resampled data for ensemble
        print("Training models for ensemble...")
        trained_models = {}
        for name, model in base_models.items():
            if name in results:
                try:
                    model.fit(X_resampled, y_resampled)
                    trained_models[name] = model
                except Exception as e:
                    print(f"Error training {name} for ensemble: {e}")

        # Create and evaluate ensemble models
        if len(trained_models) >= 3:
            print("Creating and evaluating ensemble models...")
            ensemble_models = self.create_ensemble_models(trained_models)

            for name, model in ensemble_models.items():
                try:
                    scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1', n_jobs=-1)
                    results[name] = {
                        'f1_mean': scores.mean(),
                        'f1_std': scores.std(),
                        'f1_scores': scores.tolist(),
                        'model_type': 'ensemble'
                    }
                    print(f"{name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")
                except Exception as e:
                    print(f"Error evaluating ensemble {name}: {e}")

        # Find best model
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_mean'])
        best_score = results[best_model_name]['f1_mean']

        print(f"\nBest model: {best_model_name} with F1-score: {best_score:.3f}")

        # Store best models
        if best_model_name in base_models:
            self.best_models['best'] = base_models[best_model_name]
        elif best_model_name in ensemble_models:
            self.best_models['best'] = ensemble_models[best_model_name]

        # Add metadata
        results['metadata'] = {
            'best_model': best_model_name,
            'best_f1_score': best_score,
            'features_selected': X_selected.shape[1],
            'original_features': X.shape[1],
            'resampled_size': len(X_resampled),
            'original_size': len(X),
            'feature_selection_method': 'rfecv',
            'sampling_method': 'smote_tomek',
            'feature_engineering_applied': True
        }

        return results

    def get_feature_importance(self, model_name: str = 'best') -> Dict[str, float]:
        """
        Get feature importance from the best model

        Args:
            model_name: Name of model to get importance from

        Returns:
            Dictionary of feature importances
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found")

        model = self.best_models[model_name]

        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            if self.feature_selector is not None:
                selected_features = self.feature_selector.get_feature_names_out()
            else:
                selected_features = [f"feature_{i}" for i in range(len(model.feature_importances_))]

            importance_dict = dict(zip(selected_features, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        elif hasattr(model, 'coef_'):
            # For linear models
            if self.feature_selector is not None:
                selected_features = self.feature_selector.get_feature_names_out()
            else:
                selected_features = [f"feature_{i}" for i in range(len(model.coef_[0]))]

            importance_dict = dict(zip(selected_features, np.abs(model.coef_[0])))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

        else:
            print("Model does not support feature importance extraction")
            return {}

    def predict_with_confidence(self, X: pd.DataFrame, model_name: str = 'best') -> Dict[str, Any]:
        """
        Make predictions with confidence intervals

        Args:
            X: Feature matrix for prediction
            model_name: Name of model to use

        Returns:
            Dictionary with predictions and confidence metrics
        """
        if model_name not in self.best_models:
            raise ValueError(f"Model {model_name} not found")

        model = self.best_models[model_name]

        # Apply same transformations as training
        X_enhanced = self.advanced_feature_engineering(X, pd.Series([0]*len(X)))  # Dummy y for consistency

        if self.feature_selector is not None:
            X_selected = self.feature_selector.transform(X_enhanced)
            X_final = pd.DataFrame(X_selected, index=X.index)
        else:
            X_final = X_enhanced

        # Get predictions
        predictions = model.predict(X_final)

        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_final)[:, 1]

            # Calculate confidence levels
            confidence_levels = np.where(
                probabilities > 0.7, 'High',
                np.where(probabilities > 0.3, 'Medium', 'Low')
            )
        else:
            probabilities = None
            confidence_levels = ['Unknown'] * len(predictions)

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'confidence_levels': confidence_levels,
            'high_risk_count': np.sum(predictions == 1),
            'prediction_summary': {
                'total_customers': len(predictions),
                'predicted_churn': np.sum(predictions == 1),
                'predicted_retention': np.sum(predictions == 0),
                'churn_rate': np.mean(predictions)
            }
        }

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Simple method to train models and return trained model objects
        Compatibility method for existing tests

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Dictionary of trained models
        """
        # Apply basic feature engineering
        X_enhanced = self.advanced_feature_engineering(X, y)

        # Create basic models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            ),
            'logistic_regression': LogisticRegression(
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=1000
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=self.random_state
            )
        }

        # Train models
        trained_models = {}
        for name, model in models.items():
            try:
                model.fit(X_enhanced, y)
                trained_models[name] = model
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        return trained_models
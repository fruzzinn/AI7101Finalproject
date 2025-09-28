"""
Ultra-High Performance Model Service
Advanced ML pipeline targeting 0.9+ F1-score with state-of-the-art techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Advanced preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.manifold import TSNE

# Imbalanced learning
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, NeighbourhoodCleaningRule
from imblearn.combine import SMOTETomek, SMOTEENN

# Model validation
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

# Advanced ensemble
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

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class UltraHighPerformanceModel:
    """
    Ultra-high performance model targeting 0.9+ F1-score
    """

    def __init__(self, random_state: int = 42, target_f1: float = 0.9):
        """Initialize the ultra-high performance model service"""
        self.random_state = random_state
        self.target_f1 = target_f1
        self.n_folds = 10  # More folds for better validation
        self.best_models = {}
        self.preprocessing_pipeline = {}
        self.feature_importance_scores = {}

        np.random.seed(random_state)

    def advanced_data_quality_enhancement(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Advanced data quality enhancement and outlier handling
        """
        print("🔍 Advanced data quality enhancement...")

        X_enhanced = X.copy()

        # 1. Remove low variance features
        variance_selector = VarianceThreshold(threshold=0.01)
        X_enhanced = pd.DataFrame(
            variance_selector.fit_transform(X_enhanced),
            columns=X_enhanced.columns[variance_selector.get_support()],
            index=X_enhanced.index
        )

        # 2. Handle extreme outliers using IQR method
        numerical_cols = X_enhanced.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            Q1 = X_enhanced[col].quantile(0.25)
            Q3 = X_enhanced[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More aggressive outlier detection
            upper_bound = Q3 + 3 * IQR

            # Cap outliers
            X_enhanced[col] = X_enhanced[col].clip(lower_bound, upper_bound)

        # 3. Create polynomial interactions for top features
        if len(numerical_cols) >= 2:
            # Get correlation with target for feature selection
            correlations = []
            for col in numerical_cols:
                corr = abs(np.corrcoef(X_enhanced[col], y)[0, 1])
                if not np.isnan(corr):
                    correlations.append((col, corr))

            # Sort by correlation and take top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            top_features = [feat for feat, _ in correlations[:4]]

            # Create interactions
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    X_enhanced[f'{feat1}_x_{feat2}'] = X_enhanced[feat1] * X_enhanced[feat2]
                    X_enhanced[f'{feat1}_div_{feat2}'] = X_enhanced[feat1] / (X_enhanced[feat2] + 1e-8)

        # 4. Create domain-specific features
        self._create_domain_specific_features(X_enhanced)

        return X_enhanced, y

    def _create_domain_specific_features(self, df: pd.DataFrame) -> None:
        """Create domain-specific churn prediction features"""

        # Churn risk indicators
        if 'tenure' in df.columns:
            # Tenure risk bands
            df['tenure_risk_very_high'] = (df['tenure'] <= 3).astype(int)
            df['tenure_risk_high'] = ((df['tenure'] > 3) & (df['tenure'] <= 12)).astype(int)
            df['tenure_risk_medium'] = ((df['tenure'] > 12) & (df['tenure'] <= 24)).astype(int)
            df['tenure_loyalty'] = (df['tenure'] >= 36).astype(int)

            # Tenure momentum
            df['tenure_squared'] = df['tenure'] ** 2
            df['tenure_log'] = np.log1p(df['tenure'])
            df['tenure_sqrt'] = np.sqrt(df['tenure'])

        # Financial risk indicators
        if 'monthly_charges' in df.columns:
            df['monthly_charges_squared'] = df['monthly_charges'] ** 2
            df['monthly_charges_log'] = np.log1p(df['monthly_charges'])

            # Price sensitivity bands
            monthly_std = df['monthly_charges'].std()
            monthly_mean = df['monthly_charges'].mean()
            df['price_sensitivity_high'] = (df['monthly_charges'] > monthly_mean + 2*monthly_std).astype(int)
            df['price_sensitivity_low'] = (df['monthly_charges'] < monthly_mean - monthly_std).astype(int)

        # Service adoption patterns
        service_cols = [col for col in df.columns if any(service in col.lower()
                       for service in ['streaming', 'internet', 'phone', 'tv', 'support', 'backup'])]
        if service_cols:
            df['service_adoption_count'] = df[service_cols].sum(axis=1)
            df['service_adoption_rate'] = df['service_adoption_count'] / len(service_cols)
            df['low_adoption_risk'] = (df['service_adoption_count'] <= 1).astype(int)
            df['high_adoption_loyalty'] = (df['service_adoption_count'] >= len(service_cols) * 0.7).astype(int)

    def ultra_advanced_preprocessing(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Ultra-advanced preprocessing with multiple transformation techniques
        """
        print("⚙️ Ultra-advanced preprocessing...")

        X_processed = X.copy()

        # 1. Multiple scaling approaches
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns

        if len(numerical_cols) > 0:
            # Robust scaling for main features
            robust_scaler = RobustScaler()
            X_processed[numerical_cols] = robust_scaler.fit_transform(X_processed[numerical_cols])

            # Power transformation for normality
            power_transformer = PowerTransformer(method='yeo-johnson', standardize=True)
            X_power = power_transformer.fit_transform(X_processed[numerical_cols])

            # Add power-transformed features as additional columns
            for i, col in enumerate(numerical_cols):
                X_processed[f'{col}_power'] = X_power[:, i]

            # Quantile transformation for uniform distribution
            quantile_transformer = QuantileTransformer(output_distribution='uniform', random_state=self.random_state)
            X_quantile = quantile_transformer.fit_transform(X_processed[numerical_cols])

            # Add quantile-transformed features
            for i, col in enumerate(numerical_cols):
                X_processed[f'{col}_quantile'] = X_quantile[:, i]

        # 2. Advanced feature selection
        X_selected = self._ultra_feature_selection(X_processed, y)

        # 3. Dimensionality enhancement (not reduction!)
        X_enhanced = self._create_dimensionality_features(X_selected, y)

        return X_enhanced, y

    def _ultra_feature_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Ultra-advanced feature selection combining multiple methods"""

        # 1. Statistical selection
        selector_f = SelectKBest(score_func=f_classif, k=min(50, X.shape[1]//2))
        X_f = selector_f.fit_transform(X, y)
        selected_f = X.columns[selector_f.get_support()]

        # 2. Mutual information selection
        selector_mi = SelectKBest(score_func=mutual_info_classif, k=min(50, X.shape[1]//2))
        X_mi = selector_mi.fit_transform(X, y)
        selected_mi = X.columns[selector_mi.get_support()]

        # 3. Model-based selection
        rf_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            threshold='median'
        )
        rf_selector.fit(X, y)
        selected_rf = X.columns[rf_selector.get_support()]

        # Combine all selected features
        all_selected = set(selected_f) | set(selected_mi) | set(selected_rf)

        return X[list(all_selected)]

    def _create_dimensionality_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Create dimensionality-based features"""

        X_enhanced = X.copy()

        # PCA features (first few components often capture most variance)
        if X.shape[1] >= 5:
            pca = PCA(n_components=min(5, X.shape[1]), random_state=self.random_state)
            X_pca = pca.fit_transform(X)

            for i in range(X_pca.shape[1]):
                X_enhanced[f'pca_{i}'] = X_pca[:, i]

        # ICA features for independence
        if X.shape[1] >= 3:
            ica = FastICA(n_components=min(3, X.shape[1]), random_state=self.random_state, max_iter=1000)
            try:
                X_ica = ica.fit_transform(X)
                for i in range(X_ica.shape[1]):
                    X_enhanced[f'ica_{i}'] = X_ica[:, i]
            except:
                pass  # ICA might fail on some datasets

        return X_enhanced

    def create_ultra_models(self) -> Dict[str, Any]:
        """Create ultra-high performance models"""

        models = {}

        # 1. Optimized tree-based models
        models['rf_ultra'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )

        models['et_ultra'] = ExtraTreesClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True
        )

        models['gb_ultra'] = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=self.random_state
        )

        # 2. Advanced boosting (if available)
        if XGBOOST_AVAILABLE:
            models['xgb_ultra'] = XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )

        if LIGHTGBM_AVAILABLE:
            models['lgb_ultra'] = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                min_child_samples=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
                is_unbalance=True
            )

        if CATBOOST_AVAILABLE:
            models['catboost_ultra'] = CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                subsample=0.8,
                random_state=self.random_state,
                verbose=False,
                auto_class_weights='Balanced'
            )

        # 3. Linear models with different regularizations
        models['lr_l1'] = LogisticRegression(
            penalty='l1',
            C=0.1,
            solver='liblinear',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=2000
        )

        models['lr_l2'] = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=2000
        )

        models['lr_elastic'] = LogisticRegression(
            penalty='elasticnet',
            C=1.0,
            l1_ratio=0.5,
            solver='saga',
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=2000
        )

        # 4. Support Vector Machine
        models['svm_rbf'] = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.random_state
        )

        # 5. Other algorithms
        models['knn'] = KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='manhattan'
        )

        models['nb'] = GaussianNB()

        models['lda'] = LinearDiscriminantAnalysis()

        models['qda'] = QuadraticDiscriminantAnalysis()

        return models

    def create_ultra_ensembles(self, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Create ultra-advanced ensemble models"""

        ensembles = {}

        # Select top models for ensembles
        model_list = list(base_models.values())
        model_names = list(base_models.keys())

        if len(model_list) >= 3:
            # 1. Voting ensembles
            ensembles['voting_soft_all'] = VotingClassifier(
                estimators=[(name, model) for name, model in zip(model_names, model_list)],
                voting='soft'
            )

            # 2. Bagging ensemble
            best_model = model_list[0]  # Assume first is best
            ensembles['bagging_ultra'] = BaggingClassifier(
                estimator=best_model,
                n_estimators=20,
                max_samples=0.8,
                max_features=0.8,
                random_state=self.random_state,
                n_jobs=-1
            )

            # 3. Multi-level stacking
            if len(model_list) >= 5:
                # Level 1: Use diverse base models
                level1_models = [(name, model) for name, model in zip(model_names[:5], model_list[:5])]

                # Level 2: Meta-learner
                meta_learner = LogisticRegression(
                    class_weight='balanced',
                    random_state=self.random_state
                )

                ensembles['stacking_ultra'] = StackingClassifier(
                    estimators=level1_models,
                    final_estimator=meta_learner,
                    cv=5,
                    stack_method='predict_proba',
                    n_jobs=-1
                )

        return ensembles

    def ultra_advanced_sampling(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Create multiple resampled datasets with different techniques"""

        print("🔄 Creating multiple resampled datasets...")

        resampled_datasets = []

        # Original dataset
        resampled_datasets.append((X, y, 'original'))

        # Check if resampling is needed
        class_counts = y.value_counts()
        minority_ratio = class_counts.min() / class_counts.max()

        if minority_ratio < 0.4:  # Only if significantly imbalanced

            # 1. SMOTE variants
            samplers = [
                (SMOTE(random_state=self.random_state, k_neighbors=5), 'smote'),
                (BorderlineSMOTE(random_state=self.random_state, k_neighbors=5), 'borderline_smote'),
                (SVMSMOTE(random_state=self.random_state, k_neighbors=5), 'svm_smote'),
                (ADASYN(random_state=self.random_state, n_neighbors=5), 'adasyn'),
            ]

            # 2. Combined methods
            samplers.extend([
                (SMOTETomek(random_state=self.random_state), 'smote_tomek'),
                (SMOTEENN(random_state=self.random_state), 'smote_enn'),
            ])

            for sampler, name in samplers:
                try:
                    X_resampled, y_resampled = sampler.fit_resample(X, y)
                    resampled_datasets.append((
                        pd.DataFrame(X_resampled, columns=X.columns),
                        pd.Series(y_resampled),
                        name
                    ))
                except Exception as e:
                    print(f"Warning: {name} failed: {e}")

        return resampled_datasets

    def train_and_evaluate_ultra_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train and evaluate ultra-high performance models"""

        print("🚀 Training ultra-high performance models...")

        # 1. Data quality enhancement
        X_enhanced, y_enhanced = self.advanced_data_quality_enhancement(X, y)

        # 2. Ultra-advanced preprocessing
        X_processed, y_processed = self.ultra_advanced_preprocessing(X_enhanced, y_enhanced)

        # 3. Create multiple resampled datasets
        resampled_datasets = self.ultra_advanced_sampling(X_processed, y_processed)

        # 4. Create models
        base_models = self.create_ultra_models()

        # 5. Cross-validation with more folds for stability
        cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        results = {}
        best_f1 = 0
        best_combination = None

        # 6. Evaluate each model on each resampled dataset
        for X_resamp, y_resamp, sampling_method in resampled_datasets:
            print(f"  Testing sampling method: {sampling_method}")

            for model_name, model in base_models.items():
                try:
                    scores = cross_val_score(
                        model, X_resamp, y_resamp,
                        cv=cv, scoring='f1', n_jobs=-1
                    )

                    mean_f1 = scores.mean()
                    std_f1 = scores.std()

                    combination_name = f"{model_name}_{sampling_method}"
                    results[combination_name] = {
                        'f1_mean': mean_f1,
                        'f1_std': std_f1,
                        'f1_scores': scores.tolist(),
                        'model_type': 'base',
                        'sampling_method': sampling_method
                    }

                    print(f"    {combination_name}: F1 = {mean_f1:.3f} ± {std_f1:.3f}")

                    if mean_f1 > best_f1:
                        best_f1 = mean_f1
                        best_combination = (model_name, model, X_resamp, y_resamp, sampling_method)

                except Exception as e:
                    print(f"    Error with {model_name} on {sampling_method}: {e}")

        # 7. Train ensemble models on best dataset
        if best_combination and best_f1 >= 0.7:  # Only if base models are reasonable
            print(f"  Creating ensembles with best combination (F1={best_f1:.3f})...")

            _, _, X_best, y_best, best_sampling = best_combination

            # Train base models on best dataset
            trained_models = {}
            for name, model in base_models.items():
                try:
                    model.fit(X_best, y_best)
                    trained_models[name] = model
                except:
                    continue

            # Create ensembles
            ensemble_models = self.create_ultra_ensembles(trained_models)

            # Evaluate ensembles
            for ensemble_name, ensemble_model in ensemble_models.items():
                try:
                    scores = cross_val_score(
                        ensemble_model, X_best, y_best,
                        cv=cv, scoring='f1', n_jobs=-1
                    )

                    mean_f1 = scores.mean()
                    std_f1 = scores.std()

                    combination_name = f"{ensemble_name}_{best_sampling}"
                    results[combination_name] = {
                        'f1_mean': mean_f1,
                        'f1_std': std_f1,
                        'f1_scores': scores.tolist(),
                        'model_type': 'ensemble',
                        'sampling_method': best_sampling
                    }

                    print(f"    {combination_name}: F1 = {mean_f1:.3f} ± {std_f1:.3f}")

                    if mean_f1 > best_f1:
                        best_f1 = mean_f1

                except Exception as e:
                    print(f"    Error with ensemble {ensemble_name}: {e}")

        # 8. Hyperparameter optimization for top models
        top_models = sorted(results.items(), key=lambda x: x[1]['f1_mean'], reverse=True)[:3]

        if top_models and top_models[0][1]['f1_mean'] < self.target_f1:
            print(f"  🔧 Hyperparameter optimization for top models...")
            self._hyperparameter_optimization(top_models, X_processed, y_processed, results)

        # 9. Final results
        final_best = max(results.keys(), key=lambda k: results[k]['f1_mean'])
        final_best_score = results[final_best]['f1_mean']

        results['metadata'] = {
            'best_model': final_best,
            'best_f1_score': final_best_score,
            'target_achieved': final_best_score >= self.target_f1,
            'total_combinations_tested': len(results) - 1,
            'features_final': X_processed.shape[1],
            'original_features': X.shape[1],
            'cv_folds': self.n_folds
        }

        return results

    def _hyperparameter_optimization(self, top_models: List, X: pd.DataFrame, y: pd.Series, results: Dict):
        """Advanced hyperparameter optimization for top models"""

        param_grids = {
            'rf_ultra': {
                'n_estimators': [300, 500, 700],
                'max_depth': [15, 20, 25],
                'min_samples_split': [2, 3, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2', 0.8]
            },
            'xgb_ultra': {
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.03, 0.05, 0.07],
                'max_depth': [6, 8, 10],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9]
            }
        }

        for model_full_name, model_results in top_models:
            model_base_name = model_full_name.split('_')[0] + '_' + model_full_name.split('_')[1]

            if model_base_name in param_grids:
                try:
                    # Create base model
                    if 'rf_ultra' in model_base_name:
                        base_model = RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
                    elif 'xgb_ultra' in model_base_name and XGBOOST_AVAILABLE:
                        base_model = XGBClassifier(random_state=self.random_state, n_jobs=-1)
                    else:
                        continue

                    # Grid search
                    grid_search = RandomizedSearchCV(
                        estimator=base_model,
                        param_distributions=param_grids[model_base_name],
                        n_iter=20,
                        cv=5,
                        scoring='f1',
                        n_jobs=-1,
                        random_state=self.random_state
                    )

                    grid_search.fit(X, y)

                    # Update results with optimized model
                    optimized_name = f"{model_full_name}_optimized"
                    results[optimized_name] = {
                        'f1_mean': grid_search.best_score_,
                        'f1_std': 0,  # Not available from RandomizedSearchCV
                        'best_params': grid_search.best_params_,
                        'model_type': 'optimized',
                        'sampling_method': model_results['sampling_method']
                    }

                    print(f"      {optimized_name}: F1 = {grid_search.best_score_:.3f}")

                except Exception as e:
                    print(f"      Hyperparameter optimization failed for {model_base_name}: {e}")
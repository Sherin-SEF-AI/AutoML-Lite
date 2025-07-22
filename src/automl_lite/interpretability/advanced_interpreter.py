"""
Advanced Model Interpretability for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.inspection import permutation_importance
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    from lime.lime_tabular import LimeTabularExplainer
    from lime.lime_base import LimeBase
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available. Install with: pip install lime")

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AdvancedInterpreter:
    """
    Advanced model interpretability with multiple methods.
    """
    
    def __init__(
        self,
        enable_shap: bool = True,
        enable_lime: bool = True,
        enable_permutation: bool = True,
        enable_partial_dependence: bool = True,
        enable_feature_effects: bool = True,
        n_shap_samples: int = 100,
        n_lime_samples: int = 1000,
        n_permutation_repeats: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Advanced Interpreter.
        
        Args:
            enable_shap: Enable SHAP explanations
            enable_lime: Enable LIME explanations
            enable_permutation: Enable permutation importance
            enable_partial_dependence: Enable partial dependence plots
            enable_feature_effects: Enable feature effects analysis
            n_shap_samples: Number of samples for SHAP
            n_lime_samples: Number of samples for LIME
            n_permutation_repeats: Number of repeats for permutation importance
            random_state: Random state for reproducibility
        """
        self.enable_shap = enable_shap and SHAP_AVAILABLE
        self.enable_lime = enable_lime and LIME_AVAILABLE
        self.enable_permutation = enable_permutation
        self.enable_partial_dependence = enable_partial_dependence
        self.enable_feature_effects = enable_feature_effects
        self.n_shap_samples = n_shap_samples
        self.n_lime_samples = n_lime_samples
        self.n_permutation_repeats = n_permutation_repeats
        self.random_state = random_state
        
        # Results storage
        self.shap_values_ = None
        self.shap_explainer_ = None
        self.lime_explainer_ = None
        self.permutation_importance_ = None
        self.partial_dependence_ = {}
        self.feature_effects_ = {}
        self.interpretability_summary_ = {}
        
    def fit(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        feature_names: Optional[List[str]] = None
    ) -> 'AdvancedInterpreter':
        """
        Fit the interpreter with model and data.
        
        Args:
            model: Trained model
            X: Training features
            y: Target variable
            feature_names: Feature names
            
        Returns:
            Self
        """
        logger.info("Starting advanced model interpretability analysis...")
        
        self.model = model
        self.X = X.copy()
        self.y = y
        self.feature_names = feature_names or X.columns.tolist()
        
        # 1. SHAP Analysis
        if self.enable_shap:
            self._compute_shap_values()
        
        # 2. LIME Analysis
        if self.enable_lime:
            self._setup_lime_explainer()
        
        # 3. Permutation Importance
        if self.enable_permutation and y is not None:
            self._compute_permutation_importance()
        
        # 4. Partial Dependence
        if self.enable_partial_dependence:
            self._compute_partial_dependence()
        
        # 5. Feature Effects
        if self.enable_feature_effects:
            self._compute_feature_effects()
        
        # Generate summary
        self._generate_interpretability_summary()
        
        logger.info("Advanced interpretability analysis completed")
        return self
    
    def _compute_shap_values(self):
        """Compute SHAP values."""
        try:
            logger.info("Computing SHAP values...")
            
            # Sample data if too large
            if len(self.X) > self.n_shap_samples:
                sample_indices = np.random.choice(
                    len(self.X), 
                    self.n_shap_samples, 
                    replace=False
                )
                X_sample = self.X.iloc[sample_indices]
            else:
                X_sample = self.X
            
            # Create SHAP explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                # Classification model
                self.shap_explainer_ = shap.TreeExplainer(self.model)
                self.shap_values_ = self.shap_explainer_.shap_values(X_sample)
                
                # For binary classification, use the positive class
                if isinstance(self.shap_values_, list):
                    self.shap_values_ = self.shap_values_[1]
            else:
                # Regression model
                self.shap_explainer_ = shap.TreeExplainer(self.model)
                self.shap_values_ = self.shap_explainer_.shap_values(X_sample)
            
            logger.info("SHAP values computed successfully")
            
        except Exception as e:
            logger.warning(f"Failed to compute SHAP values: {str(e)}")
            self.shap_values_ = None
            self.shap_explainer_ = None
    
    def _setup_lime_explainer(self):
        """Setup LIME explainer."""
        try:
            logger.info("Setting up LIME explainer...")
            
            # Convert to numpy arrays
            X_np = self.X.values
            
            # Determine if classification or regression
            if hasattr(self.model, 'predict_proba'):
                mode = 'classification'
            else:
                mode = 'regression'
            
            # Create LIME explainer
            self.lime_explainer_ = LimeTabularExplainer(
                X_np,
                feature_names=self.feature_names,
                class_names=['class_0', 'class_1'] if mode == 'classification' else None,
                mode=mode,
                random_state=self.random_state
            )
            
            logger.info("LIME explainer setup completed")
            
        except Exception as e:
            logger.warning(f"Failed to setup LIME explainer: {str(e)}")
            self.lime_explainer_ = None
    
    def _compute_permutation_importance(self):
        """Compute permutation importance."""
        try:
            logger.info("Computing permutation importance...")
            
            # Determine scoring function
            if hasattr(self.model, 'predict_proba'):
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            # Compute permutation importance
            result = permutation_importance(
                self.model,
                self.X,
                self.y,
                n_repeats=self.n_permutation_repeats,
                random_state=self.random_state,
                scoring=scoring
            )
            
            self.permutation_importance_ = {
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'feature_names': self.feature_names
            }
            
            logger.info("Permutation importance computed successfully")
            
        except Exception as e:
            logger.warning(f"Failed to compute permutation importance: {str(e)}")
            self.permutation_importance_ = None
    
    def _compute_partial_dependence(self):
        """Compute partial dependence plots."""
        try:
            logger.info("Computing partial dependence...")
            
            # Select top features for partial dependence
            if self.permutation_importance_:
                # Use permutation importance to select top features
                importances = self.permutation_importance_['importances_mean']
                top_indices = np.argsort(importances)[-5:]  # Top 5 features
            else:
                # Use first 5 features
                top_indices = range(min(5, len(self.feature_names)))
            
            for idx in top_indices:
                feature_name = self.feature_names[idx]
                
                try:
                    # Compute partial dependence
                    pd_result = partial_dependence(
                        self.model,
                        self.X,
                        [idx],
                        percentiles=(0.05, 0.95),
                        grid_resolution=50
                    )
                    
                    self.partial_dependence_[feature_name] = {
                        'values': pd_result[1][0],
                        'averaged_predictions': pd_result[0][0]
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to compute partial dependence for {feature_name}: {str(e)}")
                    continue
            
            logger.info(f"Partial dependence computed for {len(self.partial_dependence_)} features")
            
        except Exception as e:
            logger.warning(f"Failed to compute partial dependence: {str(e)}")
    
    def _compute_feature_effects(self):
        """Compute feature effects analysis."""
        try:
            logger.info("Computing feature effects...")
            
            # Analyze feature distributions and their relationship with target
            for feature_name in self.feature_names[:10]:  # Limit to 10 features
                try:
                    feature_values = self.X[feature_name]
                    
                    # Basic statistics
                    stats = {
                        'mean': feature_values.mean(),
                        'std': feature_values.std(),
                        'min': feature_values.min(),
                        'max': feature_values.max(),
                        'skewness': feature_values.skew(),
                        'kurtosis': feature_values.kurtosis()
                    }
                    
                    # Correlation with target if available
                    if self.y is not None:
                        correlation = feature_values.corr(self.y)
                        stats['correlation_with_target'] = correlation
                    
                    # Effect on predictions
                    if self.shap_values_ is not None:
                        feature_idx = self.feature_names.index(feature_name)
                        shap_importance = np.abs(self.shap_values_[:, feature_idx]).mean()
                        stats['shap_importance'] = shap_importance
                    
                    self.feature_effects_[feature_name] = stats
                    
                except Exception as e:
                    logger.warning(f"Failed to compute effects for {feature_name}: {str(e)}")
                    continue
            
            logger.info(f"Feature effects computed for {len(self.feature_effects_)} features")
            
        except Exception as e:
            logger.warning(f"Failed to compute feature effects: {str(e)}")
    
    def _generate_interpretability_summary(self):
        """Generate interpretability summary."""
        self.interpretability_summary_ = {
            'shap_available': self.shap_values_ is not None,
            'lime_available': self.lime_explainer_ is not None,
            'permutation_available': self.permutation_importance_ is not None,
            'partial_dependence_available': len(self.partial_dependence_) > 0,
            'feature_effects_available': len(self.feature_effects_) > 0,
            'n_features_analyzed': len(self.feature_names),
            'top_features': self._get_top_features()
        }
    
    def _get_top_features(self) -> List[str]:
        """Get top features by importance."""
        try:
            if self.permutation_importance_:
                importances = self.permutation_importance_['importances_mean']
                top_indices = np.argsort(importances)[-10:]  # Top 10
                return [self.feature_names[i] for i in top_indices[::-1]]
            elif self.shap_values_ is not None:
                shap_importances = np.abs(self.shap_values_).mean(axis=0)
                top_indices = np.argsort(shap_importances)[-10:]  # Top 10
                return [self.feature_names[i] for i in top_indices[::-1]]
            else:
                return self.feature_names[:10]
        except:
            return self.feature_names[:10]
    
    def explain_prediction(self, instance: pd.Series, method: str = 'shap') -> Dict[str, Any]:
        """
        Explain a single prediction.
        
        Args:
            instance: Single instance to explain
            method: Explanation method ('shap' or 'lime')
            
        Returns:
            Explanation dictionary
        """
        if method == 'shap' and self.shap_explainer_ is not None:
            return self._explain_with_shap(instance)
        elif method == 'lime' and self.lime_explainer_ is not None:
            return self._explain_with_lime(instance)
        else:
            raise ValueError(f"Method {method} not available")
    
    def _explain_with_shap(self, instance: pd.Series) -> Dict[str, Any]:
        """Explain prediction using SHAP."""
        try:
            # Get SHAP values for the instance
            shap_values = self.shap_explainer_.shap_values(instance)
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = {
                    'shap_value': float(shap_values[i]),
                    'feature_value': float(instance[feature])
                }
            
            return {
                'method': 'shap',
                'feature_importance': feature_importance,
                'base_value': float(self.shap_explainer_.expected_value),
                'prediction': float(self.model.predict(instance.values.reshape(1, -1))[0])
            }
            
        except Exception as e:
            logger.warning(f"Failed to explain with SHAP: {str(e)}")
            return {'method': 'shap', 'error': str(e)}
    
    def _explain_with_lime(self, instance: pd.Series) -> Dict[str, Any]:
        """Explain prediction using LIME."""
        try:
            # Get LIME explanation
            explanation = self.lime_explainer_.explain_instance(
                instance.values,
                self.model.predict,
                num_features=len(self.feature_names)
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature, weight in explanation.as_list():
                feature_importance[feature] = {
                    'lime_weight': weight,
                    'feature_value': float(instance[feature])
                }
            
            return {
                'method': 'lime',
                'feature_importance': feature_importance,
                'prediction': float(self.model.predict(instance.values.reshape(1, -1))[0])
            }
            
        except Exception as e:
            logger.warning(f"Failed to explain with LIME: {str(e)}")
            return {'method': 'lime', 'error': str(e)}
    
    def get_interpretability_report(self) -> Dict[str, Any]:
        """Get comprehensive interpretability report."""
        return {
            'summary': self.interpretability_summary_,
            'shap_values': self.shap_values_,
            'permutation_importance': self.permutation_importance_,
            'partial_dependence': self.partial_dependence_,
            'feature_effects': self.feature_effects_,
            'top_features': self._get_top_features()
        }
    
    def plot_feature_importance(self, method: str = 'permutation', top_n: int = 10):
        """Plot feature importance."""
        try:
            if method == 'permutation' and self.permutation_importance_:
                importances = self.permutation_importance_['importances_mean']
                std = self.permutation_importance_['importances_std']
                feature_names = self.permutation_importance_['feature_names']
                
                # Get top features
                top_indices = np.argsort(importances)[-top_n:]
                
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_indices)), importances[top_indices])
                plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices])
                plt.xlabel('Permutation Importance')
                plt.title('Feature Importance (Permutation)')
                plt.tight_layout()
                plt.show()
                
            elif method == 'shap' and self.shap_values_ is not None:
                # SHAP summary plot
                if SHAP_AVAILABLE:
                    shap.summary_plot(self.shap_values_, self.X, show=False)
                    plt.title('SHAP Feature Importance')
                    plt.tight_layout()
                    plt.show()
                    
        except Exception as e:
            logger.warning(f"Failed to plot feature importance: {str(e)}")
    
    def plot_partial_dependence(self, feature_name: str):
        """Plot partial dependence for a specific feature."""
        try:
            if feature_name in self.partial_dependence_:
                pd_data = self.partial_dependence_[feature_name]
                
                plt.figure(figsize=(8, 6))
                plt.plot(pd_data['values'], pd_data['averaged_predictions'])
                plt.xlabel(feature_name)
                plt.ylabel('Partial Dependence')
                plt.title(f'Partial Dependence Plot: {feature_name}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            logger.warning(f"Failed to plot partial dependence: {str(e)}") 
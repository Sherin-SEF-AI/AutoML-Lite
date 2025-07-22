"""
Auto Feature Engineering for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.decomposition import PCA
import warnings

from ..utils.logger import get_logger

logger = get_logger(__name__)


class AutoFeatureEngineer:
    """
    Automated feature engineering with multiple strategies.
    """
    
    def __init__(
        self,
        enable_polynomial_features: bool = True,
        enable_interaction_features: bool = True,
        enable_temporal_features: bool = True,
        enable_statistical_features: bool = True,
        enable_domain_features: bool = True,
        max_polynomial_degree: int = 2,
        max_feature_combinations: int = 100,
        feature_selection_method: str = 'mutual_info',
        n_best_features: Optional[int] = None,
        correlation_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize Auto Feature Engineer.
        
        Args:
            enable_polynomial_features: Generate polynomial features
            enable_interaction_features: Generate interaction features
            enable_temporal_features: Generate temporal features
            enable_statistical_features: Generate statistical features
            enable_domain_features: Generate domain-specific features
            max_polynomial_degree: Maximum polynomial degree
            max_feature_combinations: Maximum feature combinations
            feature_selection_method: Method for feature selection
            n_best_features: Number of best features to select
            correlation_threshold: Threshold for removing correlated features
            random_state: Random state for reproducibility
        """
        self.enable_polynomial_features = enable_polynomial_features
        self.enable_interaction_features = enable_interaction_features
        self.enable_temporal_features = enable_temporal_features
        self.enable_statistical_features = enable_statistical_features
        self.enable_domain_features = enable_domain_features
        self.max_polynomial_degree = max_polynomial_degree
        self.max_feature_combinations = max_feature_combinations
        self.feature_selection_method = feature_selection_method
        self.n_best_features = n_best_features
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        
        # Internal state
        self.feature_names_ = []
        self.feature_importance_ = {}
        self.correlation_matrix_ = None
        self.selected_features_ = []
        self.is_fitted_ = False
        
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the feature engineer and transform the data.
        
        Args:
            X: Input features
            y: Target variable (optional)
            
        Returns:
            Transformed features
        """
        logger.info("Starting auto feature engineering...")
        
        # Store original features
        original_features = X.copy()
        engineered_features = []
        
        # 1. Polynomial features
        if self.enable_polynomial_features:
            poly_features = self._generate_polynomial_features(X)
            engineered_features.append(poly_features)
            logger.info(f"Generated {poly_features.shape[1]} polynomial features")
        
        # 2. Interaction features
        if self.enable_interaction_features:
            interaction_features = self._generate_interaction_features(X)
            engineered_features.append(interaction_features)
            logger.info(f"Generated {interaction_features.shape[1]} interaction features")
        
        # 3. Temporal features
        if self.enable_temporal_features:
            temporal_features = self._generate_temporal_features(X)
            engineered_features.append(temporal_features)
            logger.info(f"Generated {temporal_features.shape[1]} temporal features")
        
        # 4. Statistical features
        if self.enable_statistical_features:
            statistical_features = self._generate_statistical_features(X)
            engineered_features.append(statistical_features)
            logger.info(f"Generated {statistical_features.shape[1]} statistical features")
        
        # 5. Domain-specific features
        if self.enable_domain_features:
            domain_features = self._generate_domain_features(X)
            engineered_features.append(domain_features)
            logger.info(f"Generated {domain_features.shape[1]} domain features")
        
        # Combine all features
        if engineered_features:
            combined_features = pd.concat([original_features] + engineered_features, axis=1)
        else:
            combined_features = original_features
        
        # Handle NaN values in the final dataset
        combined_features = combined_features.fillna(combined_features.median())
        
        # Remove highly correlated features
        combined_features = self._remove_correlated_features(combined_features)
        
        # Feature selection
        if self.n_best_features and y is not None:
            combined_features = self._select_best_features(combined_features, y)
        
        # Store feature information
        self.feature_names_ = combined_features.columns.tolist()
        self.is_fitted_ = True
        
        logger.info(f"Feature engineering completed. Final features: {len(self.feature_names_)}")
        
        return combined_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted feature engineer.
        
        Args:
            X: Input features
            
        Returns:
            Transformed features
        """
        if not self.is_fitted_:
            raise ValueError("Feature engineer must be fitted before transform")
        
        # Apply the same transformations as in fit_transform
        original_features = X.copy()
        engineered_features = []
        
        if self.enable_polynomial_features:
            poly_features = self._generate_polynomial_features(X)
            engineered_features.append(poly_features)
        
        if self.enable_interaction_features:
            interaction_features = self._generate_interaction_features(X)
            engineered_features.append(interaction_features)
        
        if self.enable_temporal_features:
            temporal_features = self._generate_temporal_features(X)
            engineered_features.append(temporal_features)
        
        if self.enable_statistical_features:
            statistical_features = self._generate_statistical_features(X)
            engineered_features.append(statistical_features)
        
        if self.enable_domain_features:
            domain_features = self._generate_domain_features(X)
            engineered_features.append(domain_features)
        
        if engineered_features:
            combined_features = pd.concat([original_features] + engineered_features, axis=1)
        else:
            combined_features = original_features
        
        # Handle NaN values in the final dataset
        combined_features = combined_features.fillna(combined_features.median())
        
        # Remove correlated features
        combined_features = self._remove_correlated_features(combined_features)
        
        # Select only the features that were selected during fitting
        if self.selected_features_:
            available_features = [f for f in self.selected_features_ if f in combined_features.columns]
            combined_features = combined_features[available_features]
        
        return combined_features
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features."""
        try:
            # Select numeric features
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) < 2:
                return pd.DataFrame()
            
            # Limit features to avoid explosion
            if len(numeric_features) > 10:
                numeric_features = numeric_features[:10]
            
            X_numeric = X[numeric_features]
            
            # Generate polynomial features
            poly = PolynomialFeatures(
                degree=min(self.max_polynomial_degree, 2),
                include_bias=False,
                interaction_only=True
            )
            
            poly_features = poly.fit_transform(X_numeric)
            feature_names = poly.get_feature_names_out(numeric_features)
            
            # Create DataFrame
            poly_df = pd.DataFrame(
                poly_features,
                columns=feature_names,
                index=X.index
            )
            
            # Remove original features (keep only interactions)
            original_feature_names = [f for f in feature_names if 'x0' not in f and 'x1' not in f]
            poly_df = poly_df[original_feature_names]
            
            return poly_df
            
        except Exception as e:
            logger.warning(f"Failed to generate polynomial features: {str(e)}")
            return pd.DataFrame()
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features between numeric columns."""
        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) < 2:
                return pd.DataFrame()
            
            interactions = []
            interaction_names = []
            
            # Generate pairwise interactions
            for i, feat1 in enumerate(numeric_features):
                for j, feat2 in enumerate(numeric_features[i+1:], i+1):
                    if len(interactions) >= self.max_feature_combinations:
                        break
                    
                    # Multiplication
                    interactions.append(X[feat1] * X[feat2])
                    interaction_names.append(f"{feat1}_mul_{feat2}")
                    
                    # Division (with safety check)
                    if (X[feat2] != 0).all():
                        interactions.append(X[feat1] / (X[feat2] + 1e-8))
                        interaction_names.append(f"{feat1}_div_{feat2}")
                    
                    # Sum
                    interactions.append(X[feat1] + X[feat2])
                    interaction_names.append(f"{feat1}_add_{feat2}")
                    
                    # Difference
                    interactions.append(X[feat1] - X[feat2])
                    interaction_names.append(f"{feat1}_sub_{feat2}")
                
                if len(interactions) >= self.max_feature_combinations:
                    break
            
            if interactions:
                interaction_df = pd.DataFrame(
                    dict(zip(interaction_names, interactions)),
                    index=X.index
                )
                return interaction_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to generate interaction features: {str(e)}")
            return pd.DataFrame()
    
    def _generate_temporal_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate temporal features from datetime columns."""
        try:
            temporal_features = []
            temporal_names = []
            
            # Find datetime columns
            datetime_columns = X.select_dtypes(include=['datetime64']).columns
            
            for col in datetime_columns:
                # Extract temporal components
                temporal_features.extend([
                    X[col].dt.year,
                    X[col].dt.month,
                    X[col].dt.day,
                    X[col].dt.dayofweek,
                    X[col].dt.dayofyear,
                    X[col].dt.quarter,
                    X[col].dt.hour,
                    X[col].dt.minute,
                    X[col].dt.second
                ])
                
                temporal_names.extend([
                    f"{col}_year",
                    f"{col}_month",
                    f"{col}_day",
                    f"{col}_dayofweek",
                    f"{col}_dayofyear",
                    f"{col}_quarter",
                    f"{col}_hour",
                    f"{col}_minute",
                    f"{col}_second"
                ])
            
            if temporal_features:
                temporal_df = pd.DataFrame(
                    dict(zip(temporal_names, temporal_features)),
                    index=X.index
                )
                return temporal_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to generate temporal features: {str(e)}")
            return pd.DataFrame()
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features."""
        try:
            numeric_features = X.select_dtypes(include=[np.number]).columns
            if len(numeric_features) < 2:
                return pd.DataFrame()
            
            statistical_features = []
            statistical_names = []
            
            # Rolling statistics (if enough data)
            if len(X) > 10:
                for col in numeric_features[:5]:  # Limit to 5 features
                    # Rolling mean
                    rolling_mean = X[col].rolling(window=3, min_periods=1).mean()
                    statistical_features.append(rolling_mean)
                    statistical_names.append(f"{col}_rolling_mean_3")
                    
                    # Rolling std
                    rolling_std = X[col].rolling(window=3, min_periods=1).std()
                    statistical_features.append(rolling_std)
                    statistical_names.append(f"{col}_rolling_std_3")
            
            # Percentile features
            for col in numeric_features[:5]:
                for percentile in [25, 50, 75]:
                    percentile_val = X[col].quantile(percentile / 100)
                    statistical_features.append(X[col] - percentile_val)
                    statistical_names.append(f"{col}_percentile_{percentile}_diff")
            
            if statistical_features:
                statistical_df = pd.DataFrame(
                    dict(zip(statistical_names, statistical_features)),
                    index=X.index
                )
                return statistical_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to generate statistical features: {str(e)}")
            return pd.DataFrame()
    
    def _generate_domain_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features based on column names and patterns."""
        try:
            domain_features = []
            domain_names = []
            
            # Financial domain features
            financial_keywords = ['price', 'cost', 'amount', 'revenue', 'profit', 'loss']
            for col in X.columns:
                col_lower = col.lower()
                
                # Financial ratios
                if any(keyword in col_lower for keyword in financial_keywords):
                    for other_col in X.columns:
                        if other_col != col and any(keyword in other_col.lower() for keyword in financial_keywords):
                            if (X[other_col] != 0).all():
                                ratio = X[col] / (X[other_col] + 1e-8)
                                domain_features.append(ratio)
                                domain_names.append(f"{col}_ratio_{other_col}")
                
                # Categorical encoding for string columns
                if X[col].dtype == 'object':
                    # Frequency encoding
                    freq_encoding = X[col].value_counts(normalize=True)
                    domain_features.append(X[col].map(freq_encoding))
                    domain_names.append(f"{col}_freq_encoding")
            
            if domain_features:
                domain_df = pd.DataFrame(
                    dict(zip(domain_names, domain_features)),
                    index=X.index
                )
                return domain_df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"Failed to generate domain features: {str(e)}")
            return pd.DataFrame()
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        try:
            if len(X.columns) < 2:
                return X
            
            # Calculate correlation matrix
            corr_matrix = X.corr().abs()
            self.correlation_matrix_ = corr_matrix
            
            # Find highly correlated features
            upper_tri = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [column for column in upper_tri.columns 
                      if any(upper_tri[column] > self.correlation_threshold)]
            
            if to_drop:
                logger.info(f"Removing {len(to_drop)} highly correlated features")
                X = X.drop(columns=to_drop)
            
            return X
            
        except Exception as e:
            logger.warning(f"Failed to remove correlated features: {str(e)}")
            return X
    
    def _select_best_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select the best features using statistical tests."""
        try:
            if self.n_best_features is None or self.n_best_features >= len(X.columns):
                return X
            
            # Handle NaN values by imputing with median
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Determine if classification or regression
            if y.dtype == 'object' or len(y.unique()) < 10:
                # Classification
                selector = SelectKBest(score_func=f_classif, k=self.n_best_features)
            else:
                # Regression
                selector = SelectKBest(score_func=f_regression, k=self.n_best_features)
            
            # Fit and transform
            X_selected = selector.fit_transform(X_imputed, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            # Store feature importance
            self.feature_importance_ = dict(zip(X.columns, selector.scores_))
            self.selected_features_ = selected_features
            
            logger.info(f"Selected {len(selected_features)} best features")
            
            return X[selected_features]
            
        except Exception as e:
            logger.warning(f"Failed to select best features: {str(e)}")
            return X
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance_
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of feature engineering process."""
        return {
            'original_features': len(self.feature_names_) - len(self.selected_features_) if self.selected_features_ else len(self.feature_names_),
            'engineered_features': len(self.selected_features_) if self.selected_features_ else 0,
            'total_features': len(self.feature_names_),
            'feature_importance': self.feature_importance_,
            'correlation_matrix': self.correlation_matrix_,
            'selected_features': self.selected_features_
        } 
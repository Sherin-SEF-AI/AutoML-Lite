"""
Preprocessing pipeline for AutoML Lite.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from category_encoders import OneHotEncoder, TargetEncoder, OrdinalEncoder
from typing import Any, Dict, List, Optional, Tuple, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    """
    Comprehensive preprocessing pipeline for AutoML Lite.
    
    Handles:
    - Missing value imputation
    - Categorical encoding
    - Feature scaling
    - Outlier detection
    - Feature selection
    """
    
    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.is_fitted = False
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.preprocessors = {}
        self.feature_names_out = None
        
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None
    ) -> "PreprocessingPipeline":
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable
            problem_type: Type of ML problem
            config: Preprocessing configuration
            
        Returns:
            self: Fitted pipeline
        """
        config = config or {}
        
        # Detect feature types
        self._detect_feature_types(X)
        
        # Configure preprocessing steps
        self._configure_preprocessors(X, y, problem_type, config)
        
        # Fit preprocessors
        self._fit_preprocessors(X, y)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted preprocessors.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
        
        X_transformed = X.copy()
        
        # Apply missing value imputation
        if 'missing_imputer' in self.preprocessors:
            X_transformed = self._apply_missing_imputation(X_transformed)
        
        # Apply categorical encoding
        if 'categorical_encoder' in self.preprocessors:
            X_transformed = self._apply_categorical_encoding(X_transformed)
        
        # Apply feature scaling
        if 'scaler' in self.preprocessors:
            X_transformed = self._apply_scaling(X_transformed)
        
        # Apply outlier detection
        if 'outlier_detector' in self.preprocessors:
            X_transformed = self._apply_outlier_detection(X_transformed)
        
        # Update feature names
        self.feature_names_out = X_transformed.columns.tolist()
        
        return X_transformed
    
    def fit_transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            X: Feature matrix
            y: Target variable
            problem_type: Type of ML problem
            config: Preprocessing configuration
            
        Returns:
            Transformed DataFrame
        """
        return self.fit(X, y, problem_type, config).transform(X)
    
    def _detect_feature_types(self, X: pd.DataFrame) -> None:
        """Detect and categorize feature types."""
        self.numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_features = X.select_dtypes(include=['datetime']).columns.tolist()
        
        logger.info(f"Detected {len(self.numeric_features)} numeric features")
        logger.info(f"Detected {len(self.categorical_features)} categorical features")
        logger.info(f"Detected {len(self.datetime_features)} datetime features")
    
    def _configure_preprocessors(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        problem_type: str,
        config: Dict[str, Any]
    ) -> None:
        """Configure preprocessing steps based on data and configuration."""
        
        # Missing value handling
        missing_strategy = config.get('handle_missing', 'auto')
        if missing_strategy != 'none':
            self._configure_missing_imputation(X, missing_strategy)
        
        # Categorical encoding
        encoding_strategy = config.get('categorical_encoding', 'auto')
        if encoding_strategy != 'none' and self.categorical_features:
            self._configure_categorical_encoding(X, y, problem_type, encoding_strategy)
        
        # Feature scaling
        scaling_strategy = config.get('feature_scaling', 'auto')
        if scaling_strategy != 'none':
            self._configure_scaling(scaling_strategy)
        
        # Outlier detection
        if config.get('outlier_detection', False):
            self._configure_outlier_detection()
    
    def _configure_missing_imputation(self, X: pd.DataFrame, strategy: str) -> None:
        """Configure missing value imputation."""
        if strategy == 'auto':
            # Use different strategies for different data types
            numeric_imputer = SimpleImputer(strategy='median')
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            
            self.preprocessors['missing_imputer'] = ColumnTransformer(
                transformers=[
                    ('num', numeric_imputer, self.numeric_features),
                    ('cat', categorical_imputer, self.categorical_features)
                ],
                remainder='passthrough'
            )
        else:
            # Use specified strategy for all features
            self.preprocessors['missing_imputer'] = SimpleImputer(strategy=strategy)
    
    def _configure_categorical_encoding(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        problem_type: str,
        strategy: str
    ) -> None:
        """Configure categorical encoding."""
        if strategy == 'auto':
            # Choose encoding based on cardinality and problem type
            cardinality = {col: X[col].nunique() for col in self.categorical_features}
            
            if problem_type == 'classification' and y is not None:
                # Use target encoding for high cardinality features
                high_cardinality = [col for col, card in cardinality.items() if card > 10]
                low_cardinality = [col for col, card in cardinality.items() if card <= 10]
                
                if high_cardinality:
                    self.preprocessors['categorical_encoder'] = ColumnTransformer(
                        transformers=[
                            ('target_high', TargetEncoder(), high_cardinality),
                            ('onehot_low', OneHotEncoder(), low_cardinality)
                        ],
                        remainder='passthrough'
                    )
                else:
                    self.preprocessors['categorical_encoder'] = OneHotEncoder()
            else:
                # Use one-hot encoding for regression/clustering
                self.preprocessors['categorical_encoder'] = OneHotEncoder()
        elif strategy == 'onehot':
            self.preprocessors['categorical_encoder'] = OneHotEncoder()
        elif strategy == 'label':
            self.preprocessors['categorical_encoder'] = OrdinalEncoder()
        elif strategy == 'target':
            if y is not None:
                self.preprocessors['categorical_encoder'] = TargetEncoder()
            else:
                logger.warning("Target encoding requires target variable, using label encoding instead")
                self.preprocessors['categorical_encoder'] = OrdinalEncoder()
    
    def _configure_scaling(self, strategy: str) -> None:
        """Configure feature scaling."""
        if strategy == 'auto':
            # Use robust scaling by default (handles outliers better)
            self.preprocessors['scaler'] = RobustScaler()
        elif strategy == 'standard':
            self.preprocessors['scaler'] = StandardScaler()
        elif strategy == 'minmax':
            self.preprocessors['scaler'] = MinMaxScaler()
        elif strategy == 'robust':
            self.preprocessors['scaler'] = RobustScaler()
    
    def _configure_outlier_detection(self) -> None:
        """Configure outlier detection."""
        # Use isolation forest for outlier detection
        self.preprocessors['outlier_detector'] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
    
    def _fit_preprocessors(self, X: pd.DataFrame, y: Optional[pd.Series]) -> None:
        """Fit all configured preprocessors."""
        X_current = X.copy()
        
        for name, preprocessor in self.preprocessors.items():
            if hasattr(preprocessor, 'fit'):
                if name == 'categorical_encoder' and y is not None:
                    preprocessor.fit(X_current, y)
                    # Update X_current for next preprocessor
                    if hasattr(preprocessor, 'transform'):
                        X_current = preprocessor.transform(X_current)
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            feature_names = X_current.columns.tolist()
                        X_current = pd.DataFrame(X_current, columns=feature_names, index=X.index)
                elif name == 'scaler':
                    # Fit scaler on the current transformed data
                    preprocessor.fit(X_current)
                else:
                    preprocessor.fit(X_current)
                    # Update X_current for next preprocessor
                    if hasattr(preprocessor, 'transform'):
                        X_current = preprocessor.transform(X_current)
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names = preprocessor.get_feature_names_out()
                        else:
                            feature_names = X_current.columns.tolist()
                        X_current = pd.DataFrame(X_current, columns=feature_names, index=X.index)
    
    def _apply_missing_imputation(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply missing value imputation."""
        imputer = self.preprocessors['missing_imputer']
        X_imputed = imputer.transform(X)
        
        # Convert back to DataFrame with proper column names
        if hasattr(imputer, 'get_feature_names_out'):
            feature_names = imputer.get_feature_names_out()
        else:
            feature_names = X.columns.tolist()
        
        return pd.DataFrame(X_imputed, columns=pd.Index(feature_names), index=X.index)
    
    def _apply_categorical_encoding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply categorical encoding."""
        encoder = self.preprocessors['categorical_encoder']
        X_encoded = encoder.transform(X)
        
        # Convert back to DataFrame with proper column names
        if hasattr(encoder, 'get_feature_names_out'):
            feature_names = encoder.get_feature_names_out()
        else:
            feature_names = X.columns.tolist()
        
        return pd.DataFrame(X_encoded, columns=pd.Index(feature_names), index=X.index)
    
    def _apply_scaling(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature scaling."""
        scaler = self.preprocessors['scaler']
        X_scaled = scaler.transform(X)
        
        # Convert back to DataFrame with proper column names
        if hasattr(scaler, 'get_feature_names_out'):
            feature_names = scaler.get_feature_names_out()
        else:
            feature_names = X.columns.tolist()
        
        return pd.DataFrame(X_scaled, columns=pd.Index(feature_names), index=X.index)
    
    def _apply_outlier_detection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply outlier detection and handling."""
        detector = self.preprocessors['outlier_detector']
        outlier_labels = detector.predict(X)
        
        # Remove outliers (label -1 indicates outliers)
        outlier_mask = outlier_labels == -1
        if outlier_mask.any():
            logger.info(f"Detected {outlier_mask.sum()} outliers, removing them")
            X_clean = X[~outlier_mask].copy()
        else:
            X_clean = X.copy()
        
        return X_clean
    
    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        if self.feature_names_out is None:
            raise ValueError("Pipeline must be fitted before getting feature names")
        return self.feature_names_out 
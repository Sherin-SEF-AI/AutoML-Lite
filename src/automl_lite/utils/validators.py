"""
Data validation utilities for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import Union


class DataValidator:
    """
    Validate and convert input data to appropriate formats.
    """
    
    def validate_features(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Validate and convert feature matrix to DataFrame.
        
        Args:
            X: Feature matrix
            
        Returns:
            Validated DataFrame
        """
        if isinstance(X, np.ndarray):
            # Convert numpy array to DataFrame
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            feature_names = pd.Index([f"feature_{i}" for i in range(X.shape[1])])
            X = pd.DataFrame(X, columns=feature_names)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame or numpy array")
        
        # Check for empty data
        if X.empty:
            raise ValueError("Feature matrix is empty")
        
        # Check for infinite values
        if np.isinf(X.select_dtypes(include=[np.number])).any().any():
            raise ValueError("Feature matrix contains infinite values")
        
        return X
    
    def validate_target(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Validate and convert target variable to Series.
        
        Args:
            y: Target variable
            
        Returns:
            Validated Series
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        elif not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series or numpy array")
        
        # Check for empty data
        if y.empty:
            raise ValueError("Target variable is empty")
        
        # Check for infinite values (for numeric targets)
        if pd.api.types.is_numeric_dtype(y):
            if np.isinf(y).any():
                raise ValueError("Target variable contains infinite values")
        
        return y
    
    def validate_data_consistency(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> None:
        """
        Validate consistency between features and target.
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        X = self.validate_features(X)
        y = self.validate_target(y)
        
        if len(X) != len(y):
            raise ValueError(
                f"Number of samples in X ({len(X)}) and y ({len(y)}) must match"
            )
    
    def check_missing_values(self, X: pd.DataFrame) -> dict:
        """
        Check for missing values in the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_stats = {
            'total_missing': X.isnull().sum().sum(),
            'missing_percentage': (X.isnull().sum().sum() / (X.shape[0] * X.shape[1])) * 100,
            'columns_with_missing': X.columns[X.isnull().any()].tolist(),
            'missing_by_column': X.isnull().sum().to_dict()
        }
        
        return missing_stats
    
    def check_data_types(self, X: pd.DataFrame) -> dict:
        """
        Analyze data types in the dataset.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with data type information
        """
        type_info = {
            'numeric_columns': X.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': X.select_dtypes(include=['object', 'category']).columns.tolist(),
            'datetime_columns': X.select_dtypes(include=['datetime']).columns.tolist(),
            'dtype_counts': X.dtypes.value_counts().to_dict()
        }
        
        return type_info
    
    def check_cardinality(self, X: pd.DataFrame) -> dict:
        """
        Check cardinality of categorical features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary with cardinality information
        """
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        cardinality = {}
        
        for col in categorical_cols:
            cardinality[col] = X[col].nunique()
        
        return cardinality
    
    def validate_preprocessing_config(self, config: dict) -> dict:
        """
        Validate preprocessing configuration.
        
        Args:
            config: Preprocessing configuration dictionary
            
        Returns:
            Validated configuration
        """
        valid_config = {}
        
        # Handle missing values
        if 'handle_missing' in config:
            valid_options = ['auto', 'drop', 'impute', 'none']
            if config['handle_missing'] not in valid_options:
                raise ValueError(f"handle_missing must be one of {valid_options}")
            valid_config['handle_missing'] = config['handle_missing']
        
        # Categorical encoding
        if 'categorical_encoding' in config:
            valid_options = ['auto', 'onehot', 'label', 'target', 'none']
            if config['categorical_encoding'] not in valid_options:
                raise ValueError(f"categorical_encoding must be one of {valid_options}")
            valid_config['categorical_encoding'] = config['categorical_encoding']
        
        # Feature scaling
        if 'feature_scaling' in config:
            valid_options = ['auto', 'standard', 'minmax', 'robust', 'none']
            if config['feature_scaling'] not in valid_options:
                raise ValueError(f"feature_scaling must be one of {valid_options}")
            valid_config['feature_scaling'] = config['feature_scaling']
        
        # Outlier detection
        if 'outlier_detection' in config:
            if not isinstance(config['outlier_detection'], bool):
                raise ValueError("outlier_detection must be a boolean")
            valid_config['outlier_detection'] = config['outlier_detection']
        
        # Feature selection
        if 'feature_selection' in config:
            valid_options = ['auto', 'statistical', 'model_based', 'none']
            if config['feature_selection'] not in valid_options:
                raise ValueError(f"feature_selection must be one of {valid_options}")
            valid_config['feature_selection'] = config['feature_selection']
        
        return valid_config 
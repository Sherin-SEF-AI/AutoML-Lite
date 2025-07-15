"""
Problem type detection utilities for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


class ProblemDetector:
    """
    Automatically detect the type of machine learning problem.
    """
    
    def detect(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Optional[Union[pd.Series, np.ndarray]] = None
    ) -> str:
        """
        Detect the problem type based on input data.
        
        Args:
            X: Feature matrix
            y: Target variable (None for clustering)
            
        Returns:
            Problem type: 'classification', 'regression', or 'clustering'
        """
        if y is None:
            return "clustering"
        
        # Convert to pandas Series if needed
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Check if target is numeric
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        
        # Check for classification based on unique values
        unique_values = y.nunique()
        total_samples = len(y)
        
        # If very few unique values relative to sample size, likely classification
        if unique_values <= max(2, total_samples * 0.1):
            return "classification"
        
        # If we have a reasonable number of unique values but they're all integers
        # and they form a consecutive sequence starting from 0, likely classification
        if y.dtype in ['int64', 'int32', 'int16', 'int8']:
            if y.min() == 0 and y.max() == unique_values - 1:
                return "classification"
        
        # If we have many unique values (more than 20% of samples), likely regression
        if unique_values > total_samples * 0.2:
            return "regression"
        
        # Default to regression for continuous numeric data
        return "regression"
    
    def is_binary_classification(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """
        Check if the problem is binary classification.
        
        Args:
            y: Target variable
            
        Returns:
            True if binary classification
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        return y.nunique() == 2
    
    def is_multiclass_classification(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """
        Check if the problem is multiclass classification.
        
        Args:
            y: Target variable
            
        Returns:
            True if multiclass classification
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        unique_count = y.nunique()
        return 2 < unique_count <= 20  # Reasonable limit for multiclass 
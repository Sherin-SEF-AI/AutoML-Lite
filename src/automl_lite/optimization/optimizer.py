"""
Hyperparameter optimization using Optuna.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score
)

from ..utils.logger import get_logger

logger = get_logger(__name__)


class HyperparameterOptimizer:
    """
    Hyperparameter optimization using Optuna.
    """
    
    def __init__(self, n_trials: int = 50, timeout: Optional[int] = None):
        """
        Initialize the optimizer.
        
        Args:
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
        """
        self.n_trials = n_trials
        self.timeout = timeout
    
    def optimize(
        self,
        model_class: Type,
        param_grid: Dict[str, List[Any]],
        X: pd.DataFrame,
        y: Optional[pd.Series],
        problem_type: str,
        cv_folds: int = 5,
        n_jobs: int = -1,
        early_stopping: bool = False,
        patience: int = 10,
        random_state: int = 42,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Optimize hyperparameters for a model.
        
        Args:
            model_class: Model class to optimize
            param_grid: Parameter grid
            X: Feature matrix
            y: Target variable
            problem_type: Type of problem
            cv_folds: Number of CV folds
            random_state: Random seed
            
        Returns:
            Tuple of (best_params, best_score)
        """
        def objective(trial):
            # Sample parameters from the grid
            params = self._sample_params(trial, param_grid)
            
            try:
                # Create model with sampled parameters
                model = model_class(**params)
                
                # Perform cross-validation
                score = self._cross_validate(
                    model, X, y, problem_type, cv_folds, random_state
                )
                
                return score
            except Exception as e:
                logger.warning(f"Trial failed: {str(e)}")
                return -np.inf if self._is_higher_better(problem_type) else np.inf
        
        # Create study
        direction = "maximize" if self._is_higher_better(problem_type) else "minimize"
        study = optuna.create_study(direction=direction)
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=False
        )
        
        best_params = study.best_params
        best_score = study.best_value
        
        logger.info(f"Best score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_params, best_score
    
    def _sample_params(self, trial: optuna.Trial, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Sample parameters from the grid using Optuna."""
        params = {}
        
        for param_name, param_values in param_grid.items():
            if isinstance(param_values[0], bool):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values[0], int):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values[0], float):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values[0], str):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values[0], tuple):
                # Handle tuple parameters (e.g., hidden_layer_sizes)
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_values)
        
        return params
    
    def _cross_validate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        problem_type: str,
        cv_folds: int,
        random_state: int,
    ) -> float:
        """Perform cross-validation and return score."""
        if problem_type == "clustering":
            return self._cross_validate_clustering(model, X, cv_folds, random_state)
        else:
            if y is None:
                raise ValueError("Target variable y is required for supervised learning")
            return self._cross_validate_supervised(model, X, y, problem_type, cv_folds, random_state)
    
    def _cross_validate_supervised(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        cv_folds: int,
        random_state: int,
    ) -> float:
        """Cross-validation for supervised learning."""
        if problem_type == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = self._get_classification_scoring()
        else:  # regression
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = self._get_regression_scoring()
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return np.mean(scores)
    
    def _cross_validate_clustering(
        self,
        model: Any,
        X: pd.DataFrame,
        cv_folds: int,
        random_state: int,
    ) -> float:
        """Cross-validation for clustering."""
        # For clustering, we use silhouette score
        try:
            model.fit(X)
            if hasattr(model, 'labels_'):
                labels = model.labels_
            else:
                labels = model.predict(X)
            
            # Silhouette score requires at least 2 clusters
            if len(np.unique(labels)) < 2:
                return -1.0
            
            return silhouette_score(X, labels)
        except Exception:
            return -1.0
    
    def _get_classification_scoring(self) -> str:
        """Get scoring metric for classification."""
        return 'accuracy'  # Can be extended to support other metrics
    
    def _get_regression_scoring(self) -> str:
        """Get scoring metric for regression."""
        return 'r2'  # Can be extended to support other metrics
    
    def _is_higher_better(self, problem_type: str) -> bool:
        """Check if higher scores are better for the problem type."""
        if problem_type == "classification":
            return True  # accuracy, f1, etc.
        elif problem_type == "regression":
            return True  # r2 score
        else:  # clustering
            return True  # silhouette score
    
    def optimize_multiple_models(
        self,
        models: List[Tuple[str, Type, Dict[str, List[Any]]]],
        X: pd.DataFrame,
        y: Optional[pd.Series],
        problem_type: str,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Optimize multiple models and return results.
        
        Args:
            models: List of (model_name, model_class, param_grid) tuples
            X: Feature matrix
            y: Target variable
            problem_type: Type of problem
            cv_folds: Number of CV folds
            random_state: Random seed
            
        Returns:
            List of (model_name, best_params, best_score) tuples
        """
        results = []
        
        for model_name, model_class, param_grid in models:
            try:
                logger.info(f"Optimizing {model_name}...")
                best_params, best_score = self.optimize(
                    model_class, param_grid, X, y, problem_type, cv_folds, random_state
                )
                results.append((model_name, best_params, best_score))
            except Exception as e:
                logger.warning(f"Failed to optimize {model_name}: {str(e)}")
                continue
        
        # Sort by score
        if self._is_higher_better(problem_type):
            results.sort(key=lambda x: x[2], reverse=True)
        else:
            results.sort(key=lambda x: x[2])
        
        return results 
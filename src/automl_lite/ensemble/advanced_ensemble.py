"""
Advanced Ensemble Methods for AutoML-Lite
Includes Stacking, Blending, and Meta-Learning approaches
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import logging

logger = logging.getLogger(__name__)


class StackingEnsemble(BaseEstimator):
    """
    Stacking Ensemble using K-Fold Cross-Validation

    Stacking uses predictions from base models as features for a meta-learner.
    Uses out-of-fold predictions to avoid overfitting.
    """

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: Optional[BaseEstimator] = None,
        n_folds: int = 5,
        use_probas: bool = True,
        problem_type: str = 'classification',
        stratified: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Stacking Ensemble

        Parameters:
        -----------
        base_models : List[BaseEstimator]
            List of base models for level-0 predictions
        meta_model : BaseEstimator, optional
            Meta-learner for level-1 predictions. If None, uses LogisticRegression
            for classification or Ridge for regression
        n_folds : int
            Number of folds for cross-validation
        use_probas : bool
            Whether to use probabilities for classification (True) or predictions (False)
        problem_type : str
            'classification' or 'regression'
        stratified : bool
            Whether to use stratified K-Fold for classification
        random_state : int
            Random state for reproducibility
        verbose : bool
            Whether to print progress
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.use_probas = use_probas
        self.problem_type = problem_type
        self.stratified = stratified
        self.random_state = random_state
        self.verbose = verbose

        # Will be set during fit
        self.base_models_ = None
        self.meta_model_ = None
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble

        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns:
        --------
        self : StackingEnsemble
        """
        if self.verbose:
            logger.info(f"Training Stacking Ensemble with {len(self.base_models)} base models and {self.n_folds} folds")

        # Initialize meta model if not provided
        if self.meta_model is None:
            if self.problem_type == 'classification':
                self.meta_model_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
            else:
                self.meta_model_ = Ridge(random_state=self.random_state)
        else:
            self.meta_model_ = clone(self.meta_model)

        # Initialize cross-validation
        if self.problem_type == 'classification' and self.stratified:
            kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            self.n_classes_ = len(np.unique(y))
        else:
            kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Generate out-of-fold predictions for meta features
        n_samples = X.shape[0]

        if self.problem_type == 'classification' and self.use_probas:
            # Use probabilities for each class
            meta_features = np.zeros((n_samples, len(self.base_models) * self.n_classes_))
        else:
            # Use single prediction per model
            meta_features = np.zeros((n_samples, len(self.base_models)))

        # Train base models and generate meta features
        self.base_models_ = []

        for model_idx, base_model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Training base model {model_idx + 1}/{len(self.base_models)}: {base_model.__class__.__name__}")

            # Out-of-fold predictions for this model
            oof_predictions = np.zeros((n_samples, self.n_classes_)) if (self.problem_type == 'classification' and self.use_probas) else np.zeros(n_samples)

            # K-Fold training
            fold_models = []
            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X, y if self.stratified else None)):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]

                # Clone and train model on this fold
                fold_model = clone(base_model)
                fold_model.fit(X_train_fold, y_train_fold)
                fold_models.append(fold_model)

                # Generate out-of-fold predictions
                if self.problem_type == 'classification' and self.use_probas:
                    oof_predictions[val_idx] = fold_model.predict_proba(X_val_fold)
                else:
                    oof_predictions[val_idx] = fold_model.predict(X_val_fold)

            # Store fold models
            self.base_models_.append(fold_models)

            # Add to meta features
            if self.problem_type == 'classification' and self.use_probas:
                meta_features[:, model_idx * self.n_classes_:(model_idx + 1) * self.n_classes_] = oof_predictions
            else:
                meta_features[:, model_idx] = oof_predictions

        # Train meta model on out-of-fold predictions
        if self.verbose:
            logger.info(f"Training meta-model: {self.meta_model_.__class__.__name__}")

        self.meta_model_.fit(meta_features, y)

        if self.verbose:
            logger.info("Stacking Ensemble training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble

        Parameters:
        -----------
        X : np.ndarray
            Features to predict

        Returns:
        --------
        predictions : np.ndarray
            Predictions
        """
        meta_features = self._generate_meta_features(X)
        return self.meta_model_.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (classification only)

        Parameters:
        -----------
        X : np.ndarray
            Features to predict

        Returns:
        --------
        probabilities : np.ndarray
            Class probabilities
        """
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification")

        meta_features = self._generate_meta_features(X)
        return self.meta_model_.predict_proba(meta_features)

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta features from base model predictions"""
        n_samples = X.shape[0]

        if self.problem_type == 'classification' and self.use_probas:
            meta_features = np.zeros((n_samples, len(self.base_models_) * self.n_classes_))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models_)))

        # Get predictions from all base models (average across folds)
        for model_idx, fold_models in enumerate(self.base_models_):
            if self.problem_type == 'classification' and self.use_probas:
                # Average probabilities across folds
                fold_predictions = np.zeros((n_samples, self.n_classes_))
                for fold_model in fold_models:
                    fold_predictions += fold_model.predict_proba(X)
                fold_predictions /= len(fold_models)

                meta_features[:, model_idx * self.n_classes_:(model_idx + 1) * self.n_classes_] = fold_predictions
            else:
                # Average predictions across folds
                fold_predictions = np.zeros(n_samples)
                for fold_model in fold_models:
                    fold_predictions += fold_model.predict(X)
                fold_predictions /= len(fold_models)

                meta_features[:, model_idx] = fold_predictions

        return meta_features


class BlendingEnsemble(BaseEstimator):
    """
    Blending Ensemble using holdout validation set

    Simpler than stacking - uses a single holdout set instead of K-fold CV.
    Faster but potentially less robust than stacking.
    """

    def __init__(
        self,
        base_models: List[BaseEstimator],
        meta_model: Optional[BaseEstimator] = None,
        blend_ratio: float = 0.2,
        use_probas: bool = True,
        problem_type: str = 'classification',
        stratify: bool = True,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Blending Ensemble

        Parameters:
        -----------
        base_models : List[BaseEstimator]
            List of base models
        meta_model : BaseEstimator, optional
            Meta-learner. If None, uses LogisticRegression/Ridge
        blend_ratio : float
            Ratio of data to use for blending (holdout set)
        use_probas : bool
            Whether to use probabilities for classification
        problem_type : str
            'classification' or 'regression'
        stratify : bool
            Whether to stratify the holdout split
        random_state : int
            Random state for reproducibility
        verbose : bool
            Whether to print progress
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.blend_ratio = blend_ratio
        self.use_probas = use_probas
        self.problem_type = problem_type
        self.stratify = stratify
        self.random_state = random_state
        self.verbose = verbose

        self.base_models_ = None
        self.meta_model_ = None
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BlendingEnsemble':
        """
        Fit the blending ensemble

        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns:
        --------
        self : BlendingEnsemble
        """
        from sklearn.model_selection import train_test_split

        if self.verbose:
            logger.info(f"Training Blending Ensemble with {len(self.base_models)} base models")

        # Split data into train and blend sets
        if self.problem_type == 'classification' and self.stratify:
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y, test_size=self.blend_ratio, random_state=self.random_state, stratify=y
            )
            self.n_classes_ = len(np.unique(y))
        else:
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y, test_size=self.blend_ratio, random_state=self.random_state
            )

        # Initialize meta model
        if self.meta_model is None:
            if self.problem_type == 'classification':
                self.meta_model_ = LogisticRegression(max_iter=1000, random_state=self.random_state)
            else:
                self.meta_model_ = Ridge(random_state=self.random_state)
        else:
            self.meta_model_ = clone(self.meta_model)

        # Train base models on training set
        self.base_models_ = []

        n_blend = X_blend.shape[0]
        if self.problem_type == 'classification' and self.use_probas:
            blend_features = np.zeros((n_blend, len(self.base_models) * self.n_classes_))
        else:
            blend_features = np.zeros((n_blend, len(self.base_models)))

        for model_idx, base_model in enumerate(self.base_models):
            if self.verbose:
                logger.info(f"Training base model {model_idx + 1}/{len(self.base_models)}: {base_model.__class__.__name__}")

            # Clone and train on training set
            trained_model = clone(base_model)
            trained_model.fit(X_train, y_train)
            self.base_models_.append(trained_model)

            # Generate predictions on blend set
            if self.problem_type == 'classification' and self.use_probas:
                blend_predictions = trained_model.predict_proba(X_blend)
                blend_features[:, model_idx * self.n_classes_:(model_idx + 1) * self.n_classes_] = blend_predictions
            else:
                blend_predictions = trained_model.predict(X_blend)
                blend_features[:, model_idx] = blend_predictions

        # Train meta model on blend set
        if self.verbose:
            logger.info(f"Training meta-model: {self.meta_model_.__class__.__name__}")

        self.meta_model_.fit(blend_features, y_blend)

        if self.verbose:
            logger.info("Blending Ensemble training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        meta_features = self._generate_meta_features(X)
        return self.meta_model_.predict(meta_features)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities (classification only)"""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification")

        meta_features = self._generate_meta_features(X)
        return self.meta_model_.predict_proba(meta_features)

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta features from base model predictions"""
        n_samples = X.shape[0]

        if self.problem_type == 'classification' and self.use_probas:
            meta_features = np.zeros((n_samples, len(self.base_models_) * self.n_classes_))
        else:
            meta_features = np.zeros((n_samples, len(self.base_models_)))

        for model_idx, model in enumerate(self.base_models_):
            if self.problem_type == 'classification' and self.use_probas:
                predictions = model.predict_proba(X)
                meta_features[:, model_idx * self.n_classes_:(model_idx + 1) * self.n_classes_] = predictions
            else:
                predictions = model.predict(X)
                meta_features[:, model_idx] = predictions

        return meta_features


class WeightedEnsemble(BaseEstimator):
    """
    Weighted Ensemble with optimized weights

    Learns optimal weights for combining model predictions using
    gradient descent or closed-form solution.
    """

    def __init__(
        self,
        models: List[BaseEstimator],
        problem_type: str = 'classification',
        optimization: str = 'grid',  # 'grid', 'gradient', or 'closed_form'
        use_probas: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Weighted Ensemble

        Parameters:
        -----------
        models : List[BaseEstimator]
            List of trained models
        problem_type : str
            'classification' or 'regression'
        optimization : str
            Method to optimize weights: 'grid', 'gradient', or 'closed_form'
        use_probas : bool
            Whether to use probabilities for classification
        verbose : bool
            Whether to print progress
        """
        self.models = models
        self.problem_type = problem_type
        self.optimization = optimization
        self.use_probas = use_probas
        self.verbose = verbose

        self.weights_ = None
        self.n_classes_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedEnsemble':
        """
        Fit the weighted ensemble by optimizing weights

        Parameters:
        -----------
        X : np.ndarray
            Validation features
        y : np.ndarray
            Validation targets

        Returns:
        --------
        self : WeightedEnsemble
        """
        if self.verbose:
            logger.info(f"Optimizing weights for {len(self.models)} models using {self.optimization}")

        # Get predictions from all models
        predictions = []
        for model in self.models:
            if self.problem_type == 'classification' and self.use_probas:
                pred = model.predict_proba(X)
                if self.n_classes_ is None:
                    self.n_classes_ = pred.shape[1]
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (n_models, n_samples) or (n_models, n_samples, n_classes)

        # Optimize weights
        if self.optimization == 'grid':
            self.weights_ = self._optimize_weights_grid(predictions, y)
        elif self.optimization == 'gradient':
            self.weights_ = self._optimize_weights_gradient(predictions, y)
        elif self.optimization == 'closed_form':
            self.weights_ = self._optimize_weights_closed_form(predictions, y)
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization}")

        if self.verbose:
            logger.info(f"Optimal weights: {self.weights_}")

        return self

    def _optimize_weights_grid(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using grid search"""
        from sklearn.metrics import accuracy_score, mean_squared_error
        from itertools import product

        # Generate weight combinations
        weight_options = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
        n_models = len(self.models)

        best_score = -np.inf if self.problem_type == 'classification' else np.inf
        best_weights = None

        # Brute force search (only feasible for small number of models)
        if n_models <= 3:
            for weights in product(weight_options, repeat=n_models):
                weights = np.array(weights)
                if np.sum(weights) == 0:
                    continue
                weights = weights / np.sum(weights)  # Normalize

                # Compute weighted prediction
                if self.problem_type == 'classification' and self.use_probas:
                    weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                    y_pred = np.argmax(weighted_pred, axis=1)
                    score = accuracy_score(y, y_pred)
                else:
                    weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                    score = -mean_squared_error(y, weighted_pred)  # Negative MSE for maximization

                if (self.problem_type == 'classification' and score > best_score) or \
                   (self.problem_type == 'regression' and score > best_score):
                    best_score = score
                    best_weights = weights
        else:
            # For more models, use random search
            n_trials = 1000
            for _ in range(n_trials):
                weights = np.random.dirichlet(np.ones(n_models))  # Random weights that sum to 1

                if self.problem_type == 'classification' and self.use_probas:
                    weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                    y_pred = np.argmax(weighted_pred, axis=1)
                    score = accuracy_score(y, y_pred)
                else:
                    weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                    score = -mean_squared_error(y, weighted_pred)

                if (self.problem_type == 'classification' and score > best_score) or \
                   (self.problem_type == 'regression' and score > best_score):
                    best_score = score
                    best_weights = weights

        return best_weights if best_weights is not None else np.ones(n_models) / n_models

    def _optimize_weights_gradient(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using gradient descent"""
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss, mean_squared_error

        n_models = len(self.models)

        def objective(weights):
            weights = np.abs(weights)  # Ensure positive
            weights = weights / np.sum(weights)  # Normalize

            if self.problem_type == 'classification' and self.use_probas:
                weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                # Add small epsilon to avoid log(0)
                weighted_pred = np.clip(weighted_pred, 1e-10, 1 - 1e-10)
                return log_loss(y, weighted_pred)
            else:
                weighted_pred = np.tensordot(weights, predictions, axes=([0], [0]))
                return mean_squared_error(y, weighted_pred)

        # Initial weights
        initial_weights = np.ones(n_models) / n_models

        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=[(0, 1)] * n_models,
                         constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        return result.x

    def _optimize_weights_closed_form(self, predictions: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Optimize weights using closed-form solution (regression only)"""
        if self.problem_type != 'regression':
            logger.warning("Closed-form optimization only works for regression, falling back to grid search")
            return self._optimize_weights_grid(predictions, y)

        # Reshape predictions: (n_models, n_samples) -> (n_samples, n_models)
        X_meta = predictions.T

        # Solve using least squares with non-negative constraint
        from scipy.optimize import nnls

        weights, _ = nnls(X_meta, y)

        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(self.models)) / len(self.models)

        return weights

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted predictions"""
        predictions = []
        for model in self.models:
            if self.problem_type == 'classification' and self.use_probas:
                pred = model.predict_proba(X)
            else:
                pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Apply weights
        if self.problem_type == 'classification' and self.use_probas:
            weighted_pred = np.tensordot(self.weights_, predictions, axes=([0], [0]))
            return np.argmax(weighted_pred, axis=1)
        else:
            return np.tensordot(self.weights_, predictions, axes=([0], [0]))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict weighted class probabilities (classification only)"""
        if self.problem_type != 'classification':
            raise ValueError("predict_proba is only available for classification")

        predictions = []
        for model in self.models:
            predictions.append(model.predict_proba(X))

        predictions = np.array(predictions)
        return np.tensordot(self.weights_, predictions, axes=([0], [0]))


def create_ensemble(
    models: List[BaseEstimator],
    X_val: np.ndarray,
    y_val: np.ndarray,
    ensemble_type: str = 'stacking',
    problem_type: str = 'classification',
    **kwargs
) -> BaseEstimator:
    """
    Factory function to create and train ensemble

    Parameters:
    -----------
    models : List[BaseEstimator]
        List of trained base models
    X_val : np.ndarray
        Validation features for meta-learning
    y_val : np.ndarray
        Validation targets
    ensemble_type : str
        Type of ensemble: 'stacking', 'blending', or 'weighted'
    problem_type : str
        'classification' or 'regression'
    **kwargs
        Additional arguments for ensemble constructor

    Returns:
    --------
    ensemble : BaseEstimator
        Trained ensemble model
    """
    if ensemble_type == 'stacking':
        ensemble = StackingEnsemble(models, problem_type=problem_type, **kwargs)
    elif ensemble_type == 'blending':
        ensemble = BlendingEnsemble(models, problem_type=problem_type, **kwargs)
    elif ensemble_type == 'weighted':
        ensemble = WeightedEnsemble(models, problem_type=problem_type, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    ensemble.fit(X_val, y_val)
    return ensemble

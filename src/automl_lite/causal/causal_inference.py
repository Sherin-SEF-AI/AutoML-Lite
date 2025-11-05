"""
Causal Inference Module

Implements:
- Causal effect estimation
- Propensity score matching
- Double Machine Learning (DML)
- Instrumental variable methods
- Treatment effect heterogeneity
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import logging

logger = logging.getLogger(__name__)


class PropensityScoreMatcher:
    """
    Propensity Score Matching for causal inference

    Estimates treatment effects by matching treated and control units
    """

    def __init__(
        self,
        caliper: float = 0.1,
        n_neighbors: int = 1,
        random_state: int = 42
    ):
        """
        Initialize Propensity Score Matcher

        Parameters:
        -----------
        caliper : float
            Maximum propensity score distance for matching
        n_neighbors : int
            Number of control units to match to each treated unit
        random_state : int
            Random state
        """
        self.caliper = caliper
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.propensity_model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.propensity_scores = None

    def fit(self, X: np.ndarray, treatment: np.ndarray):
        """
        Fit propensity score model

        Parameters:
        -----------
        X : np.ndarray
            Covariates (features)
        treatment : np.ndarray
            Treatment indicator (0 or 1)
        """
        self.propensity_model.fit(X, treatment)
        self.propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        return self

    def match(
        self,
        treatment: np.ndarray,
        propensity_scores: Optional[np.ndarray] = None
    ) -> Dict[int, List[int]]:
        """
        Find matches for treated units

        Parameters:
        -----------
        treatment : np.ndarray
            Treatment indicator
        propensity_scores : np.ndarray, optional
            Propensity scores (uses fitted scores if None)

        Returns:
        --------
        matches : Dict[int, List[int]]
            Dictionary mapping treated indices to control indices
        """
        if propensity_scores is None:
            propensity_scores = self.propensity_scores

        treated_indices = np.where(treatment == 1)[0]
        control_indices = np.where(treatment == 0)[0]

        matches = {}

        for treated_idx in treated_indices:
            treated_ps = propensity_scores[treated_idx]

            # Find control units within caliper
            ps_distances = np.abs(propensity_scores[control_indices] - treated_ps)
            within_caliper = ps_distances <= self.caliper

            if np.any(within_caliper):
                # Get closest matches
                close_controls = control_indices[within_caliper]
                close_distances = ps_distances[within_caliper]

                # Sort by distance and take top n_neighbors
                sorted_indices = np.argsort(close_distances)[:self.n_neighbors]
                matches[treated_idx] = close_controls[sorted_indices].tolist()

        return matches

    def estimate_ate(
        self,
        treatment: np.ndarray,
        outcome: np.ndarray,
        matches: Optional[Dict[int, List[int]]] = None
    ) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect (ATE)

        Parameters:
        -----------
        treatment : np.ndarray
            Treatment indicator
        outcome : np.ndarray
            Outcome variable
        matches : Dict[int, List[int]], optional
            Matching dictionary (computed if None)

        Returns:
        --------
        results : Dict[str, float]
            ATE and statistics
        """
        if matches is None:
            matches = self.match(treatment)

        treatment_effects = []

        for treated_idx, control_indices in matches.items():
            treated_outcome = outcome[treated_idx]
            control_outcomes = outcome[control_indices]

            # Individual treatment effect
            ite = treated_outcome - np.mean(control_outcomes)
            treatment_effects.append(ite)

        ate = np.mean(treatment_effects)
        ate_std = np.std(treatment_effects)

        return {
            'ate': float(ate),
            'ate_std': float(ate_std),
            'n_matched': len(matches),
            'n_treated': np.sum(treatment == 1)
        }


class DoubleMachineLearning:
    """
    Double/Debiased Machine Learning (DML) for causal inference

    Estimates treatment effects using cross-fitting to avoid overfitting bias
    """

    def __init__(
        self,
        model_y=None,  # Model for outcome
        model_t=None,  # Model for treatment
        n_folds: int = 5,
        random_state: int = 42
    ):
        """
        Initialize DML

        Parameters:
        -----------
        model_y : estimator, optional
            Model for outcome (uses RandomForest if None)
        model_t : estimator, optional
            Model for treatment (uses RandomForest if None)
        n_folds : int
            Number of folds for cross-fitting
        random_state : int
            Random state
        """
        self.model_y = model_y or RandomForestRegressor(random_state=random_state)
        self.model_t = model_t or RandomForestRegressor(random_state=random_state)
        self.n_folds = n_folds
        self.random_state = random_state

        self.theta = None  # Treatment effect estimate

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> 'DoubleMachineLearning':
        """
        Fit DML model

        Parameters:
        -----------
        X : np.ndarray
            Covariates
        treatment : np.ndarray
            Treatment variable
        outcome : np.ndarray
            Outcome variable
        """
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Store residuals
        y_residuals = np.zeros(len(outcome))
        t_residuals = np.zeros(len(treatment))

        # Cross-fitting
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            t_train, t_test = treatment[train_idx], treatment[test_idx]
            y_train, y_test = outcome[train_idx], outcome[test_idx]

            # Fit outcome model
            self.model_y.fit(X_train, y_train)
            y_pred = self.model_y.predict(X_test)
            y_residuals[test_idx] = y_test - y_pred

            # Fit treatment model
            self.model_t.fit(X_train, t_train)
            t_pred = self.model_t.predict(X_test)
            t_residuals[test_idx] = t_test - t_pred

        # Estimate treatment effect using residuals
        # theta = E[Y_residual * T_residual] / E[T_residual^2]
        self.theta = np.sum(y_residuals * t_residuals) / np.sum(t_residuals ** 2)

        return self

    def estimate_ate(self) -> Dict[str, float]:
        """
        Get Average Treatment Effect estimate

        Returns:
        --------
        results : Dict[str, float]
            ATE estimate
        """
        if self.theta is None:
            raise ValueError("Model not fitted yet")

        return {
            'ate': float(self.theta),
            'method': 'double_ml'
        }


class CausalForest:
    """
    Causal Forest for heterogeneous treatment effect estimation

    Simplified implementation using Random Forest with treatment interaction
    """

    def __init__(
        self,
        n_estimators: int = 100,
        min_samples_leaf: int = 10,
        random_state: int = 42
    ):
        """
        Initialize Causal Forest

        Parameters:
        -----------
        n_estimators : int
            Number of trees
        min_samples_leaf : int
            Minimum samples per leaf
        random_state : int
            Random state
        """
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.forest = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )

    def fit(
        self,
        X: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray
    ) -> 'CausalForest':
        """
        Fit causal forest

        Parameters:
        -----------
        X : np.ndarray
            Covariates
        treatment : np.ndarray
            Treatment variable
        outcome : np.ndarray
            Outcome variable
        """
        # Create features with treatment interaction
        X_augmented = np.column_stack([X, treatment, X * treatment.reshape(-1, 1)])

        self.forest.fit(X_augmented, outcome)

        return self

    def predict_ite(self, X: np.ndarray) -> np.ndarray:
        """
        Predict Individual Treatment Effects (ITE)

        Parameters:
        -----------
        X : np.ndarray
            Covariates

        Returns:
        --------
        ite : np.ndarray
            Individual treatment effects
        """
        # Predict with treatment = 1
        X_treated = np.column_stack([X, np.ones(len(X)), X])
        y1 = self.forest.predict(X_treated)

        # Predict with treatment = 0
        X_control = np.column_stack([X, np.zeros(len(X)), np.zeros_like(X)])
        y0 = self.forest.predict(X_control)

        # ITE = E[Y|X, T=1] - E[Y|X, T=0]
        return y1 - y0

    def estimate_ate(self, X: np.ndarray) -> Dict[str, float]:
        """
        Estimate Average Treatment Effect

        Parameters:
        -----------
        X : np.ndarray
            Covariates

        Returns:
        --------
        results : Dict[str, float]
            ATE estimate
        """
        ite = self.predict_ite(X)

        return {
            'ate': float(np.mean(ite)),
            'ate_std': float(np.std(ite)),
            'method': 'causal_forest'
        }


def estimate_treatment_effect(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    method: str = 'dml',
    **kwargs
) -> Dict[str, Any]:
    """
    Factory function for treatment effect estimation

    Parameters:
    -----------
    X : np.ndarray
        Covariates
    treatment : np.ndarray
        Treatment variable
    outcome : np.ndarray
        Outcome variable
    method : str
        Estimation method: 'psm', 'dml', or 'causal_forest'
    **kwargs
        Additional parameters for specific methods

    Returns:
    --------
    results : Dict[str, Any]
        Treatment effect estimates
    """
    if method == 'psm':
        estimator = PropensityScoreMatcher(**kwargs)
        estimator.fit(X, treatment)
        return estimator.estimate_ate(treatment, outcome)

    elif method == 'dml':
        estimator = DoubleMachineLearning(**kwargs)
        estimator.fit(X, treatment, outcome)
        return estimator.estimate_ate()

    elif method == 'causal_forest':
        estimator = CausalForest(**kwargs)
        estimator.fit(X, treatment, outcome)
        return estimator.estimate_ate(X)

    else:
        raise ValueError(f"Unknown method: {method}")

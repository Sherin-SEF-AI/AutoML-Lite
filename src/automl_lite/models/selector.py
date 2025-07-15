"""
Model selection for AutoML Lite.
"""

from typing import Any, Dict, List, Tuple, Type
import numpy as np

# Import scikit-learn models
try:
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor,
        GradientBoostingClassifier, GradientBoostingRegressor,
        ExtraTreesClassifier, ExtraTreesRegressor,
        VotingClassifier, VotingRegressor
    )
    from sklearn.linear_model import (
        LogisticRegression, LinearRegression,
        Ridge, Lasso, ElasticNet
    )
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import IsolationForest
    from imblearn.ensemble import BalancedRandomForestClassifier
except ImportError:
    # Fallback for missing dependencies
    pass


class ModelSelector:
    """
    Model selector for different problem types.
    """
    
    def __init__(self):
        """Initialize the model selector."""
        self.classification_models = self._get_classification_models()
        self.regression_models = self._get_regression_models()
        self.clustering_models = self._get_clustering_models()
    
    def get_models(self, problem_type: str) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """
        Get models for a specific problem type.
        
        Args:
            problem_type: Type of problem ('classification', 'regression', 'clustering')
            
        Returns:
            List of (model_name, model_class, param_grid) tuples
        """
        if problem_type == "classification":
            return self.classification_models
        elif problem_type == "regression":
            return self.regression_models
        elif problem_type == "clustering":
            return self.clustering_models
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def _get_classification_models(self) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """Get classification models with parameter grids."""
        models = []
        
        # Random Forest
        models.append((
            "Random Forest",
            RandomForestClassifier,
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        ))
        
        # Gradient Boosting
        models.append((
            "Gradient Boosting",
            GradientBoostingClassifier,
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'random_state': [42]
            }
        ))
        
        # Support Vector Machine
        models.append((
            "SVM",
            SVC,
            {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'random_state': [42]
            }
        ))
        
        # Logistic Regression
        models.append((
            "Logistic Regression",
            LogisticRegression,
            {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'random_state': [42]
            }
        ))
        
        # K-Nearest Neighbors
        models.append((
            "K-Nearest Neighbors",
            KNeighborsClassifier,
            {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        ))
        
        # Decision Tree
        models.append((
            "Decision Tree",
            DecisionTreeClassifier,
            {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        ))
        
        # Neural Network
        models.append((
            "Neural Network",
            MLPClassifier,
            {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'random_state': [42]
            }
        ))
        
        # Extra Trees
        models.append((
            "Extra Trees",
            ExtraTreesClassifier,
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'random_state': [42]
            }
        ))
        
        # Gaussian Naive Bayes
        models.append((
            "Gaussian Naive Bayes",
            GaussianNB,
            {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
            }
        ))
        
        # Linear Discriminant Analysis
        models.append((
            "Linear Discriminant Analysis",
            LinearDiscriminantAnalysis,
            {
                'solver': ['svd', 'lsqr', 'eigen'],
                'shrinkage': [None, 'auto', 0.1, 0.5, 0.9]
            }
        ))
        
        return models
    
    def _get_regression_models(self) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """Get regression models with parameter grids."""
        models = []
        
        # Random Forest
        models.append((
            "Random Forest",
            RandomForestRegressor,
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        ))
        
        # Gradient Boosting
        models.append((
            "Gradient Boosting",
            GradientBoostingRegressor,
            {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'random_state': [42]
            }
        ))
        
        # Support Vector Regression
        models.append((
            "SVR",
            SVR,
            {
                'C': [0.1, 1, 10, 100],
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        ))
        
        # Linear Regression
        models.append((
            "Linear Regression",
            LinearRegression,
            {}
        ))
        
        # Ridge Regression
        models.append((
            "Ridge Regression",
            Ridge,
            {
                'alpha': [0.1, 1, 10, 100],
                'random_state': [42]
            }
        ))
        
        # Lasso Regression
        models.append((
            "Lasso Regression",
            Lasso,
            {
                'alpha': [0.1, 1, 10, 100],
                'random_state': [42]
            }
        ))
        
        # Elastic Net
        models.append((
            "Elastic Net",
            ElasticNet,
            {
                'alpha': [0.1, 1, 10],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                'random_state': [42]
            }
        ))
        
        # K-Nearest Neighbors
        models.append((
            "K-Nearest Neighbors",
            KNeighborsRegressor,
            {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        ))
        
        # Decision Tree
        models.append((
            "Decision Tree",
            DecisionTreeRegressor,
            {
                'max_depth': [None, 5, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'random_state': [42]
            }
        ))
        
        # Neural Network
        models.append((
            "Neural Network",
            MLPRegressor,
            {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'random_state': [42]
            }
        ))
        
        # Extra Trees
        models.append((
            "Extra Trees",
            ExtraTreesRegressor,
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'random_state': [42]
            }
        ))
        
        return models
    
    def _get_clustering_models(self) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """Get clustering models with parameter grids."""
        models = []
        
        # K-Means
        models.append((
            "K-Means",
            KMeans,
            {
                'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                'init': ['k-means++', 'random'],
                'n_init': [10],
                'random_state': [42]
            }
        ))
        
        # DBSCAN
        models.append((
            "DBSCAN",
            DBSCAN,
            {
                'eps': [0.1, 0.3, 0.5, 0.7, 1.0],
                'min_samples': [2, 3, 5, 10]
            }
        ))
        
        return models
    
    def get_ensemble_models(self, problem_type: str) -> List[Tuple[str, Type, Dict[str, Any]]]:
        """
        Get ensemble models for a specific problem type.
        
        Args:
            problem_type: Type of problem
            
        Returns:
            List of ensemble models
        """
        if problem_type == "classification":
            return [
                (
                    "Voting Classifier",
                    VotingClassifier,
                    {
                        'estimators': [
                            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                            ('svm', SVC(probability=True, random_state=42))
                        ],
                        'voting': ['hard', 'soft']
                    }
                )
            ]
        elif problem_type == "regression":
            return [
                (
                    "Voting Regressor",
                    VotingRegressor,
                    {
                        'estimators': [
                            ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                            ('svr', SVR())
                        ]
                    }
                )
            ]
        else:
            return [] 
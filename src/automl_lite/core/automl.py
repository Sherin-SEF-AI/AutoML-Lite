"""
Main AutoMLite class for automated machine learning.
"""

import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier, VotingRegressor
from tqdm import tqdm

from ..preprocessing.pipeline import PreprocessingPipeline
from ..preprocessing.feature_engineering import AutoFeatureEngineer
from ..models.selector import ModelSelector
from ..models.deep_learning import DeepLearningModel
from ..models.time_series import TimeSeriesForecaster
from ..optimization.optimizer import HyperparameterOptimizer
from ..visualization.reporter import ReportGenerator
from ..interpretability.advanced_interpreter import AdvancedInterpreter
from ..config.advanced_config import AutoMLConfig, ConfigManager
from ..experiments.tracker import ExperimentTracker, ExperimentManager
from ..ui.interactive_dashboard import AutoMLDashboard
from ..utils.problem_detector import ProblemDetector
from ..utils.validators import DataValidator
from ..utils.logger import get_logger

logger = get_logger(__name__)


class AutoMLite:
    """
    Main AutoML Lite class for automated machine learning.
    
    This class provides end-to-end ML automation including:
    - Automatic problem type detection
    - Intelligent preprocessing
    - Model selection and optimization
    - Comprehensive evaluation and reporting
    - Ensemble methods
    - Model interpretability
    """
    
    def __init__(
        self,
        time_budget: int = 600,
        max_models: int = 10,
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True,
        n_jobs: int = -1,
        enable_ensemble: bool = True,
        enable_early_stopping: bool = True,
        enable_feature_selection: bool = True,
        enable_interpretability: bool = True,
        enable_auto_feature_engineering: bool = True,
        enable_deep_learning: bool = False,
        enable_time_series: bool = False,
        enable_experiment_tracking: bool = False,
        ensemble_method: str = "voting",
        top_k_models: int = 3,
        early_stopping_patience: int = 10,
        feature_selection_method: str = "mutual_info",
        feature_selection_threshold: float = 0.01,
        config: Optional[AutoMLConfig] = None,
        experiment_tracker: Optional[ExperimentTracker] = None,
    ) -> None:
        """
        Initialize AutoML Lite with production-ready features.
        
        Args:
            time_budget: Maximum time in seconds for training
            max_models: Maximum number of models to try
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            verbose: Whether to show progress
            n_jobs: Number of parallel jobs
            enable_ensemble: Whether to create ensemble models
            enable_early_stopping: Whether to use early stopping
            enable_feature_selection: Whether to perform feature selection
            enable_interpretability: Whether to enable model interpretability
            enable_auto_feature_engineering: Whether to enable auto feature engineering
            enable_deep_learning: Whether to enable deep learning models
            enable_time_series: Whether to enable time series forecasting
            enable_experiment_tracking: Whether to enable experiment tracking
            ensemble_method: Ensemble method ('voting', 'stacking', 'blending')
            top_k_models: Number of top models to use in ensemble
            early_stopping_patience: Patience for early stopping
            feature_selection_method: Feature selection method
            feature_selection_threshold: Threshold for feature selection
            config: Advanced configuration object
            experiment_tracker: Experiment tracking object
        """
        # Use config if provided, otherwise use default parameters
        if config is not None:
            self.config = config
            self.time_budget = config.time_budget
            self.max_models = config.max_models
            self.cv_folds = config.cv_folds
            self.random_state = config.random_state
            self.verbose = config.verbose
            self.n_jobs = config.n_jobs
            self.enable_ensemble = config.enable_ensemble
            self.enable_early_stopping = config.enable_early_stopping
            self.enable_feature_selection = config.enable_feature_selection
            self.enable_interpretability = config.enable_interpretability
            self.enable_auto_feature_engineering = config.enable_auto_feature_engineering
            self.ensemble_method = config.ensemble_method.value
            self.top_k_models = config.top_k_models
            self.early_stopping_patience = config.early_stopping_patience
        else:
            self.config = None
            self.time_budget = time_budget
            self.max_models = max_models
            self.cv_folds = cv_folds
            self.random_state = random_state
            self.verbose = verbose
            self.n_jobs = n_jobs
            self.enable_ensemble = enable_ensemble
            self.enable_early_stopping = enable_early_stopping
            self.enable_feature_selection = enable_feature_selection
            self.enable_interpretability = enable_interpretability
            self.enable_auto_feature_engineering = enable_auto_feature_engineering
            self.ensemble_method = ensemble_method
            self.top_k_models = top_k_models
            self.early_stopping_patience = early_stopping_patience
        
        # Additional features
        self.enable_deep_learning = enable_deep_learning
        self.enable_time_series = enable_time_series
        self.enable_experiment_tracking = enable_experiment_tracking
        self.feature_selection_method = feature_selection_method
        self.feature_selection_threshold = feature_selection_threshold
        
        # Initialize components
        self.problem_detector = ProblemDetector()
        self.data_validator = DataValidator()
        self.preprocessor = PreprocessingPipeline()
        self.model_selector = ModelSelector()
        self.optimizer = HyperparameterOptimizer()
        self.report_generator = ReportGenerator()
        
        # Production-ready components
        if self.enable_auto_feature_engineering:
            self.feature_engineer = AutoFeatureEngineer()
        
        if self.enable_interpretability:
            self.interpreter = AdvancedInterpreter()
        
        if self.enable_deep_learning:
            self.deep_learning_model = None  # Will be initialized when needed
        
        if self.enable_time_series:
            self.time_series_forecaster = TimeSeriesForecaster()
        
        if self.enable_experiment_tracking:
            self.experiment_tracker = experiment_tracker or ExperimentTracker()
            self.experiment_manager = ExperimentManager()
        
        # Configuration management
        self.config_manager = ConfigManager()
        
        # State variables
        self.is_fitted = False
        self.problem_type = None
        self.best_model = None
        self.best_model_name = None
        self.best_score = None
        self.feature_importance = None
        self.leaderboard = None
        self.training_history = []
        self.preprocessing_config = None
        self.ensemble_model = None
        self.selected_features = None
        self.interpretability_results = None
        
        # Set random seeds
        np.random.seed(random_state)
        
        if verbose:
            logger.info("AutoML Lite initialized successfully")
            logger.info(f"Advanced features: Ensemble={enable_ensemble}, "
                       f"Early Stopping={enable_early_stopping}, "
                       f"Feature Selection={enable_feature_selection}, "
                       f"Interpretability={enable_interpretability}")
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
    ) -> "AutoMLite":
        """
        Fit the AutoML pipeline.
        
        Args:
            X: Feature matrix
            y: Target variable (None for clustering)
            preprocessing_config: Custom preprocessing configuration
            
        Returns:
            self: Fitted AutoMLite instance
        """
        start_time = time.time()
        
        if self.verbose:
            logger.info("Starting AutoML training...")
        
        # Start experiment tracking if enabled
        if self.enable_experiment_tracking and hasattr(self, 'experiment_tracker'):
            self.experiment_tracker.start_run()
            self.experiment_tracker.log_params({
                'time_budget': self.time_budget,
                'max_models': self.max_models,
                'cv_folds': self.cv_folds,
                'enable_ensemble': self.enable_ensemble,
                'enable_feature_selection': self.enable_feature_selection
            })
        
        # Validate input data
        X = self._validate_input(X)
        if y is not None:
            y = self._validate_target(y)
        
        # Detect problem type
        self.problem_type = self.problem_detector.detect(X, y)
        if self.verbose:
            logger.info(f"Detected problem type: {self.problem_type}")
        
        # Configure preprocessing
        self.preprocessing_config = preprocessing_config or {}
        
        # Preprocess data
        if self.verbose:
            logger.info("Preprocessing data...")
        X_processed = self.preprocessor.fit_transform(
            X, y, self.problem_type, self.preprocessing_config
        )
        
        # Auto feature engineering
        if self.enable_auto_feature_engineering and hasattr(self, 'feature_engineer'):
            if self.verbose:
                logger.info("Performing auto feature engineering...")
            X_processed = self.feature_engineer.fit_transform(X_processed, y)
        
        # Feature selection
        if self.enable_feature_selection:
            X_processed, self.selected_features = self._perform_feature_selection(
                X_processed, y
            )
        
        # Select and optimize models
        if self.verbose:
            logger.info("Selecting and optimizing models...")
        
        self._optimize_models(X_processed, y, start_time)
        
        # Create ensemble if enabled
        if self.enable_ensemble and len(self.training_history) > 1:
            self.ensemble_model = self._create_ensemble(X_processed, y)
        
        # Calculate feature importance
        if self.best_model is not None:
            self.feature_importance = self._calculate_feature_importance(
                X_processed, y
            )
        
        # Time series forecasting
        if self.enable_time_series and hasattr(self, 'time_series_forecaster'):
            if self.verbose:
                logger.info("Training time series forecaster...")
            self.time_series_forecaster.fit(self.best_model, X_processed, y)
        
        # Deep learning models
        if self.enable_deep_learning:
            if self.verbose:
                logger.info("Training deep learning model...")
            self.deep_learning_model = DeepLearningModel(
                framework="tensorflow",
                model_type="mlp",
                output_units=len(np.unique(y)) if self.problem_type == "classification" else 1
            )
            self.deep_learning_model.fit(X_processed, y)
        
        # Advanced interpretability
        if self.enable_interpretability and hasattr(self, 'interpreter'):
            if self.verbose:
                logger.info("Performing advanced interpretability analysis...")
            self.interpreter.fit(self.best_model, X_processed, y)
            self.interpretability_results = self.interpreter.get_interpretability_report()
        
        # Log experiment results
        if self.enable_experiment_tracking and hasattr(self, 'experiment_tracker'):
            experiment_results = {
                'best_score': self.best_score,
                'training_time': time.time() - start_time,
                'n_models_tried': len(self.training_history),
                'config': {
                    'time_budget': self.time_budget,
                    'max_models': self.max_models,
                    'cv_folds': self.cv_folds,
                    'problem_type': self.problem_type
                },
                'leaderboard': self.leaderboard,
                'feature_importance': self.feature_importance
            }
            self.experiment_tracker.log_automl_results(experiment_results)
            self.experiment_tracker.end_run()
        
        self.is_fitted = True
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(f"Training completed in {elapsed_time:.2f} seconds")
            logger.info(f"Best model: {self.best_model_name}")
            logger.info(f"Best score: {self.best_score:.4f}")
            if self.ensemble_model:
                logger.info("Ensemble model created successfully")
            if self.enable_auto_feature_engineering:
                logger.info("Auto feature engineering completed")
            if self.enable_deep_learning:
                logger.info("Deep learning model trained")
            if self.enable_time_series:
                logger.info("Time series forecasting model trained")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X = self._validate_input(X)
        X_processed = self.preprocessor.transform(X)
        
        # Apply auto feature engineering if enabled
        if self.enable_auto_feature_engineering and hasattr(self, 'feature_engineer'):
            X_processed = self.feature_engineer.transform(X_processed)
        
        # Apply feature selection if enabled
        if self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
        
        # Use ensemble if available and better
        if self.ensemble_model is not None:
            return self.ensemble_model.predict(X_processed)
        else:
            return self.best_model.predict(X_processed)
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities (classification only).
        
        Args:
            X: Feature matrix
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if self.problem_type != "classification":
            raise ValueError("predict_proba is only available for classification")
        
        X = self._validate_input(X)
        X_processed = self.preprocessor.transform(X)
        
        # Apply auto feature engineering if enabled
        if self.enable_auto_feature_engineering and hasattr(self, 'feature_engineer'):
            X_processed = self.feature_engineer.transform(X_processed)
        
        # Apply feature selection if enabled
        if self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
        
        # Use ensemble if available and better
        model = self.ensemble_model if self.ensemble_model is not None else self.best_model
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X_processed)
        else:
            # Fallback: try to use decision_function or predict, or raise a clear error
            if hasattr(model, "decision_function"):
                # Convert decision_function output to probabilities using softmax
                from scipy.special import softmax
                decision = model.decision_function(X_processed)
                if decision.ndim == 1:
                    # Binary classification
                    proba = softmax(np.vstack([-decision, decision]).T, axis=1)
                else:
                    proba = softmax(decision, axis=1)
                return proba
            else:
                raise AttributeError(
                    f"The selected model ({type(model).__name__}) does not support probability prediction. "
                    "Use predict() for class labels instead."
                )
    
    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Evaluate model performance.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Model score
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before scoring")
        
        X = self._validate_input(X)
        y = self._validate_target(y)
        X_processed = self.preprocessor.transform(X)
        
        # Apply auto feature engineering if enabled
        if self.enable_auto_feature_engineering and hasattr(self, 'feature_engineer'):
            X_processed = self.feature_engineer.transform(X_processed)
        
        # Apply feature selection if enabled
        if self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
        
        # Use ensemble if available and better
        if self.ensemble_model is not None:
            return self.ensemble_model.score(X_processed, y)
        else:
            return self.best_model.score(X_processed, y)
    
    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'best_score': self.best_score,
            'problem_type': self.problem_type,
            'preprocessor': self.preprocessor,
            'ensemble_model': self.ensemble_model,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'leaderboard': self.leaderboard,
            'training_history': self.training_history,
            'interpretability_results': self.interpretability_results,
        }
        
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: Union[str, Path]) -> "AutoMLite":
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            self: Loaded AutoMLite instance
        """
        model_data = joblib.load(path)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.best_score = model_data['best_score']
        self.problem_type = model_data['problem_type']
        self.preprocessor = model_data['preprocessor']
        self.ensemble_model = model_data.get('ensemble_model')
        self.selected_features = model_data.get('selected_features')
        self.feature_importance = model_data.get('feature_importance')
        self.leaderboard = model_data.get('leaderboard')
        self.training_history = model_data.get('training_history', [])
        self.interpretability_results = model_data.get('interpretability_results')
        self.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return self
    
    @classmethod
    def load_model_from_file(cls, path: Union[str, Path]) -> "AutoMLite":
        """
        Load a trained model from disk (class method).
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded AutoMLite instance
        """
        model_data = joblib.load(path)
        
        # Create new instance
        instance = cls()
        
        # Load model data
        instance.best_model = model_data['best_model']
        instance.best_model_name = model_data['best_model_name']
        instance.best_score = model_data['best_score']
        instance.problem_type = model_data['problem_type']
        instance.preprocessor = model_data['preprocessor']
        instance.ensemble_model = model_data.get('ensemble_model')
        instance.selected_features = model_data.get('selected_features')
        instance.feature_importance = model_data.get('feature_importance')
        instance.leaderboard = model_data.get('leaderboard')
        instance.training_history = model_data.get('training_history', [])
        instance.interpretability_results = model_data.get('interpretability_results')
        instance.is_fitted = True
        
        logger.info(f"Model loaded from {path}")
        return instance
    
    def generate_report(self, path: Union[str, Path], X_test: Optional[pd.DataFrame] = None, y_test: Optional[np.ndarray] = None) -> None:
        """
        Generate a comprehensive HTML report.
        
        Args:
            path: Path to save the report
            X_test: Test features for additional visualizations
            y_test: Test targets for additional visualizations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating report")
        
        # Convert leaderboard to list of dicts if needed
        leaderboard_list = None
        if self.leaderboard is not None:
            if isinstance(self.leaderboard, pd.DataFrame):
                if not self.leaderboard.empty:
                    leaderboard_list = self.leaderboard.to_dict('records')
            elif isinstance(self.leaderboard, list):
                leaderboard_list = self.leaderboard
        
        self.report_generator.generate_report(
            path=path,
            automl=self,
            problem_type=self.problem_type,
            leaderboard=leaderboard_list,
            feature_importance=self.feature_importance,
            training_history=self.training_history,
            ensemble_info=self.get_ensemble_info(),
            interpretability_results=self.get_interpretability_report(),
            X_test=X_test,
            y_test=y_test
        )
    
    def get_leaderboard(self) -> pd.DataFrame:
        """
        Get the model leaderboard.
        
        Returns:
            DataFrame with model rankings
        """
        if self.leaderboard is None:
            return pd.DataFrame()
        return self.leaderboard.copy()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in self.feature_importance.items()
        ])
        return importance_df.sort_values('importance', ascending=False)
    
    def get_interpretability_report(self) -> Dict[str, Any]:
        """
        Get model interpretability results.
        
        Returns:
            Dictionary with interpretability results
        """
        return self.interpretability_results or {}
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get ensemble model information.
        
        Returns:
            Dictionary with ensemble information
        """
        if self.ensemble_model is None:
            return {}
        
        # Ensure all values are scalars
        ensemble_score = None
        if hasattr(self.ensemble_model, 'score_'):
            try:
                # If it's a numpy array, convert to float
                import numpy as np
                score_val = self.ensemble_model.score_
                if isinstance(score_val, np.ndarray):
                    ensemble_score = float(score_val.mean())
                else:
                    ensemble_score = float(score_val)
            except Exception:
                ensemble_score = None
        # If it's a method, set to None
        if callable(getattr(self.ensemble_model, 'score', None)):
            pass  # Do not use the method
        return {
            'ensemble_method': self.ensemble_method,
            'top_k_models': self.top_k_models,
            'ensemble_score': ensemble_score,
        }
    
    def get_feature_engineering_summary(self) -> Dict[str, Any]:
        """Get feature engineering summary."""
        if not self.enable_auto_feature_engineering or not hasattr(self, 'feature_engineer'):
            return {}
        
        return self.feature_engineer.get_feature_summary()
    
    def get_deep_learning_summary(self) -> Dict[str, Any]:
        """Get deep learning model summary."""
        if not self.enable_deep_learning or self.deep_learning_model is None:
            return {}
        
        return self.deep_learning_model.get_model_summary()
    
    def get_time_series_summary(self) -> Dict[str, Any]:
        """Get time series forecasting summary."""
        if not self.enable_time_series or not hasattr(self, 'time_series_forecaster'):
            return {}
        
        return self.time_series_forecaster.get_model_summary()
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get experiment tracking summary."""
        if not self.enable_experiment_tracking or not hasattr(self, 'experiment_tracker'):
            return {}
        
        return self.experiment_tracker.get_experiment_summary()
    
    def run_dashboard(self):
        """Run the interactive dashboard."""
        if not hasattr(self, 'dashboard'):
            self.dashboard = AutoMLDashboard("AutoML Lite Dashboard")
        
        self.dashboard.run_dashboard()
    
    def load_config(self, config_path: str) -> 'AutoMLite':
        """Load configuration from file."""
        config = self.config_manager.load_config(config_path)
        return AutoMLite(config=config)
    
    def save_config(self, config_path: str):
        """Save current configuration to file."""
        if self.config is not None:
            self.config_manager.save_config(self.config, config_path)
    
    def get_config_template(self, template_name: str) -> AutoMLConfig:
        """Get configuration template."""
        return self.config_manager.get_template(template_name)
    
    def list_config_templates(self) -> List[str]:
        """List available configuration templates."""
        return self.config_manager.list_templates()
    
    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Validate and convert input to DataFrame."""
        return self.data_validator.validate_features(X)
    
    def _validate_target(self, y: Union[pd.Series, np.ndarray]) -> pd.Series:
        """Validate and convert target to Series."""
        return self.data_validator.validate_target(y)
    
    def _optimize_models(
        self, X: pd.DataFrame, y: Optional[pd.Series], start_time: float
    ) -> None:
        """Optimize models with early stopping."""
        models = self.model_selector.get_models(self.problem_type)
        self.leaderboard = []
        best_score = float('-inf') if self.problem_type == "classification" else float('inf')
        
        for i, (model_name, model_class, param_grid) in enumerate(tqdm(
            models[:self.max_models], 
            desc="Training models",
            disable=not self.verbose
        )):
            if time.time() - start_time > self.time_budget:
                logger.info("Time budget exceeded, stopping training")
                break
            
            try:
                # Optimize hyperparameters
                best_params, score = self.optimizer.optimize(
                    model_class,
                    param_grid,
                    X, y,
                    self.problem_type,
                    self.cv_folds,
                    self.n_jobs,
                    early_stopping=self.enable_early_stopping,
                    patience=self.early_stopping_patience
                )
                
                # Create best model
                best_model = model_class(**best_params)
                best_model.fit(X, y)
                
                # Update best model if better
                is_better = (
                    (self.problem_type == "classification" and score > best_score) or
                    (self.problem_type == "regression" and score < best_score)
                )
                
                if is_better:
                    self.best_model = best_model
                    self.best_model_name = model_name
                    self.best_score = score
                    best_score = score
                
                # Store in leaderboard
                self.leaderboard.append({
                    'rank': len(self.leaderboard) + 1,
                    'model_name': model_name,
                    'score': score,
                    'params': best_params
                })
                
                # Store training history
                self.training_history.append({
                    'model_name': model_name,
                    'score': score,
                    'params': best_params,
                    'training_time': time.time() - start_time
                })
                
                if self.verbose:
                    logger.info(f"{model_name}: {score:.4f}")
                
            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {str(e)}")
                continue
        
        if not self.leaderboard:
            raise ValueError("No models were successfully trained")
        
        # Convert leaderboard to DataFrame
        self.leaderboard = pd.DataFrame(self.leaderboard)
        self.leaderboard = self.leaderboard.sort_values(
            'score', 
            ascending=(self.problem_type == "regression")
        ).reset_index(drop=True)
    
    def _create_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Optional[BaseEstimator]:
        """Create ensemble model from top performers."""
        if len(self.training_history) < 2:
            return None
        
        # Get top k models
        top_models = self.leaderboard.head(self.top_k_models)
        
        if len(top_models) < 2:
            return None
        
        # Create ensemble
        if self.ensemble_method == "voting":
            if self.problem_type == "classification":
                # Check if all models support predict_proba for soft voting
                estimators = []
                for i, row in top_models.iterrows():
                    model = self._get_model_by_name(row['model_name'])
                    if model is not None:
                        estimators.append((f"{row['model_name']}_{i}", model))
                
                if not estimators:
                    return None
                
                # Check if all models support predict_proba
                all_support_proba = all(hasattr(model, 'predict_proba') for _, model in estimators)
                
                if all_support_proba:
                    ensemble = VotingClassifier(estimators=estimators, voting='soft')
                else:
                    # Fallback to hard voting if not all models support predict_proba
                    if self.verbose:
                        logger.warning("Not all models support predict_proba, using hard voting for ensemble")
                    ensemble = VotingClassifier(estimators=estimators, voting='hard')
                
            else:
                ensemble = VotingRegressor(
                    estimators=[(f"{row['model_name']}_{i}", self._get_model_by_name(row['model_name']))
                               for i, row in top_models.iterrows()]
                )
            
            ensemble.fit(X, y)
            return ensemble
        
        return None
    
    def _get_model_by_name(self, model_name: str) -> BaseEstimator:
        """Get model instance by name from training history."""
        for history in self.training_history:
            if history['model_name'] == model_name:
                # Recreate model with best parameters
                models = self.model_selector.get_models(self.problem_type)
                for name, model_class, _ in models:
                    if name == model_name:
                        return model_class(**history['params'])
        return None
    
    def _perform_feature_selection(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Tuple[pd.DataFrame, Optional[List[str]]]:
        """Perform feature selection."""
        if y is None or self.problem_type == "clustering":
            return X, None
        
        try:
            from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression
            
            if self.problem_type == "classification":
                selector = SelectKBest(
                    score_func=mutual_info_classif,
                    k=max(1, int(X.shape[1] * 0.8))  # Keep 80% of features
                )
            else:
                selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=max(1, int(X.shape[1] * 0.8))
                )
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            if self.verbose:
                logger.info(f"Feature selection: {len(selected_features)}/{X.shape[1]} features selected")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
        
        except Exception as e:
            logger.warning(f"Feature selection failed: {str(e)}")
            return X, None
    
    def _calculate_feature_importance(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Calculate feature importance."""
        if y is None or self.problem_type == "clustering":
            return {}
        
        try:
            if hasattr(self.best_model, 'feature_importances_'):
                importance = self.best_model.feature_importances_
            elif hasattr(self.best_model, 'coef_'):
                importance = np.abs(self.best_model.coef_)
                if importance.ndim > 1:
                    importance = np.mean(importance, axis=0)
            else:
                # Use permutation importance
                from sklearn.inspection import permutation_importance
                result = permutation_importance(
                    self.best_model, X, y, n_repeats=5, random_state=self.random_state
                )
                importance = result.importances_mean
            
            return dict(zip(X.columns, importance))
        
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {str(e)}")
            return {}
    
    def _generate_interpretability_report(
        self, X: pd.DataFrame, y: Optional[pd.Series]
    ) -> Dict[str, Any]:
        """Generate model interpretability report."""
        if y is None:
            return {}
        
        try:
            import shap
            
            # SHAP values
            explainer = shap.TreeExplainer(self.best_model) if hasattr(self.best_model, 'feature_importances_') else shap.KernelExplainer(self.best_model.predict, X.iloc[:100])
            shap_values = explainer.shap_values(X.iloc[:100])
            
            return {
                'shap_values': shap_values,
                'feature_names': X.columns.tolist(),
                'explainer_type': type(explainer).__name__
            }
        
        except Exception as e:
            logger.warning(f"SHAP interpretability failed: {str(e)}")
            return {} 
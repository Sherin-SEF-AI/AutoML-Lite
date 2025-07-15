"""
Configuration management system for AutoML Lite.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProblemType(Enum):
    """Problem types supported by AutoML Lite."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


class EnsembleMethod(Enum):
    """Ensemble methods supported by AutoML Lite."""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"


class FeatureSelectionMethod(Enum):
    """Feature selection methods supported by AutoML Lite."""
    MUTUAL_INFO = "mutual_info"
    RECURSIVE = "recursive"
    SELECT_K_BEST = "select_k_best"
    VARIANCE_THRESHOLD = "variance_threshold"


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Missing value handling
    missing_value_strategy: str = "auto"  # auto, mean, median, mode, drop
    missing_value_threshold: float = 0.5  # Maximum fraction of missing values
    
    # Categorical encoding
    categorical_encoding: str = "auto"  # auto, onehot, label, target
    max_categories: int = 10  # Maximum categories for one-hot encoding
    
    # Scaling
    scaling: str = "auto"  # auto, standard, robust, minmax, none
    scaling_with_mean: bool = True
    scaling_with_std: bool = True
    
    # Outlier handling
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 1.5
    
    # Feature selection
    feature_selection: bool = True
    feature_selection_method: str = "mutual_info"
    feature_selection_threshold: float = 0.01
    max_features: Optional[int] = None
    
    # Data validation
    validate_data: bool = True
    check_duplicates: bool = True
    check_constant_features: bool = True


@dataclass
class ModelConfig:
    """Configuration for model selection and training."""
    
    # Model selection
    models_to_try: List[str] = field(default_factory=lambda: [
        "random_forest", "gradient_boosting", "svm", "logistic_regression",
        "knn", "naive_bayes", "decision_tree", "neural_network"
    ])
    
    # Hyperparameter optimization
    optimization_method: str = "optuna"  # optuna, random, grid
    max_trials: int = 100
    timeout: int = 300  # seconds
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, kfold, time_series
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001
    
    # Ensemble
    enable_ensemble: bool = True
    ensemble_method: str = "voting"
    top_k_models: int = 3
    ensemble_voting: str = "soft"  # soft, hard
    
    # Model-specific settings
    random_forest_config: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    })
    
    gradient_boosting_config: Dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 0.9, 1.0]
    })
    
    svm_config: Dict[str, Any] = field(default_factory=lambda: {
        "C": [0.1, 1, 10, 100],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto", 0.001, 0.01, 0.1]
    })
    
    neural_network_config: Dict[str, Any] = field(default_factory=lambda: {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"]
    })


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    classification_metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"
    ])
    
    regression_metrics: List[str] = field(default_factory=lambda: [
        "mse", "mae", "r2", "rmse", "mape"
    ])
    
    clustering_metrics: List[str] = field(default_factory=lambda: [
        "silhouette", "calinski_harabasz", "davies_bouldin"
    ])
    
    # Evaluation settings
    test_size: float = 0.2
    random_state: int = 42
    stratified_split: bool = True
    
    # Statistical testing
    statistical_tests: bool = True
    confidence_level: float = 0.95
    
    # Model comparison
    compare_models: bool = True
    baseline_model: str = "dummy"  # dummy, random, constant


@dataclass
class InterpretabilityConfig:
    """Configuration for model interpretability."""
    
    # SHAP analysis
    shap_analysis: bool = True
    shap_background_samples: int = 100
    shap_explainer_type: str = "auto"  # auto, tree, kernel, linear
    
    # Feature importance
    feature_importance: bool = True
    importance_method: str = "auto"  # auto, permutation, builtin
    
    # Partial dependence plots
    partial_dependence: bool = True
    pd_features: Optional[List[str]] = None  # None for auto-selection
    pd_grid_resolution: int = 50
    
    # LIME explanations
    lime_explanations: bool = False
    lime_samples: int = 1000
    lime_features: int = 10


@dataclass
class ReportingConfig:
    """Configuration for report generation."""
    
    # Report content
    include_performance_plots: bool = True
    include_feature_importance: bool = True
    include_model_comparison: bool = True
    include_interpretability: bool = True
    include_data_analysis: bool = True
    
    # Visualization settings
    plot_style: str = "seaborn"  # seaborn, matplotlib, plotly
    figure_size: tuple = (12, 8)
    dpi: int = 100
    
    # Export options
    export_format: str = "html"  # html, pdf, notebook
    include_code: bool = False
    include_timestamps: bool = True
    
    # Customization
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None
    logo_path: Optional[str] = None


@dataclass
class AutoMLConfig:
    """Complete AutoML Lite configuration."""
    
    # Basic settings
    problem_type: Optional[ProblemType] = None
    random_state: int = 42
    n_jobs: int = -1
    verbose: bool = True
    
    # Time and resource constraints
    time_budget: int = 600  # seconds
    memory_limit: Optional[str] = None  # e.g., "4GB"
    max_models: int = 10
    
    # Component configurations
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # Advanced settings
    enable_parallel: bool = True
    enable_caching: bool = True
    cache_dir: Optional[str] = None
    
    # Validation
    validate_config: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.validate_config:
            self._validate()
    
    def _validate(self):
        """Validate the configuration."""
        errors = []
        
        # Validate time budget
        if self.time_budget <= 0:
            errors.append("time_budget must be positive")
        
        # Validate max_models
        if self.max_models <= 0:
            errors.append("max_models must be positive")
        
        # Validate test_size
        if not 0 < self.evaluation.test_size < 1:
            errors.append("test_size must be between 0 and 1")
        
        # Validate confidence level
        if not 0 < self.evaluation.confidence_level < 1:
            errors.append("confidence_level must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = asdict(self)
        
        # Convert enums to strings
        if self.problem_type:
            config_dict['problem_type'] = self.problem_type.value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoMLConfig':
        """Create configuration from dictionary."""
        # Convert string to enum
        if 'problem_type' in config_dict and config_dict['problem_type']:
            config_dict['problem_type'] = ProblemType(config_dict['problem_type'])
        
        # Create nested configs
        if 'preprocessing' in config_dict:
            config_dict['preprocessing'] = PreprocessingConfig(**config_dict['preprocessing'])
        
        if 'model' in config_dict:
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'evaluation' in config_dict:
            config_dict['evaluation'] = EvaluationConfig(**config_dict['evaluation'])
        
        if 'interpretability' in config_dict:
            config_dict['interpretability'] = InterpretabilityConfig(**config_dict['interpretability'])
        
        if 'reporting' in config_dict:
            config_dict['reporting'] = ReportingConfig(**config_dict['reporting'])
        
        return cls(**config_dict)


class ConfigManager:
    """Manager for AutoML Lite configurations."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory to store configurations (optional)
        """
        self.config_dir = Path(config_dir) if config_dir else Path.home() / ".automl_lite" / "configs"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Predefined configurations
        self._preset_configs = self._create_preset_configs()
    
    def _create_preset_configs(self) -> Dict[str, AutoMLConfig]:
        """Create preset configurations for common use cases."""
        configs = {}
        
        # Quick configuration
        quick_config = AutoMLConfig(
            time_budget=300,
            max_models=5,
            model=ModelConfig(max_trials=20, timeout=60),
            evaluation=EvaluationConfig(test_size=0.3),
            interpretability=InterpretabilityConfig(shap_analysis=False),
            reporting=ReportingConfig(include_interpretability=False)
        )
        configs['quick'] = quick_config
        
        # Comprehensive configuration
        comprehensive_config = AutoMLConfig(
            time_budget=1800,
            max_models=15,
            model=ModelConfig(max_trials=200, timeout=900),
            evaluation=EvaluationConfig(statistical_tests=True),
            interpretability=InterpretabilityConfig(
                shap_analysis=True,
                lime_explanations=True
            ),
            reporting=ReportingConfig(include_code=True)
        )
        configs['comprehensive'] = comprehensive_config
        
        # Production configuration
        production_config = AutoMLConfig(
            time_budget=3600,
            max_models=20,
            model=ModelConfig(
                max_trials=500,
                timeout=1800,
                enable_ensemble=True,
                ensemble_method="stacking"
            ),
            evaluation=EvaluationConfig(
                test_size=0.2,
                statistical_tests=True,
                compare_models=True
            ),
            interpretability=InterpretabilityConfig(
                shap_analysis=True,
                feature_importance=True,
                partial_dependence=True
            ),
            reporting=ReportingConfig(
                export_format="html",
                include_timestamps=True
            ),
            enable_caching=True
        )
        configs['production'] = production_config
        
        # Research configuration
        research_config = AutoMLConfig(
            time_budget=7200,
            max_models=30,
            model=ModelConfig(
                max_trials=1000,
                timeout=3600,
                enable_ensemble=True,
                ensemble_method="blending"
            ),
            evaluation=EvaluationConfig(
                statistical_tests=True,
                compare_models=True,
                baseline_model="dummy"
            ),
            interpretability=InterpretabilityConfig(
                shap_analysis=True,
                lime_explanations=True,
                partial_dependence=True
            ),
            reporting=ReportingConfig(
                include_code=True,
                include_timestamps=True,
                export_format="html"
            ),
            enable_caching=True
        )
        configs['research'] = research_config
        
        # Classification-specific configuration
        classification_config = AutoMLConfig(
            problem_type=ProblemType.CLASSIFICATION,
            time_budget=600,
            max_models=10,
            model=ModelConfig(
                models_to_try=[
                    "random_forest", "gradient_boosting", "svm", 
                    "logistic_regression", "neural_network"
                ],
                enable_ensemble=True,
                ensemble_method="voting"
            ),
            evaluation=EvaluationConfig(
                classification_metrics=[
                    "accuracy", "precision", "recall", "f1", "roc_auc"
                ]
            )
        )
        configs['classification'] = classification_config
        
        # Regression-specific configuration
        regression_config = AutoMLConfig(
            problem_type=ProblemType.REGRESSION,
            time_budget=600,
            max_models=10,
            model=ModelConfig(
                models_to_try=[
                    "random_forest", "gradient_boosting", "svm", 
                    "linear_regression", "neural_network"
                ],
                enable_ensemble=True,
                ensemble_method="voting"
            ),
            evaluation=EvaluationConfig(
                regression_metrics=[
                    "mse", "mae", "r2", "rmse"
                ]
            )
        )
        configs['regression'] = regression_config
        
        return configs
    
    def get_preset_config(self, name: str) -> AutoMLConfig:
        """
        Get a preset configuration.
        
        Args:
            name: Name of the preset configuration
            
        Returns:
            AutoMLConfig instance
        """
        if name not in self._preset_configs:
            available = list(self._preset_configs.keys())
            raise ValueError(f"Unknown preset '{name}'. Available: {available}")
        
        return self._preset_configs[name]
    
    def list_preset_configs(self) -> List[str]:
        """List available preset configurations."""
        return list(self._preset_configs.keys())
    
    def save_config(self, config: AutoMLConfig, name: str) -> None:
        """
        Save a configuration to file.
        
        Args:
            config: Configuration to save
            name: Name for the configuration
        """
        config_path = self.config_dir / f"{name}.json"
        
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration saved to {config_path}")
    
    def load_config(self, name: str) -> AutoMLConfig:
        """
        Load a configuration from file.
        
        Args:
            name: Name of the configuration file (without extension)
            
        Returns:
            AutoMLConfig instance
        """
        config_path = self.config_dir / f"{name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        return AutoMLConfig.from_dict(config_dict)
    
    def list_saved_configs(self) -> List[str]:
        """List saved configuration files."""
        config_files = list(self.config_dir.glob("*.json"))
        return [f.stem for f in config_files]
    
    def delete_config(self, name: str) -> None:
        """
        Delete a saved configuration.
        
        Args:
            name: Name of the configuration to delete
        """
        config_path = self.config_dir / f"{name}.json"
        
        if config_path.exists():
            config_path.unlink()
            logger.info(f"Configuration deleted: {config_path}")
        else:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    def create_custom_config(
        self,
        base_config: Optional[str] = None,
        **overrides
    ) -> AutoMLConfig:
        """
        Create a custom configuration based on a preset or default.
        
        Args:
            base_config: Name of base preset configuration (optional)
            **overrides: Configuration overrides
            
        Returns:
            AutoMLConfig instance
        """
        if base_config:
            config = self.get_preset_config(base_config)
        else:
            config = AutoMLConfig()
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Try nested attributes
                parts = key.split('.')
                obj = config
                for part in parts[:-1]:
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        raise ValueError(f"Invalid configuration key: {key}")
                
                if hasattr(obj, parts[-1]):
                    setattr(obj, parts[-1], value)
                else:
                    raise ValueError(f"Invalid configuration key: {key}")
        
        return config
    
    def validate_config(self, config: AutoMLConfig) -> List[str]:
        """
        Validate a configuration and return any issues.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation issues (empty if valid)
        """
        issues = []
        
        try:
            config._validate()
        except ValueError as e:
            issues.append(str(e))
        
        # Additional validation
        if config.time_budget < config.model.timeout:
            issues.append("time_budget should be greater than model.timeout")
        
        if config.max_models < config.model.top_k_models:
            issues.append("max_models should be greater than or equal to model.top_k_models")
        
        return issues
    
    def export_config_template(self, output_path: str) -> None:
        """
        Export a configuration template.
        
        Args:
            output_path: Path to save the template
        """
        template_config = AutoMLConfig()
        
        with open(output_path, 'w') as f:
            json.dump(template_config.to_dict(), f, indent=2)
        
        logger.info(f"Configuration template exported to {output_path}")
    
    def merge_configs(self, configs: List[AutoMLConfig]) -> AutoMLConfig:
        """
        Merge multiple configurations (later configs override earlier ones).
        
        Args:
            configs: List of configurations to merge
            
        Returns:
            Merged AutoMLConfig instance
        """
        if not configs:
            raise ValueError("At least one configuration must be provided")
        
        merged_dict = configs[0].to_dict()
        
        for config in configs[1:]:
            config_dict = config.to_dict()
            self._deep_merge(merged_dict, config_dict)
        
        return AutoMLConfig.from_dict(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> None:
        """Deep merge two dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value 
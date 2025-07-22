"""
Advanced Configuration Management for AutoML Lite.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import jsonschema
from copy import deepcopy

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ProblemType(Enum):
    """Problem types supported by AutoML Lite."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    MULTI_LABEL = "multi_label"


class EnsembleMethod(Enum):
    """Ensemble methods supported by AutoML Lite."""
    VOTING = "voting"
    STACKING = "stacking"
    BAGGING = "bagging"


@dataclass
class ModelConfig:
    """Configuration for individual models."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    max_trials: int = 50
    timeout: Optional[int] = None


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering."""
    enable_polynomial_features: bool = True
    enable_interaction_features: bool = True
    enable_temporal_features: bool = True
    enable_statistical_features: bool = True
    enable_domain_features: bool = True
    max_polynomial_degree: int = 2
    max_feature_combinations: int = 100
    n_best_features: Optional[int] = None
    correlation_threshold: float = 0.95


@dataclass
class InterpretabilityConfig:
    """Configuration for model interpretability."""
    enable_shap: bool = True
    enable_lime: bool = True
    enable_permutation: bool = True
    enable_partial_dependence: bool = True
    enable_feature_effects: bool = True
    n_shap_samples: int = 100
    n_lime_samples: int = 1000
    n_permutation_repeats: int = 10


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    enable_tracking: bool = False
    tracking_backend: str = "mlflow"  # mlflow, wandb, tensorboard
    experiment_name: Optional[str] = None
    log_artifacts: bool = True
    log_metrics: bool = True
    log_params: bool = True


@dataclass
class AutoMLConfig:
    """Main AutoML configuration."""
    # Basic settings
    problem_type: ProblemType = ProblemType.CLASSIFICATION
    time_budget: int = 300
    max_models: int = 10
    cv_folds: int = 5
    random_state: int = 42
    verbose: bool = True
    
    # Advanced features
    enable_ensemble: bool = True
    enable_early_stopping: bool = True
    enable_feature_selection: bool = True
    enable_interpretability: bool = True
    enable_auto_feature_engineering: bool = True
    enable_deep_learning: bool = False
    enable_time_series: bool = False
    enable_experiment_tracking: bool = False
    
    # Ensemble settings
    ensemble_method: EnsembleMethod = EnsembleMethod.VOTING
    top_k_models: int = 3
    early_stopping_patience: int = 10
    
    # Model configurations
    models: List[ModelConfig] = field(default_factory=list)
    
    # Feature engineering
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    
    # Interpretability
    interpretability: InterpretabilityConfig = field(default_factory=InterpretabilityConfig)
    
    # Experiment tracking
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    # Advanced settings
    n_jobs: int = -1
    memory_limit: Optional[str] = None
    enable_parallel: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = asdict(self)
        
        # Convert enums to strings
        config_dict['problem_type'] = self.problem_type.value
        config_dict['ensemble_method'] = self.ensemble_method.value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AutoMLConfig':
        """Create config from dictionary."""
        # Convert string enums back to enum objects
        if 'problem_type' in config_dict:
            config_dict['problem_type'] = ProblemType(config_dict['problem_type'])
        if 'ensemble_method' in config_dict:
            config_dict['ensemble_method'] = EnsembleMethod(config_dict['ensemble_method'])
        
        # Convert model configs
        if 'models' in config_dict:
            models = []
            for model_dict in config_dict['models']:
                models.append(ModelConfig(**model_dict))
            config_dict['models'] = models
        
        # Convert nested configs
        if 'feature_engineering' in config_dict:
            config_dict['feature_engineering'] = FeatureEngineeringConfig(**config_dict['feature_engineering'])
        
        if 'interpretability' in config_dict:
            config_dict['interpretability'] = InterpretabilityConfig(**config_dict['interpretability'])
        
        if 'experiment' in config_dict:
            config_dict['experiment'] = ExperimentConfig(**config_dict['experiment'])
        
        return cls(**config_dict)


class ConfigManager:
    """
    Advanced configuration manager for AutoML Lite.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: Optional[AutoMLConfig] = None
        self.templates_dir = Path(__file__).parent / "templates"
        
        # Create templates directory if it doesn't exist
        self.templates_dir.mkdir(exist_ok=True)
        
        # Load default templates
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default configuration templates."""
        default_templates = {
            'basic': self._get_basic_template(),
            'advanced': self._get_advanced_template(),
            'production': self._get_production_template(),
            'research': self._get_research_template(),
            'customer_churn': self._get_customer_churn_template(),
            'fraud_detection': self._get_fraud_detection_template(),
            'house_price': self._get_house_price_template()
        }
        
        for name, template in default_templates.items():
            template_path = self.templates_dir / f"{name}.yaml"
            if not template_path.exists():
                self.save_config(template, template_path)
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> AutoMLConfig:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            AutoML configuration
        """
        config_path = Path(config_path) if config_path else self.config_path
        
        if not config_path or not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"Loading configuration from {config_path}")
        
        # Determine file type and load
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        # Validate configuration
        self._validate_config(config_dict)
        
        # Create config object
        self.config = AutoMLConfig.from_dict(config_dict)
        
        logger.info("Configuration loaded successfully")
        return self.config
    
    def save_config(self, config: AutoMLConfig, config_path: Optional[Union[str, Path]] = None):
        """
        Save configuration to file.
        
        Args:
            config: AutoML configuration
            config_path: Path to save configuration
        """
        config_path = Path(config_path) if config_path else self.config_path
        
        if not config_path:
            raise ValueError("No configuration path specified")
        
        logger.info(f"Saving configuration to {config_path}")
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Save based on file extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info("Configuration saved successfully")
    
    def get_template(self, template_name: str) -> AutoMLConfig:
        """
        Get configuration template.
        
        Args:
            template_name: Name of the template
            
        Returns:
            AutoML configuration template
        """
        template_path = self.templates_dir / f"{template_name}.yaml"
        
        if not template_path.exists():
            raise ValueError(f"Template not found: {template_name}")
        
        return self.load_config(template_path)
    
    def list_templates(self) -> List[str]:
        """List available templates."""
        templates = []
        for template_file in self.templates_dir.glob("*.yaml"):
            templates.append(template_file.stem)
        return sorted(templates)
    
    def create_template(self, name: str, config: AutoMLConfig):
        """
        Create a new template.
        
        Args:
            name: Template name
            config: Configuration to save as template
        """
        template_path = self.templates_dir / f"{name}.yaml"
        self.save_config(config, template_path)
        logger.info(f"Template '{name}' created successfully")
    
    def _validate_config(self, config_dict: Dict[str, Any]):
        """Validate configuration dictionary."""
        # Basic validation schema
        schema = {
            "type": "object",
            "properties": {
                "problem_type": {"type": "string", "enum": ["classification", "regression", "time_series", "multi_label"]},
                "time_budget": {"type": "integer", "minimum": 1},
                "max_models": {"type": "integer", "minimum": 1},
                "cv_folds": {"type": "integer", "minimum": 2},
                "random_state": {"type": "integer"},
                "verbose": {"type": "boolean"}
            },
            "required": ["problem_type", "time_budget", "max_models"]
        }
        
        try:
            jsonschema.validate(instance=config_dict, schema=schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {str(e)}")
    
    def _get_basic_template(self) -> AutoMLConfig:
        """Get basic configuration template."""
        return AutoMLConfig(
            problem_type=ProblemType.CLASSIFICATION,
            time_budget=300,
            max_models=5,
            cv_folds=3,
            enable_ensemble=False,
            enable_interpretability=False,
            enable_auto_feature_engineering=False
        )
    
    def _get_advanced_template(self) -> AutoMLConfig:
        """Get advanced configuration template."""
        return AutoMLConfig(
            problem_type=ProblemType.CLASSIFICATION,
            time_budget=600,
            max_models=10,
            cv_folds=5,
            enable_ensemble=True,
            enable_interpretability=True,
            enable_auto_feature_engineering=True,
            ensemble_method=EnsembleMethod.VOTING,
            top_k_models=3
        )
    
    def _get_production_template(self) -> AutoMLConfig:
        """Get production configuration template."""
        return AutoMLConfig(
            problem_type=ProblemType.CLASSIFICATION,
            time_budget=1800,
            max_models=15,
            cv_folds=5,
            enable_ensemble=True,
            enable_interpretability=True,
            enable_auto_feature_engineering=True,
            enable_early_stopping=True,
            ensemble_method=EnsembleMethod.STACKING,
            top_k_models=5,
            experiment=ExperimentConfig(
                enable_tracking=True,
                tracking_backend="mlflow",
                log_artifacts=True,
                log_metrics=True
            )
        )
    
    def _get_research_template(self) -> AutoMLConfig:
        """Get research configuration template."""
        return AutoMLConfig(
            problem_type=ProblemType.CLASSIFICATION,
            time_budget=3600,
            max_models=20,
            cv_folds=10,
            enable_ensemble=True,
            enable_interpretability=True,
            enable_auto_feature_engineering=True,
            ensemble_method=EnsembleMethod.STACKING,
            top_k_models=7,
            interpretability=InterpretabilityConfig(
                enable_shap=True,
                enable_lime=True,
                enable_permutation=True,
                enable_partial_dependence=True,
                n_shap_samples=500,
                n_lime_samples=2000
            )
        )
    
    def _get_customer_churn_template(self) -> AutoMLConfig:
        """Get customer churn specific template."""
        config = self._get_production_template()
        config.problem_type = ProblemType.CLASSIFICATION
        config.feature_engineering = FeatureEngineeringConfig(
            enable_polynomial_features=True,
            enable_interaction_features=True,
            enable_temporal_features=True,
            enable_statistical_features=True,
            enable_domain_features=True,
            max_polynomial_degree=2,
            max_feature_combinations=50,
            n_best_features=20
        )
        return config
    
    def _get_fraud_detection_template(self) -> AutoMLConfig:
        """Get fraud detection specific template."""
        config = self._get_production_template()
        config.problem_type = ProblemType.CLASSIFICATION
        config.interpretability = InterpretabilityConfig(
            enable_shap=True,
            enable_lime=True,
            enable_permutation=True,
            enable_partial_dependence=True,
            enable_feature_effects=True,
            n_shap_samples=200,
            n_lime_samples=1000,
            n_permutation_repeats=20
        )
        return config
    
    def _get_house_price_template(self) -> AutoMLConfig:
        """Get house price prediction template."""
        config = self._get_production_template()
        config.problem_type = ProblemType.REGRESSION
        config.feature_engineering = FeatureEngineeringConfig(
            enable_polynomial_features=True,
            enable_interaction_features=True,
            enable_temporal_features=False,
            enable_statistical_features=True,
            enable_domain_features=True,
            max_polynomial_degree=3,
            max_feature_combinations=100,
            n_best_features=30
        )
        return config
    
    def merge_configs(self, base_config: AutoMLConfig, override_config: Dict[str, Any]) -> AutoMLConfig:
        """
        Merge configurations with override.
        
        Args:
            base_config: Base configuration
            override_config: Override configuration dictionary
            
        Returns:
            Merged configuration
        """
        # Convert base config to dict
        merged_dict = base_config.to_dict()
        
        # Recursively merge
        self._recursive_merge(merged_dict, override_config)
        
        # Convert back to config object
        return AutoMLConfig.from_dict(merged_dict)
    
    def _recursive_merge(self, base_dict: Dict[str, Any], override_dict: Dict[str, Any]):
        """Recursively merge dictionaries."""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._recursive_merge(base_dict[key], value)
            else:
                base_dict[key] = value 
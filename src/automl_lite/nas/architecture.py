"""
Core data structures for Neural Architecture Search.

This module defines the fundamental data models for representing neural network
architectures, configurations, and search results.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np


@dataclass
class LayerConfig:
    """
    Configuration for a single neural network layer.
    
    Attributes:
        layer_type: Type of layer (e.g., 'dense', 'conv2d', 'lstm', 'dropout')
        params: Layer-specific parameters (e.g., units, filters, kernel_size)
        input_shape: Expected input shape (optional, inferred during validation)
        output_shape: Expected output shape (optional, inferred during validation)
    """
    layer_type: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert LayerConfig to dictionary for serialization."""
        return {
            'layer_type': self.layer_type,
            'params': self.params,
            'input_shape': list(self.input_shape) if self.input_shape else None,
            'output_shape': list(self.output_shape) if self.output_shape else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LayerConfig':
        """Create LayerConfig from dictionary."""
        return cls(
            layer_type=data['layer_type'],
            params=data.get('params', {}),
            input_shape=tuple(data['input_shape']) if data.get('input_shape') else None,
            output_shape=tuple(data['output_shape']) if data.get('output_shape') else None,
        )
    
    def __repr__(self) -> str:
        """String representation of LayerConfig."""
        params_str = ', '.join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.layer_type}({params_str})"


@dataclass
class Architecture:
    """
    Representation of a neural network architecture.
    
    Attributes:
        layers: List of layer configurations
        connections: List of (from_layer_idx, to_layer_idx) tuples for skip connections
        global_config: Global configuration (optimizer, learning_rate, etc.)
        metadata: Additional metadata (performance metrics, hardware metrics, etc.)
        id: Unique identifier for the architecture
    """
    layers: List[LayerConfig]
    connections: List[Tuple[int, int]] = field(default_factory=list)
    global_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate architecture after initialization."""
        if not self.layers:
            raise ValueError("Architecture must have at least one layer")
        
        # Ensure connections reference valid layer indices
        max_idx = len(self.layers) - 1
        for from_idx, to_idx in self.connections:
            if from_idx < 0 or from_idx > max_idx:
                raise ValueError(f"Invalid connection: from_idx {from_idx} out of range")
            if to_idx < 0 or to_idx > max_idx:
                raise ValueError(f"Invalid connection: to_idx {to_idx} out of range")
            if from_idx >= to_idx:
                raise ValueError(f"Invalid connection: from_idx {from_idx} must be < to_idx {to_idx}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Architecture to dictionary for serialization."""
        return {
            'id': self.id,
            'layers': [layer.to_dict() for layer in self.layers],
            'connections': [[from_idx, to_idx] for from_idx, to_idx in self.connections],
            'global_config': self.global_config,
            'metadata': self.metadata,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert Architecture to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Architecture':
        """Create Architecture from dictionary."""
        return cls(
            layers=[LayerConfig.from_dict(layer_data) for layer_data in data['layers']],
            connections=[tuple(conn) for conn in data.get('connections', [])],
            global_config=data.get('global_config', {}),
            metadata=data.get('metadata', {}),
            id=data.get('id', str(uuid.uuid4())),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Architecture':
        """Create Architecture from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def get_num_layers(self) -> int:
        """Get the number of layers in the architecture."""
        return len(self.layers)
    
    def get_num_parameters(self) -> Optional[int]:
        """Get total number of parameters if available in metadata."""
        return self.metadata.get('num_parameters')
    
    def get_performance_metric(self, metric_name: str) -> Optional[float]:
        """Get a specific performance metric from metadata."""
        metrics = self.metadata.get('performance_metrics', {})
        return metrics.get(metric_name)
    
    def set_performance_metric(self, metric_name: str, value: float):
        """Set a performance metric in metadata."""
        if 'performance_metrics' not in self.metadata:
            self.metadata['performance_metrics'] = {}
        self.metadata['performance_metrics'][metric_name] = value
    
    def get_hardware_metric(self, metric_name: str) -> Optional[float]:
        """Get a specific hardware metric from metadata."""
        metrics = self.metadata.get('hardware_metrics', {})
        return metrics.get(metric_name)
    
    def set_hardware_metric(self, metric_name: str, value: float):
        """Set a hardware metric in metadata."""
        if 'hardware_metrics' not in self.metadata:
            self.metadata['hardware_metrics'] = {}
        self.metadata['hardware_metrics'][metric_name] = value
    
    def clone(self) -> 'Architecture':
        """Create a deep copy of the architecture with a new ID."""
        import copy
        new_arch = Architecture(
            layers=copy.deepcopy(self.layers),
            connections=copy.deepcopy(self.connections),
            global_config=copy.deepcopy(self.global_config),
            metadata=copy.deepcopy(self.metadata),
        )
        return new_arch
    
    def __repr__(self) -> str:
        """String representation of Architecture."""
        layers_str = ' -> '.join(str(layer) for layer in self.layers)
        skip_str = f", {len(self.connections)} skip connections" if self.connections else ""
        return f"Architecture(id={self.id[:8]}..., {len(self.layers)} layers{skip_str})"


@dataclass
class NASConfig:
    """
    Configuration for Neural Architecture Search.
    
    This dataclass contains all parameters needed to configure the NAS process,
    including search strategy, performance estimation, hardware constraints,
    and multi-objective optimization settings.
    """
    # Search configuration
    search_strategy: str = 'evolutionary'  # 'rl', 'evolutionary', 'darts'
    search_space_type: str = 'auto'  # 'auto', 'tabular', 'vision', 'timeseries'
    time_budget: int = 3600  # seconds
    max_architectures: int = 100
    
    # Search strategy specific parameters
    rl_controller_hidden_size: int = 100
    rl_baseline_decay: float = 0.95
    evolution_population_size: int = 50
    evolution_mutation_rate: float = 0.2
    darts_supernet_epochs: int = 50
    
    # Performance estimation
    performance_estimator: str = 'early_stopping'  # 'early_stopping', 'weight_sharing', 'learning_curve'
    estimation_budget_fraction: float = 0.1
    early_stopping_patience: int = 5
    
    # Hardware constraints
    enable_hardware_aware: bool = False
    target_hardware: str = 'cpu'  # 'cpu', 'gpu', 'mobile', 'edge'
    max_latency_ms: Optional[float] = None
    max_memory_mb: Optional[float] = None
    max_model_size_mb: Optional[float] = None
    
    # Multi-objective optimization
    enable_multi_objective: bool = True
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'latency', 'model_size'])
    objective_weights: Optional[Dict[str, float]] = None
    
    # Transfer learning
    enable_transfer_learning: bool = True
    architecture_repository_path: str = '~/.automl_lite/nas_architectures.db'
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 10  # Save every N architectures
    checkpoint_path: str = './nas_checkpoint.pkl'
    
    # Logging
    verbose: bool = True
    log_all_architectures: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        # Validate search strategy
        valid_strategies = ['rl', 'evolutionary', 'darts']
        if self.search_strategy not in valid_strategies:
            raise ValueError(f"search_strategy must be one of {valid_strategies}, got '{self.search_strategy}'")
        
        # Validate search space type
        valid_spaces = ['auto', 'tabular', 'vision', 'timeseries']
        if self.search_space_type not in valid_spaces:
            raise ValueError(f"search_space_type must be one of {valid_spaces}, got '{self.search_space_type}'")
        
        # Validate time budget
        if self.time_budget <= 0:
            raise ValueError(f"time_budget must be positive, got {self.time_budget}")
        
        # Validate max architectures
        if self.max_architectures <= 0:
            raise ValueError(f"max_architectures must be positive, got {self.max_architectures}")
        
        # Validate performance estimator
        valid_estimators = ['early_stopping', 'weight_sharing', 'learning_curve']
        if self.performance_estimator not in valid_estimators:
            raise ValueError(f"performance_estimator must be one of {valid_estimators}, got '{self.performance_estimator}'")
        
        # Validate estimation budget fraction
        if not 0 < self.estimation_budget_fraction <= 1:
            raise ValueError(f"estimation_budget_fraction must be in (0, 1], got {self.estimation_budget_fraction}")
        
        # Validate target hardware
        valid_hardware = ['cpu', 'gpu', 'mobile', 'edge']
        if self.target_hardware not in valid_hardware:
            raise ValueError(f"target_hardware must be one of {valid_hardware}, got '{self.target_hardware}'")
        
        # Validate hardware constraints
        if self.max_latency_ms is not None and self.max_latency_ms <= 0:
            raise ValueError(f"max_latency_ms must be positive, got {self.max_latency_ms}")
        
        if self.max_memory_mb is not None and self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        
        if self.max_model_size_mb is not None and self.max_model_size_mb <= 0:
            raise ValueError(f"max_model_size_mb must be positive, got {self.max_model_size_mb}")
        
        # Validate objectives
        valid_objectives = ['accuracy', 'latency', 'model_size', 'memory', 'energy']
        for obj in self.objectives:
            if obj not in valid_objectives:
                raise ValueError(f"Invalid objective '{obj}'. Must be one of {valid_objectives}")
        
        # Validate objective weights
        if self.objective_weights is not None:
            for obj in self.objective_weights:
                if obj not in self.objectives:
                    raise ValueError(f"Objective weight specified for '{obj}' but it's not in objectives list")
            
            # Check weights are positive
            for obj, weight in self.objective_weights.items():
                if weight < 0:
                    raise ValueError(f"Objective weight for '{obj}' must be non-negative, got {weight}")
        
        # Validate checkpoint frequency
        if self.checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {self.checkpoint_frequency}")
        
        # Validate strategy-specific parameters
        if self.rl_controller_hidden_size <= 0:
            raise ValueError(f"rl_controller_hidden_size must be positive, got {self.rl_controller_hidden_size}")
        
        if not 0 < self.rl_baseline_decay < 1:
            raise ValueError(f"rl_baseline_decay must be in (0, 1), got {self.rl_baseline_decay}")
        
        if self.evolution_population_size <= 0:
            raise ValueError(f"evolution_population_size must be positive, got {self.evolution_population_size}")
        
        if not 0 <= self.evolution_mutation_rate <= 1:
            raise ValueError(f"evolution_mutation_rate must be in [0, 1], got {self.evolution_mutation_rate}")
        
        if self.darts_supernet_epochs <= 0:
            raise ValueError(f"darts_supernet_epochs must be positive, got {self.darts_supernet_epochs}")
        
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NASConfig to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert NASConfig to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NASConfig':
        """Create NASConfig from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NASConfig':
        """Create NASConfig from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class NASResult:
    """
    Results from a Neural Architecture Search.
    
    Contains the best architecture found, Pareto front for multi-objective
    optimization, search history, and various statistics.
    """
    best_architecture: Architecture
    pareto_front: List[Architecture] = field(default_factory=list)
    all_architectures: List[Architecture] = field(default_factory=list)
    search_history: List[Dict[str, Any]] = field(default_factory=list)
    search_time: float = 0.0
    total_architectures_evaluated: int = 0
    
    # Statistics
    best_accuracy: float = 0.0
    best_latency: Optional[float] = None
    best_model_size: Optional[float] = None
    
    # Metadata
    search_strategy: str = ''
    search_space_type: str = ''
    dataset_metadata: Dict[str, Any] = field(default_factory=dict)
    config: Optional[NASConfig] = None
    
    def get_top_k_architectures(self, k: int = 5, metric: str = 'accuracy') -> List[Architecture]:
        """
        Get top k architectures sorted by a specific metric.
        
        Args:
            k: Number of top architectures to return
            metric: Metric to sort by (default: 'accuracy')
        
        Returns:
            List of top k architectures
        """
        if not self.all_architectures:
            return []
        
        # Sort architectures by metric (descending for accuracy, ascending for latency/size)
        reverse = metric == 'accuracy'
        sorted_archs = sorted(
            self.all_architectures,
            key=lambda arch: arch.get_performance_metric(metric) or arch.get_hardware_metric(metric) or float('inf'),
            reverse=reverse
        )
        
        return sorted_archs[:k]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the search results."""
        return {
            'search_strategy': self.search_strategy,
            'search_space_type': self.search_space_type,
            'total_architectures_evaluated': self.total_architectures_evaluated,
            'search_time_seconds': self.search_time,
            'best_accuracy': self.best_accuracy,
            'best_latency_ms': self.best_latency,
            'best_model_size_mb': self.best_model_size,
            'pareto_front_size': len(self.pareto_front),
            'best_architecture_id': self.best_architecture.id,
            'best_architecture_layers': self.best_architecture.get_num_layers(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert NASResult to dictionary for serialization."""
        return {
            'best_architecture': self.best_architecture.to_dict(),
            'pareto_front': [arch.to_dict() for arch in self.pareto_front],
            'all_architectures': [arch.to_dict() for arch in self.all_architectures],
            'search_history': self.search_history,
            'search_time': self.search_time,
            'total_architectures_evaluated': self.total_architectures_evaluated,
            'best_accuracy': self.best_accuracy,
            'best_latency': self.best_latency,
            'best_model_size': self.best_model_size,
            'search_strategy': self.search_strategy,
            'search_space_type': self.search_space_type,
            'dataset_metadata': self.dataset_metadata,
            'config': self.config.to_dict() if self.config else None,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert NASResult to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NASResult':
        """Create NASResult from dictionary."""
        return cls(
            best_architecture=Architecture.from_dict(data['best_architecture']),
            pareto_front=[Architecture.from_dict(arch_data) for arch_data in data.get('pareto_front', [])],
            all_architectures=[Architecture.from_dict(arch_data) for arch_data in data.get('all_architectures', [])],
            search_history=data.get('search_history', []),
            search_time=data.get('search_time', 0.0),
            total_architectures_evaluated=data.get('total_architectures_evaluated', 0),
            best_accuracy=data.get('best_accuracy', 0.0),
            best_latency=data.get('best_latency'),
            best_model_size=data.get('best_model_size'),
            search_strategy=data.get('search_strategy', ''),
            search_space_type=data.get('search_space_type', ''),
            dataset_metadata=data.get('dataset_metadata', {}),
            config=NASConfig.from_dict(data['config']) if data.get('config') else None,
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'NASResult':
        """Create NASResult from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __repr__(self) -> str:
        """String representation of NASResult."""
        return (f"NASResult(strategy={self.search_strategy}, "
                f"evaluated={self.total_architectures_evaluated}, "
                f"best_accuracy={self.best_accuracy:.4f}, "
                f"time={self.search_time:.1f}s)")

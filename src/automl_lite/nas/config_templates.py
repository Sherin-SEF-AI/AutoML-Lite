"""
Configuration templates for Neural Architecture Search.

This module provides pre-configured NASConfig templates for common use cases,
making it easy to get started with NAS for different scenarios.
"""

from .architecture import NASConfig


def get_quick_start_config(time_budget: int = 1800) -> NASConfig:
    """
    Get a quick start configuration for NAS.
    
    This configuration uses evolutionary search with moderate settings,
    suitable for getting started quickly with NAS on most problems.
    
    Args:
        time_budget: Time budget in seconds (default: 1800 = 30 minutes)
    
    Returns:
        NASConfig configured for quick start
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=50,
        
        # Evolutionary strategy parameters
        evolution_population_size=20,
        evolution_mutation_rate=0.2,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.15,
        early_stopping_patience=5,
        
        # Hardware constraints (disabled for quick start)
        enable_hardware_aware=False,
        
        # Multi-objective optimization (simplified)
        enable_multi_objective=False,
        objectives=['accuracy'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_mobile_deployment_config(
    max_latency_ms: float = 100.0,
    max_model_size_mb: float = 10.0,
    time_budget: int = 3600
) -> NASConfig:
    """
    Get a configuration for hardware-aware NAS targeting mobile deployment.
    
    This configuration optimizes for mobile devices with strict latency
    and model size constraints.
    
    Args:
        max_latency_ms: Maximum inference latency in milliseconds (default: 100)
        max_model_size_mb: Maximum model size in MB (default: 10)
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for mobile deployment
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=100,
        
        # Evolutionary strategy parameters
        evolution_population_size=30,
        evolution_mutation_rate=0.25,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.1,
        early_stopping_patience=5,
        
        # Hardware constraints (mobile-specific)
        enable_hardware_aware=True,
        target_hardware='mobile',
        max_latency_ms=max_latency_ms,
        max_memory_mb=512.0,  # Typical mobile memory constraint
        max_model_size_mb=max_model_size_mb,
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        objective_weights={
            'accuracy': 0.5,
            'latency': 0.3,
            'model_size': 0.2,
        },
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_edge_deployment_config(
    max_latency_ms: float = 50.0,
    max_memory_mb: float = 256.0,
    time_budget: int = 3600
) -> NASConfig:
    """
    Get a configuration for hardware-aware NAS targeting edge device deployment.
    
    This configuration optimizes for edge devices with very strict resource
    constraints (e.g., IoT devices, embedded systems).
    
    Args:
        max_latency_ms: Maximum inference latency in milliseconds (default: 50)
        max_memory_mb: Maximum memory usage in MB (default: 256)
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for edge deployment
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=100,
        
        # Evolutionary strategy parameters
        evolution_population_size=25,
        evolution_mutation_rate=0.3,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.1,
        early_stopping_patience=3,
        
        # Hardware constraints (edge-specific)
        enable_hardware_aware=True,
        target_hardware='edge',
        max_latency_ms=max_latency_ms,
        max_memory_mb=max_memory_mb,
        max_model_size_mb=5.0,  # Very small models for edge
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'memory', 'model_size'],
        objective_weights={
            'accuracy': 0.4,
            'latency': 0.3,
            'memory': 0.2,
            'model_size': 0.1,
        },
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_multi_objective_config(
    objectives: list = None,
    objective_weights: dict = None,
    time_budget: int = 3600
) -> NASConfig:
    """
    Get a configuration for multi-objective optimization.
    
    This configuration focuses on finding a diverse Pareto front of
    architectures with different trade-offs between objectives.
    
    Args:
        objectives: List of objectives to optimize (default: ['accuracy', 'latency', 'model_size'])
        objective_weights: Optional weights for objectives
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for multi-objective optimization
    """
    if objectives is None:
        objectives = ['accuracy', 'latency', 'model_size']
    
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=100,
        
        # Evolutionary strategy parameters (larger population for diversity)
        evolution_population_size=50,
        evolution_mutation_rate=0.2,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.1,
        early_stopping_patience=5,
        
        # Hardware constraints (enabled for latency/memory objectives)
        enable_hardware_aware=True,
        target_hardware='cpu',
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=objectives,
        objective_weights=objective_weights,
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_high_accuracy_config(time_budget: int = 7200) -> NASConfig:
    """
    Get a configuration optimized for finding the highest accuracy architecture.
    
    This configuration uses more thorough search and evaluation, suitable
    when accuracy is the primary concern and computational resources are available.
    
    Args:
        time_budget: Time budget in seconds (default: 7200 = 2 hours)
    
    Returns:
        NASConfig configured for high accuracy
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=150,
        
        # Evolutionary strategy parameters
        evolution_population_size=50,
        evolution_mutation_rate=0.15,
        
        # Performance estimation (more thorough)
        performance_estimator='learning_curve',
        estimation_budget_fraction=0.2,
        early_stopping_patience=10,
        
        # Hardware constraints (disabled)
        enable_hardware_aware=False,
        
        # Multi-objective optimization (accuracy only)
        enable_multi_objective=False,
        objectives=['accuracy'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_rl_search_config(time_budget: int = 3600) -> NASConfig:
    """
    Get a configuration using reinforcement learning search strategy.
    
    This configuration uses an RL controller to generate architectures,
    suitable for discrete search spaces and when you want to leverage
    the RL approach (similar to NASNet).
    
    Args:
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for RL search
    """
    return NASConfig(
        # Search configuration
        search_strategy='rl',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=100,
        
        # RL strategy parameters
        rl_controller_hidden_size=100,
        rl_baseline_decay=0.95,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.1,
        early_stopping_patience=5,
        
        # Hardware constraints
        enable_hardware_aware=False,
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_darts_config(time_budget: int = 3600) -> NASConfig:
    """
    Get a configuration using DARTS (gradient-based) search strategy.
    
    This configuration uses differentiable architecture search with
    a supernet, suitable for continuous relaxation of the search space
    and when you want fast, efficient search.
    
    Args:
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for DARTS search
    """
    return NASConfig(
        # Search configuration
        search_strategy='darts',
        search_space_type='auto',
        time_budget=time_budget,
        max_architectures=50,  # DARTS evaluates fewer architectures
        
        # DARTS strategy parameters
        darts_supernet_epochs=50,
        
        # Performance estimation (weight sharing is implicit in DARTS)
        performance_estimator='weight_sharing',
        estimation_budget_fraction=0.2,
        early_stopping_patience=5,
        
        # Hardware constraints
        enable_hardware_aware=False,
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=5,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_vision_config(time_budget: int = 3600) -> NASConfig:
    """
    Get a configuration optimized for computer vision tasks.
    
    This configuration uses a vision-specific search space with
    convolutional layers and residual connections.
    
    Args:
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for vision tasks
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='vision',
        time_budget=time_budget,
        max_architectures=100,
        
        # Evolutionary strategy parameters
        evolution_population_size=40,
        evolution_mutation_rate=0.2,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.1,
        early_stopping_patience=5,
        
        # Hardware constraints
        enable_hardware_aware=False,
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


def get_timeseries_config(time_budget: int = 3600) -> NASConfig:
    """
    Get a configuration optimized for time series tasks.
    
    This configuration uses a time series-specific search space with
    recurrent layers (LSTM, GRU) and temporal convolutions.
    
    Args:
        time_budget: Time budget in seconds (default: 3600 = 1 hour)
    
    Returns:
        NASConfig configured for time series tasks
    """
    return NASConfig(
        # Search configuration
        search_strategy='evolutionary',
        search_space_type='timeseries',
        time_budget=time_budget,
        max_architectures=80,
        
        # Evolutionary strategy parameters
        evolution_population_size=30,
        evolution_mutation_rate=0.2,
        
        # Performance estimation
        performance_estimator='early_stopping',
        estimation_budget_fraction=0.15,
        early_stopping_patience=5,
        
        # Hardware constraints
        enable_hardware_aware=False,
        
        # Multi-objective optimization
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        
        # Transfer learning
        enable_transfer_learning=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_frequency=10,
        
        # Logging
        verbose=True,
        log_all_architectures=True,
    )


# Dictionary of all available templates
TEMPLATES = {
    'quick_start': get_quick_start_config,
    'mobile': get_mobile_deployment_config,
    'edge': get_edge_deployment_config,
    'multi_objective': get_multi_objective_config,
    'high_accuracy': get_high_accuracy_config,
    'rl': get_rl_search_config,
    'darts': get_darts_config,
    'vision': get_vision_config,
    'timeseries': get_timeseries_config,
}


def get_template(template_name: str, **kwargs) -> NASConfig:
    """
    Get a configuration template by name.
    
    Args:
        template_name: Name of the template (e.g., 'quick_start', 'mobile', 'edge')
        **kwargs: Additional arguments to pass to the template function
    
    Returns:
        NASConfig instance
    
    Raises:
        ValueError: If template_name is not recognized
    """
    if template_name not in TEMPLATES:
        available = ', '.join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available templates: {available}")
    
    return TEMPLATES[template_name](**kwargs)


def list_templates() -> list:
    """
    Get a list of all available template names.
    
    Returns:
        List of template names
    """
    return list(TEMPLATES.keys())


def print_template_info():
    """Print information about all available templates."""
    print("Available NAS Configuration Templates:")
    print("=" * 60)
    
    template_info = {
        'quick_start': 'Quick start with evolutionary search (30 min)',
        'mobile': 'Hardware-aware search for mobile deployment',
        'edge': 'Hardware-aware search for edge devices',
        'multi_objective': 'Multi-objective optimization with Pareto front',
        'high_accuracy': 'Thorough search for maximum accuracy (2 hours)',
        'rl': 'Reinforcement learning search strategy',
        'darts': 'Gradient-based DARTS search strategy',
        'vision': 'Optimized for computer vision tasks',
        'timeseries': 'Optimized for time series forecasting',
    }
    
    for name, description in template_info.items():
        print(f"\n{name}:")
        print(f"  {description}")
    
    print("\n" + "=" * 60)
    print("Usage: config = get_template('template_name', time_budget=3600)")

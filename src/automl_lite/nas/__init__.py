"""
Neural Architecture Search (NAS) module for AutoML Lite.

This module provides automated neural network architecture discovery
with support for multiple search strategies, hardware-aware optimization,
and multi-objective optimization.
"""

from .architecture import Architecture, LayerConfig, NASConfig, NASResult
from .validators import ArchitectureValidator
from .search_space import (
    SearchSpace,
    TabularSearchSpace,
    VisionSearchSpace,
    TimeSeriesSearchSpace,
)
from .performance_estimator import (
    PerformanceEstimator,
    PerformanceEstimate,
    EarlyStoppingEstimator,
    LearningCurveEstimator,
    WeightSharingEstimator,
)
from .search_strategy import (
    SearchStrategy,
    SearchHistory,
    EvolutionarySearchStrategy,
    RLSearchStrategy,
    DARTSSearchStrategy,
)
from .hardware_profiler import (
    HardwareProfiler,
    HardwareMetrics,
    LatencyPredictor,
    MemoryEstimator,
    HardwareConstraintChecker,
)
from .multi_objective import (
    MultiObjectiveOptimizer,
    Objective,
)
from .repository import ArchitectureRepository
from .controller import NASController
from .visualization import NASVisualizer
from .logging_utils import NASLogger, create_architecture_summary
from .config_templates import (
    get_quick_start_config,
    get_mobile_deployment_config,
    get_edge_deployment_config,
    get_multi_objective_config,
    get_high_accuracy_config,
    get_rl_search_config,
    get_darts_config,
    get_vision_config,
    get_timeseries_config,
    get_template,
    list_templates,
    print_template_info,
)
from .utils import (
    compare_architectures,
    architecture_diff,
    calculate_flops,
    calculate_parameters,
    get_architecture_complexity_metrics,
    estimate_search_space_size,
    get_layer_type_distribution,
    get_architecture_statistics,
    format_architecture_summary,
)

__all__ = [
    'Architecture',
    'LayerConfig',
    'NASConfig',
    'NASResult',
    'ArchitectureValidator',
    'SearchSpace',
    'TabularSearchSpace',
    'VisionSearchSpace',
    'TimeSeriesSearchSpace',
    'PerformanceEstimator',
    'PerformanceEstimate',
    'EarlyStoppingEstimator',
    'LearningCurveEstimator',
    'WeightSharingEstimator',
    'SearchStrategy',
    'SearchHistory',
    'EvolutionarySearchStrategy',
    'RLSearchStrategy',
    'DARTSSearchStrategy',
    'HardwareProfiler',
    'HardwareMetrics',
    'LatencyPredictor',
    'MemoryEstimator',
    'HardwareConstraintChecker',
    'MultiObjectiveOptimizer',
    'Objective',
    'ArchitectureRepository',
    'NASController',
    'NASVisualizer',
    'NASLogger',
    'create_architecture_summary',
    # Configuration templates
    'get_quick_start_config',
    'get_mobile_deployment_config',
    'get_edge_deployment_config',
    'get_multi_objective_config',
    'get_high_accuracy_config',
    'get_rl_search_config',
    'get_darts_config',
    'get_vision_config',
    'get_timeseries_config',
    'get_template',
    'list_templates',
    'print_template_info',
    # Utilities
    'compare_architectures',
    'architecture_diff',
    'calculate_flops',
    'calculate_parameters',
    'get_architecture_complexity_metrics',
    'estimate_search_space_size',
    'get_layer_type_distribution',
    'get_architecture_statistics',
    'format_architecture_summary',
]

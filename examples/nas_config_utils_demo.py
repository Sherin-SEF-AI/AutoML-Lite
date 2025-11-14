"""
Demo script for NAS configuration templates and utility functions.

This script demonstrates:
1. Using pre-configured NAS templates
2. Architecture comparison and diff
3. Complexity metrics calculation
4. Search space size estimation
"""

import numpy as np
from automl_lite.nas import (
    # Configuration templates
    get_quick_start_config,
    get_mobile_deployment_config,
    get_edge_deployment_config,
    get_multi_objective_config,
    get_template,
    list_templates,
    print_template_info,
    # Utilities
    compare_architectures,
    architecture_diff,
    calculate_flops,
    calculate_parameters,
    get_architecture_complexity_metrics,
    estimate_search_space_size,
    get_architecture_statistics,
    format_architecture_summary,
    # Core classes
    Architecture,
    LayerConfig,
    TabularSearchSpace,
)


def demo_configuration_templates():
    """Demonstrate configuration templates."""
    print("=" * 80)
    print("CONFIGURATION TEMPLATES DEMO")
    print("=" * 80)
    
    # List all available templates
    print("\n1. Available Templates:")
    print("-" * 80)
    print_template_info()
    
    # Quick start configuration
    print("\n2. Quick Start Configuration:")
    print("-" * 80)
    quick_config = get_quick_start_config(time_budget=1800)
    print(f"Search strategy: {quick_config.search_strategy}")
    print(f"Time budget: {quick_config.time_budget}s")
    print(f"Max architectures: {quick_config.max_architectures}")
    print(f"Population size: {quick_config.evolution_population_size}")
    
    # Mobile deployment configuration
    print("\n3. Mobile Deployment Configuration:")
    print("-" * 80)
    mobile_config = get_mobile_deployment_config(
        max_latency_ms=100.0,
        max_model_size_mb=10.0
    )
    print(f"Hardware-aware: {mobile_config.enable_hardware_aware}")
    print(f"Target hardware: {mobile_config.target_hardware}")
    print(f"Max latency: {mobile_config.max_latency_ms}ms")
    print(f"Max model size: {mobile_config.max_model_size_mb}MB")
    print(f"Objectives: {mobile_config.objectives}")
    print(f"Objective weights: {mobile_config.objective_weights}")
    
    # Edge deployment configuration
    print("\n4. Edge Deployment Configuration:")
    print("-" * 80)
    edge_config = get_edge_deployment_config(
        max_latency_ms=50.0,
        max_memory_mb=256.0
    )
    print(f"Target hardware: {edge_config.target_hardware}")
    print(f"Max latency: {edge_config.max_latency_ms}ms")
    print(f"Max memory: {edge_config.max_memory_mb}MB")
    print(f"Max model size: {edge_config.max_model_size_mb}MB")
    
    # Multi-objective configuration
    print("\n5. Multi-Objective Configuration:")
    print("-" * 80)
    multi_obj_config = get_multi_objective_config(
        objectives=['accuracy', 'latency', 'model_size'],
        objective_weights={'accuracy': 0.6, 'latency': 0.3, 'model_size': 0.1}
    )
    print(f"Multi-objective: {multi_obj_config.enable_multi_objective}")
    print(f"Objectives: {multi_obj_config.objectives}")
    print(f"Weights: {multi_obj_config.objective_weights}")
    print(f"Population size: {multi_obj_config.evolution_population_size}")
    
    # Get template by name
    print("\n6. Get Template by Name:")
    print("-" * 80)
    rl_config = get_template('rl', time_budget=3600)
    print(f"Template: rl")
    print(f"Search strategy: {rl_config.search_strategy}")
    print(f"Controller hidden size: {rl_config.rl_controller_hidden_size}")
    print(f"Baseline decay: {rl_config.rl_baseline_decay}")


def demo_architecture_comparison():
    """Demonstrate architecture comparison utilities."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON DEMO")
    print("=" * 80)
    
    # Create two similar architectures
    arch1 = Architecture(
        layers=[
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ],
        connections=[(0, 2)],  # Skip connection
    )
    
    arch2 = Architecture(
        layers=[
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.5}),
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),  # Different size
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ],
        connections=[(0, 2)],  # Same skip connection
    )
    
    # Compare architectures
    print("\n1. Architecture Comparison:")
    print("-" * 80)
    comparison = compare_architectures(arch1, arch2)
    print(f"Similarity score: {comparison['similarity_score']:.2%}")
    print(f"Layer count difference: {comparison['num_layers_diff']}")
    print(f"Connection difference: {comparison['connections_diff']}")
    print(f"Layer type differences: {len(comparison['layer_type_diff'])}")
    
    # Generate diff
    print("\n2. Architecture Diff:")
    print("-" * 80)
    diff = architecture_diff(arch1, arch2)
    print(diff)


def demo_complexity_metrics():
    """Demonstrate complexity metrics calculation."""
    print("\n" + "=" * 80)
    print("COMPLEXITY METRICS DEMO")
    print("=" * 80)
    
    # Create a sample architecture
    architecture = Architecture(
        layers=[
            LayerConfig('dense', {'units': 256, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ],
        connections=[(0, 2), (2, 4)],
    )
    
    input_shape = (32, 100)  # (batch_size, features)
    
    # Calculate individual metrics
    print("\n1. Individual Metrics:")
    print("-" * 80)
    num_params = calculate_parameters(architecture, input_shape)
    flops = calculate_flops(architecture, input_shape)
    print(f"Total parameters: {num_params:,}")
    print(f"Total FLOPs: {flops:,}")
    print(f"Model size (float32): {(num_params * 4) / (1024 * 1024):.2f} MB")
    
    # Get comprehensive metrics
    print("\n2. Comprehensive Metrics:")
    print("-" * 80)
    metrics = get_architecture_complexity_metrics(architecture, input_shape)
    print(f"Number of layers: {metrics['num_layers']}")
    print(f"Parameters: {metrics['num_parameters']:,}")
    print(f"FLOPs: {metrics['flops']:,}")
    print(f"Model size: {metrics['model_size_mb']:.2f} MB")
    print(f"Skip connections: {metrics['num_skip_connections']}")
    
    # Format architecture summary
    print("\n3. Architecture Summary:")
    print("-" * 80)
    summary = format_architecture_summary(architecture, input_shape)
    print(summary)


def demo_search_space_estimation():
    """Demonstrate search space size estimation."""
    print("\n" + "=" * 80)
    print("SEARCH SPACE SIZE ESTIMATION DEMO")
    print("=" * 80)
    
    # Estimate tabular search space
    print("\n1. Tabular Search Space:")
    print("-" * 80)
    layer_types = ['dense', 'dropout', 'batchnormalization']
    param_ranges = {
        'units': [16, 32, 64, 128, 256, 512],
        'activation': ['relu', 'tanh', 'elu'],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    }
    
    size_no_skip = estimate_search_space_size(
        layer_types=layer_types,
        param_ranges=param_ranges,
        min_layers=2,
        max_layers=5,
        allow_skip_connections=False
    )
    print(f"Search space size (no skip connections): {size_no_skip:,}")
    
    size_with_skip = estimate_search_space_size(
        layer_types=layer_types,
        param_ranges=param_ranges,
        min_layers=2,
        max_layers=5,
        allow_skip_connections=True
    )
    print(f"Search space size (with skip connections): {size_with_skip:,}")
    
    # Estimate vision search space
    print("\n2. Vision Search Space:")
    print("-" * 80)
    vision_layer_types = ['conv2d', 'maxpooling2d', 'dense', 'dropout']
    vision_param_ranges = {
        'filters': [16, 32, 64, 128, 256],
        'kernel_size': [3, 5, 7],
        'units': [64, 128, 256, 512],
    }
    
    vision_size = estimate_search_space_size(
        layer_types=vision_layer_types,
        param_ranges=vision_param_ranges,
        min_layers=5,
        max_layers=10,
        allow_skip_connections=True
    )
    print(f"Vision search space size: {vision_size:,}")


def demo_architecture_statistics():
    """Demonstrate architecture statistics calculation."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE STATISTICS DEMO")
    print("=" * 80)
    
    # Create a population of architectures
    search_space = TabularSearchSpace(
        input_shape=(100,),
        output_shape=(10,),
        problem_type='classification'
    )
    architectures = [search_space.sample_architecture() for _ in range(20)]
    
    # Calculate statistics
    print("\n1. Population Statistics:")
    print("-" * 80)
    stats = get_architecture_statistics(architectures)
    print(f"Number of architectures: {stats['num_architectures']}")
    print(f"Average layers: {stats['avg_num_layers']:.2f}")
    print(f"Min layers: {stats['min_num_layers']}")
    print(f"Max layers: {stats['max_num_layers']}")
    print(f"Average skip connections: {stats['avg_skip_connections']:.2f}")
    
    print("\n2. Layer Type Distribution:")
    print("-" * 80)
    for layer_type, count in sorted(stats['layer_type_distribution'].items()):
        print(f"  {layer_type}: {count}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("NAS CONFIGURATION TEMPLATES AND UTILITIES DEMO")
    print("=" * 80)
    
    # Run demos
    demo_configuration_templates()
    demo_architecture_comparison()
    demo_complexity_metrics()
    demo_search_space_estimation()
    demo_architecture_statistics()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

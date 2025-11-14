"""
Example demonstrating the NAS architecture data structures.

This example shows how to:
1. Create layer configurations
2. Build architectures with skip connections
3. Validate architectures
4. Serialize/deserialize architectures
5. Configure NAS search parameters
"""

from automl_lite.nas import (
    Architecture,
    LayerConfig,
    NASConfig,
    NASResult,
    ArchitectureValidator,
)


def example_simple_mlp():
    """Create a simple MLP architecture."""
    print("=" * 60)
    print("Example 1: Simple MLP Architecture")
    print("=" * 60)
    
    # Define layers
    layers = [
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.2}),
        LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
    ]
    
    # Create architecture
    arch = Architecture(
        layers=layers,
        global_config={
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'batch_size': 32,
        }
    )
    
    print(f"\nArchitecture: {arch}")
    print(f"Number of layers: {arch.get_num_layers()}")
    print(f"Architecture ID: {arch.id}")
    
    # Validate architecture
    validator = ArchitectureValidator()
    is_valid, errors = validator.validate_architecture(arch, input_shape=(100,))
    
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
    
    # Show inferred shapes
    print("\nLayer shapes:")
    for i, layer in enumerate(arch.layers):
        print(f"  Layer {i} ({layer.layer_type}): {layer.input_shape} -> {layer.output_shape}")
    
    return arch


def example_cnn_architecture():
    """Create a CNN architecture for image classification."""
    print("\n" + "=" * 60)
    print("Example 2: CNN Architecture with Skip Connections")
    print("=" * 60)
    
    # Define layers
    layers = [
        LayerConfig('conv2d', {'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}),
        LayerConfig('batch_normalization', {}),
        LayerConfig('conv2d', {'filters': 32, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}),
        LayerConfig('max_pooling2d', {'pool_size': 2}),
        LayerConfig('conv2d', {'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}),
        LayerConfig('batch_normalization', {}),
        LayerConfig('conv2d', {'filters': 64, 'kernel_size': 3, 'padding': 'same', 'activation': 'relu'}),
        LayerConfig('max_pooling2d', {'pool_size': 2}),
        LayerConfig('flatten', {}),
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.5}),
        LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
    ]
    
    # Add skip connections (residual connections)
    connections = [
        (0, 2),  # Skip connection from first conv to third conv
        (4, 6),  # Skip connection from fifth conv to seventh conv
    ]
    
    # Create architecture
    arch = Architecture(
        layers=layers,
        connections=connections,
        global_config={
            'optimizer': 'adam',
            'learning_rate': 0.001,
        }
    )
    
    print(f"\nArchitecture: {arch}")
    print(f"Skip connections: {arch.connections}")
    
    # Validate with image input
    validator = ArchitectureValidator()
    is_valid, errors = validator.validate_architecture(arch, input_shape=(28, 28, 1))
    
    print(f"\nValidation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Show key layer shapes
    print("\nKey layer shapes:")
    for i in [0, 3, 4, 7, 8, 9, 11]:
        layer = arch.layers[i]
        print(f"  Layer {i} ({layer.layer_type}): {layer.input_shape} -> {layer.output_shape}")
    
    return arch


def example_serialization(arch):
    """Demonstrate architecture serialization."""
    print("\n" + "=" * 60)
    print("Example 3: Architecture Serialization")
    print("=" * 60)
    
    # Add some metadata
    arch.set_performance_metric('accuracy', 0.95)
    arch.set_performance_metric('loss', 0.15)
    arch.set_hardware_metric('latency', 45.2)
    arch.set_hardware_metric('memory', 120.5)
    
    # Serialize to JSON
    arch_json = arch.to_json()
    print(f"\nSerialized architecture (first 500 chars):")
    print(arch_json[:500] + "...")
    
    # Deserialize
    arch_restored = Architecture.from_json(arch_json)
    print(f"\nRestored architecture: {arch_restored}")
    print(f"Accuracy: {arch_restored.get_performance_metric('accuracy')}")
    print(f"Latency: {arch_restored.get_hardware_metric('latency')} ms")
    
    return arch_restored


def example_nas_config():
    """Demonstrate NAS configuration."""
    print("\n" + "=" * 60)
    print("Example 4: NAS Configuration")
    print("=" * 60)
    
    # Create a basic configuration
    config = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=1800,  # 30 minutes
        max_architectures=50,
        enable_hardware_aware=True,
        target_hardware='mobile',
        max_latency_ms=100.0,
        max_model_size_mb=10.0,
    )
    
    print("\nNAS Configuration:")
    print(f"  Search strategy: {config.search_strategy}")
    print(f"  Time budget: {config.time_budget}s")
    print(f"  Max architectures: {config.max_architectures}")
    print(f"  Hardware-aware: {config.enable_hardware_aware}")
    print(f"  Target hardware: {config.target_hardware}")
    print(f"  Max latency: {config.max_latency_ms}ms")
    print(f"  Max model size: {config.max_model_size_mb}MB")
    
    # Demonstrate validation
    print("\n✓ Configuration is valid")
    
    # Try invalid configuration
    print("\nTrying invalid configuration...")
    try:
        invalid_config = NASConfig(search_strategy='invalid_strategy')
    except ValueError as e:
        print(f"✗ Validation error: {e}")
    
    return config


def example_nas_result():
    """Demonstrate NAS result structure."""
    print("\n" + "=" * 60)
    print("Example 5: NAS Result")
    print("=" * 60)
    
    # Create some example architectures
    architectures = []
    for i, (units, acc) in enumerate([(128, 0.92), (256, 0.95), (64, 0.88), (512, 0.94)]):
        layers = [
            LayerConfig('dense', {'units': units, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        arch = Architecture(layers=layers)
        arch.set_performance_metric('accuracy', acc)
        arch.set_hardware_metric('latency', 50.0 + i * 10)
        architectures.append(arch)
    
    # Create result
    best_arch = architectures[1]  # Highest accuracy
    result = NASResult(
        best_architecture=best_arch,
        all_architectures=architectures,
        pareto_front=[architectures[1], architectures[3]],
        search_time=1800.0,
        total_architectures_evaluated=50,
        best_accuracy=0.95,
        best_latency=60.0,
        search_strategy='evolutionary',
        search_space_type='tabular',
    )
    
    print(f"\nNAS Result: {result}")
    
    # Get summary
    summary = result.get_summary()
    print("\nSearch Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Get top architectures
    print("\nTop 3 architectures by accuracy:")
    top_3 = result.get_top_k_architectures(k=3, metric='accuracy')
    for i, arch in enumerate(top_3, 1):
        acc = arch.get_performance_metric('accuracy')
        lat = arch.get_hardware_metric('latency')
        print(f"  {i}. Accuracy: {acc:.3f}, Latency: {lat:.1f}ms, Layers: {arch.get_num_layers()}")
    
    return result


def example_architecture_cloning():
    """Demonstrate architecture cloning and modification."""
    print("\n" + "=" * 60)
    print("Example 6: Architecture Cloning")
    print("=" * 60)
    
    # Create original architecture
    layers = [
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
    ]
    original = Architecture(layers=layers)
    original.set_performance_metric('accuracy', 0.90)
    
    print(f"Original: {original}")
    print(f"  ID: {original.id}")
    print(f"  Accuracy: {original.get_performance_metric('accuracy')}")
    
    # Clone and modify
    cloned = original.clone()
    cloned.set_performance_metric('accuracy', 0.92)
    cloned.layers.append(LayerConfig('dense', {'units': 32, 'activation': 'relu'}))
    
    print(f"\nCloned: {cloned}")
    print(f"  ID: {cloned.id}")
    print(f"  Accuracy: {cloned.get_performance_metric('accuracy')}")
    print(f"  Layers: {cloned.get_num_layers()}")
    
    print(f"\nOriginal unchanged:")
    print(f"  Accuracy: {original.get_performance_metric('accuracy')}")
    print(f"  Layers: {original.get_num_layers()}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("NAS Architecture Data Structures Examples")
    print("=" * 60)
    
    # Run examples
    arch1 = example_simple_mlp()
    arch2 = example_cnn_architecture()
    arch_restored = example_serialization(arch1)
    config = example_nas_config()
    result = example_nas_result()
    example_architecture_cloning()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()

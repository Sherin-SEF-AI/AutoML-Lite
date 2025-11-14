"""
Demo script for NAS Search Space functionality.

This script demonstrates how to use the different search spaces to sample,
validate, and mutate neural network architectures.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from automl_lite.nas import (
    TabularSearchSpace,
    VisionSearchSpace,
    TimeSeriesSearchSpace,
)


def demo_tabular_search_space():
    """Demonstrate TabularSearchSpace for structured data."""
    print("=" * 70)
    print("TABULAR SEARCH SPACE DEMO")
    print("=" * 70)
    
    # Create search space for tabular classification
    search_space = TabularSearchSpace(
        input_shape=(100,),  # 100 input features
        output_shape=(10,),  # 10 classes
        problem_type='classification',
        random_seed=42,
        enable_skip_connections=True
    )
    
    print(f"\nSearch Space: {search_space}")
    print(f"Estimated size: {search_space.get_search_space_size():,} architectures")
    
    # Sample a random architecture
    print("\n--- Sampling Architecture ---")
    arch = search_space.sample_architecture()
    print(f"Architecture ID: {arch.id[:8]}...")
    print(f"Number of layers: {len(arch.layers)}")
    print(f"Skip connections: {len(arch.connections)}")
    
    print("\nLayers:")
    for i, layer in enumerate(arch.layers):
        print(f"  {i}: {layer}")
    
    if arch.connections:
        print(f"\nSkip connections: {arch.connections}")
    
    print(f"\nGlobal config: {arch.global_config}")
    
    # Validate the architecture
    is_valid = search_space.validate_architecture(arch)
    print(f"\nArchitecture is valid: {is_valid}")
    
    # Mutate the architecture
    print("\n--- Mutating Architecture ---")
    mutated = search_space.mutate_architecture(arch, mutation_rate=0.3)
    print(f"Mutated Architecture ID: {mutated.id[:8]}...")
    print(f"Number of layers: {len(mutated.layers)}")
    print(f"Skip connections: {len(mutated.connections)}")
    
    print("\nMutated layers:")
    for i, layer in enumerate(mutated.layers):
        print(f"  {i}: {layer}")
    
    # Crossover example
    print("\n--- Crossover Example ---")
    arch2 = search_space.sample_architecture()
    child = search_space.crossover(arch, arch2)
    print(f"Parent 1 layers: {len(arch.layers)}")
    print(f"Parent 2 layers: {len(arch2.layers)}")
    print(f"Child layers: {len(child.layers)}")
    print(f"Child is valid: {search_space.validate_architecture(child)}")


def demo_vision_search_space():
    """Demonstrate VisionSearchSpace for image data."""
    print("\n\n" + "=" * 70)
    print("VISION SEARCH SPACE DEMO")
    print("=" * 70)
    
    # Create search space for image classification
    search_space = VisionSearchSpace(
        input_shape=(28, 28, 1),  # MNIST-like images
        output_shape=(10,),  # 10 classes
        problem_type='classification',
        random_seed=42,
        enable_residual_connections=True
    )
    
    print(f"\nSearch Space: {search_space}")
    print(f"Estimated size: {search_space.get_search_space_size():,} architectures")
    
    # Sample a random architecture
    print("\n--- Sampling CNN Architecture ---")
    arch = search_space.sample_architecture()
    print(f"Architecture ID: {arch.id[:8]}...")
    print(f"Number of layers: {len(arch.layers)}")
    print(f"Residual connections: {len(arch.connections)}")
    
    # Count layer types
    conv_layers = [l for l in arch.layers if l.layer_type == 'conv2d']
    pool_layers = [l for l in arch.layers if l.layer_type in ['max_pool2d', 'avg_pool2d']]
    dense_layers = [l for l in arch.layers if l.layer_type == 'dense']
    
    print(f"\nLayer breakdown:")
    print(f"  Conv2D layers: {len(conv_layers)}")
    print(f"  Pooling layers: {len(pool_layers)}")
    print(f"  Dense layers: {len(dense_layers)}")
    
    print("\nFirst few layers:")
    for i, layer in enumerate(arch.layers[:5]):
        print(f"  {i}: {layer}")
    
    if arch.connections:
        print(f"\nResidual connections: {arch.connections}")
    
    # Validate the architecture
    is_valid = search_space.validate_architecture(arch)
    print(f"\nArchitecture is valid: {is_valid}")
    
    # Mutate the architecture
    print("\n--- Mutating CNN Architecture ---")
    mutated = search_space.mutate_architecture(arch, mutation_rate=0.3)
    print(f"Original layers: {len(arch.layers)}")
    print(f"Mutated layers: {len(mutated.layers)}")
    print(f"Mutated is valid: {search_space.validate_architecture(mutated)}")


def demo_timeseries_search_space():
    """Demonstrate TimeSeriesSearchSpace for sequential data."""
    print("\n\n" + "=" * 70)
    print("TIME SERIES SEARCH SPACE DEMO")
    print("=" * 70)
    
    # Create search space for time series classification
    search_space = TimeSeriesSearchSpace(
        input_shape=(100, 10),  # 100 timesteps, 10 features
        output_shape=(5,),  # 5 classes
        problem_type='classification',
        random_seed=42
    )
    
    print(f"\nSearch Space: {search_space}")
    print(f"Estimated size: {search_space.get_search_space_size():,} architectures")
    
    # Sample multiple architectures to show variety
    print("\n--- Sampling Multiple Architectures ---")
    
    for i in range(3):
        arch = search_space.sample_architecture()
        
        # Count layer types
        lstm_layers = [l for l in arch.layers if l.layer_type == 'lstm']
        gru_layers = [l for l in arch.layers if l.layer_type == 'gru']
        conv_layers = [l for l in arch.layers if l.layer_type == 'conv1d']
        dense_layers = [l for l in arch.layers if l.layer_type == 'dense']
        
        print(f"\nArchitecture {i+1}:")
        print(f"  Total layers: {len(arch.layers)}")
        print(f"  LSTM: {len(lstm_layers)}, GRU: {len(gru_layers)}, Conv1D: {len(conv_layers)}, Dense: {len(dense_layers)}")
        
        # Determine architecture type
        if lstm_layers or gru_layers:
            if conv_layers:
                arch_type = "Hybrid (RNN + CNN)"
            else:
                arch_type = "Pure RNN"
        else:
            arch_type = "Pure CNN"
        
        print(f"  Type: {arch_type}")
        print(f"  Valid: {search_space.validate_architecture(arch)}")
        
        # Show first few layers
        print("  First 3 layers:")
        for j, layer in enumerate(arch.layers[:3]):
            print(f"    {j}: {layer}")
    
    # Demonstrate mutation
    print("\n--- Mutation Example ---")
    arch = search_space.sample_architecture()
    print(f"Original architecture has {len(arch.layers)} layers")
    
    mutated = search_space.mutate_architecture(arch, mutation_rate=0.5)
    print(f"Mutated architecture has {len(mutated.layers)} layers")
    print(f"Mutated is valid: {search_space.validate_architecture(mutated)}")


def demo_architecture_operations():
    """Demonstrate architecture graph operations."""
    print("\n\n" + "=" * 70)
    print("ARCHITECTURE OPERATIONS DEMO")
    print("=" * 70)
    
    search_space = TabularSearchSpace(
        input_shape=(50,),
        output_shape=(5,),
        problem_type='classification'
    )
    
    # Start with a simple architecture
    arch = search_space.sample_architecture()
    print(f"\nOriginal architecture: {len(arch.layers)} layers")
    
    # Add a layer
    from automl_lite.nas import LayerConfig
    new_layer = LayerConfig('dropout', {'rate': 0.4})
    arch_with_layer = search_space.add_layer(arch, new_layer, position=-1)
    print(f"After adding dropout: {len(arch_with_layer.layers)} layers")
    
    # Modify a layer
    arch_modified = search_space.modify_layer(arch_with_layer, 0, {'units': 512})
    print(f"Modified first layer units to 512")
    
    # Add a skip connection
    if len(arch_modified.layers) >= 3:
        arch_with_skip = search_space.add_connection(arch_modified, 0, 2)
        print(f"Added skip connection: {arch_with_skip.connections}")
    
    # Remove a layer
    if len(arch_modified.layers) > 2:
        arch_removed = search_space.remove_layer(arch_modified, 1)
        print(f"After removing layer: {len(arch_removed.layers)} layers")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("NAS SEARCH SPACE DEMONSTRATION")
    print("=" * 70)
    
    demo_tabular_search_space()
    demo_vision_search_space()
    demo_timeseries_search_space()
    demo_architecture_operations()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

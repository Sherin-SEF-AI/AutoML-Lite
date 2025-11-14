"""
Demo script for NAS visualization and logging capabilities.

This script demonstrates:
1. Architecture diagram rendering
2. Search progress visualization
3. Pareto front visualization
4. Verbose logging
"""

import numpy as np
from sklearn.datasets import make_classification

from automl_lite.nas import (
    Architecture,
    LayerConfig,
    NASVisualizer,
    NASLogger,
    create_architecture_summary,
    TabularSearchSpace,
)


def create_sample_architecture():
    """Create a sample architecture for visualization."""
    layers = [
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('batchnormalization', {}),
        LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
        LayerConfig('dense', {'units': 1, 'activation': 'sigmoid'}),
    ]
    
    # Add skip connection from layer 0 to layer 4
    connections = [(0, 4)]
    
    arch = Architecture(
        layers=layers,
        connections=connections,
        global_config={'optimizer': 'adam', 'learning_rate': 0.001},
        metadata={
            'metrics': {
                'accuracy': 0.92,
                'latency': 15.5,
                'model_size': 2.3
            }
        }
    )
    
    return arch


def demo_architecture_diagram():
    """Demonstrate architecture diagram rendering."""
    print("\n" + "="*80)
    print("Demo 1: Architecture Diagram Rendering")
    print("="*80)
    
    visualizer = NASVisualizer()
    arch = create_sample_architecture()
    
    print(f"\nArchitecture ID: {arch.id[:8]}")
    print(f"Layers: {len(arch.layers)}")
    print(f"Skip connections: {len(arch.connections)}")
    
    # Render diagram
    try:
        diagram = visualizer.render_architecture_diagram(arch, format='base64')
        print(f"\n✅ Architecture diagram rendered successfully")
        print(f"   Output format: base64 encoded PNG")
        print(f"   Data length: {len(diagram)} characters")
    except Exception as e:
        print(f"\n❌ Failed to render diagram: {e}")
        print("   Note: graphviz may not be installed, using matplotlib fallback")


def demo_search_progress():
    """Demonstrate search progress visualization."""
    print("\n" + "="*80)
    print("Demo 2: Search Progress Visualization")
    print("="*80)
    
    visualizer = NASVisualizer()
    
    # Create mock search history
    search_history = []
    base_accuracy = 0.70
    for i in range(50):
        # Simulate improving accuracy with some noise
        accuracy = base_accuracy + (i * 0.004) + np.random.normal(0, 0.02)
        accuracy = min(0.95, max(0.65, accuracy))
        
        search_history.append({
            'architecture_id': f'arch_{i}',
            'accuracy': accuracy,
            'score': accuracy,
            'latency': 10 + np.random.uniform(-3, 3),
            'model_size': 2 + np.random.uniform(-0.5, 0.5)
        })
    
    print(f"\nSimulated {len(search_history)} architecture evaluations")
    print(f"Best accuracy: {max(h['accuracy'] for h in search_history):.4f}")
    
    # Create visualization
    try:
        plot = visualizer.create_search_progress_plot(search_history, format='base64')
        print(f"\n✅ Search progress plot created successfully")
        print(f"   Output format: base64 encoded PNG")
        print(f"   Data length: {len(plot)} characters")
    except Exception as e:
        print(f"\n❌ Failed to create plot: {e}")


def demo_pareto_front():
    """Demonstrate Pareto front visualization."""
    print("\n" + "="*80)
    print("Demo 3: Pareto Front Visualization")
    print("="*80)
    
    visualizer = NASVisualizer()
    
    # Create mock architectures with trade-offs
    architectures = []
    for i in range(30):
        # Create trade-off: higher accuracy -> higher latency and size
        accuracy = 0.70 + np.random.uniform(0, 0.25)
        latency = 5 + (accuracy - 0.70) * 40 + np.random.uniform(-2, 2)
        model_size = 1 + (accuracy - 0.70) * 8 + np.random.uniform(-0.3, 0.3)
        
        layers = [
            LayerConfig('dense', {'units': 64}),
            LayerConfig('dense', {'units': 32}),
            LayerConfig('dense', {'units': 1}),
        ]
        
        arch = Architecture(
            layers=layers,
            metadata={
                'metrics': {
                    'accuracy': accuracy,
                    'latency': max(5, latency),
                    'model_size': max(1, model_size)
                }
            }
        )
        architectures.append(arch)
    
    print(f"\nCreated {len(architectures)} architectures with varying trade-offs")
    
    # Create 2D Pareto front
    try:
        plot_2d = visualizer.create_pareto_front_plot(
            architectures,
            objectives=['accuracy', 'latency'],
            format='base64',
            highlight_pareto=True
        )
        print(f"\n✅ 2D Pareto front plot created successfully")
        print(f"   Objectives: accuracy vs latency")
        print(f"   Data length: {len(plot_2d)} characters")
    except Exception as e:
        print(f"\n❌ Failed to create 2D plot: {e}")
    
    # Create 3D Pareto front
    try:
        plot_3d = visualizer.create_pareto_front_plot(
            architectures,
            objectives=['accuracy', 'latency', 'model_size'],
            format='base64',
            highlight_pareto=True
        )
        print(f"\n✅ 3D Pareto front plot created successfully")
        print(f"   Objectives: accuracy vs latency vs model_size")
        print(f"   Data length: {len(plot_3d)} characters")
    except Exception as e:
        print(f"\n❌ Failed to create 3D plot: {e}")


def demo_verbose_logging():
    """Demonstrate verbose logging capabilities."""
    print("\n" + "="*80)
    print("Demo 4: Verbose Logging")
    print("="*80)
    
    logger = NASLogger(verbose=True)
    
    # Log search start
    logger.log_search_start(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=1800,
        max_architectures=100
    )
    
    # Simulate architecture evaluations
    search_space = TabularSearchSpace(
        input_shape=(10,),
        output_shape=(1,),
        problem_type='classification'
    )
    
    for i in range(10):
        # Generate architecture
        arch = search_space.sample_architecture()
        summary = create_architecture_summary(arch)
        
        logger.log_architecture_generation(arch.id, 'random_sampling')
        logger.log_architecture_evaluation_start(arch.id, summary)
        
        # Simulate evaluation
        import time
        time.sleep(0.1)  # Simulate evaluation time
        
        metrics = {
            'accuracy': 0.70 + np.random.uniform(0, 0.25),
            'latency': 10 + np.random.uniform(-3, 3),
            'model_size': 2 + np.random.uniform(-0.5, 0.5)
        }
        
        logger.log_architecture_evaluation_complete(arch.id, metrics, 0.1)
        
        # Log progress every 5 iterations
        if (i + 1) % 5 == 0:
            logger.log_search_progress(i + 1, 10)
    
    # Log search complete
    summary = logger.get_search_summary()
    logger.log_search_complete(
        total_architectures=summary['architectures_evaluated'],
        best_architecture_id=summary['best_architecture_id'],
        best_score=summary['best_score']
    )
    
    print(f"\n📊 Search Summary:")
    print(f"   Architectures evaluated: {summary['architectures_evaluated']}")
    print(f"   Best score: {summary['best_score']:.4f}")
    print(f"   Average time per architecture: {summary['avg_time_per_architecture']:.2f}s")


def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("NAS Visualization and Logging Demo")
    print("="*80)
    
    try:
        demo_architecture_diagram()
    except Exception as e:
        print(f"\n❌ Architecture diagram demo failed: {e}")
    
    try:
        demo_search_progress()
    except Exception as e:
        print(f"\n❌ Search progress demo failed: {e}")
    
    try:
        demo_pareto_front()
    except Exception as e:
        print(f"\n❌ Pareto front demo failed: {e}")
    
    try:
        demo_verbose_logging()
    except Exception as e:
        print(f"\n❌ Verbose logging demo failed: {e}")
    
    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)


if __name__ == '__main__':
    main()

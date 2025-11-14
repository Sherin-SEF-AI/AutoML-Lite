"""
Demo script for NASController - orchestrating Neural Architecture Search.

This script demonstrates the complete NAS workflow including:
- Initializing the NASController with configuration
- Running architecture search
- Accessing search results and Pareto front
- Checkpointing and resuming search
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from automl_lite.nas import NASController, NASConfig


def demo_basic_search():
    """Demonstrate basic NAS search."""
    print("=" * 80)
    print("Demo 1: Basic NAS Search")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    
    # Create NAS configuration
    config = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=60,  # 1 minute for demo
        max_architectures=10,
        enable_hardware_aware=False,
        enable_multi_objective=False,
        verbose=True
    )
    
    print(f"\nConfiguration:")
    print(f"  Strategy: {config.search_strategy}")
    print(f"  Time budget: {config.time_budget}s")
    print(f"  Max architectures: {config.max_architectures}")
    
    # Initialize controller
    controller = NASController(config)
    print(f"\nInitialized: {controller}")
    
    # Run search
    print("\nStarting search...")
    result = controller.search(X_train, y_train, problem_type='classification')
    
    # Display results
    print("\n" + "=" * 80)
    print("Search Results")
    print("=" * 80)
    print(f"Total architectures evaluated: {result.total_architectures_evaluated}")
    print(f"Search time: {result.search_time:.1f}s")
    print(f"Best accuracy: {result.best_accuracy:.4f}")
    print(f"\nBest architecture:")
    print(f"  ID: {result.best_architecture.id[:8]}...")
    print(f"  Layers: {result.best_architecture.get_num_layers()}")
    print(f"  Architecture: {result.best_architecture}")
    
    # Get top 3 architectures
    top_3 = result.get_top_k_architectures(k=3, metric='accuracy')
    print(f"\nTop 3 architectures:")
    for i, arch in enumerate(top_3, 1):
        acc = arch.get_performance_metric('accuracy')
        print(f"  {i}. {arch.id[:8]}... - Accuracy: {acc:.4f}, Layers: {arch.get_num_layers()}")
    
    return controller, result


def demo_hardware_aware_search():
    """Demonstrate hardware-aware NAS search."""
    print("\n\n" + "=" * 80)
    print("Demo 2: Hardware-Aware NAS Search")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Create hardware-aware configuration
    config = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=60,
        max_architectures=10,
        enable_hardware_aware=True,
        target_hardware='mobile',
        max_latency_ms=50.0,
        max_model_size_mb=5.0,
        verbose=True
    )
    
    print(f"Hardware constraints:")
    print(f"  Target: {config.target_hardware}")
    print(f"  Max latency: {config.max_latency_ms}ms")
    print(f"  Max model size: {config.max_model_size_mb}MB")
    
    # Run search
    controller = NASController(config)
    result = controller.search(X, y, problem_type='classification')
    
    # Display results with hardware metrics
    print("\n" + "=" * 80)
    print("Search Results (Hardware-Aware)")
    print("=" * 80)
    print(f"Best architecture:")
    print(f"  Accuracy: {result.best_accuracy:.4f}")
    if result.best_latency:
        print(f"  Latency: {result.best_latency:.2f}ms")
    if result.best_model_size:
        print(f"  Model size: {result.best_model_size:.2f}MB")
    
    return controller, result


def demo_multi_objective_search():
    """Demonstrate multi-objective NAS search."""
    print("\n\n" + "=" * 80)
    print("Demo 3: Multi-Objective NAS Search")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Create multi-objective configuration
    config = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=60,
        max_architectures=15,
        enable_hardware_aware=True,
        enable_multi_objective=True,
        objectives=['accuracy', 'latency', 'model_size'],
        objective_weights={'accuracy': 0.6, 'latency': 0.3, 'model_size': 0.1},
        verbose=True
    )
    
    print(f"Objectives: {config.objectives}")
    print(f"Weights: {config.objective_weights}")
    
    # Run search
    controller = NASController(config)
    result = controller.search(X, y, problem_type='classification')
    
    # Display Pareto front
    print("\n" + "=" * 80)
    print("Pareto Front")
    print("=" * 80)
    print(f"Number of non-dominated solutions: {len(result.pareto_front)}")
    
    if result.pareto_front:
        print("\nPareto front architectures:")
        for i, arch in enumerate(result.pareto_front[:5], 1):  # Show first 5
            acc = arch.get_performance_metric('accuracy') or 0.0
            lat = arch.get_hardware_metric('latency_ms') or 0.0
            size = arch.get_hardware_metric('model_size_mb') or 0.0
            print(f"  {i}. Accuracy: {acc:.4f}, Latency: {lat:.2f}ms, Size: {size:.2f}MB")
    
    return controller, result


def demo_checkpointing():
    """Demonstrate checkpointing and resume functionality."""
    print("\n\n" + "=" * 80)
    print("Demo 4: Checkpointing and Resume")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Create configuration with checkpointing
    config = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=30,
        max_architectures=5,
        enable_checkpointing=True,
        checkpoint_frequency=2,
        checkpoint_path='./nas_demo_checkpoint.pkl',
        verbose=True
    )
    
    print("Running initial search with checkpointing...")
    controller = NASController(config)
    result1 = controller.search(X, y, problem_type='classification')
    
    print(f"\nInitial search completed:")
    print(f"  Architectures evaluated: {result1.total_architectures_evaluated}")
    print(f"  Best accuracy: {result1.best_accuracy:.4f}")
    
    # Simulate resuming from checkpoint
    print("\n" + "-" * 80)
    print("Resuming search from checkpoint...")
    
    # Create new controller and resume
    config2 = NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=60,  # Extended budget
        max_architectures=10,  # More architectures
        enable_checkpointing=True,
        checkpoint_path='./nas_demo_checkpoint.pkl',
        verbose=True
    )
    
    controller2 = NASController(config2)
    result2 = controller2.resume_search(
        './nas_demo_checkpoint.pkl',
        X, y,
        problem_type='classification'
    )
    
    print(f"\nResumed search completed:")
    print(f"  Architectures evaluated: {result2.total_architectures_evaluated}")
    print(f"  Best accuracy: {result2.best_accuracy:.4f}")
    
    return controller2, result2


def demo_search_statistics():
    """Demonstrate search statistics and progress tracking."""
    print("\n\n" + "=" * 80)
    print("Demo 5: Search Statistics")
    print("=" * 80)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    config = NASConfig(
        search_strategy='evolutionary',
        time_budget=45,
        max_architectures=8,
        verbose=False  # Disable verbose to show custom progress
    )
    
    controller = NASController(config)
    
    # Run search
    result = controller.search(X, y, problem_type='classification')
    
    # Get and display statistics
    stats = controller.get_search_statistics()
    
    print("\nSearch Statistics:")
    print(f"  Total iterations: {stats['iteration']}")
    print(f"  Successful evaluations: {stats['architectures_evaluated']}")
    print(f"  Success rate: {stats['success_rate']:.1%}")
    print(f"  Error count: {stats['error_count']}")
    print(f"  Best performance: {stats['best_performance']:.4f}")
    print(f"  Elapsed time: {stats['elapsed_time_seconds']:.1f}s")
    
    # Print formatted progress
    controller.print_search_progress()
    
    return controller, result


if __name__ == '__main__':
    print("NASController Demo")
    print("=" * 80)
    print("This demo showcases the NASController orchestrating neural architecture search")
    print()
    
    # Run demos
    try:
        # Demo 1: Basic search
        controller1, result1 = demo_basic_search()
        
        # Demo 2: Hardware-aware search
        controller2, result2 = demo_hardware_aware_search()
        
        # Demo 3: Multi-objective search
        controller3, result3 = demo_multi_objective_search()
        
        # Demo 4: Checkpointing
        controller4, result4 = demo_checkpointing()
        
        # Demo 5: Statistics
        controller5, result5 = demo_search_statistics()
        
        print("\n" + "=" * 80)
        print("All demos completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError running demo: {e}")
        import traceback
        traceback.print_exc()

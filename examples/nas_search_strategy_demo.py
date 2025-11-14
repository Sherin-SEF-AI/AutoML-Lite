"""
Demo script for Neural Architecture Search strategies.

This script demonstrates the three search strategies:
1. EvolutionarySearchStrategy - Genetic algorithm-based search
2. RLSearchStrategy - Reinforcement learning with REINFORCE
3. DARTSSearchStrategy - Gradient-based differentiable search

Each strategy explores the architecture search space differently and
has different strengths and use cases.
"""

import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from automl_lite.nas import (
    TabularSearchSpace,
    VisionSearchSpace,
    TimeSeriesSearchSpace,
    EvolutionarySearchStrategy,
    RLSearchStrategy,
    DARTSSearchStrategy,
)


def demo_evolutionary_search():
    """Demonstrate evolutionary search strategy."""
    print("\n" + "="*70)
    print("EVOLUTIONARY SEARCH STRATEGY DEMO")
    print("="*70)
    
    # Create a simple tabular search space
    input_shape = (20,)  # 20 features
    output_shape = (3,)  # 3 classes
    
    search_space = TabularSearchSpace(
        input_shape=input_shape,
        output_shape=output_shape,
        problem_type='classification',
        random_seed=42
    )
    
    # Initialize evolutionary strategy
    strategy = EvolutionarySearchStrategy(
        search_space=search_space,
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.5,
        tournament_size=3,
        elitism_ratio=0.1,
        random_seed=42
    )
    
    print(f"\nInitialized: {strategy}")
    print(f"Population size: {strategy.population_size}")
    print(f"Mutation rate: {strategy.mutation_rate}")
    print(f"Crossover rate: {strategy.crossover_rate}")
    
    # Simulate architecture search
    print("\nSimulating architecture search...")
    
    # Generate and evaluate initial population
    for i in range(strategy.population_size):
        arch = strategy.generate_architecture()
        # Simulate performance (random for demo)
        performance = np.random.uniform(0.7, 0.95)
        strategy.update(arch, performance)
        
        print(f"  Generation 0, Individual {i+1}: "
              f"{arch.get_num_layers()} layers, "
              f"performance={performance:.4f}")
    
    print(f"\nGeneration 0 summary:")
    gen_summary = strategy.get_generation_summary()
    for key, value in gen_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Generate and evaluate next generation
    print("\nEvolving to generation 1...")
    for i in range(strategy.population_size):
        arch = strategy.generate_architecture()
        performance = np.random.uniform(0.75, 0.97)  # Slightly better
        strategy.update(arch, performance)
        
        print(f"  Generation 1, Individual {i+1}: "
              f"{arch.get_num_layers()} layers, "
              f"performance={performance:.4f}")
    
    print(f"\nGeneration 1 summary:")
    gen_summary = strategy.get_generation_summary()
    for key, value in gen_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Get best architecture
    best_arch = strategy.get_best_architecture()
    print(f"\nBest architecture found:")
    print(f"  ID: {best_arch.id[:8]}...")
    print(f"  Layers: {best_arch.get_num_layers()}")
    print(f"  Performance: {strategy.get_best_performance():.4f}")
    print(f"  Layer types: {[layer.layer_type for layer in best_arch.layers]}")
    
    # Show search history summary
    print(f"\nSearch history summary:")
    history_summary = strategy.get_history_summary()
    for key, value in history_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def demo_rl_search():
    """Demonstrate RL search strategy."""
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING SEARCH STRATEGY DEMO")
    print("="*70)
    
    # Create a vision search space
    input_shape = (28, 28, 1)  # MNIST-like images
    output_shape = (10,)  # 10 classes
    
    search_space = VisionSearchSpace(
        input_shape=input_shape,
        output_shape=output_shape,
        problem_type='classification',
        random_seed=42
    )
    
    # Initialize RL strategy
    try:
        strategy = RLSearchStrategy(
            search_space=search_space,
            controller_hidden_size=64,
            baseline_decay=0.95,
            learning_rate=0.001,
            batch_size=5,
            backend='tensorflow',
            random_seed=42
        )
        
        print(f"\nInitialized: {strategy}")
        print(f"Controller hidden size: {strategy.controller_hidden_size}")
        print(f"Baseline decay: {strategy.baseline_decay}")
        print(f"Batch size: {strategy.batch_size}")
        
        # Simulate architecture search
        print("\nSimulating architecture search...")
        
        for i in range(15):
            arch = strategy.generate_architecture()
            # Simulate performance
            performance = np.random.uniform(0.75, 0.95)
            strategy.update(arch, performance)
            
            print(f"  Architecture {i+1}: "
                  f"{arch.get_num_layers()} layers, "
                  f"performance={performance:.4f}")
            
            # Show controller summary every 5 architectures
            if (i + 1) % 5 == 0:
                controller_summary = strategy.get_controller_summary()
                print(f"\n  Controller summary after {i+1} evaluations:")
                for key, value in controller_summary.items():
                    if isinstance(value, float):
                        print(f"    {key}: {value:.4f}")
                    else:
                        print(f"    {key}: {value}")
                print()
        
        # Get best architecture
        best_arch = strategy.get_best_architecture()
        print(f"\nBest architecture found:")
        print(f"  ID: {best_arch.id[:8]}...")
        print(f"  Layers: {best_arch.get_num_layers()}")
        print(f"  Performance: {strategy.get_best_performance():.4f}")
        
    except ImportError as e:
        print(f"\nSkipping RL demo: {e}")
        print("Install TensorFlow or PyTorch to use RL search strategy")


def demo_darts_search():
    """Demonstrate DARTS search strategy."""
    print("\n" + "="*70)
    print("DARTS (DIFFERENTIABLE ARCHITECTURE SEARCH) DEMO")
    print("="*70)
    
    # Create a time series search space
    input_shape = (50, 5)  # 50 timesteps, 5 features
    output_shape = (1,)  # Single output
    
    search_space = TimeSeriesSearchSpace(
        input_shape=input_shape,
        output_shape=output_shape,
        problem_type='regression',
        random_seed=42
    )
    
    # Initialize DARTS strategy
    try:
        strategy = DARTSSearchStrategy(
            search_space=search_space,
            supernet_epochs=10,
            arch_learning_rate=3e-4,
            weight_learning_rate=0.025,
            backend='tensorflow',
            random_seed=42
        )
        
        print(f"\nInitialized: {strategy}")
        print(f"Supernet epochs: {strategy.supernet_epochs}")
        print(f"Architecture learning rate: {strategy.arch_learning_rate}")
        print(f"Weight learning rate: {strategy.weight_learning_rate}")
        print(f"Candidate operations: {len(strategy.candidate_operations)}")
        
        # Show candidate operations
        print(f"\nCandidate operations:")
        for i, op in enumerate(strategy.candidate_operations, 1):
            print(f"  {i}. {op}")
        
        # Generate sample data for supernet training
        print("\nGenerating sample data for supernet training...")
        X_train = np.random.randn(100, 50, 5)
        y_train = np.random.randn(100, 1)
        X_val = np.random.randn(20, 50, 5)
        y_val = np.random.randn(20, 1)
        
        # Build supernet
        print("Building supernet...")
        strategy.build_supernet(X_train, y_train)
        
        print(f"Supernet built: {strategy.supernet is not None}")
        
        # Extract architectures at different stages
        print("\nExtracting architectures during search...")
        
        for i in range(3):
            arch = strategy.generate_architecture()
            # Simulate performance
            performance = np.random.uniform(0.80, 0.95)
            strategy.update(arch, performance)
            
            print(f"\n  Architecture {i+1}:")
            print(f"    Layers: {arch.get_num_layers()}")
            print(f"    Performance: {performance:.4f}")
            
            if 'selected_operations' in arch.metadata:
                print(f"    Selected operations: {arch.metadata['selected_operations']}")
        
        # Get DARTS summary
        print(f"\nDARTS summary:")
        darts_summary = strategy.get_darts_summary()
        for key, value in darts_summary.items():
            print(f"  {key}: {value}")
        
        # Get best architecture
        best_arch = strategy.get_best_architecture()
        print(f"\nBest architecture found:")
        print(f"  ID: {best_arch.id[:8]}...")
        print(f"  Layers: {best_arch.get_num_layers()}")
        print(f"  Performance: {strategy.get_best_performance():.4f}")
        
    except ImportError as e:
        print(f"\nSkipping DARTS demo: {e}")
        print("Install TensorFlow or PyTorch to use DARTS search strategy")


def compare_strategies():
    """Compare different search strategies."""
    print("\n" + "="*70)
    print("SEARCH STRATEGY COMPARISON")
    print("="*70)
    
    # Create a common search space
    input_shape = (10,)
    output_shape = (2,)
    
    search_space = TabularSearchSpace(
        input_shape=input_shape,
        output_shape=output_shape,
        problem_type='classification',
        random_seed=42
    )
    
    # Initialize all strategies
    strategies = {
        'Evolutionary': EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=5,
            random_seed=42
        ),
    }
    
    # Try to add RL and DARTS if dependencies available
    try:
        strategies['RL'] = RLSearchStrategy(
            search_space=search_space,
            batch_size=5,
            backend='tensorflow',
            random_seed=42
        )
    except ImportError:
        print("\nRL strategy not available (TensorFlow/PyTorch not installed)")
    
    try:
        strategies['DARTS'] = DARTSSearchStrategy(
            search_space=search_space,
            supernet_epochs=5,
            backend='tensorflow',
            random_seed=42
        )
    except ImportError:
        print("\nDARTS strategy not available (TensorFlow/PyTorch not installed)")
    
    # Simulate search for each strategy
    print(f"\nSimulating {10} architecture evaluations for each strategy...")
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"\n{name} Strategy:")
        
        for i in range(10):
            arch = strategy.generate_architecture()
            performance = np.random.uniform(0.7, 0.95)
            strategy.update(arch, performance)
        
        results[name] = {
            'best_performance': strategy.get_best_performance(),
            'mean_performance': strategy.get_history_summary()['mean_performance'],
            'evaluations': len(strategy.history),
        }
        
        print(f"  Best performance: {results[name]['best_performance']:.4f}")
        print(f"  Mean performance: {results[name]['mean_performance']:.4f}")
        print(f"  Total evaluations: {results[name]['evaluations']}")
    
    # Summary table
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Strategy':<15} {'Best':<12} {'Mean':<12} {'Evaluations':<12}")
    print("-" * 70)
    
    for name, result in results.items():
        print(f"{name:<15} "
              f"{result['best_performance']:<12.4f} "
              f"{result['mean_performance']:<12.4f} "
              f"{result['evaluations']:<12}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("NEURAL ARCHITECTURE SEARCH - SEARCH STRATEGIES DEMO")
    print("="*70)
    print("\nThis demo showcases three different search strategies:")
    print("1. Evolutionary Search - Population-based genetic algorithm")
    print("2. RL Search - REINFORCE with LSTM controller")
    print("3. DARTS - Gradient-based differentiable search")
    
    # Run individual demos
    demo_evolutionary_search()
    demo_rl_search()
    demo_darts_search()
    
    # Compare strategies
    compare_strategies()
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey takeaways:")
    print("- Evolutionary: Simple, parallelizable, good for discrete spaces")
    print("- RL: Learns from experience, can discover novel architectures")
    print("- DARTS: Fast, gradient-based, efficient for large search spaces")
    print("\nChoose based on your computational budget and problem requirements!")


if __name__ == '__main__':
    main()

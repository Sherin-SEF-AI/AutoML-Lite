"""
Demo: Multi-Objective Optimization for Neural Architecture Search

This example demonstrates the multi-objective optimization capabilities
of the NAS system, including:
- Pareto dominance checking
- Pareto front calculation
- Objective weighting and scalarization
- Constraint satisfaction
- Pareto front visualization
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

# Import NAS components
from automl_lite.nas import (
    MultiObjectiveOptimizer,
    Objective,
    Architecture,
    LayerConfig,
)


@dataclass
class MockArchitecture:
    """Mock architecture for demonstration."""
    id: str
    metrics: Dict[str, float]
    
    def __repr__(self):
        return f"Arch({self.id})"
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, MockArchitecture):
            return False
        return self.id == other.id


def demo_basic_pareto():
    """Demonstrate basic Pareto dominance and front calculation."""
    print("=" * 80)
    print("DEMO 1: Basic Pareto Dominance and Front Calculation")
    print("=" * 80)
    
    # Define objectives
    objectives = [
        Objective('accuracy', 'maximize', weight=0.6),
        Objective('latency', 'minimize', weight=0.3),
        Objective('model_size', 'minimize', weight=0.1),
    ]
    
    optimizer = MultiObjectiveOptimizer(objectives)
    
    # Create mock architectures with different trade-offs
    architectures = [
        MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100, 'model_size': 20}),
        MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50, 'model_size': 10}),
        MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70, 'model_size': 15}),
        MockArchitecture('A4', {'accuracy': 0.88, 'latency': 120, 'model_size': 25}),  # Dominated
        MockArchitecture('A5', {'accuracy': 0.93, 'latency': 80, 'model_size': 18}),
        MockArchitecture('A6', {'accuracy': 0.85, 'latency': 40, 'model_size': 8}),
    ]
    
    print(f"\nTotal architectures: {len(architectures)}")
    print("\nArchitecture metrics:")
    for arch in architectures:
        print(f"  {arch.id}: {arch.metrics}")
    
    # Check dominance
    print("\n" + "-" * 80)
    print("Dominance relationships:")
    print("-" * 80)
    for i, arch1 in enumerate(architectures):
        for arch2 in architectures[i+1:]:
            if optimizer.dominates(arch1.metrics, arch2.metrics):
                print(f"  {arch1.id} dominates {arch2.id}")
            elif optimizer.dominates(arch2.metrics, arch1.metrics):
                print(f"  {arch2.id} dominates {arch1.id}")
    
    # Compute Pareto front
    pareto_front = optimizer.compute_pareto_front(architectures)
    
    print("\n" + "-" * 80)
    print(f"Pareto Front ({len(pareto_front)} architectures):")
    print("-" * 80)
    for arch in pareto_front:
        print(f"  {arch.id}: {arch.metrics}")
    
    # Compute Pareto ranks
    ranks = optimizer.compute_pareto_rank(architectures)
    print("\n" + "-" * 80)
    print("Pareto Ranks:")
    print("-" * 80)
    for arch in architectures:
        print(f"  {arch.id}: Rank {ranks[arch]}")


def demo_scalarization():
    """Demonstrate objective weighting and scalarization."""
    print("\n\n" + "=" * 80)
    print("DEMO 2: Objective Weighting and Scalarization")
    print("=" * 80)
    
    # Define objectives
    objectives = [
        Objective('accuracy', 'maximize', weight=0.5),
        Objective('latency', 'minimize', weight=0.3),
        Objective('model_size', 'minimize', weight=0.2),
    ]
    
    optimizer = MultiObjectiveOptimizer(objectives)
    
    # Create architectures
    architectures = [
        MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100, 'model_size': 20}),
        MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50, 'model_size': 10}),
        MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70, 'model_size': 15}),
    ]
    
    print("\nOriginal weights:")
    for obj in objectives:
        print(f"  {obj.name}: {obj.weight}")
    
    # Rank with original weights
    print("\n" + "-" * 80)
    print("Ranking with original weights:")
    print("-" * 80)
    ranked = optimizer.rank_architectures(architectures, method='scalarize')
    for i, (arch, score) in enumerate(ranked, 1):
        print(f"  {i}. {arch.id}: score={score:.4f}, metrics={arch.metrics}")
    
    # Change weights to prioritize latency
    print("\n" + "-" * 80)
    print("Changing weights to prioritize latency...")
    print("-" * 80)
    optimizer.set_objective_weights({
        'accuracy': 0.3,
        'latency': 0.6,
        'model_size': 0.1
    })
    
    print("\nNew weights:")
    for name, weight in optimizer.get_objective_weights().items():
        print(f"  {name}: {weight}")
    
    # Rank with new weights
    print("\n" + "-" * 80)
    print("Ranking with new weights:")
    print("-" * 80)
    ranked = optimizer.rank_architectures(architectures, method='scalarize')
    for i, (arch, score) in enumerate(ranked, 1):
        print(f"  {i}. {arch.id}: score={score:.4f}, metrics={arch.metrics}")
    
    # Select best with preferences
    print("\n" + "-" * 80)
    print("Selecting best with custom preferences:")
    print("-" * 80)
    best = optimizer.select_best_architecture(
        architectures,
        preferences={'accuracy': 0.8, 'latency': 0.1, 'model_size': 0.1}
    )
    print(f"  Best architecture: {best.id}")
    print(f"  Metrics: {best.metrics}")


def demo_constraints():
    """Demonstrate constraint satisfaction checking."""
    print("\n\n" + "=" * 80)
    print("DEMO 3: Constraint Satisfaction")
    print("=" * 80)
    
    # Define objectives
    objectives = [
        Objective('accuracy', 'maximize'),
        Objective('latency', 'minimize'),
        Objective('model_size', 'minimize'),
    ]
    
    # Define constraints
    constraints = [
        "accuracy > 0.90",
        "latency < 80",
    ]
    
    optimizer = MultiObjectiveOptimizer(objectives, constraints=constraints)
    
    # Create architectures
    architectures = [
        MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100, 'model_size': 20}),  # Violates latency
        MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50, 'model_size': 10}),   # Violates accuracy (boundary)
        MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70, 'model_size': 15}),   # Satisfies all
        MockArchitecture('A4', {'accuracy': 0.88, 'latency': 60, 'model_size': 12}),   # Violates accuracy
        MockArchitecture('A5', {'accuracy': 0.93, 'latency': 75, 'model_size': 18}),   # Satisfies all
    ]
    
    print(f"\nConstraints:")
    for constraint in constraints:
        print(f"  - {constraint}")
    
    print(f"\nTotal architectures: {len(architectures)}")
    
    # Check each architecture
    print("\n" + "-" * 80)
    print("Constraint checking:")
    print("-" * 80)
    for arch in architectures:
        satisfies = optimizer.check_constraints(arch.metrics)
        status = "✓ PASS" if satisfies else "✗ FAIL"
        print(f"  {arch.id}: {status} - {arch.metrics}")
    
    # Filter by constraints
    valid_architectures = optimizer.filter_by_constraints(architectures)
    
    print("\n" + "-" * 80)
    print(f"Valid architectures ({len(valid_architectures)}):")
    print("-" * 80)
    for arch in valid_architectures:
        print(f"  {arch.id}: {arch.metrics}")
    
    # Add complex constraint
    print("\n" + "-" * 80)
    print("Adding complex constraint: accuracy > 0.92 OR model_size < 12")
    print("-" * 80)
    optimizer.add_constraint("accuracy > 0.92 OR model_size < 12")
    
    valid_architectures = optimizer.filter_by_constraints(architectures)
    print(f"\nValid architectures with new constraint ({len(valid_architectures)}):")
    for arch in valid_architectures:
        print(f"  {arch.id}: {arch.metrics}")


def demo_hypervolume():
    """Demonstrate hypervolume calculation."""
    print("\n\n" + "=" * 80)
    print("DEMO 4: Hypervolume Indicator")
    print("=" * 80)
    
    # Define objectives (2D for simplicity)
    objectives = [
        Objective('accuracy', 'maximize'),
        Objective('latency', 'minimize'),
    ]
    
    optimizer = MultiObjectiveOptimizer(objectives)
    
    # Create two sets of architectures
    set1 = [
        MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),
        MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),
        MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70}),
    ]
    
    set2 = [
        MockArchitecture('B1', {'accuracy': 0.96, 'latency': 90}),
        MockArchitecture('B2', {'accuracy': 0.91, 'latency': 45}),
        MockArchitecture('B3', {'accuracy': 0.93, 'latency': 65}),
    ]
    
    # Compute Pareto fronts
    pareto1 = optimizer.compute_pareto_front(set1)
    pareto2 = optimizer.compute_pareto_front(set2)
    
    print("\nSet 1 Pareto Front:")
    for arch in pareto1:
        print(f"  {arch.id}: {arch.metrics}")
    
    print("\nSet 2 Pareto Front:")
    for arch in pareto2:
        print(f"  {arch.id}: {arch.metrics}")
    
    # Compute hypervolumes
    hv1 = optimizer.compute_hypervolume(pareto1)
    hv2 = optimizer.compute_hypervolume(pareto2)
    
    print("\n" + "-" * 80)
    print("Hypervolume Comparison:")
    print("-" * 80)
    print(f"  Set 1 hypervolume: {hv1:.4f}")
    print(f"  Set 2 hypervolume: {hv2:.4f}")
    print(f"  Improvement: {((hv2 - hv1) / hv1 * 100):.2f}%")


def demo_visualization():
    """Demonstrate Pareto front visualization."""
    print("\n\n" + "=" * 80)
    print("DEMO 5: Pareto Front Visualization")
    print("=" * 80)
    
    # Define objectives
    objectives = [
        Objective('accuracy', 'maximize'),
        Objective('latency', 'minimize'),
    ]
    
    optimizer = MultiObjectiveOptimizer(objectives)
    
    # Generate random architectures
    np.random.seed(42)
    n_archs = 50
    architectures = []
    
    for i in range(n_archs):
        # Create trade-off: higher accuracy -> higher latency
        accuracy = np.random.uniform(0.80, 0.98)
        latency = 30 + (accuracy - 0.80) * 300 + np.random.normal(0, 20)
        latency = max(20, latency)  # Ensure positive
        
        architectures.append(
            MockArchitecture(f'A{i+1}', {'accuracy': accuracy, 'latency': latency})
        )
    
    print(f"\nGenerated {len(architectures)} random architectures")
    
    # Compute Pareto front
    pareto_front = optimizer.compute_pareto_front(architectures)
    print(f"Pareto front contains {len(pareto_front)} architectures")
    
    # Try to visualize (will work if matplotlib/plotly is installed)
    print("\n" + "-" * 80)
    print("Attempting to create visualization...")
    print("-" * 80)
    
    try:
        # Try matplotlib
        fig = optimizer.visualize_pareto_front(
            architectures,
            pareto_front=pareto_front,
            backend='matplotlib',
            save_path='pareto_front_2d.png',
            show=False
        )
        if fig:
            print("✓ Saved matplotlib visualization to: pareto_front_2d.png")
    except Exception as e:
        print(f"✗ Matplotlib visualization failed: {e}")
    
    try:
        # Try plotly
        fig = optimizer.visualize_pareto_front(
            architectures,
            pareto_front=pareto_front,
            backend='plotly',
            save_path='pareto_front_2d.html',
            show=False
        )
        if fig:
            print("✓ Saved plotly visualization to: pareto_front_2d.html")
    except Exception as e:
        print(f"✗ Plotly visualization failed: {e}")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "Multi-Objective Optimization Demo" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    demo_basic_pareto()
    demo_scalarization()
    demo_constraints()
    demo_hypervolume()
    demo_visualization()
    
    print("\n\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()

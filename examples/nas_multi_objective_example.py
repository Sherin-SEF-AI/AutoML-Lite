"""
Multi-Objective Neural Architecture Search Example

This example demonstrates multi-objective NAS with Pareto front exploration.
It shows how to balance multiple competing objectives and select architectures
based on different preferences.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from automl_lite import AutoMLite
from automl_lite.nas import NASConfig, MultiObjectiveOptimizer

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Multi-Objective Neural Architecture Search")
print("=" * 80)

# Generate dataset
print("\n1. Generating dataset...")
X, y = make_classification(
    n_samples=3000,
    n_features=40,
    n_informative=30,
    n_classes=4,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Dataset: {X_train.shape[0]} samples, {X_train.shape[1]} features, {len(np.unique(y))} classes")

# Configure multi-objective NAS
print("\n2. Configuring Multi-Objective NAS...")
config = NASConfig(
    # Search settings
    search_strategy='evolutionary',
    search_space_type='tabular',
    time_budget=1200,  # 20 minutes
    max_architectures=80,
    
    # Multi-objective optimization
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size'],
    
    # No weights = pure Pareto optimization
    # (will find all non-dominated solutions)
    objective_weights=None,
    
    # Hardware profiling for latency/size metrics
    enable_hardware_aware=True,
    target_hardware='cpu',
    
    # Performance estimation
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.15,
    
    verbose=True
)

print(f"\n   Optimization Setup:")
print(f"   - Objectives: {', '.join(config.objectives)}")
print(f"   - Strategy: Pure Pareto optimization (no weights)")
print(f"   - Time budget: {config.time_budget/60:.0f} minutes")

# Run multi-objective NAS
print("\n3. Running Multi-Objective NAS...")
print("   This will take approximately 20 minutes...")
print("   " + "-" * 76)

automl = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_config=config,
    verbose=True
)

automl.fit(X_train, y_train)

print("   " + "-" * 76)
print("   Search complete!")

# Analyze Pareto front
print("\n4. Analyzing Pareto Front...")
nas_result = automl.nas_result
pareto_front = nas_result.pareto_front

print(f"\n   Search Statistics:")
print(f"   - Total architectures evaluated: {nas_result.total_architectures_evaluated}")
print(f"   - Pareto front size: {len(pareto_front)}")
print(f"   - Search time: {nas_result.search_time/60:.1f} minutes")

# Display Pareto front
print(f"\n   Pareto Front Architectures:")
print(f"   {'#':<4} {'Arch ID':<15} {'Accuracy':<12} {'Latency (ms)':<15} {'Size (MB)':<12} {'Layers':<8}")
print("   " + "-" * 76)

for i, arch in enumerate(pareto_front, 1):
    print(f"   {i:<4} {arch.id[:13]:<15} "
          f"{arch.metadata.get('accuracy', 0):.4f}      "
          f"{arch.metadata.get('latency', 0):<15.1f} "
          f"{arch.metadata.get('model_size', 0):<12.1f} "
          f"{len(arch.layers):<8}")

# Analyze objective ranges
print("\n5. Objective Ranges in Pareto Front...")
accuracies = [a.metadata.get('accuracy', 0) for a in pareto_front]
latencies = [a.metadata.get('latency', 0) for a in pareto_front]
sizes = [a.metadata.get('model_size', 0) for a in pareto_front]

print(f"\n   Accuracy:")
print(f"   - Min: {min(accuracies):.4f}")
print(f"   - Max: {max(accuracies):.4f}")
print(f"   - Range: {max(accuracies) - min(accuracies):.4f}")

print(f"\n   Latency:")
print(f"   - Min: {min(latencies):.1f}ms")
print(f"   - Max: {max(latencies):.1f}ms")
print(f"   - Range: {max(latencies) - min(latencies):.1f}ms")

print(f"\n   Model Size:")
print(f"   - Min: {min(sizes):.1f}MB")
print(f"   - Max: {max(sizes):.1f}MB")
print(f"   - Range: {max(sizes) - min(sizes):.1f}MB")

# Select architectures based on different preferences
print("\n6. Selecting Architectures for Different Scenarios...")

optimizer = MultiObjectiveOptimizer(objectives=config.objectives)

# Scenario 1: Prioritize accuracy (research/offline)
print("\n   Scenario 1: Research/Offline (Accuracy Priority)")
preferences_research = {
    'accuracy': 0.8,
    'latency': 0.1,
    'model_size': 0.1
}
arch_research = optimizer.select_best_architecture(pareto_front, preferences_research)
print(f"   - Selected: {arch_research.id[:20]}")
print(f"   - Accuracy: {arch_research.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {arch_research.metadata.get('latency', 0):.1f}ms")
print(f"   - Size: {arch_research.metadata.get('model_size', 0):.1f}MB")

# Scenario 2: Balance all objectives (production)
print("\n   Scenario 2: Production (Balanced)")
preferences_production = {
    'accuracy': 0.5,
    'latency': 0.3,
    'model_size': 0.2
}
arch_production = optimizer.select_best_architecture(pareto_front, preferences_production)
print(f"   - Selected: {arch_production.id[:20]}")
print(f"   - Accuracy: {arch_production.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {arch_production.metadata.get('latency', 0):.1f}ms")
print(f"   - Size: {arch_production.metadata.get('model_size', 0):.1f}MB")

# Scenario 3: Prioritize efficiency (mobile/edge)
print("\n   Scenario 3: Mobile/Edge (Efficiency Priority)")
preferences_mobile = {
    'accuracy': 0.3,
    'latency': 0.4,
    'model_size': 0.3
}
arch_mobile = optimizer.select_best_architecture(pareto_front, preferences_mobile)
print(f"   - Selected: {arch_mobile.id[:20]}")
print(f"   - Accuracy: {arch_mobile.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {arch_mobile.metadata.get('latency', 0):.1f}ms")
print(f"   - Size: {arch_mobile.metadata.get('model_size', 0):.1f}MB")

# Scenario 4: Real-time applications (latency critical)
print("\n   Scenario 4: Real-time (Latency Critical)")
preferences_realtime = {
    'accuracy': 0.3,
    'latency': 0.6,
    'model_size': 0.1
}
arch_realtime = optimizer.select_best_architecture(pareto_front, preferences_realtime)
print(f"   - Selected: {arch_realtime.id[:20]}")
print(f"   - Accuracy: {arch_realtime.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {arch_realtime.metadata.get('latency', 0):.1f}ms")
print(f"   - Size: {arch_realtime.metadata.get('model_size', 0):.1f}MB")

# Compare scenarios
print("\n7. Scenario Comparison...")
print(f"\n   {'Scenario':<20} {'Accuracy':<12} {'Latency (ms)':<15} {'Size (MB)':<12}")
print("   " + "-" * 76)

scenarios = [
    ("Research/Offline", arch_research),
    ("Production", arch_production),
    ("Mobile/Edge", arch_mobile),
    ("Real-time", arch_realtime)
]

for name, arch in scenarios:
    print(f"   {name:<20} "
          f"{arch.metadata.get('accuracy', 0):.4f}      "
          f"{arch.metadata.get('latency', 0):<15.1f} "
          f"{arch.metadata.get('model_size', 0):<12.1f}")

# Analyze trade-offs
print("\n8. Trade-off Analysis...")

# Accuracy vs Latency
print("\n   Accuracy vs Latency Trade-off:")
sorted_by_acc = sorted(pareto_front, key=lambda a: a.metadata.get('accuracy', 0), reverse=True)
print(f"   - Highest accuracy: {sorted_by_acc[0].metadata.get('accuracy', 0):.4f} "
      f"(latency: {sorted_by_acc[0].metadata.get('latency', 0):.1f}ms)")
print(f"   - Lowest accuracy: {sorted_by_acc[-1].metadata.get('accuracy', 0):.4f} "
      f"(latency: {sorted_by_acc[-1].metadata.get('latency', 0):.1f}ms)")
print(f"   - Accuracy sacrifice for speed: {sorted_by_acc[0].metadata.get('accuracy', 0) - sorted_by_acc[-1].metadata.get('accuracy', 0):.4f}")

# Accuracy vs Size
print("\n   Accuracy vs Model Size Trade-off:")
sorted_by_size = sorted(pareto_front, key=lambda a: a.metadata.get('model_size', 0))
print(f"   - Smallest model: {sorted_by_size[0].metadata.get('model_size', 0):.1f}MB "
      f"(accuracy: {sorted_by_size[0].metadata.get('accuracy', 0):.4f})")
print(f"   - Largest model: {sorted_by_size[-1].metadata.get('model_size', 0):.1f}MB "
      f"(accuracy: {sorted_by_size[-1].metadata.get('accuracy', 0):.4f})")

# Evaluate selected architectures
print("\n9. Test Set Evaluation...")
print(f"\n   {'Scenario':<20} {'Test Accuracy':<15}")
print("   " + "-" * 40)

# Note: In practice, you would rebuild and evaluate each architecture
# For this demo, we use the best architecture from AutoMLite
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"   {'Production (used)':<20} {test_accuracy:.4f}")

# Visualize Pareto front
print("\n10. Generating Visualizations...")
try:
    from automl_lite.nas import visualize_pareto_front
    
    # 2D plots
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='accuracy',
        save_path='pareto_multi_obj_acc_lat.html'
    )
    print("   - Saved: pareto_multi_obj_acc_lat.html")
    
    visualize_pareto_front(
        pareto_front,
        x_objective='model_size',
        y_objective='accuracy',
        save_path='pareto_multi_obj_acc_size.html'
    )
    print("   - Saved: pareto_multi_obj_acc_size.html")
    
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='model_size',
        save_path='pareto_multi_obj_lat_size.html'
    )
    print("   - Saved: pareto_multi_obj_lat_size.html")
    
    # 3D plot
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='model_size',
        z_objective='accuracy',
        save_path='pareto_multi_obj_3d.html'
    )
    print("   - Saved: pareto_multi_obj_3d.html")
    
except Exception as e:
    print(f"   Visualization skipped: {e}")

# Save results
print("\n11. Saving Results...")
automl.save_model('nas_multi_objective_model.pkl')
print("   - Model saved: nas_multi_objective_model.pkl")

automl.generate_report(save_path='nas_multi_objective_report.html')
print("   - Report saved: nas_multi_objective_report.html")

# Export Pareto front
print("\n   Exporting Pareto front...")
pareto_data = []
for arch in pareto_front:
    pareto_data.append({
        'architecture_id': arch.id,
        'accuracy': arch.metadata.get('accuracy', 0),
        'latency_ms': arch.metadata.get('latency', 0),
        'model_size_mb': arch.metadata.get('model_size', 0),
        'num_layers': len(arch.layers)
    })

df_pareto = pd.DataFrame(pareto_data)
df_pareto.to_csv('pareto_front.csv', index=False)
print("   - Pareto front saved: pareto_front.csv")

print("\n" + "=" * 80)
print("Multi-Objective NAS Complete!")
print("=" * 80)

print("\nKey Insights:")
print(f"- Found {len(pareto_front)} non-dominated architectures")
print("- Each architecture represents a different trade-off")
print("- Different scenarios benefit from different architectures")
print("- Pareto front provides flexibility for deployment decisions")

print("\nDecision Guide:")
print("┌─────────────────────────────────────────────────────────────┐")
print("│ If you prioritize...        │ Choose architecture for...    │")
print("├─────────────────────────────────────────────────────────────┤")
print("│ Highest accuracy            │ Research/Offline scenario     │")
print("│ Balanced performance        │ Production scenario           │")
print("│ Efficiency (size + latency) │ Mobile/Edge scenario          │")
print("│ Fastest inference           │ Real-time scenario            │")
print("└─────────────────────────────────────────────────────────────┘")

print("\nNext Steps:")
print("- Explore interactive Pareto front visualizations (HTML files)")
print("- Test selected architectures on actual deployment hardware")
print("- Consider A/B testing different architectures in production")
print("- Use transfer learning to warm-start future searches")
print("- Experiment with different objective weights")

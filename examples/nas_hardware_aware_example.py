"""
Hardware-Aware Neural Architecture Search Example

This example demonstrates NAS with hardware constraints for mobile deployment.
It shows how to optimize architectures for latency, memory, and model size.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from automl_lite import AutoMLite
from automl_lite.nas import NASConfig, get_config_template
from automl_lite.nas import HardwareProfiler

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Hardware-Aware Neural Architecture Search for Mobile Deployment")
print("=" * 80)

# Generate dataset
print("\n1. Generating dataset...")
X, y = make_classification(
    n_samples=2000,
    n_features=30,
    n_informative=20,
    n_classes=2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Dataset: {X_train.shape[0]} training samples, {X_train.shape[1]} features")

# Configure hardware-aware NAS for mobile
print("\n2. Configuring Hardware-Aware NAS for Mobile...")
config = NASConfig(
    # Search settings
    search_strategy='evolutionary',
    search_space_type='tabular',
    time_budget=900,  # 15 minutes
    max_architectures=50,
    
    # Hardware-aware settings
    enable_hardware_aware=True,
    target_hardware='mobile',
    
    # Mobile constraints
    max_latency_ms=100,      # 100ms inference time
    max_memory_mb=50,        # 50MB peak memory
    max_model_size_mb=10,    # 10MB model size
    
    # Multi-objective optimization
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size'],
    objective_weights={
        'accuracy': 0.5,      # 50% weight on accuracy
        'latency': 0.3,       # 30% weight on latency
        'model_size': 0.2     # 20% weight on model size
    },
    
    # Performance estimation
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.1,
    
    # Checkpointing
    enable_checkpointing=True,
    checkpoint_path='./nas_hardware_checkpoint.pkl',
    
    verbose=True
)

print(f"\n   Hardware Constraints:")
print(f"   - Target: {config.target_hardware}")
print(f"   - Max latency: {config.max_latency_ms}ms")
print(f"   - Max memory: {config.max_memory_mb}MB")
print(f"   - Max model size: {config.max_model_size_mb}MB")

print(f"\n   Optimization Objectives:")
for obj, weight in config.objective_weights.items():
    print(f"   - {obj}: {weight*100:.0f}% weight")

# Alternative: Use pre-configured template
print("\n   Alternative: Using pre-configured template...")
template_config = get_config_template('hardware_aware_mobile')
print(f"   Template loaded: {template_config.search_strategy} strategy")

# Create AutoMLite with hardware-aware NAS
print("\n3. Running Hardware-Aware NAS...")
print("   This will take approximately 15 minutes...")
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

# Analyze results
print("\n4. Analyzing Hardware-Aware Results...")
nas_result = automl.nas_result

print(f"\n   Search Statistics:")
print(f"   - Architectures evaluated: {nas_result.total_architectures_evaluated}")
print(f"   - Search time: {nas_result.search_time/60:.1f} minutes")

# Best architecture
best_arch = nas_result.best_architecture
print(f"\n   Best Architecture (balanced):")
print(f"   - Accuracy: {best_arch.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {best_arch.metadata.get('latency', 0):.1f}ms")
print(f"   - Memory: {best_arch.metadata.get('memory', 0):.1f}MB")
print(f"   - Model size: {best_arch.metadata.get('model_size', 0):.1f}MB")
print(f"   - Layers: {len(best_arch.layers)}")

# Check constraint satisfaction
print(f"\n   Constraint Satisfaction:")
print(f"   - Latency: {'✓' if best_arch.metadata.get('latency', 0) <= config.max_latency_ms else '✗'} "
      f"({best_arch.metadata.get('latency', 0):.1f}ms / {config.max_latency_ms}ms)")
print(f"   - Memory: {'✓' if best_arch.metadata.get('memory', 0) <= config.max_memory_mb else '✗'} "
      f"({best_arch.metadata.get('memory', 0):.1f}MB / {config.max_memory_mb}MB)")
print(f"   - Model size: {'✓' if best_arch.metadata.get('model_size', 0) <= config.max_model_size_mb else '✗'} "
      f"({best_arch.metadata.get('model_size', 0):.1f}MB / {config.max_model_size_mb}MB)")

# Pareto front analysis
print("\n5. Pareto Front Analysis...")
pareto_front = nas_result.pareto_front
print(f"   Pareto front contains {len(pareto_front)} non-dominated architectures")

print(f"\n   {'Arch ID':<15} {'Accuracy':<12} {'Latency (ms)':<15} {'Size (MB)':<12} {'Layers':<8}")
print("   " + "-" * 76)

for arch in pareto_front[:10]:  # Show top 10
    print(f"   {arch.id[:13]:<15} "
          f"{arch.metadata.get('accuracy', 0):.4f}      "
          f"{arch.metadata.get('latency', 0):<15.1f} "
          f"{arch.metadata.get('model_size', 0):<12.1f} "
          f"{len(arch.layers):<8}")

# Find specific trade-offs
print("\n6. Exploring Trade-offs...")

# Highest accuracy (may sacrifice efficiency)
highest_acc_arch = max(pareto_front, key=lambda a: a.metadata.get('accuracy', 0))
print(f"\n   Highest Accuracy Architecture:")
print(f"   - Accuracy: {highest_acc_arch.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {highest_acc_arch.metadata.get('latency', 0):.1f}ms")
print(f"   - Model size: {highest_acc_arch.metadata.get('model_size', 0):.1f}MB")

# Lowest latency (fastest inference)
lowest_lat_arch = min(pareto_front, key=lambda a: a.metadata.get('latency', float('inf')))
print(f"\n   Lowest Latency Architecture:")
print(f"   - Accuracy: {lowest_lat_arch.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {lowest_lat_arch.metadata.get('latency', 0):.1f}ms")
print(f"   - Model size: {lowest_lat_arch.metadata.get('model_size', 0):.1f}MB")

# Smallest model (minimal storage)
smallest_arch = min(pareto_front, key=lambda a: a.metadata.get('model_size', float('inf')))
print(f"\n   Smallest Model Architecture:")
print(f"   - Accuracy: {smallest_arch.metadata.get('accuracy', 0):.4f}")
print(f"   - Latency: {smallest_arch.metadata.get('latency', 0):.1f}ms")
print(f"   - Model size: {smallest_arch.metadata.get('model_size', 0):.1f}MB")

# Hardware profiling demonstration
print("\n7. Hardware Profiling Demonstration...")
profiler = HardwareProfiler(target_hardware='mobile')

print(f"\n   Profiling best architecture on different hardware:")

for hardware in ['mobile', 'edge', 'cpu', 'gpu']:
    profiler_hw = HardwareProfiler(target_hardware=hardware)
    latency = profiler_hw.estimate_latency(best_arch, batch_size=1)
    memory = profiler_hw.estimate_memory(best_arch, batch_size=1)
    
    print(f"   - {hardware.upper():<8}: Latency={latency:>6.1f}ms, Memory={memory:>6.1f}MB")

# Evaluate on test set
print("\n8. Test Set Evaluation...")
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"   Test Accuracy: {test_accuracy:.4f}")

# Visualize Pareto front
print("\n9. Generating Visualizations...")
try:
    from automl_lite.nas import visualize_pareto_front
    
    # 2D plot: accuracy vs latency
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='accuracy',
        save_path='pareto_accuracy_latency.html'
    )
    print("   - Saved: pareto_accuracy_latency.html")
    
    # 2D plot: latency vs model size
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='model_size',
        save_path='pareto_latency_size.html'
    )
    print("   - Saved: pareto_latency_size.html")
    
    # 3D plot: all three objectives
    visualize_pareto_front(
        pareto_front,
        x_objective='latency',
        y_objective='model_size',
        z_objective='accuracy',
        save_path='pareto_3d.html'
    )
    print("   - Saved: pareto_3d.html")
    
except Exception as e:
    print(f"   Visualization skipped: {e}")

# Save model
print("\n10. Saving Results...")
automl.save_model('nas_hardware_aware_model.pkl')
print("   - Model saved: nas_hardware_aware_model.pkl")

automl.generate_report(save_path='nas_hardware_aware_report.html')
print("   - Report saved: nas_hardware_aware_report.html")

print("\n" + "=" * 80)
print("Hardware-Aware NAS Complete!")
print("=" * 80)

print("\nKey Insights:")
print("- NAS found architectures that meet mobile deployment constraints")
print("- Pareto front shows trade-offs between accuracy, latency, and size")
print("- Multiple architectures available for different deployment scenarios")
print("- Hardware profiling helps estimate performance on target devices")

print("\nDeployment Recommendations:")
if best_arch.metadata.get('latency', 0) <= 50:
    print("✓ Suitable for real-time mobile applications")
elif best_arch.metadata.get('latency', 0) <= 100:
    print("✓ Suitable for interactive mobile applications")
else:
    print("⚠ May need optimization for real-time use")

if best_arch.metadata.get('model_size', 0) <= 5:
    print("✓ Excellent for resource-constrained devices")
elif best_arch.metadata.get('model_size', 0) <= 10:
    print("✓ Good for mobile deployment")
else:
    print("⚠ Consider model compression techniques")

print("\nNext Steps:")
print("- Test on actual mobile device for validation")
print("- Consider quantization for further optimization")
print("- Explore edge device deployment (see config.target_hardware='edge')")
print("- Try different objective weights based on priorities")

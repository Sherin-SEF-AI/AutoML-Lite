"""
Basic Neural Architecture Search Example

This example demonstrates basic NAS usage on a tabular classification dataset.
It shows how to enable NAS, run a search, and analyze the results.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from automl_lite import AutoMLite
from automl_lite.nas import NASConfig

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("Basic Neural Architecture Search Example")
print("=" * 80)

# Generate synthetic tabular dataset
print("\n1. Generating synthetic dataset...")
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=3,
    random_state=42
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")
print(f"   Features: {X_train.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# Configure NAS with basic settings
print("\n2. Configuring Neural Architecture Search...")
config = NASConfig(
    search_strategy='evolutionary',
    search_space_type='tabular',
    time_budget=600,  # 10 minutes for demo
    max_architectures=30,
    
    # Performance estimation
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.1,
    early_stopping_patience=3,
    
    # Checkpointing
    enable_checkpointing=True,
    checkpoint_path='./nas_basic_checkpoint.pkl',
    
    # Logging
    verbose=True
)

print(f"   Search strategy: {config.search_strategy}")
print(f"   Time budget: {config.time_budget}s ({config.time_budget/60:.1f} minutes)")
print(f"   Max architectures: {config.max_architectures}")

# Create AutoMLite instance with NAS enabled
print("\n3. Creating AutoMLite with NAS enabled...")
automl = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_config=config,
    verbose=True
)

# Run NAS
print("\n4. Running Neural Architecture Search...")
print("   This will take approximately 10 minutes...")
print("   " + "-" * 76)

automl.fit(X_train, y_train)

print("   " + "-" * 76)
print("   Search complete!")

# Analyze results
print("\n5. Analyzing NAS Results...")
nas_result = automl.nas_result

print(f"\n   Search Statistics:")
print(f"   - Total architectures evaluated: {nas_result.total_architectures_evaluated}")
print(f"   - Search time: {nas_result.search_time:.1f}s ({nas_result.search_time/60:.1f} minutes)")
print(f"   - Best validation accuracy: {nas_result.best_accuracy:.4f}")

# Get best architecture details
best_arch = nas_result.best_architecture
print(f"\n   Best Architecture:")
print(f"   - ID: {best_arch.id}")
print(f"   - Number of layers: {len(best_arch.layers)}")
print(f"   - Layer types: {[layer.layer_type for layer in best_arch.layers]}")

# Show layer details
print(f"\n   Layer Details:")
for i, layer in enumerate(best_arch.layers):
    print(f"   - Layer {i+1}: {layer.layer_type}")
    if layer.layer_type == 'dense':
        print(f"     Units: {layer.params.get('units', 'N/A')}")
        print(f"     Activation: {layer.params.get('activation', 'N/A')}")
    elif layer.layer_type == 'dropout':
        print(f"     Rate: {layer.params.get('rate', 'N/A')}")

# Evaluate on test set
print("\n6. Evaluating on test set...")
y_pred = automl.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n   Test Accuracy: {test_accuracy:.4f}")
print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in range(3)]))

# Show top 5 architectures
print("\n7. Top 5 Architectures:")
print(f"   {'Rank':<6} {'Architecture ID':<20} {'Accuracy':<12} {'Layers':<8}")
print("   " + "-" * 76)

all_archs = sorted(
    nas_result.all_architectures,
    key=lambda a: a.metadata.get('accuracy', 0),
    reverse=True
)[:5]

for rank, arch in enumerate(all_archs, 1):
    print(f"   {rank:<6} {arch.id:<20} {arch.metadata.get('accuracy', 0):.4f}      {len(arch.layers):<8}")

# Save model
print("\n8. Saving model...")
model_path = 'nas_basic_model.pkl'
automl.save_model(model_path)
print(f"   Model saved to: {model_path}")

# Generate report
print("\n9. Generating HTML report...")
report_path = 'nas_basic_report.html'
automl.generate_report(save_path=report_path)
print(f"   Report saved to: {report_path}")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)
print("\nKey Takeaways:")
print("- NAS automatically discovered an optimal architecture for the dataset")
print("- The search evaluated multiple architectures within the time budget")
print("- The best architecture achieved good test accuracy")
print("- Results are saved and can be reused")
print("\nNext Steps:")
print("- Try different search strategies ('rl', 'darts')")
print("- Increase time_budget for better results")
print("- Explore hardware-aware NAS (see nas_hardware_aware_example.py)")
print("- Try multi-objective optimization (see nas_multi_objective_example.py)")

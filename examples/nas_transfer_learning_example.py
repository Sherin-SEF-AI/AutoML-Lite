"""
Transfer Learning Neural Architecture Search Example

This example demonstrates how to use transfer learning in NAS to:
1. Reuse successful architectures from previous searches
2. Reduce search time by warm-starting with similar architectures
3. Build an organizational architecture repository
4. Adapt architectures to new problems
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

from automl_lite import AutoMLite
from automl_lite.nas import NASConfig, ArchitectureRepository

# Set random seed
np.random.seed(42)

print("=" * 80)
print("Transfer Learning Neural Architecture Search")
print("=" * 80)

# Initialize architecture repository
print("\n1. Initializing Architecture Repository...")
repo = ArchitectureRepository(db_path='./nas_architecture_repo.db')
print(f"   Repository: ./nas_architecture_repo.db")

# Check existing architectures
try:
    existing_count = len(repo.list_all_architectures())
    print(f"   Existing architectures: {existing_count}")
except:
    print(f"   Existing architectures: 0 (new repository)")

# Scenario 1: Initial search (no transfer learning)
print("\n" + "=" * 80)
print("SCENARIO 1: Initial Search (No Transfer Learning)")
print("=" * 80)

print("\n2. Generating first dataset...")
X1, y1 = make_classification(
    n_samples=2000,
    n_features=25,
    n_informative=20,
    n_classes=3,
    random_state=42
)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, test_size=0.2, random_state=42
)

print(f"   Dataset 1: {X1_train.shape[0]} samples, {X1_train.shape[1]} features, {len(np.unique(y1))} classes")

# Configure NAS without transfer learning
print("\n3. Running NAS without transfer learning...")
config_no_transfer = NASConfig(
    search_strategy='evolutionary',
    search_space_type='tabular',
    time_budget=600,  # 10 minutes
    max_architectures=30,
    
    # Disable transfer learning for baseline
    enable_transfer_learning=False,
    
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.1,
    verbose=False  # Less verbose for demo
)

print(f"   Time budget: {config_no_transfer.time_budget}s")
print(f"   Transfer learning: {config_no_transfer.enable_transfer_learning}")

start_time = time.time()

automl1 = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_config=config_no_transfer,
    verbose=False
)

automl1.fit(X1_train, y1_train)

search_time_no_transfer = time.time() - start_time

print(f"\n   Results:")
print(f"   - Search time: {search_time_no_transfer:.1f}s ({search_time_no_transfer/60:.1f} minutes)")
print(f"   - Architectures evaluated: {automl1.nas_result.total_architectures_evaluated}")
print(f"   - Best accuracy: {automl1.nas_result.best_accuracy:.4f}")

# Save best architecture to repository
print("\n4. Saving architecture to repository...")
best_arch1 = automl1.nas_result.best_architecture

metadata1 = {
    'dataset_name': 'dataset_1',
    'n_samples': X1_train.shape[0],
    'n_features': X1_train.shape[1],
    'problem_type': 'classification',
    'n_classes': len(np.unique(y1)),
    'accuracy': automl1.nas_result.best_accuracy,
    'search_time': search_time_no_transfer
}

arch_id1 = repo.save_architecture(
    best_arch1,
    metadata=metadata1,
    tags=['baseline', 'classification', 'tabular']
)

print(f"   Architecture saved with ID: {arch_id1}")
print(f"   Metadata: {metadata1}")

# Scenario 2: Similar problem with transfer learning
print("\n" + "=" * 80)
print("SCENARIO 2: Similar Problem (With Transfer Learning)")
print("=" * 80)

print("\n5. Generating similar dataset...")
X2, y2 = make_classification(
    n_samples=2200,  # Slightly different size
    n_features=28,   # Slightly different features
    n_informative=22,
    n_classes=3,     # Same number of classes
    random_state=43
)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

print(f"   Dataset 2: {X2_train.shape[0]} samples, {X2_train.shape[1]} features, {len(np.unique(y2))} classes")

# Find similar architectures
print("\n6. Finding similar architectures...")
dataset2_metadata = {
    'n_samples': X2_train.shape[0],
    'n_features': X2_train.shape[1],
    'problem_type': 'classification',
    'n_classes': len(np.unique(y2))
}

similar_archs = repo.find_similar_architectures(dataset2_metadata, top_k=3)
print(f"   Found {len(similar_archs)} similar architectures")

if similar_archs:
    for i, arch in enumerate(similar_archs, 1):
        print(f"   {i}. Architecture: {arch.id[:20]}")
        print(f"      Similarity score: {arch.metadata.get('similarity_score', 'N/A')}")
        print(f"      Original accuracy: {arch.metadata.get('accuracy', 'N/A'):.4f}")

# Configure NAS with transfer learning
print("\n7. Running NAS with transfer learning...")
config_with_transfer = NASConfig(
    search_strategy='evolutionary',
    search_space_type='tabular',
    time_budget=600,  # Same time budget
    max_architectures=30,
    
    # Enable transfer learning
    enable_transfer_learning=True,
    architecture_repository_path='./nas_architecture_repo.db',
    
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.1,
    verbose=False
)

print(f"   Time budget: {config_with_transfer.time_budget}s")
print(f"   Transfer learning: {config_with_transfer.enable_transfer_learning}")

start_time = time.time()

automl2 = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_config=config_with_transfer,
    verbose=False
)

automl2.fit(X2_train, y2_train)

search_time_with_transfer = time.time() - start_time

print(f"\n   Results:")
print(f"   - Search time: {search_time_with_transfer:.1f}s ({search_time_with_transfer/60:.1f} minutes)")
print(f"   - Architectures evaluated: {automl2.nas_result.total_architectures_evaluated}")
print(f"   - Best accuracy: {automl2.nas_result.best_accuracy:.4f}")

# Compare results
print("\n8. Comparing Transfer Learning Impact...")
print(f"\n   {'Metric':<30} {'No Transfer':<20} {'With Transfer':<20} {'Improvement':<15}")
print("   " + "-" * 85)

time_saved = search_time_no_transfer - search_time_with_transfer
time_saved_pct = (time_saved / search_time_no_transfer) * 100

print(f"   {'Search time (s)':<30} {search_time_no_transfer:<20.1f} {search_time_with_transfer:<20.1f} "
      f"{time_saved_pct:>6.1f}%")

archs_diff = automl2.nas_result.total_architectures_evaluated - automl1.nas_result.total_architectures_evaluated
print(f"   {'Architectures evaluated':<30} {automl1.nas_result.total_architectures_evaluated:<20} "
      f"{automl2.nas_result.total_architectures_evaluated:<20} {archs_diff:>+6}")

acc_diff = automl2.nas_result.best_accuracy - automl1.nas_result.best_accuracy
print(f"   {'Best accuracy':<30} {automl1.nas_result.best_accuracy:<20.4f} "
      f"{automl2.nas_result.best_accuracy:<20.4f} {acc_diff:>+6.4f}")

# Save second architecture
print("\n9. Saving second architecture to repository...")
best_arch2 = automl2.nas_result.best_architecture

metadata2 = {
    'dataset_name': 'dataset_2',
    'n_samples': X2_train.shape[0],
    'n_features': X2_train.shape[1],
    'problem_type': 'classification',
    'n_classes': len(np.unique(y2)),
    'accuracy': automl2.nas_result.best_accuracy,
    'search_time': search_time_with_transfer,
    'used_transfer_learning': True
}

arch_id2 = repo.save_architecture(
    best_arch2,
    metadata=metadata2,
    tags=['transfer_learning', 'classification', 'tabular']
)

print(f"   Architecture saved with ID: {arch_id2}")

# Demonstrate architecture adaptation
print("\n" + "=" * 80)
print("SCENARIO 3: Architecture Adaptation")
print("=" * 80)

print("\n10. Adapting architecture to different problem...")

# Create a problem with different dimensions
X3, y3 = make_classification(
    n_samples=1500,
    n_features=50,   # Different number of features
    n_informative=40,
    n_classes=5,     # Different number of classes
    random_state=44
)

print(f"   New problem: {X3.shape[1]} features, {len(np.unique(y3))} classes")

# Adapt the best architecture from dataset 1
adapted_arch = repo.adapt_architecture(
    best_arch1,
    new_input_shape=(X3.shape[1],),
    new_output_shape=(len(np.unique(y3)),)
)

print(f"\n   Original architecture:")
print(f"   - Input shape: {best_arch1.layers[0].input_shape}")
print(f"   - Output shape: {best_arch1.layers[-1].output_shape}")
print(f"   - Layers: {len(best_arch1.layers)}")

print(f"\n   Adapted architecture:")
print(f"   - Input shape: {adapted_arch.layers[0].input_shape}")
print(f"   - Output shape: {adapted_arch.layers[-1].output_shape}")
print(f"   - Layers: {len(adapted_arch.layers)}")

# Repository management
print("\n" + "=" * 80)
print("Repository Management")
print("=" * 80)

print("\n11. Exploring repository contents...")

# List all architectures
all_archs = repo.list_all_architectures()
print(f"\n   Total architectures in repository: {len(all_archs)}")

print(f"\n   {'Arch ID':<25} {'Dataset':<15} {'Features':<10} {'Classes':<10} {'Accuracy':<12}")
print("   " + "-" * 85)

for arch_info in all_archs:
    print(f"   {arch_info['id'][:23]:<25} "
          f"{arch_info.get('dataset_name', 'N/A'):<15} "
          f"{arch_info.get('n_features', 'N/A'):<10} "
          f"{arch_info.get('n_classes', 'N/A'):<10} "
          f"{arch_info.get('accuracy', 0):.4f}")

# Export/Import demonstration
print("\n12. Demonstrating architecture export/import...")

# Export architecture
exported_json = repo.export_architecture(arch_id1, format='json')
print(f"   Exported architecture {arch_id1[:20]} to JSON")
print(f"   JSON size: {len(exported_json)} characters")

# Save to file
with open('exported_architecture.json', 'w') as f:
    f.write(exported_json)
print(f"   Saved to: exported_architecture.json")

# Import architecture (simulating sharing across teams)
with open('exported_architecture.json', 'r') as f:
    imported_json = f.read()

imported_arch = repo.import_architecture(imported_json, format='json')
print(f"   Imported architecture: {imported_arch.id[:20]}")
print(f"   Layers: {len(imported_arch.layers)}")

# Test set evaluation
print("\n13. Test Set Evaluation...")
y2_pred = automl2.predict(X2_test)
test_accuracy = accuracy_score(y2_test, y2_pred)
print(f"   Test accuracy (with transfer learning): {test_accuracy:.4f}")

print("\n" + "=" * 80)
print("Transfer Learning NAS Complete!")
print("=" * 80)

print("\nKey Benefits of Transfer Learning:")
print(f"✓ Time savings: {time_saved_pct:.1f}% faster search")
print(f"✓ Better initialization: Started with proven architectures")
print(f"✓ Knowledge accumulation: Repository grows with each search")
print(f"✓ Architecture reuse: Adapt existing architectures to new problems")
print(f"✓ Team collaboration: Share architectures via export/import")

print("\nRepository Statistics:")
print(f"- Total architectures: {len(all_archs)}")
print(f"- Average accuracy: {np.mean([a.get('accuracy', 0) for a in all_archs]):.4f}")
print(f"- Repository size: ./nas_architecture_repo.db")

print("\nBest Practices:")
print("1. Always save successful architectures to the repository")
print("2. Use descriptive tags for easy filtering")
print("3. Include comprehensive metadata (dataset characteristics, performance)")
print("4. Regularly clean up poor-performing architectures")
print("5. Export important architectures for backup and sharing")

print("\nNext Steps:")
print("- Build your organization's architecture repository over time")
print("- Experiment with different problem types and domains")
print("- Share architectures across teams via export/import")
print("- Use similarity search to find relevant starting points")
print("- Adapt architectures to new problems for faster deployment")

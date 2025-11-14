# Architecture Repository API Reference

## Overview
The `ArchitectureRepository` provides persistent storage and retrieval of neural network architectures with support for transfer learning.

## Class: ArchitectureRepository

### Initialization

```python
ArchitectureRepository(db_path: str = '~/.automl_lite/nas_architectures.db')
```

**Parameters:**
- `db_path`: Path to SQLite database file (default: `~/.automl_lite/nas_architectures.db`)

**Example:**
```python
from automl_lite.nas import ArchitectureRepository

repo = ArchitectureRepository(db_path='./my_architectures.db')
```

---

## Core Methods

### save_architecture()

Save an architecture to the repository with optional metadata.

```python
save_architecture(
    architecture: Architecture,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    performance_metrics: Optional[Dict[str, float]] = None,
    hardware_metrics: Optional[Dict[str, float]] = None,
    search_metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> str
```

**Parameters:**
- `architecture`: Architecture object to save
- `dataset_metadata`: Dataset characteristics
  - `problem_type`: 'classification', 'regression', etc.
  - `n_samples`: Number of training samples
  - `n_features`: Number of input features
  - `n_classes`: Number of output classes (classification)
  - `dataset_name`: Name of the dataset
- `performance_metrics`: Training metrics
  - `accuracy`: Training accuracy
  - `val_accuracy`: Validation accuracy
  - `loss`: Training loss
  - `val_loss`: Validation loss
  - `training_time`: Time in seconds
- `hardware_metrics`: Hardware performance
  - `latency_ms`: Inference latency in milliseconds
  - `memory_mb`: Memory usage in MB
  - `model_size_mb`: Model size in MB
  - `flops`: Floating point operations
  - `num_parameters`: Total parameters
  - `target_hardware`: 'cpu', 'gpu', 'mobile', 'edge'
- `search_metadata`: Search process info
  - `search_strategy`: 'evolutionary', 'rl', 'darts'
  - `search_time`: Search time in seconds
  - `search_space_type`: 'tabular', 'vision', 'timeseries'
  - `generation`: Generation number (evolutionary)
- `tags`: List of tags for categorization

**Returns:** Architecture ID (string)

**Example:**
```python
arch_id = repo.save_architecture(
    architecture,
    dataset_metadata={
        'problem_type': 'classification',
        'n_samples': 60000,
        'n_features': 784,
        'n_classes': 10,
        'dataset_name': 'MNIST'
    },
    performance_metrics={
        'accuracy': 0.98,
        'val_accuracy': 0.97,
        'training_time': 120.5
    },
    hardware_metrics={
        'latency_ms': 3.2,
        'memory_mb': 25.0,
        'model_size_mb': 1.5
    },
    tags=['mnist', 'production', 'high-accuracy']
)
```

---

### load_architecture()

Load an architecture from the repository by ID.

```python
load_architecture(architecture_id: str) -> Optional[Tuple[Architecture, Dict[str, Any]]]
```

**Parameters:**
- `architecture_id`: ID of the architecture to load

**Returns:** Tuple of (Architecture, metadata_dict) or None if not found

**Example:**
```python
result = repo.load_architecture(arch_id)
if result:
    architecture, metadata = result
    print(f"Loaded: {architecture}")
    print(f"Accuracy: {metadata['performance_metrics']['accuracy']}")
```

---

### list_architectures()

List architectures with optional filtering.

```python
list_architectures(
    problem_type: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    tags: Optional[List[str]] = None,
    limit: int = 100
) -> List[Tuple[str, Dict[str, Any]]]
```

**Parameters:**
- `problem_type`: Filter by problem type
- `min_accuracy`: Minimum accuracy threshold
- `tags`: Filter by tags (must have all specified tags)
- `limit`: Maximum number of results

**Returns:** List of (architecture_id, summary_dict) tuples

**Example:**
```python
# List all classification architectures with accuracy >= 0.95
results = repo.list_architectures(
    problem_type='classification',
    min_accuracy=0.95,
    limit=10
)

for arch_id, summary in results:
    print(f"{arch_id}: {summary['accuracy']:.2%}")
```

---

### delete_architecture()

Delete an architecture from the repository.

```python
delete_architecture(architecture_id: str) -> bool
```

**Parameters:**
- `architecture_id`: ID of the architecture to delete

**Returns:** True if deleted successfully, False otherwise

**Example:**
```python
success = repo.delete_architecture(arch_id)
if success:
    print("Architecture deleted")
```

---

## Transfer Learning Methods

### compute_similarity()

Compute similarity score between two architecture metadata dictionaries.

```python
compute_similarity(
    metadata1: Dict[str, Any],
    metadata2: Dict[str, Any]
) -> float
```

**Parameters:**
- `metadata1`: First architecture's dataset metadata
- `metadata2`: Second architecture's dataset metadata

**Returns:** Similarity score between 0 and 1 (1 = most similar)

**Similarity Calculation:**
- Problem type match: 40% weight
- Dataset size similarity: 30% weight
- Feature count similarity: 30% weight

**Example:**
```python
similarity = repo.compute_similarity(
    {'problem_type': 'classification', 'n_samples': 10000, 'n_features': 784},
    {'problem_type': 'classification', 'n_samples': 12000, 'n_features': 800}
)
print(f"Similarity: {similarity:.2%}")
```

---

### find_similar_architectures()

Find architectures similar to given dataset characteristics.

```python
find_similar_architectures(
    dataset_metadata: Dict[str, Any],
    top_k: int = 3,
    min_similarity: float = 0.3
) -> List[Tuple[Architecture, Dict[str, Any], float]]
```

**Parameters:**
- `dataset_metadata`: Target dataset characteristics
- `top_k`: Number of similar architectures to return
- `min_similarity`: Minimum similarity threshold (0-1)

**Returns:** List of (Architecture, metadata, similarity_score) tuples, sorted by similarity

**Example:**
```python
# Find architectures for a new classification problem
similar = repo.find_similar_architectures(
    dataset_metadata={
        'problem_type': 'classification',
        'n_samples': 50000,
        'n_features': 1000,
        'n_classes': 20
    },
    top_k=3,
    min_similarity=0.5
)

for arch, metadata, similarity in similar:
    print(f"Similarity: {similarity:.2%}")
    print(f"Architecture: {arch}")
    print(f"Original accuracy: {metadata['performance_metrics']['accuracy']:.2%}")
```

---

### adapt_architecture()

Adapt an architecture to a new problem.

```python
adapt_architecture(
    architecture: Architecture,
    new_input_shape: Tuple[int, ...],
    new_output_shape: Tuple[int, ...],
    dataset_size: Optional[int] = None,
    scale_factor: Optional[float] = None
) -> Architecture
```

**Parameters:**
- `architecture`: Source architecture to adapt
- `new_input_shape`: New input shape (e.g., `(784,)` for MNIST)
- `new_output_shape`: New output shape (e.g., `(10,)` for 10 classes)
- `dataset_size`: Size of new dataset (for automatic scaling)
- `scale_factor`: Manual scaling factor (overrides dataset_size)

**Returns:** Adapted architecture with new ID

**Adaptation Process:**
1. Clones the architecture
2. Modifies input layer to match new input shape
3. Modifies output layer to match new output shape
4. Scales intermediate layers based on dataset size or scale factor
5. Adds adaptation metadata

**Example:**
```python
# Adapt MNIST architecture to Fashion-MNIST
adapted = repo.adapt_architecture(
    mnist_architecture,
    new_input_shape=(784,),
    new_output_shape=(10,),
    dataset_size=60000
)

# Adapt with manual scaling (1.5x larger)
scaled = repo.adapt_architecture(
    base_architecture,
    new_input_shape=(100,),
    new_output_shape=(5,),
    scale_factor=1.5
)
```

---

## Import/Export Methods

### export_architecture()

Export an architecture to a JSON file.

```python
export_architecture(
    architecture_id: str,
    output_path: str,
    include_metadata: bool = True
) -> bool
```

**Parameters:**
- `architecture_id`: ID of the architecture to export
- `output_path`: Path to output JSON file
- `include_metadata`: Whether to include all metadata

**Returns:** True if exported successfully, False otherwise

**Example:**
```python
success = repo.export_architecture(
    arch_id,
    'exported_architecture.json',
    include_metadata=True
)
```

---

### import_architecture()

Import an architecture from a JSON file.

```python
import_architecture(
    input_path: str,
    validate: bool = True,
    save_to_repository: bool = True
) -> Optional[Architecture]
```

**Parameters:**
- `input_path`: Path to input JSON file
- `validate`: Whether to validate the architecture
- `save_to_repository`: Whether to save to repository after import

**Returns:** Imported Architecture object or None if import failed

**Validation Checks:**
- Architecture has at least one layer
- All layers have valid types
- Connections reference valid layer indices
- Layer parameters are reasonable

**Example:**
```python
imported = repo.import_architecture(
    'architecture.json',
    validate=True,
    save_to_repository=True
)

if imported:
    print(f"Imported: {imported.id}")
```

---

## Utility Methods

### get_statistics()

Get statistics about the repository.

```python
get_statistics() -> Dict[str, Any]
```

**Returns:** Dictionary with repository statistics
- `total_architectures`: Total number of architectures
- `by_problem_type`: Count by problem type
- `avg_accuracy`: Average accuracy across all architectures
- `max_accuracy`: Maximum accuracy
- `top_tags`: Most common tags

**Example:**
```python
stats = repo.get_statistics()
print(f"Total architectures: {stats['total_architectures']}")
print(f"Average accuracy: {stats['avg_accuracy']:.2%}")
print(f"By problem type: {stats['by_problem_type']}")
```

---

### close()

Close the database connection.

```python
close()
```

**Example:**
```python
repo.close()
```

---

## Context Manager

The repository can be used as a context manager for automatic resource cleanup.

```python
with ArchitectureRepository(db_path='./architectures.db') as repo:
    # Use repository
    repo.save_architecture(architecture)
    # Connection automatically closed on exit
```

---

## Complete Workflow Example

```python
from automl_lite.nas import ArchitectureRepository, Architecture, LayerConfig

# Create repository
repo = ArchitectureRepository(db_path='./nas_architectures.db')

# Create an architecture
architecture = Architecture(
    layers=[
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
    ]
)

# Save with metadata
arch_id = repo.save_architecture(
    architecture,
    dataset_metadata={
        'problem_type': 'classification',
        'n_samples': 60000,
        'n_features': 784,
        'n_classes': 10
    },
    performance_metrics={'accuracy': 0.98},
    tags=['mnist', 'production']
)

# Later: Find similar architectures for transfer learning
similar = repo.find_similar_architectures(
    dataset_metadata={
        'problem_type': 'classification',
        'n_samples': 50000,
        'n_features': 800,
        'n_classes': 10
    },
    top_k=3
)

# Use most similar as starting point
if similar:
    base_arch, metadata, similarity = similar[0]
    print(f"Found similar architecture with {similarity:.2%} similarity")
    
    # Adapt to new problem
    adapted = repo.adapt_architecture(
        base_arch,
        new_input_shape=(800,),
        new_output_shape=(10,),
        dataset_size=50000
    )
    
    # Use adapted architecture in your search
    # This reduces search time by 40%+

# Export for sharing
repo.export_architecture(arch_id, 'best_architecture.json')

# Clean up
repo.close()
```

---

## Database Schema

The repository uses SQLite with the following tables:

### architectures
- `id` (TEXT, PRIMARY KEY)
- `architecture_json` (TEXT)
- `created_at` (TEXT)
- `updated_at` (TEXT)

### dataset_metadata
- `architecture_id` (TEXT, PRIMARY KEY, FOREIGN KEY)
- `problem_type` (TEXT)
- `n_samples` (INTEGER)
- `n_features` (INTEGER)
- `n_classes` (INTEGER)
- `dataset_name` (TEXT)

### performance_metrics
- `architecture_id` (TEXT, PRIMARY KEY, FOREIGN KEY)
- `accuracy` (REAL)
- `loss` (REAL)
- `val_accuracy` (REAL)
- `val_loss` (REAL)
- `training_time` (REAL)

### hardware_metrics
- `architecture_id` (TEXT, PRIMARY KEY, FOREIGN KEY)
- `latency_ms` (REAL)
- `memory_mb` (REAL)
- `model_size_mb` (REAL)
- `flops` (REAL)
- `num_parameters` (INTEGER)
- `target_hardware` (TEXT)

### search_metadata
- `architecture_id` (TEXT, PRIMARY KEY, FOREIGN KEY)
- `search_strategy` (TEXT)
- `search_time` (REAL)
- `search_space_type` (TEXT)
- `generation` (INTEGER)

### tags
- `architecture_id` (TEXT, FOREIGN KEY)
- `tag` (TEXT)
- PRIMARY KEY: (architecture_id, tag)

---

## Best Practices

1. **Use context manager** for automatic cleanup:
   ```python
   with ArchitectureRepository() as repo:
       # Your code here
   ```

2. **Tag architectures** for easy filtering:
   ```python
   tags=['production', 'high-accuracy', 'mobile-optimized']
   ```

3. **Save comprehensive metadata** for better similarity matching:
   ```python
   repo.save_architecture(
       arch,
       dataset_metadata={...},  # Always include
       performance_metrics={...},  # Always include
       hardware_metrics={...},  # Include for hardware-aware NAS
       search_metadata={...},  # Include for reproducibility
       tags=[...]  # Include for organization
   )
   ```

4. **Use transfer learning** to reduce search time:
   ```python
   similar = repo.find_similar_architectures(target_metadata, top_k=3)
   if similar:
       base_arch = similar[0][0]
       adapted = repo.adapt_architecture(base_arch, ...)
       # Use adapted as starting point
   ```

5. **Export successful architectures** for sharing:
   ```python
   repo.export_architecture(best_arch_id, 'best_model.json')
   ```

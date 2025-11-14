# Neural Architecture Search (NAS) User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration Guide](#configuration-guide)
4. [Hardware-Aware NAS](#hardware-aware-nas)
5. [Multi-Objective Optimization](#multi-objective-optimization)
6. [Transfer Learning](#transfer-learning)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

Neural Architecture Search (NAS) automates the design of neural network architectures, eliminating the need for manual architecture engineering. AutoML Lite's NAS module can:

- **Automatically discover** optimal architectures for your dataset
- **Optimize multiple objectives** (accuracy, latency, model size)
- **Consider hardware constraints** for deployment on mobile/edge devices
- **Leverage transfer learning** to reuse successful architectures
- **Support multiple search strategies** (evolutionary, RL, gradient-based)

### When to Use NAS

Use NAS when:
- You need deep learning models but lack architecture design expertise
- You want to optimize for deployment constraints (latency, memory)
- You need to balance multiple objectives (accuracy vs efficiency)
- You have sufficient computational budget (30+ minutes)

Don't use NAS when:
- Your dataset is very small (<1000 samples)
- You have strict time constraints (<10 minutes)
- Standard architectures work well for your problem
- You don't need deep learning models

---

## Quick Start

### Basic Usage

The simplest way to use NAS with AutoML Lite:

```python
from automl_lite import AutoMLite
import pandas as pd

# Load your data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Enable NAS
automl = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_time_budget=1800  # 30 minutes
)

# Fit will automatically run NAS
automl.fit(X, y)

# Use the best model
predictions = automl.predict(X_test)

# View results
print(f"Best accuracy: {automl.nas_result.best_accuracy:.3f}")
print(f"Architectures evaluated: {automl.nas_result.total_architectures_evaluated}")
```

### Choosing a Search Strategy

AutoML Lite supports three search strategies:

```python
# Evolutionary (default, good balance)
automl = AutoMLite(
    enable_nas=True,
    nas_search_strategy='evolutionary'
)

# Reinforcement Learning (explores diverse architectures)
automl = AutoMLite(
    enable_nas=True,
    nas_search_strategy='rl'
)

# DARTS (gradient-based, fastest)
automl = AutoMLite(
    enable_nas=True,
    nas_search_strategy='darts'
)
```

**Strategy Comparison:**

| Strategy | Speed | Quality | Best For |
|----------|-------|---------|----------|
| Evolutionary | Medium | Good | General purpose, robust |
| RL | Slow | Excellent | Research, diverse exploration |
| DARTS | Fast | Good | Quick results, continuous spaces |

---

## Configuration Guide

### Using Configuration Templates

Pre-configured templates for common scenarios:

```python
from automl_lite.nas import get_config_template

# Quick start (30 min, evolutionary)
config = get_config_template('quick_start')

# Hardware-aware for mobile
config = get_config_template('hardware_aware_mobile')

# Multi-objective optimization
config = get_config_template('multi_objective')

# Research (longer search, RL)
config = get_config_template('research')

# Use with AutoMLite
automl = AutoMLite(enable_nas=True, nas_config=config)
```

### Custom Configuration

Create a custom configuration for fine-grained control:

```python
from automl_lite.nas import NASConfig

config = NASConfig(
    # Search settings
    search_strategy='evolutionary',
    search_space_type='auto',  # or 'tabular', 'vision', 'timeseries'
    time_budget=3600,  # 1 hour
    max_architectures=100,
    
    # Evolutionary settings
    evolution_population_size=50,
    evolution_mutation_rate=0.2,
    
    # Performance estimation
    performance_estimator='early_stopping',
    estimation_budget_fraction=0.1,
    early_stopping_patience=5,
    
    # Checkpointing
    enable_checkpointing=True,
    checkpoint_frequency=10,
    checkpoint_path='./nas_checkpoint.pkl',
    
    # Logging
    verbose=True
)

automl = AutoMLite(enable_nas=True, nas_config=config)
```

### Configuration Parameters Explained

**Search Strategy Parameters:**

- `search_strategy`: Algorithm to use
  - `'evolutionary'`: Genetic algorithm (default)
  - `'rl'`: Reinforcement learning with REINFORCE
  - `'darts'`: Differentiable architecture search

- `search_space_type`: Type of architectures to explore
  - `'auto'`: Automatically detect based on data (default)
  - `'tabular'`: MLP architectures for structured data
  - `'vision'`: CNN architectures for images
  - `'timeseries'`: RNN/LSTM architectures for sequences

- `time_budget`: Maximum search time in seconds
  - Recommended: 1800-3600 (30-60 minutes)
  - Minimum: 600 (10 minutes)

- `max_architectures`: Maximum number of architectures to evaluate
  - Default: 100
  - More architectures = better results but longer search

**Performance Estimation:**

- `performance_estimator`: How to evaluate architectures
  - `'early_stopping'`: Train for few epochs (default, fast)
  - `'learning_curve'`: Extrapolate from partial training
  - `'weight_sharing'`: Use supernet (fastest, less accurate)

- `estimation_budget_fraction`: Fraction of full training
  - Default: 0.1 (10% of epochs)
  - Higher = more accurate but slower

**Checkpointing:**

- `enable_checkpointing`: Save search progress
  - Recommended: True for long searches
  - Allows resuming if interrupted

- `checkpoint_frequency`: Save every N architectures
  - Default: 10
  - Lower = more frequent saves, higher overhead

### Resuming from Checkpoint

If your search is interrupted, resume from the last checkpoint:

```python
from automl_lite.nas import NASController, NASConfig

config = NASConfig(checkpoint_path='./nas_checkpoint.pkl')
controller = NASController(config)

# Resume search
result = controller.resume_search('./nas_checkpoint.pkl')
```

---

## Hardware-Aware NAS

Optimize architectures for specific hardware constraints.

### Mobile Deployment

Find architectures suitable for mobile devices:

```python
from automl_lite.nas import NASConfig

config = NASConfig(
    search_strategy='evolutionary',
    time_budget=1800,
    
    # Enable hardware-aware search
    enable_hardware_aware=True,
    target_hardware='mobile',
    
    # Constraints for mobile
    max_latency_ms=100,      # 100ms inference time
    max_memory_mb=50,        # 50MB memory
    max_model_size_mb=10     # 10MB model size
)

automl = AutoMLite(enable_nas=True, nas_config=config)
automl.fit(X_train, y_train)

# Check if best architecture meets constraints
best_arch = automl.nas_result.best_architecture
print(f"Latency: {best_arch.metadata['latency']:.1f}ms")
print(f"Memory: {best_arch.metadata['memory']:.1f}MB")
print(f"Model size: {best_arch.metadata['model_size']:.1f}MB")
```

### Edge Device Deployment

Optimize for edge devices with limited resources:

```python
config = NASConfig(
    enable_hardware_aware=True,
    target_hardware='edge',
    max_latency_ms=50,       # Very low latency
    max_memory_mb=20,        # Limited memory
    max_model_size_mb=5      # Small model
)
```

### GPU Deployment

Optimize for GPU inference:

```python
config = NASConfig(
    enable_hardware_aware=True,
    target_hardware='gpu',
    max_latency_ms=10,       # Fast GPU inference
    max_memory_mb=1000       # More memory available
)
```

### Understanding Hardware Metrics

**Latency:** Time to process one input (milliseconds)
- Mobile: 50-200ms acceptable
- Edge: <50ms for real-time
- GPU: <10ms for high throughput

**Memory:** Peak memory during inference (megabytes)
- Mobile: 20-100MB
- Edge: 10-50MB
- GPU: 500-2000MB

**Model Size:** Disk size of saved model (megabytes)
- Mobile: 5-20MB
- Edge: 2-10MB
- GPU: 20-100MB

### Hardware Profiling

Profile an architecture for different hardware:

```python
from automl_lite.nas import HardwareProfiler

# Profile for mobile
profiler = HardwareProfiler(target_hardware='mobile')
latency = profiler.estimate_latency(architecture, batch_size=1)
memory = profiler.estimate_memory(architecture, batch_size=1)

print(f"Mobile latency: {latency:.1f}ms")
print(f"Mobile memory: {memory:.1f}MB")

# Check constraints
constraints = {
    'max_latency_ms': 100,
    'max_memory_mb': 50,
    'max_model_size_mb': 10
}
is_valid = profiler.check_constraints(architecture, constraints)
print(f"Meets constraints: {is_valid}")
```

---

## Multi-Objective Optimization

Balance multiple competing objectives.

### Basic Multi-Objective Search

Optimize accuracy, latency, and model size simultaneously:

```python
from automl_lite.nas import NASConfig

config = NASConfig(
    search_strategy='evolutionary',
    time_budget=1800,
    
    # Enable multi-objective
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size'],
    
    # Hardware profiling needed for latency/size
    enable_hardware_aware=True,
    target_hardware='mobile'
)

automl = AutoMLite(enable_nas=True, nas_config=config)
automl.fit(X_train, y_train)

# Get Pareto front
pareto_front = automl.nas_result.pareto_front
print(f"Pareto front contains {len(pareto_front)} architectures")

# Explore trade-offs
for arch in pareto_front:
    print(f"Accuracy: {arch.metadata['accuracy']:.3f}, "
          f"Latency: {arch.metadata['latency']:.1f}ms, "
          f"Size: {arch.metadata['model_size']:.1f}MB")
```

### Weighted Objectives

Specify importance of each objective:

```python
config = NASConfig(
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size'],
    objective_weights={
        'accuracy': 0.6,      # 60% weight on accuracy
        'latency': 0.3,       # 30% weight on latency
        'model_size': 0.1     # 10% weight on size
    }
)
```

### Hard Constraints

Specify minimum requirements:

```python
config = NASConfig(
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size'],
    
    # Only consider architectures meeting these constraints
    constraints={
        'accuracy': 'accuracy > 0.90',
        'latency': 'latency < 100',
        'model_size': 'model_size < 10'
    }
)
```

### Selecting from Pareto Front

Choose the best architecture based on your preferences:

```python
from automl_lite.nas import MultiObjectiveOptimizer

optimizer = MultiObjectiveOptimizer(
    objectives=['accuracy', 'latency', 'model_size']
)

# Get Pareto front
pareto_front = automl.nas_result.pareto_front

# Select based on preferences
preferences = {
    'accuracy': 0.7,      # Prefer accuracy
    'latency': 0.2,       # Some concern for latency
    'model_size': 0.1     # Less concern for size
}

best_arch = optimizer.select_best_architecture(pareto_front, preferences)
print(f"Selected architecture: {best_arch.id}")
```

### Visualizing Pareto Front

Visualize trade-offs between objectives:

```python
from automl_lite.nas import visualize_pareto_front

# 2D plot (accuracy vs latency)
visualize_pareto_front(
    pareto_front,
    x_objective='latency',
    y_objective='accuracy',
    save_path='pareto_2d.html'
)

# 3D plot (all three objectives)
visualize_pareto_front(
    pareto_front,
    x_objective='latency',
    y_objective='model_size',
    z_objective='accuracy',
    save_path='pareto_3d.html'
)
```

---

## Transfer Learning

Reuse successful architectures to speed up search.

### Enabling Transfer Learning

Transfer learning is enabled by default:

```python
config = NASConfig(
    enable_transfer_learning=True,
    architecture_repository_path='~/.automl_lite/nas_architectures.db'
)

automl = AutoMLite(enable_nas=True, nas_config=config)
automl.fit(X_train, y_train)
```

When enabled, NAS will:
1. Search the repository for similar architectures
2. Use them to initialize the search
3. Automatically save successful architectures

### Manual Architecture Management

Save and load architectures manually:

```python
from automl_lite.nas import ArchitectureRepository

repo = ArchitectureRepository()

# Save an architecture
metadata = {
    'dataset': 'my_dataset',
    'n_samples': 10000,
    'n_features': 50,
    'problem_type': 'classification',
    'accuracy': 0.95
}
arch_id = repo.save_architecture(
    architecture,
    metadata=metadata,
    tags=['production', 'high_accuracy']
)

# Find similar architectures
similar = repo.find_similar_architectures(
    dataset_metadata={
        'n_samples': 12000,
        'n_features': 45,
        'problem_type': 'classification'
    },
    top_k=3
)

print(f"Found {len(similar)} similar architectures")
```

### Adapting Architectures

Adapt an architecture to new input/output dimensions:

```python
# Load a saved architecture
arch = repo.load_architecture(arch_id)

# Adapt to new problem
adapted_arch = repo.adapt_architecture(
    arch,
    new_input_shape=(100,),   # 100 features
    new_output_shape=(5,)     # 5 classes
)

# Use adapted architecture
# (can be used as starting point for search)
```

### Import/Export Architectures

Share architectures across projects or teams:

```python
# Export to JSON
arch_json = repo.export_architecture(arch_id, format='json')
with open('architecture.json', 'w') as f:
    f.write(arch_json)

# Import from JSON
with open('architecture.json', 'r') as f:
    arch_json = f.read()
imported_arch = repo.import_architecture(arch_json, format='json')
```

### Transfer Learning Benefits

Expected speedup from transfer learning:

- **40-60% reduction** in search time
- **Better initial population** for evolutionary search
- **Faster convergence** to good solutions
- **Knowledge accumulation** across projects

---

## Advanced Topics

### Custom Search Spaces

Define a custom search space:

```python
from automl_lite.nas import SearchSpace, LayerConfig

class CustomSearchSpace(SearchSpace):
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def sample_architecture(self):
        # Define your sampling logic
        layers = []
        # ... create layers ...
        return Architecture(layers=layers)
    
    def validate_architecture(self, arch):
        # Define validation logic
        return True
    
    def mutate_architecture(self, arch):
        # Define mutation logic
        return mutated_arch

# Use custom search space
from automl_lite.nas import NASController, NASConfig

config = NASConfig(search_strategy='evolutionary')
controller = NASController(config)
controller.search_space = CustomSearchSpace(input_dim=50, output_dim=10)
```

### Parallel Architecture Evaluation

Speed up search with parallel evaluation:

```python
config = NASConfig(
    search_strategy='evolutionary',
    n_jobs=4,  # Use 4 parallel workers
    time_budget=1800
)
```

### Custom Performance Metrics

Use custom metrics for architecture evaluation:

```python
from sklearn.metrics import f1_score

def custom_metric(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

config = NASConfig(
    performance_metric=custom_metric,
    metric_direction='maximize'
)
```

### Architecture Visualization

Visualize architecture structure:

```python
from automl_lite.nas import visualize_architecture

# Generate architecture diagram
visualize_architecture(
    architecture,
    save_path='architecture.png',
    format='png'  # or 'svg', 'pdf'
)
```

### Search Progress Monitoring

Monitor search progress in real-time:

```python
from automl_lite.nas import NASController

controller = NASController(config, verbose=True)

# Search with progress callback
def progress_callback(iteration, best_score, current_arch):
    print(f"Iteration {iteration}: Best score = {best_score:.3f}")
    print(f"Current architecture: {current_arch.id}")

result = controller.search(
    X, y,
    problem_type='classification',
    progress_callback=progress_callback
)
```

---

## Troubleshooting

### Common Issues

**Issue: Search takes too long**

Solutions:
- Reduce `time_budget` or `max_architectures`
- Use faster `performance_estimator` ('early_stopping' or 'weight_sharing')
- Reduce `estimation_budget_fraction`
- Use 'darts' search strategy (fastest)

```python
config = NASConfig(
    time_budget=900,  # 15 minutes
    performance_estimator='weight_sharing',
    estimation_budget_fraction=0.05
)
```

**Issue: Out of memory errors**

Solutions:
- Reduce batch size
- Use smaller search space
- Enable gradient checkpointing
- Reduce `evolution_population_size`

```python
config = NASConfig(
    evolution_population_size=20,  # Smaller population
    batch_size=32  # Smaller batches
)
```

**Issue: Poor architecture quality**

Solutions:
- Increase `time_budget`
- Increase `max_architectures`
- Use 'rl' search strategy (better exploration)
- Increase `estimation_budget_fraction` (more accurate evaluation)

```python
config = NASConfig(
    search_strategy='rl',
    time_budget=3600,  # 1 hour
    max_architectures=200,
    estimation_budget_fraction=0.2
)
```

**Issue: No architectures meet hardware constraints**

Solutions:
- Relax constraints
- Use smaller search space
- Enable transfer learning (find proven architectures)

```python
config = NASConfig(
    enable_hardware_aware=True,
    max_latency_ms=150,  # Relaxed from 100
    max_memory_mb=75,    # Relaxed from 50
    enable_transfer_learning=True
)
```

**Issue: Search fails to start**

Check:
- TensorFlow or PyTorch is installed
- Input data is valid (no NaN, proper shape)
- Configuration is valid

```python
# Validate configuration
config = NASConfig(...)
config.validate()  # Raises error if invalid

# Check data
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"NaN in X: {np.isnan(X).any()}")
```

### Performance Tips

1. **Start with quick_start template** for initial experiments
2. **Use checkpointing** for long searches
3. **Enable transfer learning** to leverage past searches
4. **Use hardware-aware search** only when needed (adds overhead)
5. **Monitor search progress** to detect issues early
6. **Save successful architectures** for future reuse

### Getting Help

- Check the [API Reference](NAS_API_REFERENCE.md)
- Review [examples](../examples/)
- Open an issue on GitHub
- Check verbose logs for detailed error messages

---

## Best Practices

### For Production Use

1. **Run longer searches** (60+ minutes) for production models
2. **Use multi-objective optimization** to balance accuracy and efficiency
3. **Enable hardware-aware search** for deployment constraints
4. **Save and version architectures** in the repository
5. **Validate on held-out test set** before deployment

### For Research

1. **Use RL search strategy** for better exploration
2. **Increase max_architectures** (200+)
3. **Use learning_curve estimator** for accurate predictions
4. **Experiment with custom search spaces**
5. **Analyze Pareto fronts** for insights

### For Quick Experiments

1. **Use quick_start template**
2. **Limit time_budget** (15-30 minutes)
3. **Use evolutionary or darts strategy**
4. **Use early_stopping estimator**
5. **Enable transfer learning**

---

## Next Steps

- Explore [example notebooks](../examples/)
- Read the [API Reference](NAS_API_REFERENCE.md)
- Try different search strategies
- Experiment with hardware-aware search
- Build your architecture repository

For more information, see the [AutoML Lite documentation](../README.md).

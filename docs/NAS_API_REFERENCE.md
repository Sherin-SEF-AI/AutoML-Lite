# Neural Architecture Search (NAS) API Reference

## Overview

The Neural Architecture Search (NAS) module provides automated neural network architecture discovery capabilities for AutoML Lite. This document describes the public API for all NAS components.

## Table of Contents

- [NASController](#nascontroller)
- [Architecture](#architecture)
- [SearchSpace](#searchspace)
- [SearchStrategy](#searchstrategy)
- [PerformanceEstimator](#performanceestimator)
- [HardwareProfiler](#hardwareprofiler)
- [MultiObjectiveOptimizer](#multiobjectiveoptimizer)
- [ArchitectureRepository](#architecturerepository)
- [Configuration](#configuration)
- [Utilities](#utilities)

---

## NASController

The main orchestrator for neural architecture search.

### Class: `NASController`

**Location:** `automl_lite.nas.controller.NASController`

**Description:** Coordinates the architecture search process, managing search strategies, performance estimation, hardware profiling, and multi-objective optimization.

#### Constructor

```python
NASController(
    config: NASConfig,
    experiment_tracker: Optional[ExperimentTracker] = None,
    verbose: bool = True
)
```

**Parameters:**
- `config` (NASConfig): Configuration object specifying search parameters
- `experiment_tracker` (Optional[ExperimentTracker]): Experiment tracking instance for logging
- `verbose` (bool): Enable verbose logging output

#### Methods

##### `search()`

```python
search(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str,
    time_budget: Optional[int] = None
) -> NASResult
```

Performs neural architecture search on the provided dataset.

**Parameters:**
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels
- `problem_type` (str): Type of ML problem ('classification', 'regression')
- `time_budget` (Optional[int]): Maximum search time in seconds

**Returns:**
- `NASResult`: Search results containing best architectures and metrics

**Example:**
```python
from automl_lite.nas import NASController, NASConfig

config = NASConfig(search_strategy='evolutionary', time_budget=1800)
controller = NASController(config)
result = controller.search(X_train, y_train, problem_type='classification')
print(f"Best accuracy: {result.best_accuracy:.3f}")
```

##### `resume_search()`

```python
resume_search(checkpoint_path: str) -> NASResult
```

Resumes a previously interrupted search from a checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file

**Returns:**
- `NASResult`: Updated search results

##### `get_best_architectures()`

```python
get_best_architectures(top_k: int = 5) -> List[Architecture]
```

Returns the top-k best performing architectures.

**Parameters:**
- `top_k` (int): Number of architectures to return

**Returns:**
- `List[Architecture]`: List of best architectures

##### `get_pareto_front()`

```python
get_pareto_front() -> List[Architecture]
```

Returns the Pareto front for multi-objective optimization.

**Returns:**
- `List[Architecture]`: Non-dominated architectures

##### `save_checkpoint()`

```python
save_checkpoint(path: str) -> None
```

Saves current search state to a checkpoint file.

**Parameters:**
- `path` (str): Path where checkpoint will be saved

---

## Architecture

Data structures for representing neural network architectures.

### Class: `Architecture`

**Location:** `automl_lite.nas.architecture.Architecture`

**Description:** Represents a neural network architecture with layers, connections, and metadata.

#### Attributes

- `id` (str): Unique identifier for the architecture
- `layers` (List[LayerConfig]): List of layer configurations
- `connections` (List[Tuple[int, int]]): Layer connections (from_layer, to_layer)
- `global_config` (Dict[str, Any]): Global configuration (optimizer, learning_rate, etc.)
- `metadata` (Dict[str, Any]): Performance and hardware metrics

#### Methods

##### `to_dict()`

```python
to_dict() -> Dict[str, Any]
```

Serializes architecture to dictionary format.

**Returns:**
- `Dict[str, Any]`: Dictionary representation

##### `from_dict()`

```python
@classmethod
from_dict(cls, data: Dict[str, Any]) -> Architecture
```

Creates architecture from dictionary.

**Parameters:**
- `data` (Dict[str, Any]): Dictionary representation

**Returns:**
- `Architecture`: Reconstructed architecture

##### `validate()`

```python
validate() -> bool
```

Validates architecture structure and layer compatibility.

**Returns:**
- `bool`: True if valid, False otherwise

### Class: `LayerConfig`

**Location:** `automl_lite.nas.architecture.LayerConfig`

**Description:** Configuration for a single neural network layer.

#### Attributes

- `layer_type` (str): Type of layer ('dense', 'conv2d', 'lstm', etc.)
- `params` (Dict[str, Any]): Layer-specific parameters
- `input_shape` (Optional[Tuple[int, ...]]): Expected input shape
- `output_shape` (Optional[Tuple[int, ...]]): Output shape

---

## SearchSpace

Defines the space of possible architectures.

### Class: `SearchSpace` (Abstract Base)

**Location:** `automl_lite.nas.search_space.SearchSpace`

**Description:** Abstract base class for defining architecture search spaces.

#### Methods

##### `sample_architecture()`

```python
sample_architecture() -> Architecture
```

Samples a random architecture from the search space.

**Returns:**
- `Architecture`: Randomly sampled architecture

##### `validate_architecture()`

```python
validate_architecture(arch: Architecture) -> bool
```

Validates if an architecture is valid within this search space.

**Parameters:**
- `arch` (Architecture): Architecture to validate

**Returns:**
- `bool`: True if valid

##### `mutate_architecture()`

```python
mutate_architecture(arch: Architecture) -> Architecture
```

Creates a mutated version of an architecture.

**Parameters:**
- `arch` (Architecture): Architecture to mutate

**Returns:**
- `Architecture`: Mutated architecture

### Class: `TabularSearchSpace`

**Location:** `automl_lite.nas.search_space.TabularSearchSpace`

**Description:** Search space for tabular/structured data with MLP architectures.

**Supported Layers:**
- Dense (16-512 units)
- Dropout (0.0-0.5 rate)
- BatchNormalization

**Architecture Depth:** 1-8 layers

**Example:**
```python
from automl_lite.nas import TabularSearchSpace

search_space = TabularSearchSpace(
    input_dim=20,
    output_dim=2,
    max_layers=6,
    allow_skip_connections=True
)
arch = search_space.sample_architecture()
```

### Class: `VisionSearchSpace`

**Location:** `automl_lite.nas.search_space.VisionSearchSpace`

**Description:** Search space for computer vision tasks with CNN architectures.

**Supported Layers:**
- Conv2D (16-256 filters, kernel sizes 3/5/7)
- MaxPooling2D
- Dense
- Dropout
- BatchNormalization

**Architecture Depth:** 3-20 layers

### Class: `TimeSeriesSearchSpace`

**Location:** `automl_lite.nas.search_space.TimeSeriesSearchSpace`

**Description:** Search space for time series tasks with RNN architectures.

**Supported Layers:**
- LSTM (32-256 units)
- GRU (32-256 units)
- Conv1D
- Dense
- Dropout

**Architecture Depth:** 1-6 recurrent layers

---

## SearchStrategy

Algorithms for exploring the architecture search space.

### Class: `SearchStrategy` (Abstract Base)

**Location:** `automl_lite.nas.search_strategy.SearchStrategy`

**Description:** Abstract base class for search strategies.

#### Methods

##### `generate_architecture()`

```python
generate_architecture() -> Architecture
```

Generates the next architecture to evaluate.

**Returns:**
- `Architecture`: Next architecture candidate

##### `update()`

```python
update(architectures: List[Architecture], rewards: List[float]) -> None
```

Updates strategy based on evaluation results.

**Parameters:**
- `architectures` (List[Architecture]): Evaluated architectures
- `rewards` (List[float]): Performance scores

### Class: `EvolutionarySearchStrategy`

**Location:** `automl_lite.nas.search_strategy.EvolutionarySearchStrategy`

**Description:** Genetic algorithm-based search strategy.

**Constructor:**
```python
EvolutionarySearchStrategy(
    search_space: SearchSpace,
    population_size: int = 50,
    mutation_rate: float = 0.2,
    crossover_rate: float = 0.7,
    tournament_size: int = 3,
    elitism_ratio: float = 0.1
)
```

**Parameters:**
- `search_space` (SearchSpace): Search space to explore
- `population_size` (int): Number of individuals in population
- `mutation_rate` (float): Probability of mutation
- `crossover_rate` (float): Probability of crossover
- `tournament_size` (int): Tournament selection size
- `elitism_ratio` (float): Fraction of top individuals to preserve

### Class: `RLSearchStrategy`

**Location:** `automl_lite.nas.search_strategy.RLSearchStrategy`

**Description:** Reinforcement learning-based search using REINFORCE algorithm.

**Constructor:**
```python
RLSearchStrategy(
    search_space: SearchSpace,
    controller_hidden_size: int = 100,
    baseline_decay: float = 0.95,
    learning_rate: float = 0.001
)
```

### Class: `DARTSSearchStrategy`

**Location:** `automl_lite.nas.search_strategy.DARTSSearchStrategy`

**Description:** Differentiable architecture search using gradient-based optimization.

**Constructor:**
```python
DARTSSearchStrategy(
    search_space: SearchSpace,
    supernet_epochs: int = 50,
    arch_learning_rate: float = 3e-4
)
```

---

## PerformanceEstimator

Efficient architecture performance estimation.

### Class: `PerformanceEstimator` (Abstract Base)

**Location:** `automl_lite.nas.performance_estimator.PerformanceEstimator`

**Description:** Base class for performance estimation strategies.

#### Methods

##### `estimate_performance()`

```python
estimate_performance(
    arch: Architecture,
    X: np.ndarray,
    y: np.ndarray,
    budget_fraction: float = 0.1
) -> Tuple[float, float]
```

Estimates architecture performance without full training.

**Parameters:**
- `arch` (Architecture): Architecture to evaluate
- `X` (np.ndarray): Training features
- `y` (np.ndarray): Training labels
- `budget_fraction` (float): Fraction of full training budget to use

**Returns:**
- `Tuple[float, float]`: (performance_score, confidence)

### Class: `EarlyStoppingEstimator`

**Location:** `automl_lite.nas.performance_estimator.EarlyStoppingEstimator`

**Description:** Estimates performance using early stopping on partial training.

**Constructor:**
```python
EarlyStoppingEstimator(
    max_epochs: int = 50,
    patience: int = 5,
    min_epochs: int = 10
)
```

### Class: `LearningCurveEstimator`

**Location:** `automl_lite.nas.performance_estimator.LearningCurveEstimator`

**Description:** Predicts final performance by extrapolating learning curves.

### Class: `WeightSharingEstimator`

**Location:** `automl_lite.nas.performance_estimator.WeightSharingEstimator`

**Description:** Uses supernet weight sharing for fast evaluation.

---

## HardwareProfiler

Hardware-aware architecture profiling.

### Class: `HardwareProfiler`

**Location:** `automl_lite.nas.hardware_profiler.HardwareProfiler`

**Description:** Estimates hardware-specific metrics for architectures.

**Constructor:**
```python
HardwareProfiler(
    target_hardware: str = 'cpu',
    calibration_data: Optional[Dict] = None
)
```

**Parameters:**
- `target_hardware` (str): Target hardware ('cpu', 'gpu', 'mobile', 'edge')
- `calibration_data` (Optional[Dict]): Calibration measurements

#### Methods

##### `estimate_latency()`

```python
estimate_latency(arch: Architecture, batch_size: int = 1) -> float
```

Estimates inference latency in milliseconds.

**Parameters:**
- `arch` (Architecture): Architecture to profile
- `batch_size` (int): Batch size for inference

**Returns:**
- `float`: Estimated latency in milliseconds

##### `estimate_memory()`

```python
estimate_memory(arch: Architecture, batch_size: int = 1) -> float
```

Estimates peak memory usage in megabytes.

**Parameters:**
- `arch` (Architecture): Architecture to profile
- `batch_size` (int): Batch size

**Returns:**
- `float`: Estimated memory in MB

##### `check_constraints()`

```python
check_constraints(
    arch: Architecture,
    constraints: Dict[str, float]
) -> bool
```

Checks if architecture satisfies hardware constraints.

**Parameters:**
- `arch` (Architecture): Architecture to check
- `constraints` (Dict[str, float]): Constraint specifications

**Returns:**
- `bool`: True if all constraints satisfied

**Example:**
```python
profiler = HardwareProfiler(target_hardware='mobile')
constraints = {
    'max_latency_ms': 100,
    'max_memory_mb': 50,
    'max_model_size_mb': 10
}
is_valid = profiler.check_constraints(arch, constraints)
```

---

## MultiObjectiveOptimizer

Multi-objective optimization for NAS.

### Class: `MultiObjectiveOptimizer`

**Location:** `automl_lite.nas.multi_objective.MultiObjectiveOptimizer`

**Description:** Handles optimization of multiple competing objectives.

**Constructor:**
```python
MultiObjectiveOptimizer(
    objectives: List[str],
    weights: Optional[Dict[str, float]] = None,
    constraints: Optional[Dict[str, str]] = None
)
```

**Parameters:**
- `objectives` (List[str]): Objectives to optimize (e.g., ['accuracy', 'latency', 'model_size'])
- `weights` (Optional[Dict[str, float]]): Objective weights for scalarization
- `constraints` (Optional[Dict[str, str]]): Hard constraints (e.g., "accuracy > 0.9")

#### Methods

##### `compute_pareto_front()`

```python
compute_pareto_front(architectures: List[Architecture]) -> List[Architecture]
```

Computes the Pareto front from a set of architectures.

**Parameters:**
- `architectures` (List[Architecture]): Candidate architectures

**Returns:**
- `List[Architecture]`: Non-dominated architectures

##### `select_best_architecture()`

```python
select_best_architecture(
    pareto_front: List[Architecture],
    preferences: Dict[str, float]
) -> Architecture
```

Selects best architecture from Pareto front based on preferences.

**Parameters:**
- `pareto_front` (List[Architecture]): Pareto front architectures
- `preferences` (Dict[str, float]): User preferences for objectives

**Returns:**
- `Architecture`: Selected architecture

---

## ArchitectureRepository

Storage and retrieval of architectures for transfer learning.

### Class: `ArchitectureRepository`

**Location:** `automl_lite.nas.repository.ArchitectureRepository`

**Description:** Manages architecture storage and transfer learning.

**Constructor:**
```python
ArchitectureRepository(db_path: str = '~/.automl_lite/nas_architectures.db')
```

**Parameters:**
- `db_path` (str): Path to SQLite database file

#### Methods

##### `save_architecture()`

```python
save_architecture(
    arch: Architecture,
    metadata: Dict[str, Any],
    tags: Optional[List[str]] = None
) -> str
```

Saves an architecture to the repository.

**Parameters:**
- `arch` (Architecture): Architecture to save
- `metadata` (Dict[str, Any]): Dataset and performance metadata
- `tags` (Optional[List[str]]): Tags for categorization

**Returns:**
- `str`: Architecture ID

##### `find_similar_architectures()`

```python
find_similar_architectures(
    dataset_metadata: Dict[str, Any],
    top_k: int = 3
) -> List[Architecture]
```

Finds similar architectures based on dataset characteristics.

**Parameters:**
- `dataset_metadata` (Dict[str, Any]): Current dataset metadata
- `top_k` (int): Number of similar architectures to return

**Returns:**
- `List[Architecture]`: Similar architectures

##### `adapt_architecture()`

```python
adapt_architecture(
    arch: Architecture,
    new_input_shape: Tuple[int, ...],
    new_output_shape: Tuple[int, ...]
) -> Architecture
```

Adapts an architecture to new input/output dimensions.

**Parameters:**
- `arch` (Architecture): Architecture to adapt
- `new_input_shape` (Tuple[int, ...]): New input shape
- `new_output_shape` (Tuple[int, ...]): New output shape

**Returns:**
- `Architecture`: Adapted architecture

---

## Configuration

### Class: `NASConfig`

**Location:** `automl_lite.nas.architecture.NASConfig`

**Description:** Configuration for neural architecture search.

#### Attributes

**Search Configuration:**
- `search_strategy` (str): Strategy to use ('evolutionary', 'rl', 'darts')
- `search_space_type` (str): Search space type ('auto', 'tabular', 'vision', 'timeseries')
- `time_budget` (int): Maximum search time in seconds
- `max_architectures` (int): Maximum architectures to evaluate

**Strategy-Specific:**
- `rl_controller_hidden_size` (int): RL controller hidden units
- `rl_baseline_decay` (float): RL baseline decay rate
- `evolution_population_size` (int): Evolutionary population size
- `evolution_mutation_rate` (float): Mutation probability
- `darts_supernet_epochs` (int): DARTS supernet training epochs

**Performance Estimation:**
- `performance_estimator` (str): Estimator type ('early_stopping', 'weight_sharing', 'learning_curve')
- `estimation_budget_fraction` (float): Fraction of full training budget
- `early_stopping_patience` (int): Early stopping patience

**Hardware Constraints:**
- `enable_hardware_aware` (bool): Enable hardware-aware search
- `target_hardware` (str): Target hardware platform
- `max_latency_ms` (Optional[float]): Maximum latency constraint
- `max_memory_mb` (Optional[float]): Maximum memory constraint
- `max_model_size_mb` (Optional[float]): Maximum model size constraint

**Multi-Objective:**
- `enable_multi_objective` (bool): Enable multi-objective optimization
- `objectives` (List[str]): Objectives to optimize
- `objective_weights` (Optional[Dict[str, float]]): Objective weights

**Transfer Learning:**
- `enable_transfer_learning` (bool): Enable transfer learning
- `architecture_repository_path` (str): Repository database path

**Checkpointing:**
- `enable_checkpointing` (bool): Enable checkpointing
- `checkpoint_frequency` (int): Checkpoint save frequency
- `checkpoint_path` (str): Checkpoint file path

**Example:**
```python
from automl_lite.nas import NASConfig

config = NASConfig(
    search_strategy='evolutionary',
    time_budget=1800,
    enable_hardware_aware=True,
    target_hardware='mobile',
    max_latency_ms=100,
    enable_multi_objective=True,
    objectives=['accuracy', 'latency', 'model_size']
)
```

---

## Utilities

### Function: `get_config_template()`

**Location:** `automl_lite.nas.config_templates.get_config_template`

```python
get_config_template(template_name: str) -> NASConfig
```

Returns a pre-configured NAS configuration template.

**Parameters:**
- `template_name` (str): Template name ('quick_start', 'hardware_aware_mobile', 'multi_objective', 'research')

**Returns:**
- `NASConfig`: Configuration object

**Example:**
```python
from automl_lite.nas import get_config_template

config = get_config_template('hardware_aware_mobile')
```

### Function: `compare_architectures()`

**Location:** `automl_lite.nas.utils.compare_architectures`

```python
compare_architectures(arch1: Architecture, arch2: Architecture) -> Dict[str, Any]
```

Compares two architectures and returns differences.

**Parameters:**
- `arch1` (Architecture): First architecture
- `arch2` (Architecture): Second architecture

**Returns:**
- `Dict[str, Any]`: Comparison results

### Function: `estimate_search_space_size()`

**Location:** `automl_lite.nas.utils.estimate_search_space_size`

```python
estimate_search_space_size(search_space: SearchSpace) -> int
```

Estimates the total number of possible architectures.

**Parameters:**
- `search_space` (SearchSpace): Search space to analyze

**Returns:**
- `int`: Estimated search space size

---

## Integration with AutoMLite

### Using NAS with AutoMLite

```python
from automl_lite import AutoMLite
from automl_lite.nas import NASConfig

# Enable NAS in AutoMLite
config = NASConfig(
    search_strategy='evolutionary',
    time_budget=1800
)

automl = AutoMLite(
    enable_deep_learning=True,
    enable_nas=True,
    nas_config=config
)

# Fit will automatically run NAS
automl.fit(X_train, y_train)

# Access NAS results
print(f"Best architecture accuracy: {automl.nas_result.best_accuracy:.3f}")
print(f"Architectures evaluated: {automl.nas_result.total_architectures_evaluated}")

# Get Pareto front
pareto_front = automl.nas_result.pareto_front
for arch in pareto_front:
    print(f"Accuracy: {arch.metadata['accuracy']:.3f}, "
          f"Latency: {arch.metadata['latency']:.1f}ms")
```

---

## Error Handling

All NAS components raise standard Python exceptions:

- `ValueError`: Invalid configuration or parameters
- `RuntimeError`: Search or evaluation failures
- `FileNotFoundError`: Missing checkpoint or repository files
- `MemoryError`: Out of memory during architecture evaluation

Example error handling:

```python
try:
    result = controller.search(X, y, problem_type='classification')
except ValueError as e:
    print(f"Invalid configuration: {e}")
except RuntimeError as e:
    print(f"Search failed: {e}")
    # Resume from checkpoint if available
    result = controller.resume_search('checkpoint.pkl')
```

---

## Performance Considerations

1. **Search Time**: Typical searches take 30-60 minutes for good results
2. **Memory Usage**: Peak memory ~2-4GB for most searches
3. **Parallelization**: Use `n_jobs` parameter for parallel evaluation
4. **Caching**: Enable caching to avoid re-evaluating similar architectures

---

## See Also

- [NAS User Guide](NAS_USER_GUIDE.md)
- [Examples](../examples/)
- [AutoML Lite Documentation](API_REFERENCE.md)

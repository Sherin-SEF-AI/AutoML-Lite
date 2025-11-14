# Performance Estimator API Reference

## Quick Reference

### Classes

#### PerformanceEstimate
Result dataclass containing estimation results.

```python
@dataclass
class PerformanceEstimate:
    performance: float              # Estimated performance metric
    confidence_lower: float         # Lower bound of confidence interval
    confidence_upper: float         # Upper bound of confidence interval
    training_time: float = 0.0      # Time spent training (seconds)
    epochs_trained: int = 0         # Number of epochs trained
    metadata: Dict[str, Any] = None # Additional metadata
```

**Methods:**
- `get_confidence_interval() -> Tuple[float, float]` - Returns (lower, upper)
- `get_confidence_width() -> float` - Returns interval width

---

#### PerformanceEstimator (Abstract Base Class)

Base class for all performance estimators.

```python
class PerformanceEstimator(ABC):
    def __init__(
        self,
        budget_fraction: float = 0.1,
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    )
```

**Abstract Methods:**
- `estimate_performance(architecture, X, y, problem_type) -> PerformanceEstimate`

**Utility Methods:**
- `should_continue_training(current_epoch, metrics, patience) -> bool`
- `_build_model_from_architecture(architecture, input_shape, output_shape, problem_type)`
- `_split_data(X, y) -> Tuple[X_train, X_val, y_train, y_val]`
- `_get_num_epochs() -> int`

---

#### EarlyStoppingEstimator

Trains for a fraction of epochs with early stopping.

```python
class EarlyStoppingEstimator(PerformanceEstimator):
    def __init__(
        self,
        budget_fraction: float = 0.15,  # 15% of full training
        max_epochs: int = 100,
        patience: int = 5,              # Early stopping patience
        min_delta: float = 0.001,       # Minimum improvement
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    )
```

**Use Case:** Quick filtering of unpromising architectures

**Characteristics:**
- Budget: 10-20% of full training
- Speed: Fast
- Accuracy: Good for identifying poor candidates

---

#### LearningCurveEstimator

Extrapolates final performance from partial training curves.

```python
class LearningCurveEstimator(PerformanceEstimator):
    def __init__(
        self,
        budget_fraction: float = 0.2,   # 20% of full training
        max_epochs: int = 100,
        curve_model: str = 'power_law', # 'power_law', 'exponential', 'both'
        min_points: int = 5,            # Minimum points for extrapolation
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    )
```

**Curve Models:**
- `'power_law'`: y = a * x^b + c
- `'exponential'`: y = a * (1 - exp(-b * x)) + c
- `'both'`: Fits both and averages weighted by R²

**Use Case:** More accurate performance prediction

**Characteristics:**
- Budget: 15-25% of full training
- Speed: Moderate
- Accuracy: Better prediction of final performance

---

#### WeightSharingEstimator

Uses a supernet for weight sharing across architectures.

```python
class WeightSharingEstimator(PerformanceEstimator):
    def __init__(
        self,
        budget_fraction: float = 0.05,  # 5% per architecture
        max_epochs: int = 100,
        supernet_epochs: int = 50,      # Epochs to train supernet
        finetune_epochs: int = 5,       # Epochs to fine-tune
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    )
```

**Additional Methods:**
- `build_supernet(search_space, input_shape, output_shape, problem_type)`
- `train_supernet(X, y)`

**Use Case:** Evaluating many architectures efficiently

**Characteristics:**
- Budget: 5% per architecture (after supernet training)
- Speed: Very fast (100x speedup)
- Accuracy: Good with proper supernet

---

## Usage Examples

### Example 1: Early Stopping Estimator

```python
from src.automl_lite.nas.architecture import Architecture, LayerConfig
from src.automl_lite.nas.performance_estimator import EarlyStoppingEstimator

# Create architecture
architecture = Architecture(
    layers=[
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
    ]
)

# Create estimator
estimator = EarlyStoppingEstimator(
    budget_fraction=0.15,
    max_epochs=50,
    patience=5,
    verbose=True
)

# Estimate performance
estimate = estimator.estimate_performance(
    architecture, X_train, y_train, problem_type='classification'
)

print(f"Performance: {estimate.performance:.4f}")
print(f"CI: [{estimate.confidence_lower:.4f}, {estimate.confidence_upper:.4f}]")
print(f"Time: {estimate.training_time:.2f}s")
print(f"Epochs: {estimate.epochs_trained}")
```

### Example 2: Learning Curve Estimator

```python
from src.automl_lite.nas.performance_estimator import LearningCurveEstimator

# Create estimator with both curve models
estimator = LearningCurveEstimator(
    budget_fraction=0.2,
    max_epochs=100,
    curve_model='both',  # Use both power law and exponential
    verbose=True
)

# Estimate performance
estimate = estimator.estimate_performance(
    architecture, X_train, y_train, problem_type='classification'
)

print(f"Extrapolated Performance: {estimate.performance:.4f}")
print(f"Fit Quality: {estimate.metadata['fit_quality']}")
```

### Example 3: Weight Sharing Estimator

```python
from src.automl_lite.nas.performance_estimator import WeightSharingEstimator

# Create estimator
estimator = WeightSharingEstimator(
    supernet_epochs=50,
    finetune_epochs=5,
    verbose=True
)

# Evaluate multiple architectures (supernet trained on first call)
architectures = [arch1, arch2, arch3, arch4, arch5]
estimates = []

for arch in architectures:
    estimate = estimator.estimate_performance(
        arch, X_train, y_train, problem_type='classification'
    )
    estimates.append(estimate)
    print(f"Arch {arch.id[:8]}: {estimate.performance:.4f} "
          f"({estimate.training_time:.2f}s)")
```

### Example 4: Comparing Estimators

```python
# Compare all three estimators
estimators = {
    'Early Stopping': EarlyStoppingEstimator(budget_fraction=0.15),
    'Learning Curve': LearningCurveEstimator(budget_fraction=0.2),
    'Weight Sharing': WeightSharingEstimator(supernet_epochs=30)
}

results = {}
for name, estimator in estimators.items():
    estimate = estimator.estimate_performance(
        architecture, X_train, y_train, problem_type='classification'
    )
    results[name] = {
        'performance': estimate.performance,
        'time': estimate.training_time,
        'epochs': estimate.epochs_trained
    }

# Print comparison
for name, result in results.items():
    print(f"{name:20s}: {result['performance']:.4f} "
          f"({result['time']:.2f}s, {result['epochs']} epochs)")
```

---

## Supported Layer Types

The performance estimators support the following layer types when building models:

### Tabular Data
- `'dense'` - Fully connected layer
  - Parameters: `units`, `activation`
- `'dropout'` - Dropout regularization
  - Parameters: `rate`
- `'batchnormalization'` or `'batch_normalization'` - Batch normalization

### Vision Data
- `'conv2d'` - 2D convolution
  - Parameters: `filters`, `kernel_size`, `strides`, `padding`, `activation`
- `'maxpooling2d'` or `'max_pooling2d'` - 2D max pooling
  - Parameters: `pool_size`
- `'flatten'` - Flatten layer

### Time Series Data
- `'lstm'` - LSTM recurrent layer
  - Parameters: `units`, `return_sequences`
- `'gru'` - GRU recurrent layer
  - Parameters: `units`, `return_sequences`
- `'conv1d'` - 1D convolution
  - Parameters: `filters`, `kernel_size`, `activation`

---

## Configuration in NASConfig

```python
from src.automl_lite.nas.architecture import NASConfig

# Configure performance estimator in NAS
config = NASConfig(
    # Choose estimator type
    performance_estimator='early_stopping',  # or 'learning_curve', 'weight_sharing'
    
    # Budget configuration
    estimation_budget_fraction=0.15,  # 15% of full training
    
    # Early stopping specific
    early_stopping_patience=5,
    
    # Other settings
    max_epochs=100,
    verbose=True
)
```

---

## Error Handling

All estimators handle errors gracefully:

```python
estimate = estimator.estimate_performance(architecture, X, y)

# Check status
if estimate.metadata.get('status') == 'success':
    print(f"Success: {estimate.performance:.4f}")
elif estimate.metadata.get('status') == 'build_failed':
    print(f"Model building failed: {estimate.metadata.get('error')}")
elif estimate.metadata.get('status') == 'training_failed':
    print(f"Training failed: {estimate.metadata.get('error')}")
```

---

## Performance Metrics

### Classification
- Primary metric: `accuracy` (0.0 to 1.0)
- Confidence intervals: Based on validation accuracy variance
- Higher is better

### Regression
- Primary metric: `-mae` (negative mean absolute error)
- Confidence intervals: Based on validation MAE variance
- Higher is better (less negative = lower error)

---

## Best Practices

1. **Choose the Right Estimator:**
   - Use `EarlyStoppingEstimator` for quick filtering
   - Use `LearningCurveEstimator` for accurate estimates
   - Use `WeightSharingEstimator` when evaluating many architectures

2. **Budget Configuration:**
   - Start with default budgets (15-20%)
   - Increase for more accuracy, decrease for more speed
   - Consider total search time budget

3. **Confidence Intervals:**
   - Use confidence intervals for uncertainty-aware search
   - Wider intervals indicate less reliable estimates
   - Consider both performance and confidence width

4. **Supernet Training:**
   - Train supernet once at the beginning
   - Reuse for all architecture evaluations
   - Retrain if search space changes significantly

5. **Validation:**
   - Use consistent validation split across estimators
   - Stratify for classification problems
   - Set random_state for reproducibility

---

## Dependencies

- **Required:** numpy, scikit-learn, TensorFlow ≥ 2.8.0
- **Optional:** scipy (for learning curve fitting), PyTorch ≥ 1.12.0 (not yet implemented)

---

## Limitations

1. PyTorch support not yet implemented
2. Skip connections have simplified implementation
3. Supernet architecture is basic (could be more sophisticated)
4. Weight inheritance uses simple layer name matching

---

## See Also

- [Task 3 Summary](TASK_3_SUMMARY.md) - Detailed implementation notes
- [Design Document](design.md) - Overall NAS design
- [Requirements](requirements.md) - Performance estimator requirements
- [Demo Script](../../examples/nas_performance_estimator_demo.py) - Working examples

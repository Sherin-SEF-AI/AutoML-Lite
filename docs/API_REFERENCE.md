# AutoML Lite API Reference

## Table of Contents

- [AutoMLite Class](#automlite-class)
- [CLI Commands](#cli-commands)
- [Configuration](#configuration)
- [Data Types](#data-types)
- [Error Handling](#error-handling)

## AutoMLite Class

The main class for automated machine learning operations.

### Constructor

```python
AutoMLite(
    problem_type: str = 'auto',
    time_budget: int = 300,
    max_models: int = 10,
    cv_folds: int = 5,
    random_state: int = 42,
    enable_ensemble: bool = False,
    enable_feature_selection: bool = False,
    enable_interpretability: bool = False,
    enable_early_stopping: bool = False,
    patience: int = 10,
    min_delta: float = 0.001,
    verbose: bool = True,
    n_jobs: int = -1
)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem_type` | str | 'auto' | Type of ML problem ('classification', 'regression', 'auto') |
| `time_budget` | int | 300 | Maximum time in seconds for training |
| `max_models` | int | 10 | Maximum number of models to try |
| `cv_folds` | int | 5 | Number of cross-validation folds |
| `random_state` | int | 42 | Random seed for reproducibility |
| `enable_ensemble` | bool | False | Enable ensemble methods |
| `enable_feature_selection` | bool | False | Enable feature selection |
| `enable_interpretability` | bool | False | Enable model interpretability |
| `enable_early_stopping` | bool | False | Enable early stopping |
| `patience` | int | 10 | Patience for early stopping |
| `min_delta` | float | 0.001 | Minimum improvement for early stopping |
| `verbose` | bool | True | Enable verbose output |
| `n_jobs` | int | -1 | Number of parallel jobs (-1 for all cores) |

### Methods

#### `fit(X, y=None, target_column=None)`

Train the AutoML model.

```python
def fit(
    self,
    X: Union[pd.DataFrame, np.ndarray],
    y: Optional[Union[pd.Series, np.ndarray]] = None,
    target_column: Optional[str] = None
) -> 'AutoMLite'
```

**Parameters:**
- `X`: Training data (DataFrame or array)
- `y`: Target values (optional if target_column is provided)
- `target_column`: Name of target column in X (optional if y is provided)

**Returns:**
- Self instance for method chaining

**Example:**
```python
# Using DataFrame with target column
automl.fit(data, target_column='target')

# Using separate X and y
automl.fit(X, y)
```

#### `predict(X)`

Make predictions on new data.

```python
def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray
```

**Parameters:**
- `X`: Data to predict on

**Returns:**
- Array of predictions

**Example:**
```python
predictions = automl.predict(test_data)
```

#### `predict_proba(X)`

Get prediction probabilities (classification only).

```python
def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray
```

**Parameters:**
- `X`: Data to predict on

**Returns:**
- Array of prediction probabilities

**Raises:**
- `NotImplementedError`: If model doesn't support predict_proba

**Example:**
```python
probabilities = automl.predict_proba(test_data)
```

#### `save_model(path)`

Save the trained model to disk.

```python
def save_model(self, path: Union[str, Path]) -> None
```

**Parameters:**
- `path`: Path to save the model

**Example:**
```python
automl.save_model('model.pkl')
```

#### `load_model(path)`

Load a saved model from disk.

```python
def load_model(self, path: Union[str, Path]) -> 'AutoMLite'
```

**Parameters:**
- `path`: Path to the saved model

**Returns:**
- Self instance with loaded model

**Example:**
```python
automl.load_model('model.pkl')
```

#### `generate_report(path)`

Generate comprehensive HTML report.

```python
def generate_report(
    self,
    path: Union[str, Path],
    X_test: Optional[pd.DataFrame] = None,
    y_test: Optional[np.ndarray] = None
) -> None
```

**Parameters:**
- `path`: Path to save the report
- `X_test`: Test data for additional analysis (optional)
- `y_test`: Test labels for additional analysis (optional)

**Example:**
```python
automl.generate_report('report.html', X_test, y_test)
```

#### `get_leaderboard()`

Get model performance leaderboard.

```python
def get_leaderboard(self) -> List[Dict[str, Any]]
```

**Returns:**
- List of dictionaries containing model performance information

**Example:**
```python
leaderboard = automl.get_leaderboard()
for model in leaderboard:
    print(f"{model['model_name']}: {model['score']:.4f}")
```

#### `get_feature_importance()`

Get feature importance scores.

```python
def get_feature_importance(self) -> Dict[str, float]
```

**Returns:**
- Dictionary mapping feature names to importance scores

**Example:**
```python
importance = automl.get_feature_importance()
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {score:.4f}")
```

#### `get_interpretability_results()`

Get model interpretability results.

```python
def get_interpretability_results(self) -> Dict[str, Any]
```

**Returns:**
- Dictionary containing interpretability analysis results

**Example:**
```python
interpretability = automl.get_interpretability_results()
print(interpretability.keys())
```

### Properties

#### `best_model_name`
- **Type:** str
- **Description:** Name of the best performing model

#### `best_score`
- **Type:** float
- **Description:** Performance score of the best model

#### `best_model`
- **Type:** sklearn estimator
- **Description:** The best trained model

#### `selected_features`
- **Type:** List[str]
- **Description:** List of selected features (if feature selection is enabled)

#### `training_history`
- **Type:** List[Dict[str, Any]]
- **Description:** History of model training with scores and times

#### `ensemble_model`
- **Type:** sklearn estimator
- **Description:** The ensemble model (if ensemble is enabled)

## CLI Commands

### Training Command

```bash
python -m automl_lite.cli.main train [OPTIONS] DATA
```

**Arguments:**
- `DATA`: Path to training data file (CSV format)

**Options:**
- `--target TEXT`: Target column name (required)
- `--output PATH`: Output model file path (default: model.pkl)
- `--config PATH`: Configuration file path
- `--time-budget INTEGER`: Time budget in seconds (default: 300)
- `--max-models INTEGER`: Maximum number of models (default: 10)
- `--cv-folds INTEGER`: Cross-validation folds (default: 5)
- `--enable-ensemble`: Enable ensemble methods
- `--enable-feature-selection`: Enable feature selection
- `--enable-interpretability`: Enable model interpretability
- `--verbose`: Verbose output

### Prediction Command

```bash
python -m automl_lite.cli.main predict [OPTIONS] MODEL DATA
```

**Arguments:**
- `MODEL`: Path to trained model file
- `DATA`: Path to prediction data file

**Options:**
- `--output PATH`: Output predictions file path (default: predictions.csv)
- `--proba`: Output prediction probabilities

### Report Command

```bash
python -m automl_lite.cli.main report [OPTIONS] MODEL
```

**Arguments:**
- `MODEL`: Path to trained model file

**Options:**
- `--output PATH`: Output report file path (default: report.html)

### Interactive Command

```bash
python -m automl_lite.cli.main interactive
```

Launches an interactive session for guided model training and analysis.

## Configuration

### Configuration File Format (YAML)

```yaml
# AutoML Configuration
problem_type: classification
time_budget: 600
max_models: 15
cv_folds: 5
random_state: 42

# Advanced Features
enable_ensemble: true
enable_feature_selection: true
enable_interpretability: true
enable_early_stopping: true

# Model Parameters
models:
  - RandomForest
  - XGBoost
  - LightGBM
  - SVM
  - NeuralNetwork

# Feature Selection
feature_selection:
  method: mutual_info
  threshold: 0.01
  max_features: 20

# Ensemble
ensemble:
  method: voting
  top_k: 3
  voting: soft

# Early Stopping
early_stopping:
  patience: 10
  min_delta: 0.001
```

### Loading Configuration

```python
# From file
automl = AutoMLite.from_config('config.yaml')

# From dictionary
config = {
    'problem_type': 'classification',
    'time_budget': 600,
    'enable_ensemble': True
}
automl = AutoMLite.from_config(config)
```

## Data Types

### Supported Input Formats

#### Pandas DataFrame
```python
import pandas as pd

data = pd.read_csv('data.csv')
automl.fit(data, target_column='target')
```

#### NumPy Arrays
```python
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([0, 1])
automl.fit(X, y)
```

#### Mixed Input
```python
X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
y = np.array([0, 1, 0])
automl.fit(X, y)
```

### Output Formats

#### Predictions
```python
# Classification
predictions = automl.predict(X)  # Returns class labels
probabilities = automl.predict_proba(X)  # Returns probabilities

# Regression
predictions = automl.predict(X)  # Returns continuous values
```

#### Feature Importance
```python
importance = automl.get_feature_importance()
# Returns: {'feature1': 0.5, 'feature2': 0.3, ...}
```

#### Leaderboard
```python
leaderboard = automl.get_leaderboard()
# Returns: [
#     {'model_name': 'RandomForest', 'score': 0.85, 'params': {...}},
#     {'model_name': 'XGBoost', 'score': 0.83, 'params': {...}},
#     ...
# ]
```

## Error Handling

### Common Exceptions

#### `ValueError`
Raised when input data is invalid or missing required parameters.

```python
try:
    automl.fit(data)  # Missing target_column
except ValueError as e:
    print(f"Invalid input: {e}")
```

#### `NotImplementedError`
Raised when trying to use `predict_proba` on models that don't support it.

```python
try:
    probabilities = automl.predict_proba(X)
except NotImplementedError as e:
    print(f"Probability predictions not available: {e}")
```

#### `FileNotFoundError`
Raised when trying to load a non-existent model file.

```python
try:
    automl.load_model('nonexistent.pkl')
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
```

### Error Recovery

#### Graceful Fallbacks
```python
# AutoML Lite automatically handles many error cases
automl = AutoMLite(enable_ensemble=True)

# If some models don't support predict_proba, ensemble falls back to hard voting
automl.fit(data, target_column='target')
# No error - automatically uses hard voting if needed
```

#### Debug Mode
```python
# Enable verbose output for debugging
automl = AutoMLite(verbose=True)
automl.fit(data, target_column='target')
```

### Best Practices

1. **Always specify problem_type** for better performance
2. **Use appropriate time_budget** based on dataset size
3. **Enable features gradually** to identify issues
4. **Check data quality** before training
5. **Save models** after successful training
6. **Generate reports** for analysis and documentation

## Performance Considerations

### Memory Usage
- Large datasets: Use `max_models` to limit memory usage
- Feature selection: Reduces memory footprint
- Batch processing: Process data in chunks

### Training Time
- Time budget: Set appropriate limits
- Early stopping: Reduces unnecessary training
- Parallel processing: Uses all available cores by default

### Model Selection
- Start with fewer models for quick prototyping
- Increase model count for better performance
- Use ensemble methods for improved accuracy 
# Technology Stack

## Build System

- **Package Manager**: setuptools with pyproject.toml (PEP 621)
- **Build Backend**: setuptools.build_meta
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12

## Core Dependencies

- **ML Framework**: scikit-learn (>=1.0.0)
- **Optimization**: optuna (>=3.0.0)
- **Data Processing**: pandas (>=1.3.0), numpy (>=1.21.0)
- **Visualization**: matplotlib (>=3.5.0), seaborn (>=0.11.0), plotly (>=5.0.0)
- **Interpretability**: shap (>=0.41.0)
- **Utilities**: joblib, tqdm, jinja2, scipy, category-encoders, imbalanced-learn

## Optional Dependencies

- **Deep Learning**: tensorflow (>=2.8.0), torch (>=1.12.0)
- **Time Series**: statsmodels (>=0.13.0), prophet (>=1.1.0)
- **Experiment Tracking**: mlflow (>=1.28.0), wandb (>=0.13.0)
- **Interactive UI**: streamlit (>=1.25.0)
- **CLI**: rich (for terminal UI)

## Development Tools

- **Testing**: pytest (>=7.0.0), pytest-cov (>=4.0.0)
- **Formatting**: black (line-length=88), isort (profile="black")
- **Linting**: flake8 (>=5.0.0)
- **Type Checking**: mypy (>=0.991)
- **Pre-commit**: pre-commit hooks for code quality
- **Documentation**: sphinx (>=5.0.0), sphinx-rtd-theme

## Common Commands

### Installation
```bash
# Install from source
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install with all optional features
pip install -e ".[dev,docs,examples]"
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=automl_lite --cov-report=html --cov-report=xml

# Run specific test file
pytest tests/unit/test_automl.py

# Run in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/

# Lint code
flake8 src/ tests/

# Type check
mypy src/

# Run all pre-commit hooks
pre-commit run --all-files
```

### CLI Usage
```bash
# Interactive mode
automl-lite interactive

# Train model
automl-lite train data.csv --target target_column --output model.pkl

# Make predictions
automl-lite predict model.pkl test_data.csv --output predictions.csv

# Generate report
automl-lite report model.pkl --output report.html

# Validate data
automl-lite validate data.csv --target target_column
```

### Building and Distribution
```bash
# Build package
python -m build

# Install locally
pip install dist/automl_lite-*.whl

# Upload to PyPI (maintainers only)
twine upload dist/*
```

## Code Style Guidelines

- **Line Length**: 88 characters (black default)
- **Docstrings**: Google style
- **Type Hints**: Required for all public functions and methods
- **Import Organization**: isort with black profile
- **Naming Conventions**: PEP 8 (snake_case for functions/variables, PascalCase for classes)

## Testing Standards

- **Test Structure**: Arrange-Act-Assert pattern
- **Test Naming**: Descriptive names starting with `test_`
- **Test Organization**: Unit tests in `tests/unit/`, integration tests in `tests/integration/`
- **Coverage Target**: Aim for high coverage, exclude test files and __pycache__
- **Mocking**: Use unittest.mock for external dependencies

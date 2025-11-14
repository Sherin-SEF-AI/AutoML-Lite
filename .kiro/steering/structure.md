# Project Structure

## Source Code Organization

```
src/automl_lite/          # Main package source
├── __init__.py           # Package entry point, exports AutoMLite
├── core/                 # Core AutoML functionality
│   ├── automl.py         # Main AutoMLite class (915 lines)
├── cli/                  # Command-line interface
│   ├── __main__.py       # CLI entry point
│   ├── main.py           # CLI commands and argument parsing
├── config/               # Configuration management
│   ├── advanced_config.py
│   ├── config_manager.py
│   └── templates/        # YAML config templates (basic, production, research, etc.)
├── models/               # Model selection and algorithms
│   ├── selector.py       # Model selection logic
│   ├── deep_learning.py  # TensorFlow/PyTorch integration
│   └── time_series.py    # Time series forecasting models
├── preprocessing/        # Data preprocessing
│   ├── pipeline.py       # Preprocessing pipeline
│   └── feature_engineering.py  # Auto feature engineering
├── optimization/         # Hyperparameter optimization
│   └── optimizer.py      # Optuna-based optimization
├── visualization/        # Reporting and visualization
│   └── reporter.py       # HTML report generation
├── interpretability/     # Model interpretability
│   └── advanced_interpreter.py  # SHAP, LIME analysis
├── experiments/          # Experiment tracking
│   └── tracker.py        # MLflow, W&B integration
├── ui/                   # User interfaces
│   ├── interactive_dashboard.py  # Dashboard UI
│   └── terminal_ui.py    # Terminal-based UI
├── utils/                # Utility modules
│   ├── data_analyzer.py
│   ├── logger.py
│   ├── model_comparator.py
│   ├── problem_detector.py  # Auto problem type detection
│   └── validators.py
├── static/               # Static assets for UI
└── templates/            # HTML templates for reports
```

## Key Directories

### `/tests/`
- `unit/` - Unit tests for individual components
- `integration/` - Integration tests for workflows
- Test files follow `test_*.py` naming convention

### `/docs/`
- `API_REFERENCE.md` - API documentation
- `EXAMPLES.md` - Usage examples
- `INSTALLATION.md` - Installation guide
- `USER_GUIDE.md` - Comprehensive user guide
- `api/` - Generated API docs
- `source/` - Sphinx documentation source

### `/examples/`
- Python scripts demonstrating features
- Jupyter notebooks for tutorials
- Sample data files (test_data.csv)

### `/experiments/`
- Local experiment tracking data
- Organized by experiment name and run ID
- Contains artifacts, metrics, models, params

### `/mlruns/`
- MLflow experiment tracking data
- Organized by experiment ID
- Contains run metadata, artifacts, metrics

### `/model/`
- Saved model artifacts (.pkl, .joblib)
- Model metadata (model_info.json, feature_names.json)
- Preprocessor and sample data

### `/dist/`
- Built distribution packages (.whl, .tar.gz)

## Configuration Files

- `pyproject.toml` - Modern Python project configuration (PEP 621)
- `setup.py` - Minimal setup script (delegates to pyproject.toml)
- `requirements.txt` - Core dependencies
- `requirements_inference.txt` - Minimal inference dependencies
- `.gitignore` - Git ignore patterns
- `MANIFEST.in` - Package manifest for distribution

## Documentation Files

- `README.md` - Main project documentation
- `CONTRIBUTING.md` - Contribution guidelines
- `CHANGELOG.md` - Version history
- `LICENSE` - MIT license
- `PRODUCTION_SUMMARY.md` - Production features summary
- Various guide files for specific topics (HuggingFace, PyPI, sharing)

## Architecture Patterns

### Main Entry Point
- `AutoMLite` class in `src/automl_lite/core/automl.py` is the primary interface
- Exported from `src/automl_lite/__init__.py` for easy import

### Component Pattern
- Each major feature is a separate component (preprocessor, selector, optimizer, etc.)
- Components are initialized in `AutoMLite.__init__()` and used in `fit()`
- Components can be enabled/disabled via constructor flags

### Pipeline Pattern
- Data flows through: validation → preprocessing → feature engineering → model selection → optimization → ensemble
- Each stage is modular and can be configured independently

### Configuration Management
- YAML-based configuration templates in `src/automl_lite/config/templates/`
- ConfigManager handles loading/saving configurations
- Supports both programmatic and file-based configuration

### CLI Structure
- Rich-based terminal UI for better user experience
- Subcommands: train, predict, report, compare, batch, validate, info, interactive
- Argument parsing with argparse, output formatting with rich

### Model Persistence
- Models saved with joblib (.pkl files)
- Includes model, preprocessor, metadata, and configuration
- Load/save methods on AutoMLite class

## Naming Conventions

- **Classes**: PascalCase (e.g., `AutoMLite`, `ModelSelector`)
- **Functions/Methods**: snake_case (e.g., `fit()`, `predict()`, `generate_report()`)
- **Private Methods**: Leading underscore (e.g., `_validate_input()`, `_optimize_models()`)
- **Constants**: UPPER_SNAKE_CASE (rare in this codebase)
- **Files**: snake_case (e.g., `automl.py`, `model_selector.py`)

## Import Patterns

```python
# Relative imports within package
from ..preprocessing.pipeline import PreprocessingPipeline
from ..models.selector import ModelSelector
from ..utils.logger import get_logger

# External imports organized by: stdlib, third-party, local
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator

from ..core.automl import AutoMLite
```

## Output Artifacts

- **Models**: `*.pkl` files (joblib serialized)
- **Reports**: `*.html` files (interactive visualizations)
- **Predictions**: `*.csv` files
- **Configs**: `*.yaml` or `*.json` files
- **Experiments**: Organized in `experiments/` and `mlruns/` directories

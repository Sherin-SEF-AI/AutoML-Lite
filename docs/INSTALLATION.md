# AutoML Lite Installation Guide

Complete installation and setup instructions for AutoML Lite.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### Recommended Requirements

- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **Storage**: 5GB free space
- **CPU**: Multi-core processor (4+ cores recommended)

### Python Version Check

```bash
python --version
# Should show Python 3.8 or higher
```

## Installation Methods

### Method 1: Install from Source (Recommended)

#### Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/Sherin-SEF-AI/AutoML-Lite.git

# Navigate to the project directory
cd AutoML-Lite
```

#### Step 2: Create Virtual Environment

**On Linux/macOS:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

**On Windows:**
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

#### Step 4: Verify Installation

```bash
# Test the installation
python -c "from automl_lite import AutoMLite; print('AutoML Lite installed successfully!')"

# Test CLI
python -m automl_lite.cli.main --help
```

### Method 2: Install with pip (Future Release)

```bash
# Install directly from PyPI (when available)
pip install automl-lite
```

### Method 3: Install with conda (Future Release)

```bash
# Install with conda (when available)
conda install -c conda-forge automl-lite
```

## Dependencies

### Core Dependencies

AutoML Lite requires the following core dependencies:

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | >= 1.0.0 | Machine learning algorithms |
| pandas | >= 1.3.0 | Data manipulation |
| numpy | >= 1.21.0 | Numerical computing |
| optuna | >= 3.0.0 | Hyperparameter optimization |
| plotly | >= 5.0.0 | Interactive visualizations |
| seaborn | >= 0.11.0 | Statistical visualizations |
| matplotlib | >= 3.5.0 | Basic plotting |
| jinja2 | >= 3.0.0 | HTML template rendering |
| joblib | >= 1.1.0 | Model persistence |

### Optional Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| xgboost | >= 1.5.0 | Gradient boosting (optional) |
| lightgbm | >= 3.3.0 | Light gradient boosting (optional) |
| shap | >= 0.40.0 | Model interpretability (optional) |

### Development Dependencies

For development and testing:

```bash
# Install development dependencies
pip install -r requirements-dev.txt
```

Development dependencies include:
- pytest (testing)
- flake8 (linting)
- black (code formatting)
- mypy (type checking)

## Configuration

### Environment Variables

Set these environment variables for optimal performance:

```bash
# Set number of CPU cores to use
export OMP_NUM_THREADS=4

# Set memory limit for numpy
export MKL_NUM_THREADS=4

# Enable parallel processing
export JOBLIB_NUM_THREADS=4
```

**On Windows:**
```cmd
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set JOBLIB_NUM_THREADS=4
```

### Configuration File

Create a configuration file for custom settings:

```yaml
# config.yaml
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

## Verification

### Basic Functionality Test

Create a test script to verify the installation:

```python
# test_installation.py
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

def test_basic_functionality():
    """Test basic AutoML Lite functionality"""
    
    # Create test data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'target': np.random.binomial(1, 0.3, 100)
    })
    
    # Initialize AutoML
    automl = AutoMLite(
        problem_type='classification',
        time_budget=30,
        max_models=3
    )
    
    # Train model
    automl.fit(data, target_column='target')
    
    # Make predictions
    predictions = automl.predict(data.iloc[:10])
    
    # Generate report
    automl.generate_report('test_report.html')
    
    print("✅ Basic functionality test passed!")
    print(f"Best model: {automl.best_model_name}")
    print(f"Best score: {automl.best_score:.4f}")
    print(f"Predictions shape: {predictions.shape}")
    
    return True

if __name__ == "__main__":
    test_basic_functionality()
```

Run the test:
```bash
python test_installation.py
```

### CLI Test

Test the command-line interface:

```bash
# Test CLI help
python -m automl_lite.cli.main --help

# Test training command help
python -m automl_lite.cli.main train --help

# Test prediction command help
python -m automl_lite.cli.main predict --help

# Test report command help
python -m automl_lite.cli.main report --help
```

### Advanced Features Test

Test advanced features:

```python
# test_advanced_features.py
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

def test_advanced_features():
    """Test advanced AutoML Lite features"""
    
    # Create test data
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 200),
        'feature2': np.random.normal(0, 1, 200),
        'feature3': np.random.normal(0, 1, 200),
        'feature4': np.random.normal(0, 1, 200),
        'feature5': np.random.normal(0, 1, 200),
        'target': np.random.binomial(1, 0.3, 200)
    })
    
    # Test with all advanced features
    automl = AutoMLite(
        problem_type='classification',
        time_budget=60,
        max_models=5,
        enable_ensemble=True,
        enable_feature_selection=True,
        enable_interpretability=True
    )
    
    # Train model
    automl.fit(data, target_column='target')
    
    # Test feature selection
    if hasattr(automl, 'selected_features'):
        print(f"✅ Feature selection: {len(automl.selected_features)} features selected")
    
    # Test ensemble
    if hasattr(automl, 'ensemble_model'):
        print("✅ Ensemble model created")
    
    # Test interpretability
    try:
        interpretability = automl.get_interpretability_results()
        print("✅ Interpretability features available")
    except:
        print("⚠️ Interpretability not available")
    
    # Test predictions
    predictions = automl.predict(data.iloc[:10])
    print(f"✅ Predictions: {predictions.shape}")
    
    # Test probabilities
    try:
        probabilities = automl.predict_proba(data.iloc[:10])
        print(f"✅ Probabilities: {probabilities.shape}")
    except NotImplementedError:
        print("⚠️ Probabilities not available for this model")
    
    # Generate comprehensive report
    automl.generate_report('advanced_test_report.html')
    print("✅ Advanced features test passed!")
    
    return True

if __name__ == "__main__":
    test_advanced_features()
```

## Troubleshooting

### Common Installation Issues

#### 1. Python Version Issues

**Problem**: "Python version not supported"

**Solution**:
```bash
# Check Python version
python --version

# If version is too old, upgrade Python
# On Ubuntu/Debian:
sudo apt update
sudo apt install python3.9 python3.9-venv

# On macOS with Homebrew:
brew install python@3.9

# On Windows: Download from python.org
```

#### 2. Virtual Environment Issues

**Problem**: "venv module not found"

**Solution**:
```bash
# Install venv module
sudo apt install python3-venv  # Ubuntu/Debian
# OR
pip install virtualenv
python -m virtualenv venv
```

#### 3. Permission Issues

**Problem**: "Permission denied" errors

**Solution**:
```bash
# Fix permissions
chmod +x venv/bin/activate

# Or use sudo (not recommended for pip install)
sudo pip install -r requirements.txt
```

#### 4. Memory Issues During Installation

**Problem**: "Out of memory" during installation

**Solution**:
```bash
# Install with reduced memory usage
pip install --no-cache-dir -r requirements.txt

# Or install packages one by one
pip install scikit-learn
pip install pandas
pip install numpy
# ... etc
```

#### 5. Compilation Issues

**Problem**: "Failed to build wheel" errors

**Solution**:
```bash
# Install build tools
sudo apt install build-essential python3-dev  # Ubuntu/Debian
# OR
xcode-select --install  # macOS

# Install with pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

### Dependency Issues

#### 1. scikit-learn Installation Issues

**Problem**: scikit-learn fails to install

**Solution**:
```bash
# Install system dependencies
sudo apt install libblas-dev liblapack-dev libatlas-base-dev gfortran

# Install scikit-learn with specific version
pip install scikit-learn==1.1.3
```

#### 2. XGBoost/LightGBM Issues

**Problem**: XGBoost or LightGBM fails to install

**Solution**:
```bash
# Install system dependencies
sudo apt install cmake build-essential

# Install with conda (alternative)
conda install -c conda-forge xgboost lightgbm
```

#### 3. SHAP Installation Issues

**Problem**: SHAP fails to install

**Solution**:
```bash
# Install system dependencies
sudo apt install libgomp1

# Install SHAP with specific version
pip install shap==0.41.0
```

### Runtime Issues

#### 1. Import Errors

**Problem**: "ModuleNotFoundError"

**Solution**:
```bash
# Check if virtual environment is activated
which python

# Reinstall the package
pip uninstall automl-lite
pip install -e .
```

#### 2. Memory Issues

**Problem**: "OutOfMemoryError"

**Solution**:
```python
# Reduce memory usage in code
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# Use smaller datasets for testing
automl = AutoMLite(max_models=3)
```

#### 3. Performance Issues

**Problem**: Slow training

**Solution**:
```python
# Optimize performance
automl = AutoMLite(
    n_jobs=4,  # Use fewer cores
    time_budget=300,  # Reduce time budget
    max_models=5  # Reduce number of models
)
```

### Platform-Specific Issues

#### Windows Issues

**Problem**: Path length issues

**Solution**:
```bash
# Enable long paths in Windows
# Run as administrator:
fsutil behavior set SymlinkEvaluation L2L:1 R2R:1 L2R:1 R2L:1

# Or use shorter paths
cd C:\temp
git clone https://github.com/Sherin-SEF-AI/AutoML-Lite.git
```

#### macOS Issues

**Problem**: SSL certificate issues

**Solution**:
```bash
# Install certificates
pip install certifi

# Or use conda
conda install certifi
```

#### Linux Issues

**Problem**: Library dependency issues

**Solution**:
```bash
# Install system libraries
sudo apt update
sudo apt install python3-dev libblas-dev liblapack-dev libatlas-base-dev gfortran

# Or use conda
conda install -c conda-forge blas lapack
```

### Getting Help

If you encounter issues not covered here:

1. **Check the logs**: Look for detailed error messages
2. **Search issues**: Check existing GitHub issues
3. **Create new issue**: Provide detailed information including:
   - Operating system and version
   - Python version
   - Error messages
   - Steps to reproduce
4. **Contact support**: Email sherin@deepmost.ai or visit [sherinjosephroy.link/contact](https://sherinjosephroy.link/contact)

### Performance Optimization

#### For Large Datasets

```python
# Use memory-efficient settings
automl = AutoMLite(
    max_models=5,
    enable_ensemble=False,
    enable_feature_selection=True,
    n_jobs=2
)
```

#### For Fast Prototyping

```python
# Use quick settings
automl = AutoMLite(
    time_budget=60,
    max_models=3,
    enable_ensemble=False,
    enable_feature_selection=False
)
```

#### For Production

```python
# Use production settings
automl = AutoMLite(
    time_budget=600,
    max_models=15,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True,
    cv_folds=5
)
```

This installation guide covers all aspects of setting up AutoML Lite. Follow the steps carefully and refer to the troubleshooting section if you encounter any issues. 
# AutoML Lite User Guide

A comprehensive guide to using AutoML Lite effectively for your machine learning projects.

## Table of Contents

- [Getting Started](#getting-started)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Making Predictions](#making-predictions)
- [Generating Reports](#generating-reports)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sherin-SEF-AI/AutoML-Lite.git
cd AutoML-Lite
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install -e .
```

### Quick Test

Test your installation with a simple example:

```python
from automl_lite import AutoMLite
import pandas as pd
import numpy as np

# Create simple test data
np.random.seed(42)
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 100),
    'feature2': np.random.normal(0, 1, 100),
    'target': np.random.binomial(1, 0.3, 100)
})

# Train model
automl = AutoMLite(problem_type='classification', time_budget=30)
automl.fit(data, target_column='target')

print(f"Best model: {automl.best_model_name}")
print(f"Best score: {automl.best_score:.4f}")
```

## Data Preparation

### Supported Data Formats

AutoML Lite supports various data formats:

1. **CSV Files** (Recommended)
```python
import pandas as pd
data = pd.read_csv('your_data.csv')
```

2. **Excel Files**
```python
data = pd.read_excel('your_data.xlsx')
```

3. **NumPy Arrays**
```python
import numpy as np
X = np.array([[1, 2, 3], [4, 5, 6]])
y = np.array([0, 1])
```

### Data Requirements

#### Required Format
- **Target column**: Must be present in your data
- **Features**: All other columns will be used as features
- **Data types**: AutoML Lite handles mixed data types automatically

#### Data Quality Checklist

✅ **Clean your data before training:**

1. **Remove duplicates**
```python
data = data.drop_duplicates()
```

2. **Handle missing values**
```python
# AutoML Lite handles missing values automatically, but you can preprocess:
data = data.dropna()  # Remove rows with missing values
# OR
data = data.fillna(data.mean())  # Fill with mean values
```

3. **Check data types**
```python
print(data.dtypes)
```

4. **Remove unnecessary columns**
```python
# Remove ID columns, timestamps, etc.
data = data.drop(['id', 'timestamp'], axis=1)
```

### Example Data Preparation

```python
import pandas as pd
import numpy as np

# Load raw data
raw_data = pd.read_csv('customer_data.csv')

# Data cleaning
cleaned_data = raw_data.copy()

# Remove duplicates
cleaned_data = cleaned_data.drop_duplicates()

# Handle missing values
cleaned_data = cleaned_data.fillna(cleaned_data.mean())

# Remove unnecessary columns
columns_to_drop = ['customer_id', 'created_date', 'last_login']
cleaned_data = cleaned_data.drop(columns_to_drop, axis=1)

# Check final data
print(f"Original shape: {raw_data.shape}")
print(f"Cleaned shape: {cleaned_data.shape}")
print(f"Missing values: {cleaned_data.isnull().sum().sum()}")

# Save cleaned data
cleaned_data.to_csv('cleaned_customer_data.csv', index=False)
```

## Training Models

### Basic Training

#### Using CLI (Recommended for beginners)

```bash
# Basic training
python -m automl_lite.cli.main train data.csv --target target_column --output model.pkl

# Advanced training with all features
python -m automl_lite.cli.main train data.csv \
    --target target_column \
    --output model.pkl \
    --enable-ensemble \
    --enable-feature-selection \
    --enable-interpretability \
    --time-budget 600 \
    --max-models 15 \
    --verbose
```

#### Using Python API

```python
from automl_lite import AutoMLite
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Initialize AutoML
automl = AutoMLite(
    problem_type='classification',  # or 'regression'
    time_budget=300,
    max_models=10,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True
)

# Train model
automl.fit(data, target_column='target')

# Save model
automl.save_model('trained_model.pkl')
```

### Configuration Options

#### Time Budget
- **Small datasets (< 1000 samples)**: 60-120 seconds
- **Medium datasets (1000-10000 samples)**: 300-600 seconds
- **Large datasets (> 10000 samples)**: 600-1800 seconds

#### Number of Models
- **Quick prototyping**: 5-10 models
- **Production**: 10-20 models
- **Research**: 20+ models

#### Cross-Validation Folds
- **Small datasets**: 3-5 folds
- **Large datasets**: 5-10 folds

### Example Training Workflows

#### 1. Quick Prototyping

```python
# Fast training for initial exploration
automl = AutoMLite(
    problem_type='classification',
    time_budget=60,
    max_models=5,
    enable_ensemble=False
)

automl.fit(data, target_column='target')
print(f"Quick prototype score: {automl.best_score:.4f}")
```

#### 2. Production Training

```python
# Comprehensive training for production
automl = AutoMLite(
    problem_type='classification',
    time_budget=600,
    max_models=15,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True,
    cv_folds=5
)

automl.fit(data, target_column='target')
automl.save_model('production_model.pkl')
```

#### 3. Research/Experimentation

```python
# Extensive training for research
automl = AutoMLite(
    problem_type='classification',
    time_budget=1800,
    max_models=25,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True,
    cv_folds=10
)

automl.fit(data, target_column='target')
```

## Making Predictions

### Basic Predictions

#### Using CLI

```bash
# Regular predictions
python -m automl_lite.cli.main predict model.pkl test_data.csv --output predictions.csv

# Probability predictions (classification only)
python -m automl_lite.cli.main predict model.pkl test_data.csv --output probabilities.csv --proba
```

#### Using Python API

```python
# Load trained model
automl = AutoMLite()
automl.load_model('trained_model.pkl')

# Load test data
test_data = pd.read_csv('test_data.csv')

# Make predictions
predictions = automl.predict(test_data)

# Get probabilities (classification only)
try:
    probabilities = automl.predict_proba(test_data)
    print("Probabilities available")
except NotImplementedError:
    print("Probabilities not available for this model")
```

### Batch Processing

```python
import pandas as pd
from automl_lite import AutoMLite

def batch_predict(model_path, data_path, output_path, batch_size=1000):
    """Process predictions in batches to handle large datasets"""
    
    # Load model
    automl = AutoMLite()
    automl.load_model(model_path)
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Process in batches
    all_predictions = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i+batch_size]
        predictions = automl.predict(batch)
        all_predictions.extend(predictions)
        
        print(f"Processed batch {i//batch_size + 1}/{(len(data)-1)//batch_size + 1}")
    
    # Save results
    results = pd.DataFrame({
        'predictions': all_predictions
    })
    results.to_csv(output_path, index=False)
    
    return results

# Usage
batch_predict('model.pkl', 'large_test_data.csv', 'batch_predictions.csv')
```

### Real-time Predictions

```python
from flask import Flask, request, jsonify
from automl_lite import AutoMLite
import pandas as pd

app = Flask(__name__)

# Load model at startup
automl = AutoMLite()
automl.load_model('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Convert to DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = automl.predict(input_data)[0]
    
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
```

## Generating Reports

### Basic Report Generation

#### Using CLI

```bash
python -m automl_lite.cli.main report model.pkl --output report.html
```

#### Using Python API

```python
# Generate basic report
automl.generate_report('basic_report.html')

# Generate report with test data
automl.generate_report('comprehensive_report.html', X_test, y_test)
```

### Report Components

The generated HTML report includes:

1. **Executive Summary**
   - Problem type
   - Best model and score
   - Training time
   - Number of models tried

2. **Model Performance**
   - Leaderboard of all models
   - Performance comparison charts
   - Cross-validation results

3. **Feature Analysis**
   - Feature importance rankings
   - Feature correlation matrix
   - Selected features (if feature selection enabled)

4. **Training History**
   - Model scores over time
   - Training time per model
   - Learning curves

5. **Advanced Analysis** (if enabled)
   - Ensemble information
   - Model interpretability results
   - SHAP values analysis

### Customizing Reports

```python
# Generate report with specific components
automl.generate_report(
    'custom_report.html',
    X_test=test_data,
    y_test=test_labels
)
```

## Advanced Features

### Ensemble Methods

Ensemble methods combine multiple models for better performance:

```python
automl = AutoMLite(enable_ensemble=True)

# The ensemble automatically:
# 1. Selects the best performing models
# 2. Creates a voting classifier
# 3. Handles predict_proba compatibility
# 4. Falls back to hard voting if needed

automl.fit(data, target_column='target')
print(f"Ensemble score: {automl.best_score:.4f}")
```

### Feature Selection

Automatic feature selection improves performance and reduces overfitting:

```python
automl = AutoMLite(enable_feature_selection=True)

automl.fit(data, target_column='target')

# Check selected features
print(f"Original features: {len(data.columns) - 1}")
print(f"Selected features: {len(automl.selected_features)}")
print("Selected features:", automl.selected_features)
```

### Model Interpretability

Understand your model's decisions:

```python
automl = AutoMLite(enable_interpretability=True)

automl.fit(data, target_column='target')

# Get interpretability results
interpretability = automl.get_interpretability_results()

# Available analyses:
# - SHAP values
# - Feature effects
# - Model complexity metrics
```

### Early Stopping

Optimize training time with early stopping:

```python
automl = AutoMLite(
    enable_early_stopping=True,
    patience=10,
    min_delta=0.001
)

# Training stops early if no improvement for 'patience' trials
automl.fit(data, target_column='target')
```

## Best Practices

### 1. Data Quality

- **Clean your data** before training
- **Handle missing values** appropriately
- **Remove duplicates** and outliers
- **Check data types** and convert if needed

### 2. Training Strategy

- **Start simple**: Use basic settings for initial exploration
- **Iterate**: Gradually add advanced features
- **Monitor**: Watch training progress and adjust parameters
- **Validate**: Always test on unseen data

### 3. Model Selection

- **Use appropriate time budget** for your dataset size
- **Enable ensemble** for better performance
- **Use feature selection** for high-dimensional data
- **Enable interpretability** for business applications

### 4. Production Deployment

- **Save models** after successful training
- **Version control** your models and configurations
- **Monitor performance** in production
- **Retrain periodically** with new data

### 5. Performance Optimization

```python
# Memory optimization for large datasets
automl = AutoMLite(
    max_models=5,  # Reduce number of models
    enable_feature_selection=True,  # Reduce feature space
    enable_ensemble=False  # Disable ensemble to save memory
)

# Time optimization
automl = AutoMLite(
    time_budget=300,  # Set reasonable time limit
    enable_early_stopping=True,  # Stop early if no improvement
    patience=5  # Reduce patience for faster training
)
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Symptoms**: OutOfMemoryError or slow performance

**Solutions**:
```python
# Reduce memory usage
automl = AutoMLite(
    max_models=5,  # Fewer models
    enable_ensemble=False,  # Disable ensemble
    enable_feature_selection=True  # Reduce features
)

# Process data in chunks
for chunk in pd.read_csv('large_data.csv', chunksize=1000):
    # Process each chunk
    pass
```

#### 2. Time Budget Exceeded

**Symptoms**: Training stops before completion

**Solutions**:
```python
# Increase time budget
automl = AutoMLite(time_budget=1200)  # 20 minutes

# Use early stopping
automl = AutoMLite(
    enable_early_stopping=True,
    patience=5
)

# Reduce number of models
automl = AutoMLite(max_models=5)
```

#### 3. Poor Performance

**Symptoms**: Low accuracy scores

**Solutions**:
```python
# Check data quality
print(data.isnull().sum())
print(data.dtypes)

# Try different configurations
automl = AutoMLite(
    enable_ensemble=True,
    enable_feature_selection=True,
    cv_folds=10  # More robust validation
)

# Increase training time
automl = AutoMLite(
    time_budget=600,
    max_models=15
)
```

#### 4. Model Compatibility Issues

**Symptoms**: predict_proba not available

**Solutions**:
```python
# AutoML Lite handles this automatically
# But you can check compatibility:
try:
    probabilities = automl.predict_proba(X)
except NotImplementedError:
    print("Using regular predictions instead")
    predictions = automl.predict(X)
```

### Debug Mode

Enable verbose output for debugging:

```python
automl = AutoMLite(verbose=True)
automl.fit(data, target_column='target')
```

### Getting Help

1. **Check the logs**: Look for error messages and warnings
2. **Reduce complexity**: Start with basic settings
3. **Check data**: Ensure your data is properly formatted
4. **Update dependencies**: Make sure you have the latest versions
5. **Report issues**: Create an issue on GitHub with details

### Performance Tips

1. **Use appropriate data types**: Convert strings to categories
2. **Remove unnecessary columns**: Drop ID columns and timestamps
3. **Handle missing values**: Fill or remove missing data
4. **Scale features**: Some algorithms benefit from scaling
5. **Use cross-validation**: More robust than single train/test split

This user guide covers the essential aspects of using AutoML Lite effectively. Start with the basic examples and gradually explore advanced features as you become more comfortable with the library. 
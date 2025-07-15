# AutoML Lite Examples

This guide provides comprehensive examples for different use cases and scenarios with AutoML Lite.

## Table of Contents

- [Basic Examples](#basic-examples)
- [Classification Examples](#classification-examples)
- [Regression Examples](#regression-examples)
- [Advanced Features](#advanced-features)
- [Production Scenarios](#production-scenarios)
- [Troubleshooting Examples](#troubleshooting-examples)

## Basic Examples

### 1. Simple Classification

```python
from automl_lite import AutoMLite
import pandas as pd
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

# Initialize AutoML
automl = AutoMLite(
    problem_type='classification',
    time_budget=60,
    max_models=5
)

# Train model
automl.fit(data, target_column='target')

# Make predictions
predictions = automl.predict(data.iloc[:10])

# Print results
print(f"Best model: {automl.best_model_name}")
print(f"Best score: {automl.best_score:.4f}")
print(f"Predictions: {predictions}")
```

### 2. Simple Regression

```python
from automl_lite import AutoMLite
import pandas as pd
from sklearn.datasets import load_boston

# Load data
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['target'] = boston.target

# Initialize AutoML
automl = AutoMLite(
    problem_type='regression',
    time_budget=120,
    max_models=8
)

# Train model
automl.fit(data, target_column='target')

# Make predictions
predictions = automl.predict(data.iloc[:10])

# Print results
print(f"Best model: {automl.best_model_name}")
print(f"Best score: {automl.best_score:.4f}")
print(f"Predictions: {predictions}")
```

### 3. CLI Basic Workflow

```bash
# 1. Train a model
python -m automl_lite.cli.main train iris.csv --target target --output iris_model.pkl

# 2. Make predictions
python -m automl_lite.cli.main predict iris_model.pkl test_data.csv --output predictions.csv

# 3. Generate report
python -m automl_lite.cli.main report iris_model.pkl --output report.html
```

## Classification Examples

### 1. Customer Churn Prediction

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Simulate customer data
np.random.seed(42)
n_customers = 1000

data = pd.DataFrame({
    'age': np.random.normal(45, 15, n_customers),
    'tenure': np.random.exponential(5, n_customers),
    'monthly_charges': np.random.normal(70, 20, n_customers),
    'total_charges': np.random.normal(2000, 1000, n_customers),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers)
})

# Create target variable (churn)
data['churn'] = (
    (data['age'] < 30) | 
    (data['tenure'] < 1) | 
    (data['monthly_charges'] > 80) |
    (data['contract_type'] == 'Month-to-month')
).astype(int)

# Initialize AutoML with advanced features
automl = AutoMLite(
    problem_type='classification',
    time_budget=300,
    max_models=10,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True
)

# Train model
automl.fit(data, target_column='churn')

# Get feature importance
importance = automl.get_feature_importance()
print("Feature Importance:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.4f}")

# Generate comprehensive report
automl.generate_report('churn_analysis.html')

# Save model for production
automl.save_model('churn_model.pkl')
```

### 2. Email Spam Detection

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Simulate email features
np.random.seed(42)
n_emails = 2000

data = pd.DataFrame({
    'word_count': np.random.poisson(50, n_emails),
    'char_count': np.random.poisson(300, n_emails),
    'exclamation_count': np.random.poisson(2, n_emails),
    'capital_ratio': np.random.beta(2, 5, n_emails),
    'url_count': np.random.poisson(1, n_emails),
    'spam_words': np.random.poisson(3, n_emails),
    'sender_trust_score': np.random.beta(5, 2, n_emails)
})

# Create spam labels
data['is_spam'] = (
    (data['exclamation_count'] > 3) |
    (data['capital_ratio'] > 0.3) |
    (data['url_count'] > 2) |
    (data['spam_words'] > 5) |
    (data['sender_trust_score'] < 0.3)
).astype(int)

# Initialize AutoML
automl = AutoMLite(
    problem_type='classification',
    time_budget=200,
    max_models=8,
    enable_ensemble=True,
    enable_feature_selection=True
)

# Train model
automl.fit(data, target_column='is_spam')

# Get predictions with probabilities
probabilities = automl.predict_proba(data.iloc[:10])
print("Spam probabilities for first 10 emails:")
for i, prob in enumerate(probabilities[:, 1]):
    print(f"  Email {i+1}: {prob:.3f}")

# Generate report
automl.generate_report('spam_detection_report.html')
```

### 3. Multi-class Classification

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Simulate multi-class data
np.random.seed(42)
n_samples = 1500

data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'feature4': np.random.normal(0, 1, n_samples)
})

# Create 3 classes based on feature combinations
data['class'] = np.where(
    data['feature1'] + data['feature2'] > 0,
    np.where(data['feature3'] > 0, 0, 1),
    2
)

# Initialize AutoML
automl = AutoMLite(
    problem_type='classification',
    time_budget=180,
    max_models=10,
    enable_ensemble=True
)

# Train model
automl.fit(data, target_column='class')

# Get leaderboard
leaderboard = automl.get_leaderboard()
print("Model Performance:")
for i, model in enumerate(leaderboard[:5]):
    print(f"  {i+1}. {model['model_name']}: {model['score']:.4f}")

# Generate report
automl.generate_report('multiclass_report.html')
```

## Regression Examples

### 1. House Price Prediction

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Simulate house data
np.random.seed(42)
n_houses = 2000

data = pd.DataFrame({
    'sqft': np.random.normal(2000, 500, n_houses),
    'bedrooms': np.random.poisson(3, n_houses),
    'bathrooms': np.random.poisson(2, n_houses),
    'age': np.random.exponential(20, n_houses),
    'distance_to_city': np.random.exponential(10, n_houses),
    'crime_rate': np.random.beta(2, 8, n_houses),
    'school_rating': np.random.beta(8, 2, n_houses)
})

# Create house prices
base_price = 200000
data['price'] = (
    base_price +
    data['sqft'] * 100 +
    data['bedrooms'] * 15000 +
    data['bathrooms'] * 25000 -
    data['age'] * 1000 -
    data['distance_to_city'] * 5000 -
    data['crime_rate'] * 50000 +
    data['school_rating'] * 100000 +
    np.random.normal(0, 20000, n_houses)
)

# Initialize AutoML
automl = AutoMLite(
    problem_type='regression',
    time_budget=300,
    max_models=12,
    enable_ensemble=True,
    enable_feature_selection=True,
    enable_interpretability=True
)

# Train model
automl.fit(data, target_column='price')

# Get feature importance
importance = automl.get_feature_importance()
print("Feature Importance for House Prices:")
for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: {score:.4f}")

# Generate report
automl.generate_report('house_pricing_report.html')
```

### 2. Sales Forecasting

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Simulate sales data
np.random.seed(42)
n_periods = 1000

data = pd.DataFrame({
    'month': np.arange(1, n_periods + 1),
    'advertising_budget': np.random.normal(50000, 15000, n_periods),
    'competitor_price': np.random.normal(100, 20, n_periods),
    'seasonality': np.sin(2 * np.pi * np.arange(n_periods) / 12),
    'economic_index': np.random.normal(100, 10, n_periods),
    'customer_satisfaction': np.random.beta(8, 2, n_periods)
})

# Create sales volume with trend and seasonality
trend = np.linspace(1000, 2000, n_periods)
data['sales_volume'] = (
    trend +
    data['advertising_budget'] * 0.01 +
    data['seasonality'] * 200 -
    data['competitor_price'] * 5 +
    data['economic_index'] * 2 +
    data['customer_satisfaction'] * 500 +
    np.random.normal(0, 50, n_periods)
)

# Initialize AutoML
automl = AutoMLite(
    problem_type='regression',
    time_budget=240,
    max_models=10,
    enable_ensemble=True,
    enable_feature_selection=True
)

# Train model
automl.fit(data, target_column='sales_volume')

# Make predictions for next 3 months
future_data = pd.DataFrame({
    'month': [n_periods + 1, n_periods + 2, n_periods + 3],
    'advertising_budget': [60000, 65000, 70000],
    'competitor_price': [95, 98, 102],
    'seasonality': [np.sin(2 * np.pi * (n_periods + 1) / 12),
                   np.sin(2 * np.pi * (n_periods + 2) / 12),
                   np.sin(2 * np.pi * (n_periods + 3) / 12)],
    'economic_index': [105, 107, 108],
    'customer_satisfaction': [0.85, 0.87, 0.88]
})

predictions = automl.predict(future_data)
print("Sales Forecast for Next 3 Months:")
for i, pred in enumerate(predictions):
    print(f"  Month {n_periods + i + 1}: {pred:.0f} units")

# Generate report
automl.generate_report('sales_forecast_report.html')
```

## Advanced Features

### 1. Ensemble Methods

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Create complex dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'feature4': np.random.normal(0, 1, n_samples),
    'feature5': np.random.normal(0, 1, n_samples)
})

# Create complex target
data['target'] = (
    np.sin(data['feature1']) +
    np.cos(data['feature2']) +
    data['feature3'] ** 2 +
    data['feature4'] * data['feature5'] +
    np.random.normal(0, 0.1, n_samples)
)

# Initialize AutoML with ensemble
automl = AutoMLite(
    problem_type='regression',
    time_budget=300,
    max_models=15,
    enable_ensemble=True,
    enable_feature_selection=True
)

# Train model
automl.fit(data, target_column='target')

# Compare ensemble vs individual models
leaderboard = automl.get_leaderboard()
print("Model Performance Comparison:")
for i, model in enumerate(leaderboard):
    marker = "🏆" if "Ensemble" in model['model_name'] else "  "
    print(f"{marker} {model['model_name']}: {model['score']:.4f}")

# Generate report
automl.generate_report('ensemble_analysis.html')
```

### 2. Feature Selection

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Create dataset with irrelevant features
np.random.seed(42)
n_samples = 800

# Relevant features
relevant_features = pd.DataFrame({
    'important_feature1': np.random.normal(0, 1, n_samples),
    'important_feature2': np.random.normal(0, 1, n_samples),
    'important_feature3': np.random.normal(0, 1, n_samples)
})

# Irrelevant features
irrelevant_features = pd.DataFrame({
    'noise_feature1': np.random.normal(0, 1, n_samples),
    'noise_feature2': np.random.normal(0, 1, n_samples),
    'noise_feature3': np.random.normal(0, 1, n_samples),
    'noise_feature4': np.random.normal(0, 1, n_samples),
    'noise_feature5': np.random.normal(0, 1, n_samples)
})

# Combine features
data = pd.concat([relevant_features, irrelevant_features], axis=1)

# Create target using only relevant features
data['target'] = (
    data['important_feature1'] * 2 +
    data['important_feature2'] * 1.5 +
    data['important_feature3'] * 0.5 +
    np.random.normal(0, 0.1, n_samples)
)

# Initialize AutoML with feature selection
automl = AutoMLite(
    problem_type='regression',
    time_budget=200,
    max_models=10,
    enable_feature_selection=True
)

# Train model
automl.fit(data, target_column='target')

# Check selected features
print(f"Original features: {len(data.columns) - 1}")
print(f"Selected features: {len(automl.selected_features)}")
print("Selected features:")
for feature in automl.selected_features:
    print(f"  - {feature}")

# Generate report
automl.generate_report('feature_selection_report.html')
```

### 3. Model Interpretability

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Create interpretable dataset
np.random.seed(42)
n_samples = 1200

data = pd.DataFrame({
    'age': np.random.normal(45, 15, n_samples),
    'income': np.random.normal(60000, 20000, n_samples),
    'education_years': np.random.normal(14, 3, n_samples),
    'credit_score': np.random.normal(700, 100, n_samples),
    'debt_ratio': np.random.beta(2, 5, n_samples),
    'employment_length': np.random.exponential(5, n_samples)
})

# Create loan approval target
data['loan_approved'] = (
    (data['income'] > 50000) &
    (data['credit_score'] > 650) &
    (data['debt_ratio'] < 0.4) &
    (data['employment_length'] > 2)
).astype(int)

# Initialize AutoML with interpretability
automl = AutoMLite(
    problem_type='classification',
    time_budget=300,
    max_models=10,
    enable_interpretability=True,
    enable_ensemble=True
)

# Train model
automl.fit(data, target_column='loan_approved')

# Get interpretability results
interpretability = automl.get_interpretability_results()
print("Interpretability Analysis Available:")
for key in interpretability.keys():
    print(f"  - {key}")

# Generate comprehensive report
automl.generate_report('loan_approval_interpretability.html')
```

## Production Scenarios

### 1. Batch Processing Pipeline

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import joblib
from datetime import datetime

# Simulate production data pipeline
def load_training_data():
    """Load and preprocess training data"""
    # Simulate loading from database/file
    np.random.seed(42)
    n_samples = 5000
    
    data = pd.DataFrame({
        'customer_id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples),
        'target': np.random.binomial(1, 0.3, n_samples)
    })
    
    return data

def load_prediction_data():
    """Load new data for predictions"""
    np.random.seed(123)
    n_samples = 1000
    
    data = pd.DataFrame({
        'customer_id': range(n_samples),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.normal(0, 1, n_samples)
    })
    
    return data

# Training pipeline
print("Starting training pipeline...")
training_data = load_training_data()

automl = AutoMLite(
    problem_type='classification',
    time_budget=600,
    max_models=15,
    enable_ensemble=True,
    enable_feature_selection=True
)

# Train model
automl.fit(training_data, target_column='target')

# Save model and metadata
model_info = {
    'model_path': 'production_model.pkl',
    'training_date': datetime.now().isoformat(),
    'best_score': automl.best_score,
    'best_model': automl.best_model_name,
    'selected_features': automl.selected_features,
    'training_samples': len(training_data)
}

automl.save_model('production_model.pkl')
joblib.dump(model_info, 'model_metadata.pkl')

print(f"Model saved: {model_info}")

# Prediction pipeline
print("\nStarting prediction pipeline...")
prediction_data = load_prediction_data()

# Load model
loaded_automl = AutoMLite()
loaded_automl.load_model('production_model.pkl')

# Make predictions
predictions = loaded_automl.predict(prediction_data)
probabilities = loaded_automl.predict_proba(prediction_data)

# Save results
results = pd.DataFrame({
    'customer_id': prediction_data['customer_id'],
    'prediction': predictions,
    'probability': probabilities[:, 1]
})

results.to_csv('batch_predictions.csv', index=False)
print(f"Predictions saved for {len(results)} customers")
```

### 2. API Integration

```python
from flask import Flask, request, jsonify
from automl_lite import AutoMLite
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model at startup
automl = AutoMLite()
automl.load_model('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        input_data = pd.DataFrame(data['features'])
        
        # Make predictions
        predictions = automl.predict(input_data)
        probabilities = automl.predict_proba(input_data)
        
        # Prepare response
        response = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'model_info': {
                'model_name': automl.best_model_name,
                'score': automl.best_score
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

### 3. Model Monitoring

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import joblib
from datetime import datetime, timedelta

class ModelMonitor:
    def __init__(self, model_path, metadata_path):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.automl = AutoMLite()
        self.automl.load_model(model_path)
        self.metadata = joblib.load(metadata_path)
        
    def check_data_drift(self, new_data):
        """Check for data drift"""
        # Compare feature distributions
        drift_report = {}
        
        for feature in self.automl.selected_features:
            if feature in new_data.columns:
                old_mean = self.metadata.get(f'feature_stats', {}).get(feature, {}).get('mean', 0)
                new_mean = new_data[feature].mean()
                drift = abs(new_mean - old_mean) / (abs(old_mean) + 1e-8)
                drift_report[feature] = drift
        
        return drift_report
    
    def evaluate_performance(self, X_test, y_test):
        """Evaluate model performance on new data"""
        from sklearn.metrics import accuracy_score, classification_report
        
        predictions = self.automl.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, predictions),
            'performance_drop': self.metadata['best_score'] - accuracy
        }
    
    def generate_monitoring_report(self, new_data, X_test=None, y_test=None):
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': self.metadata,
            'data_drift': self.check_data_drift(new_data)
        }
        
        if X_test is not None and y_test is not None:
            report['performance'] = self.evaluate_performance(X_test, y_test)
        
        return report

# Usage example
monitor = ModelMonitor('production_model.pkl', 'model_metadata.pkl')

# Simulate new data
np.random.seed(123)
new_data = pd.DataFrame({
    'feature1': np.random.normal(0.1, 1, 1000),  # Slight drift
    'feature2': np.random.normal(0, 1, 1000),
    'feature3': np.random.normal(0, 1, 1000)
})

# Generate monitoring report
report = monitor.generate_monitoring_report(new_data)
print("Monitoring Report:")
print(f"Data drift detected: {any(drift > 0.1 for drift in report['data_drift'].values())}")
```

## Troubleshooting Examples

### 1. Memory Issues

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import psutil
import gc

def monitor_memory():
    """Monitor memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")

# Large dataset handling
print("Loading large dataset...")
monitor_memory()

# Create large dataset
np.random.seed(42)
n_samples = 50000
large_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'feature4': np.random.normal(0, 1, n_samples),
    'feature5': np.random.normal(0, 1, n_samples),
    'target': np.random.binomial(1, 0.3, n_samples)
})

monitor_memory()

# Use memory-efficient settings
automl = AutoMLite(
    problem_type='classification',
    time_budget=300,
    max_models=5,  # Reduce number of models
    enable_feature_selection=True,  # Reduce feature space
    enable_ensemble=False  # Disable ensemble to save memory
)

print("Training with memory-efficient settings...")
automl.fit(large_data, target_column='target')

monitor_memory()

# Clean up
del large_data
gc.collect()
monitor_memory()
```

### 2. Time Budget Issues

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import time

# Complex dataset that might take long to train
np.random.seed(42)
n_samples = 10000

complex_data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'feature4': np.random.normal(0, 1, n_samples),
    'feature5': np.random.normal(0, 1, n_samples),
    'feature6': np.random.normal(0, 1, n_samples),
    'feature7': np.random.normal(0, 1, n_samples),
    'feature8': np.random.normal(0, 1, n_samples),
    'target': np.random.binomial(1, 0.3, n_samples)
})

# Progressive time budget strategy
time_budgets = [60, 120, 300]  # Start small, increase if needed

for budget in time_budgets:
    print(f"\nTrying with {budget} second time budget...")
    
    automl = AutoMLite(
        problem_type='classification',
        time_budget=budget,
        max_models=10,
        enable_ensemble=True
    )
    
    start_time = time.time()
    try:
        automl.fit(complex_data, target_column='target')
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.1f} seconds")
        print(f"Best score: {automl.best_score:.4f}")
        
        if automl.best_score > 0.8:  # Good enough performance
            break
            
    except Exception as e:
        print(f"Training failed with {budget}s budget: {e}")
        continue
```

### 3. Model Compatibility Issues

```python
import pandas as pd
import numpy as np
from automl_lite import AutoMLite

# Dataset that might cause compatibility issues
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, n_samples),
    'feature2': np.random.normal(0, 1, n_samples),
    'feature3': np.random.normal(0, 1, n_samples),
    'target': np.random.binomial(1, 0.3, n_samples)
})

# Try different configurations
configurations = [
    {'enable_ensemble': False, 'enable_feature_selection': False},
    {'enable_ensemble': True, 'enable_feature_selection': False},
    {'enable_ensemble': False, 'enable_feature_selection': True},
    {'enable_ensemble': True, 'enable_feature_selection': True}
]

for i, config in enumerate(configurations):
    print(f"\nTrying configuration {i+1}: {config}")
    
    try:
        automl = AutoMLite(
            problem_type='classification',
            time_budget=120,
            max_models=5,
            **config
        )
        
        automl.fit(data, target_column='target')
        print(f"Success! Best score: {automl.best_score:.4f}")
        
        # Test predictions
        predictions = automl.predict(data.iloc[:10])
        print(f"Predictions work: {predictions}")
        
        # Test probabilities if available
        try:
            probabilities = automl.predict_proba(data.iloc[:10])
            print(f"Probabilities work: {probabilities.shape}")
        except NotImplementedError:
            print("Probabilities not available (expected for some models)")
        
        break
        
    except Exception as e:
        print(f"Configuration failed: {e}")
        continue
```

These examples demonstrate various use cases and scenarios with AutoML Lite, from basic usage to advanced production deployments. Each example includes error handling and best practices for real-world applications. 
---
title: AutoML Lite
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
tags:
- automl
- machine-learning
- deep-learning
- time-series
- classification
- regression
- feature-engineering
- interpretability
- experiment-tracking
- production
---

# AutoML Lite 🤖

**Automated Machine Learning Made Simple**

A lightweight, production-ready automated machine learning library that simplifies the entire ML pipeline from data preprocessing to model deployment.

## 🎬 Demo

### AutoML Lite in Action
![AutoML Lite Demo](https://github.com/Sherin-SEF-AI/AutoML-Lite/raw/main/automl-lite.gif)

### Generated HTML Reports
![AutoML Report Generation](https://github.com/Sherin-SEF-AI/AutoML-Lite/raw/main/automl-lite-report.gif)

### Weights & Biases Integration
![W&B Experiment Tracking](https://github.com/Sherin-SEF-AI/AutoML-Lite/raw/main/automl-wandb.gif)

## 🚀 Quick Start

### Installation
```bash
pip install automl-lite
```

### 5-Line ML Pipeline
```python
from automl_lite import AutoMLite
import pandas as pd

# Load your data
data = pd.read_csv('your_data.csv')

# Initialize AutoML (zero configuration!)
automl = AutoMLite(time_budget=300)

# Train and get the best model
best_model = automl.fit(data, target_column='target')

# Make predictions
predictions = automl.predict(new_data)
```

## ✨ Key Features

### 🧠 Intelligent Automation
- **Auto Feature Engineering**: 11.6x feature expansion (20→232 features)
- **Smart Model Selection**: Tests 15+ algorithms automatically
- **Hyperparameter Optimization**: Uses Optuna for efficient tuning
- **Ensemble Methods**: Automatic voting classifiers

### 🏭 Production-Ready
- **Deep Learning**: TensorFlow and PyTorch integration
- **Time Series**: ARIMA, Prophet, LSTM forecasting
- **Advanced Interpretability**: SHAP, LIME, permutation importance
- **Experiment Tracking**: MLflow, W&B, TensorBoard
- **Interactive Dashboards**: Real-time monitoring

### 📊 Comprehensive Reporting
- **Interactive HTML Reports**: Beautiful visualizations
- **Model Performance Analysis**: Confusion matrices, ROC curves
- **Feature Importance**: Detailed analysis and correlations
- **Training History**: Complete logs and metrics

## 🎯 Supported Problem Types

- ✅ **Classification** (Binary & Multi-class)
- ✅ **Regression**
- ✅ **Time Series Forecasting**
- ✅ **Deep Learning Tasks**

## 🔥 Performance Metrics

### Production Demo Results
- **Training Time**: 391.92 seconds for complete pipeline
- **Best Model**: Random Forest (80.00% accuracy)
- **Feature Engineering**: 20 → 232 features (11.6x expansion)
- **Feature Selection**: 132/166 features intelligently selected
- **Hyperparameter Optimization**: 50 trials with Optuna

## 🛠️ Advanced Usage

### Custom Configuration
```python
config = {
    'time_budget': 600,
    'max_models': 20,
    'cv_folds': 5,
    'feature_engineering': True,
    'ensemble_method': 'voting',
    'interpretability': True
}

automl = AutoMLite(**config)
```

### Time Series Forecasting
```python
automl = AutoMLite(problem_type='time_series')
model = automl.fit(data, target_column='sales', date_column='date')
forecast = automl.predict_future(periods=30)
```

### Deep Learning
```python
automl = AutoMLite(
    include_deep_learning=True,
    deep_learning_framework='tensorflow'
)
model = automl.fit(data, target_column='target')
```

## 📈 CLI Interface

```bash
# Basic usage
automl-lite train data.csv --target target_column

# With custom config
automl-lite train data.csv --target target_column --config config.yaml

# Generate report
automl-lite report --model model.pkl --output report.html
```

## 🎨 Interactive Dashboard

```python
from automl_lite.ui import launch_dashboard
launch_dashboard(automl)
```

## 🔍 Model Interpretability

```python
# Get SHAP values
shap_values = automl.explain_model(X_test)

# Feature importance
importance = automl.get_feature_importance()

# Partial dependence plots
automl.plot_partial_dependence('feature_name')
```

## 🎯 Use Cases

### Perfect For:
- 🏢 **Data Scientists** - Rapid prototyping
- 🚀 **ML Engineers** - Production development
- 📊 **Analysts** - Quick insights
- 🎓 **Students** - Learning ML concepts
- 🏭 **Startups** - Fast MVP development

### Industries:
- **Finance**: Credit scoring, fraud detection
- **Healthcare**: Disease prediction, monitoring
- **E-commerce**: Segmentation, forecasting
- **Marketing**: Campaign optimization
- **Manufacturing**: Predictive maintenance

## 🔧 Configuration Templates

- **Basic**: Quick experiments
- **Production**: Production deployment
- **Research**: Extensive search
- **Customer Churn**: Churn prediction
- **Fraud Detection**: Fraud detection
- **House Price**: Real estate prediction

## 📦 Installation Options

### From PyPI (Recommended)
```bash
pip install automl-lite
```

### From Source
```bash
git clone https://github.com/Sherin-SEF-AI/AutoML-Lite.git
cd AutoML-Lite
pip install -e .
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add tests**
5. **Submit a pull request**

## 📚 Documentation & Resources

- 📖 **Full Documentation**: [GitHub Wiki](https://github.com/Sherin-SEF-AI/AutoML-Lite/wiki)
- 🎯 **API Reference**: [API Docs](https://github.com/Sherin-SEF-AI/AutoML-Lite/blob/main/docs/API_REFERENCE.md)
- 📝 **Examples**: [Example Notebooks](https://github.com/Sherin-SEF-AI/AutoML-Lite/tree/main/examples)
- 🚀 **Quick Start**: [Installation Guide](https://github.com/Sherin-SEF-AI/AutoML-Lite/blob/main/docs/INSTALLATION.md)

## 💬 Join the Community

- 🌟 **Star the Repository**: [GitHub](https://github.com/Sherin-SEF-AI/AutoML-Lite)
- 🐛 **Report Issues**: [Issue Tracker](https://github.com/Sherin-SEF-AI/AutoML-Lite/issues)
- 💡 **Feature Requests**: [Discussions](https://github.com/Sherin-SEF-AI/AutoML-Lite/discussions)
- 📧 **Contact**: sherin.joseph2217@gmail.com

## 🏆 Why Choose AutoML Lite?

| Feature | AutoML Lite | Other Libraries |
|---------|-------------|-----------------|
| **Setup Time** | 30 seconds | 30+ minutes |
| **Configuration** | Zero required | Complex configs |
| **Production Ready** | ✅ Built-in | ❌ Manual setup |
| **Deep Learning** | ✅ Integrated | ❌ Separate setup |
| **Time Series** | ✅ Native support | ❌ Limited |
| **Interpretability** | ✅ Advanced | ❌ Basic |
| **Experiment Tracking** | ✅ Multi-platform | ❌ Limited |
| **Interactive Reports** | ✅ Beautiful HTML | ❌ Basic plots |

## 🎯 Ready to Transform Your ML Workflow?

**Stop spending hours on boilerplate code. Start building amazing ML models in minutes!**

```bash
pip install automl-lite
```

**Try it now and see the difference!** 🚀

---

*Built with ❤️ by the AutoML Lite community*

**Tags**: #python #machinelearning #automl #datascience #ml #ai #automation #productivity #opensource #deeplearning #timeseries #interpretability #experimenttracking #production #deployment 
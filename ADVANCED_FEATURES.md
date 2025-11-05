# Advanced Features in AutoML-Lite

This document describes the advanced features added to AutoML-Lite, making it a comprehensive, enterprise-grade AutoML framework.

## Table of Contents

1. [Advanced Ensemble Methods](#1-advanced-ensemble-methods)
2. [Model Monitoring & Drift Detection](#2-model-monitoring--drift-detection)
3. [Advanced Feature Engineering](#3-advanced-feature-engineering)
4. [Anomaly Detection Framework](#4-anomaly-detection-framework)
5. [Neural Architecture Search](#5-neural-architecture-search)
6. [Model Serving API](#6-model-serving-api)
7. [Causal Inference](#7-causal-inference)
8. [Fairness & Bias Detection](#8-fairness--bias-detection)

---

## 1. Advanced Ensemble Methods

### Features
- **Stacking Ensemble**: Uses K-fold cross-validation to generate out-of-fold predictions as meta-features
- **Blending Ensemble**: Simpler holdout-based ensemble for faster training
- **Weighted Ensemble**: Optimizes ensemble weights using grid search, gradient descent, or closed-form solutions

### Usage

```python
from automl_lite.ensemble import StackingEnsemble, BlendingEnsemble, WeightedEnsemble
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Train base models
base_models = [
    RandomForestClassifier(n_estimators=100),
    GradientBoostingClassifier(n_estimators=100),
    LogisticRegression(max_iter=1000)
]

for model in base_models:
    model.fit(X_train, y_train)

# Stacking Ensemble
stacking = StackingEnsemble(
    base_models=base_models,
    problem_type='classification',
    n_folds=5,
    use_probas=True
)
stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)

# Blending Ensemble
blending = BlendingEnsemble(
    base_models=base_models,
    problem_type='classification',
    blend_ratio=0.2
)
blending.fit(X_train, y_train)
predictions = blending.predict(X_test)

# Weighted Ensemble
weighted = WeightedEnsemble(
    models=base_models,
    problem_type='classification',
    optimization='gradient'
)
weighted.fit(X_val, y_val)  # Use validation set to optimize weights
predictions = weighted.predict(X_test)
```

### Key Benefits
- **Better Accuracy**: Combines multiple models for improved performance
- **Reduced Overfitting**: Out-of-fold predictions prevent information leakage
- **Flexible Meta-Learning**: Customizable meta-models and optimization strategies

---

## 2. Model Monitoring & Drift Detection

### Features
- **Data Drift Detection**: Multiple methods (KS test, Chi-squared, PSI, Wasserstein, KL divergence)
- **Prediction Drift Detection**: Monitor changes in model output distributions
- **Performance Degradation**: Track model performance over time
- **Comprehensive Reporting**: Detailed drift reports with recommendations

### Usage

```python
from automl_lite.monitoring import DriftDetector, ModelMonitor
import pandas as pd

# Setup drift detector
drift_detector = DriftDetector(
    reference_data=X_train_df,
    feature_names=feature_names,
    categorical_features=['cat_feature_1', 'cat_feature_2']
)

# Detect drift on new data
drift_report = drift_detector.detect_data_drift(
    current_data=X_new_df,
    methods=['ks', 'psi', 'wasserstein']
)

print(f"Drift Detected: {drift_report.data_drift_detected}")
print(f"Drifted Features: {drift_report.drift_scores['drifted_features_count']}")
print(f"Recommendations: {drift_report.recommendations}")

# Full model monitoring
monitor = ModelMonitor(
    model=trained_model,
    reference_data=X_train_df,
    reference_labels=y_train,
    feature_names=feature_names,
    problem_type='classification'
)

# Monitor new batch
monitoring_report = monitor.monitor(X_new_df, y_new)
```

### Key Benefits
- **Proactive Monitoring**: Detect issues before they impact production
- **Multiple Detection Methods**: Robust drift detection across different scenarios
- **Actionable Insights**: Clear recommendations for model maintenance
- **Performance Tracking**: Continuous monitoring of model quality

---

## 3. Advanced Feature Engineering

### Features
- **Target Encoding with CV**: Prevents overfitting using cross-validation
- **Frequency Encoding**: Encodes high-cardinality categoricals by frequency
- **Lag Features**: Create temporal features for time series
- **Rolling Window Statistics**: Moving averages, std, min, max
- **GroupBy Aggregations**: Summary statistics by groups
- **Fourier Features**: Capture periodic patterns
- **Wavelet Transformation**: Multi-scale feature extraction
- **Adaptive Binning**: Intelligent discretization strategies

### Usage

```python
from automl_lite.preprocessing.advanced_feature_engineering import (
    TargetEncoderCV,
    FrequencyEncoder,
    LagFeatureCreator,
    RollingWindowFeatures,
    FourierFeatures,
    AdaptiveBinner
)

# Target Encoding with Cross-Validation
target_encoder = TargetEncoderCV(
    categorical_features=['category_col1', 'category_col2'],
    n_folds=5,
    smoothing=1.0
)
X_encoded = target_encoder.fit_transform_cv(X_train, y_train)

# Frequency Encoding
freq_encoder = FrequencyEncoder(categorical_features=['high_cardinality_col'])
X_freq = freq_encoder.fit_transform(X_train)

# Lag Features (for time series)
lag_creator = LagFeatureCreator(
    target_column='sales',
    lags=[1, 2, 3, 7, 14, 30],
    time_column='date'
)
X_lagged = lag_creator.transform(X_train)

# Rolling Window Features
rolling = RollingWindowFeatures(
    feature_columns=['sales', 'temperature'],
    windows=[7, 14, 30],
    statistics=['mean', 'std', 'min', 'max']
)
X_rolling = rolling.transform(X_train)

# Adaptive Binning
binner = AdaptiveBinner(
    feature_columns=['numerical_col1', 'numerical_col2'],
    n_bins=10,
    strategy='quantile'
)
X_binned = binner.fit_transform(X_train)
```

### Key Benefits
- **Automatic Feature Generation**: Create powerful features automatically
- **Prevents Overfitting**: Cross-validation techniques for target encoding
- **Time Series Support**: Specialized features for temporal data
- **Scalable**: Efficient implementations for large datasets

---

## 4. Anomaly Detection Framework

### Features
- **Multiple Algorithms**: Isolation Forest, LOF, One-Class SVM, Elliptic Envelope, Autoencoder, Statistical methods
- **Ensemble Detection**: Combine multiple methods for robust detection
- **Customizable Thresholds**: Fine-tune sensitivity
- **Comprehensive Reports**: Detailed anomaly information

### Usage

```python
from automl_lite.anomaly import detect_anomalies, EnsembleAnomalyDetector

# Quick detection with single method
report = detect_anomalies(
    X,
    method='isolation_forest',
    contamination=0.1
)

print(f"Anomalies Detected: {report.n_anomalies}")
print(f"Anomaly Indices: {report.anomaly_indices}")

# Ensemble detection for robustness
ensemble_detector = EnsembleAnomalyDetector(
    contamination=0.1,
    voting='soft',  # or 'hard'
    scale_features=True
)

ensemble_detector.fit(X_train)
report = ensemble_detector.detect(X_test)

# Get anomaly scores
scores = ensemble_detector.predict_scores(X_test)
```

### Supported Methods
- **Isolation Forest**: Efficient for high-dimensional data
- **Local Outlier Factor (LOF)**: Density-based detection
- **One-Class SVM**: Kernel-based outlier detection
- **Elliptic Envelope**: Assumes Gaussian distribution
- **Autoencoder**: Deep learning-based detection
- **Statistical**: Z-score and IQR methods
- **Ensemble**: Combines multiple methods

### Key Benefits
- **Versatile**: Works across various data types and distributions
- **Ensemble Robustness**: Reduce false positives with voting
- **Production-Ready**: Fast inference for real-time detection
- **Interpretable**: Clear anomaly scores and reports

---

## 5. Neural Architecture Search (NAS)

### Features
- **Automatic Architecture Discovery**: Find optimal neural network structures
- **Multiple Search Methods**: Random search and evolutionary algorithms
- **Customizable Search Space**: Define layers, activations, optimizers
- **Fast Evaluation**: Early stopping and efficient cross-validation

### Usage

```python
from automl_lite.neural import NeuralArchitectureSearch, auto_neural_network

# Quick NAS with default settings
model, architecture = auto_neural_network(
    X_train, y_train,
    problem_type='classification',
    n_trials=50,
    search_method='random'
)

# Custom NAS with evolutionary search
from automl_lite.neural import NASSearchSpace

search_space = NASSearchSpace(
    n_layers_range=(2, 6),
    hidden_units_options=[64, 128, 256, 512],
    activation_options=['relu', 'elu', 'tanh'],
    dropout_options=[0.0, 0.1, 0.2, 0.3],
    optimizer_options=['adam', 'sgd'],
    learning_rate_options=[0.001, 0.01, 0.0001]
)

nas = NeuralArchitectureSearch(
    search_space=search_space,
    search_method='evolutionary',
    n_trials=100,
    epochs=50,
    problem_type='classification'
)

best_architecture = nas.search(X_train, y_train)
best_model = nas.get_best_model()

# Get search history
history = nas.get_search_history()
```

### Key Benefits
- **Automated Design**: No manual architecture engineering needed
- **Better Performance**: Discover architectures optimized for your data
- **Flexible**: Supports both classification and regression
- **Efficient Search**: Smart algorithms reduce search time

---

## 6. Model Serving API

### Features
- **FastAPI Integration**: Production-ready REST API
- **Automatic Endpoint Generation**: /predict, /info, /health
- **Batch Predictions**: Efficient batch processing
- **CORS Support**: Ready for web applications
- **Model Metadata**: Track versions, features, and performance

### Usage

```python
from automl_lite.serving import ModelServer

# Create server from trained model
server = ModelServer(
    model=trained_model,
    model_name="my_classifier",
    feature_names=['feature_1', 'feature_2', ..., 'feature_n'],
    problem_type='classification',
    version='1.0'
)

# Start serving (in production)
server.serve(host='0.0.0.0', port=8000)

# Access API at: http://localhost:8000/docs
```

**API Endpoints:**
- `GET /` - API information
- `POST /predict` - Make predictions
- `GET /info` - Model information
- `GET /health` - Health check

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": [[1.0, 2.0, 3.0, ...]]
     }'
```

**Example Response:**
```json
{
  "predictions": [1],
  "probabilities": [[0.2, 0.8]],
  "model_name": "my_classifier",
  "timestamp": "2024-01-01T12:00:00",
  "prediction_time_ms": 5.2
}
```

### Key Benefits
- **Production-Ready**: Battle-tested FastAPI framework
- **Easy Deployment**: Simple Docker containerization
- **Interactive Docs**: Automatic Swagger/OpenAPI documentation
- **Monitoring**: Built-in health checks and metrics

---

## 7. Causal Inference

### Features
- **Propensity Score Matching (PSM)**: Match treated/control units
- **Double Machine Learning (DML)**: Debiased treatment effect estimation
- **Causal Forest**: Heterogeneous treatment effects
- **Individual Treatment Effects (ITE)**: Personalized effect estimation

### Usage

```python
from automl_lite.causal import estimate_treatment_effect, DoubleMachineLearning

# Propensity Score Matching
results_psm = estimate_treatment_effect(
    X=covariates,
    treatment=treatment_indicator,
    outcome=outcome_variable,
    method='psm',
    caliper=0.1
)

print(f"Average Treatment Effect: {results_psm['ate']:.4f}")

# Double Machine Learning
results_dml = estimate_treatment_effect(
    X=covariates,
    treatment=treatment_variable,
    outcome=outcome_variable,
    method='dml',
    n_folds=5
)

print(f"Debiased ATE: {results_dml['ate']:.4f}")

# Causal Forest for heterogeneous effects
from automl_lite.causal import CausalForest

cf = CausalForest(n_estimators=100)
cf.fit(X_covariates, treatment, outcome)

# Predict individual treatment effects
individual_effects = cf.predict_ite(X_new)
```

### Key Benefits
- **Rigorous Causal Analysis**: Beyond correlation to causation
- **Multiple Methods**: Choose the right approach for your data
- **Heterogeneous Effects**: Understand who benefits most from treatment
- **Production-Ready**: Efficient implementations for large datasets

---

## 8. Fairness & Bias Detection

### Features
- **Multiple Fairness Metrics**: Demographic parity, equal opportunity, equalized odds, disparate impact
- **Calibration Analysis**: Check prediction calibration by group
- **Bias Mitigation**: Sample reweighting and threshold adjustment
- **Comprehensive Reports**: Detailed fairness analysis with recommendations

### Usage

```python
from automl_lite.fairness import BiasDetector, detect_and_mitigate_bias
import pandas as pd

# Prepare sensitive features
sensitive_df = pd.DataFrame({
    'gender': ['M', 'F', 'M', ...],
    'race': ['A', 'B', 'A', ...],
    'age_group': ['young', 'old', 'young', ...]
})

# Detect bias
detector = BiasDetector(
    sensitive_features=['gender', 'race', 'age_group']
)

reports = detector.detect_bias(
    y_true=y_true,
    y_pred=y_pred,
    sensitive_data=sensitive_df,
    y_pred_proba=y_pred_proba
)

# Analyze reports
for feature, report in reports.items():
    print(f"\nFairness Analysis for {feature}:")
    print(f"  Is Fair: {report.is_fair}")
    print(f"  Demographic Parity Diff: {report.demographic_parity_diff:.4f}")
    print(f"  Equal Opportunity Diff: {report.equal_opportunity_diff:.4f}")
    print(f"  Disparate Impact Ratio: {report.disparate_impact_ratio:.4f}")

    if report.violations:
        print(f"  Violations:")
        for violation in report.violations:
            print(f"    - {violation}")

    if report.recommendations:
        print(f"  Recommendations:")
        for rec in report.recommendations:
            print(f"    - {rec}")

# Bias mitigation
from automl_lite.fairness import FairnessIntervention

# Reweight samples
weights = FairnessIntervention.reweight_samples(
    sensitive_df['gender'].values,
    y_true
)

# Train with reweighted samples
model.fit(X_train, y_train, sample_weight=weights)

# Find fair thresholds
thresholds = FairnessIntervention.find_fair_threshold(
    y_true, y_pred_proba,
    sensitive_df['gender'].values,
    metric='equal_opportunity'
)
```

### Fairness Metrics

- **Demographic Parity**: Equal positive prediction rates across groups
- **Equal Opportunity**: Equal true positive rates across groups
- **Equalized Odds**: Equal TPR and FPR across groups
- **Disparate Impact**: Ratio of positive rates (80% rule)
- **Calibration by Group**: Prediction probabilities match outcomes within groups

### Key Benefits
- **Ethical AI**: Build fair and responsible models
- **Regulatory Compliance**: Meet fairness requirements
- **Multiple Perspectives**: Various fairness definitions supported
- **Actionable**: Concrete mitigation strategies provided

---

## Installation

Install additional dependencies for advanced features:

```bash
# Core advanced features
pip install fastapi uvicorn pydantic PyWavelets

# Optional: Deep learning for NAS and Autoencoder
pip install tensorflow>=2.8.0
# OR
pip install torch>=1.12.0

# Optional: Time series for advanced feature engineering
pip install statsmodels>=0.13.0
```

---

## Examples

Run the comprehensive demo:

```bash
python examples/advanced_features_demo.py
```

This demonstrates all 8 advanced feature categories with real examples.

---

## Performance Considerations

### Computational Requirements
- **Ensemble Methods**: 2-3x training time of single models
- **NAS**: Can be expensive (use n_trials carefully)
- **Drift Detection**: O(n) for most methods
- **Anomaly Detection**: Ensemble is slower but more robust
- **Causal Inference**: DML requires 2x model training

### Scalability Tips
1. **Parallel Processing**: Use joblib for parallel model training
2. **Sampling**: Use subset for NAS and drift detection on large datasets
3. **Caching**: Cache feature engineering results
4. **GPU**: Enable GPU for NAS with TensorFlow/PyTorch
5. **Incremental Learning**: Update drift detectors incrementally

---

## Roadmap

Future enhancements planned:
- [ ] Distributed training with Ray
- [ ] AutoML for computer vision
- [ ] Natural language processing support
- [ ] Reinforcement learning integration
- [ ] Enhanced model compression
- [ ] Cloud deployment integrations (AWS, GCP, Azure)

---

## Contributing

Contributions are welcome! See CONTRIBUTING.md for guidelines.

---

## License

See LICENSE file for details.

---

## Citation

If you use these advanced features in your research, please cite:

```bibtex
@software{automl_lite_advanced,
  title={AutoML-Lite: Advanced Features for Production ML},
  author={AutoML-Lite Contributors},
  year={2024},
  url={https://github.com/yourusername/AutoML-Lite}
}
```

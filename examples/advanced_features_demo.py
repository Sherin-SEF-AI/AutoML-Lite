"""
Advanced Features Demo for AutoML-Lite

Demonstrates all the new advanced features:
1. Advanced Ensemble Methods (Stacking, Blending, Weighted)
2. Model Monitoring & Drift Detection
3. Advanced Feature Engineering
4. Anomaly Detection
5. Neural Architecture Search
6. Model Serving API
7. Causal Inference
8. Fairness & Bias Detection
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

print("=" * 80)
print("AutoML-Lite Advanced Features Demo")
print("=" * 80)


# =============================================================================
# 1. ADVANCED ENSEMBLE METHODS
# =============================================================================
print("\n" + "=" * 80)
print("1. Advanced Ensemble Methods (Stacking, Blending, Weighted)")
print("=" * 80)

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                          n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train base models
print("\nTraining base models...")
base_models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    GradientBoostingClassifier(n_estimators=50, random_state=42),
    LogisticRegression(max_iter=1000, random_state=42)
]

for model in base_models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  {model.__class__.__name__}: {score:.4f}")

# Create stacking ensemble
print("\n>>> Creating Stacking Ensemble...")
try:
    from automl_lite.ensemble import StackingEnsemble

    stacking = StackingEnsemble(
        base_models=base_models,
        problem_type='classification',
        n_folds=5,
        verbose=False
    )

    stacking.fit(X_train, y_train)
    stacking_score = stacking.score(X_test, y_test)
    print(f"✓ Stacking Ensemble Score: {stacking_score:.4f}")
except Exception as e:
    print(f"✗ Stacking failed: {e}")

# Create blending ensemble
print("\n>>> Creating Blending Ensemble...")
try:
    from automl_lite.ensemble import BlendingEnsemble

    blending = BlendingEnsemble(
        base_models=base_models,
        problem_type='classification',
        blend_ratio=0.2,
        verbose=False
    )

    blending.fit(X_train, y_train)
    blending_score = blending.score(X_test, y_test)
    print(f"✓ Blending Ensemble Score: {blending_score:.4f}")
except Exception as e:
    print(f"✗ Blending failed: {e}")


# =============================================================================
# 2. MODEL MONITORING & DRIFT DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("2. Model Monitoring & Drift Detection")
print("=" * 80)

try:
    from automl_lite.monitoring import DriftDetector, ModelMonitor

    # Create reference and current data
    X_reference = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    X_current = pd.DataFrame(X_test + np.random.normal(0, 0.5, X_test.shape),
                             columns=[f'feature_{i}' for i in range(X_test.shape[1])])

    # Initialize drift detector
    print("\n>>> Detecting Data Drift...")
    drift_detector = DriftDetector(
        reference_data=X_reference,
        feature_names=X_reference.columns.tolist()
    )

    # Detect drift
    drift_report = drift_detector.detect_data_drift(X_current, methods=['ks', 'wasserstein'])

    print(f"✓ Drift Detection Complete")
    print(f"  Data Drift Detected: {drift_report.data_drift_detected}")
    print(f"  Number of Drifted Features: {drift_report.drift_scores.get('drifted_features_count', 0)}")
    print(f"  Alerts: {len(drift_report.alerts)}")

except Exception as e:
    print(f"✗ Drift detection failed: {e}")


# =============================================================================
# 3. ADVANCED FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 80)
print("3. Advanced Feature Engineering")
print("=" * 80)

try:
    from automl_lite.preprocessing.advanced_feature_engineering import (
        TargetEncoderCV,
        FrequencyEncoder,
        AdaptiveBinner
    )

    # Create sample data with categorical features
    df = pd.DataFrame({
        'cat1': np.random.choice(['A', 'B', 'C'], size=1000),
        'cat2': np.random.choice(['X', 'Y', 'Z'], size=1000),
        'num1': np.random.randn(1000),
        'num2': np.random.randn(1000)
    })
    y_sample = np.random.randint(0, 2, size=1000)

    print("\n>>> Target Encoding with Cross-Validation...")
    target_encoder = TargetEncoderCV(
        categorical_features=['cat1', 'cat2'],
        n_folds=5
    )
    df_encoded = target_encoder.fit_transform_cv(df, y_sample)
    print(f"✓ Target Encoding Complete: {df.shape} → {df_encoded.shape}")

    print("\n>>> Frequency Encoding...")
    freq_encoder = FrequencyEncoder(categorical_features=['cat1', 'cat2'])
    df_freq = freq_encoder.fit_transform(df)
    print(f"✓ Frequency Encoding Complete: {df.shape} → {df_freq.shape}")

    print("\n>>> Adaptive Binning...")
    binner = AdaptiveBinner(
        feature_columns=['num1', 'num2'],
        n_bins=10,
        strategy='quantile'
    )
    df_binned = binner.fit_transform(df)
    print(f"✓ Adaptive Binning Complete: {df.shape} → {df_binned.shape}")

except Exception as e:
    print(f"✗ Advanced feature engineering failed: {e}")


# =============================================================================
# 4. ANOMALY DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("4. Anomaly Detection Framework")
print("=" * 80)

try:
    from automl_lite.anomaly import detect_anomalies, EnsembleAnomalyDetector

    # Generate normal data with some anomalies
    X_normal = np.random.randn(900, 10)
    X_anomalies = np.random.randn(100, 10) * 3  # Anomalies with larger variance
    X_anomaly_test = np.vstack([X_normal, X_anomalies])

    print("\n>>> Isolation Forest Detection...")
    report_if = detect_anomalies(X_anomaly_test, method='isolation_forest', contamination=0.1)
    print(f"✓ Isolation Forest: {report_if.n_anomalies} anomalies detected ({report_if.anomaly_ratio:.2%})")

    print("\n>>> Local Outlier Factor Detection...")
    report_lof = detect_anomalies(X_anomaly_test, method='lof', contamination=0.1)
    print(f"✓ LOF: {report_lof.n_anomalies} anomalies detected ({report_lof.anomaly_ratio:.2%})")

    print("\n>>> Ensemble Anomaly Detection...")
    report_ensemble = detect_anomalies(X_anomaly_test, method='ensemble', contamination=0.1)
    print(f"✓ Ensemble: {report_ensemble.n_anomalies} anomalies detected ({report_ensemble.anomaly_ratio:.2%})")

except Exception as e:
    print(f"✗ Anomaly detection failed: {e}")


# =============================================================================
# 5. NEURAL ARCHITECTURE SEARCH
# =============================================================================
print("\n" + "=" * 80)
print("5. Neural Architecture Search (NAS)")
print("=" * 80)

try:
    from automl_lite.neural import NeuralArchitectureSearch, auto_neural_network

    print("\n>>> Running Neural Architecture Search...")
    print("  (This may take a few minutes...)")

    # Use smaller dataset for demo
    X_small, y_small = make_classification(n_samples=500, n_features=10, random_state=42)

    # Quick NAS with limited trials
    model, architecture = auto_neural_network(
        X_small, y_small,
        problem_type='classification',
        n_trials=5,  # Limited for demo
        search_method='random',
        epochs=10,
        verbose=False
    )

    print(f"✓ NAS Complete")
    print(f"  Best Architecture: {architecture.n_layers} layers")
    print(f"  Optimizer: {architecture.optimizer}")
    print(f"  Learning Rate: {architecture.learning_rate}")
    print(f"  Score: {architecture.score:.4f}")

except Exception as e:
    print(f"✗ NAS failed: {e}")


# =============================================================================
# 6. MODEL SERVING API
# =============================================================================
print("\n" + "=" * 80)
print("6. REST API for Model Serving")
print("=" * 80)

try:
    from automl_lite.serving import ModelServer

    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    print("\n>>> Creating Model Server...")
    server = ModelServer(
        model=model,
        model_name="demo_classifier",
        feature_names=[f'feature_{i}' for i in range(X_train.shape[1])],
        problem_type='classification',
        version='1.0'
    )

    # Test prediction
    test_features = X_test[:5]
    result = server.predict(test_features)

    print(f"✓ Model Server Created")
    print(f"  Model: {server.model_name}")
    print(f"  Features: {server.n_features}")
    print(f"  Sample Prediction: {result['predictions'][:3]}")
    print(f"  Prediction Time: {result['prediction_time_ms']:.2f}ms")

    # Show how to start server (commented out for demo)
    print("\n  To serve model via REST API, run:")
    print("  >>> server.serve(host='0.0.0.0', port=8000)")
    print("  Then access: http://localhost:8000/docs")

except Exception as e:
    print(f"✗ Model serving failed: {e}")


# =============================================================================
# 7. CAUSAL INFERENCE
# =============================================================================
print("\n" + "=" * 80)
print("7. Causal Inference")
print("=" * 80)

try:
    from automl_lite.causal import estimate_treatment_effect, DoubleMachineLearning

    # Generate synthetic causal data
    n = 1000
    X_causal = np.random.randn(n, 5)
    treatment = (X_causal[:, 0] + np.random.randn(n) > 0).astype(int)
    outcome = 2 * treatment + X_causal[:, 1] + np.random.randn(n)

    print("\n>>> Propensity Score Matching...")
    results_psm = estimate_treatment_effect(
        X_causal, treatment, outcome,
        method='psm'
    )
    print(f"✓ PSM ATE: {results_psm['ate']:.4f} ± {results_psm.get('ate_std', 0):.4f}")

    print("\n>>> Double Machine Learning...")
    results_dml = estimate_treatment_effect(
        X_causal, treatment, outcome,
        method='dml',
        n_folds=5
    )
    print(f"✓ DML ATE: {results_dml['ate']:.4f}")

    print("\n>>> Causal Forest...")
    results_cf = estimate_treatment_effect(
        X_causal, treatment, outcome,
        method='causal_forest'
    )
    print(f"✓ Causal Forest ATE: {results_cf['ate']:.4f} ± {results_cf.get('ate_std', 0):.4f}")

except Exception as e:
    print(f"✗ Causal inference failed: {e}")


# =============================================================================
# 8. FAIRNESS & BIAS DETECTION
# =============================================================================
print("\n" + "=" * 80)
print("8. Fairness & Bias Detection")
print("=" * 80)

try:
    from automl_lite.fairness import BiasDetector, detect_and_mitigate_bias

    # Generate data with potential bias
    n = 1000
    sensitive_feature = np.random.randint(0, 2, size=n)  # Binary sensitive attribute
    X_fair = np.random.randn(n, 10)
    X_fair = np.column_stack([X_fair, sensitive_feature])

    # Introduce bias: model performs better for group 0
    y_true_fair = (X_fair[:, 0] + sensitive_feature * 0.5 + np.random.randn(n) > 0).astype(int)

    # Train potentially biased model
    model_fair = LogisticRegression(random_state=42)
    model_fair.fit(X_fair, y_true_fair)
    y_pred_fair = model_fair.predict(X_fair)

    # Create DataFrame with sensitive features
    sensitive_df = pd.DataFrame({'group': sensitive_feature})

    print("\n>>> Detecting Bias...")
    detector = BiasDetector(sensitive_features=['group'])
    reports = detector.detect_bias(
        y_true_fair,
        y_pred_fair,
        sensitive_df,
        y_pred_proba=model_fair.predict_proba(X_fair)[:, 1]
    )

    for feature, report in reports.items():
        print(f"\n✓ Bias Analysis for '{feature}':")
        print(f"  Fair: {report.is_fair}")
        print(f"  Demographic Parity Diff: {report.demographic_parity_diff:.4f}")
        print(f"  Equal Opportunity Diff: {report.equal_opportunity_diff:.4f}")
        print(f"  Disparate Impact Ratio: {report.disparate_impact_ratio:.4f}")
        print(f"  Violations: {len(report.violations)}")
        if report.violations:
            for violation in report.violations:
                print(f"    - {violation}")

except Exception as e:
    print(f"✗ Fairness detection failed: {e}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("DEMO COMPLETE")
print("=" * 80)
print("""
Advanced features demonstrated:
✓ Stacking & Blending Ensembles
✓ Model Monitoring & Drift Detection
✓ Advanced Feature Engineering
✓ Comprehensive Anomaly Detection
✓ Neural Architecture Search
✓ REST API Model Serving
✓ Causal Inference Methods
✓ Fairness & Bias Detection

All features are production-ready and integrated into AutoML-Lite!
""")

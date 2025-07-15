#!/usr/bin/env python3
"""
Comprehensive example demonstrating all AutoML Lite features.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import os

from automl_lite import AutoMLite


def classification_example():
    """Run a comprehensive classification example."""
    print("🤖 AutoML Lite - Comprehensive Classification Example")
    print("=" * 60)
    
    # Generate sample classification data
    print("📊 Generating sample classification data...")
    X, y = make_classification(
        n_samples=2000,
        n_features=15,
        n_informative=8,
        n_redundant=4,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution:\n{y_series.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize AutoML Lite with all features enabled
    print("\n🚀 Initializing AutoML Lite with all features...")
    automl = AutoMLite(
        time_budget=300,  # 5 minutes
        max_models=5,     # Try 5 models
        cv_folds=5,       # 5-fold CV
        random_state=42,
        verbose=True,
        enable_ensemble=True,
        enable_early_stopping=True,
        enable_feature_selection=True,
        enable_interpretability=True,
        ensemble_method="voting",
        top_k_models=3,
        early_stopping_patience=10
    )
    
    # Train the model
    print("\n🎯 Training AutoML model...")
    automl.fit(X_train, y_train)
    
    # Results
    print(f"\n✅ Training completed!")
    print(f"Best model: {automl.best_model_name}")
    print(f"Best CV score: {automl.best_score:.4f}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = automl.predict(X_test)
    test_score = automl.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    
    # Show leaderboard
    print("\n🏆 Model Leaderboard:")
    leaderboard = automl.get_leaderboard()
    print(leaderboard)
    
    # Show feature importance
    print("\n🎯 Feature Importance (Top 10):")
    feature_importance = automl.get_feature_importance()
    print(feature_importance.head(10))
    
    # Show ensemble info
    print("\n🎯 Ensemble Information:")
    ensemble_info = automl.get_ensemble_info()
    print(ensemble_info)
    
    # Show interpretability results
    print("\n🔍 Interpretability Results:")
    interpretability_results = automl.get_interpretability_report()
    print(interpretability_results)
    
    # Save model
    print("\n💾 Saving model...")
    automl.save_model("comprehensive_classification_model.pkl")
    
    # Generate comprehensive report with test data
    print("\n📋 Generating comprehensive report...")
    automl.generate_report("comprehensive_classification_report.html", X_test, y_test)
    
    print("\n🎉 Comprehensive classification example completed!")
    print("📁 Files created:")
    print("  - comprehensive_classification_model.pkl (saved model)")
    print("  - comprehensive_classification_report.html (comprehensive report)")
    
    return automl, X_test, y_test


def regression_example():
    """Run a comprehensive regression example."""
    print("\n🤖 AutoML Lite - Comprehensive Regression Example")
    print("=" * 60)
    
    # Generate sample regression data
    print("📊 Generating sample regression data...")
    X, y = make_regression(
        n_samples=1500,
        n_features=12,
        n_informative=6,
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target statistics:")
    print(f"  Mean: {y_series.mean():.2f}")
    print(f"  Std: {y_series.std():.2f}")
    print(f"  Min: {y_series.min():.2f}")
    print(f"  Max: {y_series.max():.2f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize AutoML Lite with all features enabled
    print("\n🚀 Initializing AutoML Lite with all features...")
    automl = AutoMLite(
        time_budget=300,  # 5 minutes
        max_models=5,     # Try 5 models
        cv_folds=5,       # 5-fold CV
        random_state=42,
        verbose=True,
        enable_ensemble=True,
        enable_early_stopping=True,
        enable_feature_selection=True,
        enable_interpretability=True,
        ensemble_method="voting",
        top_k_models=3,
        early_stopping_patience=10
    )
    
    # Train the model
    print("\n🎯 Training AutoML model...")
    automl.fit(X_train, y_train)
    
    # Results
    print(f"\n✅ Training completed!")
    print(f"Best model: {automl.best_model_name}")
    print(f"Best CV score: {automl.best_score:.4f}")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    y_pred = automl.predict(X_test)
    test_score = automl.score(X_test, y_test)
    print(f"Test R² score: {test_score:.4f}")
    
    # Show leaderboard
    print("\n🏆 Model Leaderboard:")
    leaderboard = automl.get_leaderboard()
    print(leaderboard)
    
    # Show feature importance
    print("\n🎯 Feature Importance (Top 10):")
    feature_importance = automl.get_feature_importance()
    print(feature_importance.head(10))
    
    # Show ensemble info
    print("\n🎯 Ensemble Information:")
    ensemble_info = automl.get_ensemble_info()
    print(ensemble_info)
    
    # Show interpretability results
    print("\n🔍 Interpretability Results:")
    interpretability_results = automl.get_interpretability_report()
    print(interpretability_results)
    
    # Save model
    print("\n💾 Saving model...")
    automl.save_model("comprehensive_regression_model.pkl")
    
    # Generate comprehensive report with test data
    print("\n📋 Generating comprehensive report...")
    automl.generate_report("comprehensive_regression_report.html", X_test, y_test)
    
    print("\n🎉 Comprehensive regression example completed!")
    print("📁 Files created:")
    print("  - comprehensive_regression_model.pkl (saved model)")
    print("  - comprehensive_regression_report.html (comprehensive report)")
    
    return automl, X_test, y_test


def main():
    """Run comprehensive examples."""
    print("🚀 AutoML Lite - Production Ready Package Demo")
    print("=" * 80)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run classification example
    try:
        classification_model, X_test_clf, y_test_clf = classification_example()
        print("\n✅ Classification example completed successfully!")
    except Exception as e:
        print(f"\n❌ Classification example failed: {str(e)}")
    
    # Run regression example
    try:
        regression_model, X_test_reg, y_test_reg = regression_example()
        print("\n✅ Regression example completed successfully!")
    except Exception as e:
        print(f"\n❌ Regression example failed: {str(e)}")
    
    print("\n🎉 All examples completed!")
    print("\n📋 Summary of AutoML Lite Features:")
    print("  ✅ Automated model selection and hyperparameter optimization")
    print("  ✅ Ensemble learning with voting classifiers/regressors")
    print("  ✅ Feature selection and importance analysis")
    print("  ✅ Early stopping for efficient training")
    print("  ✅ Comprehensive HTML reports with visualizations")
    print("  ✅ Model interpretability analysis")
    print("  ✅ Test set performance analysis")
    print("  ✅ Confusion matrices, ROC curves, and residuals plots")
    print("  ✅ Feature correlation analysis")
    print("  ✅ Learning curves and training history")
    print("  ✅ Production-ready model saving and loading")
    
    return 0


if __name__ == "__main__":
    exit(main()) 
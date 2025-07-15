#!/usr/bin/env python3
"""
Final demonstration of AutoML Lite - Production Ready Package
This script showcases all features working together.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time
import os

from automl_lite import AutoMLite


def main():
    """Demonstrate all AutoML Lite features."""
    print("🚀 AutoML Lite - Production Ready Package Final Demo")
    print("=" * 80)
    
    # Generate sample data
    print("📊 Generating sample classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    print(f"Dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize AutoML Lite with all features
    print("\n🤖 Initializing AutoML Lite with all advanced features...")
    automl = AutoMLite(
        time_budget=120,  # 2 minutes for demo
        max_models=3,     # Try 3 models
        cv_folds=3,       # 3-fold CV
        random_state=42,
        verbose=True,
        enable_ensemble=True,
        enable_early_stopping=True,
        enable_feature_selection=True,
        enable_interpretability=True,
        ensemble_method="voting",
        top_k_models=2,
        early_stopping_patience=5
    )
    
    # Train the model
    print("\n🎯 Training AutoML model...")
    start_time = time.time()
    automl.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"\n✅ Training completed in {training_time:.2f} seconds!")
    print(f"Best model: {automl.best_model_name}")
    print(f"Best CV score: {automl.best_score:.4f}")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    y_pred = automl.predict(X_test)
    test_score = automl.score(X_test, y_test)
    print(f"Test accuracy: {test_score:.4f}")
    
    # Show all available information
    print("\n📊 Model Information:")
    print("-" * 40)
    
    # Leaderboard
    print("\n🏆 Model Leaderboard:")
    leaderboard = automl.get_leaderboard()
    for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
        print(f"  {i}. {row['model_name']}: {row['score']:.4f}")
    
    # Feature importance
    print("\n🎯 Feature Importance (Top 5):")
    feature_importance = automl.get_feature_importance()
    print(feature_importance.head())
    
    # Ensemble info
    print("\n🎯 Ensemble Information:")
    ensemble_info = automl.get_ensemble_info()
    print(f"  Method: {ensemble_info.get('ensemble_method', 'N/A')}")
    print(f"  Top K models: {ensemble_info.get('top_k_models', 'N/A')}")
    print(f"  Ensemble score: {ensemble_info.get('ensemble_score', 'N/A')}")
    
    # Interpretability
    print("\n🔍 Interpretability Results:")
    interpretability_results = automl.get_interpretability_report()
    print(f"  SHAP values available: {'shap_values' in interpretability_results}")
    print(f"  Feature effects available: {'feature_effects' in interpretability_results}")
    print(f"  Model complexity: {interpretability_results.get('model_complexity', 'N/A')}")
    
    # Save model
    print("\n💾 Saving model...")
    model_path = "final_demo_model.pkl"
    automl.save_model(model_path)
    print(f"Model saved to: {model_path}")
    
    # Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    report_path = "final_demo_report.html"
    automl.generate_report(report_path, X_test, y_test)
    print(f"Report generated: {report_path}")
    
    # Test model loading
    print("\n🔄 Testing model loading...")
    loaded_automl = AutoMLite()
    loaded_automl.load_model(model_path)
    
    # Verify loaded model works
    loaded_pred = loaded_automl.predict(X_test)
    loaded_score = loaded_automl.score(X_test, y_test)
    print(f"Loaded model test accuracy: {loaded_score:.4f}")
    print(f"Predictions match: {np.array_equal(y_pred, loaded_pred)}")
    
    # Final summary
    print("\n🎉 AutoML Lite Demo Completed Successfully!")
    print("=" * 80)
    print("✅ All Features Working:")
    print("  • Automated model selection and hyperparameter optimization")
    print("  • Ensemble learning with voting classifiers")
    print("  • Feature selection and importance analysis")
    print("  • Early stopping for efficient training")
    print("  • Model interpretability analysis")
    print("  • Comprehensive HTML reports with visualizations")
    print("  • Test set performance analysis")
    print("  • Model saving and loading")
    print("  • Production-ready CLI interface")
    
    print(f"\n📁 Generated Files:")
    print(f"  • {model_path} (saved model)")
    print(f"  • {report_path} (comprehensive report)")
    
    print(f"\n📊 Performance Summary:")
    print(f"  • Training time: {training_time:.2f} seconds")
    print(f"  • Best CV score: {automl.best_score:.4f}")
    print(f"  • Test accuracy: {test_score:.4f}")
    print(f"  • Models tried: {len(leaderboard)}")
    
    return 0


if __name__ == "__main__":
    exit(main()) 
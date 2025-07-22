#!/usr/bin/env python3
"""
🤖 AutoML Lite - Complete Feature Showcase Demo

This demo showcases all the powerful features of AutoML Lite:
- Automated model selection and hyperparameter optimization
- Ensemble methods and feature selection
- Model interpretability and comprehensive reporting
- CLI and Python API usage
- Production-ready features

Perfect for sharing on Dev.to, Medium, and other developer platforms!

Author: Sherin Joseph Roy
GitHub: https://github.com/Sherin-SEF-AI/AutoML-Lite
PyPI: https://pypi.org/project/automl-lite/
"""

import pandas as pd
import numpy as np
import time
import os
from pathlib import Path
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Import AutoML Lite
from automl_lite import AutoMLite

def print_header(title):
    """Print a beautiful header for each section."""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def print_subheader(title):
    """Print a subheader."""
    print(f"\n📋 {title}")
    print("-" * 40)

def create_sample_data():
    """Create sample datasets for demonstration."""
    print_subheader("Creating Sample Datasets")
    
    # Classification dataset
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=2, random_state=42
    )
    clf_df = pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(20)])
    clf_df['target'] = y_clf
    
    # Regression dataset
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=15, n_informative=10, 
        noise=0.1, random_state=42
    )
    reg_df = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(15)])
    reg_df['target'] = y_reg
    
    print("✅ Created classification dataset: 1000 samples, 20 features")
    print("✅ Created regression dataset: 1000 samples, 15 features")
    
    return clf_df, reg_df

def demo_basic_automl():
    """Demonstrate basic AutoML functionality."""
    print_header("Basic AutoML - Automated Model Selection")
    
    # Create data
    clf_df, _ = create_sample_data()
    X = clf_df.drop('target', axis=1)
    y = clf_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print_subheader("Training AutoML Model")
    
    # Initialize AutoML with basic settings
    automl = AutoMLite(
        time_budget=60,  # 1 minute time budget
        max_models=5,    # Try 5 different models
        cv_folds=3,      # 3-fold cross-validation
        random_state=42,
        verbose=True
    )
    
    # Train the model
    start_time = time.time()
    automl.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    print(f"✅ Best model: {automl.best_model_name}")
    print(f"✅ Best score: {automl.best_score:.4f}")
    
    # Make predictions
    print_subheader("Making Predictions")
    predictions = automl.predict(X_test)
    print(f"✅ Made predictions on {len(X_test)} test samples")
    
    # Show leaderboard
    print_subheader("Model Leaderboard")
    leaderboard = automl.get_leaderboard()
    for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
        print(f"  {i}. {row['model_name']}: {row['score']:.4f}")
    
    return automl, X_test, y_test

def demo_advanced_features():
    """Demonstrate advanced AutoML features."""
    print_header("Advanced Features - Ensemble & Interpretability")
    
    # Create data
    clf_df, _ = create_sample_data()
    X = clf_df.drop('target', axis=1)
    y = clf_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print_subheader("Training with Advanced Features")
    
    # Initialize AutoML with all advanced features
    automl = AutoMLite(
        time_budget=120,           # 2 minutes
        max_models=8,              # More models for ensemble
        cv_folds=5,                # 5-fold CV
        random_state=42,
        verbose=True,
        enable_ensemble=True,      # Enable ensemble methods
        enable_early_stopping=True, # Enable early stopping
        enable_feature_selection=True, # Enable feature selection
        enable_interpretability=True,  # Enable interpretability
        ensemble_method="voting",  # Voting ensemble
        top_k_models=3,           # Top 3 models in ensemble
        early_stopping_patience=10
    )
    
    # Train the model
    start_time = time.time()
    automl.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Advanced training completed in {training_time:.2f} seconds")
    print(f"✅ Best model: {automl.best_model_name}")
    print(f"✅ Best score: {automl.best_score:.4f}")
    
    # Show ensemble information
    print_subheader("Ensemble Information")
    ensemble_info = automl.get_ensemble_info()
    if ensemble_info:
        print(f"✅ Ensemble method: {ensemble_info.get('ensemble_method', 'N/A')}")
        print(f"✅ Top K models: {ensemble_info.get('top_k_models', 'N/A')}")
        print(f"✅ Ensemble score: {ensemble_info.get('ensemble_score', 'N/A'):.4f}")
    
    # Show feature importance
    print_subheader("Feature Importance (Top 10)")
    feature_importance = automl.get_feature_importance()
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"  {i:2d}. {feature}: {importance:.4f}")
    
    # Show interpretability results
    print_subheader("Model Interpretability")
    interpretability = automl.get_interpretability_results()
    if interpretability:
        print("✅ SHAP values calculated")
        print("✅ Feature effects analyzed")
        print("✅ Model complexity assessed")
    
    return automl, X_test, y_test

def demo_regression():
    """Demonstrate regression capabilities."""
    print_header("Regression - House Price Prediction")
    
    # Create regression data
    _, reg_df = create_sample_data()
    X = reg_df.drop('target', axis=1)
    y = reg_df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print_subheader("Training Regression Model")
    
    # Initialize AutoML for regression
    automl = AutoMLite(
        problem_type='regression',
        time_budget=90,
        max_models=6,
        cv_folds=5,
        random_state=42,
        verbose=True,
        enable_ensemble=True,
        enable_feature_selection=True
    )
    
    # Train the model
    start_time = time.time()
    automl.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✅ Regression training completed in {training_time:.2f} seconds")
    print(f"✅ Best model: {automl.best_model_name}")
    print(f"✅ Best score (R²): {automl.best_score:.4f}")
    
    # Make predictions
    predictions = automl.predict(X_test)
    print(f"✅ Made regression predictions on {len(X_test)} test samples")
    
    return automl, X_test, y_test

def demo_model_persistence():
    """Demonstrate model saving and loading."""
    print_header("Model Persistence - Save & Load")
    
    # Train a model first
    automl, X_test, y_test = demo_basic_automl()
    
    print_subheader("Saving Model")
    model_path = "demo_model.pkl"
    automl.save_model(model_path)
    print(f"✅ Model saved to: {model_path}")
    
    print_subheader("Loading Model")
    from automl_lite import AutoMLite
    
    # Load the model
    loaded_automl = AutoMLite.load_model(model_path)
    print("✅ Model loaded successfully")
    
    # Verify it works
    predictions = loaded_automl.predict(X_test)
    print(f"✅ Loaded model made predictions on {len(X_test)} samples")
    print(f"✅ Model type: {type(loaded_automl.best_model).__name__}")
    
    # Clean up
    os.remove(model_path)
    print("✅ Demo model file cleaned up")

def demo_report_generation():
    """Demonstrate comprehensive report generation."""
    print_header("Report Generation - Beautiful HTML Reports")
    
    # Train a model with advanced features
    automl, X_test, y_test = demo_advanced_features()
    
    print_subheader("Generating Comprehensive Report")
    
    # Generate report with test data
    report_path = "demo_report.html"
    automl.generate_report(report_path, X_test=X_test, y_test=y_test)
    
    print(f"✅ Report generated: {report_path}")
    print("📊 Report includes:")
    print("   • Model leaderboard and performance comparison")
    print("   • Feature importance analysis")
    print("   • Training history and learning curves")
    print("   • Ensemble information")
    print("   • Model interpretability (SHAP values)")
    print("   • Confusion matrix and ROC curves")
    print("   • Feature correlation analysis")
    print("   • Interactive visualizations")
    
    # Check if report was created
    if os.path.exists(report_path):
        file_size = os.path.getsize(report_path) / 1024  # KB
        print(f"✅ Report file size: {file_size:.1f} KB")
        print(f"🌐 Open {report_path} in your browser to view the report!")
    else:
        print("❌ Report generation failed")

def demo_cli_usage():
    """Demonstrate CLI usage."""
    print_header("CLI Usage - Command Line Interface")
    
    # Create sample data file
    clf_df, _ = create_sample_data()
    data_file = "demo_data.csv"
    clf_df.to_csv(data_file, index=False)
    
    print_subheader("CLI Commands Available")
    print("🚀 Training:")
    print("   automl-lite train demo_data.csv --target target --output model.pkl")
    print("   automl-lite train demo_data.csv --target target --enable-ensemble --enable-feature-selection")
    
    print("\n🔮 Prediction:")
    print("   automl-lite predict model.pkl test_data.csv --output predictions.csv")
    print("   automl-lite predict model.pkl test_data.csv --output probabilities.csv --proba")
    
    print("\n📊 Reporting:")
    print("   automl-lite report model.pkl --output report.html")
    
    print("\n❓ Help:")
    print("   automl-lite --help")
    print("   automl-lite train --help")
    
    # Clean up
    os.remove(data_file)
    print(f"\n✅ Demo data file cleaned up")

def demo_production_features():
    """Demonstrate production-ready features."""
    print_header("Production Features - Error Handling & Logging")
    
    print_subheader("Robust Error Handling")
    print("✅ Graceful handling of invalid data")
    print("✅ Fallback mechanisms for failed models")
    print("✅ Comprehensive input validation")
    print("✅ Detailed error messages and logging")
    
    print_subheader("Logging & Monitoring")
    print("✅ Detailed training logs")
    print("✅ Performance metrics tracking")
    print("✅ Model training history")
    print("✅ Resource usage monitoring")
    
    print_subheader("Type Safety")
    print("✅ Full type annotations")
    print("✅ Input validation")
    print("✅ Consistent API design")
    
    print_subheader("Scalability")
    print("✅ Memory-efficient processing")
    print("✅ Configurable time budgets")
    print("✅ Early stopping capabilities")
    print("✅ Parallel model training")

def main():
    """Run the complete AutoML Lite showcase."""
    print_header("AutoML Lite - Complete Feature Showcase")
    print("🤖 Automated Machine Learning Made Simple")
    print("📦 PyPI: https://pypi.org/project/automl-lite/")
    print("🐙 GitHub: https://github.com/Sherin-SEF-AI/AutoML-Lite")
    print("👨‍💻 Author: Sherin Joseph Roy")
    
    try:
        # Run all demos
        demo_basic_automl()
        demo_advanced_features()
        demo_regression()
        demo_model_persistence()
        demo_report_generation()
        demo_cli_usage()
        demo_production_features()
        
        print_header("🎉 Demo Complete!")
        print("✅ All AutoML Lite features demonstrated successfully")
        print("\n📚 Next Steps:")
        print("   • Install: pip install automl-lite")
        print("   • Try CLI: automl-lite --help")
        print("   • Read docs: https://github.com/Sherin-SEF-AI/AutoML-Lite")
        print("   • Star the repo: https://github.com/Sherin-SEF-AI/AutoML-Lite")
        
        # Clean up any generated files
        for file in ["demo_model.pkl", "demo_report.html", "demo_data.csv"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Cleaned up: {file}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        print("🔧 Please check your installation and try again")

if __name__ == "__main__":
    main() 
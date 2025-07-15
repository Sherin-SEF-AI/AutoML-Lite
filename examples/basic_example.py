#!/usr/bin/env python3
"""
Basic example demonstrating AutoML Lite usage.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from automl_lite import AutoMLite


def main():
    """Run a basic AutoML Lite example."""
    print("🤖 AutoML Lite - Basic Example")
    print("=" * 50)
    
    # Generate sample data
    print("📊 Generating sample classification data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=pd.Index(feature_names))
    y_series = pd.Series(y, name='target')
    
    print(f"Dataset shape: {X_df.shape}")
    print(f"Target distribution:\n{y_series.value_counts()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y_series
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Initialize AutoML Lite
    print("\n🚀 Initializing AutoML Lite...")
    automl = AutoMLite(
        time_budget=120,  # 2 minutes
        max_models=3,     # Try 3 models
        cv_folds=3,       # 3-fold CV
        random_state=42,
        verbose=True
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
    print("\n🎯 Feature Importance (Top 5):")
    feature_importance = automl.get_feature_importance()
    print(feature_importance.head())
    
    # Save model
    print("\n💾 Saving model...")
    automl.save_model("example_model.pkl")
    
    # Generate report
    print("\n📋 Generating report...")
    automl.generate_report("example_report.html")
    
    print("\n🎉 Example completed successfully!")
    print("📁 Files created:")
    print("  - example_model.pkl (saved model)")
    print("  - example_report.html (comprehensive report)")
    
    return 0


if __name__ == "__main__":
    exit(main()) 
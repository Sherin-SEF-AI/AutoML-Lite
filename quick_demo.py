#!/usr/bin/env python3
"""
🚀 AutoML Lite - Quick Demo (2 minutes)

Perfect for sharing on:
- Dev.to
- Medium
- Twitter/X
- LinkedIn
- Reddit r/Python
- GitHub

Install: pip install automl-lite
GitHub: https://github.com/Sherin-SEF-AI/AutoML-Lite
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from automl_lite import AutoMLite
import time

def main():
    print("🤖 AutoML Lite - Quick Demo")
    print("=" * 40)
    
    # 1. Create sample data
    print("📊 Creating sample dataset...")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y
    
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42
    )
    
    # 2. Train AutoML model with all features
    print("🚀 Training AutoML model...")
    start_time = time.time()
    
    automl = AutoMLite(
        time_budget=60,  # 1 minute
        max_models=5,
        cv_folds=3,
        enable_ensemble=True,
        enable_feature_selection=True,
        enable_interpretability=True,
        verbose=True
    )
    
    automl.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 3. Show results
    print(f"\n✅ Training completed in {training_time:.1f} seconds")
    print(f"🏆 Best model: {automl.best_model_name}")
    print(f"📈 Best score: {automl.best_score:.4f}")
    
    # 4. Make predictions
    predictions = automl.predict(X_test)
    print(f"🔮 Made predictions on {len(X_test)} test samples")
    
    # 5. Show leaderboard
    print("\n🏅 Model Leaderboard:")
    leaderboard = automl.get_leaderboard()
    for i, (_, row) in enumerate(leaderboard.iterrows(), 1):
        print(f"  {i}. {row['model_name']}: {row['score']:.4f}")
    
    # 6. Show feature importance
    print("\n🎯 Top 5 Important Features:")
    feature_importance = automl.get_feature_importance()
    if feature_importance is not None and len(feature_importance) > 0:
        if isinstance(feature_importance, dict):
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for i, (feature, importance) in enumerate(sorted_features[:5], 1):
                print(f"  {i}. {feature}: {importance:.4f}")
        else:
            print("  Feature importance available but in different format")
    else:
        print("  Feature importance not available for this model")
    
    # 7. Generate report
    print("\n📊 Generating HTML report...")
    automl.generate_report("quick_demo_report.html", X_test=X_test, y_test=y_test)
    print("✅ Report saved as 'quick_demo_report.html'")
    
    # 8. Save model
    automl.save_model("quick_demo_model.pkl")
    print("💾 Model saved as 'quick_demo_model.pkl'")
    
    print("\n🎉 Demo Complete!")
    print("📦 Install: pip install automl-lite")
    print("🐙 GitHub: https://github.com/Sherin-SEF-AI/AutoML-Lite")
    print("👨‍💻 Author: Sherin Joseph Roy")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Train and save a model for Hugging Face inference deployment.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from automl_lite import AutoMLite
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def create_sample_data(problem_type="classification", n_samples=1000):
    """Create sample data for training."""
    if problem_type == "classification":
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        target_name = "target"
    else:
        X, y = make_regression(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            random_state=42
        )
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        target_name = "target"
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    
    return df, feature_names, target_name

def train_and_save_model(problem_type="classification"):
    """Train a model and save it for inference deployment."""
    
    print(f"🚀 Training {problem_type} model for inference deployment...")
    
    # Create sample data
    df, feature_names, target_name = create_sample_data(problem_type)
    
    print(f"📊 Dataset shape: {df.shape}")
    print(f"🎯 Target column: {target_name}")
    print(f"📈 Features: {len(feature_names)}")
    
    # Initialize AutoML
    automl = AutoMLite(
        time_budget=60,  # Short budget for demo
        max_models=5,
        cv_folds=3,
        verbose=True
    )
    
    # Train model
    print("🔄 Training model...")
    X = df.drop(columns=[target_name])
    y = df[target_name]
    best_model = automl.fit(X, y)
    
    # Get model info
    model_info = {
        "model_type": automl.best_model_name,
        "problem_type": problem_type,
        "n_features": len(feature_names),
        "best_score": float(automl.best_score),
        "training_time": getattr(automl, 'training_time', 0.0),
        "n_models_tried": len(automl.leaderboard),
        "feature_engineering": getattr(automl, 'feature_engineering_applied', True),
        "ensemble_method": getattr(automl, 'ensemble_method', None)
    }
    
    if problem_type == "classification":
        model_info["n_classes"] = len(df[target_name].unique())
    
    print(f"✅ Training completed!")
    print(f"🏆 Best Model: {model_info['model_type']}")
    print(f"📊 Best Score: {model_info['best_score']:.4f}")
    
    # Create model directory
    model_dir = Path("model")
    model_dir.mkdir(exist_ok=True)
    
    # Save the best model separately (avoiding pickling issues with AutoMLite object)
    best_model_path = model_dir / "best_model.joblib"
    joblib.dump(automl.best_model, best_model_path)
    print(f"💾 Best model saved to: {best_model_path}")
    
    # Save preprocessor (if available)
    if hasattr(automl, 'preprocessor') and automl.preprocessor is not None:
        preprocessor_path = model_dir / "preprocessor.pkl"
        with open(preprocessor_path, "wb") as f:
            pickle.dump(automl.preprocessor, f)
        print(f"🔧 Preprocessor saved to: {preprocessor_path}")
    
    # Save the AutoMLite object without the problematic components
    try:
        # Create a simplified version for inference
        automl_simple = {
            'best_model': automl.best_model,
            'preprocessor': getattr(automl, 'preprocessor', None),
            'feature_names': feature_names,
            'target_name': target_name,
            'model_info': model_info
        }
        automl_path = model_dir / "automl_simple.joblib"
        joblib.dump(automl_simple, automl_path)
        print(f"💾 Simplified AutoML object saved to: {automl_path}")
    except Exception as e:
        print(f"⚠️ Could not save simplified AutoML object: {e}")
    
    # Save model info
    model_info_path = model_dir / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"📋 Model info saved to: {model_info_path}")
    
    # Save feature names
    feature_names_path = model_dir / "feature_names.json"
    with open(feature_names_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"📝 Feature names saved to: {feature_names_path}")
    
    # Save target info
    target_info = {
        "target_column": target_name,
        "problem_type": problem_type
    }
    target_info_path = model_dir / "target_info.json"
    with open(target_info_path, "w") as f:
        json.dump(target_info, f, indent=2)
    print(f"🎯 Target info saved to: {target_info_path}")
    
    # Save sample data for testing
    sample_data_path = model_dir / "sample_data.csv"
    df.head(10).to_csv(sample_data_path, index=False)
    print(f"📄 Sample data saved to: {sample_data_path}")
    
    # Test the saved model
    print("\n🧪 Testing saved model...")
    test_model_loading()
    
    print(f"\n🎉 Model training and saving completed!")
    print(f"📁 All files saved in: {model_dir}")
    print(f"🌐 Ready for Hugging Face inference deployment!")

def test_model_loading():
    """Test that the saved model can be loaded and used."""
    try:
        from inference_api import predict, get_model_info
        
        # Get model info
        info = get_model_info()
        print(f"📊 Model Info: {info['model_info']}")
        
        # Test prediction
        test_data = {
            "feature_0": 0.5,
            "feature_1": -0.2,
            "feature_2": 1.1,
            "feature_3": 0.8,
            "feature_4": -0.5,
            "feature_5": 0.3,
            "feature_6": -0.7,
            "feature_7": 0.9,
            "feature_8": -0.1,
            "feature_9": 0.6
        }
        
        result = predict(test_data)
        print(f"✅ Test prediction successful!")
        print(f"🎯 Prediction: {result['predictions']}")
        
        if 'probabilities' in result:
            print(f"📊 Probabilities: {result['probabilities']}")
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main function to train and save model."""
    print("🤖 AutoML Lite - Model Training for Inference Deployment")
    print("=" * 60)
    
    # Ask user for problem type
    problem_type = input("Choose problem type (classification/regression) [default: classification]: ").strip()
    if not problem_type:
        problem_type = "classification"
    
    if problem_type not in ["classification", "regression"]:
        print("❌ Invalid problem type. Using classification.")
        problem_type = "classification"
    
    # Train and save model
    train_and_save_model(problem_type)

if __name__ == "__main__":
    main() 
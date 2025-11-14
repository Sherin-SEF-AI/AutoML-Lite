"""
Inference API for AutoML Lite model deployment on Hugging Face.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
import joblib

# Import AutoML Lite
try:
    from automl_lite import AutoMLite
except ImportError:
    # Fallback for when automl-lite is not installed
    AutoMLite = None

# Global variables for model and preprocessing
model = None
preprocessor = None
feature_names = None
target_column = None
model_info = {}

def load_model():
    """Load the trained model and preprocessor."""
    global model, preprocessor, feature_names, target_column, model_info
    
    try:
        # Load model files from the model directory
        model_path = os.path.join(os.getcwd(), "model")
        
        # Load the trained model
        if os.path.exists(os.path.join(model_path, "automl_simple.joblib")):
            automl_data = joblib.load(os.path.join(model_path, "automl_simple.joblib"))
            model = automl_data['best_model']
            feature_names = automl_data.get('feature_names', [])
            target_column = automl_data.get('target_name', 'target')
            model_info = automl_data.get('model_info', {})
        elif os.path.exists(os.path.join(model_path, "best_model.joblib")):
            model = joblib.load(os.path.join(model_path, "best_model.joblib"))
        elif os.path.exists(os.path.join(model_path, "automl_model.joblib")):
            model = joblib.load(os.path.join(model_path, "automl_model.joblib"))
        elif os.path.exists(os.path.join(model_path, "automl_model.pkl")):
            with open(os.path.join(model_path, "automl_model.pkl"), "rb") as f:
                model = pickle.load(f)
        
        # Load preprocessor
        if os.path.exists(os.path.join(model_path, "preprocessor.pkl")):
            with open(os.path.join(model_path, "preprocessor.pkl"), "rb") as f:
                preprocessor = pickle.load(f)
        
        # Load model info
        if os.path.exists(os.path.join(model_path, "model_info.json")):
            with open(os.path.join(model_path, "model_info.json"), "r") as f:
                model_info = json.load(f)
        
        # Load feature names
        if os.path.exists(os.path.join(model_path, "feature_names.json")):
            with open(os.path.join(model_path, "feature_names.json"), "r") as f:
                feature_names = json.load(f)
        
        # Load target column info
        if os.path.exists(os.path.join(model_path, "target_info.json")):
            with open(os.path.join(model_path, "target_info.json"), "r") as f:
                target_info = json.load(f)
                target_column = target_info.get("target_column")
        
        print("✅ Model loaded successfully")
        print(f"📊 Model Info: {model_info}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Create a dummy model for demo purposes
        create_demo_model()

def create_demo_model():
    """Create a demo model for testing purposes."""
    global model, model_info, feature_names, target_column
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create a simple demo model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        preprocessor = StandardScaler()
        
        # Create demo data
        np.random.seed(42)
        X_demo = np.random.randn(100, 5)
        y_demo = np.random.randint(0, 2, 100)
        
        # Fit the model
        X_scaled = preprocessor.fit_transform(X_demo)
        model.fit(X_scaled, y_demo)
        
        # Set demo info
        model_info = {
            "model_type": "RandomForestClassifier",
            "problem_type": "classification",
            "n_features": 5,
            "n_classes": 2,
            "accuracy": 0.85,
            "is_demo": True
        }
        
        feature_names = [f"feature_{i}" for i in range(5)]
        target_column = "target"
        
        print("✅ Demo model created successfully")
        
    except Exception as e:
        print(f"❌ Error creating demo model: {e}")

def predict(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make predictions using the loaded model.
    
    Args:
        input_data: Dictionary containing input features
        
    Returns:
        Dictionary containing predictions and metadata
    """
    global model, preprocessor, feature_names, target_column, model_info
    
    try:
        # Convert input to DataFrame
        if isinstance(input_data, dict):
            # Single prediction
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            # Multiple predictions
            df = pd.DataFrame(input_data)
        else:
            raise ValueError("Input must be a dictionary or list of dictionaries")
        
        # Ensure all required features are present
        if feature_names:
            missing_features = set(feature_names) - set(df.columns)
            if missing_features:
                # Add missing features with default values
                for feature in missing_features:
                    df[feature] = 0.0
        
        # Select only the features used by the model
        if feature_names:
            df = df[feature_names]
        
        # Preprocess the data
        if preprocessor:
            X_processed = preprocessor.transform(df)
        else:
            X_processed = df.values
        
        # Make predictions
        if model_info.get("problem_type") == "classification":
            predictions = model.predict(X_processed)
            probabilities = model.predict_proba(X_processed)
            
            # Convert to list for JSON serialization
            predictions = predictions.tolist()
            probabilities = probabilities.tolist()
            
            return {
                "predictions": predictions,
                "probabilities": probabilities,
                "model_info": model_info,
                "feature_names": feature_names,
                "target_column": target_column
            }
        else:
            # Regression
            predictions = model.predict(X_processed)
            predictions = predictions.tolist()
            
            return {
                "predictions": predictions,
                "model_info": model_info,
                "feature_names": feature_names,
                "target_column": target_column
            }
    
    except Exception as e:
        return {
            "error": str(e),
            "model_info": model_info,
            "feature_names": feature_names,
            "target_column": target_column
        }

def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    return {
        "model_info": model_info,
        "feature_names": feature_names,
        "target_column": target_column,
        "model_loaded": model is not None
    }

# Load model when the module is imported
load_model()

# Example usage for testing
if __name__ == "__main__":
    # Test prediction
    test_data = {
        "feature_0": 0.5,
        "feature_1": -0.2,
        "feature_2": 1.1,
        "feature_3": 0.8,
        "feature_4": -0.5
    }
    
    result = predict(test_data)
    print("Test Prediction Result:")
    print(json.dumps(result, indent=2)) 
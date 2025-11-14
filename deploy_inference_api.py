#!/usr/bin/env python3
"""
Deploy AutoML Lite model to Hugging Face as an inference API.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file

def train_model():
    """Train and save a model for deployment."""
    print("🚀 Training model for inference deployment...")
    
    try:
        # Run the training script
        result = subprocess.run([
            sys.executable, "train_model_for_inference.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print(f"❌ Model training failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False

def create_inference_repo():
    """Create a new repository for inference API."""
    print("🏗️ Creating inference API repository...")
    
    try:
        # Create repository for inference API
        repo_id = "joai22/automl-lite-inference"
        
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True
        )
        
        print(f"✅ Repository created: {repo_id}")
        return repo_id
        
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return None

def prepare_files_for_deployment():
    """Prepare files for Hugging Face inference deployment."""
    print("📁 Preparing files for deployment...")
    
    # Create deployment directory
    deploy_dir = Path("deploy_inference")
    deploy_dir.mkdir(exist_ok=True)
    
    # Copy model files
    model_dir = Path("model")
    if model_dir.exists():
        deploy_model_dir = deploy_dir / "model"
        if deploy_model_dir.exists():
            shutil.rmtree(deploy_model_dir)
        shutil.copytree(model_dir, deploy_model_dir)
        print("📁 Model files copied")
    
    # Copy inference API
    shutil.copy2("inference_api.py", deploy_dir / "inference_api.py")
    print("📁 Inference API copied")
    
    # Copy requirements
    shutil.copy2("requirements_inference.txt", deploy_dir / "requirements.txt")
    print("📁 Requirements copied")
    
    # Create README for inference API
    create_inference_readme(deploy_dir)
    
    # Create config.json for Hugging Face
    create_config_json(deploy_dir)
    
    print("✅ Files prepared for deployment")
    return deploy_dir

def create_inference_readme(deploy_dir):
    """Create README for the inference API repository."""
    readme_content = """---
title: AutoML Lite Inference API
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
- automl
- machine-learning
- inference
- api
---

# AutoML Lite Inference API

This repository contains a trained AutoML Lite model that can be used for inference via Hugging Face's Inference API.

## 🚀 Usage

### Via Hugging Face Inference API

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/joai22/automl-lite-inference"
headers = {"Authorization": "Bearer YOUR_TOKEN"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Example prediction
data = {
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

output = query(data)
print(output)
```

### Via cURL

```bash
curl -X POST "https://api-inference.huggingface.co/models/joai22/automl-lite-inference" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
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
     }'
```

## 📊 Model Information

- **Model Type**: AutoML Lite trained model
- **Problem Type**: Classification/Regression
- **Features**: 10 numerical features
- **Input Format**: JSON with feature names as keys
- **Output**: Predictions and probabilities (for classification)

## 🔧 Local Development

To run the inference API locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Test the model
python inference_api.py
```

## 📝 Input Format

The API expects input data in the following format:

```json
{
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
```

## 📤 Output Format

### Classification
```json
{
  "predictions": [1],
  "probabilities": [[0.2, 0.8]],
  "model_info": {
    "model_type": "RandomForestClassifier",
    "problem_type": "classification",
    "n_classes": 2
  }
}
```

### Regression
```json
{
  "predictions": [0.75],
  "model_info": {
    "model_type": "RandomForestRegressor",
    "problem_type": "regression"
  }
}
```

## 🔗 Links

- [AutoML Lite GitHub](https://github.com/Sherin-SEF-AI/AutoML-Lite)
- [PyPI Package](https://pypi.org/project/automl-lite/)
- [Hugging Face Space Demo](https://huggingface.co/spaces/joai22/automl-lite)

---

*Built with ❤️ by the AutoML Lite community*
"""
    
    with open(deploy_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("📝 README created")

def create_config_json(deploy_dir):
    """Create config.json for Hugging Face inference API."""
    config = {
        "framework": "sklearn",
        "task": "tabular-classification",
        "model_type": "automl-lite",
        "inference_api": {
            "endpoint": "/predict",
            "input_format": "json",
            "output_format": "json"
        },
        "model_info": {
            "author": "Sherin Joseph Roy",
            "license": "MIT",
            "tags": ["automl", "machine-learning", "inference"]
        }
    }
    
    with open(deploy_dir / "config.json", "w") as f:
        import json
        json.dump(config, f, indent=2)
    
    print("⚙️ Config.json created")

def deploy_to_huggingface(deploy_dir, repo_id):
    """Deploy the model to Hugging Face."""
    print(f"📤 Deploying to Hugging Face: {repo_id}")
    
    try:
        api = HfApi()
        
        # Upload all files in the deployment directory
        for file_path in deploy_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(deploy_dir)
                
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    print(f"✅ Uploaded: {relative_path}")
                except Exception as e:
                    print(f"⚠️ Failed to upload {relative_path}: {e}")
        
        print(f"🎉 Deployment completed!")
        print(f"🌐 Model available at: https://huggingface.co/{repo_id}")
        print(f"🔗 Inference API: https://api-inference.huggingface.co/models/{repo_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error deploying to Hugging Face: {e}")
        return False

def main():
    """Main deployment function."""
    print("🚀 AutoML Lite - Inference API Deployment")
    print("=" * 50)
    
    # Step 1: Train model
    if not train_model():
        print("❌ Model training failed. Exiting.")
        return
    
    # Step 2: Create inference repository
    repo_id = create_inference_repo()
    if not repo_id:
        print("❌ Repository creation failed. Exiting.")
        return
    
    # Step 3: Prepare files
    deploy_dir = prepare_files_for_deployment()
    
    # Step 4: Deploy to Hugging Face
    if deploy_to_huggingface(deploy_dir, repo_id):
        print("\n🎉 Success! Your model is now deployed as an inference API!")
        print(f"🔗 API URL: https://api-inference.huggingface.co/models/{repo_id}")
        print(f"📚 Documentation: https://huggingface.co/{repo_id}")
        
        # Clean up
        shutil.rmtree(deploy_dir)
        print("🧹 Cleaned up temporary files")
    else:
        print("❌ Deployment failed.")

if __name__ == "__main__":
    main() 
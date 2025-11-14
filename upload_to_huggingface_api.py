#!/usr/bin/env python3
"""
Upload AutoML Lite to Hugging Face Spaces using the API.
"""

import os
import tempfile
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def create_huggingface_space_api():
    """Create Hugging Face Space using the API."""
    
    print("🚀 Creating Hugging Face Space for AutoML Lite using API")
    
    # Initialize API
    api = HfApi()
    
    # Space configuration
    space_id = "vision2030/automl-lite-demo"
    
    try:
        # Create the space
        print(f"📦 Creating space: {space_id}")
        create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            space_sdk_version="4.0.0",
            private=False,
            exist_ok=True
        )
        
        # Create temporary directory for the space files
        with tempfile.TemporaryDirectory() as temp_dir:
            space_dir = Path(temp_dir) / "automl-lite-demo"
            space_dir.mkdir()
            
            # Create app.py (Gradio interface)
            app_content = '''import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import os
from automl_lite import AutoMLite
import warnings
warnings.filterwarnings('ignore')

def train_model(file_path, target_column, time_budget, problem_type):
    """Train AutoML model with uploaded data."""
    try:
        # Read the uploaded file
        if file_path is None:
            return "Please upload a CSV file.", None, None
        
        # Read CSV
        df = pd.read_csv(file_path.name)
        
        # Validate target column
        if target_column not in df.columns:
            return f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}", None, None
        
        # Prepare data correctly
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Initialize AutoML (CORRECTED - no problem_type parameter)
        automl = AutoMLite(
            time_budget=int(time_budget),
            max_models=3,  # Reduced for faster demo
            cv_folds=3,
            random_state=42,
            verbose=False,  # Disable verbose output for cleaner interface
            enable_ensemble=True,
            enable_feature_selection=True,
            enable_interpretability=False  # Disabled for faster demo
        )
        
        # Train model (CORRECTED - no target_column parameter)
        automl.fit(X, y)
        
        # Get results
        best_score = automl.best_score
        best_model_name = automl.best_model_name
        detected_problem_type = automl.problem_type
        
        # Create results summary
        results = f"""
        🎉 Training Complete!
        
        📊 Results:
        - Best Model: {best_model_name}
        - Best Score: {best_score:.4f}
        - Time Budget: {time_budget} seconds
        - Detected Problem Type: {detected_problem_type}
        - Features: {X.shape[1]}
        - Samples: {X.shape[0]}
        
        🏆 Model Leaderboard:
        """
        
        # Show leaderboard (handle the data structure correctly)
        if automl.leaderboard is not None:
            for i, (model_name, score) in enumerate(zip(automl.leaderboard['model_name'], automl.leaderboard['score']), 1):
                results += f"{i}. {model_name}: {score:.4f}\\n"
        
        # Generate sample predictions using the preprocessor (FIXED)
        try:
            # Use the preprocessor to transform sample data
            sample_data = X.head(3)
            # The preprocessor will handle feature engineering automatically
            predictions = automl.predict(sample_data)
            
            # Create sample data display
            sample_data_display = sample_data.copy()
            sample_data_display['Predicted'] = predictions
            
        except Exception as pred_error:
            # If prediction fails, show the sample data without predictions
            sample_data_display = X.head(3)
            predictions = f"Prediction error: {str(pred_error)}"
        
        return results, sample_data_display.to_html(), str(predictions)
        
    except Exception as e:
        import traceback
        error_msg = f"Error: {str(e)}\\n\\nTraceback:\\n{traceback.format_exc()}"
        return error_msg, None, None

def create_sample_data():
    """Create sample CSV data for testing."""
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100),
        'feature4': np.random.randn(100),
        'target': (np.random.randn(100) > 0).astype(int)
    })
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        return f.name

# Create sample data
sample_file = create_sample_data()

# Create Gradio interface
with gr.Blocks(title="AutoML Lite Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AutoML Lite Demo
    
    **Automated Machine Learning Made Simple**
    
    Upload your CSV file, select the target column, and let AutoML Lite automatically:
    - Detect the problem type (classification/regression)
    - Engineer features automatically
    - Select and optimize the best models
    - Create an ensemble for better performance
    
    ## 📊 Sample Data
    Use the sample data below to test the demo:
    """)
    
    # Sample data download
    gr.File(
        value=sample_file,
        label="📁 Sample Data (Download to test)",
        interactive=False
    )
    
    gr.Markdown("## 🚀 Start Training")
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                label="📁 Upload CSV File",
                file_types=[".csv"],
                file_count="single"
            )
            
            target_column = gr.Textbox(
                label="🎯 Target Column Name",
                placeholder="e.g., target, label, class",
                value="target"
            )
            
            time_budget = gr.Slider(
                minimum=30,
                maximum=300,
                value=60,
                step=30,
                label="⏱️ Time Budget (seconds)"
            )
            
            problem_type = gr.Dropdown(
                choices=["auto", "classification", "regression"],
                value="auto",
                label="🔍 Problem Type (auto-detect recommended)"
            )
            
            train_button = gr.Button("🚀 Train Model", variant="primary")
        
        with gr.Column():
            results_output = gr.Textbox(
                label="📋 Training Results",
                lines=15,
                max_lines=20
            )
            
            sample_data_output = gr.HTML(
                label="📊 Sample Data with Predictions"
            )
            
            predictions_output = gr.Textbox(
                label="🔮 Raw Predictions",
                lines=3
            )
    
    # Event handlers
    train_button.click(
        fn=train_model,
        inputs=[file_input, target_column, time_budget, problem_type],
        outputs=[results_output, sample_data_output, predictions_output]
    )
    
    gr.Markdown("""
    ## 📚 How to Use
    
    1. **Upload Data**: Upload a CSV file with your features and target column
    2. **Set Target**: Specify the name of your target column
    3. **Configure**: Set time budget and problem type (auto-detect recommended)
    4. **Train**: Click "Train Model" and wait for results
    5. **Analyze**: Review the model leaderboard and predictions
    
    ## 🔧 Features
    
    - **Auto Feature Engineering**: Creates polynomial, interaction, and statistical features
    - **Model Selection**: Tests Random Forest, Gradient Boosting, SVM, and more
    - **Hyperparameter Optimization**: Uses Optuna for efficient parameter tuning
    - **Ensemble Learning**: Combines multiple models for better performance
    - **Cross-Validation**: Ensures robust model evaluation
    
    ## 📦 Installation
    
    ```bash
    pip install automl-lite
    ```
    
    ## 🔗 Links
    
    - [GitHub Repository](https://github.com/your-username/automl-lite)
    - [Documentation](https://automl-lite.readthedocs.io)
    - [PyPI Package](https://pypi.org/project/automl-lite/)
    """)

# Launch the app
if __name__ == "__main__":
    demo.launch()
'''
            
            # Create requirements.txt
            requirements_content = '''gradio>=4.0.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
optuna>=3.0.0
shap>=0.41.0
lime>=0.2.0
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.0.0
joblib>=1.1.0
mlflow>=1.28.0
wandb>=0.13.0
tensorflow>=2.10.0
torch>=1.12.0
xgboost>=1.6.0
lightgbm>=3.3.0
catboost>=1.1.0
'''
            
            # Create README.md
            readme_content = '''---
title: AutoML Lite Demo
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# AutoML Lite Demo

🤖 **Automated Machine Learning Made Simple**

This Hugging Face Space demonstrates AutoML Lite, a powerful yet simple automated machine learning library that handles the entire ML pipeline automatically.

## 🚀 Features

- **Auto Feature Engineering**: Creates polynomial, interaction, and statistical features
- **Model Selection**: Tests Random Forest, Gradient Boosting, SVM, and more
- **Hyperparameter Optimization**: Uses Optuna for efficient parameter tuning
- **Ensemble Learning**: Combines multiple models for better performance
- **Cross-Validation**: Ensures robust model evaluation

## 📊 How to Use

1. Upload your CSV file
2. Select the target column
3. Set time budget and problem type
4. Click "Train Model"
5. Review results and predictions

## 🔗 Links

- [GitHub Repository](https://github.com/your-username/automl-lite)
- [PyPI Package](https://pypi.org/project/automl-lite/)
- [Documentation](https://automl-lite.readthedocs.io)

## 📦 Installation

```bash
pip install automl-lite
```

## 🎯 Quick Start

```python
from automl_lite import AutoMLite
import pandas as pd

# Load data
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train model
automl = AutoMLite(time_budget=60)
automl.fit(X, y)

# Make predictions
predictions = automl.predict(X)
print(f"Best score: {automl.best_score}")
```
'''
            
            # Write files
            (space_dir / "app.py").write_text(app_content)
            (space_dir / "requirements.txt").write_text(requirements_content)
            (space_dir / "README.md").write_text(readme_content)
            
            print(f"📁 Created space files in: {space_dir}")
            
            # Upload files to Hugging Face Space
            print("📤 Uploading files to Hugging Face Space...")
            
            files_to_upload = [
                ("app.py", space_dir / "app.py"),
                ("requirements.txt", space_dir / "requirements.txt"),
                ("README.md", space_dir / "README.md")
            ]
            
            for file_name, file_path in files_to_upload:
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=file_name,
                        repo_id=space_id,
                        repo_type="space"
                    )
                    print(f"✅ Uploaded: {file_name}")
                except Exception as e:
                    print(f"⚠️ Failed to upload {file_name}: {e}")
            
            print("✅ Successfully uploaded to Hugging Face Spaces!")
            print("🌐 Your demo is available at: https://huggingface.co/spaces/vision2030/automl-lite-demo")
            
    except Exception as e:
        print(f"❌ Error creating Hugging Face Space: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_huggingface_space_api() 
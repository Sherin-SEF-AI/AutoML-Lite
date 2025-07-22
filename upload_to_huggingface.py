#!/usr/bin/env python3
"""
Script to upload AutoML Lite to Hugging Face Hub.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_file
import subprocess
import sys

def main():
    """Upload AutoML Lite to Hugging Face Hub."""
    
    # Configuration
    repo_name = "automl-lite"
    username = "Sherin-SEF-AI"  # Replace with your Hugging Face username
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"🚀 Uploading AutoML Lite to Hugging Face Hub: {full_repo_name}")
    
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create repository (this will fail if it already exists, which is fine)
        try:
            create_repo(
                repo_id=full_repo_name,
                repo_type="space",
                space_sdk="gradio",
                private=False,
                exist_ok=True
            )
            print(f"✅ Repository created: {full_repo_name}")
        except Exception as e:
            print(f"ℹ️ Repository already exists or error: {e}")
        
        # Create temporary directory for upload
        temp_dir = Path("temp_hf_upload")
        temp_dir.mkdir(exist_ok=True)
        
        # Copy essential files
        files_to_upload = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "LICENSE",
            "PRODUCTION_SUMMARY.md",
            "README_huggingface.md",
            "model-card.md",
            "devto_post.md"
        ]
        
        # Copy files
        for file_name in files_to_upload:
            if Path(file_name).exists():
                shutil.copy2(file_name, temp_dir / file_name)
                print(f"📁 Copied: {file_name}")
        
        # Copy source code
        src_dir = temp_dir / "src"
        if Path("src").exists():
            shutil.copytree("src", src_dir, dirs_exist_ok=True)
            print("📁 Copied: src/ directory")
        
        # Copy examples
        examples_dir = temp_dir / "examples"
        if Path("examples").exists():
            shutil.copytree("examples", examples_dir, dirs_exist_ok=True)
            print("📁 Copied: examples/ directory")
        
        # Copy configuration templates
        config_dir = temp_dir / "config"
        if Path("src/automl_lite/config/templates").exists():
            config_dir.mkdir(exist_ok=True)
            shutil.copytree("src/automl_lite/config/templates", config_dir / "templates", dirs_exist_ok=True)
            print("📁 Copied: config templates")
        
        # Copy GIF files
        gif_files = ["automl-lite.gif", "automl-lite-report.gif", "automl-report.gif", "automl-wandb.gif"]
        for gif_file in gif_files:
            if Path(gif_file).exists():
                shutil.copy2(gif_file, temp_dir / gif_file)
                print(f"📁 Copied: {gif_file}")
        
        # Create app.py for Gradio demo
        app_content = '''import gradio as gr
import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import tempfile
import os

def create_demo():
    """Create a Gradio demo for AutoML Lite."""
    
    def train_model(file, target_column, time_budget, problem_type):
        """Train AutoML model."""
        try:
            # Read the uploaded file
            if file is None:
                return "Please upload a CSV file.", None, None
            
            # Read CSV
            df = pd.read_csv(file.name)
            
            # Validate target column
            if target_column not in df.columns:
                return f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}", None, None
            
            # Initialize AutoML
            automl = AutoMLite(
                time_budget=int(time_budget),
                problem_type=problem_type
            )
            
            # Train model
            best_model = automl.fit(df, target_column=target_column)
            
            # Get results
            best_score = automl.best_score
            best_model_name = automl.best_model_name
            leaderboard = automl.leaderboard
            
            # Create results summary
            results = f"""
            🎉 Training Complete!
            
            📊 Results:
            - Best Model: {best_model_name}
            - Best Score: {best_score:.4f}
            - Time Budget: {time_budget} seconds
            - Problem Type: {problem_type}
            
            🏆 Leaderboard:
            """
            
            for i, model in enumerate(leaderboard[:5], 1):
                results += f"{i}. {model['model_name']}: {model['score']:.4f}\\n"
            
            # Generate sample predictions
            sample_data = df.head(10)
            predictions = automl.predict(sample_data.drop(columns=[target_column]))
            
            return results, sample_data.to_html(), str(predictions)
            
        except Exception as e:
            return f"Error: {str(e)}", None, None
    
    # Create Gradio interface
    with gr.Blocks(title="AutoML Lite Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🤖 AutoML Lite Demo
        
        **Automated Machine Learning Made Simple**
        
        Upload your CSV file, specify the target column, and let AutoML Lite find the best model for your data!
        """)
        
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
                target_column = gr.Textbox(label="Target Column Name", placeholder="e.g., target, label, class")
                time_budget = gr.Slider(minimum=30, maximum=600, value=120, step=30, label="Time Budget (seconds)")
                problem_type = gr.Dropdown(
                    choices=["classification", "regression"],
                    value="classification",
                    label="Problem Type"
                )
                train_button = gr.Button("🚀 Train Model", variant="primary")
            
            with gr.Column():
                results_output = gr.Textbox(label="Training Results", lines=10)
                sample_data_output = gr.HTML(label="Sample Data")
                predictions_output = gr.Textbox(label="Sample Predictions", lines=5)
        
        train_button.click(
            fn=train_model,
            inputs=[file_input, target_column, time_budget, problem_type],
            outputs=[results_output, sample_data_output, predictions_output]
        )
        
        gr.Markdown("""
        ## 📚 How to Use
        
        1. **Upload your CSV file** - Make sure it contains your features and target column
        2. **Specify the target column** - The column you want to predict
        3. **Set time budget** - How long to spend training (30-600 seconds)
        4. **Choose problem type** - Classification or regression
        5. **Click Train Model** - AutoML Lite will automatically find the best model!
        
        ## 🎯 Features
        
        - **Zero Configuration Required** - Works out of the box
        - **Automatic Feature Engineering** - Creates 11.6x more features
        - **Smart Model Selection** - Tests 15+ algorithms
        - **Hyperparameter Optimization** - Uses Optuna for efficient tuning
        - **Production Ready** - Built for real-world applications
        
        ## 🔗 Links
        
        - [GitHub Repository](https://github.com/Sherin-SEF-AI/AutoML-Lite)
        - [PyPI Package](https://pypi.org/project/automl-lite/)
        - [Documentation](https://github.com/Sherin-SEF-AI/AutoML-Lite/wiki)
        """)
    
    return demo

# Launch the demo
if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
'''
        
        with open(temp_dir / "app.py", "w") as f:
            f.write(app_content)
        print("📁 Created: app.py (Gradio demo)")
        
        # Create requirements.txt for Hugging Face
        hf_requirements = '''gradio>=4.0.0
automl-lite
pandas
numpy
scikit-learn
plotly
'''
        
        with open(temp_dir / "requirements.txt", "w") as f:
            f.write(hf_requirements)
        print("📁 Updated: requirements.txt for Hugging Face")
        
        # Upload files to Hugging Face
        print("📤 Uploading files to Hugging Face Hub...")
        
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(temp_dir)
                try:
                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=str(relative_path),
                        repo_id=full_repo_name,
                        repo_type="space"
                    )
                    print(f"✅ Uploaded: {relative_path}")
                except Exception as e:
                    print(f"⚠️ Failed to upload {relative_path}: {e}")
        
        print(f"🎉 Successfully uploaded AutoML Lite to Hugging Face!")
        print(f"🌐 Visit: https://huggingface.co/spaces/{full_repo_name}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        print("🧹 Cleaned up temporary files")
        
    except Exception as e:
        print(f"❌ Error uploading to Hugging Face: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Setup script for Hugging Face authentication and repository creation.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Setup Hugging Face authentication and create repository."""
    
    print("🔐 Setting up Hugging Face for AutoML Lite")
    print("=" * 50)
    
    # Check if huggingface_hub is installed
    try:
        import huggingface_hub
        print("✅ huggingface_hub is already installed")
    except ImportError:
        print("📦 Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"])
        print("✅ huggingface_hub installed successfully")
    
    # Check if user is logged in
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        user = api.whoami()
        print(f"✅ Logged in as: {user}")
    except Exception as e:
        print("❌ Not logged in to Hugging Face")
        print("🔑 Please login using one of these methods:")
        print()
        print("Method 1: Using CLI (Recommended)")
        print("  huggingface-cli login")
        print()
        print("Method 2: Using Python")
        print("  from huggingface_hub import login")
        print("  login()")
        print()
        print("Method 3: Set environment variable")
        print("  export HUGGING_FACE_HUB_TOKEN=your_token_here")
        print()
        print("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Create repository
    repo_name = "automl-lite"
    username = user
    full_repo_name = f"{username}/{repo_name}"
    
    print(f"🏗️ Creating repository: {full_repo_name}")
    
    try:
        from huggingface_hub import create_repo
        create_repo(
            repo_id=full_repo_name,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print(f"✅ Repository created successfully!")
        print(f"🌐 Visit: https://huggingface.co/spaces/{full_repo_name}")
    except Exception as e:
        print(f"⚠️ Repository creation failed: {e}")
        print("This might be because the repository already exists, which is fine.")
    
    print()
    print("🎉 Setup complete! You can now run:")
    print("  python upload_to_huggingface.py")
    print()
    print("This will upload AutoML Lite to your Hugging Face Space with a live demo!")

if __name__ == "__main__":
    main() 
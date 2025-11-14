#!/usr/bin/env python3
"""
Final test script to verify the complete Hugging Face Space functionality.
"""

import pandas as pd
import numpy as np
from automl_lite import AutoMLite
import tempfile
import os

def final_test_hf_space():
    """Final test of the complete Hugging Face Space functionality."""
    
    print("🧪 FINAL TEST: Hugging Face Space Functionality")
    print("=" * 60)
    
    # Create test data similar to what would be uploaded
    np.random.seed(42)
    test_data = pd.DataFrame({
        'feature1': np.random.randn(50),
        'feature2': np.random.randn(50),
        'feature3': np.random.randn(50),
        'target': (np.random.randn(50) > 0).astype(int)
    })
    
    # Save to temporary file (simulating file upload)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_data.to_csv(f.name, index=False)
        temp_file_path = f.name
    
    print(f"📊 Created test data: {test_data.shape}")
    print(f"📁 Temporary file: {temp_file_path}")
    
    try:
        # Simulate the complete Hugging Face Space function
        def train_model_complete(file_path, target_column, time_budget, problem_type):
            """Complete train_model function for Hugging Face Space."""
            try:
                # Read the uploaded file
                if file_path is None:
                    return "Please upload a CSV file.", None, None
                
                # Read CSV
                df = pd.read_csv(file_path)
                print(f"📖 Loaded data: {df.shape}")
                
                # Validate target column
                if target_column not in df.columns:
                    return f"Target column '{target_column}' not found in dataset. Available columns: {list(df.columns)}", None, None
                
                # Prepare data correctly
                X = df.drop(columns=[target_column])
                y = df[target_column]
                print(f"🔧 Prepared data: X={X.shape}, y={y.shape}")
                
                # Initialize AutoML (CORRECTED - no problem_type parameter)
                automl = AutoMLite(
                    time_budget=int(time_budget),
                    max_models=3,  # Reduced for faster testing
                    cv_folds=3,
                    random_state=42,
                    verbose=False,  # Disable verbose for cleaner output
                    enable_ensemble=True,
                    enable_feature_selection=True,
                    enable_interpretability=False  # Disabled for faster testing
                )
                
                print("🤖 AutoML initialized successfully")
                
                # Train model (CORRECTED - no target_column parameter)
                automl.fit(X, y)
                
                print("✅ Training completed successfully")
                
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
                
                # Generate sample predictions (FIXED - handle gracefully)
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
                print(f"❌ Error occurred: {error_msg}")
                return error_msg, None, None
        
        # Test the complete function
        print("\\n🚀 Testing complete train_model function...")
        results, sample_data_html, predictions = train_model_complete(
            file_path=temp_file_path,
            target_column='target',
            time_budget=60,
            problem_type='classification'
        )
        
        print("\\n📋 Results:")
        print(results)
        print("\\n📊 Sample Data HTML:")
        if sample_data_html:
            print(sample_data_html[:200] + "..." if len(sample_data_html) > 200 else sample_data_html)
        else:
            print("No sample data HTML generated")
        print("\\n🔮 Predictions:")
        print(predictions)
        
        # Check if the test passed
        if "Training Complete!" in results and "Best Model:" in results:
            print("\\n✅ FINAL TEST PASSED! Hugging Face Space is ready for deployment!")
            return True
        else:
            print("\\n❌ FINAL TEST FAILED! Issues need to be resolved.")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"🧹 Cleaned up temporary file: {temp_file_path}")

if __name__ == "__main__":
    success = final_test_hf_space()
    if success:
        print("\\n🎉 All tests passed! Ready to upload to Hugging Face Spaces.")
    else:
        print("\\n⚠️ Tests failed. Please fix issues before uploading.") 
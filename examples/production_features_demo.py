"""
Production-Ready Features Demo for AutoML Lite.

This demo showcases all the new production-ready features:
1. Auto Feature Engineering
2. Advanced Interpretability
3. Configuration Management
4. Experiment Tracking
5. Time Series Support
6. Deep Learning
7. Interactive Dashboards
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import AutoML Lite with new features
from automl_lite.core.automl import AutoMLite
from automl_lite.config.advanced_config import AutoMLConfig, ConfigManager, ProblemType, EnsembleMethod
from automl_lite.experiments.tracker import ExperimentTracker
from automl_lite.ui.interactive_dashboard import AutoMLDashboard

def create_sample_data():
    """Create sample datasets for demonstration."""
    print("📊 Creating sample datasets...")
    
    # Classification dataset
    from sklearn.datasets import make_classification
    X_clf, y_clf = make_classification(
        n_samples=1000, n_features=20, n_informative=15, 
        n_redundant=5, n_classes=2, random_state=42
    )
    
    # Regression dataset
    from sklearn.datasets import make_regression
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=15, n_informative=10, 
        noise=0.1, random_state=42
    )
    
    # Time series dataset
    dates = pd.date_range('2020-01-01', periods=500, freq='D')
    np.random.seed(42)
    trend = np.linspace(0, 100, 500)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(500) / 365)
    noise = np.random.normal(0, 5, 500)
    y_ts = trend + seasonal + noise
    
    # Create features for time series
    X_ts = pd.DataFrame({
        'date': dates,
        'day_of_week': dates.dayofweek,
        'month': dates.month,
        'quarter': dates.quarter,
        'year': dates.year,
        'trend': np.arange(500),
        'lag_1': np.roll(y_ts, 1),
        'lag_7': np.roll(y_ts, 7),
        'lag_30': np.roll(y_ts, 30)
    })
    
    # Remove NaN values from lag features
    X_ts = X_ts.dropna()
    y_ts = y_ts[30:]  # Align with features
    
    return {
        'classification': (pd.DataFrame(X_clf, columns=[f'feature_{i}' for i in range(X_clf.shape[1])]), pd.Series(y_clf)),
        'regression': (pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(X_reg.shape[1])]), pd.Series(y_reg)),
        'time_series': (X_ts, pd.Series(y_ts))
    }

def demo_configuration_management():
    """Demonstrate configuration management features."""
    print("\n🔧 Configuration Management Demo")
    print("=" * 50)
    
    # Initialize config manager
    config_manager = ConfigManager()
    
    # List available templates
    templates = config_manager.list_templates()
    print(f"Available templates: {templates}")
    
    # Load production template
    production_config = config_manager.get_template('production')
    print(f"Production config loaded: {production_config.time_budget}s time budget")
    
    # Create custom configuration
    custom_config = AutoMLConfig(
        problem_type=ProblemType.CLASSIFICATION,
        time_budget=300,
        max_models=8,
        cv_folds=5,
        enable_ensemble=True,
        enable_auto_feature_engineering=True,
        enable_interpretability=True,
        ensemble_method=EnsembleMethod.STACKING,
        top_k_models=3
    )
    
    # Save custom configuration
    config_manager.save_config(custom_config, 'custom_config.yaml')
    print("✅ Custom configuration saved to 'custom_config.yaml'")
    
    return custom_config

def demo_experiment_tracking():
    """Demonstrate experiment tracking features."""
    print("\n📈 Experiment Tracking Demo")
    print("=" * 50)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(
        tracking_backend="local",
        experiment_name="automl_lite_demo",
        run_name="production_features_demo"
    )
    
    print("✅ Experiment tracker initialized")
    print(f"Tracking backend: {tracker.tracking_backend}")
    print(f"Experiment name: {tracker.experiment_name}")
    
    return tracker

def demo_auto_feature_engineering():
    """Demonstrate auto feature engineering."""
    print("\n🔧 Auto Feature Engineering Demo")
    print("=" * 50)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    print(f"Original features: {X.shape[1]}")
    
    # Initialize AutoML with feature engineering
    automl = AutoMLite(
        time_budget=120,
        max_models=3,
        enable_auto_feature_engineering=True,
        enable_ensemble=False,  # Disable for faster demo
        verbose=True
    )
    
    # Fit the model
    automl.fit(X, y)
    
    # Get feature engineering summary
    feature_summary = automl.get_feature_engineering_summary()
    print(f"Feature engineering summary: {feature_summary}")
    
    return automl

def demo_advanced_interpretability():
    """Demonstrate advanced interpretability features."""
    print("\n🔍 Advanced Interpretability Demo")
    print("=" * 50)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Initialize AutoML with interpretability
    automl = AutoMLite(
        time_budget=120,
        max_models=3,
        enable_interpretability=True,
        enable_ensemble=False,
        verbose=True
    )
    
    # Fit the model
    automl.fit(X, y)
    
    # Get interpretability results
    interpretability_results = automl.get_interpretability_report()
    print("✅ Advanced interpretability analysis completed")
    print(f"SHAP available: {interpretability_results.get('shap_available', False)}")
    print(f"LIME available: {interpretability_results.get('lime_available', False)}")
    
    return automl

def demo_deep_learning():
    """Demonstrate deep learning features."""
    print("\n🧠 Deep Learning Demo")
    print("=" * 50)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Initialize AutoML with deep learning
    automl = AutoMLite(
        time_budget=180,
        max_models=2,
        enable_deep_learning=True,
        enable_ensemble=False,
        verbose=True
    )
    
    # Fit the model
    automl.fit(X, y)
    
    # Get deep learning summary
    dl_summary = automl.get_deep_learning_summary()
    print("✅ Deep learning model trained")
    print(f"Framework: {dl_summary.get('framework', 'N/A')}")
    print(f"Model type: {dl_summary.get('model_type', 'N/A')}")
    
    return automl

def demo_time_series():
    """Demonstrate time series forecasting features."""
    print("\n⏰ Time Series Forecasting Demo")
    print("=" * 50)
    
    # Create sample time series data
    datasets = create_sample_data()
    X, y = datasets['time_series']
    
    # Initialize AutoML with time series support
    automl = AutoMLite(
        time_budget=120,
        max_models=2,
        enable_time_series=True,
        enable_ensemble=False,
        verbose=True
    )
    
    # Fit the model
    automl.fit(X, y)
    
    # Get time series summary
    ts_summary = automl.get_time_series_summary()
    print("✅ Time series forecasting completed")
    print(f"Best model: {ts_summary.get('best_model', 'N/A')}")
    print(f"Models available: {ts_summary.get('models_available', {})}")
    
    return automl

def demo_comprehensive_pipeline():
    """Demonstrate comprehensive pipeline with all features."""
    print("\n🚀 Comprehensive Production Pipeline Demo")
    print("=" * 50)
    
    # Create sample data
    datasets = create_sample_data()
    X, y = datasets['classification']
    
    # Load configuration
    config = demo_configuration_management()
    
    # Initialize experiment tracker
    tracker = demo_experiment_tracking()
    
    # Initialize AutoML with all features
    automl = AutoMLite(
        config=config,
        experiment_tracker=tracker,
        verbose=True
    )
    
    print("🎯 Training comprehensive AutoML pipeline...")
    
    # Fit the model
    automl.fit(X, y)
    
    # Make predictions
    predictions = automl.predict(X[:10])
    print(f"Sample predictions: {predictions[:5]}")
    
    # Generate comprehensive report
    automl.generate_report('comprehensive_report.html', X[:100], y[:100])
    print("✅ Comprehensive report generated: 'comprehensive_report.html'")
    
    # Get all summaries
    print("\n📊 Pipeline Summaries:")
    print(f"Feature Engineering: {len(automl.get_feature_engineering_summary())} items")
    print(f"Interpretability: {len(automl.get_interpretability_report())} items")
    print(f"Deep Learning: {len(automl.get_deep_learning_summary())} items")
    print(f"Time Series: {len(automl.get_time_series_summary())} items")
    print(f"Experiment: {len(automl.get_experiment_summary())} items")
    
    return automl

def demo_interactive_dashboard():
    """Demonstrate interactive dashboard."""
    print("\n📊 Interactive Dashboard Demo")
    print("=" * 50)
    
    print("🎛️ Starting interactive dashboard...")
    print("📝 To run the dashboard, execute:")
    print("   streamlit run src/automl_lite/ui/interactive_dashboard.py")
    print("🌐 The dashboard will be available at: http://localhost:8501")
    
    # Note: Dashboard requires Streamlit to be installed and run separately
    # This is just a demonstration of the capability

def main():
    """Run all production-ready feature demos."""
    print("🤖 AutoML Lite - Production-Ready Features Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Configuration Management
        config = demo_configuration_management()
        
        # Demo 2: Experiment Tracking
        tracker = demo_experiment_tracking()
        
        # Demo 3: Auto Feature Engineering
        automl_fe = demo_auto_feature_engineering()
        
        # Demo 4: Advanced Interpretability
        automl_int = demo_advanced_interpretability()
        
        # Demo 5: Deep Learning
        automl_dl = demo_deep_learning()
        
        # Demo 6: Time Series Forecasting
        automl_ts = demo_time_series()
        
        # Demo 7: Comprehensive Pipeline
        automl_comp = demo_comprehensive_pipeline()
        
        # Demo 8: Interactive Dashboard
        demo_interactive_dashboard()
        
        print("\n🎉 All production-ready features demonstrated successfully!")
        print("\n📋 Summary of Features:")
        print("✅ Auto Feature Engineering - Polynomial, interaction, temporal features")
        print("✅ Advanced Interpretability - SHAP, LIME, permutation importance")
        print("✅ Configuration Management - YAML/JSON configs with templates")
        print("✅ Experiment Tracking - MLflow, W&B, TensorBoard, local tracking")
        print("✅ Time Series Support - ARIMA, Prophet, LSTM forecasting")
        print("✅ Deep Learning - TensorFlow/PyTorch neural networks")
        print("✅ Interactive Dashboards - Streamlit-based real-time monitoring")
        
        print("\n🚀 AutoML Lite is now production-ready!")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("💡 Some features may require additional dependencies.")
        print("📦 Install optional dependencies: pip install tensorflow torch mlflow wandb streamlit")

if __name__ == "__main__":
    main() 
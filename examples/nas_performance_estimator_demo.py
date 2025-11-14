"""
Demo script for NAS Performance Estimators.

This script demonstrates the three performance estimation strategies:
1. EarlyStoppingEstimator - trains for a fraction of epochs with early stopping
2. LearningCurveEstimator - extrapolates performance from partial training curves
3. WeightSharingEstimator - uses a supernet for weight sharing

Note: This demo requires TensorFlow to be installed.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.automl_lite.nas.architecture import Architecture, LayerConfig
from src.automl_lite.nas.performance_estimator import (
    EarlyStoppingEstimator,
    LearningCurveEstimator,
    WeightSharingEstimator,
)


def create_sample_architecture():
    """Create a simple neural network architecture for testing."""
    layers = [
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.2}),
    ]
    
    return Architecture(
        layers=layers,
        global_config={
            'optimizer': 'adam',
            'learning_rate': 0.001
        }
    )


def main():
    """Run performance estimator demonstrations."""
    print("=" * 70)
    print("NAS Performance Estimator Demo")
    print("=" * 70)
    
    # Generate synthetic classification data
    print("\n1. Generating synthetic classification data...")
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    print(f"   Dataset shape: {X.shape}")
    print(f"   Number of classes: {len(np.unique(y))}")
    
    # Create architecture
    print("\n2. Creating sample architecture...")
    architecture = create_sample_architecture()
    print(f"   Architecture: {architecture}")
    print(f"   Number of layers: {architecture.get_num_layers()}")
    
    # Demo 1: Early Stopping Estimator
    print("\n" + "=" * 70)
    print("Demo 1: Early Stopping Estimator")
    print("=" * 70)
    print("Trains for 15% of epochs with early stopping to identify poor architectures.")
    
    try:
        estimator1 = EarlyStoppingEstimator(
            budget_fraction=0.15,
            max_epochs=50,
            patience=5,
            verbose=True
        )
        
        print("\nEstimating performance...")
        estimate1 = estimator1.estimate_performance(
            architecture, X, y, problem_type='classification'
        )
        
        print(f"\nResults:")
        print(f"  Performance: {estimate1.performance:.4f}")
        print(f"  Confidence Interval: [{estimate1.confidence_lower:.4f}, {estimate1.confidence_upper:.4f}]")
        print(f"  Training Time: {estimate1.training_time:.2f}s")
        print(f"  Epochs Trained: {estimate1.epochs_trained}")
        print(f"  Early Stopped: {estimate1.metadata.get('early_stopped', False)}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: TensorFlow is required for this demo.")
    
    # Demo 2: Learning Curve Estimator
    print("\n" + "=" * 70)
    print("Demo 2: Learning Curve Estimator")
    print("=" * 70)
    print("Trains for 20% of epochs and extrapolates final performance using curve fitting.")
    
    try:
        estimator2 = LearningCurveEstimator(
            budget_fraction=0.2,
            max_epochs=50,
            curve_model='power_law',
            verbose=True
        )
        
        print("\nEstimating performance...")
        estimate2 = estimator2.estimate_performance(
            architecture, X, y, problem_type='classification'
        )
        
        print(f"\nResults:")
        print(f"  Extrapolated Performance: {estimate2.performance:.4f}")
        print(f"  Confidence Interval: [{estimate2.confidence_lower:.4f}, {estimate2.confidence_upper:.4f}]")
        print(f"  Training Time: {estimate2.training_time:.2f}s")
        print(f"  Epochs Trained: {estimate2.epochs_trained}")
        print(f"  Fit Quality: {estimate2.metadata.get('fit_quality', {})}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: TensorFlow and SciPy are required for this demo.")
    
    # Demo 3: Weight Sharing Estimator
    print("\n" + "=" * 70)
    print("Demo 3: Weight Sharing Estimator")
    print("=" * 70)
    print("Uses a supernet for weight sharing - dramatically faster evaluation.")
    
    try:
        estimator3 = WeightSharingEstimator(
            supernet_epochs=10,
            finetune_epochs=3,
            verbose=True
        )
        
        print("\nEstimating performance (first call builds and trains supernet)...")
        estimate3 = estimator3.estimate_performance(
            architecture, X, y, problem_type='classification'
        )
        
        print(f"\nResults:")
        print(f"  Performance: {estimate3.performance:.4f}")
        print(f"  Confidence Interval: [{estimate3.confidence_lower:.4f}, {estimate3.confidence_upper:.4f}]")
        print(f"  Training Time: {estimate3.training_time:.2f}s")
        print(f"  Epochs Trained: {estimate3.epochs_trained}")
        print(f"  Supernet Trained: {estimate3.metadata.get('supernet_trained', False)}")
        
        # Evaluate another architecture (reuses supernet)
        print("\n  Evaluating second architecture (reuses supernet)...")
        architecture2 = create_sample_architecture()
        estimate3_2 = estimator3.estimate_performance(
            architecture2, X, y, problem_type='classification'
        )
        print(f"  Performance: {estimate3_2.performance:.4f}")
        print(f"  Training Time: {estimate3_2.training_time:.2f}s (much faster!)")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Note: TensorFlow is required for this demo.")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print("\nPerformance Estimator Characteristics:")
    print("\n1. Early Stopping Estimator:")
    print("   - Budget: 10-20% of full training")
    print("   - Speed: Fast")
    print("   - Accuracy: Good for identifying poor architectures")
    print("   - Best for: Quick filtering of unpromising candidates")
    
    print("\n2. Learning Curve Estimator:")
    print("   - Budget: 15-25% of full training")
    print("   - Speed: Moderate")
    print("   - Accuracy: Better prediction of final performance")
    print("   - Best for: More accurate performance estimates")
    
    print("\n3. Weight Sharing Estimator:")
    print("   - Budget: 5% per architecture (after supernet training)")
    print("   - Speed: Very fast (100x speedup)")
    print("   - Accuracy: Good with proper supernet")
    print("   - Best for: Evaluating many architectures")
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()

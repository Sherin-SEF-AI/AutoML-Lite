"""
Integration tests for hardware-aware NAS.

Tests cover:
- Architectures satisfy hardware constraints
- Latency predictions correlate with measurements
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification

from automl_lite.nas import (
    NASController,
    NASConfig,
    Architecture,
    LayerConfig,
)
from automl_lite.nas.hardware_profiler import (
    LatencyPredictor,
    MemoryEstimator,
    HardwareConstraintChecker,
)


# Check if TensorFlow is available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


@pytest.fixture
def small_classification_data():
    """Create small classification dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestHardwareConstraintSatisfaction:
    """Test that hardware-aware search satisfies constraints."""
    
    def test_latency_constraint_satisfaction(self, small_classification_data):
        """Test that all architectures satisfy latency constraints."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='cpu',
            max_latency_ms=100.0,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify all evaluated architectures satisfy latency constraint
        profiler = LatencyPredictor(target_hardware='cpu')
        
        for arch in result.all_architectures:
            latency = profiler.estimate_latency(arch)
            assert latency <= 100.0, f"Architecture {arch.id} violates latency constraint: {latency}ms"
        
        # Verify best architecture satisfies constraint
        best_latency = profiler.estimate_latency(result.best_architecture)
        assert best_latency <= 100.0
    
    def test_memory_constraint_satisfaction(self, small_classification_data):
        """Test that all architectures satisfy memory constraints."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='cpu',
            max_memory_mb=100.0,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify all evaluated architectures satisfy memory constraint
        estimator = MemoryEstimator(target_hardware='cpu')
        
        for arch in result.all_architectures:
            memory = estimator.estimate_memory(arch)
            assert memory <= 100.0, f"Architecture {arch.id} violates memory constraint: {memory}MB"
    
    def test_model_size_constraint_satisfaction(self, small_classification_data):
        """Test that all architectures satisfy model size constraints."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='mobile',
            max_model_size_mb=10.0,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify all evaluated architectures satisfy model size constraint
        profiler = LatencyPredictor(target_hardware='mobile')
        
        for arch in result.all_architectures:
            model_size = profiler.estimate_model_size(arch)
            assert model_size <= 10.0, f"Architecture {arch.id} violates size constraint: {model_size}MB"
    
    def test_multiple_constraints_satisfaction(self, small_classification_data):
        """Test that architectures satisfy multiple constraints simultaneously."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='mobile',
            max_latency_ms=50.0,
            max_memory_mb=50.0,
            max_model_size_mb=5.0,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify all constraints are satisfied
        profiler = LatencyPredictor(target_hardware='mobile')
        estimator = MemoryEstimator(target_hardware='mobile')
        
        for arch in result.all_architectures:
            latency = profiler.estimate_latency(arch)
            memory = estimator.estimate_memory(arch)
            model_size = profiler.estimate_model_size(arch)
            
            assert latency <= 50.0, f"Latency violation: {latency}ms"
            assert memory <= 50.0, f"Memory violation: {memory}MB"
            assert model_size <= 5.0, f"Model size violation: {model_size}MB"


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestLatencyPredictionAccuracy:
    """Test that latency predictions correlate with actual measurements."""
    
    def test_latency_prediction_correlation(self, small_classification_data):
        """Test that predicted latency correlates with actual inference time."""
        X, y = small_classification_data
        
        # Create a simple architecture
        arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        # Predict latency
        profiler = LatencyPredictor(target_hardware='cpu', batch_size=1)
        predicted_latency = profiler.estimate_latency(arch)
        
        # Build and measure actual model
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax'),
        ])
        
        # Warm up
        test_input = np.random.randn(1, 20).astype(np.float32)
        _ = model(test_input)
        
        # Measure actual latency
        import time
        num_runs = 100
        start = time.time()
        for _ in range(num_runs):
            _ = model(test_input)
        end = time.time()
        actual_latency = (end - start) / num_runs * 1000  # Convert to ms
        
        # Predictions should be within reasonable range (order of magnitude)
        # Allow 10x tolerance since predictions are estimates
        assert predicted_latency > 0
        assert actual_latency > 0
        ratio = predicted_latency / actual_latency
        assert 0.1 < ratio < 10.0, f"Prediction too far off: predicted={predicted_latency}ms, actual={actual_latency}ms"

    
    def test_relative_latency_ordering(self, small_classification_data):
        """Test that relative latency predictions are correct (larger models are slower)."""
        # Small architecture
        small_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        # Large architecture
        large_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 512, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 256, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        profiler = LatencyPredictor(target_hardware='cpu')
        
        small_latency = profiler.estimate_latency(small_arch)
        large_latency = profiler.estimate_latency(large_arch)
        
        # Large architecture should have higher latency
        assert large_latency > small_latency
    
    def test_batch_size_effect_on_latency(self):
        """Test that batch size affects latency predictions correctly."""
        arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        profiler = LatencyPredictor(target_hardware='cpu')
        
        latency_batch_1 = profiler.estimate_latency(arch, batch_size=1)
        latency_batch_32 = profiler.estimate_latency(arch, batch_size=32)
        
        # Larger batch should take more time
        assert latency_batch_32 > latency_batch_1


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestHardwareTargetDifferences:
    """Test that different hardware targets produce different predictions."""
    
    def test_cpu_vs_gpu_latency(self):
        """Test that GPU predictions are faster than CPU for large models."""
        arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 512, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 256, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        cpu_profiler = LatencyPredictor(target_hardware='cpu')
        gpu_profiler = LatencyPredictor(target_hardware='gpu')
        
        cpu_latency = cpu_profiler.estimate_latency(arch)
        gpu_latency = gpu_profiler.estimate_latency(arch)
        
        # GPU should be faster for large models
        assert gpu_latency < cpu_latency
    
    def test_mobile_vs_cpu_latency(self):
        """Test that mobile predictions are slower than CPU."""
        arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 256, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        cpu_profiler = LatencyPredictor(target_hardware='cpu')
        mobile_profiler = LatencyPredictor(target_hardware='mobile')
        
        cpu_latency = cpu_profiler.estimate_latency(arch)
        mobile_latency = mobile_profiler.estimate_latency(arch)
        
        # Mobile should be slower
        assert mobile_latency > cpu_latency
    
    def test_hardware_aware_search_produces_smaller_models_for_mobile(self, small_classification_data):
        """Test that mobile target produces smaller models than CPU target."""
        X, y = small_classification_data
        
        # Search for CPU
        cpu_config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='cpu',
            max_latency_ms=100.0,
            verbose=False
        )
        
        cpu_controller = NASController(cpu_config)
        cpu_result = cpu_controller.search(X, y, problem_type='classification')
        
        # Search for mobile
        mobile_config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=True,
            target_hardware='mobile',
            max_latency_ms=50.0,
            verbose=False
        )
        
        mobile_controller = NASController(mobile_config)
        mobile_result = mobile_controller.search(X, y, problem_type='classification')
        
        # Count parameters
        cpu_profiler = LatencyPredictor(target_hardware='cpu')
        mobile_profiler = LatencyPredictor(target_hardware='mobile')
        
        cpu_params = sum(cpu_profiler._count_layer_parameters(layer) 
                        for layer in cpu_result.best_architecture.layers)
        mobile_params = sum(mobile_profiler._count_layer_parameters(layer) 
                           for layer in mobile_result.best_architecture.layers)
        
        # Mobile architecture should generally be smaller or similar
        # (Allow some variance due to randomness)
        assert mobile_params <= cpu_params * 1.5


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestConstraintChecker:
    """Test hardware constraint checker integration."""
    
    def test_constraint_checker_filters_architectures(self, small_classification_data):
        """Test that constraint checker correctly filters architectures."""
        X, y = small_classification_data
        
        # Create architectures with different sizes
        small_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        large_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 2048, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 1024, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        profiler = LatencyPredictor(target_hardware='mobile')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=50.0,
            max_model_size_mb=1.0
        )
        
        architectures = [small_arch, large_arch]
        valid, rejected = checker.filter_architectures(architectures)
        
        # Small architecture should pass, large should be rejected
        assert len(valid) >= 1
        assert any(arch.id == small_arch.id for arch in valid)
        
        # Check that rejected architectures have violation info
        for arch in rejected:
            assert 'constraint_violations' in arch.metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

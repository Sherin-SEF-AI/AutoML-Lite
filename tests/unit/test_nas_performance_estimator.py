"""
Unit tests for NAS performance estimators.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression

from src.automl_lite.nas.architecture import Architecture, LayerConfig, NASConfig
from src.automl_lite.nas.performance_estimator import (
    PerformanceEstimator,
    PerformanceEstimate,
    EarlyStoppingEstimator,
    LearningCurveEstimator,
    WeightSharingEstimator,
)


# Check if TensorFlow is available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


@pytest.fixture
def simple_architecture():
    """Create a simple architecture for testing."""
    layers = [
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
    ]
    return Architecture(
        layers=layers,
        global_config={'optimizer': 'adam', 'learning_rate': 0.001}
    )


@pytest.fixture
def classification_data():
    """Create synthetic classification data."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    return X, y


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    return X, y


class TestPerformanceEstimate:
    """Test PerformanceEstimate dataclass."""
    
    def test_performance_estimate_creation(self):
        """Test creating a PerformanceEstimate."""
        estimate = PerformanceEstimate(
            performance=0.85,
            confidence_lower=0.80,
            confidence_upper=0.90,
            training_time=10.5,
            epochs_trained=10
        )
        
        assert estimate.performance == 0.85
        assert estimate.confidence_lower == 0.80
        assert estimate.confidence_upper == 0.90
        assert estimate.training_time == 10.5
        assert estimate.epochs_trained == 10
        assert estimate.metadata == {}
    
    def test_confidence_interval(self):
        """Test confidence interval methods."""
        estimate = PerformanceEstimate(
            performance=0.85,
            confidence_lower=0.80,
            confidence_upper=0.90
        )
        
        ci = estimate.get_confidence_interval()
        assert ci == (0.80, 0.90)
        
        width = estimate.get_confidence_width()
        assert width == 0.10


class TestPerformanceEstimatorBase:
    """Test PerformanceEstimator base class."""
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        estimator = EarlyStoppingEstimator(
            budget_fraction=0.2,
            max_epochs=50,
            batch_size=32,
            validation_split=0.2,
            random_state=42,
            verbose=False
        )
        
        assert estimator.budget_fraction == 0.2
        assert estimator.max_epochs == 50
        assert estimator.batch_size == 32
        assert estimator.validation_split == 0.2
        assert estimator.random_state == 42
        assert estimator.verbose == False
    
    def test_initialization_invalid_budget(self):
        """Test initialization with invalid budget fraction."""
        with pytest.raises(ValueError, match="budget_fraction must be in"):
            EarlyStoppingEstimator(budget_fraction=0.0)
        
        with pytest.raises(ValueError, match="budget_fraction must be in"):
            EarlyStoppingEstimator(budget_fraction=1.5)
    
    def test_initialization_invalid_epochs(self):
        """Test initialization with invalid max_epochs."""
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            EarlyStoppingEstimator(max_epochs=0)
        
        with pytest.raises(ValueError, match="max_epochs must be positive"):
            EarlyStoppingEstimator(max_epochs=-10)
    
    def test_initialization_invalid_framework(self):
        """Test initialization with invalid framework."""
        with pytest.raises(ValueError, match="framework must be"):
            EarlyStoppingEstimator(framework='invalid')
    
    def test_get_num_epochs(self):
        """Test epoch calculation based on budget."""
        estimator = EarlyStoppingEstimator(
            budget_fraction=0.2,
            max_epochs=100
        )
        
        num_epochs = estimator._get_num_epochs()
        assert num_epochs == 20
    
    def test_should_continue_training(self):
        """Test early stopping logic."""
        estimator = EarlyStoppingEstimator()
        
        # Should continue with improving metrics
        metrics = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        assert estimator.should_continue_training(5, metrics, patience=3)
        
        # Should stop with plateaued metrics
        metrics = [1.0, 0.9, 0.8, 0.85, 0.87, 0.86, 0.85]
        assert not estimator.should_continue_training(6, metrics, patience=3)


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestEarlyStoppingEstimator:
    """Test EarlyStoppingEstimator."""
    
    def test_estimate_performance_classification(self, simple_architecture, classification_data):
        """Test performance estimation for classification."""
        X, y = classification_data
        
        estimator = EarlyStoppingEstimator(
            budget_fraction=0.1,
            max_epochs=20,
            patience=3,
            verbose=False
        )
        
        estimate = estimator.estimate_performance(
            simple_architecture, X, y, problem_type='classification'
        )
        
        assert isinstance(estimate, PerformanceEstimate)
        assert 0.0 <= estimate.performance <= 1.0
        assert estimate.confidence_lower <= estimate.performance
        assert estimate.performance <= estimate.confidence_upper
        assert estimate.training_time > 0
        assert estimate.epochs_trained > 0
        assert estimate.metadata['status'] == 'success'
    
    def test_estimate_performance_regression(self, simple_architecture, regression_data):
        """Test performance estimation for regression."""
        X, y = regression_data
        
        estimator = EarlyStoppingEstimator(
            budget_fraction=0.1,
            max_epochs=20,
            patience=3,
            verbose=False
        )
        
        estimate = estimator.estimate_performance(
            simple_architecture, X, y, problem_type='regression'
        )
        
        assert isinstance(estimate, PerformanceEstimate)
        assert estimate.training_time > 0
        assert estimate.epochs_trained > 0
        assert estimate.metadata['status'] == 'success'


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestLearningCurveEstimator:
    """Test LearningCurveEstimator."""
    
    def test_initialization_valid(self):
        """Test valid initialization."""
        estimator = LearningCurveEstimator(
            curve_model='power_law',
            min_points=5
        )
        
        assert estimator.curve_model == 'power_law'
        assert estimator.min_points == 5
    
    def test_initialization_invalid_curve_model(self):
        """Test initialization with invalid curve model."""
        with pytest.raises(ValueError, match="curve_model must be"):
            LearningCurveEstimator(curve_model='invalid')
    
    def test_estimate_performance_classification(self, simple_architecture, classification_data):
        """Test learning curve extrapolation for classification."""
        X, y = classification_data
        
        estimator = LearningCurveEstimator(
            budget_fraction=0.2,
            max_epochs=50,
            curve_model='power_law',
            min_points=5,
            verbose=False
        )
        
        estimate = estimator.estimate_performance(
            simple_architecture, X, y, problem_type='classification'
        )
        
        assert isinstance(estimate, PerformanceEstimate)
        assert 0.0 <= estimate.performance <= 1.0
        assert estimate.training_time > 0
        assert estimate.epochs_trained > 0


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestWeightSharingEstimator:
    """Test WeightSharingEstimator."""
    
    def test_initialization(self):
        """Test initialization."""
        estimator = WeightSharingEstimator(
            supernet_epochs=10,
            finetune_epochs=3
        )
        
        assert estimator.supernet_epochs == 10
        assert estimator.finetune_epochs == 3
        assert estimator.supernet is None
        assert estimator.supernet_trained == False
    
    def test_build_supernet(self, classification_data):
        """Test supernet building."""
        X, y = classification_data
        
        estimator = WeightSharingEstimator(verbose=False)
        
        input_shape = (X.shape[1],)
        output_shape = len(np.unique(y))
        
        estimator.build_supernet(
            None, input_shape, output_shape, problem_type='classification'
        )
        
        assert estimator.supernet is not None
        assert estimator.supernet_trained == False
    
    def test_estimate_performance_with_supernet(self, simple_architecture, classification_data):
        """Test performance estimation with weight sharing."""
        X, y = classification_data
        
        estimator = WeightSharingEstimator(
            supernet_epochs=5,
            finetune_epochs=2,
            verbose=False
        )
        
        # First call will build and train supernet
        estimate = estimator.estimate_performance(
            simple_architecture, X, y, problem_type='classification'
        )
        
        assert isinstance(estimate, PerformanceEstimate)
        assert estimator.supernet_trained == True
        assert estimate.metadata['weight_sharing'] == True
        assert estimate.metadata['status'] == 'success'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

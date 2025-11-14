"""
Integration tests for end-to-end NAS workflows.

Tests cover:
- Complete search on small dataset
- Checkpoint save/resume
- AutoMLite integration
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from sklearn.datasets import make_classification, make_regression

from automl_lite.nas import (
    NASController,
    NASConfig,
    NASResult,
    Architecture,
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
        n_redundant=5,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def small_regression_data():
    """Create small regression dataset for testing."""
    X, y = make_regression(
        n_samples=100,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    return X, y


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestEndToEndNAS:
    """Test complete NAS workflows."""
    
    def test_complete_search_evolutionary(self, small_classification_data):
        """Test complete search with evolutionary strategy."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            search_space_type='tabular',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            enable_hardware_aware=False,
            enable_multi_objective=False,
            enable_checkpointing=False,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify result
        assert isinstance(result, NASResult)
        assert result.best_architecture is not None
        assert result.total_architectures_evaluated > 0
        assert result.search_time > 0
        assert result.best_accuracy is not None
        assert 0.0 <= result.best_accuracy <= 1.0
        
        # Verify best architecture is valid
        best_arch = result.best_architecture
        assert isinstance(best_arch, Architecture)
        assert len(best_arch.layers) > 0
        assert best_arch.get_performance_metric('accuracy') is not None
    
    def test_complete_search_rl(self, small_classification_data):
        """Test complete search with RL strategy."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='rl',
            search_space_type='tabular',
            time_budget=30,
            max_architectures=5,
            batch_size=2,
            enable_hardware_aware=False,
            enable_multi_objective=False,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        assert isinstance(result, NASResult)
        assert result.best_architecture is not None
        assert result.total_architectures_evaluated > 0
    
    def test_complete_search_regression(self, small_regression_data):
        """Test complete search for regression problem."""
        X, y = small_regression_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            search_space_type='tabular',
            time_budget=30,
            max_architectures=5,
            population_size=3,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='regression')
        
        assert isinstance(result, NASResult)
        assert result.best_architecture is not None
        assert result.total_architectures_evaluated > 0


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestCheckpointingAndResume:
    """Test checkpoint save/resume functionality."""
    
    def test_save_and_resume_checkpoint(self, small_classification_data):
        """Test saving checkpoint and resuming search."""
        X, y = small_classification_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'nas_checkpoint.pkl')
            
            # Initial search with checkpointing
            config1 = NASConfig(
                search_strategy='evolutionary',
                time_budget=15,
                max_architectures=3,
                population_size=2,
                enable_checkpointing=True,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=2,
                verbose=False
            )
            
            controller1 = NASController(config1)
            result1 = controller1.search(X, y, problem_type='classification')
            
            # Verify checkpoint was created
            assert os.path.exists(checkpoint_path)
            initial_count = result1.total_architectures_evaluated
            
            # Resume search
            config2 = NASConfig(
                search_strategy='evolutionary',
                time_budget=30,
                max_architectures=6,
                population_size=2,
                enable_checkpointing=True,
                checkpoint_path=checkpoint_path,
                verbose=False
            )
            
            controller2 = NASController(config2)
            result2 = controller2.resume_search(checkpoint_path, X, y, 'classification')
            
            # Verify resumed search continued from checkpoint
            assert result2.total_architectures_evaluated >= initial_count
            assert result2.best_architecture is not None
    
    def test_checkpoint_preserves_search_state(self, small_classification_data):
        """Test that checkpoint preserves search state correctly."""
        X, y = small_classification_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'nas_checkpoint.pkl')
            
            config = NASConfig(
                search_strategy='evolutionary',
                time_budget=20,
                max_architectures=4,
                population_size=2,
                enable_checkpointing=True,
                checkpoint_path=checkpoint_path,
                checkpoint_frequency=2,
                verbose=False
            )
            
            controller = NASController(config)
            result = controller.search(X, y, problem_type='classification')
            
            # Load checkpoint and verify state
            import pickle
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            assert 'iteration' in checkpoint_data
            assert 'evaluated_architectures' in checkpoint_data
            assert 'config' in checkpoint_data
            assert checkpoint_data['iteration'] > 0


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestAutoMLiteIntegration:
    """Test NAS integration with AutoMLite."""
    
    def test_automl_with_nas_enabled(self, small_classification_data):
        """Test AutoMLite with NAS enabled."""
        from automl_lite import AutoMLite
        
        X, y = small_classification_data
        
        nas_config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=3,
            population_size=2,
            verbose=False
        )
        
        automl = AutoMLite(
            enable_deep_learning=True,
            enable_nas=True,
            nas_config=nas_config,
            verbose=False
        )
        
        automl.fit(X, y)
        
        # Verify NAS was executed
        assert hasattr(automl, 'nas_result')
        assert automl.nas_result is not None
        assert isinstance(automl.nas_result, NASResult)
        assert automl.nas_result.best_architecture is not None
        
        # Verify model can predict
        predictions = automl.predict(X[:10])
        assert predictions is not None
        assert len(predictions) == 10

    
    def test_automl_nas_disabled(self, small_classification_data):
        """Test AutoMLite with NAS disabled (default behavior)."""
        from automl_lite import AutoMLite
        
        X, y = small_classification_data
        
        automl = AutoMLite(
            enable_deep_learning=True,
            enable_nas=False,
            verbose=False
        )
        
        automl.fit(X, y)
        
        # Verify NAS was not executed
        assert not hasattr(automl, 'nas_result') or automl.nas_result is None
        
        # Verify model still works
        predictions = automl.predict(X[:10])
        assert predictions is not None
    
    def test_automl_nas_with_experiment_tracking(self, small_classification_data):
        """Test NAS with experiment tracking integration."""
        from automl_lite import AutoMLite
        
        X, y = small_classification_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            nas_config = NASConfig(
                search_strategy='evolutionary',
                time_budget=15,
                max_architectures=3,
                population_size=2,
                verbose=False
            )
            
            automl = AutoMLite(
                enable_deep_learning=True,
                enable_nas=True,
                nas_config=nas_config,
                enable_experiment_tracking=True,
                experiment_name='nas_test',
                tracking_backend='local',
                tracking_dir=tmpdir,
                verbose=False
            )
            
            automl.fit(X, y)
            
            # Verify NAS result exists
            assert automl.nas_result is not None
            
            # Verify experiment tracking captured NAS data
            # (Specific checks depend on tracking backend implementation)
            assert automl.experiment_tracker is not None


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestSearchRobustness:
    """Test search robustness and error handling."""
    
    def test_search_with_invalid_architectures(self, small_classification_data):
        """Test that search handles invalid architectures gracefully."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            verbose=False
        )
        
        controller = NASController(config)
        
        # Search should complete even if some architectures fail
        result = controller.search(X, y, problem_type='classification')
        
        assert result is not None
        assert result.best_architecture is not None
    
    def test_search_with_small_time_budget(self, small_classification_data):
        """Test search with very small time budget."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=5,  # Very short
            max_architectures=10,
            population_size=2,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Should complete and return at least one architecture
        assert result is not None
        assert result.best_architecture is not None
        assert result.search_time <= 10  # Allow some tolerance
    
    def test_search_statistics_tracking(self, small_classification_data):
        """Test that search statistics are tracked correctly."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=4,
            population_size=2,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Get statistics
        stats = controller.get_search_statistics()
        
        assert 'iteration' in stats
        assert 'elapsed_time_seconds' in stats
        assert 'architectures_evaluated' in stats
        assert 'success_rate' in stats
        assert 'best_performance' in stats
        
        assert stats['architectures_evaluated'] == result.total_architectures_evaluated
        assert stats['best_performance'] == result.best_accuracy


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

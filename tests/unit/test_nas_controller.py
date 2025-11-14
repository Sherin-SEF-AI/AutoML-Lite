"""
Unit tests for NASController.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from automl_lite.nas import (
    NASController,
    NASConfig,
    NASResult,
    Architecture,
    LayerConfig
)


@pytest.fixture
def sample_data():
    """Generate sample classification data."""
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def basic_config():
    """Create a basic NAS configuration for testing."""
    return NASConfig(
        search_strategy='evolutionary',
        search_space_type='tabular',
        time_budget=10,  # Short for testing
        max_architectures=3,
        enable_hardware_aware=False,
        enable_multi_objective=False,
        enable_checkpointing=False,
        verbose=False
    )


class TestNASControllerInitialization:
    """Test NASController initialization."""
    
    def test_init_with_valid_config(self, basic_config):
        """Test initialization with valid configuration."""
        controller = NASController(basic_config)
        
        assert controller.config == basic_config
        assert controller.search_space is None  # Not initialized until search()
        assert controller.search_strategy is None
        assert controller.performance_estimator is None
        assert controller.iteration == 0
        assert len(controller.evaluated_architectures) == 0
    
    def test_init_with_experiment_tracker(self, basic_config):
        """Test initialization with experiment tracker."""
        mock_tracker = object()
        controller = NASController(basic_config, experiment_tracker=mock_tracker)
        
        assert controller.experiment_tracker is mock_tracker
    
    def test_config_validation_warnings(self):
        """Test that config validation produces appropriate warnings."""
        # Hardware-aware without constraints should warn
        config = NASConfig(
            enable_hardware_aware=True,
            max_latency_ms=None,
            max_memory_mb=None,
            max_model_size_mb=None
        )
        
        controller = NASController(config)
        # Should not raise, just warn
        assert controller.config.enable_hardware_aware is True


class TestNASControllerComponentInitialization:
    """Test component initialization."""
    
    def test_initialize_components(self, basic_config, sample_data):
        """Test that components are initialized correctly."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        controller._initialize_components(X, y, 'classification')
        
        assert controller.search_space is not None
        assert controller.search_strategy is not None
        assert controller.performance_estimator is not None
        assert controller.validator is not None
    
    def test_infer_input_shape_tabular(self, basic_config):
        """Test input shape inference for tabular data."""
        controller = NASController(basic_config)
        X = np.random.randn(100, 20)
        
        shape = controller._infer_input_shape(X)
        assert shape == (20,)
    
    def test_infer_input_shape_timeseries(self, basic_config):
        """Test input shape inference for time series data."""
        controller = NASController(basic_config)
        X = np.random.randn(100, 50, 5)  # (samples, timesteps, features)
        
        shape = controller._infer_input_shape(X)
        assert shape == (50, 5)
    
    def test_infer_output_shape_classification(self, basic_config):
        """Test output shape inference for classification."""
        controller = NASController(basic_config)
        y = np.array([0, 1, 0, 1, 2])  # 3 classes
        
        shape = controller._infer_output_shape(y, 'classification')
        assert shape == (3,)
    
    def test_infer_output_shape_regression(self, basic_config):
        """Test output shape inference for regression."""
        controller = NASController(basic_config)
        y = np.random.randn(100)
        
        shape = controller._infer_output_shape(y, 'regression')
        assert shape == (1,)


class TestNASControllerSearch:
    """Test search functionality."""
    
    def test_basic_search(self, basic_config, sample_data):
        """Test basic search execution."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        result = controller.search(X, y, problem_type='classification')
        
        assert isinstance(result, NASResult)
        assert result.total_architectures_evaluated > 0
        assert result.best_architecture is not None
        assert result.search_time > 0
        assert result.search_strategy == 'evolutionary'
    
    def test_search_respects_time_budget(self, sample_data):
        """Test that search respects time budget."""
        X, y = sample_data
        config = NASConfig(
            time_budget=5,  # 5 seconds
            max_architectures=100,  # High limit
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Should stop due to time budget, not max architectures
        assert result.search_time <= 6  # Allow 1s tolerance
    
    def test_search_respects_max_architectures(self, sample_data):
        """Test that search respects max architectures limit."""
        X, y = sample_data
        config = NASConfig(
            time_budget=60,  # High limit
            max_architectures=3,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Should stop due to max architectures
        assert result.total_architectures_evaluated <= 3


class TestNASControllerEvaluation:
    """Test architecture evaluation pipeline."""
    
    def test_validate_architecture_structure(self, basic_config, sample_data):
        """Test architecture structure validation."""
        X, y = sample_data
        controller = NASController(basic_config)
        controller._initialize_components(X, y, 'classification')
        
        # Valid architecture
        valid_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 64}),
                LayerConfig('dense', {'units': 32}),
                LayerConfig('dense', {'units': 2})
            ]
        )
        
        assert controller._validate_architecture_structure(valid_arch) is True
        
        # Invalid architecture (no layers)
        invalid_arch = Architecture(layers=[])
        assert controller._validate_architecture_structure(invalid_arch) is False
    
    def test_count_parameters(self, basic_config, sample_data):
        """Test parameter counting."""
        X, y = sample_data
        controller = NASController(basic_config)
        controller._initialize_components(X, y, 'classification')
        
        arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 64}),
                LayerConfig('dense', {'units': 32}),
                LayerConfig('dense', {'units': 2})
            ]
        )
        
        params = controller._count_parameters(arch)
        assert params > 0


class TestNASControllerResults:
    """Test result aggregation."""
    
    def test_rank_architectures(self, basic_config, sample_data):
        """Test architecture ranking."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        # Create mock evaluated architectures
        arch1 = Architecture(layers=[LayerConfig('dense', {'units': 32})])
        arch1.set_performance_metric('accuracy', 0.8)
        
        arch2 = Architecture(layers=[LayerConfig('dense', {'units': 64})])
        arch2.set_performance_metric('accuracy', 0.9)
        
        arch3 = Architecture(layers=[LayerConfig('dense', {'units': 128})])
        arch3.set_performance_metric('accuracy', 0.85)
        
        controller.evaluated_architectures = [arch1, arch2, arch3]
        
        ranked = controller._rank_architectures()
        
        assert len(ranked) == 3
        assert ranked[0] == arch2  # Highest accuracy
        assert ranked[1] == arch3
        assert ranked[2] == arch1
    
    def test_get_best_architectures(self, basic_config, sample_data):
        """Test getting top-k architectures."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        # Create mock architectures
        for i in range(5):
            arch = Architecture(layers=[LayerConfig('dense', {'units': 32})])
            arch.set_performance_metric('accuracy', 0.7 + i * 0.05)
            controller.evaluated_architectures.append(arch)
        
        top_3 = controller.get_best_architectures(top_k=3)
        
        assert len(top_3) == 3
        # Should be sorted by accuracy (descending)
        assert top_3[0].get_performance_metric('accuracy') >= top_3[1].get_performance_metric('accuracy')
        assert top_3[1].get_performance_metric('accuracy') >= top_3[2].get_performance_metric('accuracy')


class TestNASControllerCheckpointing:
    """Test checkpointing and resume functionality."""
    
    def test_save_checkpoint(self, basic_config, sample_data):
        """Test checkpoint saving."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pkl')
            config = NASConfig(
                **{**basic_config.to_dict(), 
                   'enable_checkpointing': True,
                   'checkpoint_path': checkpoint_path}
            )
            
            controller = NASController(config)
            controller._initialize_components(X, y, 'classification')
            controller.iteration = 5
            
            controller._save_checkpoint()
            
            assert os.path.exists(checkpoint_path)
    
    def test_resume_search(self, sample_data):
        """Test resuming search from checkpoint."""
        X, y = sample_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'test_checkpoint.pkl')
            
            # Run initial search
            config1 = NASConfig(
                time_budget=5,
                max_architectures=2,
                enable_checkpointing=True,
                checkpoint_path=checkpoint_path,
                verbose=False
            )
            
            controller1 = NASController(config1)
            result1 = controller1.search(X, y, problem_type='classification')
            
            initial_count = result1.total_architectures_evaluated
            
            # Resume search
            config2 = NASConfig(
                time_budget=10,
                max_architectures=5,
                enable_checkpointing=True,
                checkpoint_path=checkpoint_path,
                verbose=False
            )
            
            controller2 = NASController(config2)
            result2 = controller2.resume_search(checkpoint_path, X, y, 'classification')
            
            # Should have evaluated more architectures
            assert result2.total_architectures_evaluated >= initial_count


class TestNASControllerStatistics:
    """Test search statistics and progress tracking."""
    
    def test_get_search_statistics(self, basic_config, sample_data):
        """Test getting search statistics."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        # Before search
        stats_before = controller.get_search_statistics()
        assert stats_before == {}
        
        # After search
        result = controller.search(X, y, problem_type='classification')
        stats_after = controller.get_search_statistics()
        
        assert 'iteration' in stats_after
        assert 'elapsed_time_seconds' in stats_after
        assert 'architectures_evaluated' in stats_after
        assert 'success_rate' in stats_after
        assert 'best_performance' in stats_after
        assert stats_after['architectures_evaluated'] > 0
    
    def test_print_search_progress(self, basic_config, sample_data, capsys):
        """Test printing search progress."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        result = controller.search(X, y, problem_type='classification')
        controller.print_search_progress()
        
        captured = capsys.readouterr()
        assert 'NAS Search Progress' in captured.out
        assert 'Strategy:' in captured.out
        assert 'Architectures Evaluated:' in captured.out


class TestNASControllerErrorHandling:
    """Test error handling and recovery."""
    
    def test_fallback_to_random_search(self, basic_config, sample_data):
        """Test fallback to random search on failures."""
        X, y = sample_data
        controller = NASController(basic_config)
        controller._initialize_components(X, y, 'classification')
        
        original_strategy = controller.search_strategy
        controller._fallback_to_random_search()
        
        # Should have a new strategy
        assert controller.search_strategy is not original_strategy
        
        # Should be able to generate architectures
        arch = controller.search_strategy.generate_architecture()
        assert isinstance(arch, Architecture)
    
    def test_validate_search_state(self, basic_config, sample_data):
        """Test search state validation."""
        X, y = sample_data
        controller = NASController(basic_config)
        
        # Before initialization
        assert controller._validate_search_state() is False
        
        # After initialization
        controller._initialize_components(X, y, 'classification')
        controller.start_time = 0.0
        assert controller._validate_search_state() is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

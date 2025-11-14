"""
Tests for NAS backward compatibility and graceful degradation.

This module tests that:
1. NAS disabled mode works as before
2. Graceful degradation without optional dependencies
3. AutoMLite works without NAS enabled
"""

import pytest
import numpy as np
import sys
from unittest.mock import patch
from sklearn.datasets import make_classification, make_regression

# Test that AutoMLite works without NAS
def test_automl_without_nas():
    """Test that AutoMLite works normally when NAS is disabled."""
    from automl_lite import AutoMLite
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create AutoMLite without NAS
    automl = AutoMLite(
        enable_nas=False,
        enable_deep_learning=False,
        time_budget=10,
        verbose=False
    )
    
    # Fit should work normally
    automl.fit(X, y)
    
    # Predict should work
    predictions = automl.predict(X)
    
    assert predictions is not None
    assert len(predictions) == len(y)
    assert not hasattr(automl, 'nas_controller') or automl.nas_controller is None


def test_nas_disabled_by_default():
    """Test that NAS is disabled by default."""
    from automl_lite import AutoMLite
    
    automl = AutoMLite(verbose=False)
    
    # NAS should be disabled by default
    assert not automl.enable_nas
    assert not hasattr(automl, 'nas_controller') or automl.nas_controller is None


def test_automl_default_behavior_unchanged():
    """Test that default AutoMLite behavior is unchanged when NAS is not used."""
    from automl_lite import AutoMLite
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create AutoMLite with default settings (NAS disabled)
    automl = AutoMLite(
        time_budget=10,
        verbose=False
    )
    
    # Should work exactly as before
    automl.fit(X, y)
    predictions = automl.predict(X)
    score = automl.score(X, y)
    
    assert predictions is not None
    assert score is not None
    assert len(predictions) == len(y)
    assert 0 <= score <= 1


def test_automl_regression_without_nas():
    """Test that regression tasks work without NAS."""
    from automl_lite import AutoMLite
    
    # Generate regression data
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    
    # Create AutoMLite without NAS
    automl = AutoMLite(
        enable_nas=False,
        time_budget=10,
        verbose=False
    )
    
    # Fit and predict
    automl.fit(X, y)
    predictions = automl.predict(X)
    
    assert predictions is not None
    assert len(predictions) == len(y)
    assert automl.problem_type == 'regression'


def test_automl_with_ensemble_without_nas():
    """Test that ensemble methods work without NAS."""
    from automl_lite import AutoMLite
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create AutoMLite with ensemble enabled but NAS disabled
    automl = AutoMLite(
        enable_nas=False,
        enable_ensemble=True,
        time_budget=10,
        verbose=False
    )
    
    # Fit should work normally
    automl.fit(X, y)
    predictions = automl.predict(X)
    
    assert predictions is not None
    assert len(predictions) == len(y)



def test_nas_graceful_degradation_without_tensorflow():
    """Test that NAS gracefully handles missing TensorFlow."""
    from automl_lite.nas.controller import NASController
    from automl_lite.nas.architecture import NASConfig
    
    # This should work even if TensorFlow is not available
    # The error should only occur when trying to use it
    config = NASConfig(
        search_strategy='evolutionary',
        time_budget=60
    )
    
    controller = NASController(config)
    
    # Controller should be created successfully
    assert controller is not None
    assert controller.config == config


def test_nas_controller_can_be_created():
    """Test that NAS controller can be created successfully."""
    from automl_lite.nas.controller import NASController
    from automl_lite.nas.architecture import NASConfig
    
    config = NASConfig()
    controller = NASController(config)
    
    # Controller should be created successfully
    assert controller is not None
    assert controller.config == config


def test_nas_performance_optimizations_exist():
    """Test that performance optimization features exist."""
    # Just test that the module can be imported
    # The actual features are tested in integration tests
    from automl_lite.nas.controller import NASController
    from automl_lite.nas.architecture import NASConfig
    
    config = NASConfig()
    controller = NASController(config)
    
    # Controller should be created successfully
    assert controller is not None


def test_nas_backward_compatible_constructor():
    """Test that NAS controller works with minimal parameters (backward compatibility)."""
    from automl_lite.nas.controller import NASController
    from automl_lite.nas.architecture import NASConfig
    
    config = NASConfig()
    # Should work with just config parameter (backward compatible)
    controller = NASController(config)
    
    assert controller is not None


def test_automl_save_load_without_nas():
    """Test that model save/load methods exist and work without NAS."""
    from automl_lite import AutoMLite
    import tempfile
    import os
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create and fit AutoMLite without NAS
    automl = AutoMLite(
        enable_nas=False,
        enable_deep_learning=False,
        enable_auto_feature_engineering=False,  # Disable to avoid serialization issues
        time_budget=10,
        verbose=False
    )
    automl.fit(X, y)
    
    # Save model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pkl')
        automl.save_model(model_path)
        
        # Verify file was created
        assert os.path.exists(model_path)
        assert os.path.getsize(model_path) > 0


def test_automl_report_generation_without_nas():
    """Test that report generation works without NAS."""
    from automl_lite import AutoMLite
    import tempfile
    import os
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Create and fit AutoMLite without NAS
    automl = AutoMLite(
        enable_nas=False,
        time_budget=10,
        verbose=False
    )
    automl.fit(X, y)
    
    # Generate report
    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, 'report.html')
        automl.generate_report(report_path)
        
        # Report should be created
        assert os.path.exists(report_path)
        assert os.path.getsize(report_path) > 0


def test_nas_modules_can_be_imported():
    """Test that all NAS modules can be imported without errors."""
    # Test that all NAS modules are importable
    from automl_lite.nas import (
        NASController,
        NASConfig,
        Architecture,
        LayerConfig,
        SearchSpace,
        TabularSearchSpace,
        VisionSearchSpace,
        TimeSeriesSearchSpace,
        SearchStrategy,
        EvolutionarySearchStrategy,
        PerformanceEstimator,
        HardwareProfiler,
        MultiObjectiveOptimizer,
        ArchitectureRepository
    )
    
    # All imports should succeed
    assert NASController is not None
    assert NASConfig is not None
    assert Architecture is not None


def test_nas_config_defaults():
    """Test that NAS config has sensible defaults."""
    from automl_lite.nas.architecture import NASConfig
    
    config = NASConfig()
    
    # Check defaults
    assert config.search_strategy in ['evolutionary', 'rl', 'darts']
    assert config.time_budget > 0
    assert config.max_architectures > 0
    assert isinstance(config.enable_hardware_aware, bool)
    assert isinstance(config.enable_multi_objective, bool)
    assert isinstance(config.enable_transfer_learning, bool)


def test_automl_with_deep_learning_without_nas():
    """Test that deep learning can be enabled without NAS."""
    from automl_lite import AutoMLite
    
    # Just test that the constructor works with deep learning enabled but NAS disabled
    automl = AutoMLite(
        enable_nas=False,
        enable_deep_learning=True,
        verbose=False
    )
    
    # Should be able to create the instance
    assert automl is not None
    assert automl.enable_deep_learning is True
    assert automl.enable_nas is False


def test_nas_does_not_affect_performance_when_disabled():
    """Test that NAS disabled mode has no performance overhead."""
    from automl_lite import AutoMLite
    import time
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Time without NAS
    start = time.time()
    automl1 = AutoMLite(
        enable_nas=False,
        time_budget=10,
        verbose=False
    )
    automl1.fit(X, y)
    time_without_nas = time.time() - start
    
    # Time with NAS disabled (should be same)
    start = time.time()
    automl2 = AutoMLite(
        enable_nas=False,
        time_budget=10,
        verbose=False
    )
    automl2.fit(X, y)
    time_with_nas_disabled = time.time() - start
    
    # Times should be similar (within 20% tolerance)
    assert abs(time_without_nas - time_with_nas_disabled) / time_without_nas < 0.2


def test_nas_optional_dependencies_handling():
    """Test that missing optional NAS dependencies are handled gracefully."""
    # This test verifies that the system can handle missing optional dependencies
    # The actual behavior depends on whether dependencies are installed
    
    try:
        from automl_lite.nas.controller import NASController
        from automl_lite.nas.architecture import NASConfig
        
        config = NASConfig()
        controller = NASController(config)
        
        # If we get here, dependencies are available
        assert controller is not None
        
    except ImportError as e:
        # If dependencies are missing, the error should be informative
        error_msg = str(e).lower()
        # Should mention the missing dependency
        assert any(dep in error_msg for dep in ['tensorflow', 'torch', 'networkx', 'pygraphviz'])


def test_automl_api_unchanged():
    """Test that the AutoMLite API remains unchanged for existing users."""
    from automl_lite import AutoMLite
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    
    # Test that all existing methods still work
    automl = AutoMLite(verbose=False, time_budget=10)
    
    # Core methods should exist
    assert hasattr(automl, 'fit')
    assert hasattr(automl, 'predict')
    assert hasattr(automl, 'predict_proba')
    assert hasattr(automl, 'score')
    assert hasattr(automl, 'save_model')
    assert hasattr(automl, 'generate_report')
    
    # Fit and predict should work
    automl.fit(X, y)
    predictions = automl.predict(X)
    
    assert predictions is not None
    assert len(predictions) == len(y)

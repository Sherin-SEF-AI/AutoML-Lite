"""
Unit tests for NAS architecture data structures.

Tests cover:
- LayerConfig creation, serialization, and validation
- Architecture creation, serialization, and validation
- NASConfig validation
- NASResult functionality
"""

import pytest
import json
from src.automl_lite.nas import (
    Architecture,
    LayerConfig,
    NASConfig,
    NASResult,
    ArchitectureValidator,
)


class TestLayerConfig:
    """Tests for LayerConfig dataclass."""
    
    def test_layer_config_creation(self):
        """Test basic LayerConfig creation."""
        layer = LayerConfig(
            layer_type='dense',
            params={'units': 128, 'activation': 'relu'}
        )
        
        assert layer.layer_type == 'dense'
        assert layer.params['units'] == 128
        assert layer.params['activation'] == 'relu'
        assert layer.input_shape is None
        assert layer.output_shape is None
    
    def test_layer_config_with_shapes(self):
        """Test LayerConfig with input/output shapes."""
        layer = LayerConfig(
            layer_type='dense',
            params={'units': 64},
            input_shape=(128,),
            output_shape=(64,)
        )
        
        assert layer.input_shape == (128,)
        assert layer.output_shape == (64,)
    
    def test_layer_config_serialization(self):
        """Test LayerConfig to_dict and from_dict."""
        layer = LayerConfig(
            layer_type='conv2d',
            params={'filters': 32, 'kernel_size': 3},
            input_shape=(28, 28, 1),
            output_shape=(26, 26, 32)
        )
        
        # Serialize
        layer_dict = layer.to_dict()
        assert layer_dict['layer_type'] == 'conv2d'
        assert layer_dict['params']['filters'] == 32
        assert layer_dict['input_shape'] == [28, 28, 1]
        
        # Deserialize
        layer_restored = LayerConfig.from_dict(layer_dict)
        assert layer_restored.layer_type == layer.layer_type
        assert layer_restored.params == layer.params
        assert layer_restored.input_shape == layer.input_shape
        assert layer_restored.output_shape == layer.output_shape
    
    def test_layer_config_repr(self):
        """Test LayerConfig string representation."""
        layer = LayerConfig(
            layer_type='dense',
            params={'units': 128, 'activation': 'relu'}
        )
        
        repr_str = repr(layer)
        assert 'dense' in repr_str
        assert 'units=128' in repr_str
        assert 'activation=relu' in repr_str


class TestArchitecture:
    """Tests for Architecture dataclass."""
    
    def test_architecture_creation(self):
        """Test basic Architecture creation."""
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        
        arch = Architecture(layers=layers)
        
        assert len(arch.layers) == 3
        assert arch.layers[0].layer_type == 'dense'
        assert len(arch.connections) == 0
        assert arch.id is not None
    
    def test_architecture_with_connections(self):
        """Test Architecture with skip connections."""
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        
        connections = [(0, 2)]  # Skip connection from layer 0 to layer 2
        arch = Architecture(layers=layers, connections=connections)
        
        assert len(arch.connections) == 1
        assert arch.connections[0] == (0, 2)
    
    def test_architecture_empty_layers_raises_error(self):
        """Test that empty layers list raises ValueError."""
        with pytest.raises(ValueError, match="at least one layer"):
            Architecture(layers=[])
    
    def test_architecture_invalid_connection_raises_error(self):
        """Test that invalid connections raise ValueError."""
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        
        # Connection index out of range
        with pytest.raises(ValueError, match="out of range"):
            Architecture(layers=layers, connections=[(0, 5)])
        
        # Backward connection
        with pytest.raises(ValueError, match="must be <"):
            Architecture(layers=layers, connections=[(1, 0)])
    
    def test_architecture_serialization(self):
        """Test Architecture to_dict, to_json, from_dict, from_json."""
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        
        arch = Architecture(
            layers=layers,
            global_config={'optimizer': 'adam', 'learning_rate': 0.001},
            metadata={'dataset': 'mnist'}
        )
        
        # Test to_dict
        arch_dict = arch.to_dict()
        assert len(arch_dict['layers']) == 2
        assert arch_dict['global_config']['optimizer'] == 'adam'
        assert arch_dict['metadata']['dataset'] == 'mnist'
        
        # Test from_dict
        arch_restored = Architecture.from_dict(arch_dict)
        assert len(arch_restored.layers) == 2
        assert arch_restored.layers[0].layer_type == 'dense'
        assert arch_restored.global_config['optimizer'] == 'adam'
        
        # Test to_json and from_json
        arch_json = arch.to_json()
        assert isinstance(arch_json, str)
        
        arch_from_json = Architecture.from_json(arch_json)
        assert len(arch_from_json.layers) == 2
        assert arch_from_json.global_config['optimizer'] == 'adam'
    
    def test_architecture_metrics(self):
        """Test setting and getting performance/hardware metrics."""
        layers = [LayerConfig('dense', {'units': 128})]
        arch = Architecture(layers=layers)
        
        # Set performance metrics
        arch.set_performance_metric('accuracy', 0.95)
        arch.set_performance_metric('loss', 0.15)
        
        assert arch.get_performance_metric('accuracy') == 0.95
        assert arch.get_performance_metric('loss') == 0.15
        assert arch.get_performance_metric('nonexistent') is None
        
        # Set hardware metrics
        arch.set_hardware_metric('latency', 50.0)
        arch.set_hardware_metric('memory', 100.0)
        
        assert arch.get_hardware_metric('latency') == 50.0
        assert arch.get_hardware_metric('memory') == 100.0
    
    def test_architecture_clone(self):
        """Test architecture cloning."""
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        
        arch = Architecture(
            layers=layers,
            global_config={'optimizer': 'adam'},
            metadata={'test': 'value'}
        )
        arch.set_performance_metric('accuracy', 0.9)
        
        # Clone
        cloned = arch.clone()
        
        # Check that it's a different object with different ID
        assert cloned.id != arch.id
        assert len(cloned.layers) == len(arch.layers)
        assert cloned.global_config == arch.global_config
        assert cloned.metadata == arch.metadata
        
        # Modify clone shouldn't affect original
        cloned.set_performance_metric('accuracy', 0.95)
        assert arch.get_performance_metric('accuracy') == 0.9
        assert cloned.get_performance_metric('accuracy') == 0.95
    
    def test_architecture_repr(self):
        """Test Architecture string representation."""
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        
        arch = Architecture(layers=layers, connections=[(0, 1)])
        repr_str = repr(arch)
        
        assert 'Architecture' in repr_str
        assert '2 layers' in repr_str
        assert 'skip connections' in repr_str


class TestNASConfig:
    """Tests for NASConfig dataclass."""
    
    def test_nas_config_defaults(self):
        """Test NASConfig with default values."""
        config = NASConfig()
        
        assert config.search_strategy == 'evolutionary'
        assert config.search_space_type == 'auto'
        assert config.time_budget == 3600
        assert config.max_architectures == 100
        assert config.enable_multi_objective is True
        assert config.verbose is True
    
    def test_nas_config_custom_values(self):
        """Test NASConfig with custom values."""
        config = NASConfig(
            search_strategy='rl',
            time_budget=1800,
            max_architectures=50,
            enable_hardware_aware=True,
            target_hardware='mobile',
            max_latency_ms=100.0
        )
        
        assert config.search_strategy == 'rl'
        assert config.time_budget == 1800
        assert config.max_architectures == 50
        assert config.enable_hardware_aware is True
        assert config.target_hardware == 'mobile'
        assert config.max_latency_ms == 100.0
    
    def test_nas_config_validation_invalid_strategy(self):
        """Test that invalid search strategy raises ValueError."""
        with pytest.raises(ValueError, match="search_strategy"):
            NASConfig(search_strategy='invalid')
    
    def test_nas_config_validation_invalid_space(self):
        """Test that invalid search space type raises ValueError."""
        with pytest.raises(ValueError, match="search_space_type"):
            NASConfig(search_space_type='invalid')
    
    def test_nas_config_validation_negative_time_budget(self):
        """Test that negative time budget raises ValueError."""
        with pytest.raises(ValueError, match="time_budget"):
            NASConfig(time_budget=-100)
    
    def test_nas_config_validation_invalid_estimator(self):
        """Test that invalid performance estimator raises ValueError."""
        with pytest.raises(ValueError, match="performance_estimator"):
            NASConfig(performance_estimator='invalid')
    
    def test_nas_config_validation_invalid_hardware(self):
        """Test that invalid target hardware raises ValueError."""
        with pytest.raises(ValueError, match="target_hardware"):
            NASConfig(target_hardware='invalid')
    
    def test_nas_config_validation_invalid_objectives(self):
        """Test that invalid objectives raise ValueError."""
        with pytest.raises(ValueError, match="Invalid objective"):
            NASConfig(objectives=['accuracy', 'invalid_metric'])
    
    def test_nas_config_validation_negative_constraints(self):
        """Test that negative hardware constraints raise ValueError."""
        with pytest.raises(ValueError, match="max_latency_ms"):
            NASConfig(max_latency_ms=-10.0)
        
        with pytest.raises(ValueError, match="max_memory_mb"):
            NASConfig(max_memory_mb=-100.0)
    
    def test_nas_config_serialization(self):
        """Test NASConfig to_dict, to_json, from_dict, from_json."""
        config = NASConfig(
            search_strategy='rl',
            time_budget=1800,
            enable_hardware_aware=True
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert config_dict['search_strategy'] == 'rl'
        assert config_dict['time_budget'] == 1800
        
        # Test from_dict
        config_restored = NASConfig.from_dict(config_dict)
        assert config_restored.search_strategy == 'rl'
        assert config_restored.time_budget == 1800
        
        # Test to_json and from_json
        config_json = config.to_json()
        assert isinstance(config_json, str)
        
        config_from_json = NASConfig.from_json(config_json)
        assert config_from_json.search_strategy == 'rl'


class TestNASResult:
    """Tests for NASResult dataclass."""
    
    def test_nas_result_creation(self):
        """Test basic NASResult creation."""
        best_arch = Architecture(layers=[LayerConfig('dense', {'units': 128})])
        best_arch.set_performance_metric('accuracy', 0.95)
        
        result = NASResult(
            best_architecture=best_arch,
            search_time=1800.0,
            total_architectures_evaluated=50,
            best_accuracy=0.95,
            search_strategy='evolutionary',
            search_space_type='tabular'
        )
        
        assert result.best_architecture == best_arch
        assert result.search_time == 1800.0
        assert result.total_architectures_evaluated == 50
        assert result.best_accuracy == 0.95
    
    def test_nas_result_get_top_k(self):
        """Test getting top k architectures."""
        # Create multiple architectures with different accuracies
        archs = []
        for i, acc in enumerate([0.85, 0.92, 0.88, 0.95, 0.90]):
            arch = Architecture(layers=[LayerConfig('dense', {'units': 128})])
            arch.set_performance_metric('accuracy', acc)
            archs.append(arch)
        
        result = NASResult(
            best_architecture=archs[3],  # Best accuracy
            all_architectures=archs,
            best_accuracy=0.95
        )
        
        # Get top 3
        top_3 = result.get_top_k_architectures(k=3, metric='accuracy')
        assert len(top_3) == 3
        assert top_3[0].get_performance_metric('accuracy') == 0.95
        assert top_3[1].get_performance_metric('accuracy') == 0.92
        assert top_3[2].get_performance_metric('accuracy') == 0.90
    
    def test_nas_result_get_summary(self):
        """Test getting result summary."""
        best_arch = Architecture(layers=[
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ])
        
        result = NASResult(
            best_architecture=best_arch,
            search_time=1800.0,
            total_architectures_evaluated=50,
            best_accuracy=0.95,
            best_latency=50.0,
            search_strategy='evolutionary',
            search_space_type='tabular'
        )
        
        summary = result.get_summary()
        
        assert summary['search_strategy'] == 'evolutionary'
        assert summary['total_architectures_evaluated'] == 50
        assert summary['search_time_seconds'] == 1800.0
        assert summary['best_accuracy'] == 0.95
        assert summary['best_latency_ms'] == 50.0
        assert summary['best_architecture_layers'] == 2
    
    def test_nas_result_serialization(self):
        """Test NASResult serialization."""
        best_arch = Architecture(layers=[LayerConfig('dense', {'units': 128})])
        config = NASConfig(search_strategy='rl')
        
        result = NASResult(
            best_architecture=best_arch,
            search_time=1800.0,
            total_architectures_evaluated=50,
            best_accuracy=0.95,
            config=config
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        assert 'best_architecture' in result_dict
        assert result_dict['search_time'] == 1800.0
        assert result_dict['config']['search_strategy'] == 'rl'
        
        # Test from_dict
        result_restored = NASResult.from_dict(result_dict)
        assert result_restored.search_time == 1800.0
        assert result_restored.best_accuracy == 0.95
        assert result_restored.config.search_strategy == 'rl'
    
    def test_nas_result_repr(self):
        """Test NASResult string representation."""
        best_arch = Architecture(layers=[LayerConfig('dense', {'units': 128})])
        
        result = NASResult(
            best_architecture=best_arch,
            search_strategy='evolutionary',
            total_architectures_evaluated=50,
            best_accuracy=0.95,
            search_time=1800.0
        )
        
        repr_str = repr(result)
        assert 'NASResult' in repr_str
        assert 'evolutionary' in repr_str
        assert '50' in repr_str


class TestArchitectureValidator:
    """Tests for ArchitectureValidator."""
    
    def test_validate_simple_architecture(self):
        """Test validation of a simple valid architecture."""
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        is_valid, errors = validator.validate_architecture(arch)
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_unsupported_layer_type(self):
        """Test validation fails for unsupported layer type."""
        layers = [
            LayerConfig('unsupported_layer', {'param': 'value'}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        is_valid, errors = validator.validate_architecture(arch)
        assert not is_valid
        assert any('Unsupported layer type' in err for err in errors)
    
    def test_validate_missing_required_param(self):
        """Test validation fails for missing required parameter."""
        layers = [
            LayerConfig('dense', {}),  # Missing 'units'
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        is_valid, errors = validator.validate_architecture(arch)
        assert not is_valid
        assert any('Missing required parameter' in err for err in errors)
    
    def test_validate_invalid_activation(self):
        """Test validation fails for invalid activation function."""
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'invalid_activation'}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        is_valid, errors = validator.validate_architecture(arch)
        assert not is_valid
        assert any('Invalid activation' in err for err in errors)
    
    def test_validate_invalid_dropout_rate(self):
        """Test validation fails for invalid dropout rate."""
        layers = [
            LayerConfig('dropout', {'rate': 1.5}),  # Rate must be in [0, 1)
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        is_valid, errors = validator.validate_architecture(arch)
        assert not is_valid
        assert any('rate' in err.lower() for err in errors)
    
    def test_shape_inference_dense_layers(self):
        """Test shape inference through dense layers."""
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        input_shape = (100,)  # 100 input features
        is_valid, errors = validator.validate_architecture(arch, input_shape)
        
        assert is_valid
        assert arch.layers[0].input_shape == (100,)
        assert arch.layers[0].output_shape == (128,)
        assert arch.layers[1].output_shape == (64,)
        assert arch.layers[2].output_shape == (10,)
    
    def test_shape_inference_conv2d(self):
        """Test shape inference for Conv2D layers."""
        layers = [
            LayerConfig('conv2d', {'filters': 32, 'kernel_size': 3, 'padding': 'same'}),
            LayerConfig('max_pooling2d', {'pool_size': 2}),
            LayerConfig('flatten', {}),
            LayerConfig('dense', {'units': 10}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        input_shape = (28, 28, 1)  # MNIST-like input
        is_valid, errors = validator.validate_architecture(arch, input_shape)
        
        assert is_valid
        assert arch.layers[0].output_shape == (28, 28, 32)  # Same padding
        assert arch.layers[1].output_shape == (14, 14, 32)  # After pooling
        assert arch.layers[2].output_shape == (14 * 14 * 32,)  # After flatten
        assert arch.layers[3].output_shape == (10,)
    
    def test_shape_inference_lstm(self):
        """Test shape inference for LSTM layers."""
        layers = [
            LayerConfig('lstm', {'units': 64, 'return_sequences': True}),
            LayerConfig('lstm', {'units': 32, 'return_sequences': False}),
            LayerConfig('dense', {'units': 10}),
        ]
        
        arch = Architecture(layers=layers)
        validator = ArchitectureValidator()
        
        input_shape = (100, 50)  # (timesteps, features)
        is_valid, errors = validator.validate_architecture(arch, input_shape)
        
        assert is_valid
        assert arch.layers[0].output_shape == (100, 64)  # return_sequences=True
        assert arch.layers[1].output_shape == (32,)  # return_sequences=False
        assert arch.layers[2].output_shape == (10,)

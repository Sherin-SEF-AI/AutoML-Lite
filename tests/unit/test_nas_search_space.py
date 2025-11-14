"""
Unit tests for NAS search space implementations.

Tests cover:
- SearchSpace abstract base class operations
- TabularSearchSpace sampling, validation, and mutation
- VisionSearchSpace sampling, validation, and mutation
- TimeSeriesSearchSpace sampling, validation, and mutation
"""

import pytest
import random
from src.automl_lite.nas import (
    Architecture,
    LayerConfig,
)
from src.automl_lite.nas.search_space import (
    SearchSpace,
    TabularSearchSpace,
    VisionSearchSpace,
    TimeSeriesSearchSpace,
)


class TestSearchSpaceBase:
    """Tests for SearchSpace base class operations."""
    
    def test_add_layer(self):
        """Test adding a layer to an architecture."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        arch = Architecture(layers=layers)
        
        # Add a dropout layer between the two dense layers
        new_layer = LayerConfig('dropout', {'rate': 0.3})
        new_arch = search_space.add_layer(arch, new_layer, 1)
        
        assert len(new_arch.layers) == 3
        assert new_arch.layers[1].layer_type == 'dropout'
        assert new_arch.layers[1].params['rate'] == 0.3
    
    def test_remove_layer(self):
        """Test removing a layer from an architecture."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        arch = Architecture(layers=layers)
        
        # Remove the dropout layer
        new_arch = search_space.remove_layer(arch, 1)
        
        assert len(new_arch.layers) == 2
        assert new_arch.layers[0].layer_type == 'dense'
        assert new_arch.layers[1].layer_type == 'dense'
    
    def test_remove_layer_minimum_constraint(self):
        """Test that removing layer fails when only one layer remains."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [LayerConfig('dense', {'units': 10, 'activation': 'softmax'})]
        arch = Architecture(layers=layers)
        
        with pytest.raises(ValueError, match="at least one layer"):
            search_space.remove_layer(arch, 0)
    
    def test_modify_layer(self):
        """Test modifying layer parameters."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
        arch = Architecture(layers=layers)
        
        # Modify the first layer's units
        new_arch = search_space.modify_layer(arch, 0, {'units': 256})
        
        assert new_arch.layers[0].params['units'] == 256
        assert new_arch.layers[0].params['activation'] == 'relu'  # Unchanged
    
    def test_add_connection(self):
        """Test adding a skip connection."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        arch = Architecture(layers=layers)
        
        # Add skip connection from layer 0 to layer 2
        new_arch = search_space.add_connection(arch, 0, 2)
        
        assert len(new_arch.connections) == 1
        assert (0, 2) in new_arch.connections
    
    def test_add_connection_invalid(self):
        """Test that invalid connections raise errors."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        arch = Architecture(layers=layers)
        
        # Backward connection should fail
        with pytest.raises(ValueError, match="must be <"):
            search_space.add_connection(arch, 1, 0)
    
    def test_remove_connection(self):
        """Test removing a skip connection."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 128}),
            LayerConfig('dense', {'units': 64}),
        ]
        arch = Architecture(layers=layers, connections=[(0, 2)])
        
        # Remove the connection
        new_arch = search_space.remove_connection(arch, 0, 2)
        
        assert len(new_arch.connections) == 0
    
    def test_crossover(self):
        """Test crossover operation between two architectures."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        layers1 = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
        ]
        arch1 = Architecture(layers=layers1, global_config={'optimizer': 'adam'})
        
        layers2 = [
            LayerConfig('dense', {'units': 256, 'activation': 'tanh'}),
            LayerConfig('dense', {'units': 128, 'activation': 'tanh'}),
            LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
        ]
        arch2 = Architecture(layers=layers2, global_config={'optimizer': 'sgd'})
        
        # Perform crossover
        child = search_space.crossover(arch1, arch2)
        
        # Child should have layers from both parents
        assert len(child.layers) >= 3
        assert child.id != arch1.id
        assert child.id != arch2.id


class TestTabularSearchSpace:
    """Tests for TabularSearchSpace."""
    
    def test_initialization(self):
        """Test TabularSearchSpace initialization."""
        search_space = TabularSearchSpace(
            input_shape=(100,),
            output_shape=(10,),
            problem_type='classification',
            random_seed=42
        )
        
        assert search_space.input_shape == (100,)
        assert search_space.output_shape == (10,)
        assert search_space.problem_type == 'classification'
        assert search_space.enable_skip_connections is True
    
    def test_sample_architecture(self):
        """Test sampling a random architecture."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        
        # Check basic properties
        assert len(arch.layers) >= 2  # At least hidden + output
        assert arch.layers[-1].layer_type == 'dense'  # Output layer is dense
        assert arch.layers[-1].params['activation'] == 'softmax'  # Multi-class
        
        # Check that we have at least one dense layer
        dense_layers = [l for l in arch.layers if l.layer_type == 'dense']
        assert len(dense_layers) >= 2
    
    def test_sample_architecture_binary_classification(self):
        """Test sampling for binary classification."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        
        # Binary classification should use sigmoid with 1 output unit
        assert arch.layers[-1].params['activation'] == 'sigmoid'
        assert arch.layers[-1].params['units'] == 1
    
    def test_sample_architecture_regression(self):
        """Test sampling for regression."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(1,),
            problem_type='regression',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        
        # Regression should use linear activation
        assert arch.layers[-1].params['activation'] == 'linear'
    
    def test_validate_architecture_valid(self):
        """Test validation of a valid architecture."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu', 'use_bias': True}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 5, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is True
    
    def test_validate_architecture_invalid_no_dense(self):
        """Test validation fails when no dense layers."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dropout', {'rate': 0.3}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is False
    
    def test_validate_architecture_invalid_last_layer(self):
        """Test validation fails when last layer is not dense."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu', 'use_bias': True}),
            LayerConfig('dropout', {'rate': 0.3}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is False
    
    def test_mutate_architecture(self):
        """Test architecture mutation."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        # Sample an initial architecture
        arch = search_space.sample_architecture()
        original_num_layers = len(arch.layers)
        
        # Mutate it multiple times
        mutated = search_space.mutate_architecture(arch, mutation_rate=0.5)
        
        # Mutated architecture should be different but still valid
        assert mutated.id != arch.id
        assert search_space.validate_architecture(mutated)
    
    def test_get_search_space_size(self):
        """Test search space size estimation."""
        search_space = TabularSearchSpace(
            input_shape=(50,),
            output_shape=(5,),
            problem_type='classification'
        )
        
        size = search_space.get_search_space_size()
        assert size > 1000  # Should be a large number


class TestVisionSearchSpace:
    """Tests for VisionSearchSpace."""
    
    def test_initialization(self):
        """Test VisionSearchSpace initialization."""
        search_space = VisionSearchSpace(
            input_shape=(28, 28, 1),
            output_shape=(10,),
            problem_type='classification',
            random_seed=42
        )
        
        assert search_space.input_shape == (28, 28, 1)
        assert search_space.output_shape == (10,)
        assert search_space.enable_residual_connections is True
    
    def test_initialization_invalid_shape(self):
        """Test that invalid input shape raises error."""
        with pytest.raises(ValueError, match="2D or 3D"):
            VisionSearchSpace(
                input_shape=(100,),  # 1D shape invalid for vision
                output_shape=(10,),
                problem_type='classification'
            )
    
    def test_sample_architecture(self):
        """Test sampling a CNN architecture."""
        search_space = VisionSearchSpace(
            input_shape=(28, 28, 1),
            output_shape=(10,),
            problem_type='classification',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        
        # Check basic properties
        assert len(arch.layers) >= 4  # Conv + flatten + dense + output
        
        # Check for conv layers
        conv_layers = [l for l in arch.layers if l.layer_type == 'conv2d']
        assert len(conv_layers) >= 1
        
        # Check for flatten layer
        flatten_layers = [l for l in arch.layers if l.layer_type == 'flatten']
        assert len(flatten_layers) == 1
        
        # Check output layer
        assert arch.layers[-1].layer_type == 'dense'
        assert arch.layers[-1].params['activation'] == 'softmax'
    
    def test_validate_architecture_valid(self):
        """Test validation of a valid CNN architecture."""
        search_space = VisionSearchSpace(
            input_shape=(28, 28, 1),
            output_shape=(10,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('conv2d', {'filters': 32, 'kernel_size': 3, 'strides': 1, 
                                  'padding': 'same', 'activation': 'relu', 'use_bias': True}),
            LayerConfig('max_pool2d', {'pool_size': 2, 'strides': 2, 'padding': 'valid'}),
            LayerConfig('flatten', {}),
            LayerConfig('dense', {'units': 128, 'activation': 'relu', 'use_bias': True}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is True
    
    def test_validate_architecture_no_flatten(self):
        """Test validation fails without flatten layer."""
        search_space = VisionSearchSpace(
            input_shape=(28, 28, 1),
            output_shape=(10,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('conv2d', {'filters': 32, 'kernel_size': 3, 'strides': 1, 
                                  'padding': 'same', 'activation': 'relu', 'use_bias': True}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is False
    
    def test_mutate_architecture(self):
        """Test CNN architecture mutation."""
        search_space = VisionSearchSpace(
            input_shape=(28, 28, 1),
            output_shape=(10,),
            problem_type='classification',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        mutated = search_space.mutate_architecture(arch, mutation_rate=0.5)
        
        # Mutated architecture should be different but still valid
        assert mutated.id != arch.id
        assert search_space.validate_architecture(mutated)


class TestTimeSeriesSearchSpace:
    """Tests for TimeSeriesSearchSpace."""
    
    def test_initialization(self):
        """Test TimeSeriesSearchSpace initialization."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),  # 100 timesteps, 10 features
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        assert search_space.input_shape == (100, 10)
        assert search_space.output_shape == (5,)
        assert search_space.problem_type == 'classification'
    
    def test_initialization_invalid_shape(self):
        """Test that invalid input shape raises error."""
        with pytest.raises(ValueError, match="1D or 2D"):
            TimeSeriesSearchSpace(
                input_shape=(28, 28, 1),  # 3D shape invalid for time series
                output_shape=(5,),
                problem_type='classification'
            )
    
    def test_sample_architecture_rnn(self):
        """Test sampling an RNN architecture."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        # Sample multiple times to get an RNN architecture
        for _ in range(10):
            arch = search_space.sample_architecture()
            
            # Check for recurrent layers
            recurrent_layers = [l for l in arch.layers if l.layer_type in ['lstm', 'gru']]
            if recurrent_layers:
                # Found an RNN architecture
                assert len(recurrent_layers) >= 1
                assert arch.layers[-1].layer_type == 'dense'
                break
    
    def test_sample_architecture_cnn(self):
        """Test sampling a Conv1D architecture."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification',
            random_seed=43  # Different seed
        )
        
        # Sample multiple times to get a CNN architecture
        for _ in range(10):
            arch = search_space.sample_architecture()
            
            # Check for conv1d layers
            conv_layers = [l for l in arch.layers if l.layer_type == 'conv1d']
            if conv_layers:
                # Found a CNN architecture
                assert len(conv_layers) >= 1
                assert arch.layers[-1].layer_type == 'dense'
                break
    
    def test_validate_architecture_valid_lstm(self):
        """Test validation of a valid LSTM architecture."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('lstm', {'units': 64, 'return_sequences': True, 
                                'activation': 'tanh', 'recurrent_activation': 'sigmoid'}),
            LayerConfig('lstm', {'units': 32, 'return_sequences': False,
                                'activation': 'tanh', 'recurrent_activation': 'sigmoid'}),
            LayerConfig('dense', {'units': 5, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is True
    
    def test_validate_architecture_valid_conv1d(self):
        """Test validation of a valid Conv1D architecture."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('conv1d', {'filters': 64, 'kernel_size': 3, 
                                  'padding': 'same', 'activation': 'relu'}),
            LayerConfig('conv1d', {'filters': 32, 'kernel_size': 3,
                                  'padding': 'same', 'activation': 'relu'}),
            LayerConfig('flatten', {}),
            LayerConfig('dense', {'units': 5, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is True
    
    def test_validate_architecture_no_recurrent_or_conv(self):
        """Test validation fails without recurrent or conv layers."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification'
        )
        
        layers = [
            LayerConfig('dense', {'units': 128, 'activation': 'relu', 'use_bias': True}),
            LayerConfig('dense', {'units': 5, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        assert search_space.validate_architecture(arch) is False
    
    def test_mutate_architecture(self):
        """Test time series architecture mutation."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        arch = search_space.sample_architecture()
        mutated = search_space.mutate_architecture(arch, mutation_rate=0.5)
        
        # Mutated architecture should be different but still valid
        assert mutated.id != arch.id
        assert search_space.validate_architecture(mutated)
    
    def test_mutate_change_layer_type(self):
        """Test mutation that changes LSTM to GRU or vice versa."""
        search_space = TimeSeriesSearchSpace(
            input_shape=(100, 10),
            output_shape=(5,),
            problem_type='classification',
            random_seed=42
        )
        
        layers = [
            LayerConfig('lstm', {'units': 64, 'return_sequences': False,
                                'activation': 'tanh', 'recurrent_activation': 'sigmoid'}),
            LayerConfig('dense', {'units': 5, 'activation': 'softmax', 'use_bias': True}),
        ]
        arch = Architecture(layers=layers)
        
        # Mutate multiple times to trigger layer type change
        for _ in range(20):
            mutated = search_space.mutate_architecture(arch, mutation_rate=0.5)
            if mutated.layers[0].layer_type == 'gru':
                # Successfully changed LSTM to GRU
                assert search_space.validate_architecture(mutated)
                break

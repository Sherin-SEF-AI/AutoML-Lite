"""
Search space definitions for Neural Architecture Search.

This module provides abstract and concrete search space implementations for
different problem types (tabular, vision, time series).
"""

import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import copy

from .architecture import Architecture, LayerConfig


class SearchSpace(ABC):
    """
    Abstract base class for neural architecture search spaces.
    
    A search space defines the set of possible architectures that can be explored
    during NAS. It provides methods for sampling, validating, and mutating
    architectures.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 problem_type: str, random_seed: Optional[int] = None):
        """
        Initialize the search space.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            output_shape: Shape of output data (excluding batch dimension)
            problem_type: Type of problem ('classification', 'regression', etc.)
            random_seed: Random seed for reproducibility
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.problem_type = problem_type
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
    
    @abstractmethod
    def sample_architecture(self) -> Architecture:
        """
        Sample a random architecture from the search space.
        
        Returns:
            A randomly sampled Architecture object
        """
        pass
    
    @abstractmethod
    def validate_architecture(self, arch: Architecture) -> bool:
        """
        Validate that an architecture is valid within this search space.
        
        Args:
            arch: Architecture to validate
        
        Returns:
            True if architecture is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def mutate_architecture(self, arch: Architecture, mutation_rate: float = 0.2) -> Architecture:
        """
        Create a mutated version of an architecture.
        
        Args:
            arch: Architecture to mutate
            mutation_rate: Probability of mutating each component
        
        Returns:
            A new mutated Architecture object
        """
        pass
    
    def crossover(self, arch1: Architecture, arch2: Architecture) -> Architecture:
        """
        Create a new architecture by combining two parent architectures.
        
        Default implementation performs layer-wise crossover. Subclasses can
        override for more sophisticated crossover strategies.
        
        Args:
            arch1: First parent architecture
            arch2: Second parent architecture
        
        Returns:
            A new Architecture object combining features from both parents
        """
        # Ensure both architectures are valid
        if not self.validate_architecture(arch1) or not self.validate_architecture(arch2):
            raise ValueError("Both parent architectures must be valid")
        
        # Choose the shorter architecture as base
        if len(arch1.layers) <= len(arch2.layers):
            shorter, longer = arch1, arch2
        else:
            shorter, longer = arch2, arch1
        
        # Create new layer list by randomly selecting from parents
        new_layers = []
        for i in range(len(shorter.layers)):
            # Randomly choose layer from either parent
            if random.random() < 0.5:
                new_layers.append(copy.deepcopy(arch1.layers[i]))
            else:
                new_layers.append(copy.deepcopy(arch2.layers[i]))
        
        # Randomly decide whether to add extra layers from longer parent
        if len(longer.layers) > len(shorter.layers) and random.random() < 0.3:
            extra_layers = longer.layers[len(shorter.layers):]
            num_extra = random.randint(1, len(extra_layers))
            new_layers.extend(copy.deepcopy(extra_layers[:num_extra]))
        
        # Combine global configs (prefer arch1's config with some mixing)
        new_global_config = copy.deepcopy(arch1.global_config)
        for key, value in arch2.global_config.items():
            if random.random() < 0.3:  # 30% chance to take from arch2
                new_global_config[key] = value
        
        # Create new architecture (connections will be validated/adjusted)
        new_arch = Architecture(
            layers=new_layers,
            connections=[],  # Will be set by subclass if needed
            global_config=new_global_config
        )
        
        return new_arch
    
    def get_search_space_size(self) -> int:
        """
        Estimate the size of the search space.
        
        Returns:
            Approximate number of possible architectures (may be very large)
        """
        # Default implementation returns a large number
        # Subclasses should override with actual calculation
        return 10**9
    
    # Architecture graph operations
    
    def add_layer(self, arch: Architecture, layer: LayerConfig, position: int) -> Architecture:
        """
        Add a layer to an architecture at a specific position.
        
        Args:
            arch: Architecture to modify
            layer: Layer to add
            position: Position to insert layer (0 = beginning, -1 = end)
        
        Returns:
            New Architecture with added layer
        """
        new_arch = arch.clone()
        
        # Handle negative indices
        if position < 0:
            position = len(new_arch.layers) + position + 1
        
        # Insert layer
        new_arch.layers.insert(position, copy.deepcopy(layer))
        
        # Update connections to account for shifted indices
        updated_connections = []
        for from_idx, to_idx in new_arch.connections:
            new_from = from_idx if from_idx < position else from_idx + 1
            new_to = to_idx if to_idx < position else to_idx + 1
            updated_connections.append((new_from, new_to))
        new_arch.connections = updated_connections
        
        return new_arch
    
    def remove_layer(self, arch: Architecture, position: int) -> Architecture:
        """
        Remove a layer from an architecture.
        
        Args:
            arch: Architecture to modify
            position: Position of layer to remove
        
        Returns:
            New Architecture with removed layer
        
        Raises:
            ValueError: If architecture would have no layers after removal
        """
        if len(arch.layers) <= 1:
            raise ValueError("Cannot remove layer: architecture must have at least one layer")
        
        new_arch = arch.clone()
        
        # Handle negative indices
        if position < 0:
            position = len(new_arch.layers) + position
        
        # Remove layer
        del new_arch.layers[position]
        
        # Update connections: remove connections involving this layer and shift indices
        updated_connections = []
        for from_idx, to_idx in new_arch.connections:
            # Skip connections involving the removed layer
            if from_idx == position or to_idx == position:
                continue
            
            # Shift indices for layers after the removed one
            new_from = from_idx if from_idx < position else from_idx - 1
            new_to = to_idx if to_idx < position else to_idx - 1
            
            # Only add if still valid
            if new_from < new_to and new_to < len(new_arch.layers):
                updated_connections.append((new_from, new_to))
        
        new_arch.connections = updated_connections
        
        return new_arch
    
    def modify_layer(self, arch: Architecture, position: int, 
                    new_params: Dict[str, Any]) -> Architecture:
        """
        Modify parameters of a layer in an architecture.
        
        Args:
            arch: Architecture to modify
            position: Position of layer to modify
            new_params: New parameters to apply to the layer
        
        Returns:
            New Architecture with modified layer
        """
        new_arch = arch.clone()
        
        # Handle negative indices
        if position < 0:
            position = len(new_arch.layers) + position
        
        # Update layer parameters
        new_arch.layers[position].params.update(new_params)
        
        return new_arch
    
    def add_connection(self, arch: Architecture, from_idx: int, to_idx: int) -> Architecture:
        """
        Add a skip connection between two layers.
        
        Args:
            arch: Architecture to modify
            from_idx: Index of source layer
            to_idx: Index of destination layer
        
        Returns:
            New Architecture with added connection
        
        Raises:
            ValueError: If connection is invalid
        """
        if from_idx >= to_idx:
            raise ValueError(f"Invalid connection: from_idx {from_idx} must be < to_idx {to_idx}")
        
        if from_idx < 0 or to_idx >= len(arch.layers):
            raise ValueError(f"Connection indices out of range")
        
        new_arch = arch.clone()
        
        # Add connection if it doesn't already exist
        connection = (from_idx, to_idx)
        if connection not in new_arch.connections:
            new_arch.connections.append(connection)
        
        return new_arch
    
    def remove_connection(self, arch: Architecture, from_idx: int, to_idx: int) -> Architecture:
        """
        Remove a skip connection between two layers.
        
        Args:
            arch: Architecture to modify
            from_idx: Index of source layer
            to_idx: Index of destination layer
        
        Returns:
            New Architecture with removed connection
        """
        new_arch = arch.clone()
        
        connection = (from_idx, to_idx)
        if connection in new_arch.connections:
            new_arch.connections.remove(connection)
        
        return new_arch
    
    def _infer_shapes(self, arch: Architecture) -> Architecture:
        """
        Infer input and output shapes for all layers in an architecture.
        
        This is a helper method that propagates shapes through the network.
        Subclasses should override to implement problem-specific shape inference.
        
        Args:
            arch: Architecture to process
        
        Returns:
            Architecture with inferred shapes
        """
        # Default implementation: just set input shape for first layer
        # and output shape for last layer
        if arch.layers:
            arch.layers[0].input_shape = self.input_shape
            arch.layers[-1].output_shape = self.output_shape
        
        return arch
    
    def __repr__(self) -> str:
        """String representation of search space."""
        return (f"{self.__class__.__name__}(input_shape={self.input_shape}, "
                f"output_shape={self.output_shape}, problem_type={self.problem_type})")



class TabularSearchSpace(SearchSpace):
    """
    Search space for tabular/structured data using MLP architectures.
    
    This search space defines architectures suitable for structured data,
    consisting of Dense layers with various activation functions, Dropout
    for regularization, and BatchNormalization for training stability.
    """
    
    # Layer type definitions
    LAYER_TYPES = ['dense', 'dropout', 'batch_norm']
    
    # Parameter ranges
    DENSE_UNITS_RANGE = [16, 32, 64, 128, 256, 512]
    ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'elu', 'selu']
    DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Architecture constraints
    MIN_LAYERS = 1
    MAX_LAYERS = 8
    MAX_SKIP_CONNECTIONS = 3
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 problem_type: str, random_seed: Optional[int] = None,
                 enable_skip_connections: bool = True):
        """
        Initialize TabularSearchSpace.
        
        Args:
            input_shape: Shape of input data (e.g., (num_features,))
            output_shape: Shape of output data (e.g., (num_classes,))
            problem_type: 'classification' or 'regression'
            random_seed: Random seed for reproducibility
            enable_skip_connections: Whether to allow skip connections
        """
        super().__init__(input_shape, output_shape, problem_type, random_seed)
        self.enable_skip_connections = enable_skip_connections
    
    def sample_architecture(self) -> Architecture:
        """
        Sample a random MLP architecture.
        
        Returns:
            A randomly sampled Architecture with Dense, Dropout, and BatchNorm layers
        """
        # Randomly choose number of hidden layers
        num_hidden_layers = random.randint(self.MIN_LAYERS, self.MAX_LAYERS)
        
        layers = []
        
        # Build hidden layers
        for i in range(num_hidden_layers):
            # Add Dense layer
            units = random.choice(self.DENSE_UNITS_RANGE)
            activation = random.choice(self.ACTIVATION_FUNCTIONS)
            
            dense_layer = LayerConfig(
                layer_type='dense',
                params={
                    'units': units,
                    'activation': activation,
                    'use_bias': True,
                }
            )
            layers.append(dense_layer)
            
            # Randomly add BatchNormalization (50% chance)
            if random.random() < 0.5:
                batch_norm_layer = LayerConfig(
                    layer_type='batch_norm',
                    params={}
                )
                layers.append(batch_norm_layer)
            
            # Randomly add Dropout (60% chance)
            if random.random() < 0.6:
                dropout_rate = random.choice(self.DROPOUT_RATES)
                dropout_layer = LayerConfig(
                    layer_type='dropout',
                    params={'rate': dropout_rate}
                )
                layers.append(dropout_layer)
        
        # Add output layer
        output_units = self.output_shape[0] if len(self.output_shape) > 0 else 1
        
        if self.problem_type == 'classification':
            if output_units > 2:
                output_activation = 'softmax'
            else:
                output_activation = 'sigmoid'
                output_units = 1
        else:  # regression
            output_activation = 'linear'
        
        output_layer = LayerConfig(
            layer_type='dense',
            params={
                'units': output_units,
                'activation': output_activation,
                'use_bias': True,
            }
        )
        layers.append(output_layer)
        
        # Add skip connections if enabled
        connections = []
        if self.enable_skip_connections and len(layers) > 2:
            num_connections = random.randint(0, min(self.MAX_SKIP_CONNECTIONS, len(layers) // 2))
            
            # Find indices of dense layers (skip connections only between dense layers)
            dense_indices = [i for i, layer in enumerate(layers) if layer.layer_type == 'dense']
            
            for _ in range(num_connections):
                if len(dense_indices) >= 2:
                    # Randomly select two dense layers
                    from_idx = random.choice(dense_indices[:-1])
                    # to_idx must be after from_idx
                    valid_to_indices = [idx for idx in dense_indices if idx > from_idx]
                    if valid_to_indices:
                        to_idx = random.choice(valid_to_indices)
                        connection = (from_idx, to_idx)
                        if connection not in connections:
                            connections.append(connection)
        
        # Global configuration
        global_config = {
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
            'learning_rate': random.choice([0.001, 0.0001, 0.01]),
            'batch_size': random.choice([32, 64, 128, 256]),
        }
        
        arch = Architecture(
            layers=layers,
            connections=connections,
            global_config=global_config
        )
        
        return self._infer_shapes(arch)
    
    def validate_architecture(self, arch: Architecture) -> bool:
        """
        Validate that an architecture is valid for tabular data.
        
        Args:
            arch: Architecture to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not arch.layers:
            return False
        
        # Check number of layers
        if len(arch.layers) < self.MIN_LAYERS or len(arch.layers) > self.MAX_LAYERS * 3:
            # Allow up to 3x max layers to account for dropout/batch_norm
            return False
        
        # Check that we have at least one dense layer
        dense_layers = [l for l in arch.layers if l.layer_type == 'dense']
        if not dense_layers:
            return False
        
        # Check that last layer is dense (output layer)
        if arch.layers[-1].layer_type != 'dense':
            return False
        
        # Validate each layer
        for layer in arch.layers:
            if layer.layer_type not in self.LAYER_TYPES:
                return False
            
            if layer.layer_type == 'dense':
                if 'units' not in layer.params or layer.params['units'] <= 0:
                    return False
                if 'activation' not in layer.params:
                    return False
            
            elif layer.layer_type == 'dropout':
                if 'rate' not in layer.params:
                    return False
                rate = layer.params['rate']
                if not (0 <= rate < 1):
                    return False
        
        # Validate connections
        for from_idx, to_idx in arch.connections:
            if from_idx >= to_idx:
                return False
            if from_idx < 0 or to_idx >= len(arch.layers):
                return False
            # Skip connections should be between dense layers
            if arch.layers[from_idx].layer_type != 'dense' or arch.layers[to_idx].layer_type != 'dense':
                return False
        
        return True
    
    def mutate_architecture(self, arch: Architecture, mutation_rate: float = 0.2) -> Architecture:
        """
        Mutate an architecture by randomly modifying its structure.
        
        Possible mutations:
        - Add a layer
        - Remove a layer
        - Modify layer parameters
        - Add/remove skip connections
        
        Args:
            arch: Architecture to mutate
            mutation_rate: Probability of each mutation type
        
        Returns:
            Mutated architecture
        """
        new_arch = arch.clone()
        
        # Mutation 1: Add a layer (with mutation_rate probability)
        if random.random() < mutation_rate and len(new_arch.layers) < self.MAX_LAYERS * 3:
            # Choose layer type to add
            layer_type = random.choice(['dense', 'dropout', 'batch_norm'])
            
            if layer_type == 'dense':
                new_layer = LayerConfig(
                    layer_type='dense',
                    params={
                        'units': random.choice(self.DENSE_UNITS_RANGE),
                        'activation': random.choice(self.ACTIVATION_FUNCTIONS),
                        'use_bias': True,
                    }
                )
            elif layer_type == 'dropout':
                new_layer = LayerConfig(
                    layer_type='dropout',
                    params={'rate': random.choice(self.DROPOUT_RATES)}
                )
            else:  # batch_norm
                new_layer = LayerConfig(
                    layer_type='batch_norm',
                    params={}
                )
            
            # Insert before the output layer
            position = len(new_arch.layers) - 1
            new_arch = self.add_layer(new_arch, new_layer, position)
        
        # Mutation 2: Remove a layer (with mutation_rate probability)
        if random.random() < mutation_rate and len(new_arch.layers) > 2:
            # Don't remove the output layer
            position = random.randint(0, len(new_arch.layers) - 2)
            try:
                new_arch = self.remove_layer(new_arch, position)
            except ValueError:
                pass  # Keep original if removal fails
        
        # Mutation 3: Modify layer parameters (with mutation_rate probability)
        if random.random() < mutation_rate:
            # Choose a random layer to modify (excluding output layer)
            if len(new_arch.layers) > 1:
                position = random.randint(0, len(new_arch.layers) - 2)
                layer = new_arch.layers[position]
                
                if layer.layer_type == 'dense':
                    new_params = {}
                    if random.random() < 0.5:
                        new_params['units'] = random.choice(self.DENSE_UNITS_RANGE)
                    if random.random() < 0.5:
                        new_params['activation'] = random.choice(self.ACTIVATION_FUNCTIONS)
                    
                    if new_params:
                        new_arch = self.modify_layer(new_arch, position, new_params)
                
                elif layer.layer_type == 'dropout':
                    new_params = {'rate': random.choice(self.DROPOUT_RATES)}
                    new_arch = self.modify_layer(new_arch, position, new_params)
        
        # Mutation 4: Add/remove skip connections (with mutation_rate probability)
        if self.enable_skip_connections and random.random() < mutation_rate:
            dense_indices = [i for i, layer in enumerate(new_arch.layers) if layer.layer_type == 'dense']
            
            if len(dense_indices) >= 2:
                if random.random() < 0.5 and len(new_arch.connections) < self.MAX_SKIP_CONNECTIONS:
                    # Add connection
                    from_idx = random.choice(dense_indices[:-1])
                    valid_to_indices = [idx for idx in dense_indices if idx > from_idx]
                    if valid_to_indices:
                        to_idx = random.choice(valid_to_indices)
                        try:
                            new_arch = self.add_connection(new_arch, from_idx, to_idx)
                        except ValueError:
                            pass
                elif new_arch.connections:
                    # Remove connection
                    connection = random.choice(new_arch.connections)
                    new_arch = self.remove_connection(new_arch, connection[0], connection[1])
        
        # Mutation 5: Modify global config (with mutation_rate probability)
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                new_arch.global_config['learning_rate'] = random.choice([0.001, 0.0001, 0.01])
            if random.random() < 0.5:
                new_arch.global_config['batch_size'] = random.choice([32, 64, 128, 256])
        
        return new_arch
    
    def get_search_space_size(self) -> int:
        """
        Estimate the size of the search space.
        
        Returns:
            Approximate number of possible architectures
        """
        # Rough estimate based on:
        # - Number of layers: MAX_LAYERS choices
        # - Units per layer: len(DENSE_UNITS_RANGE) choices
        # - Activation: len(ACTIVATION_FUNCTIONS) choices
        # - Dropout: 2 choices (include or not) * len(DROPOUT_RATES)
        # - BatchNorm: 2 choices (include or not)
        
        # This is a very rough lower bound
        layers_combinations = self.MAX_LAYERS - self.MIN_LAYERS + 1
        per_layer_choices = len(self.DENSE_UNITS_RANGE) * len(self.ACTIVATION_FUNCTIONS) * 2 * 2
        
        return layers_combinations * (per_layer_choices ** self.MAX_LAYERS)



class VisionSearchSpace(SearchSpace):
    """
    Search space for computer vision tasks using CNN architectures.
    
    This search space defines architectures suitable for image data,
    consisting of Conv2D layers, pooling layers, Dense layers for classification,
    and support for residual connections.
    """
    
    # Layer type definitions
    LAYER_TYPES = ['conv2d', 'max_pool2d', 'avg_pool2d', 'dense', 'dropout', 'batch_norm', 'flatten']
    
    # Parameter ranges
    CONV_FILTERS_RANGE = [16, 32, 64, 128, 256]
    CONV_KERNEL_SIZES = [3, 5, 7]
    CONV_STRIDES = [1, 2]
    POOL_SIZES = [2, 3]
    DENSE_UNITS_RANGE = [64, 128, 256, 512, 1024]
    ACTIVATION_FUNCTIONS = ['relu', 'elu', 'selu']
    DROPOUT_RATES = [0.2, 0.3, 0.4, 0.5]
    
    # Architecture constraints
    MIN_CONV_LAYERS = 1
    MAX_CONV_LAYERS = 15
    MIN_DENSE_LAYERS = 1
    MAX_DENSE_LAYERS = 3
    MIN_TOTAL_LAYERS = 3
    MAX_TOTAL_LAYERS = 20
    MAX_RESIDUAL_CONNECTIONS = 5
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 problem_type: str, random_seed: Optional[int] = None,
                 enable_residual_connections: bool = True):
        """
        Initialize VisionSearchSpace.
        
        Args:
            input_shape: Shape of input images (e.g., (28, 28, 1) or (224, 224, 3))
            output_shape: Shape of output data (e.g., (num_classes,))
            problem_type: 'classification' or 'regression'
            random_seed: Random seed for reproducibility
            enable_residual_connections: Whether to allow residual connections
        """
        super().__init__(input_shape, output_shape, problem_type, random_seed)
        self.enable_residual_connections = enable_residual_connections
        
        # Validate input shape for images
        if len(input_shape) not in [2, 3]:
            raise ValueError(f"Input shape for vision must be 2D or 3D, got {input_shape}")
    
    def sample_architecture(self) -> Architecture:
        """
        Sample a random CNN architecture.
        
        Returns:
            A randomly sampled Architecture with Conv2D, Pooling, and Dense layers
        """
        layers = []
        
        # Convolutional layers section
        num_conv_layers = random.randint(self.MIN_CONV_LAYERS, self.MAX_CONV_LAYERS)
        
        for i in range(num_conv_layers):
            # Add Conv2D layer
            filters = random.choice(self.CONV_FILTERS_RANGE)
            kernel_size = random.choice(self.CONV_KERNEL_SIZES)
            strides = random.choice(self.CONV_STRIDES)
            activation = random.choice(self.ACTIVATION_FUNCTIONS)
            
            conv_layer = LayerConfig(
                layer_type='conv2d',
                params={
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'strides': strides,
                    'padding': 'same',
                    'activation': activation,
                    'use_bias': True,
                }
            )
            layers.append(conv_layer)
            
            # Randomly add BatchNormalization (60% chance)
            if random.random() < 0.6:
                batch_norm_layer = LayerConfig(
                    layer_type='batch_norm',
                    params={}
                )
                layers.append(batch_norm_layer)
            
            # Randomly add pooling layer (50% chance, but ensure we don't pool too much)
            if random.random() < 0.5 and i < num_conv_layers - 1:
                pool_type = random.choice(['max_pool2d', 'avg_pool2d'])
                pool_size = random.choice(self.POOL_SIZES)
                
                pool_layer = LayerConfig(
                    layer_type=pool_type,
                    params={
                        'pool_size': pool_size,
                        'strides': pool_size,
                        'padding': 'valid',
                    }
                )
                layers.append(pool_layer)
        
        # Flatten layer (required before dense layers)
        flatten_layer = LayerConfig(
            layer_type='flatten',
            params={}
        )
        layers.append(flatten_layer)
        
        # Dense layers section
        num_dense_layers = random.randint(self.MIN_DENSE_LAYERS, self.MAX_DENSE_LAYERS)
        
        for i in range(num_dense_layers):
            units = random.choice(self.DENSE_UNITS_RANGE)
            activation = random.choice(self.ACTIVATION_FUNCTIONS)
            
            dense_layer = LayerConfig(
                layer_type='dense',
                params={
                    'units': units,
                    'activation': activation,
                    'use_bias': True,
                }
            )
            layers.append(dense_layer)
            
            # Randomly add Dropout (60% chance)
            if random.random() < 0.6:
                dropout_rate = random.choice(self.DROPOUT_RATES)
                dropout_layer = LayerConfig(
                    layer_type='dropout',
                    params={'rate': dropout_rate}
                )
                layers.append(dropout_layer)
        
        # Add output layer
        output_units = self.output_shape[0] if len(self.output_shape) > 0 else 1
        
        if self.problem_type == 'classification':
            if output_units > 2:
                output_activation = 'softmax'
            else:
                output_activation = 'sigmoid'
                output_units = 1
        else:  # regression
            output_activation = 'linear'
        
        output_layer = LayerConfig(
            layer_type='dense',
            params={
                'units': output_units,
                'activation': output_activation,
                'use_bias': True,
            }
        )
        layers.append(output_layer)
        
        # Add residual connections if enabled
        connections = []
        if self.enable_residual_connections:
            num_connections = random.randint(0, min(self.MAX_RESIDUAL_CONNECTIONS, num_conv_layers // 2))
            
            # Find indices of conv2d layers (residual connections only between conv layers)
            conv_indices = [i for i, layer in enumerate(layers) if layer.layer_type == 'conv2d']
            
            for _ in range(num_connections):
                if len(conv_indices) >= 2:
                    # Residual connections typically skip 1-3 layers
                    from_idx = random.choice(conv_indices[:-1])
                    # Find conv layers that are 1-3 positions ahead
                    valid_to_indices = [idx for idx in conv_indices 
                                       if idx > from_idx and idx <= from_idx + 6]
                    if valid_to_indices:
                        to_idx = random.choice(valid_to_indices)
                        connection = (from_idx, to_idx)
                        if connection not in connections:
                            connections.append(connection)
        
        # Global configuration
        global_config = {
            'optimizer': random.choice(['adam', 'sgd', 'rmsprop']),
            'learning_rate': random.choice([0.001, 0.0001, 0.01]),
            'batch_size': random.choice([16, 32, 64, 128]),
        }
        
        arch = Architecture(
            layers=layers,
            connections=connections,
            global_config=global_config
        )
        
        return self._infer_shapes(arch)
    
    def validate_architecture(self, arch: Architecture) -> bool:
        """
        Validate that an architecture is valid for vision tasks.
        
        Args:
            arch: Architecture to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not arch.layers:
            return False
        
        # Check total number of layers
        if len(arch.layers) < self.MIN_TOTAL_LAYERS or len(arch.layers) > self.MAX_TOTAL_LAYERS * 2:
            return False
        
        # Check that we have at least one conv layer and one dense layer
        conv_layers = [l for l in arch.layers if l.layer_type == 'conv2d']
        dense_layers = [l for l in arch.layers if l.layer_type == 'dense']
        
        if not conv_layers or not dense_layers:
            return False
        
        # Check that last layer is dense (output layer)
        if arch.layers[-1].layer_type != 'dense':
            return False
        
        # Check that there's a flatten layer between conv and dense sections
        flatten_layers = [i for i, l in enumerate(arch.layers) if l.layer_type == 'flatten']
        if not flatten_layers:
            return False
        
        # Validate each layer
        for layer in arch.layers:
            if layer.layer_type not in self.LAYER_TYPES:
                return False
            
            if layer.layer_type == 'conv2d':
                if 'filters' not in layer.params or layer.params['filters'] <= 0:
                    return False
                if 'kernel_size' not in layer.params or layer.params['kernel_size'] <= 0:
                    return False
            
            elif layer.layer_type == 'dense':
                if 'units' not in layer.params or layer.params['units'] <= 0:
                    return False
            
            elif layer.layer_type == 'dropout':
                if 'rate' not in layer.params:
                    return False
                rate = layer.params['rate']
                if not (0 <= rate < 1):
                    return False
            
            elif layer.layer_type in ['max_pool2d', 'avg_pool2d']:
                if 'pool_size' not in layer.params or layer.params['pool_size'] <= 0:
                    return False
        
        # Validate connections
        for from_idx, to_idx in arch.connections:
            if from_idx >= to_idx:
                return False
            if from_idx < 0 or to_idx >= len(arch.layers):
                return False
            # Residual connections should be between conv layers
            if arch.layers[from_idx].layer_type != 'conv2d' or arch.layers[to_idx].layer_type != 'conv2d':
                return False
        
        return True
    
    def mutate_architecture(self, arch: Architecture, mutation_rate: float = 0.2) -> Architecture:
        """
        Mutate a CNN architecture by randomly modifying its structure.
        
        Possible mutations:
        - Add a convolutional block
        - Remove a convolutional block
        - Modify layer parameters
        - Add/remove residual connections
        - Modify dense layers
        
        Args:
            arch: Architecture to mutate
            mutation_rate: Probability of each mutation type
        
        Returns:
            Mutated architecture
        """
        new_arch = arch.clone()
        
        # Find flatten layer index (separates conv and dense sections)
        flatten_idx = next((i for i, l in enumerate(new_arch.layers) if l.layer_type == 'flatten'), -1)
        
        if flatten_idx == -1:
            return new_arch  # Invalid architecture, return as-is
        
        # Mutation 1: Add a convolutional layer (with mutation_rate probability)
        if random.random() < mutation_rate and flatten_idx < self.MAX_TOTAL_LAYERS:
            filters = random.choice(self.CONV_FILTERS_RANGE)
            kernel_size = random.choice(self.CONV_KERNEL_SIZES)
            strides = random.choice(self.CONV_STRIDES)
            activation = random.choice(self.ACTIVATION_FUNCTIONS)
            
            new_layer = LayerConfig(
                layer_type='conv2d',
                params={
                    'filters': filters,
                    'kernel_size': kernel_size,
                    'strides': strides,
                    'padding': 'same',
                    'activation': activation,
                    'use_bias': True,
                }
            )
            
            # Insert before flatten layer
            position = random.randint(0, flatten_idx)
            new_arch = self.add_layer(new_arch, new_layer, position)
            flatten_idx += 1  # Update flatten index
        
        # Mutation 2: Remove a convolutional layer (with mutation_rate probability)
        if random.random() < mutation_rate and flatten_idx > 1:
            conv_indices = [i for i in range(flatten_idx) if new_arch.layers[i].layer_type == 'conv2d']
            if len(conv_indices) > 1:  # Keep at least one conv layer
                position = random.choice(conv_indices)
                try:
                    new_arch = self.remove_layer(new_arch, position)
                    flatten_idx -= 1  # Update flatten index
                except ValueError:
                    pass
        
        # Mutation 3: Modify convolutional layer parameters (with mutation_rate probability)
        if random.random() < mutation_rate and flatten_idx > 0:
            conv_indices = [i for i in range(flatten_idx) if new_arch.layers[i].layer_type == 'conv2d']
            if conv_indices:
                position = random.choice(conv_indices)
                new_params = {}
                
                if random.random() < 0.5:
                    new_params['filters'] = random.choice(self.CONV_FILTERS_RANGE)
                if random.random() < 0.5:
                    new_params['kernel_size'] = random.choice(self.CONV_KERNEL_SIZES)
                if random.random() < 0.3:
                    new_params['activation'] = random.choice(self.ACTIVATION_FUNCTIONS)
                
                if new_params:
                    new_arch = self.modify_layer(new_arch, position, new_params)
        
        # Mutation 4: Modify dense layer parameters (with mutation_rate probability)
        if random.random() < mutation_rate:
            # Update flatten_idx in case it changed
            flatten_idx = next((i for i, l in enumerate(new_arch.layers) if l.layer_type == 'flatten'), -1)
            
            if flatten_idx != -1 and flatten_idx < len(new_arch.layers) - 1:
                dense_indices = [i for i in range(flatten_idx + 1, len(new_arch.layers) - 1) 
                               if new_arch.layers[i].layer_type == 'dense']
                if dense_indices:
                    position = random.choice(dense_indices)
                    new_params = {}
                    
                    if random.random() < 0.5:
                        new_params['units'] = random.choice(self.DENSE_UNITS_RANGE)
                    if random.random() < 0.5:
                        new_params['activation'] = random.choice(self.ACTIVATION_FUNCTIONS)
                    
                    if new_params:
                        new_arch = self.modify_layer(new_arch, position, new_params)
        
        # Mutation 5: Add/remove residual connections (with mutation_rate probability)
        if self.enable_residual_connections and random.random() < mutation_rate:
            # Update flatten_idx
            flatten_idx = next((i for i, l in enumerate(new_arch.layers) if l.layer_type == 'flatten'), -1)
            
            if flatten_idx != -1:
                conv_indices = [i for i in range(flatten_idx) if new_arch.layers[i].layer_type == 'conv2d']
                
                if len(conv_indices) >= 2:
                    if random.random() < 0.5 and len(new_arch.connections) < self.MAX_RESIDUAL_CONNECTIONS:
                        # Add connection
                        from_idx = random.choice(conv_indices[:-1])
                        valid_to_indices = [idx for idx in conv_indices 
                                          if idx > from_idx and idx <= from_idx + 6]
                        if valid_to_indices:
                            to_idx = random.choice(valid_to_indices)
                            try:
                                new_arch = self.add_connection(new_arch, from_idx, to_idx)
                            except ValueError:
                                pass
                    elif new_arch.connections:
                        # Remove connection
                        connection = random.choice(new_arch.connections)
                        new_arch = self.remove_connection(new_arch, connection[0], connection[1])
        
        # Mutation 6: Modify global config (with mutation_rate probability)
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                new_arch.global_config['learning_rate'] = random.choice([0.001, 0.0001, 0.01])
            if random.random() < 0.5:
                new_arch.global_config['batch_size'] = random.choice([16, 32, 64, 128])
        
        return new_arch
    
    def get_search_space_size(self) -> int:
        """
        Estimate the size of the search space.
        
        Returns:
            Approximate number of possible architectures
        """
        # Rough estimate based on convolutional and dense layer combinations
        conv_combinations = (len(self.CONV_FILTERS_RANGE) * 
                           len(self.CONV_KERNEL_SIZES) * 
                           len(self.ACTIVATION_FUNCTIONS))
        
        dense_combinations = len(self.DENSE_UNITS_RANGE) * len(self.ACTIVATION_FUNCTIONS)
        
        # Very rough estimate
        return (conv_combinations ** self.MAX_CONV_LAYERS) * (dense_combinations ** self.MAX_DENSE_LAYERS)



class TimeSeriesSearchSpace(SearchSpace):
    """
    Search space for time series and sequential data using RNN architectures.
    
    This search space defines architectures suitable for sequential data,
    consisting of LSTM, GRU, and Conv1D layers, with Dense layers for output.
    """
    
    # Layer type definitions
    LAYER_TYPES = ['lstm', 'gru', 'conv1d', 'dense', 'dropout', 'batch_norm', 'flatten']
    
    # Parameter ranges
    RECURRENT_UNITS_RANGE = [32, 64, 128, 256]
    CONV1D_FILTERS_RANGE = [16, 32, 64, 128]
    CONV1D_KERNEL_SIZES = [3, 5, 7, 9]
    DENSE_UNITS_RANGE = [32, 64, 128, 256]
    ACTIVATION_FUNCTIONS = ['relu', 'tanh', 'elu']
    RECURRENT_ACTIVATIONS = ['tanh', 'sigmoid']
    DROPOUT_RATES = [0.2, 0.3, 0.4, 0.5]
    
    # Architecture constraints
    MIN_RECURRENT_LAYERS = 1
    MAX_RECURRENT_LAYERS = 6
    MAX_CONV1D_LAYERS = 4
    MAX_DENSE_LAYERS = 2
    MIN_TOTAL_LAYERS = 1
    MAX_TOTAL_LAYERS = 12
    
    def __init__(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...],
                 problem_type: str, random_seed: Optional[int] = None,
                 enable_attention: bool = False):
        """
        Initialize TimeSeriesSearchSpace.
        
        Args:
            input_shape: Shape of input sequences (e.g., (timesteps, features))
            output_shape: Shape of output data (e.g., (num_classes,) or (forecast_horizon,))
            problem_type: 'classification', 'regression', or 'forecasting'
            random_seed: Random seed for reproducibility
            enable_attention: Whether to allow attention mechanisms (future feature)
        """
        super().__init__(input_shape, output_shape, problem_type, random_seed)
        self.enable_attention = enable_attention
        
        # Validate input shape for sequences
        if len(input_shape) not in [1, 2]:
            raise ValueError(f"Input shape for time series must be 1D or 2D, got {input_shape}")
    
    def sample_architecture(self) -> Architecture:
        """
        Sample a random RNN/CNN architecture for time series.
        
        Returns:
            A randomly sampled Architecture with LSTM/GRU/Conv1D and Dense layers
        """
        layers = []
        
        # Decide on architecture type: pure RNN, pure CNN, or hybrid
        arch_type = random.choice(['rnn', 'cnn', 'hybrid'])
        
        if arch_type in ['rnn', 'hybrid']:
            # Recurrent layers section
            num_recurrent_layers = random.randint(self.MIN_RECURRENT_LAYERS, self.MAX_RECURRENT_LAYERS)
            
            for i in range(num_recurrent_layers):
                # Choose between LSTM and GRU
                layer_type = random.choice(['lstm', 'gru'])
                units = random.choice(self.RECURRENT_UNITS_RANGE)
                
                # Only the last recurrent layer should not return sequences (unless followed by more recurrent layers)
                return_sequences = (i < num_recurrent_layers - 1) or arch_type == 'hybrid'
                
                recurrent_layer = LayerConfig(
                    layer_type=layer_type,
                    params={
                        'units': units,
                        'return_sequences': return_sequences,
                        'activation': random.choice(self.RECURRENT_ACTIVATIONS),
                        'recurrent_activation': random.choice(self.RECURRENT_ACTIVATIONS),
                    }
                )
                layers.append(recurrent_layer)
                
                # Randomly add Dropout (50% chance)
                if random.random() < 0.5:
                    dropout_rate = random.choice(self.DROPOUT_RATES)
                    dropout_layer = LayerConfig(
                        layer_type='dropout',
                        params={'rate': dropout_rate}
                    )
                    layers.append(dropout_layer)
        
        if arch_type in ['cnn', 'hybrid']:
            # Conv1D layers section
            num_conv_layers = random.randint(1, self.MAX_CONV1D_LAYERS)
            
            for i in range(num_conv_layers):
                filters = random.choice(self.CONV1D_FILTERS_RANGE)
                kernel_size = random.choice(self.CONV1D_KERNEL_SIZES)
                activation = random.choice(self.ACTIVATION_FUNCTIONS)
                
                conv_layer = LayerConfig(
                    layer_type='conv1d',
                    params={
                        'filters': filters,
                        'kernel_size': kernel_size,
                        'padding': 'same',
                        'activation': activation,
                    }
                )
                layers.append(conv_layer)
                
                # Randomly add BatchNormalization (40% chance)
                if random.random() < 0.4:
                    batch_norm_layer = LayerConfig(
                        layer_type='batch_norm',
                        params={}
                    )
                    layers.append(batch_norm_layer)
            
            # Add flatten layer if we have conv layers and need to go to dense
            if arch_type == 'cnn' or (arch_type == 'hybrid' and random.random() < 0.5):
                flatten_layer = LayerConfig(
                    layer_type='flatten',
                    params={}
                )
                layers.append(flatten_layer)
        
        # Dense layers section (optional, for classification/regression)
        if self.problem_type in ['classification', 'regression'] or random.random() < 0.5:
            num_dense_layers = random.randint(0, self.MAX_DENSE_LAYERS)
            
            for i in range(num_dense_layers):
                units = random.choice(self.DENSE_UNITS_RANGE)
                activation = random.choice(self.ACTIVATION_FUNCTIONS)
                
                dense_layer = LayerConfig(
                    layer_type='dense',
                    params={
                        'units': units,
                        'activation': activation,
                        'use_bias': True,
                    }
                )
                layers.append(dense_layer)
                
                # Randomly add Dropout (50% chance)
                if random.random() < 0.5:
                    dropout_rate = random.choice(self.DROPOUT_RATES)
                    dropout_layer = LayerConfig(
                        layer_type='dropout',
                        params={'rate': dropout_rate}
                    )
                    layers.append(dropout_layer)
        
        # Add output layer
        output_units = self.output_shape[0] if len(self.output_shape) > 0 else 1
        
        if self.problem_type == 'classification':
            if output_units > 2:
                output_activation = 'softmax'
            else:
                output_activation = 'sigmoid'
                output_units = 1
        else:  # regression or forecasting
            output_activation = 'linear'
        
        output_layer = LayerConfig(
            layer_type='dense',
            params={
                'units': output_units,
                'activation': output_activation,
                'use_bias': True,
            }
        )
        layers.append(output_layer)
        
        # No skip connections for time series (they're less common in RNNs)
        connections = []
        
        # Global configuration
        global_config = {
            'optimizer': random.choice(['adam', 'rmsprop']),
            'learning_rate': random.choice([0.001, 0.0001, 0.01]),
            'batch_size': random.choice([32, 64, 128]),
        }
        
        arch = Architecture(
            layers=layers,
            connections=connections,
            global_config=global_config
        )
        
        return self._infer_shapes(arch)
    
    def validate_architecture(self, arch: Architecture) -> bool:
        """
        Validate that an architecture is valid for time series tasks.
        
        Args:
            arch: Architecture to validate
        
        Returns:
            True if valid, False otherwise
        """
        if not arch.layers:
            return False
        
        # Check total number of layers
        if len(arch.layers) < self.MIN_TOTAL_LAYERS or len(arch.layers) > self.MAX_TOTAL_LAYERS * 2:
            return False
        
        # Check that last layer is dense (output layer)
        if arch.layers[-1].layer_type != 'dense':
            return False
        
        # Check that we have at least one recurrent or conv1d layer
        recurrent_layers = [l for l in arch.layers if l.layer_type in ['lstm', 'gru']]
        conv_layers = [l for l in arch.layers if l.layer_type == 'conv1d']
        
        if not recurrent_layers and not conv_layers:
            return False
        
        # Validate each layer
        for i, layer in enumerate(arch.layers):
            if layer.layer_type not in self.LAYER_TYPES:
                return False
            
            if layer.layer_type in ['lstm', 'gru']:
                if 'units' not in layer.params or layer.params['units'] <= 0:
                    return False
                if 'return_sequences' not in layer.params:
                    return False
                
                # If not the last recurrent layer, should return sequences
                is_last_recurrent = all(
                    arch.layers[j].layer_type not in ['lstm', 'gru'] 
                    for j in range(i + 1, len(arch.layers))
                )
                # This is a soft check - we allow flexibility
            
            elif layer.layer_type == 'conv1d':
                if 'filters' not in layer.params or layer.params['filters'] <= 0:
                    return False
                if 'kernel_size' not in layer.params or layer.params['kernel_size'] <= 0:
                    return False
            
            elif layer.layer_type == 'dense':
                if 'units' not in layer.params or layer.params['units'] <= 0:
                    return False
            
            elif layer.layer_type == 'dropout':
                if 'rate' not in layer.params:
                    return False
                rate = layer.params['rate']
                if not (0 <= rate < 1):
                    return False
        
        # Validate connections (should be empty for time series)
        # We allow connections but don't encourage them
        for from_idx, to_idx in arch.connections:
            if from_idx >= to_idx:
                return False
            if from_idx < 0 or to_idx >= len(arch.layers):
                return False
        
        return True
    
    def mutate_architecture(self, arch: Architecture, mutation_rate: float = 0.2) -> Architecture:
        """
        Mutate a time series architecture by randomly modifying its structure.
        
        Possible mutations:
        - Add a recurrent/conv layer
        - Remove a layer
        - Modify layer parameters
        - Change layer type (LSTM <-> GRU)
        
        Args:
            arch: Architecture to mutate
            mutation_rate: Probability of each mutation type
        
        Returns:
            Mutated architecture
        """
        new_arch = arch.clone()
        
        # Mutation 1: Add a layer (with mutation_rate probability)
        if random.random() < mutation_rate and len(new_arch.layers) < self.MAX_TOTAL_LAYERS:
            # Choose layer type to add
            layer_type = random.choice(['lstm', 'gru', 'conv1d', 'dense', 'dropout'])
            
            if layer_type in ['lstm', 'gru']:
                new_layer = LayerConfig(
                    layer_type=layer_type,
                    params={
                        'units': random.choice(self.RECURRENT_UNITS_RANGE),
                        'return_sequences': True,  # Will be adjusted if needed
                        'activation': random.choice(self.RECURRENT_ACTIVATIONS),
                        'recurrent_activation': random.choice(self.RECURRENT_ACTIVATIONS),
                    }
                )
            elif layer_type == 'conv1d':
                new_layer = LayerConfig(
                    layer_type='conv1d',
                    params={
                        'filters': random.choice(self.CONV1D_FILTERS_RANGE),
                        'kernel_size': random.choice(self.CONV1D_KERNEL_SIZES),
                        'padding': 'same',
                        'activation': random.choice(self.ACTIVATION_FUNCTIONS),
                    }
                )
            elif layer_type == 'dense':
                new_layer = LayerConfig(
                    layer_type='dense',
                    params={
                        'units': random.choice(self.DENSE_UNITS_RANGE),
                        'activation': random.choice(self.ACTIVATION_FUNCTIONS),
                        'use_bias': True,
                    }
                )
            else:  # dropout
                new_layer = LayerConfig(
                    layer_type='dropout',
                    params={'rate': random.choice(self.DROPOUT_RATES)}
                )
            
            # Insert before the output layer
            position = len(new_arch.layers) - 1
            new_arch = self.add_layer(new_arch, new_layer, position)
        
        # Mutation 2: Remove a layer (with mutation_rate probability)
        if random.random() < mutation_rate and len(new_arch.layers) > 2:
            # Don't remove the output layer
            position = random.randint(0, len(new_arch.layers) - 2)
            try:
                new_arch = self.remove_layer(new_arch, position)
            except ValueError:
                pass
        
        # Mutation 3: Modify layer parameters (with mutation_rate probability)
        if random.random() < mutation_rate and len(new_arch.layers) > 1:
            # Choose a random layer to modify (excluding output layer)
            position = random.randint(0, len(new_arch.layers) - 2)
            layer = new_arch.layers[position]
            
            if layer.layer_type in ['lstm', 'gru']:
                new_params = {}
                if random.random() < 0.5:
                    new_params['units'] = random.choice(self.RECURRENT_UNITS_RANGE)
                if random.random() < 0.3:
                    new_params['activation'] = random.choice(self.RECURRENT_ACTIVATIONS)
                
                if new_params:
                    new_arch = self.modify_layer(new_arch, position, new_params)
            
            elif layer.layer_type == 'conv1d':
                new_params = {}
                if random.random() < 0.5:
                    new_params['filters'] = random.choice(self.CONV1D_FILTERS_RANGE)
                if random.random() < 0.5:
                    new_params['kernel_size'] = random.choice(self.CONV1D_KERNEL_SIZES)
                
                if new_params:
                    new_arch = self.modify_layer(new_arch, position, new_params)
            
            elif layer.layer_type == 'dense':
                new_params = {}
                if random.random() < 0.5:
                    new_params['units'] = random.choice(self.DENSE_UNITS_RANGE)
                if random.random() < 0.5:
                    new_params['activation'] = random.choice(self.ACTIVATION_FUNCTIONS)
                
                if new_params:
                    new_arch = self.modify_layer(new_arch, position, new_params)
            
            elif layer.layer_type == 'dropout':
                new_params = {'rate': random.choice(self.DROPOUT_RATES)}
                new_arch = self.modify_layer(new_arch, position, new_params)
        
        # Mutation 4: Change layer type (LSTM <-> GRU) (with mutation_rate probability)
        if random.random() < mutation_rate:
            recurrent_indices = [i for i, l in enumerate(new_arch.layers[:-1]) 
                               if l.layer_type in ['lstm', 'gru']]
            if recurrent_indices:
                position = random.choice(recurrent_indices)
                current_type = new_arch.layers[position].layer_type
                new_type = 'gru' if current_type == 'lstm' else 'lstm'
                
                # Create new layer with same parameters but different type
                new_layer = LayerConfig(
                    layer_type=new_type,
                    params=copy.deepcopy(new_arch.layers[position].params)
                )
                new_arch.layers[position] = new_layer
        
        # Mutation 5: Modify global config (with mutation_rate probability)
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                new_arch.global_config['learning_rate'] = random.choice([0.001, 0.0001, 0.01])
            if random.random() < 0.5:
                new_arch.global_config['batch_size'] = random.choice([32, 64, 128])
        
        return new_arch
    
    def get_search_space_size(self) -> int:
        """
        Estimate the size of the search space.
        
        Returns:
            Approximate number of possible architectures
        """
        # Rough estimate based on recurrent and conv layer combinations
        recurrent_combinations = (len(self.RECURRENT_UNITS_RANGE) * 
                                 len(self.RECURRENT_ACTIVATIONS) * 2)  # LSTM or GRU
        
        conv_combinations = (len(self.CONV1D_FILTERS_RANGE) * 
                           len(self.CONV1D_KERNEL_SIZES) * 
                           len(self.ACTIVATION_FUNCTIONS))
        
        # Very rough estimate
        return (recurrent_combinations ** self.MAX_RECURRENT_LAYERS) * (conv_combinations ** self.MAX_CONV1D_LAYERS)

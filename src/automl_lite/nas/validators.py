"""
Architecture validation utilities for Neural Architecture Search.

This module provides functions for validating architecture configurations,
checking layer compatibility, and inferring shapes through the network.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from .architecture import Architecture, LayerConfig


class ArchitectureValidator:
    """
    Validates neural network architectures for correctness and compatibility.
    
    Performs checks including:
    - Layer compatibility (valid layer types and parameters)
    - Shape inference through the network
    - Connection validity (skip connections)
    - Hardware constraint satisfaction
    """
    
    # Supported layer types and their required/optional parameters
    LAYER_SPECS = {
        # Dense/Fully Connected layers
        'dense': {
            'required': ['units'],
            'optional': ['activation', 'use_bias', 'kernel_initializer', 'bias_initializer'],
            'valid_activations': ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'softmax', 'linear', None],
        },
        'dropout': {
            'required': ['rate'],
            'optional': ['seed'],
        },
        'batch_normalization': {
            'required': [],
            'optional': ['momentum', 'epsilon', 'center', 'scale'],
        },
        
        # Convolutional layers
        'conv2d': {
            'required': ['filters', 'kernel_size'],
            'optional': ['strides', 'padding', 'activation', 'use_bias'],
            'valid_activations': ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'linear', None],
        },
        'max_pooling2d': {
            'required': ['pool_size'],
            'optional': ['strides', 'padding'],
        },
        'average_pooling2d': {
            'required': ['pool_size'],
            'optional': ['strides', 'padding'],
        },
        'global_average_pooling2d': {
            'required': [],
            'optional': [],
        },
        'flatten': {
            'required': [],
            'optional': [],
        },
        
        # Recurrent layers
        'lstm': {
            'required': ['units'],
            'optional': ['activation', 'recurrent_activation', 'return_sequences', 'dropout', 'recurrent_dropout'],
            'valid_activations': ['tanh', 'sigmoid', 'relu', 'elu'],
        },
        'gru': {
            'required': ['units'],
            'optional': ['activation', 'recurrent_activation', 'return_sequences', 'dropout', 'recurrent_dropout'],
            'valid_activations': ['tanh', 'sigmoid', 'relu', 'elu'],
        },
        
        # 1D Convolutional layers (for time series)
        'conv1d': {
            'required': ['filters', 'kernel_size'],
            'optional': ['strides', 'padding', 'activation', 'use_bias'],
            'valid_activations': ['relu', 'tanh', 'sigmoid', 'elu', 'selu', 'linear', None],
        },
        'max_pooling1d': {
            'required': ['pool_size'],
            'optional': ['strides', 'padding'],
        },
        'global_average_pooling1d': {
            'required': [],
            'optional': [],
        },
    }
    
    def __init__(self):
        """Initialize the architecture validator."""
        pass
    
    def validate_architecture(self, architecture: Architecture, 
                            input_shape: Optional[Tuple[int, ...]] = None) -> Tuple[bool, List[str]]:
        """
        Validate an architecture for correctness.
        
        Args:
            architecture: Architecture to validate
            input_shape: Expected input shape (excluding batch dimension)
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check that architecture has at least one layer
        if not architecture.layers:
            errors.append("Architecture must have at least one layer")
            return False, errors
        
        # Validate each layer
        for i, layer in enumerate(architecture.layers):
            layer_errors = self._validate_layer(layer, i)
            errors.extend(layer_errors)
        
        # Validate connections
        connection_errors = self._validate_connections(architecture)
        errors.extend(connection_errors)
        
        # Infer shapes if input shape is provided
        if input_shape is not None:
            shape_errors = self._infer_and_validate_shapes(architecture, input_shape)
            errors.extend(shape_errors)
        
        return len(errors) == 0, errors
    
    def _validate_layer(self, layer: LayerConfig, layer_idx: int) -> List[str]:
        """
        Validate a single layer configuration.
        
        Args:
            layer: Layer configuration to validate
            layer_idx: Index of the layer in the architecture
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Check if layer type is supported
        if layer.layer_type not in self.LAYER_SPECS:
            errors.append(f"Layer {layer_idx}: Unsupported layer type '{layer.layer_type}'")
            return errors
        
        spec = self.LAYER_SPECS[layer.layer_type]
        
        # Check required parameters
        for param in spec['required']:
            if param not in layer.params:
                errors.append(f"Layer {layer_idx} ({layer.layer_type}): Missing required parameter '{param}'")
        
        # Check for unknown parameters
        all_valid_params = set(spec['required'] + spec['optional'])
        for param in layer.params:
            if param not in all_valid_params:
                errors.append(f"Layer {layer_idx} ({layer.layer_type}): Unknown parameter '{param}'")
        
        # Validate activation functions if applicable
        if 'valid_activations' in spec and 'activation' in layer.params:
            activation = layer.params['activation']
            if activation not in spec['valid_activations']:
                errors.append(
                    f"Layer {layer_idx} ({layer.layer_type}): Invalid activation '{activation}'. "
                    f"Must be one of {spec['valid_activations']}"
                )
        
        # Validate parameter values
        param_errors = self._validate_layer_params(layer, layer_idx)
        errors.extend(param_errors)
        
        return errors
    
    def _validate_layer_params(self, layer: LayerConfig, layer_idx: int) -> List[str]:
        """
        Validate parameter values for a layer.
        
        Args:
            layer: Layer configuration
            layer_idx: Index of the layer
        
        Returns:
            List of error messages
        """
        errors = []
        params = layer.params
        
        # Validate units/filters (must be positive integers)
        if 'units' in params:
            if not isinstance(params['units'], int) or params['units'] <= 0:
                errors.append(f"Layer {layer_idx}: 'units' must be a positive integer, got {params['units']}")
        
        if 'filters' in params:
            if not isinstance(params['filters'], int) or params['filters'] <= 0:
                errors.append(f"Layer {layer_idx}: 'filters' must be a positive integer, got {params['filters']}")
        
        # Validate dropout rate (must be in [0, 1))
        if 'rate' in params:
            if not 0 <= params['rate'] < 1:
                errors.append(f"Layer {layer_idx}: 'rate' must be in [0, 1), got {params['rate']}")
        
        if 'dropout' in params:
            if not 0 <= params['dropout'] <= 1:
                errors.append(f"Layer {layer_idx}: 'dropout' must be in [0, 1], got {params['dropout']}")
        
        # Validate kernel_size (must be positive)
        if 'kernel_size' in params:
            kernel_size = params['kernel_size']
            if isinstance(kernel_size, int):
                if kernel_size <= 0:
                    errors.append(f"Layer {layer_idx}: 'kernel_size' must be positive, got {kernel_size}")
            elif isinstance(kernel_size, (list, tuple)):
                if any(k <= 0 for k in kernel_size):
                    errors.append(f"Layer {layer_idx}: All 'kernel_size' values must be positive, got {kernel_size}")
        
        # Validate pool_size (must be positive)
        if 'pool_size' in params:
            pool_size = params['pool_size']
            if isinstance(pool_size, int):
                if pool_size <= 0:
                    errors.append(f"Layer {layer_idx}: 'pool_size' must be positive, got {pool_size}")
            elif isinstance(pool_size, (list, tuple)):
                if any(p <= 0 for p in pool_size):
                    errors.append(f"Layer {layer_idx}: All 'pool_size' values must be positive, got {pool_size}")
        
        # Validate strides (must be positive)
        if 'strides' in params:
            strides = params['strides']
            if isinstance(strides, int):
                if strides <= 0:
                    errors.append(f"Layer {layer_idx}: 'strides' must be positive, got {strides}")
            elif isinstance(strides, (list, tuple)):
                if any(s <= 0 for s in strides):
                    errors.append(f"Layer {layer_idx}: All 'strides' values must be positive, got {strides}")
        
        return errors
    
    def _validate_connections(self, architecture: Architecture) -> List[str]:
        """
        Validate skip connections in the architecture.
        
        Args:
            architecture: Architecture to validate
        
        Returns:
            List of error messages
        """
        errors = []
        max_idx = len(architecture.layers) - 1
        
        for from_idx, to_idx in architecture.connections:
            # Check indices are in valid range
            if from_idx < 0 or from_idx > max_idx:
                errors.append(f"Connection: from_idx {from_idx} out of range [0, {max_idx}]")
            
            if to_idx < 0 or to_idx > max_idx:
                errors.append(f"Connection: to_idx {to_idx} out of range [0, {max_idx}]")
            
            # Check that connection goes forward
            if from_idx >= to_idx:
                errors.append(f"Connection: from_idx {from_idx} must be < to_idx {to_idx} (no backward connections)")
        
        # Check for duplicate connections
        if len(architecture.connections) != len(set(architecture.connections)):
            errors.append("Architecture contains duplicate connections")
        
        return errors
    
    def _infer_and_validate_shapes(self, architecture: Architecture, 
                                   input_shape: Tuple[int, ...]) -> List[str]:
        """
        Infer output shapes for all layers and validate compatibility.
        
        Args:
            architecture: Architecture to validate
            input_shape: Input shape (excluding batch dimension)
        
        Returns:
            List of error messages
        """
        errors = []
        current_shape = input_shape
        
        for i, layer in enumerate(architecture.layers):
            try:
                output_shape = self._infer_layer_output_shape(layer, current_shape)
                
                # Update layer shapes
                layer.input_shape = current_shape
                layer.output_shape = output_shape
                
                current_shape = output_shape
            except Exception as e:
                errors.append(f"Layer {i} ({layer.layer_type}): Shape inference failed - {str(e)}")
                break
        
        return errors
    
    def _infer_layer_output_shape(self, layer: LayerConfig, 
                                  input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Infer the output shape of a layer given its input shape.
        
        Args:
            layer: Layer configuration
            input_shape: Input shape (excluding batch dimension)
        
        Returns:
            Output shape (excluding batch dimension)
        
        Raises:
            ValueError: If shape cannot be inferred
        """
        layer_type = layer.layer_type
        params = layer.params
        
        if layer_type == 'dense':
            return (params['units'],)
        
        elif layer_type == 'dropout':
            return input_shape
        
        elif layer_type == 'batch_normalization':
            return input_shape
        
        elif layer_type == 'flatten':
            return (int(np.prod(input_shape)),)
        
        elif layer_type == 'conv2d':
            if len(input_shape) != 3:
                raise ValueError(f"conv2d expects 3D input (H, W, C), got shape {input_shape}")
            
            h, w, c = input_shape
            filters = params['filters']
            kernel_size = params.get('kernel_size', 3)
            strides = params.get('strides', 1)
            padding = params.get('padding', 'valid')
            
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            if isinstance(strides, int):
                strides = (strides, strides)
            
            if padding == 'same':
                out_h = int(np.ceil(h / strides[0]))
                out_w = int(np.ceil(w / strides[1]))
            else:  # 'valid'
                out_h = int(np.ceil((h - kernel_size[0] + 1) / strides[0]))
                out_w = int(np.ceil((w - kernel_size[1] + 1) / strides[1]))
            
            return (out_h, out_w, filters)
        
        elif layer_type in ['max_pooling2d', 'average_pooling2d']:
            if len(input_shape) != 3:
                raise ValueError(f"{layer_type} expects 3D input (H, W, C), got shape {input_shape}")
            
            h, w, c = input_shape
            pool_size = params.get('pool_size', 2)
            strides = params.get('strides', pool_size)
            padding = params.get('padding', 'valid')
            
            if isinstance(pool_size, int):
                pool_size = (pool_size, pool_size)
            if isinstance(strides, int):
                strides = (strides, strides)
            
            if padding == 'same':
                out_h = int(np.ceil(h / strides[0]))
                out_w = int(np.ceil(w / strides[1]))
            else:  # 'valid'
                out_h = int(np.ceil((h - pool_size[0] + 1) / strides[0]))
                out_w = int(np.ceil((w - pool_size[1] + 1) / strides[1]))
            
            return (out_h, out_w, c)
        
        elif layer_type == 'global_average_pooling2d':
            if len(input_shape) != 3:
                raise ValueError(f"global_average_pooling2d expects 3D input (H, W, C), got shape {input_shape}")
            return (input_shape[2],)
        
        elif layer_type in ['lstm', 'gru']:
            units = params['units']
            return_sequences = params.get('return_sequences', False)
            
            if len(input_shape) == 2:  # (timesteps, features)
                if return_sequences:
                    return (input_shape[0], units)
                else:
                    return (units,)
            else:
                raise ValueError(f"{layer_type} expects 2D input (timesteps, features), got shape {input_shape}")
        
        elif layer_type == 'conv1d':
            if len(input_shape) != 2:
                raise ValueError(f"conv1d expects 2D input (timesteps, features), got shape {input_shape}")
            
            timesteps, features = input_shape
            filters = params['filters']
            kernel_size = params.get('kernel_size', 3)
            strides = params.get('strides', 1)
            padding = params.get('padding', 'valid')
            
            if padding == 'same':
                out_timesteps = int(np.ceil(timesteps / strides))
            else:  # 'valid'
                out_timesteps = int(np.ceil((timesteps - kernel_size + 1) / strides))
            
            return (out_timesteps, filters)
        
        elif layer_type in ['max_pooling1d']:
            if len(input_shape) != 2:
                raise ValueError(f"{layer_type} expects 2D input (timesteps, features), got shape {input_shape}")
            
            timesteps, features = input_shape
            pool_size = params.get('pool_size', 2)
            strides = params.get('strides', pool_size)
            padding = params.get('padding', 'valid')
            
            if padding == 'same':
                out_timesteps = int(np.ceil(timesteps / strides))
            else:  # 'valid'
                out_timesteps = int(np.ceil((timesteps - pool_size + 1) / strides))
            
            return (out_timesteps, features)
        
        elif layer_type == 'global_average_pooling1d':
            if len(input_shape) != 2:
                raise ValueError(f"global_average_pooling1d expects 2D input (timesteps, features), got shape {input_shape}")
            return (input_shape[1],)
        
        else:
            raise ValueError(f"Shape inference not implemented for layer type '{layer_type}'")
    
    def check_layer_compatibility(self, layer1: LayerConfig, layer2: LayerConfig) -> Tuple[bool, str]:
        """
        Check if two layers can be connected sequentially.
        
        Args:
            layer1: First layer (output)
            layer2: Second layer (input)
        
        Returns:
            Tuple of (is_compatible, error_message)
        """
        # If shapes are known, check compatibility
        if layer1.output_shape is not None and layer2.input_shape is not None:
            if layer1.output_shape != layer2.input_shape:
                return False, f"Shape mismatch: {layer1.output_shape} -> {layer2.input_shape}"
        
        # Check specific incompatibilities
        # For example, can't have LSTM after Conv2D without flattening
        if layer1.layer_type in ['conv2d', 'max_pooling2d', 'average_pooling2d']:
            if layer2.layer_type in ['lstm', 'gru', 'conv1d']:
                return False, f"Cannot connect {layer1.layer_type} to {layer2.layer_type} without flattening or reshaping"
        
        return True, ""

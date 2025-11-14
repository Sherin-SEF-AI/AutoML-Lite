"""
Utility functions for Neural Architecture Search.

This module provides utility functions for architecture comparison, complexity
metrics calculation, and search space size estimation.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from .architecture import Architecture, LayerConfig, NASConfig


def compare_architectures(arch1: Architecture, arch2: Architecture) -> Dict[str, Any]:
    """
    Compare two architectures and return differences.
    
    Args:
        arch1: First architecture
        arch2: Second architecture
    
    Returns:
        Dictionary containing comparison results with keys:
        - 'num_layers_diff': Difference in number of layers
        - 'layer_type_diff': List of layer type differences
        - 'connections_diff': Difference in skip connections
        - 'similarity_score': Overall similarity score (0-1)
    """
    comparison = {
        'num_layers_diff': len(arch1.layers) - len(arch2.layers),
        'layer_type_diff': [],
        'connections_diff': len(arch1.connections) - len(arch2.connections),
        'similarity_score': 0.0,
    }
    
    # Compare layer types
    min_layers = min(len(arch1.layers), len(arch2.layers))
    matching_layers = 0
    
    for i in range(min_layers):
        layer1 = arch1.layers[i]
        layer2 = arch2.layers[i]
        
        if layer1.layer_type != layer2.layer_type:
            comparison['layer_type_diff'].append({
                'position': i,
                'arch1_type': layer1.layer_type,
                'arch2_type': layer2.layer_type,
            })
        else:
            matching_layers += 1
    
    # Calculate similarity score
    max_layers = max(len(arch1.layers), len(arch2.layers))
    if max_layers > 0:
        layer_similarity = matching_layers / max_layers
        
        # Consider connections similarity
        max_connections = max(len(arch1.connections), len(arch2.connections))
        if max_connections > 0:
            common_connections = len(set(arch1.connections) & set(arch2.connections))
            connection_similarity = common_connections / max_connections
        else:
            connection_similarity = 1.0 if len(arch1.connections) == len(arch2.connections) else 0.0
        
        # Weighted average (layers are more important)
        comparison['similarity_score'] = 0.8 * layer_similarity + 0.2 * connection_similarity
    
    return comparison


def architecture_diff(arch1: Architecture, arch2: Architecture) -> str:
    """
    Generate a human-readable diff between two architectures.
    
    Args:
        arch1: First architecture
        arch2: Second architecture
    
    Returns:
        String representation of the differences
    """
    comparison = compare_architectures(arch1, arch2)
    
    lines = []
    lines.append(f"Architecture Comparison (Similarity: {comparison['similarity_score']:.2%})")
    lines.append("=" * 60)
    
    # Layer count difference
    if comparison['num_layers_diff'] != 0:
        lines.append(f"Layer count: {len(arch1.layers)} vs {len(arch2.layers)} "
                    f"(diff: {comparison['num_layers_diff']:+d})")
    else:
        lines.append(f"Layer count: {len(arch1.layers)} (same)")
    
    # Connection difference
    if comparison['connections_diff'] != 0:
        lines.append(f"Skip connections: {len(arch1.connections)} vs {len(arch2.connections)} "
                    f"(diff: {comparison['connections_diff']:+d})")
    else:
        lines.append(f"Skip connections: {len(arch1.connections)} (same)")
    
    # Layer type differences
    if comparison['layer_type_diff']:
        lines.append("\nLayer type differences:")
        for diff in comparison['layer_type_diff']:
            lines.append(f"  Position {diff['position']}: "
                        f"{diff['arch1_type']} -> {diff['arch2_type']}")
    else:
        lines.append("\nNo layer type differences")
    
    return "\n".join(lines)


def calculate_flops(architecture: Architecture, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the number of FLOPs (Floating Point Operations) for an architecture.
    
    Args:
        architecture: Architecture to analyze
        input_shape: Input shape (batch_size, features) or (batch_size, height, width, channels)
    
    Returns:
        Total number of FLOPs
    """
    total_flops = 0
    current_shape = input_shape
    
    for layer in architecture.layers:
        layer_type = layer.layer_type.lower()
        params = layer.params
        
        if layer_type == 'dense':
            # Dense layer: input_size * output_size * 2 (multiply-add)
            units = params.get('units', 0)
            if len(current_shape) >= 2:
                input_size = current_shape[-1]
                layer_flops = input_size * units * 2
                total_flops += layer_flops
                current_shape = current_shape[:-1] + (units,)
        
        elif layer_type == 'conv2d':
            # Conv2D: output_h * output_w * kernel_h * kernel_w * in_channels * out_channels * 2
            filters = params.get('filters', 0)
            kernel_size = params.get('kernel_size', 3)
            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size
            
            if len(current_shape) == 4:  # (batch, height, width, channels)
                _, h, w, in_channels = current_shape
                strides = params.get('strides', 1)
                if isinstance(strides, int):
                    stride_h = stride_w = strides
                else:
                    stride_h, stride_w = strides
                
                out_h = (h - kernel_h) // stride_h + 1
                out_w = (w - kernel_w) // stride_w + 1
                
                layer_flops = out_h * out_w * kernel_h * kernel_w * in_channels * filters * 2
                total_flops += layer_flops
                current_shape = (current_shape[0], out_h, out_w, filters)
        
        elif layer_type == 'lstm':
            # LSTM: 4 * (input_size + hidden_size) * hidden_size * seq_length
            units = params.get('units', 0)
            if len(current_shape) >= 3:  # (batch, seq_length, features)
                seq_length = current_shape[1]
                input_size = current_shape[2]
                layer_flops = 4 * (input_size + units) * units * seq_length
                total_flops += layer_flops
                current_shape = (current_shape[0], seq_length, units)
        
        elif layer_type == 'gru':
            # GRU: 3 * (input_size + hidden_size) * hidden_size * seq_length
            units = params.get('units', 0)
            if len(current_shape) >= 3:  # (batch, seq_length, features)
                seq_length = current_shape[1]
                input_size = current_shape[2]
                layer_flops = 3 * (input_size + units) * units * seq_length
                total_flops += layer_flops
                current_shape = (current_shape[0], seq_length, units)
        
        # Dropout, BatchNorm, Pooling layers have negligible FLOPs
    
    return total_flops


def calculate_parameters(architecture: Architecture, input_shape: Tuple[int, ...]) -> int:
    """
    Calculate the total number of trainable parameters in an architecture.
    
    Args:
        architecture: Architecture to analyze
        input_shape: Input shape (batch_size, features) or (batch_size, height, width, channels)
    
    Returns:
        Total number of parameters
    """
    total_params = 0
    current_shape = input_shape
    
    for layer in architecture.layers:
        layer_type = layer.layer_type.lower()
        params = layer.params
        
        if layer_type == 'dense':
            # Dense layer: (input_size + 1) * output_size (weights + bias)
            units = params.get('units', 0)
            use_bias = params.get('use_bias', True)
            if len(current_shape) >= 2:
                input_size = current_shape[-1]
                layer_params = input_size * units
                if use_bias:
                    layer_params += units
                total_params += layer_params
                current_shape = current_shape[:-1] + (units,)
        
        elif layer_type == 'conv2d':
            # Conv2D: kernel_h * kernel_w * in_channels * out_channels + out_channels (bias)
            filters = params.get('filters', 0)
            kernel_size = params.get('kernel_size', 3)
            use_bias = params.get('use_bias', True)
            if isinstance(kernel_size, int):
                kernel_h = kernel_w = kernel_size
            else:
                kernel_h, kernel_w = kernel_size
            
            if len(current_shape) == 4:  # (batch, height, width, channels)
                _, h, w, in_channels = current_shape
                layer_params = kernel_h * kernel_w * in_channels * filters
                if use_bias:
                    layer_params += filters
                total_params += layer_params
                
                strides = params.get('strides', 1)
                if isinstance(strides, int):
                    stride_h = stride_w = strides
                else:
                    stride_h, stride_w = strides
                out_h = (h - kernel_h) // stride_h + 1
                out_w = (w - kernel_w) // stride_w + 1
                current_shape = (current_shape[0], out_h, out_w, filters)
        
        elif layer_type == 'lstm':
            # LSTM: 4 * ((input_size + hidden_size) * hidden_size + hidden_size)
            units = params.get('units', 0)
            if len(current_shape) >= 3:  # (batch, seq_length, features)
                input_size = current_shape[2]
                layer_params = 4 * ((input_size + units) * units + units)
                total_params += layer_params
                current_shape = (current_shape[0], current_shape[1], units)
        
        elif layer_type == 'gru':
            # GRU: 3 * ((input_size + hidden_size) * hidden_size + hidden_size)
            units = params.get('units', 0)
            if len(current_shape) >= 3:  # (batch, seq_length, features)
                input_size = current_shape[2]
                layer_params = 3 * ((input_size + units) * units + units)
                total_params += layer_params
                current_shape = (current_shape[0], current_shape[1], units)
        
        elif layer_type == 'batchnormalization':
            # BatchNorm: 2 * num_features (gamma and beta)
            if len(current_shape) >= 2:
                num_features = current_shape[-1]
                total_params += 2 * num_features
    
    return total_params


def get_architecture_complexity_metrics(
    architecture: Architecture,
    input_shape: Tuple[int, ...]
) -> Dict[str, Any]:
    """
    Calculate comprehensive complexity metrics for an architecture.
    
    Args:
        architecture: Architecture to analyze
        input_shape: Input shape
    
    Returns:
        Dictionary containing:
        - 'num_layers': Number of layers
        - 'num_parameters': Total trainable parameters
        - 'flops': Total FLOPs
        - 'model_size_mb': Estimated model size in MB (assuming float32)
        - 'num_skip_connections': Number of skip connections
    """
    num_params = calculate_parameters(architecture, input_shape)
    flops = calculate_flops(architecture, input_shape)
    
    # Estimate model size (4 bytes per float32 parameter)
    model_size_mb = (num_params * 4) / (1024 * 1024)
    
    return {
        'num_layers': len(architecture.layers),
        'num_parameters': num_params,
        'flops': flops,
        'model_size_mb': model_size_mb,
        'num_skip_connections': len(architecture.connections),
    }


def estimate_search_space_size(
    layer_types: List[str],
    param_ranges: Dict[str, List[Any]],
    min_layers: int,
    max_layers: int,
    allow_skip_connections: bool = False
) -> int:
    """
    Estimate the size of a search space.
    
    Args:
        layer_types: List of possible layer types
        param_ranges: Dictionary mapping parameter names to possible values
        min_layers: Minimum number of layers
        max_layers: Maximum number of layers
        allow_skip_connections: Whether skip connections are allowed
    
    Returns:
        Estimated number of possible architectures (may be approximate for large spaces)
    """
    total_size = 0
    
    for num_layers in range(min_layers, max_layers + 1):
        # Number of ways to choose layer types
        layer_combinations = len(layer_types) ** num_layers
        
        # Number of parameter combinations per layer (simplified)
        param_combinations = 1
        for param_values in param_ranges.values():
            param_combinations *= len(param_values)
        
        # Total combinations for this depth
        depth_combinations = layer_combinations * (param_combinations ** num_layers)
        
        # If skip connections are allowed, multiply by possible connection patterns
        if allow_skip_connections and num_layers > 1:
            # Approximate: each pair of layers can have a connection or not
            max_connections = num_layers * (num_layers - 1) // 2
            # This is a rough approximation
            connection_combinations = 2 ** min(max_connections, 10)  # Cap to avoid overflow
            depth_combinations *= connection_combinations
        
        total_size += depth_combinations
    
    return total_size


def get_layer_type_distribution(architectures: List[Architecture]) -> Dict[str, int]:
    """
    Get the distribution of layer types across a list of architectures.
    
    Args:
        architectures: List of architectures to analyze
    
    Returns:
        Dictionary mapping layer types to their counts
    """
    distribution = {}
    
    for arch in architectures:
        for layer in arch.layers:
            layer_type = layer.layer_type
            distribution[layer_type] = distribution.get(layer_type, 0) + 1
    
    return distribution


def get_architecture_statistics(architectures: List[Architecture]) -> Dict[str, Any]:
    """
    Calculate statistics across a list of architectures.
    
    Args:
        architectures: List of architectures to analyze
    
    Returns:
        Dictionary containing statistics:
        - 'num_architectures': Total number of architectures
        - 'avg_num_layers': Average number of layers
        - 'min_num_layers': Minimum number of layers
        - 'max_num_layers': Maximum number of layers
        - 'avg_skip_connections': Average number of skip connections
        - 'layer_type_distribution': Distribution of layer types
    """
    if not architectures:
        return {
            'num_architectures': 0,
            'avg_num_layers': 0,
            'min_num_layers': 0,
            'max_num_layers': 0,
            'avg_skip_connections': 0,
            'layer_type_distribution': {},
        }
    
    num_layers_list = [len(arch.layers) for arch in architectures]
    num_connections_list = [len(arch.connections) for arch in architectures]
    
    return {
        'num_architectures': len(architectures),
        'avg_num_layers': np.mean(num_layers_list),
        'min_num_layers': min(num_layers_list),
        'max_num_layers': max(num_layers_list),
        'avg_skip_connections': np.mean(num_connections_list),
        'layer_type_distribution': get_layer_type_distribution(architectures),
    }


def format_architecture_summary(architecture: Architecture, input_shape: Optional[Tuple[int, ...]] = None) -> str:
    """
    Generate a human-readable summary of an architecture.
    
    Args:
        architecture: Architecture to summarize
        input_shape: Optional input shape for complexity metrics
    
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append(f"Architecture ID: {architecture.id}")
    lines.append(f"Number of layers: {len(architecture.layers)}")
    lines.append(f"Skip connections: {len(architecture.connections)}")
    
    if input_shape:
        metrics = get_architecture_complexity_metrics(architecture, input_shape)
        lines.append(f"Parameters: {metrics['num_parameters']:,}")
        lines.append(f"FLOPs: {metrics['flops']:,}")
        lines.append(f"Model size: {metrics['model_size_mb']:.2f} MB")
    
    lines.append("\nLayers:")
    for i, layer in enumerate(architecture.layers):
        lines.append(f"  {i}: {layer}")
    
    if architecture.connections:
        lines.append("\nSkip connections:")
        for from_idx, to_idx in architecture.connections:
            lines.append(f"  {from_idx} -> {to_idx}")
    
    # Performance metrics
    perf_metrics = architecture.metadata.get('performance_metrics', {})
    if perf_metrics:
        lines.append("\nPerformance metrics:")
        for metric, value in perf_metrics.items():
            lines.append(f"  {metric}: {value:.4f}")
    
    # Hardware metrics
    hw_metrics = architecture.metadata.get('hardware_metrics', {})
    if hw_metrics:
        lines.append("\nHardware metrics:")
        for metric, value in hw_metrics.items():
            lines.append(f"  {metric}: {value:.2f}")
    
    return "\n".join(lines)

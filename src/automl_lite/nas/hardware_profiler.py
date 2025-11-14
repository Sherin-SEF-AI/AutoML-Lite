"""
Hardware profiling components for Neural Architecture Search.

This module provides tools for estimating hardware-specific metrics like
latency, memory usage, and model size for neural network architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from .architecture import Architecture, LayerConfig


@dataclass
class HardwareMetrics:
    """Container for hardware-related metrics."""
    latency_ms: float
    memory_mb: float
    model_size_mb: float
    flops: int
    num_parameters: int


class HardwareProfiler(ABC):
    """
    Abstract base class for hardware profiling.
    
    Provides interface for estimating latency, memory usage, and other
    hardware-specific metrics for neural network architectures.
    """
    
    def __init__(self, target_hardware: str = 'cpu', batch_size: int = 1):
        """
        Initialize hardware profiler.
        
        Args:
            target_hardware: Target hardware platform ('cpu', 'gpu', 'mobile', 'edge')
            batch_size: Batch size for inference
        """
        self.target_hardware = target_hardware
        self.batch_size = batch_size
        self._validate_hardware()
    
    def _validate_hardware(self):
        """Validate target hardware specification."""
        valid_hardware = ['cpu', 'gpu', 'mobile', 'edge']
        if self.target_hardware not in valid_hardware:
            raise ValueError(
                f"target_hardware must be one of {valid_hardware}, "
                f"got '{self.target_hardware}'"
            )
    
    @abstractmethod
    def estimate_latency(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate inference latency for an architecture.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated latency in milliseconds
        """
        pass
    
    @abstractmethod
    def estimate_memory(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate peak memory usage for an architecture.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated peak memory in MB
        """
        pass
    
    def estimate_model_size(self, arch: Architecture) -> float:
        """
        Estimate model size (parameters only).
        
        Args:
            arch: Architecture to profile
        
        Returns:
            Estimated model size in MB
        """
        num_params = self.count_parameters(arch)
        # Assume float32 (4 bytes per parameter)
        size_bytes = num_params * 4
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    
    def count_flops(self, arch: Architecture, input_shape: Optional[Tuple[int, ...]] = None) -> int:
        """
        Count floating point operations (FLOPs) for an architecture.
        
        Args:
            arch: Architecture to profile
            input_shape: Input shape (excluding batch dimension)
        
        Returns:
            Total FLOPs for forward pass
        """
        total_flops = 0
        current_shape = input_shape
        
        for layer in arch.layers:
            layer_flops, output_shape = self._count_layer_flops(layer, current_shape)
            total_flops += layer_flops
            current_shape = output_shape
        
        return total_flops
    
    def count_parameters(self, arch: Architecture) -> int:
        """
        Count total number of trainable parameters.
        
        Args:
            arch: Architecture to profile
        
        Returns:
            Total number of parameters
        """
        total_params = 0
        
        for layer in arch.layers:
            layer_params = self._count_layer_parameters(layer)
            total_params += layer_params
        
        return total_params
    
    def _count_layer_flops(
        self, 
        layer: LayerConfig, 
        input_shape: Optional[Tuple[int, ...]]
    ) -> Tuple[int, Optional[Tuple[int, ...]]]:
        """
        Count FLOPs for a single layer.
        
        Args:
            layer: Layer configuration
            input_shape: Input shape to the layer
        
        Returns:
            Tuple of (flops, output_shape)
        """
        layer_type = layer.layer_type.lower()
        params = layer.params
        
        if layer_type == 'dense':
            units = params.get('units', 128)
            if input_shape:
                input_size = np.prod(input_shape)
                # FLOPs = 2 * input_size * units (multiply-add operations)
                flops = 2 * input_size * units
                output_shape = (units,)
            else:
                flops = 0
                output_shape = (units,)
            return flops, output_shape
        
        elif layer_type == 'conv2d':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', 3)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            if input_shape and len(input_shape) >= 3:
                # input_shape: (height, width, channels)
                h, w, c = input_shape[-3], input_shape[-2], input_shape[-1]
                strides = params.get('strides', 1)
                if isinstance(strides, int):
                    strides = (strides, strides)
                
                # Output dimensions
                out_h = (h - kernel_size[0]) // strides[0] + 1
                out_w = (w - kernel_size[1]) // strides[1] + 1
                
                # FLOPs = 2 * kernel_h * kernel_w * in_channels * out_h * out_w * out_channels
                flops = 2 * kernel_size[0] * kernel_size[1] * c * out_h * out_w * filters
                output_shape = (out_h, out_w, filters)
            else:
                flops = 0
                output_shape = None
            return flops, output_shape
        
        elif layer_type == 'conv1d':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', 3)
            
            if input_shape and len(input_shape) >= 2:
                # input_shape: (length, channels)
                length, c = input_shape[-2], input_shape[-1]
                strides = params.get('strides', 1)
                
                # Output dimensions
                out_length = (length - kernel_size) // strides + 1
                
                # FLOPs = 2 * kernel_size * in_channels * out_length * out_channels
                flops = 2 * kernel_size * c * out_length * filters
                output_shape = (out_length, filters)
            else:
                flops = 0
                output_shape = None
            return flops, output_shape
        
        elif layer_type in ['lstm', 'gru']:
            units = params.get('units', 128)
            if input_shape and len(input_shape) >= 2:
                # input_shape: (timesteps, features)
                timesteps, features = input_shape[-2], input_shape[-1]
                
                if layer_type == 'lstm':
                    # LSTM has 4 gates, each with input and recurrent weights
                    # FLOPs per timestep = 4 * (2 * features * units + 2 * units * units)
                    flops_per_step = 4 * (2 * features * units + 2 * units * units)
                else:  # GRU
                    # GRU has 3 gates
                    flops_per_step = 3 * (2 * features * units + 2 * units * units)
                
                flops = flops_per_step * timesteps
                
                return_sequences = params.get('return_sequences', False)
                if return_sequences:
                    output_shape = (timesteps, units)
                else:
                    output_shape = (units,)
            else:
                flops = 0
                output_shape = None
            return flops, output_shape
        
        elif layer_type in ['maxpooling2d', 'avgpooling2d', 'maxpooling1d', 'avgpooling1d']:
            # Pooling operations have minimal FLOPs
            if input_shape:
                output_shape = input_shape  # Simplified, actual depends on pool size
            else:
                output_shape = None
            return 0, output_shape
        
        elif layer_type in ['dropout', 'batchnormalization', 'flatten']:
            # These layers have minimal computational cost
            return 0, input_shape
        
        else:
            # Unknown layer type
            return 0, input_shape
    
    def _count_layer_parameters(self, layer: LayerConfig) -> int:
        """
        Count parameters for a single layer.
        
        Args:
            layer: Layer configuration
        
        Returns:
            Number of parameters
        """
        layer_type = layer.layer_type.lower()
        params = layer.params
        
        if layer_type == 'dense':
            units = params.get('units', 128)
            # Need input size, use layer's input_shape if available
            if layer.input_shape:
                input_size = np.prod(layer.input_shape)
                # weights + biases
                return input_size * units + units
            else:
                # Can't determine without input shape
                return 0
        
        elif layer_type == 'conv2d':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', 3)
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            
            if layer.input_shape and len(layer.input_shape) >= 3:
                in_channels = layer.input_shape[-1]
                # weights + biases
                return kernel_size[0] * kernel_size[1] * in_channels * filters + filters
            else:
                return 0
        
        elif layer_type == 'conv1d':
            filters = params.get('filters', 32)
            kernel_size = params.get('kernel_size', 3)
            
            if layer.input_shape and len(layer.input_shape) >= 2:
                in_channels = layer.input_shape[-1]
                # weights + biases
                return kernel_size * in_channels * filters + filters
            else:
                return 0
        
        elif layer_type == 'lstm':
            units = params.get('units', 128)
            if layer.input_shape and len(layer.input_shape) >= 2:
                features = layer.input_shape[-1]
                # LSTM has 4 gates: input, forget, cell, output
                # Each gate has input weights, recurrent weights, and biases
                # params = 4 * (features * units + units * units + units)
                return 4 * (features * units + units * units + units)
            else:
                return 0
        
        elif layer_type == 'gru':
            units = params.get('units', 128)
            if layer.input_shape and len(layer.input_shape) >= 2:
                features = layer.input_shape[-1]
                # GRU has 3 gates: reset, update, new
                return 3 * (features * units + units * units + units)
            else:
                return 0
        
        elif layer_type == 'batchnormalization':
            # BatchNorm has gamma, beta parameters (2 per feature)
            if layer.input_shape:
                num_features = layer.input_shape[-1]
                return 2 * num_features
            else:
                return 0
        
        elif layer_type in ['dropout', 'maxpooling2d', 'avgpooling2d', 
                           'maxpooling1d', 'avgpooling1d', 'flatten']:
            # These layers have no trainable parameters
            return 0
        
        else:
            # Unknown layer type
            return 0
    
    def profile_architecture(
        self, 
        arch: Architecture, 
        batch_size: Optional[int] = None
    ) -> HardwareMetrics:
        """
        Profile an architecture and return all hardware metrics.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            HardwareMetrics object with all metrics
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        latency = self.estimate_latency(arch, batch_size)
        memory = self.estimate_memory(arch, batch_size)
        model_size = self.estimate_model_size(arch)
        flops = self.count_flops(arch)
        num_params = self.count_parameters(arch)
        
        return HardwareMetrics(
            latency_ms=latency,
            memory_mb=memory,
            model_size_mb=model_size,
            flops=flops,
            num_parameters=num_params
        )
    
    def check_constraints(
        self, 
        arch: Architecture, 
        constraints: Dict[str, float]
    ) -> bool:
        """
        Check if an architecture satisfies hardware constraints.
        
        Args:
            arch: Architecture to check
            constraints: Dictionary of constraint name -> max value
                        Supported: 'max_latency_ms', 'max_memory_mb', 'max_model_size_mb'
        
        Returns:
            True if all constraints are satisfied, False otherwise
        """
        metrics = self.profile_architecture(arch)
        
        if 'max_latency_ms' in constraints:
            if metrics.latency_ms > constraints['max_latency_ms']:
                return False
        
        if 'max_memory_mb' in constraints:
            if metrics.memory_mb > constraints['max_memory_mb']:
                return False
        
        if 'max_model_size_mb' in constraints:
            if metrics.model_size_mb > constraints['max_model_size_mb']:
                return False
        
        return True



class LatencyPredictor(HardwareProfiler):
    """
    Predicts inference latency on target hardware.
    
    Uses lookup tables and analytical models to estimate layer-wise latency
    based on hardware characteristics.
    """
    
    # Lookup tables for layer latency (microseconds per operation)
    # These are approximate values and should be calibrated for specific hardware
    LATENCY_TABLES = {
        'cpu': {
            'dense': 0.001,  # per MAC operation
            'conv2d': 0.002,
            'conv1d': 0.0015,
            'lstm': 0.005,
            'gru': 0.004,
            'maxpooling2d': 0.0001,
            'avgpooling2d': 0.0001,
            'maxpooling1d': 0.0001,
            'avgpooling1d': 0.0001,
            'dropout': 0.00001,
            'batchnormalization': 0.0002,
            'flatten': 0.00001,
        },
        'gpu': {
            'dense': 0.0002,
            'conv2d': 0.0003,
            'conv1d': 0.00025,
            'lstm': 0.001,
            'gru': 0.0008,
            'maxpooling2d': 0.00002,
            'avgpooling2d': 0.00002,
            'maxpooling1d': 0.00002,
            'avgpooling1d': 0.00002,
            'dropout': 0.000001,
            'batchnormalization': 0.00003,
            'flatten': 0.000001,
        },
        'mobile': {
            'dense': 0.005,
            'conv2d': 0.008,
            'conv1d': 0.006,
            'lstm': 0.02,
            'gru': 0.015,
            'maxpooling2d': 0.0005,
            'avgpooling2d': 0.0005,
            'maxpooling1d': 0.0005,
            'avgpooling1d': 0.0005,
            'dropout': 0.0001,
            'batchnormalization': 0.001,
            'flatten': 0.0001,
        },
        'edge': {
            'dense': 0.01,
            'conv2d': 0.015,
            'conv1d': 0.012,
            'lstm': 0.04,
            'gru': 0.03,
            'maxpooling2d': 0.001,
            'avgpooling2d': 0.001,
            'maxpooling1d': 0.001,
            'avgpooling1d': 0.001,
            'dropout': 0.0002,
            'batchnormalization': 0.002,
            'flatten': 0.0002,
        },
    }
    
    # Memory bandwidth (GB/s) for different hardware
    MEMORY_BANDWIDTH = {
        'cpu': 50,
        'gpu': 300,
        'mobile': 20,
        'edge': 10,
    }
    
    def __init__(
        self, 
        target_hardware: str = 'cpu', 
        batch_size: int = 1,
        calibration_data: Optional[Dict[str, float]] = None
    ):
        """
        Initialize latency predictor.
        
        Args:
            target_hardware: Target hardware platform
            batch_size: Batch size for inference
            calibration_data: Optional calibration data to adjust predictions
        """
        super().__init__(target_hardware, batch_size)
        self.calibration_data = calibration_data or {}
        self.latency_table = self.LATENCY_TABLES[target_hardware]
        self.memory_bandwidth = self.MEMORY_BANDWIDTH[target_hardware]
    
    def estimate_latency(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate inference latency for an architecture.
        
        Latency model: latency = compute_time + memory_access_time + overhead
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated latency in milliseconds
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        total_latency_us = 0.0
        current_shape = None
        
        for layer in arch.layers:
            # Use layer's input_shape if current_shape is None
            input_shape = current_shape if current_shape is not None else layer.input_shape
            layer_latency = self._estimate_layer_latency(layer, input_shape, batch_size)
            total_latency_us += layer_latency
            
            # Update current shape for next layer
            if layer.output_shape:
                current_shape = layer.output_shape
        
        # Add communication overhead (increases with number of layers and batch size)
        overhead_us = len(arch.layers) * 10 * batch_size  # 10 microseconds per layer per sample
        total_latency_us += overhead_us
        
        # Apply calibration if available
        if 'latency_multiplier' in self.calibration_data:
            total_latency_us *= self.calibration_data['latency_multiplier']
        
        # Convert to milliseconds
        latency_ms = total_latency_us / 1000.0
        
        return latency_ms
    
    def _estimate_layer_latency(
        self, 
        layer: LayerConfig, 
        input_shape: Optional[Tuple[int, ...]], 
        batch_size: int
    ) -> float:
        """
        Estimate latency for a single layer.
        
        Args:
            layer: Layer configuration
            input_shape: Input shape to the layer
            batch_size: Batch size
        
        Returns:
            Estimated latency in microseconds
        """
        layer_type = layer.layer_type.lower()
        
        # Get base latency per operation
        base_latency = self.latency_table.get(layer_type, 0.001)
        
        # Count operations
        flops, _ = self._count_layer_flops(layer, input_shape)
        
        # Compute time = FLOPs * latency_per_op * batch_size
        compute_time_us = flops * base_latency * batch_size
        
        # Memory access time
        num_params = self._count_layer_parameters(layer)
        param_size_bytes = num_params * 4  # float32
        
        # Memory access time = data_size / bandwidth
        # bandwidth in GB/s, convert to bytes/microsecond
        bandwidth_bytes_per_us = self.memory_bandwidth * 1e9 / 1e6
        memory_time_us = param_size_bytes / bandwidth_bytes_per_us
        
        total_latency_us = compute_time_us + memory_time_us
        
        return total_latency_us
    
    def estimate_memory(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate peak memory usage (delegates to MemoryEstimator logic).
        
        This is a simplified implementation for compatibility.
        For detailed memory estimation, use MemoryEstimator directly.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated peak memory in MB
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Simple memory estimation based on parameters and activations
        num_params = self.count_parameters(arch)
        parameter_memory_bytes = num_params * 4  # float32
        
        # Rough activation memory estimate
        activation_memory_bytes = 0
        for layer in arch.layers:
            if layer.output_shape:
                activation_size = np.prod(layer.output_shape) * batch_size * 4
                activation_memory_bytes += activation_size
        
        total_memory_bytes = activation_memory_bytes + parameter_memory_bytes
        memory_mb = total_memory_bytes / (1024 * 1024)
        
        return memory_mb
    
    def calibrate(self, measurements: Dict[str, float]):
        """
        Calibrate the latency predictor using actual measurements.
        
        Args:
            measurements: Dictionary mapping architecture IDs to measured latencies (ms)
        """
        # Simple calibration: compute average ratio of measured to predicted
        if not measurements:
            return
        
        ratios = []
        for arch_id, measured_latency in measurements.items():
            # This is a simplified calibration
            # In practice, you'd need the actual architecture objects
            pass
        
        # Store calibration data
        if ratios:
            self.calibration_data['latency_multiplier'] = np.mean(ratios)


class MemoryEstimator(HardwareProfiler):
    """
    Estimates peak memory usage during inference and training.
    
    Tracks activation memory, parameter memory, and gradient memory.
    """
    
    def __init__(self, target_hardware: str = 'cpu', batch_size: int = 1):
        """
        Initialize memory estimator.
        
        Args:
            target_hardware: Target hardware platform
            batch_size: Batch size for inference
        """
        super().__init__(target_hardware, batch_size)
    
    def estimate_latency(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate inference latency (simplified implementation).
        
        This is a basic implementation for compatibility.
        For detailed latency estimation, use LatencyPredictor directly.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated latency in milliseconds
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Simple latency estimate based on FLOPs
        flops = self.count_flops(arch)
        # Assume 1 GFLOPS throughput as baseline
        latency_ms = (flops * batch_size) / 1e6  # Convert to milliseconds
        
        return latency_ms
    
    def estimate_memory(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate peak memory usage for an architecture.
        
        Memory = activation_memory + parameter_memory + gradient_memory (training only)
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated peak memory in MB
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Calculate activation memory
        activation_memory_bytes = self._estimate_activation_memory(arch, batch_size)
        
        # Calculate parameter memory
        num_params = self.count_parameters(arch)
        parameter_memory_bytes = num_params * 4  # float32
        
        # For inference, we don't need gradient memory
        # For training, gradient memory ≈ parameter memory
        # We'll estimate for inference here
        
        total_memory_bytes = activation_memory_bytes + parameter_memory_bytes
        
        # Convert to MB
        memory_mb = total_memory_bytes / (1024 * 1024)
        
        return memory_mb
    
    def estimate_training_memory(self, arch: Architecture, batch_size: Optional[int] = None) -> float:
        """
        Estimate peak memory usage during training.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
        
        Returns:
            Estimated peak memory in MB
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        # Activation memory (forward + backward)
        activation_memory_bytes = self._estimate_activation_memory(arch, batch_size)
        # During backprop, we need to store activations for gradient computation
        activation_memory_bytes *= 2
        
        # Parameter memory
        num_params = self.count_parameters(arch)
        parameter_memory_bytes = num_params * 4  # float32
        
        # Gradient memory (same size as parameters)
        gradient_memory_bytes = parameter_memory_bytes
        
        # Optimizer state (e.g., Adam stores momentum and velocity)
        # Approximately 2x parameter memory for Adam
        optimizer_memory_bytes = parameter_memory_bytes * 2
        
        total_memory_bytes = (
            activation_memory_bytes + 
            parameter_memory_bytes + 
            gradient_memory_bytes + 
            optimizer_memory_bytes
        )
        
        # Convert to MB
        memory_mb = total_memory_bytes / (1024 * 1024)
        
        return memory_mb
    
    def _estimate_activation_memory(self, arch: Architecture, batch_size: int) -> int:
        """
        Estimate memory required for activations.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size
        
        Returns:
            Activation memory in bytes
        """
        total_activation_bytes = 0
        current_shape = None
        
        for layer in arch.layers:
            # Get output shape
            if layer.output_shape:
                output_shape = layer.output_shape
            else:
                # Estimate output shape
                _, output_shape = self._count_layer_flops(layer, current_shape)
            
            if output_shape:
                # Calculate activation size
                activation_size = np.prod(output_shape) * batch_size * 4  # float32
                total_activation_bytes += activation_size
                current_shape = output_shape
        
        return int(total_activation_bytes)
    
    def get_memory_breakdown(
        self, 
        arch: Architecture, 
        batch_size: Optional[int] = None,
        training: bool = False
    ) -> Dict[str, float]:
        """
        Get detailed memory breakdown.
        
        Args:
            arch: Architecture to profile
            batch_size: Batch size (uses default if None)
            training: Whether to estimate for training (vs inference)
        
        Returns:
            Dictionary with memory breakdown in MB
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        activation_memory_bytes = self._estimate_activation_memory(arch, batch_size)
        if training:
            activation_memory_bytes *= 2  # Forward + backward
        
        num_params = self.count_parameters(arch)
        parameter_memory_bytes = num_params * 4
        
        breakdown = {
            'activation_mb': activation_memory_bytes / (1024 * 1024),
            'parameter_mb': parameter_memory_bytes / (1024 * 1024),
        }
        
        if training:
            gradient_memory_bytes = parameter_memory_bytes
            optimizer_memory_bytes = parameter_memory_bytes * 2
            breakdown['gradient_mb'] = gradient_memory_bytes / (1024 * 1024)
            breakdown['optimizer_mb'] = optimizer_memory_bytes / (1024 * 1024)
        
        breakdown['total_mb'] = sum(breakdown.values())
        
        return breakdown


class HardwareConstraintChecker:
    """
    Validates architectures against hardware constraints.
    
    Filters architectures that violate latency, memory, or model size constraints.
    """
    
    def __init__(
        self,
        profiler: HardwareProfiler,
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        max_model_size_mb: Optional[float] = None
    ):
        """
        Initialize constraint checker.
        
        Args:
            profiler: Hardware profiler to use for estimation
            max_latency_ms: Maximum allowed latency in milliseconds
            max_memory_mb: Maximum allowed memory in MB
            max_model_size_mb: Maximum allowed model size in MB
        """
        self.profiler = profiler
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb
        self.max_model_size_mb = max_model_size_mb
    
    def check_constraints(self, arch: Architecture) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if an architecture satisfies all constraints.
        
        Args:
            arch: Architecture to check
        
        Returns:
            Tuple of (satisfies_constraints, violation_details)
        """
        violations = {}
        
        # Profile the architecture
        metrics = self.profiler.profile_architecture(arch)
        
        # Check latency constraint
        if self.max_latency_ms is not None:
            if metrics.latency_ms > self.max_latency_ms:
                violations['latency'] = {
                    'actual': metrics.latency_ms,
                    'max': self.max_latency_ms,
                    'violation': metrics.latency_ms - self.max_latency_ms
                }
        
        # Check memory constraint
        if self.max_memory_mb is not None:
            if metrics.memory_mb > self.max_memory_mb:
                violations['memory'] = {
                    'actual': metrics.memory_mb,
                    'max': self.max_memory_mb,
                    'violation': metrics.memory_mb - self.max_memory_mb
                }
        
        # Check model size constraint
        if self.max_model_size_mb is not None:
            if metrics.model_size_mb > self.max_model_size_mb:
                violations['model_size'] = {
                    'actual': metrics.model_size_mb,
                    'max': self.max_model_size_mb,
                    'violation': metrics.model_size_mb - self.max_model_size_mb
                }
        
        satisfies = len(violations) == 0
        
        return satisfies, violations
    
    def filter_architectures(
        self, 
        architectures: list
    ) -> Tuple[list, list]:
        """
        Filter architectures based on constraints.
        
        Args:
            architectures: List of Architecture objects to filter
        
        Returns:
            Tuple of (valid_architectures, rejected_architectures)
        """
        valid = []
        rejected = []
        
        for arch in architectures:
            satisfies, violations = self.check_constraints(arch)
            if satisfies:
                valid.append(arch)
            else:
                # Store violation info in metadata
                arch.metadata['constraint_violations'] = violations
                rejected.append(arch)
        
        return valid, rejected
    
    def get_constraint_summary(self) -> Dict[str, Optional[float]]:
        """
        Get summary of configured constraints.
        
        Returns:
            Dictionary of constraint names and values
        """
        return {
            'max_latency_ms': self.max_latency_ms,
            'max_memory_mb': self.max_memory_mb,
            'max_model_size_mb': self.max_model_size_mb,
        }

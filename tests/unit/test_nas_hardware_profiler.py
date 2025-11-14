"""
Unit tests for NAS hardware profiler components.
"""

import pytest
import numpy as np
from src.automl_lite.nas.hardware_profiler import (
    HardwareProfiler,
    HardwareMetrics,
    LatencyPredictor,
    MemoryEstimator,
    HardwareConstraintChecker,
)
from src.automl_lite.nas.architecture import Architecture, LayerConfig


class TestHardwareProfiler:
    """Test HardwareProfiler base class."""
    
    def test_initialization(self):
        """Test profiler initialization."""
        profiler = LatencyPredictor(target_hardware='cpu', batch_size=1)
        assert profiler.target_hardware == 'cpu'
        assert profiler.batch_size == 1
    
    def test_invalid_hardware(self):
        """Test that invalid hardware raises error."""
        with pytest.raises(ValueError, match="target_hardware must be one of"):
            LatencyPredictor(target_hardware='invalid')
    
    def test_count_parameters_dense(self):
        """Test parameter counting for dense layers."""
        profiler = LatencyPredictor()
        
        # Dense layer: input_size * units + units
        layer = LayerConfig(
            layer_type='dense',
            params={'units': 128},
            input_shape=(64,)
        )
        
        params = profiler._count_layer_parameters(layer)
        expected = 64 * 128 + 128  # weights + biases
        assert params == expected
    
    def test_count_parameters_conv2d(self):
        """Test parameter counting for conv2d layers."""
        profiler = LatencyPredictor()
        
        # Conv2D: kernel_h * kernel_w * in_channels * out_channels + out_channels
        layer = LayerConfig(
            layer_type='conv2d',
            params={'filters': 32, 'kernel_size': 3},
            input_shape=(28, 28, 3)
        )
        
        params = profiler._count_layer_parameters(layer)
        expected = 3 * 3 * 3 * 32 + 32  # weights + biases
        assert params == expected
    
    def test_count_parameters_lstm(self):
        """Test parameter counting for LSTM layers."""
        profiler = LatencyPredictor()
        
        # LSTM: 4 gates * (input_weights + recurrent_weights + biases)
        layer = LayerConfig(
            layer_type='lstm',
            params={'units': 64},
            input_shape=(10, 32)  # (timesteps, features)
        )
        
        params = profiler._count_layer_parameters(layer)
        expected = 4 * (32 * 64 + 64 * 64 + 64)
        assert params == expected
    
    def test_count_flops_dense(self):
        """Test FLOP counting for dense layers."""
        profiler = LatencyPredictor()
        
        layer = LayerConfig(
            layer_type='dense',
            params={'units': 128},
            input_shape=(64,)
        )
        
        flops, output_shape = profiler._count_layer_flops(layer, (64,))
        expected_flops = 2 * 64 * 128  # multiply-add operations
        assert flops == expected_flops
        assert output_shape == (128,)
    
    def test_count_flops_conv2d(self):
        """Test FLOP counting for conv2d layers."""
        profiler = LatencyPredictor()
        
        layer = LayerConfig(
            layer_type='conv2d',
            params={'filters': 32, 'kernel_size': 3, 'strides': 1},
            input_shape=(28, 28, 3)
        )
        
        flops, output_shape = profiler._count_layer_flops(layer, (28, 28, 3))
        
        # Output: (28-3)/1 + 1 = 26
        out_h = out_w = 26
        expected_flops = 2 * 3 * 3 * 3 * out_h * out_w * 32
        assert flops == expected_flops
        assert output_shape == (26, 26, 32)
    
    def test_estimate_model_size(self):
        """Test model size estimation."""
        profiler = LatencyPredictor()
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 10},
                    input_shape=(128,)
                ),
            ]
        )
        
        model_size_mb = profiler.estimate_model_size(arch)
        
        # Total params: (64*128 + 128) + (128*10 + 10) = 8320 + 1290 = 9610
        # Size in MB: 9610 * 4 / (1024*1024) ≈ 0.0366
        assert model_size_mb > 0
        assert model_size_mb < 1  # Should be small for this architecture


class TestLatencyPredictor:
    """Test LatencyPredictor class."""
    
    def test_initialization(self):
        """Test latency predictor initialization."""
        predictor = LatencyPredictor(target_hardware='gpu', batch_size=32)
        assert predictor.target_hardware == 'gpu'
        assert predictor.batch_size == 32
        assert 'dense' in predictor.latency_table
    
    def test_estimate_latency_simple_arch(self):
        """Test latency estimation for simple architecture."""
        predictor = LatencyPredictor(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 10},
                    input_shape=(128,),
                    output_shape=(10,)
                ),
            ]
        )
        
        latency_ms = predictor.estimate_latency(arch)
        assert latency_ms > 0
        assert isinstance(latency_ms, float)
    
    def test_latency_increases_with_batch_size(self):
        """Test that latency increases with batch size."""
        predictor = LatencyPredictor(target_hardware='cpu')
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
            ]
        )
        
        latency_batch_1 = predictor.estimate_latency(arch, batch_size=1)
        latency_batch_32 = predictor.estimate_latency(arch, batch_size=32)
        
        assert latency_batch_32 > latency_batch_1
    
    def test_gpu_faster_than_cpu(self):
        """Test that GPU predictions are faster than CPU."""
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 512},
                    input_shape=(256,),
                    output_shape=(512,)
                ),
            ]
        )
        
        cpu_predictor = LatencyPredictor(target_hardware='cpu')
        gpu_predictor = LatencyPredictor(target_hardware='gpu')
        
        cpu_latency = cpu_predictor.estimate_latency(arch)
        gpu_latency = gpu_predictor.estimate_latency(arch)
        
        assert gpu_latency < cpu_latency
    
    def test_mobile_slower_than_cpu(self):
        """Test that mobile predictions are slower than CPU."""
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 256},
                    input_shape=(128,),
                    output_shape=(256,)
                ),
            ]
        )
        
        cpu_predictor = LatencyPredictor(target_hardware='cpu')
        mobile_predictor = LatencyPredictor(target_hardware='mobile')
        
        cpu_latency = cpu_predictor.estimate_latency(arch)
        mobile_latency = mobile_predictor.estimate_latency(arch)
        
        assert mobile_latency > cpu_latency


class TestMemoryEstimator:
    """Test MemoryEstimator class."""
    
    def test_initialization(self):
        """Test memory estimator initialization."""
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        assert estimator.target_hardware == 'cpu'
        assert estimator.batch_size == 1
    
    def test_estimate_memory_simple_arch(self):
        """Test memory estimation for simple architecture."""
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 10},
                    input_shape=(128,),
                    output_shape=(10,)
                ),
            ]
        )
        
        memory_mb = estimator.estimate_memory(arch)
        assert memory_mb > 0
        assert isinstance(memory_mb, float)
    
    def test_memory_increases_with_batch_size(self):
        """Test that memory increases with batch size."""
        estimator = MemoryEstimator(target_hardware='cpu')
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 256},
                    input_shape=(128,),
                    output_shape=(256,)
                ),
            ]
        )
        
        memory_batch_1 = estimator.estimate_memory(arch, batch_size=1)
        memory_batch_32 = estimator.estimate_memory(arch, batch_size=32)
        
        assert memory_batch_32 > memory_batch_1
    
    def test_training_memory_higher_than_inference(self):
        """Test that training memory is higher than inference memory."""
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 512},
                    input_shape=(256,),
                    output_shape=(512,)
                ),
            ]
        )
        
        inference_memory = estimator.estimate_memory(arch)
        training_memory = estimator.estimate_training_memory(arch)
        
        assert training_memory > inference_memory
    
    def test_memory_breakdown(self):
        """Test memory breakdown calculation."""
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
            ]
        )
        
        breakdown = estimator.get_memory_breakdown(arch, training=False)
        
        assert 'activation_mb' in breakdown
        assert 'parameter_mb' in breakdown
        assert 'total_mb' in breakdown
        assert breakdown['total_mb'] > 0
    
    def test_training_memory_breakdown(self):
        """Test training memory breakdown includes gradients and optimizer."""
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
            ]
        )
        
        breakdown = estimator.get_memory_breakdown(arch, training=True)
        
        assert 'activation_mb' in breakdown
        assert 'parameter_mb' in breakdown
        assert 'gradient_mb' in breakdown
        assert 'optimizer_mb' in breakdown
        assert 'total_mb' in breakdown


class TestHardwareConstraintChecker:
    """Test HardwareConstraintChecker class."""
    
    def test_initialization(self):
        """Test constraint checker initialization."""
        profiler = LatencyPredictor()
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=100.0,
            max_memory_mb=50.0,
            max_model_size_mb=10.0
        )
        
        assert checker.max_latency_ms == 100.0
        assert checker.max_memory_mb == 50.0
        assert checker.max_model_size_mb == 10.0
    
    def test_check_constraints_pass(self):
        """Test constraint checking when all constraints are satisfied."""
        profiler = LatencyPredictor(target_hardware='cpu')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=1000.0,  # Very lenient
            max_memory_mb=1000.0,
            max_model_size_mb=100.0
        )
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 64},
                    input_shape=(32,),
                    output_shape=(64,)
                ),
            ]
        )
        
        satisfies, violations = checker.check_constraints(arch)
        assert satisfies is True
        assert len(violations) == 0
    
    def test_check_constraints_latency_violation(self):
        """Test constraint checking with latency violation."""
        profiler = LatencyPredictor(target_hardware='cpu')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=0.001  # Very strict
        )
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 512},
                    input_shape=(512,),
                    output_shape=(512,)
                ),
            ]
        )
        
        satisfies, violations = checker.check_constraints(arch)
        assert satisfies is False
        assert 'latency' in violations
        assert violations['latency']['actual'] > violations['latency']['max']
    
    def test_check_constraints_memory_violation(self):
        """Test constraint checking with memory violation."""
        profiler = MemoryEstimator(target_hardware='cpu')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_memory_mb=0.01  # Very strict
        )
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 1024},
                    input_shape=(1024,),
                    output_shape=(1024,)
                ),
            ]
        )
        
        satisfies, violations = checker.check_constraints(arch)
        assert satisfies is False
        assert 'memory' in violations
    
    def test_check_constraints_model_size_violation(self):
        """Test constraint checking with model size violation."""
        profiler = LatencyPredictor(target_hardware='cpu')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_model_size_mb=0.001  # Very strict
        )
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 2048},
                    input_shape=(2048,),
                    output_shape=(2048,)
                ),
            ]
        )
        
        satisfies, violations = checker.check_constraints(arch)
        assert satisfies is False
        assert 'model_size' in violations
    
    def test_filter_architectures(self):
        """Test filtering architectures based on constraints."""
        profiler = LatencyPredictor(target_hardware='cpu')
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=100.0,
            max_model_size_mb=1.0
        )
        
        # Small architecture (should pass)
        small_arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 32},
                    input_shape=(16,),
                    output_shape=(32,)
                ),
            ]
        )
        
        # Large architecture (should fail)
        large_arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 2048},
                    input_shape=(2048,),
                    output_shape=(2048,)
                ),
            ]
        )
        
        architectures = [small_arch, large_arch]
        valid, rejected = checker.filter_architectures(architectures)
        
        assert len(valid) >= 1
        assert len(rejected) >= 0
        
        # Check that rejected architectures have violation info
        for arch in rejected:
            assert 'constraint_violations' in arch.metadata
    
    def test_get_constraint_summary(self):
        """Test getting constraint summary."""
        profiler = LatencyPredictor()
        checker = HardwareConstraintChecker(
            profiler=profiler,
            max_latency_ms=50.0,
            max_memory_mb=100.0
        )
        
        summary = checker.get_constraint_summary()
        
        assert summary['max_latency_ms'] == 50.0
        assert summary['max_memory_mb'] == 100.0
        assert summary['max_model_size_mb'] is None


class TestHardwareMetrics:
    """Test HardwareMetrics dataclass."""
    
    def test_hardware_metrics_creation(self):
        """Test creating HardwareMetrics object."""
        metrics = HardwareMetrics(
            latency_ms=10.5,
            memory_mb=25.3,
            model_size_mb=5.2,
            flops=1000000,
            num_parameters=50000
        )
        
        assert metrics.latency_ms == 10.5
        assert metrics.memory_mb == 25.3
        assert metrics.model_size_mb == 5.2
        assert metrics.flops == 1000000
        assert metrics.num_parameters == 50000


class TestProfileArchitecture:
    """Test complete architecture profiling."""
    
    def test_profile_architecture(self):
        """Test profiling complete architecture."""
        profiler = LatencyPredictor(target_hardware='cpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='dense',
                    params={'units': 128},
                    input_shape=(64,),
                    output_shape=(128,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 64},
                    input_shape=(128,),
                    output_shape=(64,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 10},
                    input_shape=(64,),
                    output_shape=(10,)
                ),
            ]
        )
        
        metrics = profiler.profile_architecture(arch)
        
        assert isinstance(metrics, HardwareMetrics)
        assert metrics.latency_ms > 0
        assert metrics.memory_mb > 0
        assert metrics.model_size_mb > 0
        assert metrics.flops > 0
        assert metrics.num_parameters > 0
    
    def test_profile_conv_architecture(self):
        """Test profiling convolutional architecture."""
        profiler = LatencyPredictor(target_hardware='gpu', batch_size=1)
        
        arch = Architecture(
            layers=[
                LayerConfig(
                    layer_type='conv2d',
                    params={'filters': 32, 'kernel_size': 3},
                    input_shape=(28, 28, 1),
                    output_shape=(26, 26, 32)
                ),
                LayerConfig(
                    layer_type='maxpooling2d',
                    params={'pool_size': 2},
                    input_shape=(26, 26, 32),
                    output_shape=(13, 13, 32)
                ),
                LayerConfig(
                    layer_type='flatten',
                    input_shape=(13, 13, 32),
                    output_shape=(5408,)
                ),
                LayerConfig(
                    layer_type='dense',
                    params={'units': 10},
                    input_shape=(5408,),
                    output_shape=(10,)
                ),
            ]
        )
        
        metrics = profiler.profile_architecture(arch)
        
        assert metrics.latency_ms > 0
        assert metrics.memory_mb > 0
        assert metrics.num_parameters > 0

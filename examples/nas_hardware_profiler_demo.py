"""
Demo of NAS Hardware Profiler components.

This example demonstrates how to use the hardware profiling tools to estimate
latency, memory usage, and check hardware constraints for neural architectures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from automl_lite.nas import (
    Architecture,
    LayerConfig,
    LatencyPredictor,
    MemoryEstimator,
    HardwareConstraintChecker,
)


def create_sample_architectures():
    """Create sample architectures for demonstration."""
    
    # Small MLP for tabular data
    small_mlp = Architecture(
        layers=[
            LayerConfig(
                layer_type='dense',
                params={'units': 64, 'activation': 'relu'},
                input_shape=(20,),
                output_shape=(64,)
            ),
            LayerConfig(
                layer_type='dropout',
                params={'rate': 0.2},
                input_shape=(64,),
                output_shape=(64,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 32, 'activation': 'relu'},
                input_shape=(64,),
                output_shape=(32,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 1, 'activation': 'sigmoid'},
                input_shape=(32,),
                output_shape=(1,)
            ),
        ]
    )
    
    # Large MLP
    large_mlp = Architecture(
        layers=[
            LayerConfig(
                layer_type='dense',
                params={'units': 512, 'activation': 'relu'},
                input_shape=(100,),
                output_shape=(512,)
            ),
            LayerConfig(
                layer_type='batchnormalization',
                input_shape=(512,),
                output_shape=(512,)
            ),
            LayerConfig(
                layer_type='dropout',
                params={'rate': 0.3},
                input_shape=(512,),
                output_shape=(512,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 256, 'activation': 'relu'},
                input_shape=(512,),
                output_shape=(256,)
            ),
            LayerConfig(
                layer_type='batchnormalization',
                input_shape=(256,),
                output_shape=(256,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 128, 'activation': 'relu'},
                input_shape=(256,),
                output_shape=(128,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 10, 'activation': 'softmax'},
                input_shape=(128,),
                output_shape=(10,)
            ),
        ]
    )
    
    # CNN for image classification
    cnn = Architecture(
        layers=[
            LayerConfig(
                layer_type='conv2d',
                params={'filters': 32, 'kernel_size': 3, 'activation': 'relu'},
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
                layer_type='conv2d',
                params={'filters': 64, 'kernel_size': 3, 'activation': 'relu'},
                input_shape=(13, 13, 32),
                output_shape=(11, 11, 64)
            ),
            LayerConfig(
                layer_type='maxpooling2d',
                params={'pool_size': 2},
                input_shape=(11, 11, 64),
                output_shape=(5, 5, 64)
            ),
            LayerConfig(
                layer_type='flatten',
                input_shape=(5, 5, 64),
                output_shape=(1600,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 128, 'activation': 'relu'},
                input_shape=(1600,),
                output_shape=(128,)
            ),
            LayerConfig(
                layer_type='dropout',
                params={'rate': 0.5},
                input_shape=(128,),
                output_shape=(128,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 10, 'activation': 'softmax'},
                input_shape=(128,),
                output_shape=(10,)
            ),
        ]
    )
    
    # LSTM for time series
    lstm = Architecture(
        layers=[
            LayerConfig(
                layer_type='lstm',
                params={'units': 128, 'return_sequences': True},
                input_shape=(50, 10),  # (timesteps, features)
                output_shape=(50, 128)
            ),
            LayerConfig(
                layer_type='dropout',
                params={'rate': 0.2},
                input_shape=(50, 128),
                output_shape=(50, 128)
            ),
            LayerConfig(
                layer_type='lstm',
                params={'units': 64, 'return_sequences': False},
                input_shape=(50, 128),
                output_shape=(64,)
            ),
            LayerConfig(
                layer_type='dense',
                params={'units': 1},
                input_shape=(64,),
                output_shape=(1,)
            ),
        ]
    )
    
    return {
        'small_mlp': small_mlp,
        'large_mlp': large_mlp,
        'cnn': cnn,
        'lstm': lstm,
    }


def demo_latency_prediction():
    """Demonstrate latency prediction across different hardware."""
    print("=" * 80)
    print("LATENCY PREDICTION DEMO")
    print("=" * 80)
    
    architectures = create_sample_architectures()
    hardware_types = ['cpu', 'gpu', 'mobile', 'edge']
    
    for arch_name, arch in architectures.items():
        print(f"\n{arch_name.upper()} Architecture:")
        print(f"  Layers: {len(arch.layers)}")
        
        for hw in hardware_types:
            predictor = LatencyPredictor(target_hardware=hw, batch_size=1)
            latency = predictor.estimate_latency(arch)
            print(f"  {hw.upper():8s} latency: {latency:8.3f} ms")


def demo_memory_estimation():
    """Demonstrate memory estimation."""
    print("\n" + "=" * 80)
    print("MEMORY ESTIMATION DEMO")
    print("=" * 80)
    
    architectures = create_sample_architectures()
    
    for arch_name, arch in architectures.items():
        print(f"\n{arch_name.upper()} Architecture:")
        
        estimator = MemoryEstimator(target_hardware='cpu', batch_size=1)
        
        # Inference memory
        inference_memory = estimator.estimate_memory(arch)
        print(f"  Inference memory: {inference_memory:.2f} MB")
        
        # Training memory
        training_memory = estimator.estimate_training_memory(arch)
        print(f"  Training memory:  {training_memory:.2f} MB")
        
        # Memory breakdown
        breakdown = estimator.get_memory_breakdown(arch, training=True)
        print(f"  Memory breakdown:")
        print(f"    Activations: {breakdown['activation_mb']:.2f} MB")
        print(f"    Parameters:  {breakdown['parameter_mb']:.2f} MB")
        print(f"    Gradients:   {breakdown['gradient_mb']:.2f} MB")
        print(f"    Optimizer:   {breakdown['optimizer_mb']:.2f} MB")


def demo_model_complexity():
    """Demonstrate model complexity metrics."""
    print("\n" + "=" * 80)
    print("MODEL COMPLEXITY DEMO")
    print("=" * 80)
    
    architectures = create_sample_architectures()
    
    for arch_name, arch in architectures.items():
        print(f"\n{arch_name.upper()} Architecture:")
        
        profiler = LatencyPredictor(target_hardware='cpu')
        
        # Count parameters
        num_params = profiler.count_parameters(arch)
        print(f"  Parameters: {num_params:,}")
        
        # Count FLOPs
        flops = profiler.count_flops(arch)
        print(f"  FLOPs:      {flops:,}")
        
        # Model size
        model_size = profiler.estimate_model_size(arch)
        print(f"  Model size: {model_size:.2f} MB")


def demo_hardware_constraints():
    """Demonstrate hardware constraint checking."""
    print("\n" + "=" * 80)
    print("HARDWARE CONSTRAINT CHECKING DEMO")
    print("=" * 80)
    
    architectures = create_sample_architectures()
    
    # Define constraints for mobile deployment
    print("\nMobile Deployment Constraints:")
    print("  Max latency:    100 ms")
    print("  Max memory:     50 MB")
    print("  Max model size: 10 MB")
    
    profiler = LatencyPredictor(target_hardware='mobile', batch_size=1)
    checker = HardwareConstraintChecker(
        profiler=profiler,
        max_latency_ms=100.0,
        max_memory_mb=50.0,
        max_model_size_mb=10.0
    )
    
    print("\nChecking architectures:")
    for arch_name, arch in architectures.items():
        satisfies, violations = checker.check_constraints(arch)
        
        if satisfies:
            print(f"  ✓ {arch_name}: PASSES all constraints")
        else:
            print(f"  ✗ {arch_name}: VIOLATES constraints")
            for constraint, details in violations.items():
                print(f"      {constraint}: {details['actual']:.2f} > {details['max']:.2f}")


def demo_batch_size_effects():
    """Demonstrate effects of batch size on latency and memory."""
    print("\n" + "=" * 80)
    print("BATCH SIZE EFFECTS DEMO")
    print("=" * 80)
    
    arch = create_sample_architectures()['small_mlp']
    batch_sizes = [1, 8, 32, 128]
    
    print("\nSmall MLP Architecture:")
    print(f"  Batch Size | Latency (ms) | Memory (MB)")
    print(f"  {'-' * 10} | {'-' * 12} | {'-' * 11}")
    
    latency_predictor = LatencyPredictor(target_hardware='cpu')
    memory_estimator = MemoryEstimator(target_hardware='cpu')
    
    for batch_size in batch_sizes:
        latency = latency_predictor.estimate_latency(arch, batch_size=batch_size)
        memory = memory_estimator.estimate_memory(arch, batch_size=batch_size)
        print(f"  {batch_size:10d} | {latency:12.3f} | {memory:11.2f}")


def demo_hardware_comparison():
    """Compare hardware platforms for the same architecture."""
    print("\n" + "=" * 80)
    print("HARDWARE PLATFORM COMPARISON")
    print("=" * 80)
    
    arch = create_sample_architectures()['cnn']
    
    print("\nCNN Architecture on different hardware:")
    print(f"  Hardware | Latency (ms) | Memory (MB) | Model Size (MB)")
    print(f"  {'-' * 8} | {'-' * 12} | {'-' * 11} | {'-' * 15}")
    
    for hw in ['cpu', 'gpu', 'mobile', 'edge']:
        latency_predictor = LatencyPredictor(target_hardware=hw, batch_size=1)
        memory_estimator = MemoryEstimator(target_hardware=hw, batch_size=1)
        
        latency = latency_predictor.estimate_latency(arch)
        memory = memory_estimator.estimate_memory(arch)
        model_size = latency_predictor.estimate_model_size(arch)
        
        print(f"  {hw:8s} | {latency:12.3f} | {memory:11.2f} | {model_size:15.2f}")


def demo_complete_profiling():
    """Demonstrate complete architecture profiling."""
    print("\n" + "=" * 80)
    print("COMPLETE ARCHITECTURE PROFILING")
    print("=" * 80)
    
    arch = create_sample_architectures()['large_mlp']
    
    print("\nLarge MLP Architecture:")
    print(f"  Layers: {len(arch.layers)}")
    
    profiler = LatencyPredictor(target_hardware='cpu', batch_size=1)
    metrics = profiler.profile_architecture(arch)
    
    print(f"\nHardware Metrics:")
    print(f"  Latency:        {metrics.latency_ms:.3f} ms")
    print(f"  Memory:         {metrics.memory_mb:.2f} MB")
    print(f"  Model Size:     {metrics.model_size_mb:.2f} MB")
    print(f"  FLOPs:          {metrics.flops:,}")
    print(f"  Parameters:     {metrics.num_parameters:,}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("NAS HARDWARE PROFILER DEMONSTRATION")
    print("=" * 80)
    
    demo_latency_prediction()
    demo_memory_estimation()
    demo_model_complexity()
    demo_hardware_constraints()
    demo_batch_size_effects()
    demo_hardware_comparison()
    demo_complete_profiling()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()

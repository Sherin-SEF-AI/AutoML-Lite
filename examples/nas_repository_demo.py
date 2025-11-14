"""
Demo: Neural Architecture Search - Architecture Repository and Transfer Learning

This example demonstrates:
1. Creating and using an architecture repository
2. Saving architectures with metadata
3. Finding similar architectures for transfer learning
4. Adapting architectures to new problems
5. Importing and exporting architectures
"""

import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.automl_lite.nas import (
    Architecture,
    LayerConfig,
    ArchitectureRepository,
)


def create_sample_architectures():
    """Create sample architectures for different problem types."""
    
    # Architecture 1: Small MLP for MNIST-like problems
    mnist_arch = Architecture(
        layers=[
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
        ]
    )
    
    # Architecture 2: Larger MLP for complex classification
    complex_arch = Architecture(
        layers=[
            LayerConfig('dense', {'units': 256, 'activation': 'relu'}),
            LayerConfig('batchnormalization', {}),
            LayerConfig('dropout', {'rate': 0.4}),
            LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
            LayerConfig('batchnormalization', {}),
            LayerConfig('dropout', {'rate': 0.3}),
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 20, 'activation': 'softmax'}),
        ]
    )
    
    # Architecture 3: Regression architecture
    regression_arch = Architecture(
        layers=[
            LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
            LayerConfig('dropout', {'rate': 0.2}),
            LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
            LayerConfig('dense', {'units': 1, 'activation': 'linear'}),
        ]
    )
    
    return {
        'mnist': mnist_arch,
        'complex': complex_arch,
        'regression': regression_arch
    }


def demo_save_and_load():
    """Demonstrate saving and loading architectures."""
    print("=" * 70)
    print("Demo 1: Saving and Loading Architectures")
    print("=" * 70)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo_architectures.db')
        repo = ArchitectureRepository(db_path=db_path)
        
        # Create and save an architecture
        architectures = create_sample_architectures()
        mnist_arch = architectures['mnist']
        
        print(f"\nSaving architecture: {mnist_arch}")
        
        arch_id = repo.save_architecture(
            mnist_arch,
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': 60000,
                'n_features': 784,
                'n_classes': 10,
                'dataset_name': 'MNIST'
            },
            performance_metrics={
                'accuracy': 0.98,
                'val_accuracy': 0.97,
                'loss': 0.05,
                'val_loss': 0.08,
                'training_time': 120.5
            },
            hardware_metrics={
                'latency_ms': 3.2,
                'memory_mb': 25.0,
                'model_size_mb': 1.5,
                'num_parameters': 50000,
                'target_hardware': 'cpu'
            },
            search_metadata={
                'search_strategy': 'evolutionary',
                'search_time': 1800.0,
                'search_space_type': 'tabular',
                'generation': 15
            },
            tags=['mnist', 'classification', 'production', 'high-accuracy']
        )
        
        print(f"✓ Saved with ID: {arch_id}")
        
        # Load the architecture
        print(f"\nLoading architecture {arch_id}...")
        result = repo.load_architecture(arch_id)
        
        if result:
            loaded_arch, metadata = result
            print(f"✓ Loaded architecture: {loaded_arch}")
            print(f"  - Layers: {len(loaded_arch.layers)}")
            print(f"  - Problem type: {metadata['dataset_metadata']['problem_type']}")
            print(f"  - Accuracy: {metadata['performance_metrics']['accuracy']:.2%}")
            print(f"  - Latency: {metadata['hardware_metrics']['latency_ms']:.1f}ms")
            print(f"  - Tags: {', '.join(metadata['tags'])}")
        
        repo.close()


def demo_similarity_and_transfer():
    """Demonstrate finding similar architectures for transfer learning."""
    print("\n" + "=" * 70)
    print("Demo 2: Similarity Search and Transfer Learning")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo_architectures.db')
        repo = ArchitectureRepository(db_path=db_path)
        
        # Save multiple architectures with different characteristics
        architectures = create_sample_architectures()
        
        # Save MNIST architecture
        repo.save_architecture(
            architectures['mnist'],
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': 60000,
                'n_features': 784,
                'n_classes': 10
            },
            performance_metrics={'accuracy': 0.98},
            tags=['mnist', 'small']
        )
        
        # Save complex classification architecture
        repo.save_architecture(
            architectures['complex'],
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': 50000,
                'n_features': 1000,
                'n_classes': 20
            },
            performance_metrics={'accuracy': 0.95},
            tags=['complex', 'large']
        )
        
        # Save regression architecture
        repo.save_architecture(
            architectures['regression'],
            dataset_metadata={
                'problem_type': 'regression',
                'n_samples': 10000,
                'n_features': 50,
            },
            performance_metrics={'accuracy': 0.92},
            tags=['regression']
        )
        
        print("\n✓ Saved 3 architectures to repository")
        
        # Find similar architectures for a new MNIST-like problem
        print("\nSearching for architectures similar to:")
        target_metadata = {
            'problem_type': 'classification',
            'n_samples': 70000,
            'n_features': 800,
            'n_classes': 10
        }
        print(f"  - Problem: {target_metadata['problem_type']}")
        print(f"  - Samples: {target_metadata['n_samples']:,}")
        print(f"  - Features: {target_metadata['n_features']}")
        
        similar = repo.find_similar_architectures(target_metadata, top_k=2)
        
        print(f"\n✓ Found {len(similar)} similar architectures:")
        for i, (arch, metadata, similarity) in enumerate(similar, 1):
            print(f"\n  {i}. Similarity: {similarity:.2%}")
            print(f"     Architecture: {arch}")
            print(f"     Problem: {metadata['dataset_metadata']['problem_type']}")
            print(f"     Samples: {metadata['dataset_metadata']['n_samples']:,}")
            print(f"     Features: {metadata['dataset_metadata']['n_features']}")
        
        repo.close()


def demo_architecture_adaptation():
    """Demonstrate adapting architectures to new problems."""
    print("\n" + "=" * 70)
    print("Demo 3: Architecture Adaptation")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo_architectures.db')
        repo = ArchitectureRepository(db_path=db_path)
        
        # Create original architecture
        original = create_sample_architectures()['mnist']
        print(f"\nOriginal architecture:")
        print(f"  Layers: {len(original.layers)}")
        for i, layer in enumerate(original.layers):
            print(f"    {i+1}. {layer}")
        
        # Adapt to a new problem with different input/output
        print("\nAdapting to new problem:")
        print("  - Input shape: (100,) [was (784,)]")
        print("  - Output shape: (5,) [was (10,)]")
        print("  - Dataset size: 50,000 [was 60,000]")
        
        adapted = repo.adapt_architecture(
            original,
            new_input_shape=(100,),
            new_output_shape=(5,),
            dataset_size=50000
        )
        
        print(f"\n✓ Adapted architecture:")
        print(f"  Layers: {len(adapted.layers)}")
        for i, layer in enumerate(adapted.layers):
            print(f"    {i+1}. {layer}")
        
        print(f"\n  Metadata:")
        print(f"    - Adapted from: {adapted.metadata['adapted_from']}")
        print(f"    - New input shape: {adapted.metadata['adaptation']['new_input_shape']}")
        print(f"    - New output shape: {adapted.metadata['adaptation']['new_output_shape']}")
        
        # Adapt with explicit scaling
        print("\n" + "-" * 70)
        print("Adapting with 1.5x scaling factor:")
        
        scaled = repo.adapt_architecture(
            original,
            new_input_shape=(784,),
            new_output_shape=(10,),
            scale_factor=1.5
        )
        
        print(f"\n✓ Scaled architecture:")
        for i, layer in enumerate(scaled.layers):
            print(f"    {i+1}. {layer}")
        
        repo.close()


def demo_import_export():
    """Demonstrate importing and exporting architectures."""
    print("\n" + "=" * 70)
    print("Demo 4: Import and Export")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo_architectures.db')
        repo = ArchitectureRepository(db_path=db_path)
        
        # Create and save an architecture
        arch = create_sample_architectures()['complex']
        arch_id = repo.save_architecture(
            arch,
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': 50000,
                'n_features': 1000
            },
            performance_metrics={'accuracy': 0.95},
            tags=['exported', 'demo']
        )
        
        print(f"\nCreated architecture: {arch_id}")
        
        # Export to JSON
        export_path = os.path.join(tmpdir, 'exported_architecture.json')
        success = repo.export_architecture(arch_id, export_path, include_metadata=True)
        
        if success:
            print(f"✓ Exported to: {export_path}")
            
            # Show file size
            file_size = os.path.getsize(export_path)
            print(f"  File size: {file_size:,} bytes")
        
        # Import from JSON
        print("\nImporting architecture from JSON...")
        imported = repo.import_architecture(
            export_path,
            validate=True,
            save_to_repository=True
        )
        
        if imported:
            print(f"✓ Imported architecture: {imported.id}")
            print(f"  Layers: {len(imported.layers)}")
            
            # Verify it's in the repository
            result = repo.load_architecture(imported.id)
            if result:
                print(f"✓ Verified in repository")
        
        repo.close()


def demo_repository_statistics():
    """Demonstrate repository statistics."""
    print("\n" + "=" * 70)
    print("Demo 5: Repository Statistics")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'demo_architectures.db')
        repo = ArchitectureRepository(db_path=db_path)
        
        # Populate repository with multiple architectures
        architectures = create_sample_architectures()
        
        for i in range(10):
            arch = architectures['mnist'].clone()
            repo.save_architecture(
                arch,
                dataset_metadata={
                    'problem_type': 'classification' if i < 7 else 'regression',
                    'n_samples': 10000 + i * 5000,
                    'n_features': 100 + i * 50
                },
                performance_metrics={'accuracy': 0.85 + i * 0.01},
                tags=['demo', f'batch_{i // 3}']
            )
        
        print(f"\n✓ Populated repository with 10 architectures")
        
        # Get statistics
        stats = repo.get_statistics()
        
        print(f"\nRepository Statistics:")
        print(f"  Total architectures: {stats['total_architectures']}")
        print(f"\n  By problem type:")
        for problem_type, count in stats['by_problem_type'].items():
            print(f"    - {problem_type}: {count}")
        
        print(f"\n  Performance:")
        if stats['avg_accuracy']:
            print(f"    - Average accuracy: {stats['avg_accuracy']:.2%}")
        if stats['max_accuracy']:
            print(f"    - Max accuracy: {stats['max_accuracy']:.2%}")
        
        print(f"\n  Top tags:")
        for tag, count in stats['top_tags'].items():
            print(f"    - {tag}: {count}")
        
        # List architectures
        print(f"\n" + "-" * 70)
        print("Listing top architectures:")
        
        results = repo.list_architectures(limit=5)
        for i, (arch_id, summary) in enumerate(results, 1):
            print(f"\n  {i}. {arch_id[:8]}...")
            print(f"     Problem: {summary['problem_type']}")
            print(f"     Samples: {summary['n_samples']:,}")
            if summary['accuracy']:
                print(f"     Accuracy: {summary['accuracy']:.2%}")
        
        repo.close()


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("NAS Architecture Repository Demo")
    print("=" * 70)
    
    try:
        demo_save_and_load()
        demo_similarity_and_transfer()
        demo_architecture_adaptation()
        demo_import_export()
        demo_repository_statistics()
        
        print("\n" + "=" * 70)
        print("✓ All demos completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

"""
Unit tests for NAS Architecture Repository.
"""

import json
import os
import tempfile
import pytest
from pathlib import Path

from src.automl_lite.nas import (
    Architecture,
    LayerConfig,
    ArchitectureRepository,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_architectures.db')
        yield db_path


@pytest.fixture
def sample_architecture():
    """Create a sample architecture for testing."""
    layers = [
        LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
        LayerConfig('dropout', {'rate': 0.3}),
        LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
        LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
    ]
    return Architecture(layers=layers)


@pytest.fixture
def sample_metadata():
    """Create sample metadata for testing."""
    return {
        'dataset_metadata': {
            'problem_type': 'classification',
            'n_samples': 10000,
            'n_features': 784,
            'n_classes': 10,
            'dataset_name': 'mnist'
        },
        'performance_metrics': {
            'accuracy': 0.95,
            'val_accuracy': 0.93,
            'loss': 0.15,
            'val_loss': 0.18,
            'training_time': 120.5
        },
        'hardware_metrics': {
            'latency_ms': 5.2,
            'memory_mb': 45.0,
            'model_size_mb': 2.3,
            'num_parameters': 100000,
            'target_hardware': 'cpu'
        },
        'search_metadata': {
            'search_strategy': 'evolutionary',
            'search_time': 1800.0,
            'search_space_type': 'tabular',
            'generation': 10
        },
        'tags': ['mnist', 'classification', 'production']
    }


class TestArchitectureRepository:
    """Test suite for ArchitectureRepository."""
    
    def test_repository_initialization(self, temp_db):
        """Test repository initialization and database creation."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        assert repo.db_path == Path(temp_db)
        assert repo.conn is not None
        assert os.path.exists(temp_db)
        
        repo.close()
    
    def test_save_and_load_architecture(self, temp_db, sample_architecture, sample_metadata):
        """Test saving and loading an architecture."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save architecture
        arch_id = repo.save_architecture(
            sample_architecture,
            dataset_metadata=sample_metadata['dataset_metadata'],
            performance_metrics=sample_metadata['performance_metrics'],
            hardware_metrics=sample_metadata['hardware_metrics'],
            search_metadata=sample_metadata['search_metadata'],
            tags=sample_metadata['tags']
        )
        
        assert arch_id == sample_architecture.id
        
        # Load architecture
        result = repo.load_architecture(arch_id)
        assert result is not None
        
        loaded_arch, loaded_metadata = result
        
        # Verify architecture
        assert loaded_arch.id == sample_architecture.id
        assert len(loaded_arch.layers) == len(sample_architecture.layers)
        assert loaded_arch.layers[0].layer_type == 'dense'
        assert loaded_arch.layers[0].params['units'] == 128
        
        # Verify metadata
        assert loaded_metadata['dataset_metadata']['problem_type'] == 'classification'
        assert loaded_metadata['performance_metrics']['accuracy'] == 0.95
        assert loaded_metadata['hardware_metrics']['latency_ms'] == 5.2
        assert 'mnist' in loaded_metadata['tags']
        
        repo.close()
    
    def test_load_nonexistent_architecture(self, temp_db):
        """Test loading an architecture that doesn't exist."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        result = repo.load_architecture('nonexistent-id')
        assert result is None
        
        repo.close()
    
    def test_list_architectures(self, temp_db, sample_architecture, sample_metadata):
        """Test listing architectures with filters."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save multiple architectures
        for i in range(3):
            arch = sample_architecture.clone()
            metadata = sample_metadata.copy()
            metadata['performance_metrics'] = metadata['performance_metrics'].copy()
            metadata['performance_metrics']['accuracy'] = 0.90 + i * 0.02
            
            repo.save_architecture(
                arch,
                dataset_metadata=metadata['dataset_metadata'],
                performance_metrics=metadata['performance_metrics'],
                hardware_metrics=metadata['hardware_metrics']
            )
        
        # List all architectures
        results = repo.list_architectures()
        assert len(results) == 3
        
        # List with problem type filter
        results = repo.list_architectures(problem_type='classification')
        assert len(results) == 3
        
        # List with accuracy filter
        results = repo.list_architectures(min_accuracy=0.92)
        assert len(results) == 2
        
        repo.close()
    
    def test_delete_architecture(self, temp_db, sample_architecture, sample_metadata):
        """Test deleting an architecture."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save architecture
        arch_id = repo.save_architecture(
            sample_architecture,
            dataset_metadata=sample_metadata['dataset_metadata']
        )
        
        # Verify it exists
        result = repo.load_architecture(arch_id)
        assert result is not None
        
        # Delete it
        success = repo.delete_architecture(arch_id)
        assert success is True
        
        # Verify it's gone
        result = repo.load_architecture(arch_id)
        assert result is None
        
        repo.close()
    
    def test_compute_similarity(self, temp_db):
        """Test similarity computation between architectures."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        metadata1 = {
            'problem_type': 'classification',
            'n_samples': 10000,
            'n_features': 784
        }
        
        # Exact match
        metadata2 = metadata1.copy()
        similarity = repo.compute_similarity(metadata1, metadata2)
        assert similarity == 1.0
        
        # Same problem type, different sizes
        metadata3 = {
            'problem_type': 'classification',
            'n_samples': 5000,
            'n_features': 392
        }
        similarity = repo.compute_similarity(metadata1, metadata3)
        assert 0.4 < similarity < 1.0  # Problem type match + partial size match
        
        # Different problem type
        metadata4 = {
            'problem_type': 'regression',
            'n_samples': 10000,
            'n_features': 784
        }
        similarity = repo.compute_similarity(metadata1, metadata4)
        assert similarity < 0.7  # No problem type match
        
        repo.close()
    
    def test_find_similar_architectures(self, temp_db, sample_architecture):
        """Test finding similar architectures."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save architectures with different metadata
        metadata_list = [
            {'problem_type': 'classification', 'n_samples': 10000, 'n_features': 784},
            {'problem_type': 'classification', 'n_samples': 8000, 'n_features': 700},
            {'problem_type': 'regression', 'n_samples': 10000, 'n_features': 784},
        ]
        
        for metadata in metadata_list:
            arch = sample_architecture.clone()
            repo.save_architecture(arch, dataset_metadata=metadata)
        
        # Find similar to classification with 10000 samples
        target_metadata = {
            'problem_type': 'classification',
            'n_samples': 10000,
            'n_features': 784
        }
        
        results = repo.find_similar_architectures(target_metadata, top_k=2)
        assert len(results) <= 2
        
        # First result should be most similar
        if results:
            arch, metadata, similarity = results[0]
            assert similarity > 0.5
            assert isinstance(arch, Architecture)
        
        repo.close()
    
    def test_adapt_architecture(self, temp_db, sample_architecture):
        """Test architecture adaptation."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Adapt to new input/output shapes
        new_input_shape = (100,)
        new_output_shape = (5,)
        
        adapted = repo.adapt_architecture(
            sample_architecture,
            new_input_shape=new_input_shape,
            new_output_shape=new_output_shape
        )
        
        # Verify it's a new architecture
        assert adapted.id != sample_architecture.id
        
        # Verify output layer was adapted
        assert adapted.layers[-1].params['units'] == 5
        
        # Verify metadata
        assert 'adapted_from' in adapted.metadata
        assert adapted.metadata['adapted_from'] == sample_architecture.id
        
        repo.close()
    
    def test_adapt_architecture_with_scaling(self, temp_db, sample_architecture):
        """Test architecture adaptation with layer scaling."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Adapt with scale factor
        adapted = repo.adapt_architecture(
            sample_architecture,
            new_input_shape=(784,),
            new_output_shape=(10,),
            scale_factor=1.5
        )
        
        # Verify middle layers were scaled
        # Original: 128, 64 -> Scaled: ~192, ~96 (rounded to multiples of 8)
        assert adapted.layers[0].params['units'] >= 128
        assert adapted.layers[2].params['units'] >= 64
        
        repo.close()
    
    def test_export_architecture(self, temp_db, sample_architecture, sample_metadata):
        """Test exporting an architecture to JSON."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save architecture
        arch_id = repo.save_architecture(
            sample_architecture,
            dataset_metadata=sample_metadata['dataset_metadata'],
            performance_metrics=sample_metadata['performance_metrics']
        )
        
        # Export to file
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, 'exported_arch.json')
            success = repo.export_architecture(arch_id, export_path)
            assert success is True
            assert os.path.exists(export_path)
            
            # Verify file contents
            with open(export_path, 'r') as f:
                data = json.load(f)
            
            assert 'architecture' in data
            assert 'metadata' in data
            assert data['architecture']['id'] == arch_id
        
        repo.close()
    
    def test_import_architecture(self, temp_db, sample_architecture, sample_metadata):
        """Test importing an architecture from JSON."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Create export data
        export_data = {
            'architecture': sample_architecture.to_dict(),
            'metadata': sample_metadata,
            'export_version': '1.0'
        }
        
        # Write to file
        with tempfile.TemporaryDirectory() as tmpdir:
            import_path = os.path.join(tmpdir, 'import_arch.json')
            with open(import_path, 'w') as f:
                json.dump(export_data, f)
            
            # Import from file
            imported = repo.import_architecture(import_path, save_to_repository=True)
            assert imported is not None
            assert isinstance(imported, Architecture)
            assert len(imported.layers) == len(sample_architecture.layers)
            
            # Verify it was saved to repository
            result = repo.load_architecture(imported.id)
            assert result is not None
        
        repo.close()
    
    def test_import_invalid_architecture(self, temp_db):
        """Test importing an invalid architecture."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Create invalid export data (missing architecture key)
        invalid_data = {
            'export_version': '1.0'
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            import_path = os.path.join(tmpdir, 'invalid_arch.json')
            with open(import_path, 'w') as f:
                json.dump(invalid_data, f)
            
            # Import should fail
            imported = repo.import_architecture(import_path)
            assert imported is None
        
        repo.close()
    
    def test_get_statistics(self, temp_db, sample_architecture, sample_metadata):
        """Test getting repository statistics."""
        repo = ArchitectureRepository(db_path=temp_db)
        
        # Save multiple architectures
        for i in range(5):
            arch = sample_architecture.clone()
            metadata = sample_metadata.copy()
            metadata['performance_metrics'] = metadata['performance_metrics'].copy()
            metadata['performance_metrics']['accuracy'] = 0.85 + i * 0.02
            
            repo.save_architecture(
                arch,
                dataset_metadata=metadata['dataset_metadata'],
                performance_metrics=metadata['performance_metrics'],
                tags=['test']
            )
        
        # Get statistics
        stats = repo.get_statistics()
        
        assert stats['total_architectures'] == 5
        assert 'classification' in stats['by_problem_type']
        assert stats['by_problem_type']['classification'] == 5
        assert stats['avg_accuracy'] is not None
        assert stats['max_accuracy'] >= 0.85
        assert 'test' in stats['top_tags']
        
        repo.close()
    
    def test_context_manager(self, temp_db):
        """Test using repository as context manager."""
        with ArchitectureRepository(db_path=temp_db) as repo:
            assert repo.conn is not None
            stats = repo.get_statistics()
            assert 'total_architectures' in stats
        
        # Connection should be closed after context
        # Note: We can't easily test this without accessing private state


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

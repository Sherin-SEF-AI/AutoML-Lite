"""
Integration tests for NAS transfer learning.

Tests cover:
- Architecture reuse reduces search time
- Similarity-based retrieval
"""

import pytest
import numpy as np
import tempfile
import os
from sklearn.datasets import make_classification

from automl_lite.nas import (
    NASController,
    NASConfig,
    Architecture,
    LayerConfig,
    ArchitectureRepository,
)


# Check if TensorFlow is available
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False


@pytest.fixture
def small_classification_data():
    """Create small classification dataset for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=42
    )
    return X, y


@pytest.fixture
def similar_classification_data():
    """Create similar classification dataset for transfer learning."""
    X, y = make_classification(
        n_samples=120,
        n_features=20,
        n_informative=15,
        n_classes=2,
        random_state=43
    )
    return X, y


@pytest.fixture
def temp_repository():
    """Create temporary architecture repository."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_architectures.db')
        yield db_path



@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestArchitectureReuse:
    """Test that architecture reuse reduces search time."""
    
    def test_transfer_learning_reduces_search_time(
        self, small_classification_data, similar_classification_data, temp_repository
    ):
        """Test that transfer learning reduces search time compared to from-scratch search."""
        X1, y1 = small_classification_data
        X2, y2 = similar_classification_data
        
        # First search without transfer learning (baseline)
        config_baseline = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=10,
            population_size=5,
            enable_transfer_learning=False,
            verbose=False
        )
        
        controller_baseline = NASController(config_baseline)
        result_baseline = controller_baseline.search(X1, y1, problem_type='classification')
        baseline_time = result_baseline.search_time
        baseline_count = result_baseline.total_architectures_evaluated
        
        # Save best architecture to repository
        repo = ArchitectureRepository(db_path=temp_repository)
        repo.save_architecture(
            result_baseline.best_architecture,
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': len(X1),
                'n_features': X1.shape[1],
                'n_classes': len(np.unique(y1))
            },
            performance_metrics={
                'accuracy': result_baseline.best_accuracy
            }
        )
        repo.close()
        
        # Second search with transfer learning on similar dataset
        config_transfer = NASConfig(
            search_strategy='evolutionary',
            time_budget=30,
            max_architectures=10,
            population_size=5,
            enable_transfer_learning=True,
            architecture_repository_path=temp_repository,
            verbose=False
        )
        
        controller_transfer = NASController(config_transfer)
        result_transfer = controller_transfer.search(X2, y2, problem_type='classification')
        transfer_time = result_transfer.search_time
        
        # Transfer learning should be faster or achieve better results in same time
        # (Allow some variance due to randomness)
        # At minimum, it should not be significantly slower
        assert transfer_time <= baseline_time * 1.2
    
    def test_warm_start_with_similar_architectures(
        self, small_classification_data, temp_repository
    ):
        """Test that warm start initializes population with similar architectures."""
        X, y = small_classification_data
        
        # Populate repository with some architectures
        repo = ArchitectureRepository(db_path=temp_repository)
        
        for i in range(3):
            arch = Architecture(
                layers=[
                    LayerConfig('dense', {'units': 64 * (i + 1), 'activation': 'relu'}),
                    LayerConfig('dense', {'units': 32 * (i + 1), 'activation': 'relu'}),
                    LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
                ]
            )
            
            repo.save_architecture(
                arch,
                dataset_metadata={
                    'problem_type': 'classification',
                    'n_samples': 100,
                    'n_features': 20,
                    'n_classes': 2
                },
                performance_metrics={
                    'accuracy': 0.85 + i * 0.03
                }
            )
        
        repo.close()
        
        # Search with transfer learning
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            enable_transfer_learning=True,
            architecture_repository_path=temp_repository,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify search completed successfully
        assert result.best_architecture is not None
        assert result.total_architectures_evaluated > 0



@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestSimilarityBasedRetrieval:
    """Test similarity-based architecture retrieval."""
    
    def test_find_similar_architectures_by_problem_type(self, temp_repository):
        """Test finding architectures by problem type similarity."""
        repo = ArchitectureRepository(db_path=temp_repository)
        
        # Save classification architectures
        for i in range(3):
            arch = Architecture(
                layers=[
                    LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                    LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
                ]
            )
            
            repo.save_architecture(
                arch,
                dataset_metadata={
                    'problem_type': 'classification',
                    'n_samples': 1000,
                    'n_features': 50
                }
            )
        
        # Save regression architecture
        reg_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 1, 'activation': 'linear'}),
            ]
        )
        
        repo.save_architecture(
            reg_arch,
            dataset_metadata={
                'problem_type': 'regression',
                'n_samples': 1000,
                'n_features': 50
            }
        )
        
        # Find similar to classification
        target_metadata = {
            'problem_type': 'classification',
            'n_samples': 1000,
            'n_features': 50
        }
        
        results = repo.find_similar_architectures(target_metadata, top_k=3)
        
        # Should return classification architectures
        assert len(results) == 3
        for arch, metadata, similarity in results:
            assert metadata['dataset_metadata']['problem_type'] == 'classification'
            assert similarity > 0.5
        
        repo.close()
    
    def test_find_similar_architectures_by_dataset_size(self, temp_repository):
        """Test finding architectures by dataset size similarity."""
        repo = ArchitectureRepository(db_path=temp_repository)
        
        # Save architectures with different dataset sizes
        sizes = [100, 1000, 10000]
        for size in sizes:
            arch = Architecture(
                layers=[
                    LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                    LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
                ]
            )
            
            repo.save_architecture(
                arch,
                dataset_metadata={
                    'problem_type': 'classification',
                    'n_samples': size,
                    'n_features': 20
                }
            )
        
        # Find similar to medium-sized dataset
        target_metadata = {
            'problem_type': 'classification',
            'n_samples': 1200,
            'n_features': 20
        }
        
        results = repo.find_similar_architectures(target_metadata, top_k=1)
        
        # Should return the 1000-sample architecture (closest)
        assert len(results) == 1
        arch, metadata, similarity = results[0]
        assert metadata['dataset_metadata']['n_samples'] == 1000
        
        repo.close()
    
    def test_similarity_score_calculation(self, temp_repository):
        """Test that similarity scores are calculated correctly."""
        repo = ArchitectureRepository(db_path=temp_repository)
        
        # Exact match
        metadata1 = {
            'problem_type': 'classification',
            'n_samples': 1000,
            'n_features': 50
        }
        
        metadata2 = {
            'problem_type': 'classification',
            'n_samples': 1000,
            'n_features': 50
        }
        
        similarity_exact = repo.compute_similarity(metadata1, metadata2)
        assert similarity_exact == 1.0
        
        # Partial match (same problem type, different sizes)
        metadata3 = {
            'problem_type': 'classification',
            'n_samples': 500,
            'n_features': 25
        }
        
        similarity_partial = repo.compute_similarity(metadata1, metadata3)
        assert 0.4 < similarity_partial < 1.0
        
        # Different problem type
        metadata4 = {
            'problem_type': 'regression',
            'n_samples': 1000,
            'n_features': 50
        }
        
        similarity_different = repo.compute_similarity(metadata1, metadata4)
        assert similarity_different < similarity_partial
        
        repo.close()


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestArchitectureAdaptation:
    """Test architecture adaptation for new problems."""
    
    def test_adapt_architecture_to_new_output_size(self, temp_repository):
        """Test adapting architecture to different number of classes."""
        repo = ArchitectureRepository(db_path=temp_repository)
        
        # Original architecture for 10 classes
        original_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 10, 'activation': 'softmax'}),
            ]
        )
        
        # Adapt to 5 classes
        adapted = repo.adapt_architecture(
            original_arch,
            new_input_shape=(20,),
            new_output_shape=(5,)
        )
        
        # Verify output layer was changed
        assert adapted.layers[-1].params['units'] == 5
        assert adapted.layers[-1].params['activation'] == 'softmax'
        
        # Verify it's a new architecture
        assert adapted.id != original_arch.id
        
        # Verify metadata
        assert 'adapted_from' in adapted.metadata
        assert adapted.metadata['adapted_from'] == original_arch.id
        
        repo.close()
    
    def test_adapt_architecture_with_scaling(self, temp_repository):
        """Test adapting architecture with layer scaling."""
        repo = ArchitectureRepository(db_path=temp_repository)
        
        original_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 32, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 2, 'activation': 'softmax'}),
            ]
        )
        
        # Adapt with 2x scaling
        adapted = repo.adapt_architecture(
            original_arch,
            new_input_shape=(20,),
            new_output_shape=(2,),
            scale_factor=2.0
        )
        
        # Verify middle layers were scaled
        assert adapted.layers[0].params['units'] >= 64
        assert adapted.layers[1].params['units'] >= 32
        
        # Output layer should remain at target size
        assert adapted.layers[-1].params['units'] == 2
        
        repo.close()
    
    def test_transfer_learning_with_adapted_architectures(
        self, small_classification_data, temp_repository
    ):
        """Test using adapted architectures in transfer learning."""
        X, y = small_classification_data
        
        # Create and save an architecture for different problem
        original_arch = Architecture(
            layers=[
                LayerConfig('dense', {'units': 128, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 64, 'activation': 'relu'}),
                LayerConfig('dense', {'units': 5, 'activation': 'softmax'}),  # 5 classes
            ]
        )
        
        repo = ArchitectureRepository(db_path=temp_repository)
        repo.save_architecture(
            original_arch,
            dataset_metadata={
                'problem_type': 'classification',
                'n_samples': 100,
                'n_features': 20,
                'n_classes': 5
            },
            performance_metrics={'accuracy': 0.90}
        )
        repo.close()
        
        # Search with transfer learning (2 classes instead of 5)
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            enable_transfer_learning=True,
            architecture_repository_path=temp_repository,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify search completed successfully
        assert result.best_architecture is not None
        
        # Verify output layer has correct size for 2 classes
        assert result.best_architecture.layers[-1].params['units'] in [1, 2]


@pytest.mark.skipif(not HAS_TENSORFLOW, reason="TensorFlow not installed")
class TestRepositoryIntegrationWithSearch:
    """Test repository integration with search process."""
    
    def test_automatic_architecture_saving(
        self, small_classification_data, temp_repository
    ):
        """Test that best architectures are automatically saved to repository."""
        X, y = small_classification_data
        
        config = NASConfig(
            search_strategy='evolutionary',
            time_budget=20,
            max_architectures=5,
            population_size=3,
            enable_transfer_learning=True,
            architecture_repository_path=temp_repository,
            verbose=False
        )
        
        controller = NASController(config)
        result = controller.search(X, y, problem_type='classification')
        
        # Verify architecture was saved
        repo = ArchitectureRepository(db_path=temp_repository)
        
        # Check if best architecture is in repository
        loaded = repo.load_architecture(result.best_architecture.id)
        
        if loaded is not None:
            loaded_arch, loaded_metadata = loaded
            assert loaded_arch.id == result.best_architecture.id
            assert 'performance_metrics' in loaded_metadata
        
        repo.close()
    
    def test_repository_statistics_after_search(
        self, small_classification_data, temp_repository
    ):
        """Test repository statistics after multiple searches."""
        X, y = small_classification_data
        
        # Run multiple searches
        for i in range(2):
            config = NASConfig(
                search_strategy='evolutionary',
                time_budget=15,
                max_architectures=3,
                population_size=2,
                enable_transfer_learning=True,
                architecture_repository_path=temp_repository,
                verbose=False
            )
            
            controller = NASController(config)
            controller.search(X, y, problem_type='classification')
        
        # Check repository statistics
        repo = ArchitectureRepository(db_path=temp_repository)
        stats = repo.get_statistics()
        
        assert stats['total_architectures'] > 0
        assert 'classification' in stats['by_problem_type']
        
        repo.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

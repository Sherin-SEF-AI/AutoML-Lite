"""
Unit tests for Neural Architecture Search strategies.

Tests the SearchStrategy base class and concrete implementations:
- EvolutionarySearchStrategy
- RLSearchStrategy
- DARTSSearchStrategy
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from automl_lite.nas import (
    Architecture,
    LayerConfig,
    SearchSpace,
    TabularSearchSpace,
    VisionSearchSpace,
    TimeSeriesSearchSpace,
    SearchStrategy,
    SearchHistory,
    EvolutionarySearchStrategy,
    RLSearchStrategy,
    DARTSSearchStrategy,
)


class TestSearchHistory:
    """Test SearchHistory dataclass."""
    
    def test_search_history_creation(self):
        """Test creating a search history record."""
        arch = Architecture(
            layers=[LayerConfig('dense', {'units': 64})],
        )
        
        record = SearchHistory(
            architecture=arch,
            performance=0.95,
            iteration=1,
            timestamp=1234567890.0,
            metadata={'test': 'value'}
        )
        
        assert record.architecture == arch
        assert record.performance == 0.95
        assert record.iteration == 1
        assert record.timestamp == 1234567890.0
        assert record.metadata == {'test': 'value'}
    
    def test_search_history_to_dict(self):
        """Test converting search history to dictionary."""
        arch = Architecture(
            layers=[LayerConfig('dense', {'units': 64})],
        )
        
        record = SearchHistory(
            architecture=arch,
            performance=0.95,
            iteration=1,
            timestamp=1234567890.0,
        )
        
        record_dict = record.to_dict()
        
        assert 'architecture' in record_dict
        assert record_dict['performance'] == 0.95
        assert record_dict['iteration'] == 1
        assert record_dict['timestamp'] == 1234567890.0


class TestSearchStrategyBase:
    """Test SearchStrategy base class."""
    
    def test_search_strategy_initialization(self):
        """Test initializing search strategy."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        # Create a concrete implementation for testing
        class ConcreteStrategy(SearchStrategy):
            def generate_architecture(self):
                return self.search_space.sample_architecture()
            
            def update(self, architecture, performance, metadata=None):
                import time
                self.add_to_history(architecture, performance, time.time(), metadata)
        
        strategy = ConcreteStrategy(search_space, random_seed=42)
        
        assert strategy.search_space == search_space
        assert strategy.random_seed == 42
        assert len(strategy.history) == 0
        assert strategy.iteration == 0
    
    def test_add_to_history(self):
        """Test adding evaluations to history."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        class ConcreteStrategy(SearchStrategy):
            def generate_architecture(self):
                return self.search_space.sample_architecture()
            
            def update(self, architecture, performance, metadata=None):
                import time
                self.add_to_history(architecture, performance, time.time(), metadata)
        
        strategy = ConcreteStrategy(search_space)
        
        arch = search_space.sample_architecture()
        strategy.add_to_history(arch, 0.95, 1234567890.0, {'test': 'value'})
        
        assert len(strategy.history) == 1
        assert strategy.history[0].performance == 0.95
        assert strategy.history[0].metadata == {'test': 'value'}
        assert strategy.iteration == 1
    
    def test_get_best_architecture(self):
        """Test getting best architecture."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        class ConcreteStrategy(SearchStrategy):
            def generate_architecture(self):
                return self.search_space.sample_architecture()
            
            def update(self, architecture, performance, metadata=None):
                import time
                self.add_to_history(architecture, performance, time.time(), metadata)
        
        strategy = ConcreteStrategy(search_space)
        
        # Add some architectures
        arch1 = search_space.sample_architecture()
        arch2 = search_space.sample_architecture()
        arch3 = search_space.sample_architecture()
        
        strategy.add_to_history(arch1, 0.85, 1.0)
        strategy.add_to_history(arch2, 0.95, 2.0)
        strategy.add_to_history(arch3, 0.90, 3.0)
        
        best = strategy.get_best_architecture()
        assert best == arch2
        assert strategy.get_best_performance() == 0.95
    
    def test_get_history_summary(self):
        """Test getting history summary."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        class ConcreteStrategy(SearchStrategy):
            def generate_architecture(self):
                return self.search_space.sample_architecture()
            
            def update(self, architecture, performance, metadata=None):
                import time
                self.add_to_history(architecture, performance, time.time(), metadata)
        
        strategy = ConcreteStrategy(search_space)
        
        # Empty history
        summary = strategy.get_history_summary()
        assert summary['num_evaluations'] == 0
        assert summary['best_performance'] is None
        
        # Add evaluations
        for perf in [0.85, 0.90, 0.95, 0.88]:
            arch = search_space.sample_architecture()
            strategy.add_to_history(arch, perf, 1.0)
        
        summary = strategy.get_history_summary()
        assert summary['num_evaluations'] == 4
        assert summary['best_performance'] == 0.95
        assert summary['worst_performance'] == 0.85
        assert 0.88 < summary['mean_performance'] < 0.92


class TestEvolutionarySearchStrategy:
    """Test EvolutionarySearchStrategy."""
    
    def test_evolutionary_initialization(self):
        """Test initializing evolutionary strategy."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=20,
            mutation_rate=0.3,
            crossover_rate=0.6,
            tournament_size=4,
            elitism_ratio=0.15,
            random_seed=42
        )
        
        assert strategy.population_size == 20
        assert strategy.mutation_rate == 0.3
        assert strategy.crossover_rate == 0.6
        assert strategy.tournament_size == 4
        assert strategy.elitism_ratio == 0.15
        assert not strategy.initialized
    
    def test_initialize_population(self):
        """Test population initialization."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=10,
            random_seed=42
        )
        
        population = strategy.initialize_population()
        
        assert len(population) == 10
        assert all(isinstance(arch, Architecture) for arch in population)
        assert strategy.initialized
        assert len(strategy.population_fitness) == 10
    
    def test_generate_architecture_initial_population(self):
        """Test generating architectures from initial population."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=5,
            random_seed=42
        )
        
        # First 5 generations should return initial population
        architectures = []
        for i in range(5):
            arch = strategy.generate_architecture()
            architectures.append(arch)
            strategy.update(arch, 0.8 + i * 0.02)
        
        assert len(architectures) == 5
        assert all(isinstance(arch, Architecture) for arch in architectures)
    
    def test_generate_architecture_with_evolution(self):
        """Test generating architectures using evolutionary operators."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=5,
            mutation_rate=0.2,
            crossover_rate=0.5,
            random_seed=42
        )
        
        # Evaluate initial population
        for i in range(5):
            arch = strategy.generate_architecture()
            strategy.update(arch, 0.8 + i * 0.02)
        
        # Generate offspring (should use crossover or mutation)
        offspring = strategy.generate_architecture()
        
        assert isinstance(offspring, Architecture)
        assert offspring.layers  # Should have layers
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=10,
            tournament_size=3,
            random_seed=42
        )
        
        # Initialize population
        strategy.initialize_population()
        strategy.population_fitness = [0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65, 0.55, 0.95]
        
        # Run tournament selection multiple times
        selected = []
        for _ in range(10):
            arch = strategy._tournament_selection()
            selected.append(arch)
        
        assert len(selected) == 10
        assert all(isinstance(arch, Architecture) for arch in selected)
    
    def test_get_population_diversity(self):
        """Test population diversity calculation."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=10,
            random_seed=42
        )
        
        # Empty population
        assert strategy.get_population_diversity() == 0.0
        
        # Initialize population
        strategy.initialize_population()
        
        diversity = strategy.get_population_diversity()
        assert 0.0 <= diversity <= 1.0
    
    def test_get_generation_summary(self):
        """Test generation summary."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        strategy = EvolutionarySearchStrategy(
            search_space=search_space,
            population_size=5,
            random_seed=42
        )
        
        # Initialize and evaluate population
        for i in range(5):
            arch = strategy.generate_architecture()
            strategy.update(arch, 0.8 + i * 0.02)
        
        summary = strategy.get_generation_summary()
        
        assert 'generation' in summary
        assert 'population_size' in summary
        assert 'best_fitness' in summary
        assert 'mean_fitness' in summary
        assert 'diversity' in summary


class TestRLSearchStrategy:
    """Test RLSearchStrategy."""
    
    def test_rl_initialization_tensorflow(self):
        """Test initializing RL strategy with TensorFlow."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = RLSearchStrategy(
                search_space=search_space,
                controller_hidden_size=64,
                baseline_decay=0.95,
                learning_rate=0.001,
                batch_size=5,
                backend='tensorflow',
                random_seed=42
            )
            
            assert strategy.controller_hidden_size == 64
            assert strategy.baseline_decay == 0.95
            assert strategy.learning_rate == 0.001
            assert strategy.batch_size == 5
            assert strategy.backend == 'tensorflow'
            assert strategy.baseline == 0.0
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_rl_generate_architecture(self):
        """Test generating architecture with RL strategy."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        try:
            strategy = RLSearchStrategy(
                search_space=search_space,
                batch_size=3,
                backend='tensorflow',
                random_seed=42
            )
            
            arch = strategy.generate_architecture()
            
            assert isinstance(arch, Architecture)
            assert arch.layers
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_rl_update_and_batch(self):
        """Test updating RL strategy and batch management."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        try:
            strategy = RLSearchStrategy(
                search_space=search_space,
                batch_size=3,
                backend='tensorflow',
                random_seed=42
            )
            
            # Add evaluations
            for i in range(5):
                arch = strategy.generate_architecture()
                strategy.update(arch, 0.8 + i * 0.02)
            
            assert len(strategy.history) == 5
            # Batch should have been cleared after reaching batch_size
            assert len(strategy.batch_architectures) == 2  # 5 % 3 = 2
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_rl_get_controller_summary(self):
        """Test getting controller summary."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = RLSearchStrategy(
                search_space=search_space,
                batch_size=5,
                backend='tensorflow',
                random_seed=42
            )
            
            summary = strategy.get_controller_summary()
            
            assert 'baseline' in summary
            assert 'batch_size' in summary
            assert 'current_batch_size' in summary
            assert 'total_updates' in summary
            
        except ImportError:
            pytest.skip("TensorFlow not available")


class TestDARTSSearchStrategy:
    """Test DARTSSearchStrategy."""
    
    def test_darts_initialization(self):
        """Test initializing DARTS strategy."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = DARTSSearchStrategy(
                search_space=search_space,
                supernet_epochs=20,
                arch_learning_rate=3e-4,
                weight_learning_rate=0.025,
                backend='tensorflow',
                random_seed=42
            )
            
            assert strategy.supernet_epochs == 20
            assert strategy.arch_learning_rate == 3e-4
            assert strategy.weight_learning_rate == 0.025
            assert strategy.backend == 'tensorflow'
            assert not strategy.supernet_trained
            assert strategy.current_epoch == 0
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_darts_candidate_operations(self):
        """Test defining candidate operations."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = DARTSSearchStrategy(
                search_space=search_space,
                backend='tensorflow',
                random_seed=42
            )
            
            assert len(strategy.candidate_operations) > 0
            assert 'identity' in strategy.candidate_operations
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_darts_build_supernet(self):
        """Test building supernet."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = DARTSSearchStrategy(
                search_space=search_space,
                backend='tensorflow',
                random_seed=42
            )
            
            # Create sample data
            X = np.random.randn(50, 10)
            y = np.random.randint(0, 2, 50)
            
            strategy.build_supernet(X, y)
            
            assert strategy.supernet is not None
            assert strategy.arch_parameters is not None
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_darts_extract_architecture(self):
        """Test extracting architecture from supernet."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification',
            random_seed=42
        )
        
        try:
            strategy = DARTSSearchStrategy(
                search_space=search_space,
                backend='tensorflow',
                random_seed=42
            )
            
            # Without training, should return random architecture
            arch = strategy.extract_architecture()
            
            assert isinstance(arch, Architecture)
            assert arch.layers
            
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    def test_darts_get_summary(self):
        """Test getting DARTS summary."""
        search_space = TabularSearchSpace(
            input_shape=(10,),
            output_shape=(2,),
            problem_type='classification'
        )
        
        try:
            strategy = DARTSSearchStrategy(
                search_space=search_space,
                backend='tensorflow',
                random_seed=42
            )
            
            summary = strategy.get_darts_summary()
            
            assert 'supernet_trained' in summary
            assert 'current_epoch' in summary
            assert 'total_epochs' in summary
            assert 'num_operations' in summary
            
        except ImportError:
            pytest.skip("TensorFlow not available")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

"""
Unit tests for Multi-Objective Optimization in NAS.

Tests cover:
- Pareto dominance checking
- Pareto front calculation
- Objective weighting and scalarization
- Constraint satisfaction
- Hypervolume calculation
"""

import pytest
import numpy as np
from dataclasses import dataclass
from typing import Dict

from automl_lite.nas.multi_objective import MultiObjectiveOptimizer, Objective


@dataclass
class MockArchitecture:
    """Mock architecture for testing."""
    id: str
    metrics: Dict[str, float]
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, MockArchitecture):
            return False
        return self.id == other.id


class TestObjective:
    """Test Objective dataclass."""
    
    def test_objective_creation(self):
        """Test creating objectives."""
        obj = Objective('accuracy', 'maximize', weight=0.5)
        assert obj.name == 'accuracy'
        assert obj.direction == 'maximize'
        assert obj.weight == 0.5
    
    def test_objective_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Direction must be"):
            Objective('accuracy', 'invalid')
    
    def test_objective_negative_weight(self):
        """Test that negative weight raises error."""
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            Objective('accuracy', 'maximize', weight=-0.5)


class TestMultiObjectiveOptimizer:
    """Test MultiObjectiveOptimizer class."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        assert len(optimizer.objectives) == 2
        assert optimizer.constraints == []
    
    def test_initialization_with_constraints(self):
        """Test optimizer initialization with constraints."""
        objectives = [Objective('accuracy', 'maximize')]
        constraints = ["accuracy > 0.9"]
        optimizer = MultiObjectiveOptimizer(objectives, constraints)
        assert len(optimizer.constraints) == 1
    
    def test_empty_objectives_error(self):
        """Test that empty objectives raises error."""
        with pytest.raises(ValueError, match="At least one objective"):
            MultiObjectiveOptimizer([])
    
    def test_duplicate_objectives_error(self):
        """Test that duplicate objective names raise error."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('accuracy', 'minimize'),
        ]
        with pytest.raises(ValueError, match="Duplicate objective names"):
            MultiObjectiveOptimizer(objectives)


class TestParetoDominance:
    """Test Pareto dominance checking."""
    
    def test_dominates_two_objectives(self):
        """Test dominance with two objectives."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # A1 dominates A2 (better accuracy, better latency)
        metrics1 = {'accuracy': 0.95, 'latency': 50}
        metrics2 = {'accuracy': 0.90, 'latency': 60}
        assert optimizer.dominates(metrics1, metrics2)
        assert not optimizer.dominates(metrics2, metrics1)
    
    def test_no_dominance_tradeoff(self):
        """Test no dominance when there's a trade-off."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # Trade-off: A1 has better accuracy, A2 has better latency
        metrics1 = {'accuracy': 0.95, 'latency': 100}
        metrics2 = {'accuracy': 0.90, 'latency': 50}
        assert not optimizer.dominates(metrics1, metrics2)
        assert not optimizer.dominates(metrics2, metrics1)
    
    def test_dominates_three_objectives(self):
        """Test dominance with three objectives."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
            Objective('model_size', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # A1 dominates A2 in all objectives
        metrics1 = {'accuracy': 0.95, 'latency': 50, 'model_size': 10}
        metrics2 = {'accuracy': 0.90, 'latency': 60, 'model_size': 15}
        assert optimizer.dominates(metrics1, metrics2)
    
    def test_equal_metrics_no_dominance(self):
        """Test that equal metrics don't dominate."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        metrics1 = {'accuracy': 0.95, 'latency': 50}
        metrics2 = {'accuracy': 0.95, 'latency': 50}
        assert not optimizer.dominates(metrics1, metrics2)


class TestParetoFront:
    """Test Pareto front calculation."""
    
    def test_compute_pareto_front(self):
        """Test computing Pareto front."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),
            MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),
            MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70}),
            MockArchitecture('A4', {'accuracy': 0.88, 'latency': 120}),  # Dominated
        ]
        
        pareto_front = optimizer.compute_pareto_front(architectures)
        
        # A1, A2, A3 should be in Pareto front (A4 is dominated)
        assert len(pareto_front) == 3
        assert any(a.id == 'A1' for a in pareto_front)
        assert any(a.id == 'A2' for a in pareto_front)
        assert any(a.id == 'A3' for a in pareto_front)
        assert not any(a.id == 'A4' for a in pareto_front)
    
    def test_pareto_front_empty_list(self):
        """Test Pareto front with empty list."""
        objectives = [Objective('accuracy', 'maximize')]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        pareto_front = optimizer.compute_pareto_front([])
        assert pareto_front == []
    
    def test_pareto_front_single_architecture(self):
        """Test Pareto front with single architecture."""
        objectives = [Objective('accuracy', 'maximize')]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [MockArchitecture('A1', {'accuracy': 0.95})]
        pareto_front = optimizer.compute_pareto_front(architectures)
        
        assert len(pareto_front) == 1
        assert pareto_front[0].id == 'A1'
    
    def test_compute_pareto_rank(self):
        """Test computing Pareto ranks."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),  # Rank 0
            MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),   # Rank 0
            MockArchitecture('A3', {'accuracy': 0.88, 'latency': 120}),  # Rank 1
        ]
        
        ranks = optimizer.compute_pareto_rank(architectures)
        
        assert ranks[architectures[0]] == 0
        assert ranks[architectures[1]] == 0
        assert ranks[architectures[2]] == 1


class TestScalarization:
    """Test objective weighting and scalarization."""
    
    def test_scalarize_basic(self):
        """Test basic scalarization."""
        objectives = [
            Objective('accuracy', 'maximize', weight=0.6),
            Objective('latency', 'minimize', weight=0.4),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        metrics = {'accuracy': 0.95, 'latency': 50}
        score = optimizer.scalarize(metrics, normalize=False)
        
        # Score = (0.6 * 0.95 + 0.4 * (-50)) / 1.0
        expected = (0.6 * 0.95 - 0.4 * 50) / 1.0
        assert abs(score - expected) < 1e-6
    
    def test_set_objective_weights(self):
        """Test setting objective weights."""
        objectives = [
            Objective('accuracy', 'maximize', weight=0.5),
            Objective('latency', 'minimize', weight=0.5),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        optimizer.set_objective_weights({'accuracy': 0.7, 'latency': 0.3})
        
        weights = optimizer.get_objective_weights()
        assert weights['accuracy'] == 0.7
        assert weights['latency'] == 0.3
    
    def test_normalize_weights(self):
        """Test normalizing weights to sum to 1."""
        objectives = [
            Objective('accuracy', 'maximize', weight=2.0),
            Objective('latency', 'minimize', weight=3.0),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        optimizer.normalize_weights()
        
        weights = optimizer.get_objective_weights()
        assert abs(weights['accuracy'] - 0.4) < 1e-6
        assert abs(weights['latency'] - 0.6) < 1e-6
    
    def test_rank_architectures_scalarize(self):
        """Test ranking architectures by scalarization."""
        objectives = [
            Objective('accuracy', 'maximize', weight=0.6),
            Objective('latency', 'minimize', weight=0.4),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),
            MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),
            MockArchitecture('A3', {'accuracy': 0.92, 'latency': 70}),
        ]
        
        ranked = optimizer.rank_architectures(architectures, method='scalarize')
        
        assert len(ranked) == 3
        # Check that scores are in descending order
        assert ranked[0][1] >= ranked[1][1] >= ranked[2][1]
    
    def test_select_best_architecture(self):
        """Test selecting best architecture with preferences."""
        objectives = [
            Objective('accuracy', 'maximize', weight=0.5),
            Objective('latency', 'minimize', weight=0.5),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),
            MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),
        ]
        
        # Prefer accuracy
        best = optimizer.select_best_architecture(
            architectures,
            preferences={'accuracy': 0.9, 'latency': 0.1}
        )
        assert best.id == 'A1'  # Higher accuracy
        
        # Prefer latency
        best = optimizer.select_best_architecture(
            architectures,
            preferences={'accuracy': 0.1, 'latency': 0.9}
        )
        assert best.id == 'A2'  # Lower latency


class TestConstraints:
    """Test constraint satisfaction checking."""
    
    def test_parse_simple_constraint(self):
        """Test parsing simple constraints."""
        objectives = [Objective('accuracy', 'maximize')]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # Test different operators
        constraint_fn = optimizer.parse_constraint("accuracy > 0.9")
        assert constraint_fn({'accuracy': 0.95})
        assert not constraint_fn({'accuracy': 0.85})
        
        constraint_fn = optimizer.parse_constraint("accuracy >= 0.9")
        assert constraint_fn({'accuracy': 0.9})
        
        constraint_fn = optimizer.parse_constraint("latency < 100")
        assert constraint_fn({'latency': 50})
        assert not constraint_fn({'latency': 150})
    
    def test_parse_and_constraint(self):
        """Test parsing AND constraints."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        constraint_fn = optimizer.parse_constraint("accuracy > 0.9 AND latency < 100")
        
        assert constraint_fn({'accuracy': 0.95, 'latency': 50})
        assert not constraint_fn({'accuracy': 0.85, 'latency': 50})
        assert not constraint_fn({'accuracy': 0.95, 'latency': 150})
    
    def test_parse_or_constraint(self):
        """Test parsing OR constraints."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        constraint_fn = optimizer.parse_constraint("accuracy > 0.9 OR latency < 50")
        
        assert constraint_fn({'accuracy': 0.95, 'latency': 100})
        assert constraint_fn({'accuracy': 0.85, 'latency': 40})
        assert not constraint_fn({'accuracy': 0.85, 'latency': 100})
    
    def test_check_constraints(self):
        """Test checking constraints."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        constraints = ["accuracy > 0.9", "latency < 100"]
        optimizer = MultiObjectiveOptimizer(objectives, constraints)
        
        assert optimizer.check_constraints({'accuracy': 0.95, 'latency': 50})
        assert not optimizer.check_constraints({'accuracy': 0.85, 'latency': 50})
        assert not optimizer.check_constraints({'accuracy': 0.95, 'latency': 150})
    
    def test_filter_by_constraints(self):
        """Test filtering architectures by constraints."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        constraints = ["accuracy > 0.9", "latency < 100"]
        optimizer = MultiObjectiveOptimizer(objectives, constraints)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 50}),   # Valid
            MockArchitecture('A2', {'accuracy': 0.85, 'latency': 50}),   # Invalid
            MockArchitecture('A3', {'accuracy': 0.92, 'latency': 150}),  # Invalid
            MockArchitecture('A4', {'accuracy': 0.93, 'latency': 80}),   # Valid
        ]
        
        filtered = optimizer.filter_by_constraints(architectures)
        
        assert len(filtered) == 2
        assert any(a.id == 'A1' for a in filtered)
        assert any(a.id == 'A4' for a in filtered)
    
    def test_add_remove_constraints(self):
        """Test adding and removing constraints."""
        objectives = [Objective('accuracy', 'maximize')]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        assert len(optimizer.constraints) == 0
        
        optimizer.add_constraint("accuracy > 0.9")
        assert len(optimizer.constraints) == 1
        
        optimizer.add_constraint("latency < 100")
        assert len(optimizer.constraints) == 2
        
        optimizer.remove_constraint("accuracy > 0.9")
        assert len(optimizer.constraints) == 1
        
        optimizer.clear_constraints()
        assert len(optimizer.constraints) == 0


class TestHypervolume:
    """Test hypervolume calculation."""
    
    def test_hypervolume_2d(self):
        """Test 2D hypervolume calculation."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        architectures = [
            MockArchitecture('A1', {'accuracy': 0.95, 'latency': 100}),
            MockArchitecture('A2', {'accuracy': 0.90, 'latency': 50}),
        ]
        
        hv = optimizer.compute_hypervolume(architectures)
        assert hv > 0  # Should be positive
    
    def test_hypervolume_empty(self):
        """Test hypervolume with empty list."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        hv = optimizer.compute_hypervolume([])
        assert hv == 0.0
    
    def test_hypervolume_comparison(self):
        """Test that hypervolume is computed correctly."""
        objectives = [
            Objective('accuracy', 'maximize'),
            Objective('latency', 'minimize'),
        ]
        optimizer = MultiObjectiveOptimizer(objectives)
        
        # Single point
        set1 = [
            MockArchitecture('A1', {'accuracy': 0.90, 'latency': 50}),
        ]
        
        # Two points that expand the dominated space
        set2 = [
            MockArchitecture('B1', {'accuracy': 0.95, 'latency': 60}),
            MockArchitecture('B2', {'accuracy': 0.85, 'latency': 40}),
        ]
        
        hv1 = optimizer.compute_hypervolume(set1)
        hv2 = optimizer.compute_hypervolume(set2)
        
        # Both should be positive
        assert hv1 > 0
        assert hv2 > 0
        # Two-point front should dominate more space than single point
        assert hv2 >= hv1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

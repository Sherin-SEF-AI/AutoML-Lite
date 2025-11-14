"""
NASController - Orchestrates Neural Architecture Search.

This module provides the main controller that coordinates all NAS components
including search space, search strategy, performance estimation, hardware profiling,
and multi-objective optimization.
"""

import time
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

from .architecture import Architecture, NASConfig, NASResult
from .validators import ArchitectureValidator
from .search_space import SearchSpace, TabularSearchSpace, VisionSearchSpace, TimeSeriesSearchSpace
from .performance_estimator import (
    PerformanceEstimator,
    EarlyStoppingEstimator,
    LearningCurveEstimator,
    WeightSharingEstimator,
)
from .search_strategy import (
    SearchStrategy,
    EvolutionarySearchStrategy,
    RLSearchStrategy,
    DARTSSearchStrategy,
)
from .hardware_profiler import HardwareProfiler, HardwareConstraintChecker
from .multi_objective import MultiObjectiveOptimizer
from .repository import ArchitectureRepository


class NASController:
    """
    Main controller for Neural Architecture Search.
    
    Orchestrates the complete NAS workflow including:
    - Search space definition
    - Architecture generation via search strategy
    - Performance estimation
    - Hardware profiling and constraint checking
    - Multi-objective optimization
    - Checkpointing and resume
    """
    
    def __init__(
        self,
        config: NASConfig,
        experiment_tracker: Optional[Any] = None
    ):
        """
        Initialize NAS Controller.
        
        Args:
            config: NAS configuration
            experiment_tracker: Optional experiment tracker for logging
        """
        self.config = config
        self.experiment_tracker = experiment_tracker
        
        # Components (initialized during search)
        self.search_space: Optional[SearchSpace] = None
        self.search_strategy: Optional[SearchStrategy] = None
        self.performance_estimator: Optional[PerformanceEstimator] = None
        self.hardware_profiler: Optional[HardwareProfiler] = None
        self.hardware_constraint_checker: Optional[HardwareConstraintChecker] = None
        self.multi_objective_optimizer: Optional[MultiObjectiveOptimizer] = None
        self.repository: Optional[ArchitectureRepository] = None
        self.validator: Optional[ArchitectureValidator] = None
        
        # Search state
        self.evaluated_architectures: List[Architecture] = []
        self.iteration = 0
        self.start_time: Optional[float] = None
        self.error_count = 0
        self.consecutive_failures = 0
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate NAS configuration."""
        if self.config.enable_hardware_aware:
            if (self.config.max_latency_ms is None and 
                self.config.max_memory_mb is None and 
                self.config.max_model_size_mb is None):
                warnings.warn(
                    "Hardware-aware search enabled but no constraints specified. "
                    "Consider setting max_latency_ms, max_memory_mb, or max_model_size_mb."
                )
    
    def search(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        problem_type: str = 'classification'
    ) -> NASResult:
        """Run neural architecture search."""
        self.start_time = time.time()
        
        if self.config.verbose:
            print(f"Starting NAS with {self.config.search_strategy} strategy...")
        
        self._initialize_components(X, y, problem_type)
        
        while self._should_continue_search():
            self.iteration += 1
            
            try:
                architecture = self.search_strategy.generate_architecture()
                success = self._evaluate_architecture(architecture, X, y, problem_type)
                
                if success:
                    self.evaluated_architectures.append(architecture)
                    self.consecutive_failures = 0
                    self.search_strategy.update([architecture])
                else:
                    self.consecutive_failures += 1
                    self.error_count += 1
                
                if self.consecutive_failures >= 5:
                    self._fallback_to_random_search()
                    self.consecutive_failures = 0
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"Error in iteration {self.iteration}: {e}")
                self.error_count += 1
                continue
        
        return self._aggregate_results(problem_type)
    
    def resume_search(self, checkpoint_path: str, X, y, problem_type: str = 'classification') -> NASResult:
        """Resume search from checkpoint."""
        self._load_checkpoint(checkpoint_path)
        return self.search(X, y, problem_type)
    
    def get_best_architectures(self, top_k: int = 5) -> List[Architecture]:
        """Get top-k best architectures."""
        return self._rank_architectures()[:top_k]
    
    def get_pareto_front(self) -> List[Architecture]:
        """Get Pareto front for multi-objective optimization."""
        if self.multi_objective_optimizer is None:
            return []
        return self.multi_objective_optimizer.compute_pareto_front(self.evaluated_architectures)
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        if not self.evaluated_architectures:
            return {}
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        ranked = self._rank_architectures()
        best_perf = 0.0
        if ranked:
            best_arch = ranked[0]
            best_perf = best_arch.get_performance_metric('accuracy') or best_arch.get_performance_metric('mse') or 0.0
        
        return {
            'iteration': self.iteration,
            'architectures_evaluated': len(self.evaluated_architectures),
            'error_count': self.error_count,
            'success_rate': len(self.evaluated_architectures) / max(1, self.iteration),
            'best_performance': best_perf,
            'elapsed_time_seconds': elapsed_time,
        }
    
    def print_search_progress(self) -> None:
        """Print formatted search progress."""
        stats = self.get_search_statistics()
        if not stats:
            print("No search statistics available")
            return
        
        print("\n" + "=" * 60)
        print("NAS Search Progress")
        print("=" * 60)
        print(f"Strategy: {self.config.search_strategy}")
        print(f"Iterations: {stats['iteration']}")
        print(f"Architectures Evaluated: {stats['architectures_evaluated']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")
        print(f"Best Performance: {stats['best_performance']:.4f}")
        print(f"Elapsed Time: {stats['elapsed_time_seconds']:.1f}s")
        print("=" * 60)
    
    def _initialize_components(self, X, y, problem_type: str) -> None:
        """Initialize NAS components."""
        input_shape = self._infer_input_shape(X)
        output_shape = self._infer_output_shape(y, problem_type)
        
        self.validator = ArchitectureValidator()
        
        if self.config.search_space_type == 'tabular':
            self.search_space = TabularSearchSpace(input_shape, output_shape)
        elif self.config.search_space_type == 'vision':
            self.search_space = VisionSearchSpace(input_shape, output_shape)
        elif self.config.search_space_type == 'timeseries':
            self.search_space = TimeSeriesSearchSpace(input_shape, output_shape)
        else:
            self.search_space = TabularSearchSpace(input_shape, output_shape)
        
        if self.config.search_strategy == 'evolutionary':
            self.search_strategy = EvolutionarySearchStrategy(
                self.search_space,
                population_size=self.config.evolution_population_size,
                mutation_rate=self.config.evolution_mutation_rate
            )
        elif self.config.search_strategy == 'rl':
            self.search_strategy = RLSearchStrategy(self.search_space)
        elif self.config.search_strategy == 'darts':
            self.search_strategy = DARTSSearchStrategy(self.search_space)
        
        self.performance_estimator = EarlyStoppingEstimator(
            budget_fraction=self.config.estimation_budget_fraction,
            patience=self.config.early_stopping_patience
        )
        
        if self.config.enable_hardware_aware:
            self.hardware_profiler = HardwareProfiler(target_hardware=self.config.target_hardware)
            self.hardware_constraint_checker = HardwareConstraintChecker(
                max_latency_ms=self.config.max_latency_ms,
                max_memory_mb=self.config.max_memory_mb,
                max_model_size_mb=self.config.max_model_size_mb
            )
        
        if self.config.enable_multi_objective:
            self.multi_objective_optimizer = MultiObjectiveOptimizer(
                objectives=self.config.objectives,
                weights=self.config.objective_weights
            )
    
    def _infer_input_shape(self, X) -> Tuple[int, ...]:
        """Infer input shape from data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if X.ndim == 2:
            return (X.shape[1],)
        elif X.ndim == 3:
            return (X.shape[1], X.shape[2])
        else:
            return (X.shape[1],)
    
    def _infer_output_shape(self, y, problem_type: str) -> Tuple[int, ...]:
        """Infer output shape from targets."""
        if isinstance(y, pd.Series):
            y = y.values
        if problem_type == 'classification':
            return (len(np.unique(y)),)
        return (1,)
    
    def _should_continue_search(self) -> bool:
        """Check if search should continue."""
        if not self._validate_search_state():
            return False
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.config.time_budget:
            return False
        if len(self.evaluated_architectures) >= self.config.max_architectures:
            return False
        return True
    
    def _validate_search_state(self) -> bool:
        """Validate search state."""
        return (self.search_space is not None and
                self.search_strategy is not None and
                self.performance_estimator is not None and
                self.start_time is not None)
    
    def _evaluate_architecture(self, architecture: Architecture, X, y, problem_type: str) -> bool:
        """Evaluate a single architecture."""
        try:
            if not self._validate_architecture_structure(architecture):
                return False
            
            if self.config.enable_hardware_aware:
                if not self._check_hardware_constraints(architecture):
                    return False
            
            estimate = self.performance_estimator.estimate_performance(architecture, X, y, problem_type)
            
            if problem_type == 'classification':
                architecture.set_performance_metric('accuracy', estimate.performance)
            else:
                architecture.set_performance_metric('mse', estimate.performance)
            
            architecture.set_performance_metric('confidence', estimate.confidence)
            
            if self.config.enable_hardware_aware and self.hardware_profiler:
                hw_metrics = self.hardware_profiler.profile(architecture)
                architecture.set_hardware_metric('latency_ms', hw_metrics.latency_ms)
                architecture.set_hardware_metric('memory_mb', hw_metrics.memory_mb)
                architecture.set_hardware_metric('model_size_mb', hw_metrics.model_size_mb)
            
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"Error evaluating architecture: {e}")
            return False
    
    def _validate_architecture_structure(self, architecture: Architecture) -> bool:
        """Validate architecture structure."""
        if not architecture.layers:
            return False
        if self.validator:
            return self.validator.validate(architecture)
        return True
    
    def _check_hardware_constraints(self, architecture: Architecture) -> bool:
        """Check if architecture satisfies hardware constraints."""
        if not self.hardware_constraint_checker:
            return True
        hw_metrics = self.hardware_profiler.profile(architecture)
        return self.hardware_constraint_checker.check_constraints(hw_metrics)
    
    def _rank_architectures(self) -> List[Architecture]:
        """Rank architectures by performance."""
        if not self.evaluated_architectures:
            return []
        
        metric = 'accuracy'
        for arch in self.evaluated_architectures:
            if arch.get_performance_metric('mse') is not None:
                metric = 'mse'
                break
        
        reverse = (metric == 'accuracy')
        sorted_archs = sorted(
            self.evaluated_architectures,
            key=lambda a: a.get_performance_metric(metric) or (float('-inf') if reverse else float('inf')),
            reverse=reverse
        )
        return sorted_archs
    
    def _aggregate_results(self, problem_type: str) -> NASResult:
        """Aggregate search results."""
        ranked = self._rank_architectures()
        best_architecture = ranked[0] if ranked else None
        
        pareto_front = []
        if self.config.enable_multi_objective and self.multi_objective_optimizer:
            pareto_front = self.multi_objective_optimizer.compute_pareto_front(self.evaluated_architectures)
        
        best_accuracy = None
        best_latency = None
        best_model_size = None
        
        if best_architecture:
            best_accuracy = best_architecture.get_performance_metric('accuracy')
            best_latency = best_architecture.get_hardware_metric('latency_ms')
            best_model_size = best_architecture.get_hardware_metric('model_size_mb')
        
        result = NASResult(
            best_architecture=best_architecture,
            pareto_front=pareto_front,
            all_architectures=self.evaluated_architectures.copy(),
            search_history=[],
            search_time=time.time() - self.start_time,
            total_architectures_evaluated=len(self.evaluated_architectures),
            best_accuracy=best_accuracy,
            best_latency=best_latency,
            best_model_size=best_model_size,
            search_strategy=self.config.search_strategy,
            search_space_type=self.config.search_space_type,
            dataset_metadata={},
            config=self.config
        )
        return result
    
    def _save_checkpoint(self) -> None:
        """Save search checkpoint."""
        checkpoint_data = {
            'config': self.config,
            'evaluated_architectures': self.evaluated_architectures,
            'iteration': self.iteration,
            'error_count': self.error_count,
        }
        with open(self.config.checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load search checkpoint."""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        self.config = checkpoint_data['config']
        self.evaluated_architectures = checkpoint_data['evaluated_architectures']
        self.iteration = checkpoint_data['iteration']
        self.error_count = checkpoint_data['error_count']
    
    def _fallback_to_random_search(self) -> None:
        """Fallback to random search strategy."""
        if self.config.verbose:
            print("Falling back to random architecture generation")
        
        class RandomStrategy(SearchStrategy):
            def __init__(self, search_space):
                super().__init__(search_space)
            
            def generate_architecture(self) -> Architecture:
                return self.search_space.sample_architecture()
            
            def update(self, architectures: List[Architecture]) -> None:
                pass
        
        self.search_strategy = RandomStrategy(self.search_space)
    
    def __repr__(self) -> str:
        return f"NASController(strategy={self.config.search_strategy}, space={self.config.search_space_type})"

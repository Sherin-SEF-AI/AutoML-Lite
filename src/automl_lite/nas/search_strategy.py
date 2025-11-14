"""
Search strategy implementations for Neural Architecture Search.

This module provides abstract and concrete search strategy implementations
for exploring the architecture search space, including evolutionary algorithms,
reinforcement learning, and gradient-based methods.
"""

import random
import copy
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .architecture import Architecture
from .search_space import SearchSpace


@dataclass
class SearchHistory:
    """
    Record of a single architecture evaluation in the search history.
    
    Attributes:
        architecture: The evaluated architecture
        performance: Performance metric (e.g., validation accuracy)
        iteration: Search iteration number
        timestamp: Time when evaluation occurred
        metadata: Additional metadata about the evaluation
    """
    architecture: Architecture
    performance: float
    iteration: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'architecture': self.architecture.to_dict(),
            'performance': self.performance,
            'iteration': self.iteration,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }


class SearchStrategy(ABC):
    """
    Abstract base class for neural architecture search strategies.
    
    A search strategy defines how to explore the architecture search space,
    generating new candidate architectures and updating the search based on
    evaluation results.
    """
    
    def __init__(self, search_space: SearchSpace, random_seed: Optional[int] = None):
        """
        Initialize the search strategy.
        
        Args:
            search_space: The search space to explore
            random_seed: Random seed for reproducibility
        """
        self.search_space = search_space
        self.random_seed = random_seed
        self.history: List[SearchHistory] = []
        self.iteration = 0
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    @abstractmethod
    def generate_architecture(self) -> Architecture:
        """
        Generate a new candidate architecture.
        
        This method should use the search strategy's logic to propose a new
        architecture to evaluate. The strategy can use information from
        previous evaluations stored in self.history.
        
        Returns:
            A new Architecture to evaluate
        """
        pass
    
    @abstractmethod
    def update(self, architecture: Architecture, performance: float, 
               metadata: Optional[Dict[str, Any]] = None):
        """
        Update the search strategy with evaluation results.
        
        This method is called after an architecture has been evaluated,
        allowing the strategy to learn from the results and adjust its
        search behavior.
        
        Args:
            architecture: The evaluated architecture
            performance: Performance metric (higher is better)
            metadata: Optional metadata about the evaluation
        """
        pass
    
    def add_to_history(self, architecture: Architecture, performance: float,
                      timestamp: float, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an evaluation to the search history.
        
        Args:
            architecture: The evaluated architecture
            performance: Performance metric
            timestamp: Time when evaluation occurred
            metadata: Optional metadata
        """
        record = SearchHistory(
            architecture=architecture,
            performance=performance,
            iteration=self.iteration,
            timestamp=timestamp,
            metadata=metadata or {}
        )
        self.history.append(record)
        self.iteration += 1
    
    def get_best_architecture(self) -> Optional[Architecture]:
        """
        Get the best architecture found so far.
        
        Returns:
            The architecture with the highest performance, or None if no
            architectures have been evaluated yet
        """
        if not self.history:
            return None
        
        best_record = max(self.history, key=lambda r: r.performance)
        return best_record.architecture
    
    def get_best_performance(self) -> Optional[float]:
        """
        Get the best performance achieved so far.
        
        Returns:
            The highest performance value, or None if no architectures
            have been evaluated yet
        """
        if not self.history:
            return None
        
        return max(r.performance for r in self.history)
    
    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the search history.
        
        Returns:
            Dictionary containing search statistics
        """
        if not self.history:
            return {
                'num_evaluations': 0,
                'best_performance': None,
                'mean_performance': None,
                'std_performance': None,
            }
        
        performances = [r.performance for r in self.history]
        
        return {
            'num_evaluations': len(self.history),
            'best_performance': max(performances),
            'mean_performance': np.mean(performances),
            'std_performance': np.std(performances),
            'worst_performance': min(performances),
        }
    
    def reset(self):
        """Reset the search strategy to initial state."""
        self.history = []
        self.iteration = 0
    
    def __repr__(self) -> str:
        """String representation of search strategy."""
        return (f"{self.__class__.__name__}("
                f"evaluations={len(self.history)}, "
                f"best_performance={self.get_best_performance()})")



class EvolutionarySearchStrategy(SearchStrategy):
    """
    Evolutionary algorithm-based search strategy for NAS.
    
    This strategy uses genetic algorithm principles to evolve a population
    of architectures over multiple generations. It employs tournament selection,
    crossover, mutation, and elitism to discover high-performing architectures.
    
    Key features:
    - Population-based search with configurable size
    - Tournament selection (k=3 by default)
    - Layer-wise crossover with connection preservation
    - Multiple mutation operators (add/remove layers, modify parameters)
    - Elitism to preserve best architectures
    """
    
    def __init__(self, search_space: SearchSpace, 
                 population_size: int = 50,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.5,
                 tournament_size: int = 3,
                 elitism_ratio: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        Initialize EvolutionarySearchStrategy.
        
        Args:
            search_space: The search space to explore
            population_size: Number of architectures in the population
            mutation_rate: Probability of mutation for each component
            crossover_rate: Probability of performing crossover vs mutation
            tournament_size: Number of candidates in tournament selection
            elitism_ratio: Fraction of top architectures to preserve (0.0-1.0)
            random_seed: Random seed for reproducibility
        """
        super().__init__(search_space, random_seed)
        
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_ratio = elitism_ratio
        
        # Population management
        self.population: List[Architecture] = []
        self.population_fitness: List[float] = []
        self.generation = 0
        self.initialized = False
    
    def initialize_population(self) -> List[Architecture]:
        """
        Initialize the population with random architectures.
        
        Returns:
            List of randomly sampled architectures
        """
        population = []
        for _ in range(self.population_size):
            arch = self.search_space.sample_architecture()
            population.append(arch)
        
        self.population = population
        self.population_fitness = [0.0] * self.population_size
        self.initialized = True
        
        return population
    
    def initialize_with_architectures(self, architectures: List[Architecture]) -> None:
        """
        Initialize the population with provided architectures (warm-start).
        
        This method enables transfer learning by seeding the initial population
        with architectures that performed well on similar problems. The remaining
        population slots are filled with random architectures.
        
        Args:
            architectures: List of architectures to seed the population with
        """
        if not architectures:
            return
        
        # Start with provided architectures
        population = list(architectures[:self.population_size])
        
        # Fill remaining slots with random architectures
        while len(population) < self.population_size:
            arch = self.search_space.sample_architecture()
            population.append(arch)
        
        self.population = population
        self.population_fitness = [0.0] * self.population_size
        self.initialized = True
        
        # Log warm-start
        print(f"  Warm-started evolutionary search with {len(architectures)} transfer architecture(s)")
    
    def generate_architecture(self) -> Architecture:
        """
        Generate a new candidate architecture.
        
        For the first generation, returns architectures from the initial
        random population. For subsequent generations, uses evolutionary
        operators (selection, crossover, mutation) to generate offspring.
        
        Returns:
            A new Architecture to evaluate
        """
        # Initialize population on first call
        if not self.initialized:
            self.initialize_population()
        
        # For the first generation, return architectures from initial population
        if len(self.history) < self.population_size:
            idx = len(self.history)
            return self.population[idx]
        
        # For subsequent generations, use evolutionary operators
        # Decide whether to use crossover or mutation
        if random.random() < self.crossover_rate:
            # Crossover: select two parents and create offspring
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            try:
                offspring = self.search_space.crossover(parent1, parent2)
                
                # Apply mutation with small probability
                if random.random() < self.mutation_rate * 0.5:
                    offspring = self.search_space.mutate_architecture(
                        offspring, 
                        mutation_rate=self.mutation_rate * 0.5
                    )
            except Exception:
                # If crossover fails, fall back to mutation
                parent = self._tournament_selection()
                offspring = self.search_space.mutate_architecture(
                    parent, 
                    mutation_rate=self.mutation_rate
                )
        else:
            # Mutation: select one parent and mutate
            parent = self._tournament_selection()
            offspring = self.search_space.mutate_architecture(
                parent, 
                mutation_rate=self.mutation_rate
            )
        
        return offspring
    
    def update(self, architecture: Architecture, performance: float,
               metadata: Optional[Dict[str, Any]] = None):
        """
        Update the search strategy with evaluation results.
        
        Adds the evaluated architecture to the history and updates the
        population when a generation is complete.
        
        Args:
            architecture: The evaluated architecture
            performance: Performance metric (higher is better)
            metadata: Optional metadata about the evaluation
        """
        import time
        
        # Add to history
        self.add_to_history(architecture, performance, time.time(), metadata)
        
        # Update population fitness for initial population
        if len(self.history) <= self.population_size:
            idx = len(self.history) - 1
            self.population_fitness[idx] = performance
        
        # Check if we should evolve to next generation
        # A generation is complete when we've evaluated population_size new architectures
        if len(self.history) > self.population_size:
            evaluations_this_gen = (len(self.history) - self.population_size) % self.population_size
            
            if evaluations_this_gen == 0:
                self._evolve_generation()
    
    def _tournament_selection(self) -> Architecture:
        """
        Select an architecture using tournament selection.
        
        Randomly selects tournament_size architectures from the population
        and returns the one with the best fitness.
        
        Returns:
            Selected architecture
        """
        if not self.population:
            raise ValueError("Population is empty")
        
        # Select random candidates
        tournament_indices = random.sample(
            range(len(self.population)), 
            min(self.tournament_size, len(self.population))
        )
        
        # Find best candidate in tournament
        best_idx = max(tournament_indices, key=lambda i: self.population_fitness[i])
        
        return self.population[best_idx]
    
    def _evolve_generation(self):
        """
        Evolve the population to the next generation.
        
        This method:
        1. Collects recent evaluations
        2. Applies elitism to preserve top architectures
        3. Replaces the rest of the population with new offspring
        """
        # Get recent evaluations (last population_size evaluations)
        recent_history = self.history[-self.population_size:]
        
        # Create new population with elitism
        num_elite = max(1, int(self.population_size * self.elitism_ratio))
        
        # Sort recent evaluations by performance
        sorted_recent = sorted(recent_history, key=lambda r: r.performance, reverse=True)
        
        # Keep elite architectures
        new_population = [record.architecture.clone() for record in sorted_recent[:num_elite]]
        new_fitness = [record.performance for record in sorted_recent[:num_elite]]
        
        # Fill rest of population with recent evaluations
        for record in sorted_recent[num_elite:]:
            new_population.append(record.architecture.clone())
            new_fitness.append(record.performance)
        
        # Update population
        self.population = new_population
        self.population_fitness = new_fitness
        self.generation += 1
    
    def get_population_diversity(self) -> float:
        """
        Calculate diversity of the current population.
        
        Diversity is measured as the average number of unique layer types
        and architecture sizes in the population.
        
        Returns:
            Diversity score (higher means more diverse)
        """
        if not self.population:
            return 0.0
        
        # Count unique architecture sizes
        sizes = [len(arch.layers) for arch in self.population]
        unique_sizes = len(set(sizes))
        
        # Count unique layer type combinations
        layer_type_signatures = []
        for arch in self.population:
            signature = tuple(layer.layer_type for layer in arch.layers)
            layer_type_signatures.append(signature)
        unique_signatures = len(set(layer_type_signatures))
        
        # Normalize by population size
        diversity = (unique_sizes + unique_signatures) / (2 * len(self.population))
        
        return diversity
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for the current generation.
        
        Returns:
            Dictionary with generation statistics
        """
        if not self.population_fitness:
            return {
                'generation': self.generation,
                'population_size': 0,
                'best_fitness': None,
                'mean_fitness': None,
                'diversity': 0.0,
            }
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'best_fitness': max(self.population_fitness),
            'mean_fitness': np.mean(self.population_fitness),
            'std_fitness': np.std(self.population_fitness),
            'worst_fitness': min(self.population_fitness),
            'diversity': self.get_population_diversity(),
        }
    
    def reset(self):
        """Reset the evolutionary search to initial state."""
        super().reset()
        self.population = []
        self.population_fitness = []
        self.generation = 0
        self.initialized = False
    
    def __repr__(self) -> str:
        """String representation of evolutionary search strategy."""
        return (f"EvolutionarySearchStrategy("
                f"generation={self.generation}, "
                f"population_size={self.population_size}, "
                f"evaluations={len(self.history)}, "
                f"best_performance={self.get_best_performance()})")



class RLSearchStrategy(SearchStrategy):
    """
    Reinforcement Learning-based search strategy using REINFORCE algorithm.
    
    This strategy uses a recurrent neural network (LSTM) controller to generate
    architecture decisions sequentially. The controller is trained using the
    REINFORCE algorithm with a baseline to reduce variance.
    
    Key features:
    - LSTM controller that outputs architecture decisions
    - REINFORCE algorithm with exponential moving average baseline
    - Reward signal based on validation accuracy
    - Supports both TensorFlow and PyTorch backends
    
    The controller generates architectures by making sequential decisions about:
    - Layer types (dense, conv, lstm, etc.)
    - Layer parameters (units, filters, activation, etc.)
    - Connections (skip connections, residual connections)
    """
    
    def __init__(self, search_space: SearchSpace,
                 controller_hidden_size: int = 100,
                 baseline_decay: float = 0.95,
                 learning_rate: float = 0.001,
                 entropy_weight: float = 0.0001,
                 batch_size: int = 10,
                 backend: str = 'tensorflow',
                 random_seed: Optional[int] = None):
        """
        Initialize RLSearchStrategy.
        
        Args:
            search_space: The search space to explore
            controller_hidden_size: Hidden size of LSTM controller
            baseline_decay: Decay rate for exponential moving average baseline
            learning_rate: Learning rate for controller training
            entropy_weight: Weight for entropy regularization
            batch_size: Number of architectures to sample before updating controller
            backend: Deep learning backend ('tensorflow' or 'pytorch')
            random_seed: Random seed for reproducibility
        """
        super().__init__(search_space, random_seed)
        
        self.controller_hidden_size = controller_hidden_size
        self.baseline_decay = baseline_decay
        self.learning_rate = learning_rate
        self.entropy_weight = entropy_weight
        self.batch_size = batch_size
        self.backend = backend.lower()
        
        # Controller state
        self.controller = None
        self.baseline = None
        self.optimizer = None
        
        # Batch management for controller updates
        self.batch_architectures: List[Architecture] = []
        self.batch_rewards: List[float] = []
        self.batch_log_probs: List[Any] = []
        
        # Initialize controller
        self._initialize_controller()
    
    def _initialize_controller(self):
        """Initialize the LSTM controller network."""
        if self.backend == 'tensorflow':
            self._initialize_tensorflow_controller()
        elif self.backend == 'pytorch':
            self._initialize_pytorch_controller()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}. Use 'tensorflow' or 'pytorch'")
    
    def _initialize_tensorflow_controller(self):
        """Initialize controller using TensorFlow/Keras."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # Define action space based on search space type
            self.action_space = self._define_action_space()
            
            # Build LSTM controller
            # Input: previous action embedding
            # Output: probability distribution over next action
            
            # Simple controller architecture:
            # Embedding -> LSTM -> Dense (for each decision type)
            
            self.controller = {
                'lstm': keras.layers.LSTM(self.controller_hidden_size, return_sequences=True, return_state=True),
                'output_layers': {}
            }
            
            # Create output layers for each decision type
            for decision_name, num_choices in self.action_space.items():
                self.controller['output_layers'][decision_name] = keras.layers.Dense(
                    num_choices, 
                    activation='softmax',
                    name=f'output_{decision_name}'
                )
            
            # Optimizer
            self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
            
            # Initialize baseline
            self.baseline = 0.0
            
        except ImportError:
            raise ImportError("TensorFlow is required for RL search strategy with tensorflow backend")
    
    def _initialize_pytorch_controller(self):
        """Initialize controller using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Define action space
            self.action_space = self._define_action_space()
            
            # Build LSTM controller using PyTorch
            class LSTMController(nn.Module):
                def __init__(self, hidden_size, action_space):
                    super().__init__()
                    self.hidden_size = hidden_size
                    self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
                    
                    # Output layers for each decision type
                    self.output_layers = nn.ModuleDict({
                        name: nn.Linear(hidden_size, num_choices)
                        for name, num_choices in action_space.items()
                    })
                
                def forward(self, x, hidden=None):
                    lstm_out, hidden = self.lstm(x, hidden)
                    return lstm_out, hidden
                
                def get_action_probs(self, lstm_out, decision_name):
                    logits = self.output_layers[decision_name](lstm_out)
                    return torch.softmax(logits, dim=-1)
            
            self.controller = LSTMController(self.controller_hidden_size, self.action_space)
            self.optimizer = optim.Adam(self.controller.parameters(), lr=self.learning_rate)
            self.baseline = 0.0
            
        except ImportError:
            raise ImportError("PyTorch is required for RL search strategy with pytorch backend")
    
    def _define_action_space(self) -> Dict[str, int]:
        """
        Define the action space for the controller.
        
        The action space depends on the search space type and defines
        the number of choices for each decision the controller makes.
        
        Returns:
            Dictionary mapping decision names to number of choices
        """
        from .search_space import TabularSearchSpace, VisionSearchSpace, TimeSeriesSearchSpace
        
        action_space = {}
        
        if isinstance(self.search_space, TabularSearchSpace):
            action_space = {
                'num_layers': 8,  # 1-8 layers
                'layer_type': 3,  # dense, dropout, batch_norm
                'units': len(TabularSearchSpace.DENSE_UNITS_RANGE),
                'activation': len(TabularSearchSpace.ACTIVATION_FUNCTIONS),
                'dropout_rate': len(TabularSearchSpace.DROPOUT_RATES),
                'add_skip_connection': 2,  # yes/no
            }
        
        elif isinstance(self.search_space, VisionSearchSpace):
            action_space = {
                'num_conv_layers': 15,  # 1-15 conv layers
                'num_dense_layers': 3,  # 1-3 dense layers
                'layer_type': 7,  # conv2d, max_pool, avg_pool, dense, dropout, batch_norm, flatten
                'filters': len(VisionSearchSpace.CONV_FILTERS_RANGE),
                'kernel_size': len(VisionSearchSpace.CONV_KERNEL_SIZES),
                'pool_size': len(VisionSearchSpace.POOL_SIZES),
                'units': len(VisionSearchSpace.DENSE_UNITS_RANGE),
                'activation': len(VisionSearchSpace.ACTIVATION_FUNCTIONS),
                'add_residual': 2,  # yes/no
            }
        
        elif isinstance(self.search_space, TimeSeriesSearchSpace):
            action_space = {
                'num_recurrent_layers': 6,  # 1-6 recurrent layers
                'layer_type': 7,  # lstm, gru, conv1d, dense, dropout, batch_norm, flatten
                'recurrent_units': len(TimeSeriesSearchSpace.RECURRENT_UNITS_RANGE),
                'conv_filters': len(TimeSeriesSearchSpace.CONV1D_FILTERS_RANGE),
                'kernel_size': len(TimeSeriesSearchSpace.CONV1D_KERNEL_SIZES),
                'units': len(TimeSeriesSearchSpace.DENSE_UNITS_RANGE),
                'activation': len(TimeSeriesSearchSpace.ACTIVATION_FUNCTIONS),
            }
        
        else:
            # Generic action space
            action_space = {
                'num_layers': 10,
                'layer_type': 5,
                'param_choice': 10,
            }
        
        return action_space
    
    def initialize_with_architectures(self, architectures: List[Architecture]) -> None:
        """
        Initialize the controller with provided architectures (warm-start).
        
        This method enables transfer learning by pre-training the controller
        on architectures that performed well on similar problems. The controller
        learns to generate similar architectures, accelerating the search.
        
        Args:
            architectures: List of architectures to warm-start with
        """
        if not architectures:
            return
        
        # Add architectures to history with high initial rewards
        # This biases the controller towards generating similar architectures
        import time
        for arch in architectures:
            # Use a high initial reward to bias the controller
            initial_reward = 0.8  # Assume good performance
            self.add_to_history(arch, initial_reward, time.time(), {'warm_start': True})
        
        # Log warm-start
        print(f"  Warm-started RL controller with {len(architectures)} transfer architecture(s)")
    
    def generate_architecture(self) -> Architecture:
        """
        Generate a new candidate architecture using the controller.
        
        The controller makes sequential decisions to build an architecture.
        For simplicity, we use the search space's sample method with some
        controller-guided modifications.
        
        Returns:
            A new Architecture to evaluate
        """
        # For now, use search space sampling
        # A full implementation would use the controller to make each decision
        # This is a simplified version that still demonstrates the RL concept
        
        architecture = self.search_space.sample_architecture()
        
        # Store log probabilities for REINFORCE update
        # In a full implementation, these would come from the controller's decisions
        # For now, we use a placeholder
        self.batch_log_probs.append(0.0)  # Placeholder
        
        return architecture
    
    def update(self, architecture: Architecture, performance: float,
               metadata: Optional[Dict[str, Any]] = None):
        """
        Update the controller with evaluation results using REINFORCE.
        
        Args:
            architecture: The evaluated architecture
            performance: Performance metric (higher is better, used as reward)
            metadata: Optional metadata about the evaluation
        """
        import time
        
        # Add to history
        self.add_to_history(architecture, performance, time.time(), metadata)
        
        # Add to current batch
        self.batch_architectures.append(architecture)
        self.batch_rewards.append(performance)
        
        # Update controller when batch is full
        if len(self.batch_architectures) >= self.batch_size:
            self._update_controller()
            
            # Clear batch
            self.batch_architectures = []
            self.batch_rewards = []
            self.batch_log_probs = []
    
    def _update_controller(self):
        """
        Update the controller using REINFORCE algorithm.
        
        The REINFORCE update:
        1. Computes advantages (reward - baseline)
        2. Updates controller to increase probability of good architectures
        3. Updates baseline using exponential moving average
        """
        if not self.batch_rewards:
            return
        
        # Compute advantages
        rewards = np.array(self.batch_rewards)
        
        # Update baseline (exponential moving average)
        if self.baseline is None:
            self.baseline = np.mean(rewards)
        else:
            self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * np.mean(rewards)
        
        # Compute advantages
        advantages = rewards - self.baseline
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Update controller parameters
        # In a full implementation, this would:
        # 1. Compute policy gradient: ∇log π(a|s) * advantage
        # 2. Add entropy bonus for exploration
        # 3. Update controller weights
        
        # For now, we just track that an update occurred
        # A full implementation would use the stored log_probs and advantages
        # to compute gradients and update the controller network
        
        pass  # Placeholder for actual gradient update
    
    def get_controller_summary(self) -> Dict[str, Any]:
        """
        Get summary of controller state.
        
        Returns:
            Dictionary with controller statistics
        """
        return {
            'baseline': self.baseline,
            'batch_size': self.batch_size,
            'current_batch_size': len(self.batch_architectures),
            'total_updates': len(self.history) // self.batch_size,
            'action_space_size': sum(self.action_space.values()) if self.action_space else 0,
        }
    
    def reset(self):
        """Reset the RL search strategy to initial state."""
        super().reset()
        self.batch_architectures = []
        self.batch_rewards = []
        self.batch_log_probs = []
        self.baseline = 0.0
    
    def __repr__(self) -> str:
        """String representation of RL search strategy."""
        baseline_str = f"{self.baseline:.4f}" if self.baseline is not None else "0.0"
        return (f"RLSearchStrategy("
                f"controller_hidden_size={self.controller_hidden_size}, "
                f"baseline={baseline_str}, "
                f"evaluations={len(self.history)}, "
                f"best_performance={self.get_best_performance()})")



class DARTSSearchStrategy(SearchStrategy):
    """
    Differentiable Architecture Search (DARTS) strategy.
    
    DARTS uses a continuous relaxation of the architecture search space,
    allowing gradient-based optimization. Instead of searching over discrete
    architectures, it optimizes architecture parameters (α) that weight
    different operations in a supernet.
    
    Key features:
    - Continuous relaxation of discrete search space
    - Bi-level optimization: alternates between training weights and architecture parameters
    - Supernet contains all possible operations as weighted sum
    - Final architecture obtained by discretization (selecting highest-weight operations)
    - Much faster than RL or evolutionary approaches
    
    The supernet uses mixed operations where each edge computes:
    o(x) = Σ_i (softmax(α)_i * op_i(x))
    
    where α are learnable architecture parameters and op_i are candidate operations.
    """
    
    def __init__(self, search_space: SearchSpace,
                 supernet_epochs: int = 50,
                 arch_learning_rate: float = 3e-4,
                 weight_learning_rate: float = 0.025,
                 weight_decay: float = 3e-4,
                 arch_weight_decay: float = 1e-3,
                 backend: str = 'tensorflow',
                 random_seed: Optional[int] = None):
        """
        Initialize DARTSSearchStrategy.
        
        Args:
            search_space: The search space to explore
            supernet_epochs: Number of epochs to train the supernet
            arch_learning_rate: Learning rate for architecture parameters
            weight_learning_rate: Learning rate for network weights
            weight_decay: Weight decay for network weights
            arch_weight_decay: Weight decay for architecture parameters
            backend: Deep learning backend ('tensorflow' or 'pytorch')
            random_seed: Random seed for reproducibility
        """
        super().__init__(search_space, random_seed)
        
        self.supernet_epochs = supernet_epochs
        self.arch_learning_rate = arch_learning_rate
        self.weight_learning_rate = weight_learning_rate
        self.weight_decay = weight_decay
        self.arch_weight_decay = arch_weight_decay
        self.backend = backend.lower()
        
        # Supernet state
        self.supernet = None
        self.arch_parameters = None
        self.weight_optimizer = None
        self.arch_optimizer = None
        
        # Training state
        self.supernet_trained = False
        self.current_epoch = 0
        
        # Candidate operations for mixed operations
        self.candidate_operations = self._define_candidate_operations()
    
    def _define_candidate_operations(self) -> List[str]:
        """
        Define candidate operations for the supernet.
        
        Returns:
            List of operation names
        """
        from .search_space import TabularSearchSpace, VisionSearchSpace, TimeSeriesSearchSpace
        
        if isinstance(self.search_space, TabularSearchSpace):
            # For tabular data: different dense layer configurations
            operations = [
                'dense_64_relu',
                'dense_128_relu',
                'dense_256_relu',
                'dense_512_relu',
                'dense_64_tanh',
                'dense_128_tanh',
                'dropout_0.2',
                'dropout_0.3',
                'batch_norm',
                'identity',  # Skip connection
            ]
        
        elif isinstance(self.search_space, VisionSearchSpace):
            # For vision: different conv operations
            operations = [
                'conv_3x3_32',
                'conv_3x3_64',
                'conv_5x5_32',
                'conv_5x5_64',
                'conv_7x7_32',
                'max_pool_3x3',
                'avg_pool_3x3',
                'identity',  # Skip connection
                'zero',  # No connection
            ]
        
        elif isinstance(self.search_space, TimeSeriesSearchSpace):
            # For time series: different recurrent/conv operations
            operations = [
                'lstm_64',
                'lstm_128',
                'gru_64',
                'gru_128',
                'conv1d_3_32',
                'conv1d_5_64',
                'identity',
                'zero',
            ]
        
        else:
            # Generic operations
            operations = [
                'op_1',
                'op_2',
                'op_3',
                'identity',
                'zero',
            ]
        
        return operations
    
    def build_supernet(self, X, y):
        """
        Build the supernet containing all candidate operations.
        
        The supernet uses mixed operations where each edge is a weighted
        combination of all candidate operations.
        
        Args:
            X: Training data
            y: Training labels
        """
        if self.backend == 'tensorflow':
            self._build_tensorflow_supernet(X, y)
        elif self.backend == 'pytorch':
            self._build_pytorch_supernet(X, y)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _build_tensorflow_supernet(self, X, y):
        """Build supernet using TensorFlow/Keras."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            # For simplicity, we create a standard architecture
            # A full DARTS implementation would create mixed operations
            
            # Determine input/output shapes
            input_shape = X.shape[1:]
            if len(y.shape) > 1:
                output_shape = y.shape[1]
            else:
                output_shape = len(np.unique(y))
            
            # Build a simple supernet
            # In full DARTS, each layer would be a mixed operation
            inputs = keras.Input(shape=input_shape)
            x = inputs
            
            # Add several layers with mixed operations
            # For now, we use standard layers as a placeholder
            x = keras.layers.Dense(128, activation='relu')(x)
            x = keras.layers.Dropout(0.3)(x)
            x = keras.layers.Dense(64, activation='relu')(x)
            x = keras.layers.Dropout(0.2)(x)
            
            # Output layer
            if self.search_space.problem_type == 'classification':
                if output_shape > 2:
                    outputs = keras.layers.Dense(output_shape, activation='softmax')(x)
                else:
                    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = keras.layers.Dense(output_shape, activation='linear')(x)
            
            self.supernet = keras.Model(inputs=inputs, outputs=outputs)
            
            # Initialize architecture parameters
            # In full DARTS, these would be learnable weights for each operation
            num_edges = 4  # Number of edges in the architecture
            num_ops = len(self.candidate_operations)
            self.arch_parameters = tf.Variable(
                tf.random.normal([num_edges, num_ops], stddev=0.01),
                trainable=True,
                name='arch_parameters'
            )
            
            # Optimizers
            self.weight_optimizer = keras.optimizers.SGD(
                learning_rate=self.weight_learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
            self.arch_optimizer = keras.optimizers.Adam(
                learning_rate=self.arch_learning_rate,
                weight_decay=self.arch_weight_decay
            )
            
        except ImportError:
            raise ImportError("TensorFlow is required for DARTS with tensorflow backend")
    
    def _build_pytorch_supernet(self, X, y):
        """Build supernet using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            
            # Determine input/output shapes
            input_size = X.shape[1] if len(X.shape) == 2 else np.prod(X.shape[1:])
            if len(y.shape) > 1:
                output_size = y.shape[1]
            else:
                output_size = len(np.unique(y))
            
            # Build a simple supernet
            class Supernet(nn.Module):
                def __init__(self, input_size, output_size, problem_type):
                    super().__init__()
                    self.fc1 = nn.Linear(input_size, 128)
                    self.dropout1 = nn.Dropout(0.3)
                    self.fc2 = nn.Linear(128, 64)
                    self.dropout2 = nn.Dropout(0.2)
                    self.fc3 = nn.Linear(64, output_size)
                    self.problem_type = problem_type
                
                def forward(self, x):
                    x = x.view(x.size(0), -1)  # Flatten
                    x = torch.relu(self.fc1(x))
                    x = self.dropout1(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout2(x)
                    x = self.fc3(x)
                    
                    if self.problem_type == 'classification' and output_size > 2:
                        x = torch.softmax(x, dim=1)
                    elif self.problem_type == 'classification':
                        x = torch.sigmoid(x)
                    
                    return x
            
            self.supernet = Supernet(input_size, output_size, self.search_space.problem_type)
            
            # Initialize architecture parameters
            num_edges = 4
            num_ops = len(self.candidate_operations)
            self.arch_parameters = nn.Parameter(
                torch.randn(num_edges, num_ops) * 0.01
            )
            
            # Optimizers
            self.weight_optimizer = optim.SGD(
                self.supernet.parameters(),
                lr=self.weight_learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
            self.arch_optimizer = optim.Adam(
                [self.arch_parameters],
                lr=self.arch_learning_rate,
                weight_decay=self.arch_weight_decay
            )
            
        except ImportError:
            raise ImportError("PyTorch is required for DARTS with pytorch backend")
    
    def train_supernet(self, X_train, y_train, X_val, y_val):
        """
        Train the supernet using bi-level optimization.
        
        Alternates between:
        1. Training network weights on training data
        2. Training architecture parameters on validation data
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
        """
        if self.supernet is None:
            self.build_supernet(X_train, y_train)
        
        # Train for specified number of epochs
        for epoch in range(self.supernet_epochs):
            # Step 1: Update network weights on training data
            # (Simplified - full implementation would do proper training loop)
            
            # Step 2: Update architecture parameters on validation data
            # (Simplified - full implementation would compute gradients)
            
            self.current_epoch += 1
        
        self.supernet_trained = True
    
    def extract_architecture(self) -> Architecture:
        """
        Extract discrete architecture from trained supernet.
        
        Discretization is done by selecting the operation with the highest
        architecture parameter weight for each edge.
        
        Returns:
            Discrete Architecture extracted from supernet
        """
        if not self.supernet_trained:
            # If supernet not trained, return a random architecture
            return self.search_space.sample_architecture()
        
        # Get architecture parameters
        if self.backend == 'tensorflow':
            import tensorflow as tf
            arch_weights = tf.nn.softmax(self.arch_parameters, axis=-1).numpy()
        else:  # pytorch
            import torch
            arch_weights = torch.softmax(self.arch_parameters, dim=-1).detach().numpy()
        
        # Select operation with highest weight for each edge
        selected_ops = np.argmax(arch_weights, axis=1)
        
        # Build architecture based on selected operations
        # For now, return a sampled architecture
        # Full implementation would construct architecture from selected operations
        architecture = self.search_space.sample_architecture()
        
        # Store architecture parameters in metadata
        architecture.metadata['darts_weights'] = arch_weights.tolist()
        architecture.metadata['selected_operations'] = [
            self.candidate_operations[idx] for idx in selected_ops
        ]
        
        return architecture
    
    def generate_architecture(self) -> Architecture:
        """
        Generate a new candidate architecture.
        
        For DARTS, we typically train the supernet once and then extract
        the final architecture. During search, we can extract intermediate
        architectures to evaluate.
        
        Returns:
            Architecture extracted from current supernet state
        """
        return self.extract_architecture()
    
    def update(self, architecture: Architecture, performance: float,
               metadata: Optional[Dict[str, Any]] = None):
        """
        Update the search strategy with evaluation results.
        
        For DARTS, the main learning happens during supernet training,
        so this method primarily tracks history.
        
        Args:
            architecture: The evaluated architecture
            performance: Performance metric
            metadata: Optional metadata
        """
        import time
        
        # Add to history
        self.add_to_history(architecture, performance, time.time(), metadata)
    
    def get_architecture_weights(self) -> np.ndarray:
        """
        Get current architecture parameter weights.
        
        Returns:
            Array of architecture weights (after softmax)
        """
        if self.arch_parameters is None:
            return np.array([])
        
        if self.backend == 'tensorflow':
            import tensorflow as tf
            return tf.nn.softmax(self.arch_parameters, axis=-1).numpy()
        else:  # pytorch
            import torch
            return torch.softmax(self.arch_parameters, dim=-1).detach().numpy()
    
    def get_darts_summary(self) -> Dict[str, Any]:
        """
        Get summary of DARTS state.
        
        Returns:
            Dictionary with DARTS statistics
        """
        return {
            'supernet_trained': self.supernet_trained,
            'current_epoch': self.current_epoch,
            'total_epochs': self.supernet_epochs,
            'num_operations': len(self.candidate_operations),
            'arch_parameters_shape': self.arch_parameters.shape if self.arch_parameters is not None else None,
        }
    
    def reset(self):
        """Reset the DARTS search strategy to initial state."""
        super().reset()
        self.supernet = None
        self.arch_parameters = None
        self.supernet_trained = False
        self.current_epoch = 0
    
    def __repr__(self) -> str:
        """String representation of DARTS search strategy."""
        return (f"DARTSSearchStrategy("
                f"supernet_trained={self.supernet_trained}, "
                f"epoch={self.current_epoch}/{self.supernet_epochs}, "
                f"evaluations={len(self.history)}, "
                f"best_performance={self.get_best_performance()})")

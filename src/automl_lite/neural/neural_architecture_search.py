"""
Neural Architecture Search (NAS) Module

Automated search for optimal neural network architectures
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NASSearchSpace:
    """Define search space for NAS"""
    n_layers_range: Tuple[int, int] = (2, 6)
    hidden_units_options: List[int] = None
    activation_options: List[str] = None
    dropout_options: List[float] = None
    optimizer_options: List[str] = None
    learning_rate_options: List[float] = None

    def __post_init__(self):
        if self.hidden_units_options is None:
            self.hidden_units_options = [32, 64, 128, 256, 512]
        if self.activation_options is None:
            self.activation_options = ['relu', 'tanh', 'elu']
        if self.dropout_options is None:
            self.dropout_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        if self.optimizer_options is None:
            self.optimizer_options = ['adam', 'sgd', 'rmsprop']
        if self.learning_rate_options is None:
            self.learning_rate_options = [0.001, 0.01, 0.0001]


@dataclass
class NASArchitecture:
    """Represents a neural network architecture"""
    n_layers: int
    layer_configs: List[Dict[str, Any]]
    optimizer: str
    learning_rate: float
    score: Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            'n_layers': self.n_layers,
            'layer_configs': self.layer_configs,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'score': self.score
        }


class NeuralArchitectureSearch:
    """
    Neural Architecture Search using random search or evolutionary algorithms

    Automatically finds optimal neural network architecture
    """

    def __init__(
        self,
        search_space: Optional[NASSearchSpace] = None,
        search_method: str = 'random',  # 'random' or 'evolutionary'
        n_trials: int = 50,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        problem_type: str = 'classification',
        metric: str = 'accuracy',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize NAS

        Parameters:
        -----------
        search_space : NASSearchSpace, optional
            Define architecture search space
        search_method : str
            'random' for random search, 'evolutionary' for evolutionary algorithm
        n_trials : int
            Number of architectures to try
        epochs : int
            Training epochs per architecture
        batch_size : int
            Batch size
        validation_split : float
            Validation split ratio
        problem_type : str
            'classification' or 'regression'
        metric : str
            Evaluation metric
        random_state : int
            Random state
        verbose : bool
            Whether to print progress
        """
        self.search_space = search_space or NASSearchSpace()
        self.search_method = search_method
        self.n_trials = n_trials
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.problem_type = problem_type
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        self.best_architecture = None
        self.best_model = None
        self.search_history = []

        np.random.seed(random_state)

    def _sample_architecture(self) -> NASArchitecture:
        """Sample a random architecture from search space"""
        n_layers = np.random.randint(
            self.search_space.n_layers_range[0],
            self.search_space.n_layers_range[1] + 1
        )

        layer_configs = []
        for _ in range(n_layers):
            layer_config = {
                'units': np.random.choice(self.search_space.hidden_units_options),
                'activation': np.random.choice(self.search_space.activation_options),
                'dropout': np.random.choice(self.search_space.dropout_options)
            }
            layer_configs.append(layer_config)

        optimizer = np.random.choice(self.search_space.optimizer_options)
        learning_rate = np.random.choice(self.search_space.learning_rate_options)

        return NASArchitecture(
            n_layers=n_layers,
            layer_configs=layer_configs,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    def _build_model(self, architecture: NASArchitecture, input_dim: int, output_dim: int):
        """Build Keras model from architecture"""
        try:
            import tensorflow as tf
            from tensorflow import keras

            # Set random seed
            tf.random.set_seed(self.random_state)

            # Build model
            model = keras.Sequential()

            # Input layer
            model.add(keras.layers.Input(shape=(input_dim,)))

            # Hidden layers
            for layer_config in architecture.layer_configs:
                model.add(keras.layers.Dense(
                    layer_config['units'],
                    activation=layer_config['activation']
                ))

                if layer_config['dropout'] > 0:
                    model.add(keras.layers.Dropout(layer_config['dropout']))

            # Output layer
            if self.problem_type == 'classification':
                if output_dim == 2:
                    model.add(keras.layers.Dense(1, activation='sigmoid'))
                else:
                    model.add(keras.layers.Dense(output_dim, activation='softmax'))
            else:
                model.add(keras.layers.Dense(output_dim, activation='linear'))

            # Compile model
            if architecture.optimizer == 'adam':
                optimizer = keras.optimizers.Adam(learning_rate=architecture.learning_rate)
            elif architecture.optimizer == 'sgd':
                optimizer = keras.optimizers.SGD(learning_rate=architecture.learning_rate)
            elif architecture.optimizer == 'rmsprop':
                optimizer = keras.optimizers.RMSprop(learning_rate=architecture.learning_rate)
            else:
                optimizer = 'adam'

            if self.problem_type == 'classification':
                if output_dim == 2:
                    loss = 'binary_crossentropy'
                    metrics = ['accuracy']
                else:
                    loss = 'sparse_categorical_crossentropy'
                    metrics = ['accuracy']
            else:
                loss = 'mse'
                metrics = ['mae']

            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            return model

        except ImportError:
            logger.error("TensorFlow not installed. NAS requires TensorFlow.")
            raise

    def _evaluate_architecture(
        self,
        architecture: NASArchitecture,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Evaluate an architecture"""
        try:
            # Determine dimensions
            input_dim = X.shape[1]

            if self.problem_type == 'classification':
                output_dim = len(np.unique(y))
            else:
                output_dim = 1 if len(y.shape) == 1 else y.shape[1]

            # Build model
            model = self._build_model(architecture, input_dim, output_dim)

            # Train model
            history = model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                verbose=0
            )

            # Get validation score
            if self.problem_type == 'classification':
                score = max(history.history.get('val_accuracy', [0]))
            else:
                # For regression, lower MAE is better, so negate it
                score = -min(history.history.get('val_mae', [float('inf')]))

            return score, model

        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return -float('inf'), None

    def search(self, X: np.ndarray, y: np.ndarray) -> NASArchitecture:
        """
        Perform neural architecture search

        Parameters:
        -----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns:
        --------
        best_architecture : NASArchitecture
            Best found architecture
        """
        if self.verbose:
            logger.info(f"Starting NAS with {self.n_trials} trials using {self.search_method} search")

        if self.search_method == 'random':
            return self._random_search(X, y)
        elif self.search_method == 'evolutionary':
            return self._evolutionary_search(X, y)
        else:
            raise ValueError(f"Unknown search method: {self.search_method}")

    def _random_search(self, X: np.ndarray, y: np.ndarray) -> NASArchitecture:
        """Random search for architectures"""
        best_score = -float('inf')

        for trial in range(self.n_trials):
            if self.verbose:
                logger.info(f"Trial {trial + 1}/{self.n_trials}")

            # Sample architecture
            architecture = self._sample_architecture()

            # Evaluate
            score, model = self._evaluate_architecture(architecture, X, y)
            architecture.score = score

            # Store in history
            self.search_history.append(architecture)

            # Update best
            if score > best_score:
                best_score = score
                self.best_architecture = architecture
                self.best_model = model

                if self.verbose:
                    logger.info(f"New best architecture found! Score: {score:.4f}")

        if self.verbose:
            logger.info(f"Search complete. Best score: {best_score:.4f}")

        return self.best_architecture

    def _evolutionary_search(self, X: np.ndarray, y: np.ndarray) -> NASArchitecture:
        """Evolutionary search using genetic algorithm"""
        population_size = max(10, self.n_trials // 5)
        n_generations = self.n_trials // population_size

        # Initialize population
        population = [self._sample_architecture() for _ in range(population_size)]

        # Evaluate initial population
        for arch in population:
            score, model = self._evaluate_architecture(arch, X, y)
            arch.score = score
            self.search_history.append(arch)

        best_score = max(arch.score for arch in population)
        self.best_architecture = max(population, key=lambda x: x.score)

        if self.verbose:
            logger.info(f"Initial population. Best score: {best_score:.4f}")

        # Evolution
        for generation in range(1, n_generations):
            if self.verbose:
                logger.info(f"Generation {generation}/{n_generations}")

            # Selection: keep top 50%
            population.sort(key=lambda x: x.score, reverse=True)
            parents = population[:population_size // 2]

            # Crossover and mutation
            offspring = []
            while len(offspring) < population_size - len(parents):
                # Select two parents
                parent1, parent2 = np.random.choice(parents, size=2, replace=False)

                # Crossover
                child = self._crossover(parent1, parent2)

                # Mutation
                child = self._mutate(child)

                # Evaluate
                score, model = self._evaluate_architecture(child, X, y)
                child.score = score
                self.search_history.append(child)

                offspring.append(child)

                # Update best
                if score > best_score:
                    best_score = score
                    self.best_architecture = child
                    self.best_model = model

                    if self.verbose:
                        logger.info(f"New best architecture! Score: {score:.4f}")

            # New population
            population = parents + offspring

        if self.verbose:
            logger.info(f"Evolution complete. Best score: {best_score:.4f}")

        return self.best_architecture

    def _crossover(self, parent1: NASArchitecture, parent2: NASArchitecture) -> NASArchitecture:
        """Crossover two parent architectures"""
        # Choose number of layers from one parent
        n_layers = np.random.choice([parent1.n_layers, parent2.n_layers])

        # Mix layer configs
        layer_configs = []
        for i in range(n_layers):
            if i < len(parent1.layer_configs) and i < len(parent2.layer_configs):
                # Randomly choose from parents
                if np.random.random() < 0.5:
                    layer_configs.append(parent1.layer_configs[i].copy())
                else:
                    layer_configs.append(parent2.layer_configs[i].copy())
            elif i < len(parent1.layer_configs):
                layer_configs.append(parent1.layer_configs[i].copy())
            elif i < len(parent2.layer_configs):
                layer_configs.append(parent2.layer_configs[i].copy())
            else:
                # Create new random layer
                layer_configs.append({
                    'units': np.random.choice(self.search_space.hidden_units_options),
                    'activation': np.random.choice(self.search_space.activation_options),
                    'dropout': np.random.choice(self.search_space.dropout_options)
                })

        # Mix optimizer and learning rate
        optimizer = np.random.choice([parent1.optimizer, parent2.optimizer])
        learning_rate = np.random.choice([parent1.learning_rate, parent2.learning_rate])

        return NASArchitecture(
            n_layers=n_layers,
            layer_configs=layer_configs,
            optimizer=optimizer,
            learning_rate=learning_rate
        )

    def _mutate(self, architecture: NASArchitecture, mutation_rate: float = 0.2) -> NASArchitecture:
        """Mutate an architecture"""
        mutated = NASArchitecture(
            n_layers=architecture.n_layers,
            layer_configs=[cfg.copy() for cfg in architecture.layer_configs],
            optimizer=architecture.optimizer,
            learning_rate=architecture.learning_rate
        )

        # Mutate layers
        for layer_config in mutated.layer_configs:
            if np.random.random() < mutation_rate:
                layer_config['units'] = np.random.choice(self.search_space.hidden_units_options)

            if np.random.random() < mutation_rate:
                layer_config['activation'] = np.random.choice(self.search_space.activation_options)

            if np.random.random() < mutation_rate:
                layer_config['dropout'] = np.random.choice(self.search_space.dropout_options)

        # Mutate optimizer
        if np.random.random() < mutation_rate:
            mutated.optimizer = np.random.choice(self.search_space.optimizer_options)

        # Mutate learning rate
        if np.random.random() < mutation_rate:
            mutated.learning_rate = np.random.choice(self.search_space.learning_rate_options)

        return mutated

    def get_best_model(self):
        """Get the best trained model"""
        return self.best_model

    def get_search_history(self) -> pd.DataFrame:
        """Get search history as DataFrame"""
        history_data = []
        for arch in self.search_history:
            history_data.append({
                'n_layers': arch.n_layers,
                'optimizer': arch.optimizer,
                'learning_rate': arch.learning_rate,
                'score': arch.score
            })

        return pd.DataFrame(history_data)


def auto_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    problem_type: str = 'classification',
    n_trials: int = 30,
    search_method: str = 'random',
    **kwargs
) -> Tuple[Any, NASArchitecture]:
    """
    Automated neural network architecture search

    Parameters:
    -----------
    X : np.ndarray
        Training features
    y : np.ndarray
        Training targets
    problem_type : str
        'classification' or 'regression'
    n_trials : int
        Number of architectures to try
    search_method : str
        'random' or 'evolutionary'
    **kwargs
        Additional parameters for NAS

    Returns:
    --------
    model : keras.Model
        Best trained model
    architecture : NASArchitecture
        Best architecture
    """
    nas = NeuralArchitectureSearch(
        n_trials=n_trials,
        search_method=search_method,
        problem_type=problem_type,
        **kwargs
    )

    best_architecture = nas.search(X, y)
    best_model = nas.get_best_model()

    return best_model, best_architecture

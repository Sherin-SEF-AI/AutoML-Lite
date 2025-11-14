"""
Performance estimation components for Neural Architecture Search.

This module provides efficient methods to estimate architecture performance
without full training, including early stopping, learning curve extrapolation,
and weight sharing through supernets.
"""

import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from .architecture import Architecture

# Optional deep learning imports
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    tf = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_PYTORCH = False  # Not implemented yet
except ImportError:
    HAS_PYTORCH = False
    torch = None
    nn = None
    optim = None


@dataclass
class PerformanceEstimate:
    """
    Result of performance estimation for an architecture.
    
    Attributes:
        performance: Estimated performance metric (e.g., validation accuracy)
        confidence_lower: Lower bound of confidence interval
        confidence_upper: Upper bound of confidence interval
        training_time: Time spent training/estimating (seconds)
        epochs_trained: Number of epochs trained
        metadata: Additional metadata about the estimation
    """
    performance: float
    confidence_lower: float
    confidence_upper: float
    training_time: float = 0.0
    epochs_trained: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def get_confidence_interval(self) -> Tuple[float, float]:
        """Get the confidence interval as a tuple."""
        return (self.confidence_lower, self.confidence_upper)
    
    def get_confidence_width(self) -> float:
        """Get the width of the confidence interval."""
        return self.confidence_upper - self.confidence_lower


class PerformanceEstimator(ABC):
    """
    Abstract base class for architecture performance estimation.
    
    Performance estimators evaluate neural network architectures efficiently
    without requiring full training. Different strategies include early stopping,
    learning curve extrapolation, and weight sharing.
    """
    
    def __init__(
        self,
        budget_fraction: float = 0.1,
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    ):
        """
        Initialize the performance estimator.
        
        Args:
            budget_fraction: Fraction of full training budget to use (0 < x <= 1)
            max_epochs: Maximum number of epochs for full training
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            framework: Deep learning framework ('tensorflow' or 'pytorch')
        """
        if not 0 < budget_fraction <= 1:
            raise ValueError(f"budget_fraction must be in (0, 1], got {budget_fraction}")
        
        if max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {max_epochs}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if not 0 < validation_split < 1:
            raise ValueError(f"validation_split must be in (0, 1), got {validation_split}")
        
        if framework not in ['tensorflow', 'pytorch']:
            raise ValueError(f"framework must be 'tensorflow' or 'pytorch', got '{framework}'")
        
        self.budget_fraction = budget_fraction
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.framework = framework
        
        # Check framework availability
        if framework == 'tensorflow' and not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed. Install with: pip install tensorflow")
        
        if framework == 'pytorch' and not HAS_PYTORCH:
            raise ImportError("PyTorch is required but not installed. Install with: pip install torch")
        
        # Training history
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def estimate_performance(
        self,
        architecture: Architecture,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str = 'classification'
    ) -> PerformanceEstimate:
        """
        Estimate the performance of an architecture.
        
        Args:
            architecture: Architecture to evaluate
            X: Training data features
            y: Training data labels
            problem_type: Type of problem ('classification' or 'regression')
        
        Returns:
            PerformanceEstimate with estimated performance and confidence interval
        """
        pass
    
    def _build_model_from_architecture(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        output_shape: int,
        problem_type: str = 'classification'
    ):
        """
        Build a model from an architecture specification.
        
        Args:
            architecture: Architecture to build
            input_shape: Shape of input data
            output_shape: Number of output units
            problem_type: Type of problem ('classification' or 'regression')
        
        Returns:
            Compiled model (TensorFlow or PyTorch)
        """
        if self.framework == 'tensorflow':
            return self._build_tensorflow_model(architecture, input_shape, output_shape, problem_type)
        elif self.framework == 'pytorch':
            return self._build_pytorch_model(architecture, input_shape, output_shape, problem_type)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
    
    def _build_tensorflow_model(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        output_shape: int,
        problem_type: str = 'classification'
    ):
        """Build a TensorFlow/Keras model from architecture."""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required but not installed")
        
        # Create input layer
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Track layer outputs for skip connections
        layer_outputs = {-1: x}  # -1 represents input
        
        # Build layers
        for idx, layer_config in enumerate(architecture.layers):
            layer_type = layer_config.layer_type.lower()
            params = layer_config.params.copy()
            
            # Create layer based on type
            if layer_type == 'dense':
                units = params.get('units', 64)
                activation = params.get('activation', 'relu')
                x = tf.keras.layers.Dense(units, activation=activation)(x)
            
            elif layer_type == 'dropout':
                rate = params.get('rate', 0.5)
                x = tf.keras.layers.Dropout(rate)(x)
            
            elif layer_type == 'batchnormalization' or layer_type == 'batch_normalization':
                x = tf.keras.layers.BatchNormalization()(x)
            
            elif layer_type == 'conv2d':
                filters = params.get('filters', 32)
                kernel_size = params.get('kernel_size', 3)
                activation = params.get('activation', 'relu')
                strides = params.get('strides', 1)
                padding = params.get('padding', 'same')
                x = tf.keras.layers.Conv2D(
                    filters, kernel_size, strides=strides, 
                    padding=padding, activation=activation
                )(x)
            
            elif layer_type == 'maxpooling2d' or layer_type == 'max_pooling2d':
                pool_size = params.get('pool_size', 2)
                x = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(x)
            
            elif layer_type == 'flatten':
                x = tf.keras.layers.Flatten()(x)
            
            elif layer_type == 'lstm':
                units = params.get('units', 64)
                return_sequences = params.get('return_sequences', False)
                x = tf.keras.layers.LSTM(units, return_sequences=return_sequences)(x)
            
            elif layer_type == 'gru':
                units = params.get('units', 64)
                return_sequences = params.get('return_sequences', False)
                x = tf.keras.layers.GRU(units, return_sequences=return_sequences)(x)
            
            elif layer_type == 'conv1d':
                filters = params.get('filters', 32)
                kernel_size = params.get('kernel_size', 3)
                activation = params.get('activation', 'relu')
                x = tf.keras.layers.Conv1D(filters, kernel_size, activation=activation)(x)
            
            else:
                warnings.warn(f"Unknown layer type: {layer_type}, skipping")
                continue
            
            # Store layer output for potential skip connections
            layer_outputs[idx] = x
        
        # Handle skip connections
        if architecture.connections:
            # For simplicity, we'll use Add for skip connections
            for from_idx, to_idx in architecture.connections:
                if from_idx in layer_outputs and to_idx in layer_outputs:
                    # This is a simplified implementation
                    # In practice, you'd need to handle shape mismatches
                    try:
                        layer_outputs[to_idx] = tf.keras.layers.Add()([
                            layer_outputs[from_idx],
                            layer_outputs[to_idx]
                        ])
                    except Exception as e:
                        warnings.warn(f"Could not add skip connection {from_idx}->{to_idx}: {e}")
        
        # Add output layer
        if problem_type == 'classification':
            if output_shape == 2:
                # Binary classification
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                # Multi-class classification
                outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
        else:
            # Regression
            outputs = tf.keras.layers.Dense(output_shape, activation='linear')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer_name = architecture.global_config.get('optimizer', 'adam')
        learning_rate = architecture.global_config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == 'rmsprop':
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        if problem_type == 'classification':
            if output_shape == 2:
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
    
    def _build_pytorch_model(
        self,
        architecture: Architecture,
        input_shape: Tuple[int, ...],
        output_shape: int,
        problem_type: str = 'classification'
    ):
        """Build a PyTorch model from architecture."""
        raise NotImplementedError("PyTorch model building not yet implemented")
    
    def _split_data(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and validation sets."""
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.validation_split,
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) < 20 else None  # Stratify for classification
        )
        
        return X_train, X_val, y_train, y_val
    
    def _get_num_epochs(self) -> int:
        """Get the number of epochs to train based on budget fraction."""
        return max(1, int(self.max_epochs * self.budget_fraction))
    
    def should_continue_training(
        self,
        current_epoch: int,
        metrics: List[float],
        patience: int = 5
    ) -> bool:
        """
        Determine if training should continue based on metrics history.
        
        Args:
            current_epoch: Current epoch number
            metrics: List of validation metrics (e.g., validation loss)
            patience: Number of epochs to wait for improvement
        
        Returns:
            True if training should continue, False otherwise
        """
        if len(metrics) < patience + 1:
            return True
        
        # Check if there's been improvement in the last 'patience' epochs
        recent_metrics = metrics[-patience:]
        best_recent = min(recent_metrics)
        best_overall = min(metrics[:-patience])
        
        # Continue if recent performance is better than historical
        return best_recent < best_overall
    
    def _log(self, message: str):
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            print(f"[PerformanceEstimator] {message}")
    
    def _cleanup_model(self, model):
        """
        Clean up model to free memory.
        
        This method deletes the model and clears backend session to free GPU/CPU memory.
        Essential for preventing memory leaks during architecture search.
        
        Args:
            model: Model to clean up (TensorFlow or PyTorch)
        """
        try:
            if self.framework == 'tensorflow' and HAS_TENSORFLOW:
                # Clear TensorFlow session
                import gc
                del model
                tf.keras.backend.clear_session()
                gc.collect()
                self._log("Cleaned up TensorFlow model and session")
            elif self.framework == 'pytorch' and HAS_PYTORCH:
                # Clear PyTorch model
                import gc
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                self._log("Cleaned up PyTorch model and cache")
        except Exception as e:
            self._log(f"Warning: Failed to clean up model: {e}")



class EarlyStoppingEstimator(PerformanceEstimator):
    """
    Performance estimator using early stopping.
    
    Trains architectures for a fraction of total epochs (10-20%) and uses
    early stopping to identify unpromising candidates. Provides confidence
    intervals based on validation performance variance.
    """
    
    def __init__(
        self,
        budget_fraction: float = 0.15,
        max_epochs: int = 100,
        patience: int = 5,
        min_delta: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    ):
        """
        Initialize the early stopping estimator.
        
        Args:
            budget_fraction: Fraction of full training budget (default: 0.15 = 15%)
            max_epochs: Maximum number of epochs for full training
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            framework: Deep learning framework ('tensorflow' or 'pytorch')
        """
        super().__init__(
            budget_fraction=budget_fraction,
            max_epochs=max_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            verbose=verbose,
            framework=framework
        )
        
        self.patience = patience
        self.min_delta = min_delta
    
    def estimate_performance(
        self,
        architecture: Architecture,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str = 'classification'
    ) -> PerformanceEstimate:
        """
        Estimate architecture performance using early stopping.
        
        Trains the architecture for a limited number of epochs with early stopping.
        If validation performance doesn't improve for 'patience' epochs, training
        is terminated early.
        
        Args:
            architecture: Architecture to evaluate
            X: Training data features
            y: Training data labels
            problem_type: Type of problem ('classification' or 'regression')
        
        Returns:
            PerformanceEstimate with estimated performance and confidence interval
        """
        start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # Determine input and output shapes
        if len(X.shape) == 2:
            input_shape = (X.shape[1],)
        else:
            input_shape = X.shape[1:]
        
        if problem_type == 'classification':
            output_shape = len(np.unique(y))
        else:
            output_shape = 1 if len(y.shape) == 1 else y.shape[1]
        
        # Build model
        try:
            model = self._build_model_from_architecture(
                architecture, input_shape, output_shape, problem_type
            )
        except Exception as e:
            self._log(f"Failed to build model: {e}")
            # Return poor performance estimate
            return PerformanceEstimate(
                performance=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                training_time=time.time() - start_time,
                epochs_trained=0,
                metadata={'error': str(e), 'status': 'build_failed'}
            )
        
        # Calculate number of epochs to train
        num_epochs = self._get_num_epochs()
        
        # Train with early stopping
        if self.framework == 'tensorflow':
            history, epochs_trained = self._train_tensorflow_with_early_stopping(
                model, X_train, y_train, X_val, y_val, num_epochs, problem_type
            )
        else:
            raise NotImplementedError("PyTorch training not yet implemented")
        
        # Extract performance metrics
        if problem_type == 'classification':
            metric_name = 'val_accuracy'
            # Get best validation accuracy
            if metric_name in history:
                val_metrics = history[metric_name]
                performance = max(val_metrics)
                
                # Calculate confidence interval based on variance
                mean_val = np.mean(val_metrics)
                std_val = np.std(val_metrics)
                confidence_lower = max(0.0, mean_val - 1.96 * std_val)
                confidence_upper = min(1.0, mean_val + 1.96 * std_val)
            else:
                performance = 0.0
                confidence_lower = 0.0
                confidence_upper = 0.0
        else:
            # Regression - use negative MAE as performance (higher is better)
            metric_name = 'val_mae'
            if metric_name in history:
                val_metrics = history[metric_name]
                performance = -min(val_metrics)  # Negative because lower MAE is better
                
                mean_val = np.mean(val_metrics)
                std_val = np.std(val_metrics)
                confidence_lower = -(mean_val + 1.96 * std_val)
                confidence_upper = -(mean_val - 1.96 * std_val)
            else:
                performance = 0.0
                confidence_lower = 0.0
                confidence_upper = 0.0
        
        training_time = time.time() - start_time
        
        self._log(f"Architecture {architecture.id[:8]} - Performance: {performance:.4f}, "
                  f"Epochs: {epochs_trained}/{num_epochs}, Time: {training_time:.2f}s")
        
        # Clean up model to free memory
        self._cleanup_model(model)
        
        return PerformanceEstimate(
            performance=performance,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            training_time=training_time,
            epochs_trained=epochs_trained,
            metadata={
                'history': history,
                'early_stopped': epochs_trained < num_epochs,
                'status': 'success'
            }
        )
    
    def _train_tensorflow_with_early_stopping(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int,
        problem_type: str
    ) -> Tuple[Dict[str, List[float]], int]:
        """
        Train a TensorFlow model with early stopping.
        
        Returns:
            Tuple of (history dict, number of epochs trained)
        """
        # Create early stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            min_delta=self.min_delta,
            restore_best_weights=True,
            verbose=0
        )
        
        # Train model
        try:
            history_obj = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=num_epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            history = history_obj.history
            epochs_trained = len(history['loss'])
            
        except Exception as e:
            self._log(f"Training failed: {e}")
            history = {}
            epochs_trained = 0
        
        return history, epochs_trained



class LearningCurveEstimator(PerformanceEstimator):
    """
    Performance estimator using learning curve extrapolation.
    
    Trains architectures for a fraction of total epochs and fits parametric
    models (power law, exponential) to the learning curve to extrapolate
    final performance. Provides confidence intervals based on curve fitting quality.
    """
    
    def __init__(
        self,
        budget_fraction: float = 0.2,
        max_epochs: int = 100,
        curve_model: str = 'power_law',
        min_points: int = 5,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    ):
        """
        Initialize the learning curve estimator.
        
        Args:
            budget_fraction: Fraction of full training budget (default: 0.2 = 20%)
            max_epochs: Maximum number of epochs for full training
            curve_model: Model to fit ('power_law', 'exponential', or 'both')
            min_points: Minimum number of data points needed for extrapolation
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            framework: Deep learning framework ('tensorflow' or 'pytorch')
        """
        super().__init__(
            budget_fraction=budget_fraction,
            max_epochs=max_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            verbose=verbose,
            framework=framework
        )
        
        if curve_model not in ['power_law', 'exponential', 'both']:
            raise ValueError(f"curve_model must be 'power_law', 'exponential', or 'both', got '{curve_model}'")
        
        self.curve_model = curve_model
        self.min_points = min_points
    
    def estimate_performance(
        self,
        architecture: Architecture,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str = 'classification'
    ) -> PerformanceEstimate:
        """
        Estimate architecture performance using learning curve extrapolation.
        
        Trains the architecture for a limited number of epochs, fits a parametric
        model to the learning curve, and extrapolates to predict final performance.
        
        Args:
            architecture: Architecture to evaluate
            X: Training data features
            y: Training data labels
            problem_type: Type of problem ('classification' or 'regression')
        
        Returns:
            PerformanceEstimate with extrapolated performance and confidence interval
        """
        start_time = time.time()
        
        # Split data
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # Determine input and output shapes
        if len(X.shape) == 2:
            input_shape = (X.shape[1],)
        else:
            input_shape = X.shape[1:]
        
        if problem_type == 'classification':
            output_shape = len(np.unique(y))
        else:
            output_shape = 1 if len(y.shape) == 1 else y.shape[1]
        
        # Build model
        try:
            model = self._build_model_from_architecture(
                architecture, input_shape, output_shape, problem_type
            )
        except Exception as e:
            self._log(f"Failed to build model: {e}")
            return PerformanceEstimate(
                performance=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                training_time=time.time() - start_time,
                epochs_trained=0,
                metadata={'error': str(e), 'status': 'build_failed'}
            )
        
        # Calculate number of epochs to train
        num_epochs = self._get_num_epochs()
        
        # Train and collect learning curve
        if self.framework == 'tensorflow':
            history, epochs_trained = self._train_tensorflow(
                model, X_train, y_train, X_val, y_val, num_epochs
            )
        else:
            raise NotImplementedError("PyTorch training not yet implemented")
        
        # Extract validation metrics
        if problem_type == 'classification':
            metric_name = 'val_accuracy'
            maximize = True
        else:
            metric_name = 'val_mae'
            maximize = False
        
        if metric_name not in history or len(history[metric_name]) < self.min_points:
            # Not enough data for extrapolation
            self._log(f"Insufficient data for extrapolation: {len(history.get(metric_name, []))} points")
            performance = history[metric_name][-1] if metric_name in history and history[metric_name] else 0.0
            return PerformanceEstimate(
                performance=performance,
                confidence_lower=performance * 0.9,
                confidence_upper=performance * 1.1,
                training_time=time.time() - start_time,
                epochs_trained=epochs_trained,
                metadata={'status': 'insufficient_data'}
            )
        
        val_metrics = np.array(history[metric_name])
        epochs = np.arange(1, len(val_metrics) + 1)
        
        # Extrapolate to full training
        extrapolated_performance, confidence_lower, confidence_upper, fit_quality = \
            self._extrapolate_learning_curve(
                epochs, val_metrics, self.max_epochs, maximize
            )
        
        training_time = time.time() - start_time
        
        self._log(f"Architecture {architecture.id[:8]} - Extrapolated: {extrapolated_performance:.4f}, "
                  f"Actual: {val_metrics[-1]:.4f}, Epochs: {epochs_trained}, Time: {training_time:.2f}s")
        
        # Clean up model to free memory
        self._cleanup_model(model)
        
        return PerformanceEstimate(
            performance=extrapolated_performance,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            training_time=training_time,
            epochs_trained=epochs_trained,
            metadata={
                'history': history,
                'fit_quality': fit_quality,
                'curve_model': self.curve_model,
                'status': 'success'
            }
        )
    
    def _train_tensorflow(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int
    ) -> Tuple[Dict[str, List[float]], int]:
        """
        Train a TensorFlow model without early stopping.
        
        Returns:
            Tuple of (history dict, number of epochs trained)
        """
        try:
            history_obj = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=num_epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            history = history_obj.history
            epochs_trained = len(history['loss'])
            
        except Exception as e:
            self._log(f"Training failed: {e}")
            history = {}
            epochs_trained = 0
        
        return history, epochs_trained
    
    def _extrapolate_learning_curve(
        self,
        epochs: np.ndarray,
        metrics: np.ndarray,
        target_epoch: int,
        maximize: bool = True
    ) -> Tuple[float, float, float, Dict[str, float]]:
        """
        Extrapolate learning curve to predict final performance.
        
        Fits parametric models to the learning curve and extrapolates to
        the target epoch.
        
        Args:
            epochs: Array of epoch numbers
            metrics: Array of validation metrics
            target_epoch: Epoch to extrapolate to
            maximize: Whether higher metric values are better
        
        Returns:
            Tuple of (extrapolated_performance, confidence_lower, confidence_upper, fit_quality)
        """
        from scipy.optimize import curve_fit
        
        # Define curve models
        def power_law(x, a, b, c):
            """Power law: y = a * x^b + c"""
            return a * np.power(x, b) + c
        
        def exponential(x, a, b, c):
            """Exponential: y = a * (1 - exp(-b * x)) + c"""
            return a * (1 - np.exp(-b * x)) + c
        
        predictions = []
        fit_qualities = {}
        
        # Try power law fit
        if self.curve_model in ['power_law', 'both']:
            try:
                # Initial parameter guesses
                p0_power = [metrics[-1] - metrics[0], -0.5, metrics[-1]]
                popt_power, _ = curve_fit(
                    power_law, epochs, metrics, p0=p0_power, maxfev=5000
                )
                
                # Extrapolate
                pred_power = power_law(target_epoch, *popt_power)
                
                # Calculate fit quality (R²)
                residuals = metrics - power_law(epochs, *popt_power)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((metrics - np.mean(metrics)) ** 2)
                r2_power = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                predictions.append(pred_power)
                fit_qualities['power_law_r2'] = r2_power
                
            except Exception as e:
                self._log(f"Power law fit failed: {e}")
                fit_qualities['power_law_r2'] = 0.0
        
        # Try exponential fit
        if self.curve_model in ['exponential', 'both']:
            try:
                # Initial parameter guesses
                p0_exp = [metrics[-1] - metrics[0], 0.1, metrics[0]]
                popt_exp, _ = curve_fit(
                    exponential, epochs, metrics, p0=p0_exp, maxfev=5000
                )
                
                # Extrapolate
                pred_exp = exponential(target_epoch, *popt_exp)
                
                # Calculate fit quality (R²)
                residuals = metrics - exponential(epochs, *popt_exp)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((metrics - np.mean(metrics)) ** 2)
                r2_exp = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                predictions.append(pred_exp)
                fit_qualities['exponential_r2'] = r2_exp
                
            except Exception as e:
                self._log(f"Exponential fit failed: {e}")
                fit_qualities['exponential_r2'] = 0.0
        
        # Select best prediction or average
        if not predictions:
            # Fallback: use last observed value
            extrapolated = metrics[-1]
            confidence_lower = metrics[-1] * 0.9
            confidence_upper = metrics[-1] * 1.1
        elif len(predictions) == 1:
            extrapolated = predictions[0]
            # Confidence based on fit quality
            best_r2 = max(fit_qualities.values())
            uncertainty = (1 - best_r2) * abs(extrapolated) * 0.5
            confidence_lower = extrapolated - uncertainty
            confidence_upper = extrapolated + uncertainty
        else:
            # Average predictions weighted by fit quality
            weights = np.array([fit_qualities.get('power_law_r2', 0), 
                               fit_qualities.get('exponential_r2', 0)])
            if weights.sum() > 0:
                weights = weights / weights.sum()
                extrapolated = np.average(predictions, weights=weights)
            else:
                extrapolated = np.mean(predictions)
            
            # Confidence based on prediction variance and fit quality
            pred_std = np.std(predictions)
            avg_r2 = np.mean(list(fit_qualities.values()))
            uncertainty = pred_std + (1 - avg_r2) * abs(extrapolated) * 0.3
            confidence_lower = extrapolated - uncertainty
            confidence_upper = extrapolated + uncertainty
        
        # Clip to valid range
        if maximize:
            confidence_lower = max(0.0, min(confidence_lower, 1.0))
            confidence_upper = max(0.0, min(confidence_upper, 1.0))
            extrapolated = max(0.0, min(extrapolated, 1.0))
        
        return extrapolated, confidence_lower, confidence_upper, fit_qualities



class WeightSharingEstimator(PerformanceEstimator):
    """
    Performance estimator using weight sharing through a supernet.
    
    Builds a supernet containing all possible sub-architectures in the search space.
    Sampled architectures inherit weights from the supernet, requiring only minimal
    fine-tuning. This dramatically reduces training time (100x speedup) with minimal
    accuracy loss.
    """
    
    def __init__(
        self,
        budget_fraction: float = 0.05,
        max_epochs: int = 100,
        supernet_epochs: int = 50,
        finetune_epochs: int = 5,
        batch_size: int = 32,
        validation_split: float = 0.2,
        random_state: int = 42,
        verbose: bool = False,
        framework: str = 'tensorflow'
    ):
        """
        Initialize the weight sharing estimator.
        
        Args:
            budget_fraction: Fraction of full training budget (default: 0.05 = 5%)
            max_epochs: Maximum number of epochs for full training
            supernet_epochs: Number of epochs to train the supernet
            finetune_epochs: Number of epochs to fine-tune sampled architectures
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            framework: Deep learning framework ('tensorflow' or 'pytorch')
        """
        super().__init__(
            budget_fraction=budget_fraction,
            max_epochs=max_epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            random_state=random_state,
            verbose=verbose,
            framework=framework
        )
        
        self.supernet_epochs = supernet_epochs
        self.finetune_epochs = finetune_epochs
        self.supernet = None
        self.supernet_trained = False
        self.search_space = None
    
    def build_supernet(
        self,
        search_space,
        input_shape: Tuple[int, ...],
        output_shape: int,
        problem_type: str = 'classification'
    ):
        """
        Build a supernet containing all possible sub-architectures.
        
        The supernet is a large network that contains all operations in the
        search space. Sub-architectures can be extracted by selecting specific
        paths through the supernet.
        
        Args:
            search_space: SearchSpace object defining possible architectures
            input_shape: Shape of input data
            output_shape: Number of output units
            problem_type: Type of problem ('classification' or 'regression')
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for supernet")
        
        self.search_space = search_space
        
        # For simplicity, we'll create a supernet with maximum capacity
        # In practice, this would be more sophisticated
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        
        # Build a large network with multiple parallel paths
        # This is a simplified implementation
        # Real supernets use more sophisticated architectures (e.g., ENAS, DARTS)
        
        # Add multiple dense layers with different sizes
        layer_outputs = []
        for units in [64, 128, 256, 512]:
            branch = tf.keras.layers.Dense(units, activation='relu')(x)
            branch = tf.keras.layers.Dropout(0.3)(branch)
            layer_outputs.append(branch)
        
        # Concatenate all branches
        if len(layer_outputs) > 1:
            x = tf.keras.layers.Concatenate()(layer_outputs)
        else:
            x = layer_outputs[0]
        
        # Add output layer
        if problem_type == 'classification':
            if output_shape == 2:
                outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
            else:
                outputs = tf.keras.layers.Dense(output_shape, activation='softmax')(x)
        else:
            outputs = tf.keras.layers.Dense(output_shape, activation='linear')(x)
        
        # Create supernet model
        self.supernet = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile supernet
        if problem_type == 'classification':
            if output_shape == 2:
                loss = 'binary_crossentropy'
                metrics = ['accuracy']
            else:
                loss = 'sparse_categorical_crossentropy'
                metrics = ['accuracy']
        else:
            loss = 'mse'
            metrics = ['mae']
        
        self.supernet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        self._log(f"Built supernet with {self.supernet.count_params()} parameters")
    
    def train_supernet(
        self,
        X: np.ndarray,
        y: np.ndarray
    ):
        """
        Train the supernet on the full dataset.
        
        The supernet is trained once and then reused for all architecture
        evaluations, dramatically reducing computational cost.
        
        Args:
            X: Training data features
            y: Training data labels
        """
        if self.supernet is None:
            raise ValueError("Supernet not built. Call build_supernet() first.")
        
        self._log(f"Training supernet for {self.supernet_epochs} epochs...")
        
        # Split data
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # Train supernet
        try:
            history = self.supernet.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.supernet_epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            self.supernet_trained = True
            self._log(f"Supernet training complete. "
                     f"Final val_loss: {history.history['val_loss'][-1]:.4f}")
            
        except Exception as e:
            self._log(f"Supernet training failed: {e}")
            self.supernet_trained = False
    
    def estimate_performance(
        self,
        architecture: Architecture,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str = 'classification'
    ) -> PerformanceEstimate:
        """
        Estimate architecture performance using weight sharing.
        
        Extracts a sub-architecture from the supernet, inherits weights,
        and fine-tunes for a few epochs to get performance estimate.
        
        Args:
            architecture: Architecture to evaluate
            X: Training data features
            y: Training data labels
            problem_type: Type of problem ('classification' or 'regression')
        
        Returns:
            PerformanceEstimate with estimated performance and confidence interval
        """
        start_time = time.time()
        
        # Check if supernet is trained
        if not self.supernet_trained:
            self._log("Supernet not trained. Building and training supernet...")
            
            # Determine shapes
            if len(X.shape) == 2:
                input_shape = (X.shape[1],)
            else:
                input_shape = X.shape[1:]
            
            if problem_type == 'classification':
                output_shape = len(np.unique(y))
            else:
                output_shape = 1 if len(y.shape) == 1 else y.shape[1]
            
            # Build and train supernet
            self.build_supernet(None, input_shape, output_shape, problem_type)
            self.train_supernet(X, y)
        
        # For this simplified implementation, we'll build the architecture
        # and initialize with supernet weights where possible
        # In practice, this would involve more sophisticated weight inheritance
        
        # Split data
        X_train, X_val, y_train, y_val = self._split_data(X, y)
        
        # Determine shapes
        if len(X.shape) == 2:
            input_shape = (X.shape[1],)
        else:
            input_shape = X.shape[1:]
        
        if problem_type == 'classification':
            output_shape = len(np.unique(y))
        else:
            output_shape = 1 if len(y.shape) == 1 else y.shape[1]
        
        # Build model from architecture
        try:
            model = self._build_model_from_architecture(
                architecture, input_shape, output_shape, problem_type
            )
        except Exception as e:
            self._log(f"Failed to build model: {e}")
            return PerformanceEstimate(
                performance=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                training_time=time.time() - start_time,
                epochs_trained=0,
                metadata={'error': str(e), 'status': 'build_failed'}
            )
        
        # Inherit weights from supernet (simplified - just copy compatible layers)
        self._inherit_weights_from_supernet(model)
        
        # Fine-tune for a few epochs
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.finetune_epochs,
                batch_size=self.batch_size,
                verbose=0
            )
            
            epochs_trained = len(history.history['loss'])
            
        except Exception as e:
            self._log(f"Fine-tuning failed: {e}")
            return PerformanceEstimate(
                performance=0.0,
                confidence_lower=0.0,
                confidence_upper=0.0,
                training_time=time.time() - start_time,
                epochs_trained=0,
                metadata={'error': str(e), 'status': 'training_failed'}
            )
        
        # Extract performance
        if problem_type == 'classification':
            metric_name = 'val_accuracy'
            val_metrics = history.history[metric_name]
            performance = max(val_metrics)
            
            # Confidence based on variance
            mean_val = np.mean(val_metrics)
            std_val = np.std(val_metrics)
            confidence_lower = max(0.0, mean_val - 1.96 * std_val)
            confidence_upper = min(1.0, mean_val + 1.96 * std_val)
        else:
            metric_name = 'val_mae'
            val_metrics = history.history[metric_name]
            performance = -min(val_metrics)
            
            mean_val = np.mean(val_metrics)
            std_val = np.std(val_metrics)
            confidence_lower = -(mean_val + 1.96 * std_val)
            confidence_upper = -(mean_val - 1.96 * std_val)
        
        training_time = time.time() - start_time
        
        self._log(f"Architecture {architecture.id[:8]} - Performance: {performance:.4f}, "
                  f"Fine-tuned: {epochs_trained} epochs, Time: {training_time:.2f}s")
        
        return PerformanceEstimate(
            performance=performance,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            training_time=training_time,
            epochs_trained=epochs_trained,
            metadata={
                'history': history.history,
                'weight_sharing': True,
                'supernet_trained': self.supernet_trained,
                'status': 'success'
            }
        )
    
    def _inherit_weights_from_supernet(self, model):
        """
        Inherit weights from supernet to the model.
        
        This is a simplified implementation that copies weights from
        compatible layers. In practice, this would be more sophisticated.
        
        Args:
            model: Model to initialize with supernet weights
        """
        if self.supernet is None or not self.supernet_trained:
            return
        
        # Try to copy weights from supernet layers to model layers
        supernet_layers = {layer.name: layer for layer in self.supernet.layers}
        
        for layer in model.layers:
            if layer.name in supernet_layers:
                try:
                    supernet_layer = supernet_layers[layer.name]
                    if layer.get_weights() and supernet_layer.get_weights():
                        # Check if shapes match
                        model_shapes = [w.shape for w in layer.get_weights()]
                        supernet_shapes = [w.shape for w in supernet_layer.get_weights()]
                        
                        if model_shapes == supernet_shapes:
                            layer.set_weights(supernet_layer.get_weights())
                            self._log(f"Inherited weights for layer: {layer.name}")
                except Exception as e:
                    # Skip if weight copying fails
                    pass

"""
Deep Learning Models for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import warnings

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (
        Dense, Dropout, BatchNormalization, Activation,
        Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten,
        LSTM, GRU, Bidirectional, Input, Concatenate
    )
    from tensorflow.keras.optimizers import Adam, SGD
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TENSORFLOW_AVAILABLE = True
    Model = Model  # Make Model available in global scope
except ImportError:
    TENSORFLOW_AVAILABLE = False
    Model = None  # Define Model as None when TensorFlow is not available
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DeepLearningModel:
    """
    Base class for deep learning models.
    """
    
    def __init__(
        self,
        framework: str = "tensorflow",
        model_type: str = "mlp",
        input_shape: Optional[Tuple] = None,
        output_units: int = 1,
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        random_state: int = 42,
        architecture: Optional[Any] = None
    ):
        """
        Initialize Deep Learning Model.
        
        Args:
            framework: Deep learning framework ('tensorflow' or 'pytorch')
            model_type: Type of model ('mlp', 'cnn', 'lstm', 'autoencoder', 'custom')
            input_shape: Input shape for the model
            output_units: Number of output units
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            random_state: Random state for reproducibility
            architecture: NAS Architecture object for custom models
        """
        self.framework = framework.lower()
        self.model_type = model_type.lower()
        self.input_shape = input_shape
        self.output_units = output_units
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.architecture = architecture
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False
        
        # Training history
        self.history = {}
        self.training_time = 0
        
        # Set random seeds
        self._set_random_seeds()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.random_state)
        
        if self.framework == "tensorflow" and TENSORFLOW_AVAILABLE:
            tf.random.set_seed(self.random_state)
        elif self.framework == "pytorch" and PYTORCH_AVAILABLE:
            torch.manual_seed(self.random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> 'DeepLearningModel':
        """
        Fit the deep learning model.
        
        Args:
            X: Input features
            y: Target variable
            validation_split: Fraction of data for validation
            
        Returns:
            Self
        """
        import time
        start_time = time.time()
        
        logger.info(f"Training {self.model_type} model with {self.framework}")
        
        # Preprocess data
        X_processed, y_processed = self._preprocess_data(X, y)
        
        # Build model
        self._build_model(X_processed.shape[1:])
        
        # Train model
        if self.framework == "tensorflow":
            self._train_tensorflow(X_processed, y_processed, validation_split)
        elif self.framework == "pytorch":
            self._train_pytorch(X_processed, y_processed, validation_split)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        logger.info(f"Model training completed in {self.training_time:.2f} seconds")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        
        if self.framework == "tensorflow":
            predictions = self.model.predict(X_processed, verbose=0)
        elif self.framework == "pytorch":
            predictions = self._predict_pytorch(X_processed)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        return self._postprocess_predictions(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_processed = self._preprocess_features(X)
        
        if self.framework == "tensorflow":
            probabilities = self.model.predict(X_processed, verbose=0)
        elif self.framework == "pytorch":
            probabilities = self._predict_proba_pytorch(X_processed)
        else:
            raise ValueError(f"Unsupported framework: {self.framework}")
        
        return probabilities
    
    def _preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess input data."""
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle target variable
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        
        # For classification, encode labels
        if self.output_units > 1:
            y_encoded = self.label_encoder.fit_transform(y.flatten())
            y_processed = tf.keras.utils.to_categorical(y_encoded, num_classes=self.output_units)
        else:
            y_processed = y
        
        return X_scaled, y_processed
    
    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Preprocess features for prediction."""
        return self.scaler.transform(X)
    
    def _postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Postprocess predictions."""
        if self.output_units > 1:
            # Classification: return class labels
            return self.label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        else:
            # Regression: return raw predictions
            return predictions.flatten()
    
    def _build_model(self, input_shape: Tuple):
        """Build the neural network model."""
        if self.framework == "tensorflow":
            self._build_tensorflow_model(input_shape)
        elif self.framework == "pytorch":
            self._build_pytorch_model(input_shape)
    
    def _build_tensorflow_model(self, input_shape: Tuple):
        """Build TensorFlow model."""
        if self.model_type == "custom" and self.architecture is not None:
            self.model = self._build_from_nas_architecture_tensorflow(input_shape)
        elif self.model_type == "mlp":
            self.model = self._build_mlp_tensorflow(input_shape)
        elif self.model_type == "cnn":
            self.model = self._build_cnn_tensorflow(input_shape)
        elif self.model_type == "lstm":
            self.model = self._build_lstm_tensorflow(input_shape)
        elif self.model_type == "autoencoder":
            self.model = self._build_autoencoder_tensorflow(input_shape)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _build_from_nas_architecture_tensorflow(self, input_shape: Tuple) -> Model:
        """Build model from NAS architecture."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for NAS architecture building")
        
        model = Sequential()
        
        # Build layers from architecture
        for i, layer_config in enumerate(self.architecture.layers):
            layer_type = layer_config.layer_type.lower()
            params = layer_config.params
            
            if layer_type == 'dense':
                units = params.get('units', 64)
                activation = params.get('activation', 'relu')
                
                if i == 0:
                    model.add(Dense(units, input_shape=input_shape, activation=activation))
                else:
                    model.add(Dense(units, activation=activation))
                    
            elif layer_type == 'dropout':
                rate = params.get('rate', 0.3)
                model.add(Dropout(rate))
                
            elif layer_type == 'batchnormalization' or layer_type == 'batch_normalization':
                model.add(BatchNormalization())
                
            elif layer_type == 'conv2d':
                filters = params.get('filters', 32)
                kernel_size = params.get('kernel_size', 3)
                activation = params.get('activation', 'relu')
                
                if i == 0:
                    model.add(Conv2D(filters, kernel_size, input_shape=input_shape, activation=activation))
                else:
                    model.add(Conv2D(filters, kernel_size, activation=activation))
                    
            elif layer_type == 'maxpooling2d' or layer_type == 'max_pooling2d':
                pool_size = params.get('pool_size', 2)
                model.add(MaxPooling2D(pool_size=pool_size))
                
            elif layer_type == 'flatten':
                model.add(Flatten())
                
            elif layer_type == 'lstm':
                units = params.get('units', 64)
                return_sequences = params.get('return_sequences', False)
                
                if i == 0:
                    model.add(LSTM(units, input_shape=input_shape, return_sequences=return_sequences))
                else:
                    model.add(LSTM(units, return_sequences=return_sequences))
                    
            elif layer_type == 'gru':
                units = params.get('units', 64)
                return_sequences = params.get('return_sequences', False)
                
                if i == 0:
                    model.add(GRU(units, input_shape=input_shape, return_sequences=return_sequences))
                else:
                    model.add(GRU(units, return_sequences=return_sequences))
        
        # Add output layer if not already present
        if not model.layers or model.layers[-1].output_shape[-1] != self.output_units:
            if self.output_units == 1:
                model.add(Dense(1, activation='linear'))
            else:
                model.add(Dense(self.output_units, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        if self.output_units == 1:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_mlp_tensorflow(self, input_shape: Tuple) -> Model:
        """Build Multi-Layer Perceptron with TensorFlow."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.output_units == 1:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(self.output_units, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        if self.output_units == 1:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_cnn_tensorflow(self, input_shape: Tuple) -> Model:
        """Build Convolutional Neural Network with TensorFlow."""
        model = Sequential()
        
        # Reshape input for 1D convolution
        if len(input_shape) == 1:
            model.add(Conv1D(32, 3, activation='relu', input_shape=input_shape))
            model.add(MaxPooling1D(2))
            model.add(Conv1D(64, 3, activation='relu'))
            model.add(MaxPooling1D(2))
            model.add(Flatten())
        else:
            # 2D convolution
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
        
        # Dense layers
        for units in self.hidden_layers:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.output_units == 1:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(self.output_units, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        if self.output_units == 1:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_lstm_tensorflow(self, input_shape: Tuple) -> Model:
        """Build LSTM model with TensorFlow."""
        model = Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.hidden_layers):
            if i == 0:
                model.add(LSTM(units, return_sequences=i < len(self.hidden_layers) - 1, input_shape=input_shape))
            else:
                model.add(LSTM(units, return_sequences=i < len(self.hidden_layers) - 1))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        if self.output_units == 1:
            model.add(Dense(1, activation='linear'))
        else:
            model.add(Dense(self.output_units, activation='softmax'))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        if self.output_units == 1:
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def _build_autoencoder_tensorflow(self, input_shape: Tuple) -> Model:
        """Build Autoencoder with TensorFlow."""
        # Encoder
        encoder_input = Input(shape=input_shape)
        encoded = Dense(self.hidden_layers[0], activation='relu')(encoder_input)
        encoded = Dense(self.hidden_layers[1], activation='relu')(encoded)
        
        # Decoder
        decoded = Dense(self.hidden_layers[0], activation='relu')(encoded)
        decoded = Dense(input_shape[0], activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(encoder_input, decoded)
        autoencoder.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        
        return autoencoder
    
    def _train_tensorflow(self, X: np.ndarray, y: np.ndarray, validation_split: float):
        """Train TensorFlow model."""
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=0
        ).history
    
    def _build_pytorch_model(self, input_shape: Tuple):
        """Build PyTorch model."""
        if self.model_type == "custom" and self.architecture is not None:
            self.model = self._build_from_nas_architecture_pytorch(input_shape)
        elif self.model_type == "mlp":
            self.model = self._build_mlp_pytorch(input_shape)
        else:
            raise ValueError(f"PyTorch support not implemented for {self.model_type}")
    
    def _build_from_nas_architecture_pytorch(self, input_shape: Tuple):
        """Build model from NAS architecture using PyTorch."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NAS architecture building")
        
        class NASModel(nn.Module):
            def __init__(self, architecture, input_size, output_size):
                super(NASModel, self).__init__()
                self.layers = nn.ModuleList()
                
                prev_size = input_size
                for layer_config in architecture.layers:
                    layer_type = layer_config.layer_type.lower()
                    params = layer_config.params
                    
                    if layer_type == 'dense':
                        units = params.get('units', 64)
                        self.layers.append(nn.Linear(prev_size, units))
                        prev_size = units
                        
                        activation = params.get('activation', 'relu')
                        if activation == 'relu':
                            self.layers.append(nn.ReLU())
                        elif activation == 'tanh':
                            self.layers.append(nn.Tanh())
                        elif activation == 'sigmoid':
                            self.layers.append(nn.Sigmoid())
                            
                    elif layer_type == 'dropout':
                        rate = params.get('rate', 0.3)
                        self.layers.append(nn.Dropout(rate))
                        
                    elif layer_type == 'batchnormalization' or layer_type == 'batch_normalization':
                        self.layers.append(nn.BatchNorm1d(prev_size))
                
                # Add output layer if needed
                if prev_size != output_size:
                    self.layers.append(nn.Linear(prev_size, output_size))
                
                # Add final activation
                if output_size > 1:
                    self.layers.append(nn.Softmax(dim=1))
            
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x
        
        input_size = input_shape[0] if len(input_shape) == 1 else np.prod(input_shape)
        return NASModel(self.architecture, input_size, self.output_units)
    
    def _build_mlp_pytorch(self, input_shape: Tuple):
        """Build Multi-Layer Perceptron with PyTorch."""
        class MLP(nn.Module):
            def __init__(self, input_size, hidden_layers, output_size, dropout_rate):
                super(MLP, self).__init__()
                layers = []
                
                # Input layer
                layers.append(nn.Linear(input_size, hidden_layers[0]))
                layers.append(nn.BatchNorm1d(hidden_layers[0]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                
                # Hidden layers
                for i in range(len(hidden_layers) - 1):
                    layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
                    layers.append(nn.BatchNorm1d(hidden_layers[i + 1]))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                
                # Output layer
                layers.append(nn.Linear(hidden_layers[-1], output_size))
                if output_size == 1:
                    layers.append(nn.Identity())  # Linear activation for regression
                else:
                    layers.append(nn.Softmax(dim=1))  # Softmax for classification
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        return MLP(input_shape[0], self.hidden_layers, self.output_units, self.dropout_rate)
    
    def _train_pytorch(self, X: np.ndarray, y: np.ndarray, validation_split: float):
        """Train PyTorch model."""
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        # Create data loaders
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Loss function and optimizer
        if self.output_units == 1:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Early stopping (simplified)
            if epoch > self.early_stopping_patience and total_loss < 1e-6:
                break
    
    def _predict_pytorch(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with PyTorch model."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor)
            return predictions.numpy()
    
    def _predict_proba_pytorch(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions with PyTorch model."""
        return self._predict_pytorch(X)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary."""
        return {
            'framework': self.framework,
            'model_type': self.model_type,
            'input_shape': self.input_shape,
            'output_units': self.output_units,
            'hidden_layers': self.hidden_layers,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'history': self.history
        }


class TransferLearningModel:
    """
    Transfer learning models using pre-trained networks.
    """
    
    def __init__(
        self,
        base_model: str = "resnet50",
        input_shape: Tuple = (224, 224, 3),
        num_classes: int = 1000,
        fine_tune_layers: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50
    ):
        """
        Initialize Transfer Learning Model.
        
        Args:
            base_model: Pre-trained model to use
            input_shape: Input image shape
            num_classes: Number of output classes
            fine_tune_layers: Number of layers to fine-tune
            learning_rate: Learning rate for fine-tuning
            batch_size: Batch size for training
            epochs: Number of training epochs
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required for transfer learning")
        
        self.base_model_name = base_model
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.fine_tune_layers = fine_tune_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.model = None
        self.is_fitted = False
    
    def build_model(self) -> Model:
        """Build transfer learning model."""
        # Load pre-trained model
        if self.base_model_name == "resnet50":
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == "vgg16":
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif self.base_model_name == "mobilenetv2":
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            raise ValueError(f"Unsupported base model: {self.base_model_name}")
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add classification head
        model = Sequential([
            base_model,
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def fine_tune(self, train_data: np.ndarray, train_labels: np.ndarray):
        """Fine-tune the model."""
        if self.model is None:
            self.build_model()
        
        # Unfreeze some layers for fine-tuning
        base_model = self.model.layers[0]
        base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in base_model.layers[:-self.fine_tune_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate / 10),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train
        self.model.fit(
            train_data, train_labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            verbose=1
        )
        
        self.is_fitted = True 
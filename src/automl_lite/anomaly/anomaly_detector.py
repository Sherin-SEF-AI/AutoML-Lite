"""
Comprehensive Anomaly Detection Framework

Includes multiple anomaly detection algorithms:
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Elliptic Envelope
- Autoencoder-based detection
- Statistical methods
- Ensemble anomaly detection
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnomalyReport:
    """Report containing anomaly detection results"""
    n_samples: int
    n_anomalies: int
    anomaly_ratio: float
    anomaly_indices: np.ndarray
    anomaly_scores: np.ndarray
    method: str
    threshold: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'n_samples': self.n_samples,
            'n_anomalies': self.n_anomalies,
            'anomaly_ratio': self.anomaly_ratio,
            'method': self.method,
            'threshold': self.threshold
        }


class IsolationForestDetector:
    """
    Isolation Forest for anomaly detection

    Works well for high-dimensional data and is efficient
    """

    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, str] = 'auto',
        random_state: int = 42
    ):
        """
        Initialize Isolation Forest Detector

        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies (0.0 to 0.5)
        n_estimators : int
            Number of trees
        max_samples : int or 'auto'
            Number of samples to draw for each tree
        random_state : int
            Random state for reproducibility
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )

    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the detector"""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies

        Returns:
        --------
        predictions : np.ndarray
            1 for normal, -1 for anomaly
        """
        return self.model.predict(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower = more anomalous)

        Returns:
        --------
        scores : np.ndarray
            Anomaly scores
        """
        return self.model.score_samples(X)

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method='isolation_forest',
            threshold=self.contamination
        )


class LOFDetector:
    """
    Local Outlier Factor for anomaly detection

    Detects anomalies based on local density deviation
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.1,
        novelty: bool = False
    ):
        """
        Initialize LOF Detector

        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for density estimation
        contamination : float
            Expected proportion of anomalies
        novelty : bool
            Whether to use for novelty detection (fit on normal data only)
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.novelty = novelty

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=novelty
        )

    def fit(self, X: np.ndarray) -> 'LOFDetector':
        """Fit the detector"""
        if self.novelty:
            self.model.fit(X)
        else:
            # For outlier detection, fit_predict is used
            self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.novelty:
            return self.model.predict(X)
        else:
            return self.model.fit_predict(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if self.novelty:
            return self.model.score_samples(X)
        else:
            # Negative outlier factor
            return self.model.negative_outlier_factor_

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method='local_outlier_factor',
            threshold=self.contamination
        )


class OneClassSVMDetector:
    """
    One-Class SVM for anomaly detection

    Uses support vector machine with RBF kernel
    """

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str = 'scale'
    ):
        """
        Initialize One-Class SVM Detector

        Parameters:
        -----------
        nu : float
            Upper bound on fraction of training errors (similar to contamination)
        kernel : str
            Kernel type
        gamma : str or float
            Kernel coefficient
        """
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the detector"""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        return self.model.predict(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get decision scores"""
        return self.model.score_samples(X)

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method='one_class_svm',
            threshold=self.nu
        )


class EllipticEnvelopeDetector:
    """
    Elliptic Envelope for anomaly detection

    Assumes data follows a Gaussian distribution
    """

    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Elliptic Envelope Detector

        Parameters:
        -----------
        contamination : float
            Expected proportion of anomalies
        random_state : int
            Random state
        """
        self.contamination = contamination
        self.random_state = random_state

        self.model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X: np.ndarray) -> 'EllipticEnvelopeDetector':
        """Fit the detector"""
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        return self.model.predict(X)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get Mahalanobis distances"""
        return self.model.score_samples(X)

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method='elliptic_envelope',
            threshold=self.contamination
        )


class AutoencoderDetector:
    """
    Autoencoder-based anomaly detection

    Uses reconstruction error as anomaly score
    """

    def __init__(
        self,
        encoding_dim: int = 10,
        hidden_layers: List[int] = [32, 16],
        epochs: int = 50,
        batch_size: int = 32,
        contamination: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize Autoencoder Detector

        Parameters:
        -----------
        encoding_dim : int
            Dimension of encoded representation
        hidden_layers : List[int]
            Hidden layer sizes
        epochs : int
            Training epochs
        batch_size : int
            Batch size
        contamination : float
            Expected proportion of anomalies
        random_state : int
            Random state
        """
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.contamination = contamination
        self.random_state = random_state

        self.autoencoder = None
        self.threshold = None

    def fit(self, X: np.ndarray) -> 'AutoencoderDetector':
        """Fit the autoencoder"""
        try:
            import tensorflow as tf
            from tensorflow import keras

            # Set random seed
            tf.random.set_seed(self.random_state)

            input_dim = X.shape[1]

            # Build encoder
            encoder_input = keras.layers.Input(shape=(input_dim,))
            x = encoder_input

            for hidden_dim in self.hidden_layers:
                x = keras.layers.Dense(hidden_dim, activation='relu')(x)

            encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(x)

            # Build decoder
            x = encoded
            for hidden_dim in reversed(self.hidden_layers):
                x = keras.layers.Dense(hidden_dim, activation='relu')(x)

            decoded = keras.layers.Dense(input_dim, activation='linear')(x)

            # Autoencoder model
            self.autoencoder = keras.Model(encoder_input, decoded)
            self.autoencoder.compile(optimizer='adam', loss='mse')

            # Train
            self.autoencoder.fit(
                X, X,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                verbose=0,
                validation_split=0.1
            )

            # Compute threshold based on training reconstruction errors
            reconstruction = self.autoencoder.predict(X, verbose=0)
            reconstruction_errors = np.mean(np.square(X - reconstruction), axis=1)

            # Threshold at contamination percentile
            self.threshold = np.percentile(reconstruction_errors, (1 - self.contamination) * 100)

        except ImportError:
            logger.error("TensorFlow not installed. Autoencoder detector not available.")
            raise

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        scores = self.predict_scores(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get reconstruction errors"""
        if self.autoencoder is None:
            raise ValueError("Model not fitted yet")

        reconstruction = self.autoencoder.predict(X, verbose=0)
        reconstruction_errors = np.mean(np.square(X - reconstruction), axis=1)
        return reconstruction_errors

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method='autoencoder',
            threshold=self.threshold
        )


class StatisticalDetector:
    """
    Statistical anomaly detection using Z-score and IQR methods
    """

    def __init__(
        self,
        method: str = 'zscore',  # 'zscore' or 'iqr'
        threshold: float = 3.0  # Z-score threshold or IQR multiplier
    ):
        """
        Initialize Statistical Detector

        Parameters:
        -----------
        method : str
            'zscore' or 'iqr'
        threshold : float
            For zscore: number of standard deviations
            For IQR: multiplier for IQR range (typically 1.5 or 3.0)
        """
        self.method = method
        self.threshold = threshold

        self.means_ = None
        self.stds_ = None
        self.q1_ = None
        self.q3_ = None
        self.iqr_ = None

    def fit(self, X: np.ndarray) -> 'StatisticalDetector':
        """Fit the detector"""
        if self.method == 'zscore':
            self.means_ = np.mean(X, axis=0)
            self.stds_ = np.std(X, axis=0)
        elif self.method == 'iqr':
            self.q1_ = np.percentile(X, 25, axis=0)
            self.q3_ = np.percentile(X, 75, axis=0)
            self.iqr_ = self.q3_ - self.q1_
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies"""
        if self.method == 'zscore':
            z_scores = np.abs((X - self.means_) / (self.stds_ + 1e-10))
            is_anomaly = np.any(z_scores > self.threshold, axis=1)
        elif self.method == 'iqr':
            lower_bound = self.q1_ - self.threshold * self.iqr_
            upper_bound = self.q3_ + self.threshold * self.iqr_
            is_anomaly = np.any((X < lower_bound) | (X > upper_bound), axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return np.where(is_anomaly, -1, 1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        if self.method == 'zscore':
            z_scores = np.abs((X - self.means_) / (self.stds_ + 1e-10))
            # Max z-score across features
            return np.max(z_scores, axis=1)
        elif self.method == 'iqr':
            lower_bound = self.q1_ - self.threshold * self.iqr_
            upper_bound = self.q3_ + self.threshold * self.iqr_

            # Distance from IQR range
            lower_distances = np.maximum(0, lower_bound - X)
            upper_distances = np.maximum(0, X - upper_bound)
            distances = lower_distances + upper_distances

            # Max distance across features
            return np.max(distances, axis=1)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method=f'statistical_{self.method}',
            threshold=self.threshold
        )


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods

    Uses voting or score aggregation to improve robustness
    """

    def __init__(
        self,
        detectors: Optional[List] = None,
        voting: str = 'soft',  # 'soft' or 'hard'
        contamination: float = 0.1,
        scale_features: bool = True
    ):
        """
        Initialize Ensemble Anomaly Detector

        Parameters:
        -----------
        detectors : List, optional
            List of detector instances. If None, uses default ensemble
        voting : str
            'soft': aggregate scores, 'hard': majority voting
        contamination : float
            Expected proportion of anomalies
        scale_features : bool
            Whether to scale features before detection
        """
        self.detectors = detectors
        self.voting = voting
        self.contamination = contamination
        self.scale_features = scale_features

        self.scaler = StandardScaler() if scale_features else None
        self.fitted_detectors = []

    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all detectors"""
        # Scale features if needed
        if self.scale_features:
            X = self.scaler.fit_transform(X)

        # Initialize default detectors if none provided
        if self.detectors is None:
            self.detectors = [
                IsolationForestDetector(contamination=self.contamination),
                LOFDetector(contamination=self.contamination),
                EllipticEnvelopeDetector(contamination=self.contamination)
            ]

        # Fit each detector
        self.fitted_detectors = []
        for detector in self.detectors:
            try:
                detector.fit(X)
                self.fitted_detectors.append(detector)
            except Exception as e:
                logger.warning(f"Detector {detector.__class__.__name__} failed to fit: {e}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble"""
        # Scale if needed
        if self.scale_features:
            X = self.scaler.transform(X)

        if self.voting == 'hard':
            # Majority voting
            predictions = []
            for detector in self.fitted_detectors:
                predictions.append(detector.predict(X))

            predictions = np.array(predictions)  # Shape: (n_detectors, n_samples)

            # Count votes for anomaly (-1)
            anomaly_votes = np.sum(predictions == -1, axis=0)
            majority_threshold = len(self.fitted_detectors) / 2

            return np.where(anomaly_votes > majority_threshold, -1, 1)

        else:  # soft voting
            # Aggregate scores
            scores = self.predict_scores(X)

            # Threshold at contamination level
            threshold = np.percentile(scores, (1 - self.contamination) * 100)

            return np.where(scores > threshold, -1, 1)

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Get aggregated anomaly scores"""
        # Scale if needed
        if self.scale_features:
            X = self.scaler.transform(X)

        all_scores = []
        for detector in self.fitted_detectors:
            scores = detector.predict_scores(X)

            # Normalize scores to [0, 1]
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)

            all_scores.append(scores_normalized)

        # Average scores
        aggregated_scores = np.mean(all_scores, axis=0)

        return aggregated_scores

    def detect(self, X: np.ndarray) -> AnomalyReport:
        """Full anomaly detection with report"""
        predictions = self.predict(X)
        scores = self.predict_scores(X)

        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]

        return AnomalyReport(
            n_samples=len(X),
            n_anomalies=len(anomaly_indices),
            anomaly_ratio=len(anomaly_indices) / len(X),
            anomaly_indices=anomaly_indices,
            anomaly_scores=scores,
            method=f'ensemble_{self.voting}',
            threshold=self.contamination
        )


def detect_anomalies(
    X: np.ndarray,
    method: str = 'ensemble',
    contamination: float = 0.1,
    **kwargs
) -> AnomalyReport:
    """
    Factory function for anomaly detection

    Parameters:
    -----------
    X : np.ndarray
        Data to check for anomalies
    method : str
        Detection method: 'isolation_forest', 'lof', 'one_class_svm',
        'elliptic_envelope', 'autoencoder', 'statistical', 'ensemble'
    contamination : float
        Expected proportion of anomalies
    **kwargs
        Additional parameters for specific methods

    Returns:
    --------
    report : AnomalyReport
        Detection results
    """
    if method == 'isolation_forest':
        detector = IsolationForestDetector(contamination=contamination, **kwargs)
    elif method == 'lof':
        detector = LOFDetector(contamination=contamination, **kwargs)
    elif method == 'one_class_svm':
        detector = OneClassSVMDetector(nu=contamination, **kwargs)
    elif method == 'elliptic_envelope':
        detector = EllipticEnvelopeDetector(contamination=contamination, **kwargs)
    elif method == 'autoencoder':
        detector = AutoencoderDetector(contamination=contamination, **kwargs)
    elif method == 'statistical':
        detector = StatisticalDetector(**kwargs)
    elif method == 'ensemble':
        detector = EnsembleAnomalyDetector(contamination=contamination, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    detector.fit(X)
    return detector.detect(X)

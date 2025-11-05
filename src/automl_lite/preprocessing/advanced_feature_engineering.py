"""
Advanced Feature Engineering Module

Includes:
- Target encoding with cross-validation
- Frequency encoding
- Lag features for time series
- Rolling window statistics
- Groupby aggregations
- Fourier features
- Wavelet transformations
- Advanced binning strategies
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
import logging
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """
    Target Encoding with Cross-Validation to prevent overfitting

    Encodes categorical features with target statistics using out-of-fold predictions
    """

    def __init__(
        self,
        categorical_features: List[str],
        n_folds: int = 5,
        smoothing: float = 1.0,
        min_samples_leaf: int = 1,
        noise_level: float = 0.01,
        random_state: int = 42
    ):
        """
        Initialize Target Encoder with CV

        Parameters:
        -----------
        categorical_features : List[str]
            List of categorical feature names
        n_folds : int
            Number of folds for cross-validation
        smoothing : float
            Smoothing factor (higher = more regularization)
        min_samples_leaf : int
            Minimum samples required to encode a category
        noise_level : float
            Small noise to add for regularization
        random_state : int
            Random state for reproducibility
        """
        self.categorical_features = categorical_features
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.random_state = random_state

        self.encodings_ = {}
        self.global_means_ = {}

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'TargetEncoderCV':
        """Fit target encoder"""
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue

            # Compute global mean
            self.global_means_[feature] = np.mean(y)

            # Compute encodings for each category
            encodings = {}
            for category in X[feature].unique():
                mask = X[feature] == category
                category_target = y[mask]

                if len(category_target) >= self.min_samples_leaf:
                    # Smoothed mean
                    category_mean = np.mean(category_target)
                    n_samples = len(category_target)

                    # Apply smoothing
                    smoothed_mean = (
                        category_mean * n_samples + self.global_means_[feature] * self.smoothing
                    ) / (n_samples + self.smoothing)

                    encodings[category] = smoothed_mean
                else:
                    encodings[category] = self.global_means_[feature]

            self.encodings_[feature] = encodings

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical features using target encoding"""
        X_transformed = X.copy()

        for feature in self.categorical_features:
            if feature not in X.columns:
                continue

            # Map categories to encodings
            X_transformed[f'{feature}_target_encoded'] = X[feature].map(
                lambda x: self.encodings_[feature].get(x, self.global_means_[feature])
            )

            # Add small noise for regularization
            if self.noise_level > 0:
                noise = np.random.normal(0, self.noise_level, len(X_transformed))
                X_transformed[f'{feature}_target_encoded'] += noise

        return X_transformed

    def fit_transform_cv(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        """
        Fit and transform using cross-validation to prevent overfitting

        Returns out-of-fold encodings for training data
        """
        X_transformed = X.copy()

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        # Initialize encoded columns
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            X_transformed[f'{feature}_target_encoded'] = 0.0

        # Generate out-of-fold encodings
        for train_idx, val_idx in kfold.split(X):
            X_train, y_train = X.iloc[train_idx], y[train_idx]

            # Fit on training fold
            for feature in self.categorical_features:
                if feature not in X.columns:
                    continue

                # Compute global mean for this fold
                global_mean = np.mean(y_train)

                # Compute encodings
                encodings = {}
                for category in X_train[feature].unique():
                    mask = X_train[feature] == category
                    category_target = y_train[mask]

                    if len(category_target) >= self.min_samples_leaf:
                        category_mean = np.mean(category_target)
                        n_samples = len(category_target)

                        smoothed_mean = (
                            category_mean * n_samples + global_mean * self.smoothing
                        ) / (n_samples + self.smoothing)

                        encodings[category] = smoothed_mean
                    else:
                        encodings[category] = global_mean

                # Transform validation fold
                X_transformed.loc[X.index[val_idx], f'{feature}_target_encoded'] = X.iloc[val_idx][feature].map(
                    lambda x: encodings.get(x, global_mean)
                )

        # Fit on entire dataset for future transforms
        self.fit(X, y)

        return X_transformed


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Frequency Encoding: Encodes categories by their frequency

    Useful for high-cardinality categorical features
    """

    def __init__(self, categorical_features: List[str]):
        self.categorical_features = categorical_features
        self.frequencies_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FrequencyEncoder':
        """Fit frequency encoder"""
        for feature in self.categorical_features:
            if feature not in X.columns:
                continue

            # Compute frequencies
            value_counts = X[feature].value_counts(normalize=True)
            self.frequencies_[feature] = value_counts.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using frequency encoding"""
        X_transformed = X.copy()

        for feature in self.categorical_features:
            if feature not in X.columns:
                continue

            # Map to frequencies
            X_transformed[f'{feature}_frequency'] = X[feature].map(
                lambda x: self.frequencies_[feature].get(x, 0.0)
            )

        return X_transformed


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Create lag features for time series data

    Useful for capturing temporal dependencies
    """

    def __init__(
        self,
        target_column: str,
        lags: List[int] = [1, 2, 3, 7, 14, 30],
        time_column: Optional[str] = None
    ):
        """
        Initialize Lag Feature Creator

        Parameters:
        -----------
        target_column : str
            Name of the target column to create lags for
        lags : List[int]
            List of lag periods
        time_column : str, optional
            Name of time/date column for proper ordering
        """
        self.target_column = target_column
        self.lags = lags
        self.time_column = time_column

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'LagFeatureCreator':
        """Fit (no-op for this transformer)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create lag features"""
        X_transformed = X.copy()

        # Sort by time if time column is provided
        if self.time_column and self.time_column in X.columns:
            X_transformed = X_transformed.sort_values(self.time_column)

        # Create lag features
        for lag in self.lags:
            X_transformed[f'{self.target_column}_lag_{lag}'] = X_transformed[self.target_column].shift(lag)

        return X_transformed


class RollingWindowFeatures(BaseEstimator, TransformerMixin):
    """
    Create rolling window statistics

    Useful for time series and sequential data
    """

    def __init__(
        self,
        feature_columns: List[str],
        windows: List[int] = [3, 7, 14, 30],
        statistics: List[str] = ['mean', 'std', 'min', 'max'],
        time_column: Optional[str] = None
    ):
        """
        Initialize Rolling Window Feature Creator

        Parameters:
        -----------
        feature_columns : List[str]
            Columns to compute rolling statistics for
        windows : List[int]
            Window sizes
        statistics : List[str]
            Statistics to compute: 'mean', 'std', 'min', 'max', 'sum'
        time_column : str, optional
            Time column for proper ordering
        """
        self.feature_columns = feature_columns
        self.windows = windows
        self.statistics = statistics
        self.time_column = time_column

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'RollingWindowFeatures':
        """Fit (no-op)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        X_transformed = X.copy()

        # Sort by time if provided
        if self.time_column and self.time_column in X.columns:
            X_transformed = X_transformed.sort_values(self.time_column)

        # Create rolling features
        for feature in self.feature_columns:
            if feature not in X.columns:
                continue

            for window in self.windows:
                for stat in self.statistics:
                    if stat == 'mean':
                        X_transformed[f'{feature}_rolling_{window}_mean'] = X_transformed[feature].rolling(window).mean()
                    elif stat == 'std':
                        X_transformed[f'{feature}_rolling_{window}_std'] = X_transformed[feature].rolling(window).std()
                    elif stat == 'min':
                        X_transformed[f'{feature}_rolling_{window}_min'] = X_transformed[feature].rolling(window).min()
                    elif stat == 'max':
                        X_transformed[f'{feature}_rolling_{window}_max'] = X_transformed[feature].rolling(window).max()
                    elif stat == 'sum':
                        X_transformed[f'{feature}_rolling_{window}_sum'] = X_transformed[feature].rolling(window).sum()

        return X_transformed


class GroupByAggregator(BaseEstimator, TransformerMixin):
    """
    Create features using groupby aggregations

    Useful for creating summary statistics by groups
    """

    def __init__(
        self,
        groupby_columns: List[str],
        agg_columns: List[str],
        aggregations: List[str] = ['mean', 'sum', 'count', 'min', 'max', 'std']
    ):
        """
        Initialize GroupBy Aggregator

        Parameters:
        -----------
        groupby_columns : List[str]
            Columns to group by
        agg_columns : List[str]
            Columns to aggregate
        aggregations : List[str]
            Aggregation functions
        """
        self.groupby_columns = groupby_columns
        self.agg_columns = agg_columns
        self.aggregations = aggregations

        self.agg_mappings_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'GroupByAggregator':
        """Fit by computing aggregations on training data"""
        for groupby_col in self.groupby_columns:
            if groupby_col not in X.columns:
                continue

            for agg_col in self.agg_columns:
                if agg_col not in X.columns:
                    continue

                for agg_func in self.aggregations:
                    key = f'{groupby_col}_{agg_col}_{agg_func}'

                    # Compute aggregation
                    if agg_func == 'mean':
                        agg_result = X.groupby(groupby_col)[agg_col].mean()
                    elif agg_func == 'sum':
                        agg_result = X.groupby(groupby_col)[agg_col].sum()
                    elif agg_func == 'count':
                        agg_result = X.groupby(groupby_col)[agg_col].count()
                    elif agg_func == 'min':
                        agg_result = X.groupby(groupby_col)[agg_col].min()
                    elif agg_func == 'max':
                        agg_result = X.groupby(groupby_col)[agg_col].max()
                    elif agg_func == 'std':
                        agg_result = X.groupby(groupby_col)[agg_col].std()
                    else:
                        continue

                    self.agg_mappings_[key] = agg_result.to_dict()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform by mapping aggregations"""
        X_transformed = X.copy()

        for key, mapping in self.agg_mappings_.items():
            groupby_col = key.split('_')[0]

            if groupby_col in X.columns:
                # Map aggregated values
                default_value = np.mean(list(mapping.values())) if mapping else 0.0
                X_transformed[key] = X[groupby_col].map(lambda x: mapping.get(x, default_value))

        return X_transformed


class FourierFeatures(BaseEstimator, TransformerMixin):
    """
    Create Fourier features for capturing periodic patterns

    Useful for time series with seasonality
    """

    def __init__(
        self,
        n_components: int = 5,
        period: Optional[float] = None
    ):
        """
        Initialize Fourier Features

        Parameters:
        -----------
        n_components : int
            Number of Fourier components to create
        period : float, optional
            Period of seasonality (e.g., 365 for yearly, 7 for weekly)
        """
        self.n_components = n_components
        self.period = period

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'FourierFeatures':
        """Fit (no-op)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create Fourier features"""
        X_transformed = X.copy()

        # Assume index represents time steps
        time_index = np.arange(len(X))

        if self.period is None:
            # Use length as period
            self.period = len(X)

        # Create sine and cosine features
        for k in range(1, self.n_components + 1):
            X_transformed[f'fourier_sin_{k}'] = np.sin(2 * np.pi * k * time_index / self.period)
            X_transformed[f'fourier_cos_{k}'] = np.cos(2 * np.pi * k * time_index / self.period)

        return X_transformed


class WaveletTransformer(BaseEstimator, TransformerMixin):
    """
    Apply wavelet transformation for feature extraction

    Useful for capturing multi-scale patterns
    """

    def __init__(
        self,
        feature_columns: List[str],
        wavelet: str = 'haar',
        levels: int = 3
    ):
        """
        Initialize Wavelet Transformer

        Parameters:
        -----------
        feature_columns : List[str]
            Columns to apply wavelet transform
        wavelet : str
            Wavelet type (e.g., 'haar', 'db1', 'sym2')
        levels : int
            Number of decomposition levels
        """
        self.feature_columns = feature_columns
        self.wavelet = wavelet
        self.levels = levels

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'WaveletTransformer':
        """Fit (no-op)"""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply wavelet transformation"""
        import pywt

        X_transformed = X.copy()

        for feature in self.feature_columns:
            if feature not in X.columns:
                continue

            # Get feature values
            feature_values = X[feature].values

            try:
                # Apply wavelet decomposition
                coeffs = pywt.wavedec(feature_values, self.wavelet, level=self.levels)

                # Create features from coefficients
                for level, coeff in enumerate(coeffs):
                    # Statistics of coefficients at each level
                    X_transformed[f'{feature}_wavelet_level_{level}_mean'] = np.mean(coeff)
                    X_transformed[f'{feature}_wavelet_level_{level}_std'] = np.std(coeff)
                    X_transformed[f'{feature}_wavelet_level_{level}_max'] = np.max(coeff)
                    X_transformed[f'{feature}_wavelet_level_{level}_min'] = np.min(coeff)

            except Exception as e:
                logger.warning(f"Wavelet transform failed for {feature}: {e}")

        return X_transformed


class AdaptiveBinner(BaseEstimator, TransformerMixin):
    """
    Advanced binning strategies for numerical features

    Includes: quantile binning, decision tree binning, and custom binning
    """

    def __init__(
        self,
        feature_columns: List[str],
        n_bins: int = 10,
        strategy: str = 'quantile',  # 'quantile', 'uniform', 'kmeans', 'tree'
        encode: str = 'ordinal'  # 'ordinal', 'onehot'
    ):
        """
        Initialize Adaptive Binner

        Parameters:
        -----------
        feature_columns : List[str]
            Columns to bin
        n_bins : int
            Number of bins
        strategy : str
            Binning strategy
        encode : str
            Encoding method
        """
        self.feature_columns = feature_columns
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode

        self.binners_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'AdaptiveBinner':
        """Fit binners"""
        from sklearn.preprocessing import KBinsDiscretizer

        for feature in self.feature_columns:
            if feature not in X.columns:
                continue

            feature_values = X[feature].values.reshape(-1, 1)

            # Create binner
            binner = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode=self.encode,
                strategy=self.strategy if self.strategy != 'tree' else 'quantile'
            )

            binner.fit(feature_values, y)
            self.binners_[feature] = binner

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using binning"""
        X_transformed = X.copy()

        for feature, binner in self.binners_.items():
            if feature not in X.columns:
                continue

            feature_values = X[feature].values.reshape(-1, 1)

            # Transform
            if self.encode == 'ordinal':
                X_transformed[f'{feature}_binned'] = binner.transform(feature_values).flatten()
            else:  # onehot
                binned = binner.transform(feature_values)
                for i in range(binned.shape[1]):
                    X_transformed[f'{feature}_bin_{i}'] = binned[:, i]

        return X_transformed


class AdvancedFeatureEngineer:
    """
    Comprehensive advanced feature engineering pipeline

    Combines multiple advanced feature engineering techniques
    """

    def __init__(
        self,
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        time_column: Optional[str] = None,
        enable_target_encoding: bool = True,
        enable_frequency_encoding: bool = True,
        enable_lag_features: bool = False,
        enable_rolling_features: bool = False,
        enable_groupby_agg: bool = False,
        enable_fourier: bool = False,
        enable_wavelet: bool = False,
        enable_binning: bool = True,
        verbose: bool = True
    ):
        """
        Initialize Advanced Feature Engineer

        Parameters:
        -----------
        categorical_features : List[str], optional
            List of categorical feature names
        numerical_features : List[str], optional
            List of numerical feature names
        time_column : str, optional
            Time/date column name
        enable_* : bool
            Flags to enable specific feature engineering techniques
        verbose : bool
            Whether to print progress
        """
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.time_column = time_column
        self.enable_target_encoding = enable_target_encoding
        self.enable_frequency_encoding = enable_frequency_encoding
        self.enable_lag_features = enable_lag_features
        self.enable_rolling_features = enable_rolling_features
        self.enable_groupby_agg = enable_groupby_agg
        self.enable_fourier = enable_fourier
        self.enable_wavelet = enable_wavelet
        self.enable_binning = enable_binning
        self.verbose = verbose

        # Transformers
        self.transformers_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> 'AdvancedFeatureEngineer':
        """Fit all enabled transformers"""
        if self.verbose:
            logger.info("Fitting advanced feature engineering pipeline")

        # Target encoding
        if self.enable_target_encoding and y is not None and self.categorical_features:
            self.transformers_['target_encoder'] = TargetEncoderCV(self.categorical_features)
            self.transformers_['target_encoder'].fit(X, y)

        # Frequency encoding
        if self.enable_frequency_encoding and self.categorical_features:
            self.transformers_['frequency_encoder'] = FrequencyEncoder(self.categorical_features)
            self.transformers_['frequency_encoder'].fit(X)

        # Binning
        if self.enable_binning and self.numerical_features:
            self.transformers_['binner'] = AdaptiveBinner(self.numerical_features[:5])  # Limit to first 5
            self.transformers_['binner'].fit(X, y)

        # Fourier features
        if self.enable_fourier:
            self.transformers_['fourier'] = FourierFeatures(n_components=3)
            self.transformers_['fourier'].fit(X)

        if self.verbose:
            logger.info(f"Fitted {len(self.transformers_)} feature engineering transformers")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations"""
        X_transformed = X.copy()

        for name, transformer in self.transformers_.items():
            try:
                if self.verbose:
                    logger.info(f"Applying {name}")
                X_transformed = transformer.transform(X_transformed)
            except Exception as e:
                logger.warning(f"Transformer {name} failed: {e}")

        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)

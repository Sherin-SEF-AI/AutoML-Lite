"""
Time Series Forecasting for AutoML Lite.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import warnings

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Install with: pip install statsmodels")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. Install with: pip install tensorflow")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)


class TimeSeriesForecaster:
    """
    Comprehensive time series forecasting with multiple models.
    """
    
    def __init__(
        self,
        enable_arima: bool = True,
        enable_prophet: bool = True,
        enable_lstm: bool = True,
        enable_seasonal_decomposition: bool = True,
        forecast_horizon: int = 12,
        seasonality_detection: bool = True,
        auto_arima: bool = True,
        lstm_units: int = 50,
        lstm_layers: int = 2,
        random_state: int = 42
    ):
        """
        Initialize Time Series Forecaster.
        
        Args:
            enable_arima: Enable ARIMA models
            enable_prophet: Enable Prophet models
            enable_lstm: Enable LSTM models
            enable_seasonal_decomposition: Enable seasonal decomposition
            forecast_horizon: Number of periods to forecast
            seasonality_detection: Auto-detect seasonality
            auto_arima: Use auto ARIMA for parameter selection
            lstm_units: Number of LSTM units
            lstm_layers: Number of LSTM layers
            random_state: Random state for reproducibility
        """
        self.enable_arima = enable_arima and STATSMODELS_AVAILABLE
        self.enable_prophet = enable_prophet and PROPHET_AVAILABLE
        self.enable_lstm = enable_lstm and TENSORFLOW_AVAILABLE
        self.enable_seasonal_decomposition = enable_seasonal_decomposition and STATSMODELS_AVAILABLE
        self.forecast_horizon = forecast_horizon
        self.seasonality_detection = seasonality_detection
        self.auto_arima = auto_arima
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.random_state = random_state
        
        # Models
        self.arima_model = None
        self.prophet_model = None
        self.lstm_model = None
        self.seasonal_decomposition = None
        
        # Results
        self.forecasts = {}
        self.model_scores = {}
        self.best_model = None
        self.best_forecast = None
        
        # Data preprocessing
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> 'TimeSeriesForecaster':
        """
        Fit time series models.
        
        Args:
            y: Time series data
            X: Exogenous variables (optional)
            
        Returns:
            Self
        """
        logger.info("Starting time series forecasting...")
        
        self.y = y.copy()
        self.X = X.copy() if X is not None else None
        
        # Ensure index is datetime
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("Time series index must be DatetimeIndex")
        
        # Detect seasonality
        if self.seasonality_detection:
            self._detect_seasonality()
        
        # Seasonal decomposition
        if self.enable_seasonal_decomposition:
            self._perform_seasonal_decomposition()
        
        # Fit models
        if self.enable_arima:
            self._fit_arima()
        
        if self.enable_prophet:
            self._fit_prophet()
        
        if self.enable_lstm:
            self._fit_lstm()
        
        # Select best model
        self._select_best_model()
        
        self.is_fitted = True
        logger.info("Time series forecasting completed")
        
        return self
    
    def _detect_seasonality(self):
        """Detect seasonality in the time series."""
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                self.y, 
                model='additive', 
                period=self._estimate_period()
            )
            
            # Calculate seasonality strength
            seasonal_strength = np.var(decomposition.seasonal) / np.var(decomposition.resid + decomposition.seasonal)
            
            self.seasonality_info = {
                'has_seasonality': seasonal_strength > 0.1,
                'seasonal_strength': seasonal_strength,
                'period': self._estimate_period(),
                'decomposition': decomposition
            }
            
            logger.info(f"Seasonality detected: {self.seasonality_info['has_seasonality']}")
            
        except Exception as e:
            logger.warning(f"Failed to detect seasonality: {str(e)}")
            self.seasonality_info = {'has_seasonality': False, 'seasonal_strength': 0}
    
    def _estimate_period(self) -> int:
        """Estimate the seasonal period."""
        # Simple heuristics for common periods
        freq = pd.infer_freq(self.y.index)
        
        if freq == 'D':  # Daily
            return 7  # Weekly seasonality
        elif freq == 'W':  # Weekly
            return 52  # Yearly seasonality
        elif freq == 'M':  # Monthly
            return 12  # Yearly seasonality
        elif freq == 'Q':  # Quarterly
            return 4  # Yearly seasonality
        else:
            # Default to 12 for monthly-like data
            return min(12, len(self.y) // 4)
    
    def _perform_seasonal_decomposition(self):
        """Perform seasonal decomposition."""
        try:
            period = self.seasonality_info.get('period', self._estimate_period())
            
            self.seasonal_decomposition = seasonal_decompose(
                self.y,
                model='additive',
                period=period
            )
            
            logger.info("Seasonal decomposition completed")
            
        except Exception as e:
            logger.warning(f"Failed to perform seasonal decomposition: {str(e)}")
    
    def _fit_arima(self):
        """Fit ARIMA model."""
        try:
            logger.info("Fitting ARIMA model...")
            
            if self.auto_arima:
                # Use auto ARIMA for parameter selection
                from pmdarima import auto_arima
                
                self.arima_model = auto_arima(
                    self.y,
                    seasonal=self.seasonality_info.get('has_seasonality', False),
                    m=self.seasonality_info.get('period', 1),
                    suppress_warnings=True,
                    error_action='ignore',
                    stepwise=True,
                    random_state=self.random_state
                )
            else:
                # Manual ARIMA with default parameters
                self.arima_model = ARIMA(self.y, order=(1, 1, 1))
                self.arima_model = self.arima_model.fit()
            
            logger.info("ARIMA model fitted successfully")
            
        except Exception as e:
            logger.warning(f"Failed to fit ARIMA model: {str(e)}")
            self.arima_model = None
    
    def _fit_prophet(self):
        """Fit Prophet model."""
        try:
            logger.info("Fitting Prophet model...")
            
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': self.y.index,
                'y': self.y.values
            })
            
            # Create Prophet model
            self.prophet_model = Prophet(
                yearly_seasonality=self.seasonality_info.get('has_seasonality', False),
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive'
            )
            
            # Add exogenous variables if available
            if self.X is not None:
                for col in self.X.columns:
                    self.prophet_model.add_regressor(col)
                prophet_data = pd.concat([prophet_data, self.X.reset_index(drop=True)], axis=1)
            
            # Fit model
            self.prophet_model.fit(prophet_data)
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.warning(f"Failed to fit Prophet model: {str(e)}")
            self.prophet_model = None
    
    def _fit_lstm(self):
        """Fit LSTM model."""
        try:
            logger.info("Fitting LSTM model...")
            
            # Prepare data for LSTM
            data = self.y.values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(data)
            
            # Create sequences
            X, y = self._create_sequences(scaled_data, lookback=12)
            
            # Build LSTM model
            self.lstm_model = Sequential()
            
            # Add LSTM layers
            for i in range(self.lstm_layers):
                if i == 0:
                    self.lstm_model.add(LSTM(
                        units=self.lstm_units,
                        return_sequences=i < self.lstm_layers - 1,
                        input_shape=(X.shape[1], X.shape[2])
                    ))
                else:
                    self.lstm_model.add(LSTM(
                        units=self.lstm_units,
                        return_sequences=i < self.lstm_layers - 1
                    ))
                self.lstm_model.add(Dropout(0.2))
            
            # Add output layer
            self.lstm_model.add(Dense(1))
            
            # Compile model
            self.lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse'
            )
            
            # Fit model
            self.lstm_model.fit(
                X, y,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            logger.info("LSTM model fitted successfully")
            
        except Exception as e:
            logger.warning(f"Failed to fit LSTM model: {str(e)}")
            self.lstm_model = None
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM."""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _select_best_model(self):
        """Select the best model based on validation performance."""
        try:
            # Use last 20% of data for validation
            split_idx = int(len(self.y) * 0.8)
            train_y = self.y[:split_idx]
            val_y = self.y[split_idx:]
            
            best_score = float('inf')
            best_model_name = None
            
            # Evaluate ARIMA
            if self.arima_model is not None:
                try:
                    arima_forecast = self.arima_model.predict(
                        start=split_idx,
                        end=len(self.y) - 1
                    )
                    arima_score = mean_squared_error(val_y, arima_forecast)
                    self.model_scores['arima'] = arima_score
                    
                    if arima_score < best_score:
                        best_score = arima_score
                        best_model_name = 'arima'
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate ARIMA: {str(e)}")
            
            # Evaluate Prophet
            if self.prophet_model is not None:
                try:
                    future = self.prophet_model.make_future_dataframe(
                        periods=len(val_y),
                        freq=self.y.index.freq or 'D'
                    )
                    prophet_forecast = self.prophet_model.predict(future)
                    prophet_pred = prophet_forecast['yhat'].iloc[split_idx:split_idx+len(val_y)]
                    prophet_score = mean_squared_error(val_y, prophet_pred)
                    self.model_scores['prophet'] = prophet_score
                    
                    if prophet_score < best_score:
                        best_score = prophet_score
                        best_model_name = 'prophet'
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate Prophet: {str(e)}")
            
            # Evaluate LSTM
            if self.lstm_model is not None:
                try:
                    # Prepare validation data
                    val_data = self.y.values.reshape(-1, 1)
                    scaled_val_data = self.scaler.transform(val_data)
                    
                    # Create sequences for validation
                    X_val, y_val = self._create_sequences(scaled_val_data, lookback=12)
                    
                    # Make predictions
                    lstm_pred_scaled = self.lstm_model.predict(X_val)
                    lstm_pred = self.scaler.inverse_transform(lstm_pred_scaled)
                    
                    # Calculate score
                    lstm_score = mean_squared_error(val_y.iloc[12:], lstm_pred.flatten())
                    self.model_scores['lstm'] = lstm_score
                    
                    if lstm_score < best_score:
                        best_score = lstm_score
                        best_model_name = 'lstm'
                        
                except Exception as e:
                    logger.warning(f"Failed to evaluate LSTM: {str(e)}")
            
            self.best_model = best_model_name
            logger.info(f"Best model selected: {best_model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to select best model: {str(e)}")
            self.best_model = 'arima'  # Default fallback
    
    def predict(self, steps: Optional[int] = None) -> pd.Series:
        """
        Make predictions using the best model.
        
        Args:
            steps: Number of steps to predict (default: forecast_horizon)
            
        Returns:
            Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        steps = steps or self.forecast_horizon
        
        if self.best_model == 'arima' and self.arima_model is not None:
            return self._predict_arima(steps)
        elif self.best_model == 'prophet' and self.prophet_model is not None:
            return self._predict_prophet(steps)
        elif self.best_model == 'lstm' and self.lstm_model is not None:
            return self._predict_lstm(steps)
        else:
            raise ValueError("No valid model available for prediction")
    
    def _predict_arima(self, steps: int) -> pd.Series:
        """Make ARIMA predictions."""
        forecast = self.arima_model.forecast(steps=steps)
        
        # Create future index
        last_date = self.y.index[-1]
        if hasattr(self.y.index, 'freq') and self.y.index.freq:
            future_index = pd.date_range(
                start=last_date + self.y.index.freq,
                periods=steps,
                freq=self.y.index.freq
            )
        else:
            # Estimate frequency
            freq = pd.infer_freq(self.y.index)
            future_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
        
        return pd.Series(forecast, index=future_index)
    
    def _predict_prophet(self, steps: int) -> pd.Series:
        """Make Prophet predictions."""
        future = self.prophet_model.make_future_dataframe(
            periods=steps,
            freq=self.y.index.freq or 'D'
        )
        
        # Add exogenous variables if available
        if self.X is not None:
            # Extend exogenous variables (simple forward fill)
            future_X = self.X.iloc[-1:].repeat(steps, axis=0)
            future_X.index = future.index[-steps:]
            
            for col in self.X.columns:
                future[col] = future_X[col].values
        
        forecast = self.prophet_model.predict(future)
        
        # Extract predictions
        predictions = forecast['yhat'].iloc[-steps:]
        predictions.index = future.index[-steps:]
        
        return predictions
    
    def _predict_lstm(self, steps: int) -> pd.Series:
        """Make LSTM predictions."""
        # Prepare input data
        last_sequence = self.y.values[-12:].reshape(-1, 1)
        scaled_sequence = self.scaler.transform(last_sequence)
        
        predictions = []
        current_sequence = scaled_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape(1, 12, 1)
            
            # Make prediction
            pred_scaled = self.lstm_model.predict(X_pred, verbose=0)
            pred = self.scaler.inverse_transform(pred_scaled)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_scaled
        
        # Create future index
        last_date = self.y.index[-1]
        if hasattr(self.y.index, 'freq') and self.y.index.freq:
            future_index = pd.date_range(
                start=last_date + self.y.index.freq,
                periods=steps,
                freq=self.y.index.freq
            )
        else:
            freq = pd.infer_freq(self.y.index)
            future_index = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=steps,
                freq=freq
            )
        
        return pd.Series(predictions, index=future_index)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all models."""
        return {
            'best_model': self.best_model,
            'model_scores': self.model_scores,
            'seasonality_info': self.seasonality_info,
            'forecast_horizon': self.forecast_horizon,
            'n_observations': len(self.y),
            'models_available': {
                'arima': self.arima_model is not None,
                'prophet': self.prophet_model is not None,
                'lstm': self.lstm_model is not None
            }
        }
    
    def plot_forecast(self, forecast: pd.Series, title: str = "Time Series Forecast"):
        """Plot the forecast."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 6))
            
            # Plot historical data
            plt.plot(self.y.index, self.y.values, label='Historical', color='blue')
            
            # Plot forecast
            plt.plot(forecast.index, forecast.values, label='Forecast', color='red', linestyle='--')
            
            plt.title(title)
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.warning(f"Failed to plot forecast: {str(e)}") 
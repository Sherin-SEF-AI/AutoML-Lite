"""
Model Monitoring and Drift Detection Module

Detects:
- Data drift (feature distribution changes)
- Prediction drift (output distribution changes)
- Model performance degradation
- Concept drift
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from scipy import stats
from scipy.spatial import distance
import warnings

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Report containing drift detection results"""
    timestamp: datetime
    data_drift_detected: bool
    prediction_drift_detected: bool
    performance_degradation: bool
    drift_scores: Dict[str, float] = field(default_factory=dict)
    feature_drifts: Dict[str, Dict[str, float]] = field(default_factory=dict)
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'data_drift_detected': self.data_drift_detected,
            'prediction_drift_detected': self.prediction_drift_detected,
            'performance_degradation': self.performance_degradation,
            'drift_scores': self.drift_scores,
            'feature_drifts': self.feature_drifts,
            'alerts': self.alerts,
            'recommendations': self.recommendations
        }


class DriftDetector:
    """
    Comprehensive drift detection system

    Implements multiple drift detection methods:
    - Kolmogorov-Smirnov test
    - Chi-squared test
    - Population Stability Index (PSI)
    - Wasserstein distance
    - KL divergence
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        drift_threshold: float = 0.05,  # p-value threshold
        psi_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1,
        window_size: int = 1000,
        verbose: bool = True
    ):
        """
        Initialize Drift Detector

        Parameters:
        -----------
        reference_data : pd.DataFrame, optional
            Reference dataset (training data)
        feature_names : List[str], optional
            Names of features to monitor
        categorical_features : List[str], optional
            Names of categorical features
        drift_threshold : float
            P-value threshold for statistical tests (0.05 = 5% significance)
        psi_threshold : float
            PSI threshold (>0.1 indicates drift, >0.2 significant drift)
        wasserstein_threshold : float
            Wasserstein distance threshold
        window_size : int
            Size of sliding window for monitoring
        verbose : bool
            Whether to print detailed information
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.drift_threshold = drift_threshold
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.window_size = window_size
        self.verbose = verbose

        # Store reference statistics
        self.reference_stats = {}
        if reference_data is not None:
            self._compute_reference_stats()

        # History for monitoring
        self.drift_history = []

    def _compute_reference_stats(self):
        """Compute statistics for reference data"""
        if self.feature_names is None:
            self.feature_names = self.reference_data.columns.tolist()

        for feature in self.feature_names:
            if feature not in self.reference_data.columns:
                continue

            feature_data = self.reference_data[feature].dropna()

            if feature in self.categorical_features:
                # For categorical: compute value counts
                value_counts = feature_data.value_counts(normalize=True)
                self.reference_stats[feature] = {
                    'type': 'categorical',
                    'distribution': value_counts.to_dict(),
                    'unique_values': set(feature_data.unique())
                }
            else:
                # For numerical: compute distribution parameters
                self.reference_stats[feature] = {
                    'type': 'numerical',
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'median': float(feature_data.median()),
                    'q25': float(feature_data.quantile(0.25)),
                    'q75': float(feature_data.quantile(0.75)),
                    'values': feature_data.values  # Store for statistical tests
                }

    def set_reference(self, reference_data: pd.DataFrame):
        """Set or update reference data"""
        self.reference_data = reference_data
        self._compute_reference_stats()

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        methods: List[str] = ['ks', 'psi', 'wasserstein']
    ) -> DriftReport:
        """
        Detect data drift between reference and current data

        Parameters:
        -----------
        current_data : pd.DataFrame
            Current/production data to check for drift
        methods : List[str]
            Drift detection methods to use:
            - 'ks': Kolmogorov-Smirnov test
            - 'chi2': Chi-squared test
            - 'psi': Population Stability Index
            - 'wasserstein': Wasserstein distance
            - 'kl': KL divergence

        Returns:
        --------
        report : DriftReport
            Comprehensive drift report
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if self.verbose:
            logger.info(f"Detecting data drift using methods: {methods}")

        report = DriftReport(
            timestamp=datetime.now(),
            data_drift_detected=False,
            prediction_drift_detected=False,
            performance_degradation=False
        )

        drifted_features = []

        # Check each feature for drift
        for feature in self.feature_names:
            if feature not in current_data.columns:
                report.alerts.append(f"Feature '{feature}' missing in current data")
                continue

            current_feature = current_data[feature].dropna()

            if len(current_feature) == 0:
                report.alerts.append(f"Feature '{feature}' has no valid values")
                continue

            feature_drift_scores = {}

            # Determine feature type
            is_categorical = feature in self.categorical_features

            if is_categorical:
                # Categorical feature drift detection
                if 'chi2' in methods:
                    chi2_score = self._chi_squared_test(feature, current_feature)
                    feature_drift_scores['chi2_pvalue'] = chi2_score

                if 'psi' in methods:
                    psi_score = self._population_stability_index(feature, current_feature)
                    feature_drift_scores['psi'] = psi_score

            else:
                # Numerical feature drift detection
                if 'ks' in methods:
                    ks_score = self._kolmogorov_smirnov_test(feature, current_feature)
                    feature_drift_scores['ks_pvalue'] = ks_score

                if 'wasserstein' in methods:
                    wasserstein_score = self._wasserstein_distance(feature, current_feature)
                    feature_drift_scores['wasserstein'] = wasserstein_score

                if 'kl' in methods:
                    kl_score = self._kl_divergence(feature, current_feature)
                    feature_drift_scores['kl_divergence'] = kl_score

                if 'psi' in methods:
                    psi_score = self._population_stability_index_numerical(feature, current_feature)
                    feature_drift_scores['psi'] = psi_score

            # Determine if feature has drifted
            feature_drifted = self._is_feature_drifted(feature_drift_scores)

            if feature_drifted:
                drifted_features.append(feature)
                report.alerts.append(f"Drift detected in feature '{feature}': {feature_drift_scores}")

            report.feature_drifts[feature] = feature_drift_scores

        # Overall drift detection
        if len(drifted_features) > 0:
            report.data_drift_detected = True
            report.drift_scores['drifted_features_count'] = len(drifted_features)
            report.drift_scores['drifted_features_ratio'] = len(drifted_features) / len(self.feature_names)

            # Recommendations
            if report.drift_scores['drifted_features_ratio'] > 0.3:
                report.recommendations.append("Significant drift detected (>30% features). Consider retraining the model.")
            elif report.drift_scores['drifted_features_ratio'] > 0.1:
                report.recommendations.append("Moderate drift detected (>10% features). Monitor closely and consider retraining soon.")
            else:
                report.recommendations.append("Minor drift detected (<10% features). Continue monitoring.")

        # Store in history
        self.drift_history.append(report)

        if self.verbose:
            logger.info(f"Drift detection complete. Drifted features: {len(drifted_features)}/{len(self.feature_names)}")

        return report

    def _kolmogorov_smirnov_test(self, feature: str, current_data: pd.Series) -> float:
        """Kolmogorov-Smirnov test for numerical features"""
        reference_values = self.reference_stats[feature]['values']
        statistic, pvalue = stats.ks_2samp(reference_values, current_data.values)
        return float(pvalue)

    def _chi_squared_test(self, feature: str, current_data: pd.Series) -> float:
        """Chi-squared test for categorical features"""
        reference_dist = self.reference_stats[feature]['distribution']

        # Get current distribution
        current_dist = current_data.value_counts(normalize=True).to_dict()

        # Align distributions
        all_categories = set(reference_dist.keys()) | set(current_dist.keys())

        reference_counts = []
        current_counts = []

        for category in all_categories:
            reference_counts.append(reference_dist.get(category, 1e-10))
            current_counts.append(current_dist.get(category, 1e-10))

        # Chi-squared test
        try:
            statistic, pvalue = stats.chisquare(current_counts, reference_counts)
            return float(pvalue)
        except Exception as e:
            logger.warning(f"Chi-squared test failed for {feature}: {e}")
            return 1.0

    def _wasserstein_distance(self, feature: str, current_data: pd.Series) -> float:
        """Wasserstein distance (Earth Mover's Distance) for numerical features"""
        reference_values = self.reference_stats[feature]['values']
        wasserstein = stats.wasserstein_distance(reference_values, current_data.values)
        return float(wasserstein)

    def _kl_divergence(self, feature: str, current_data: pd.Series) -> float:
        """KL divergence for numerical features"""
        # Discretize into bins
        reference_values = self.reference_stats[feature]['values']

        # Determine bins
        min_val = min(reference_values.min(), current_data.min())
        max_val = max(reference_values.max(), current_data.max())
        bins = np.linspace(min_val, max_val, 20)

        # Compute histograms
        ref_hist, _ = np.histogram(reference_values, bins=bins, density=True)
        curr_hist, _ = np.histogram(current_data.values, bins=bins, density=True)

        # Add small epsilon to avoid log(0)
        ref_hist = ref_hist + 1e-10
        curr_hist = curr_hist + 1e-10

        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()

        # KL divergence
        kl_div = np.sum(curr_hist * np.log(curr_hist / ref_hist))
        return float(kl_div)

    def _population_stability_index(self, feature: str, current_data: pd.Series) -> float:
        """PSI for categorical features"""
        reference_dist = self.reference_stats[feature]['distribution']
        current_dist = current_data.value_counts(normalize=True).to_dict()

        # Align distributions
        all_categories = set(reference_dist.keys()) | set(current_dist.keys())

        psi = 0.0
        for category in all_categories:
            ref_pct = reference_dist.get(category, 1e-10)
            curr_pct = current_dist.get(category, 1e-10)

            psi += (curr_pct - ref_pct) * np.log(curr_pct / ref_pct)

        return float(psi)

    def _population_stability_index_numerical(self, feature: str, current_data: pd.Series) -> float:
        """PSI for numerical features (using quantile bins)"""
        reference_values = self.reference_stats[feature]['values']

        # Create quantile bins
        bins = np.percentile(reference_values, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        bins = np.unique(bins)  # Remove duplicates

        # Compute distributions
        ref_hist, _ = np.histogram(reference_values, bins=bins)
        curr_hist, _ = np.histogram(current_data.values, bins=bins)

        # Normalize
        ref_pct = ref_hist / len(reference_values) + 1e-10
        curr_pct = curr_hist / len(current_data) + 1e-10

        # PSI
        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def _is_feature_drifted(self, drift_scores: Dict[str, float]) -> bool:
        """Determine if a feature has drifted based on multiple scores"""
        # P-value based tests (KS, Chi-squared)
        for key in ['ks_pvalue', 'chi2_pvalue']:
            if key in drift_scores and drift_scores[key] < self.drift_threshold:
                return True

        # PSI threshold
        if 'psi' in drift_scores and abs(drift_scores['psi']) > self.psi_threshold:
            return True

        # Wasserstein threshold
        if 'wasserstein' in drift_scores and drift_scores['wasserstein'] > self.wasserstein_threshold:
            return True

        # KL divergence threshold (typically > 0.1 indicates drift)
        if 'kl_divergence' in drift_scores and drift_scores['kl_divergence'] > 0.1:
            return True

        return False

    def detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
        problem_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions

        Parameters:
        -----------
        reference_predictions : np.ndarray
            Predictions from reference period
        current_predictions : np.ndarray
            Current predictions
        problem_type : str
            'classification' or 'regression'

        Returns:
        --------
        drift_info : Dict[str, Any]
            Prediction drift statistics
        """
        drift_info = {
            'drift_detected': False,
            'statistics': {}
        }

        if problem_type == 'classification':
            # For classification: check distribution of predicted classes
            ref_dist = pd.Series(reference_predictions).value_counts(normalize=True).to_dict()
            curr_dist = pd.Series(current_predictions).value_counts(normalize=True).to_dict()

            # Chi-squared test
            all_classes = set(ref_dist.keys()) | set(curr_dist.keys())
            ref_counts = [ref_dist.get(c, 1e-10) for c in all_classes]
            curr_counts = [curr_dist.get(c, 1e-10) for c in all_classes]

            try:
                statistic, pvalue = stats.chisquare(curr_counts, ref_counts)
                drift_info['statistics']['chi2_pvalue'] = float(pvalue)
                drift_info['drift_detected'] = pvalue < self.drift_threshold
            except Exception as e:
                logger.warning(f"Prediction drift test failed: {e}")

        else:
            # For regression: check distribution statistics
            ref_mean, ref_std = np.mean(reference_predictions), np.std(reference_predictions)
            curr_mean, curr_std = np.mean(current_predictions), np.std(current_predictions)

            # Two-sample t-test for means
            statistic, pvalue = stats.ttest_ind(reference_predictions, current_predictions)

            drift_info['statistics'] = {
                'reference_mean': float(ref_mean),
                'current_mean': float(curr_mean),
                'reference_std': float(ref_std),
                'current_std': float(curr_std),
                'ttest_pvalue': float(pvalue)
            }

            drift_info['drift_detected'] = pvalue < self.drift_threshold

        return drift_info

    def detect_performance_degradation(
        self,
        reference_scores: List[float],
        current_scores: List[float],
        degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect model performance degradation

        Parameters:
        -----------
        reference_scores : List[float]
            Performance scores from reference period
        current_scores : List[float]
            Current performance scores
        degradation_threshold : float
            Threshold for detecting degradation (e.g., 0.05 = 5% drop)

        Returns:
        --------
        degradation_info : Dict[str, Any]
            Performance degradation statistics
        """
        ref_mean = np.mean(reference_scores)
        curr_mean = np.mean(current_scores)

        relative_change = (curr_mean - ref_mean) / ref_mean

        degradation_info = {
            'degradation_detected': relative_change < -degradation_threshold,
            'reference_mean': float(ref_mean),
            'current_mean': float(curr_mean),
            'relative_change': float(relative_change),
            'absolute_change': float(curr_mean - ref_mean)
        }

        # Statistical test
        statistic, pvalue = stats.ttest_ind(reference_scores, current_scores)
        degradation_info['ttest_pvalue'] = float(pvalue)

        return degradation_info

    def get_drift_summary(self, last_n: int = 10) -> pd.DataFrame:
        """
        Get summary of recent drift detections

        Parameters:
        -----------
        last_n : int
            Number of recent detections to include

        Returns:
        --------
        summary : pd.DataFrame
            Summary DataFrame
        """
        if not self.drift_history:
            return pd.DataFrame()

        recent_reports = self.drift_history[-last_n:]

        summary_data = []
        for report in recent_reports:
            summary_data.append({
                'timestamp': report.timestamp,
                'data_drift': report.data_drift_detected,
                'prediction_drift': report.prediction_drift_detected,
                'performance_degradation': report.performance_degradation,
                'num_drifted_features': report.drift_scores.get('drifted_features_count', 0)
            })

        return pd.DataFrame(summary_data)


class ModelMonitor:
    """
    Comprehensive model monitoring system

    Combines drift detection with performance monitoring and alerting
    """

    def __init__(
        self,
        model,
        reference_data: pd.DataFrame,
        reference_labels: np.ndarray,
        feature_names: List[str],
        categorical_features: Optional[List[str]] = None,
        problem_type: str = 'classification',
        alert_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Model Monitor

        Parameters:
        -----------
        model : BaseEstimator
            Trained model to monitor
        reference_data : pd.DataFrame
            Reference dataset
        reference_labels : np.ndarray
            Reference labels
        feature_names : List[str]
            Feature names
        categorical_features : List[str], optional
            Categorical feature names
        problem_type : str
            'classification' or 'regression'
        alert_thresholds : Dict[str, float], optional
            Custom thresholds for alerting
        """
        self.model = model
        self.reference_data = reference_data
        self.reference_labels = reference_labels
        self.feature_names = feature_names
        self.problem_type = problem_type

        # Initialize drift detector
        self.drift_detector = DriftDetector(
            reference_data=reference_data,
            feature_names=feature_names,
            categorical_features=categorical_features
        )

        # Get reference predictions
        self.reference_predictions = model.predict(reference_data)

        # Store reference performance
        if problem_type == 'classification':
            from sklearn.metrics import accuracy_score
            self.reference_performance = accuracy_score(reference_labels, self.reference_predictions)
        else:
            from sklearn.metrics import r2_score
            self.reference_performance = r2_score(reference_labels, self.reference_predictions)

        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'performance_degradation': 0.05,
            'drift_ratio': 0.3
        }

        # Monitoring history
        self.monitoring_history = []

    def monitor(
        self,
        current_data: pd.DataFrame,
        current_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Monitor model on current data

        Parameters:
        -----------
        current_data : pd.DataFrame
            Current data to monitor
        current_labels : np.ndarray, optional
            True labels if available

        Returns:
        --------
        monitoring_report : Dict[str, Any]
            Comprehensive monitoring report
        """
        report = {
            'timestamp': datetime.now(),
            'alerts': [],
            'recommendations': []
        }

        # 1. Data drift detection
        drift_report = self.drift_detector.detect_data_drift(current_data)
        report['data_drift'] = drift_report.to_dict()

        # 2. Prediction drift detection
        current_predictions = self.model.predict(current_data)
        prediction_drift = self.drift_detector.detect_prediction_drift(
            self.reference_predictions,
            current_predictions,
            problem_type=self.problem_type
        )
        report['prediction_drift'] = prediction_drift

        # 3. Performance monitoring (if labels available)
        if current_labels is not None:
            if self.problem_type == 'classification':
                from sklearn.metrics import accuracy_score
                current_performance = accuracy_score(current_labels, current_predictions)
            else:
                from sklearn.metrics import r2_score
                current_performance = r2_score(current_labels, current_predictions)

            performance_change = (current_performance - self.reference_performance) / self.reference_performance

            report['performance'] = {
                'reference_performance': self.reference_performance,
                'current_performance': current_performance,
                'relative_change': performance_change
            }

            # Check for degradation
            if performance_change < -self.alert_thresholds['performance_degradation']:
                report['alerts'].append(f"Performance degradation detected: {performance_change:.2%}")
                report['recommendations'].append("Consider retraining the model with recent data")

        # 4. Generate alerts and recommendations
        if drift_report.data_drift_detected:
            report['alerts'].extend(drift_report.alerts)
            report['recommendations'].extend(drift_report.recommendations)

        if prediction_drift['drift_detected']:
            report['alerts'].append("Prediction distribution has drifted significantly")
            report['recommendations'].append("Investigate changes in input data or model behavior")

        # Store in history
        self.monitoring_history.append(report)

        return report

    def get_monitoring_summary(self, last_n: int = 10) -> pd.DataFrame:
        """Get summary of recent monitoring results"""
        if not self.monitoring_history:
            return pd.DataFrame()

        recent_reports = self.monitoring_history[-last_n:]

        summary_data = []
        for report in recent_reports:
            summary_data.append({
                'timestamp': report['timestamp'],
                'data_drift_detected': report['data_drift']['data_drift_detected'],
                'prediction_drift_detected': report['prediction_drift']['drift_detected'],
                'num_alerts': len(report['alerts']),
                'current_performance': report.get('performance', {}).get('current_performance', None)
            })

        return pd.DataFrame(summary_data)

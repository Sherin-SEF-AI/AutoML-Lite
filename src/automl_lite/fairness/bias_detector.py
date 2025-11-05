"""
Fairness and Bias Detection Framework

Implements metrics and methods for detecting and mitigating bias in ML models:
- Demographic parity
- Equal opportunity
- Equalized odds
- Disparate impact
- Calibration by group
- Fairness interventions
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FairnessReport:
    """Report containing fairness metrics"""
    demographic_parity_diff: float
    equal_opportunity_diff: float
    equalized_odds_diff: float
    disparate_impact_ratio: float
    calibration_by_group: Dict[str, float]
    is_fair: bool
    violations: List[str]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            'demographic_parity_diff': self.demographic_parity_diff,
            'equal_opportunity_diff': self.equal_opportunity_diff,
            'equalized_odds_diff': self.equalized_odds_diff,
            'disparate_impact_ratio': self.disparate_impact_ratio,
            'calibration_by_group': self.calibration_by_group,
            'is_fair': self.is_fair,
            'violations': self.violations,
            'recommendations': self.recommendations
        }


class FairnessMetrics:
    """
    Fairness metrics calculator

    Computes various fairness metrics for binary classification
    """

    @staticmethod
    def demographic_parity(
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Demographic Parity: P(Y_pred=1 | A=0) ≈ P(Y_pred=1 | A=1)

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted labels
        sensitive_feature : np.ndarray
            Sensitive attribute (e.g., 0=male, 1=female)

        Returns:
        --------
        metrics : Dict[str, float]
            Demographic parity metrics
        """
        groups = np.unique(sensitive_feature)

        if len(groups) != 2:
            logger.warning(f"Expected 2 groups for sensitive feature, got {len(groups)}")

        # Positive rate for each group
        rates = {}
        for group in groups:
            mask = sensitive_feature == group
            rates[str(group)] = np.mean(y_pred[mask] == 1)

        # Difference between groups
        diff = abs(rates[str(groups[0])] - rates[str(groups[1])])

        return {
            'group_rates': rates,
            'difference': float(diff),
            'satisfied': diff < 0.1  # Common threshold: 10%
        }

    @staticmethod
    def equal_opportunity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Equal Opportunity: TPR should be equal across groups
        P(Y_pred=1 | Y=1, A=0) ≈ P(Y_pred=1 | Y=1, A=1)

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        sensitive_feature : np.ndarray
            Sensitive attribute

        Returns:
        --------
        metrics : Dict[str, float]
            Equal opportunity metrics
        """
        groups = np.unique(sensitive_feature)

        # True positive rate for each group
        tpr = {}
        for group in groups:
            group_mask = sensitive_feature == group
            positive_mask = y_true == 1
            mask = group_mask & positive_mask

            if np.sum(mask) > 0:
                tpr[str(group)] = np.mean(y_pred[mask] == 1)
            else:
                tpr[str(group)] = 0.0

        # Difference
        if len(groups) >= 2:
            diff = abs(tpr[str(groups[0])] - tpr[str(groups[1])])
        else:
            diff = 0.0

        return {
            'tpr_by_group': tpr,
            'difference': float(diff),
            'satisfied': diff < 0.1
        }

    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Equalized Odds: Both TPR and FPR should be equal across groups

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        sensitive_feature : np.ndarray
            Sensitive attribute

        Returns:
        --------
        metrics : Dict[str, float]
            Equalized odds metrics
        """
        groups = np.unique(sensitive_feature)

        tpr = {}
        fpr = {}

        for group in groups:
            group_mask = sensitive_feature == group

            # TPR
            positive_mask = y_true == 1
            tpr_mask = group_mask & positive_mask
            if np.sum(tpr_mask) > 0:
                tpr[str(group)] = np.mean(y_pred[tpr_mask] == 1)
            else:
                tpr[str(group)] = 0.0

            # FPR
            negative_mask = y_true == 0
            fpr_mask = group_mask & negative_mask
            if np.sum(fpr_mask) > 0:
                fpr[str(group)] = np.mean(y_pred[fpr_mask] == 1)
            else:
                fpr[str(group)] = 0.0

        # Average difference
        if len(groups) >= 2:
            tpr_diff = abs(tpr[str(groups[0])] - tpr[str(groups[1])])
            fpr_diff = abs(fpr[str(groups[0])] - fpr[str(groups[1])])
            avg_diff = (tpr_diff + fpr_diff) / 2
        else:
            avg_diff = 0.0

        return {
            'tpr_by_group': tpr,
            'fpr_by_group': fpr,
            'difference': float(avg_diff),
            'satisfied': avg_diff < 0.1
        }

    @staticmethod
    def disparate_impact(
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray
    ) -> Dict[str, float]:
        """
        Disparate Impact Ratio: P(Y_pred=1 | A=0) / P(Y_pred=1 | A=1)

        Should be close to 1.0. Typically, ratio < 0.8 indicates bias.

        Parameters:
        -----------
        y_pred : np.ndarray
            Predicted labels
        sensitive_feature : np.ndarray
            Sensitive attribute

        Returns:
        --------
        metrics : Dict[str, float]
            Disparate impact metrics
        """
        groups = np.unique(sensitive_feature)

        rates = {}
        for group in groups:
            mask = sensitive_feature == group
            rates[str(group)] = np.mean(y_pred[mask] == 1)

        # Ratio (unprivileged / privileged)
        if len(groups) >= 2:
            # Assume first group is unprivileged
            ratio = rates[str(groups[0])] / (rates[str(groups[1])] + 1e-10)
        else:
            ratio = 1.0

        return {
            'group_rates': rates,
            'ratio': float(ratio),
            'satisfied': 0.8 <= ratio <= 1.25  # 80% rule
        }

    @staticmethod
    def calibration_by_group(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_feature: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Calibration by Group: Check if predicted probabilities match actual outcomes
        within each sensitive group

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        sensitive_feature : np.ndarray
            Sensitive attribute
        n_bins : int
            Number of bins for calibration

        Returns:
        --------
        metrics : Dict[str, float]
            Calibration scores by group
        """
        from sklearn.calibration import calibration_curve

        groups = np.unique(sensitive_feature)

        calibration_errors = {}

        for group in groups:
            mask = sensitive_feature == group

            if np.sum(mask) < n_bins:
                calibration_errors[str(group)] = None
                continue

            try:
                # Compute calibration curve
                prob_true, prob_pred = calibration_curve(
                    y_true[mask],
                    y_pred_proba[mask],
                    n_bins=min(n_bins, np.sum(mask) // 2),
                    strategy='uniform'
                )

                # Expected Calibration Error (ECE)
                ece = np.mean(np.abs(prob_true - prob_pred))
                calibration_errors[str(group)] = float(ece)

            except Exception as e:
                logger.warning(f"Calibration calculation failed for group {group}: {e}")
                calibration_errors[str(group)] = None

        return calibration_errors


class BiasDetector:
    """
    Comprehensive bias detection system

    Analyzes model predictions for various types of bias
    """

    def __init__(
        self,
        sensitive_features: List[str],
        fairness_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize Bias Detector

        Parameters:
        -----------
        sensitive_features : List[str]
            Names of sensitive features (e.g., ['gender', 'race', 'age'])
        fairness_thresholds : Dict[str, float], optional
            Custom thresholds for fairness metrics
        """
        self.sensitive_features = sensitive_features
        self.fairness_thresholds = fairness_thresholds or {
            'demographic_parity': 0.1,
            'equal_opportunity': 0.1,
            'equalized_odds': 0.1,
            'disparate_impact_min': 0.8,
            'disparate_impact_max': 1.25
        }

    def detect_bias(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_data: pd.DataFrame,
        y_pred_proba: Optional[np.ndarray] = None
    ) -> Dict[str, FairnessReport]:
        """
        Detect bias for all sensitive features

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        sensitive_data : pd.DataFrame
            DataFrame containing sensitive features
        y_pred_proba : np.ndarray, optional
            Predicted probabilities

        Returns:
        --------
        reports : Dict[str, FairnessReport]
            Fairness reports for each sensitive feature
        """
        reports = {}

        for feature in self.sensitive_features:
            if feature not in sensitive_data.columns:
                logger.warning(f"Sensitive feature '{feature}' not found in data")
                continue

            report = self._analyze_feature(
                y_true, y_pred, sensitive_data[feature].values, y_pred_proba, feature
            )

            reports[feature] = report

        return reports

    def _analyze_feature(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_feature: np.ndarray,
        y_pred_proba: Optional[np.ndarray],
        feature_name: str
    ) -> FairnessReport:
        """Analyze bias for a single sensitive feature"""
        violations = []
        recommendations = []

        # Demographic parity
        dp_metrics = FairnessMetrics.demographic_parity(y_pred, sensitive_feature)
        dp_diff = dp_metrics['difference']

        if not dp_metrics['satisfied']:
            violations.append(f"Demographic parity violated for {feature_name}")
            recommendations.append(f"Consider reweighting or resampling based on {feature_name}")

        # Equal opportunity
        eo_metrics = FairnessMetrics.equal_opportunity(y_true, y_pred, sensitive_feature)
        eo_diff = eo_metrics['difference']

        if not eo_metrics['satisfied']:
            violations.append(f"Equal opportunity violated for {feature_name}")
            recommendations.append(f"Adjust decision threshold by {feature_name} groups")

        # Equalized odds
        eqo_metrics = FairnessMetrics.equalized_odds(y_true, y_pred, sensitive_feature)
        eqo_diff = eqo_metrics['difference']

        if not eqo_metrics['satisfied']:
            violations.append(f"Equalized odds violated for {feature_name}")

        # Disparate impact
        di_metrics = FairnessMetrics.disparate_impact(y_pred, sensitive_feature)
        di_ratio = di_metrics['ratio']

        if not di_metrics['satisfied']:
            violations.append(f"Disparate impact detected for {feature_name} (ratio: {di_ratio:.2f})")
            recommendations.append("Consider bias mitigation techniques during training")

        # Calibration by group
        calibration_errors = {}
        if y_pred_proba is not None:
            calibration_errors = FairnessMetrics.calibration_by_group(
                y_true, y_pred_proba, sensitive_feature
            )

        # Overall fairness
        is_fair = len(violations) == 0

        return FairnessReport(
            demographic_parity_diff=dp_diff,
            equal_opportunity_diff=eo_diff,
            equalized_odds_diff=eqo_diff,
            disparate_impact_ratio=di_ratio,
            calibration_by_group=calibration_errors,
            is_fair=is_fair,
            violations=violations,
            recommendations=recommendations
        )


class FairnessIntervention:
    """
    Fairness intervention methods

    Includes:
    - Reweighting
    - Threshold adjustment
    - Adversarial debiasing (simplified)
    """

    @staticmethod
    def reweight_samples(
        sensitive_feature: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """
        Compute sample weights to achieve demographic parity

        Parameters:
        -----------
        sensitive_feature : np.ndarray
            Sensitive attribute
        y : np.ndarray
            Target labels

        Returns:
        --------
        weights : np.ndarray
            Sample weights
        """
        # Compute group proportions
        groups = np.unique(sensitive_feature)
        weights = np.ones(len(y))

        for group in groups:
            mask = sensitive_feature == group

            # Target: equal representation of positive class across groups
            group_positive_rate = np.mean(y[mask])
            overall_positive_rate = np.mean(y)

            if group_positive_rate > 0:
                weight_multiplier = overall_positive_rate / group_positive_rate
                weights[mask] *= weight_multiplier

        # Normalize
        weights = weights / np.mean(weights)

        return weights

    @staticmethod
    def find_fair_threshold(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_feature: np.ndarray,
        metric: str = 'equal_opportunity'
    ) -> Dict[str, float]:
        """
        Find group-specific thresholds to satisfy fairness constraints

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities
        sensitive_feature : np.ndarray
            Sensitive attribute
        metric : str
            Fairness metric to optimize

        Returns:
        --------
        thresholds : Dict[str, float]
            Optimal thresholds for each group
        """
        groups = np.unique(sensitive_feature)
        thresholds = {}

        if metric == 'equal_opportunity':
            # Find thresholds that equalize TPR

            # Target TPR: average across groups
            target_tpr = 0.0
            for group in groups:
                group_mask = sensitive_feature == group
                positive_mask = y_true == 1
                mask = group_mask & positive_mask

                if np.sum(mask) > 0:
                    # Use 0.5 threshold as baseline
                    baseline_tpr = np.mean((y_pred_proba[mask] > 0.5))
                    target_tpr += baseline_tpr

            target_tpr /= len(groups)

            # Find threshold for each group to achieve target TPR
            for group in groups:
                group_mask = sensitive_feature == group
                positive_mask = y_true == 1
                mask = group_mask & positive_mask

                if np.sum(mask) > 0:
                    # Search for threshold
                    group_proba = y_pred_proba[mask]
                    sorted_proba = np.sort(group_proba)

                    best_threshold = 0.5
                    best_diff = float('inf')

                    for threshold in sorted_proba:
                        tpr = np.mean(group_proba >= threshold)
                        diff = abs(tpr - target_tpr)

                        if diff < best_diff:
                            best_diff = diff
                            best_threshold = threshold

                    thresholds[str(group)] = float(best_threshold)
                else:
                    thresholds[str(group)] = 0.5

        return thresholds


def detect_and_mitigate_bias(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sensitive_data: pd.DataFrame,
    sensitive_features: List[str],
    mitigation: str = 'reweight'
) -> Dict[str, Any]:
    """
    Factory function for bias detection and mitigation

    Parameters:
    -----------
    model : estimator
        Trained model
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    sensitive_data : pd.DataFrame
        Sensitive features
    sensitive_features : List[str]
        Names of sensitive features
    mitigation : str
        Mitigation strategy: 'reweight', 'threshold', or 'none'

    Returns:
    --------
    results : Dict[str, Any]
        Bias detection results and mitigation recommendations
    """
    # Detect bias
    detector = BiasDetector(sensitive_features)

    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None

    reports = detector.detect_bias(y, y_pred, sensitive_data, y_pred_proba)

    results = {
        'fairness_reports': {k: v.to_dict() for k, v in reports.items()},
        'mitigation_applied': mitigation
    }

    # Apply mitigation if requested
    if mitigation == 'reweight':
        # Compute sample weights
        for feature in sensitive_features:
            if feature in sensitive_data.columns:
                weights = FairnessIntervention.reweight_samples(
                    sensitive_data[feature].values, y
                )
                results[f'{feature}_weights'] = weights

    elif mitigation == 'threshold' and y_pred_proba is not None:
        # Find fair thresholds
        for feature in sensitive_features:
            if feature in sensitive_data.columns:
                thresholds = FairnessIntervention.find_fair_threshold(
                    y, y_pred_proba, sensitive_data[feature].values
                )
                results[f'{feature}_thresholds'] = thresholds

    return results

"""
Advanced model comparison utilities for AutoML Lite.
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, log_loss
)
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..core.automl import AutoMLite
from ..utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


class ModelComparator:
    """
    Advanced model comparison and benchmarking utility.
    
    Provides comprehensive model comparison including:
    - Performance metrics comparison
    - Statistical significance testing
    - Cross-validation stability analysis
    - Hyperparameter sensitivity analysis
    - Ensemble performance analysis
    - Model interpretability comparison
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize ModelComparator.
        
        Args:
            verbose: Whether to show progress messages
        """
        self.verbose = verbose
        self.comparison_results = {}
        self.models = {}
        
    def add_model(self, name: str, model: AutoMLite, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a model for comparison.
        
        Args:
            name: Model name/identifier
            model: Trained AutoMLite model
            config: Model configuration (optional)
        """
        if not model.is_fitted:
            raise ValueError(f"Model '{name}' must be fitted before comparison")
        
        self.models[name] = {
            'model': model,
            'config': config or {},
            'problem_type': model.problem_type
        }
        
        if self.verbose:
            logger.info(f"Added model '{name}' for comparison")
    
    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        metrics: Optional[List[str]] = None,
        statistical_tests: bool = True,
        stability_analysis: bool = True,
        interpretability_comparison: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive model comparison.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of cross-validation folds
            metrics: List of metrics to compute (if None, uses default metrics)
            statistical_tests: Whether to perform statistical significance tests
            stability_analysis: Whether to analyze cross-validation stability
            interpretability_comparison: Whether to compare model interpretability
            
        Returns:
            Dictionary containing comparison results
        """
        if not self.models:
            raise ValueError("No models added for comparison")
        
        if self.verbose:
            logger.info(f"Starting model comparison with {len(self.models)} models")
        
        # Determine problem type
        problem_types = set(model_info['problem_type'] for model_info in self.models.values())
        if len(problem_types) > 1:
            raise ValueError("All models must have the same problem type")
        
        problem_type = list(problem_types)[0]
        
        # Set default metrics based on problem type
        if metrics is None:
            metrics = self._get_default_metrics(problem_type)
        
        results = {
            'basic_comparison': self._basic_performance_comparison(X, y, metrics),
            'cross_validation': self._cross_validation_comparison(X, y, cv_folds, metrics),
            'model_info': self._collect_model_info()
        }
        
        if statistical_tests:
            results['statistical_tests'] = self._statistical_significance_tests(X, y, cv_folds)
        
        if stability_analysis:
            results['stability_analysis'] = self._stability_analysis(X, y, cv_folds, metrics)
        
        if interpretability_comparison:
            results['interpretability_comparison'] = self._interpretability_comparison(X, y)
        
        # Add ensemble analysis
        results['ensemble_analysis'] = self._ensemble_analysis(X, y)
        
        self.comparison_results = results
        
        if self.verbose:
            logger.info("Model comparison completed successfully")
        
        return results
    
    def _get_default_metrics(self, problem_type: str) -> List[str]:
        """Get default metrics for the problem type."""
        if problem_type == "classification":
            return ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        elif problem_type == "regression":
            return ['mse', 'mae', 'r2', 'rmse']
        else:
            return ['accuracy']  # Default for clustering
    
    def _basic_performance_comparison(
        self, X: pd.DataFrame, y: pd.Series, metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform basic performance comparison."""
        results = {}
        
        for metric in metrics:
            metric_results = {}
            for name, model_info in self.models.items():
                model = model_info['model']
                try:
                    score = self._compute_metric(model, X, y, metric)
                    metric_results[name] = score
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not compute {metric} for {name}: {str(e)}")
                    metric_results[name] = None
            
            results[metric] = metric_results
        
        # Add ranking
        results['rankings'] = self._compute_rankings(results)
        
        return results
    
    def _cross_validation_comparison(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int, metrics: List[str]
    ) -> Dict[str, Any]:
        """Perform cross-validation comparison."""
        results = {}
        
        # Determine CV strategy
        if self._is_classification_problem():
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for metric in metrics:
            metric_results = {}
            for name, model_info in self.models.items():
                model = model_info['model']
                try:
                    cv_scores = self._cross_validate_metric(model, X, y, metric, cv)
                    metric_results[name] = {
                        'mean': np.mean(cv_scores),
                        'std': np.std(cv_scores),
                        'scores': cv_scores.tolist(),
                        'min': np.min(cv_scores),
                        'max': np.max(cv_scores)
                    }
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not compute CV {metric} for {name}: {str(e)}")
                    metric_results[name] = None
            
            results[metric] = metric_results
        
        return results
    
    def _statistical_significance_tests(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        from scipy import stats
        
        results = {}
        
        # Get CV scores for primary metric
        primary_metric = 'accuracy' if self._is_classification_problem() else 'r2'
        
        cv_scores = {}
        for name, model_info in self.models.items():
            model = model_info['model']
            try:
                if self._is_classification_problem():
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                scores = self._cross_validate_metric(model, X, y, primary_metric, cv)
                cv_scores[name] = scores
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not get CV scores for {name}: {str(e)}")
                continue
        
        if len(cv_scores) < 2:
            return {'error': 'Insufficient models for statistical testing'}
        
        # Perform pairwise t-tests
        model_names = list(cv_scores.keys())
        pairwise_tests = {}
        
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                test_name = f"{name1}_vs_{name2}"
                try:
                    t_stat, p_value = stats.ttest_rel(cv_scores[name1], cv_scores[name2])
                    pairwise_tests[test_name] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05,
                        'effect_size': self._compute_cohens_d(cv_scores[name1], cv_scores[name2])
                    }
                except Exception as e:
                    pairwise_tests[test_name] = {'error': str(e)}
        
        results['pairwise_t_tests'] = pairwise_tests
        
        # Perform Friedman test (non-parametric alternative to repeated measures ANOVA)
        try:
            all_scores = [cv_scores[name] for name in model_names]
            friedman_stat, friedman_p = stats.friedmanchisquare(*all_scores)
            results['friedman_test'] = {
                'statistic': float(friedman_stat),
                'p_value': float(friedman_p),
                'significant': friedman_p < 0.05
            }
        except Exception as e:
            results['friedman_test'] = {'error': str(e)}
        
        return results
    
    def _stability_analysis(
        self, X: pd.DataFrame, y: pd.Series, cv_folds: int, metrics: List[str]
    ) -> Dict[str, Any]:
        """Analyze cross-validation stability."""
        results = {}
        
        for metric in metrics:
            metric_stability = {}
            for name, model_info in self.models.items():
                model = model_info['model']
                try:
                    if self._is_classification_problem():
                        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    else:
                        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    
                    scores = self._cross_validate_metric(model, X, y, metric, cv)
                    
                    # Compute stability metrics
                    stability_metrics = {
                        'cv_mean': float(np.mean(scores)),
                        'cv_std': float(np.std(scores)),
                        'cv_cv': float(np.std(scores) / np.mean(scores)) if np.mean(scores) != 0 else 0,
                        'cv_range': float(np.max(scores) - np.min(scores)),
                        'cv_iqr': float(np.percentile(scores, 75) - np.percentile(scores, 25))
                    }
                    
                    metric_stability[name] = stability_metrics
                    
                except Exception as e:
                    if self.verbose:
                        logger.warning(f"Could not analyze stability for {name}: {str(e)}")
                    metric_stability[name] = None
            
            results[metric] = metric_stability
        
        return results
    
    def _interpretability_comparison(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compare model interpretability."""
        results = {
            'feature_importance': {},
            'model_complexity': {},
            'training_time': {}
        }
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Feature importance comparison
            if model.feature_importance:
                results['feature_importance'][name] = {
                    'num_features': len(model.feature_importance),
                    'top_features': list(model.feature_importance.keys())[:5],
                    'importance_entropy': self._compute_importance_entropy(model.feature_importance)
                }
            
            # Model complexity
            results['model_complexity'][name] = self._compute_model_complexity(model)
            
            # Training time (if available)
            if model.training_history:
                total_time = sum(entry.get('training_time', 0) for entry in model.training_history)
                results['training_time'][name] = total_time
        
        return results
    
    def _ensemble_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze ensemble performance."""
        results = {
            'ensemble_models': {},
            'ensemble_performance': {},
            'diversity_analysis': {}
        }
        
        # Find models with ensembles
        ensemble_models = {}
        for name, model_info in self.models.items():
            model = model_info['model']
            if model.ensemble_model:
                ensemble_models[name] = model
        
        if not ensemble_models:
            return {'message': 'No ensemble models found'}
        
        # Compare ensemble vs individual performance
        for name, model in ensemble_models.items():
            try:
                # Individual model performance
                individual_score = model.best_model.score(X, y)
                
                # Ensemble performance
                ensemble_score = model.ensemble_model.score(X, y)
                
                results['ensemble_performance'][name] = {
                    'individual_score': individual_score,
                    'ensemble_score': ensemble_score,
                    'improvement': ensemble_score - individual_score,
                    'improvement_percentage': ((ensemble_score - individual_score) / individual_score) * 100 if individual_score != 0 else 0
                }
                
                # Ensemble info
                ensemble_info = model.get_ensemble_info()
                results['ensemble_models'][name] = ensemble_info
                
            except Exception as e:
                if self.verbose:
                    logger.warning(f"Could not analyze ensemble for {name}: {str(e)}")
        
        return results
    
    def _compute_metric(self, model: AutoMLite, X: pd.DataFrame, y: pd.Series, metric: str) -> float:
        """Compute a specific metric for a model."""
        if metric == 'accuracy':
            return accuracy_score(y, model.predict(X))
        elif metric == 'precision':
            return precision_score(y, model.predict(X), average='weighted')
        elif metric == 'recall':
            return recall_score(y, model.predict(X), average='weighted')
        elif metric == 'f1':
            return f1_score(y, model.predict(X), average='weighted')
        elif metric == 'roc_auc':
            if model.problem_type == "classification":
                return roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
            else:
                raise ValueError("ROC AUC only available for classification")
        elif metric == 'mse':
            return mean_squared_error(y, model.predict(X))
        elif metric == 'mae':
            return mean_absolute_error(y, model.predict(X))
        elif metric == 'r2':
            return r2_score(y, model.predict(X))
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y, model.predict(X)))
        elif metric == 'log_loss':
            if model.problem_type == "classification":
                return log_loss(y, model.predict_proba(X))
            else:
                raise ValueError("Log loss only available for classification")
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _cross_validate_metric(
        self, model: AutoMLite, X: pd.DataFrame, y: pd.Series, metric: str, cv
    ) -> np.ndarray:
        """Perform cross-validation for a specific metric."""
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Retrain model on this fold
            model_copy = AutoMLite(
                time_budget=60,  # Reduced for CV
                max_models=3,    # Reduced for CV
                verbose=False
            )
            model_copy.fit(X_train, y_train)
            
            # Compute metric
            score = self._compute_metric(model_copy, X_val, y_val, metric)
            scores.append(score)
        
        return np.array(scores)
    
    def _compute_rankings(self, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Compute rankings for each metric."""
        rankings = {}
        
        for metric, metric_results in results.items():
            if metric == 'rankings':
                continue
            
            # Sort models by performance (higher is better for most metrics)
            sorted_models = sorted(
                metric_results.items(),
                key=lambda x: x[1] if x[1] is not None else float('-inf'),
                reverse=True
            )
            
            rankings[metric] = [name for name, _ in sorted_models]
        
        return rankings
    
    def _compute_cohens_d(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """Compute Cohen's d effect size."""
        n1, n2 = len(scores1), len(scores2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(scores1, ddof=1) + (n2 - 1) * np.var(scores2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    def _compute_importance_entropy(self, feature_importance: Dict[str, float]) -> float:
        """Compute entropy of feature importance distribution."""
        values = np.array(list(feature_importance.values()))
        values = values / np.sum(values)  # Normalize
        entropy = -np.sum(values * np.log2(values + 1e-10))
        return float(entropy)
    
    def _compute_model_complexity(self, model: AutoMLite) -> Dict[str, Any]:
        """Compute model complexity metrics."""
        complexity = {
            'num_models_trained': len(model.training_history) if model.training_history else 0,
            'has_ensemble': model.ensemble_model is not None,
            'feature_selection': model.selected_features is not None,
            'num_selected_features': len(model.selected_features) if model.selected_features else None
        }
        
        # Estimate model parameters
        if model.best_model:
            try:
                if hasattr(model.best_model, 'n_estimators'):
                    complexity['n_estimators'] = model.best_model.n_estimators
                if hasattr(model.best_model, 'n_features_in_'):
                    complexity['n_features'] = model.best_model.n_features_in_
                if hasattr(model.best_model, 'n_classes_'):
                    complexity['n_classes'] = model.best_model.n_classes_
            except:
                pass
        
        return complexity
    
    def _is_classification_problem(self) -> bool:
        """Check if this is a classification problem."""
        problem_types = set(model_info['problem_type'] for model_info in self.models.values())
        return 'classification' in problem_types
    
    def _collect_model_info(self) -> Dict[str, Any]:
        """Collect basic information about all models."""
        info = {}
        
        for name, model_info in self.models.items():
            model = model_info['model']
            info[name] = {
                'problem_type': model.problem_type,
                'best_model_name': model.best_model_name,
                'best_score': model.best_score,
                'config': model_info['config'],
                'has_ensemble': model.ensemble_model is not None,
                'has_feature_selection': model.selected_features is not None,
                'has_interpretability': model.interpretability_results is not None
            }
        
        return info
    
    def display_comparison_summary(self) -> None:
        """Display a summary of the comparison results."""
        if not self.comparison_results:
            console.print("❌ No comparison results available", style="bold red")
            return
        
        # Basic performance comparison
        if 'basic_comparison' in self.comparison_results:
            console.print("\n[bold cyan]Basic Performance Comparison[/bold cyan]")
            
            table = Table()
            table.add_column("Model", style="cyan")
            
            # Get all metrics
            metrics = list(self.comparison_results['basic_comparison'].keys())
            metrics = [m for m in metrics if m != 'rankings']
            
            for metric in metrics:
                table.add_column(metric.title(), style="green")
            
            # Add rows
            for model_name in self.models.keys():
                row = [model_name]
                for metric in metrics:
                    score = self.comparison_results['basic_comparison'][metric].get(model_name)
                    row.append(f"{score:.4f}" if score is not None else "N/A")
                table.add_row(*row)
            
            console.print(table)
        
        # Cross-validation results
        if 'cross_validation' in self.comparison_results:
            console.print("\n[bold cyan]Cross-Validation Results (Mean ± Std)[/bold cyan]")
            
            table = Table()
            table.add_column("Model", style="cyan")
            
            metrics = list(self.comparison_results['cross_validation'].keys())
            for metric in metrics:
                table.add_column(f"{metric.title()} (CV)", style="green")
            
            for model_name in self.models.keys():
                row = [model_name]
                for metric in metrics:
                    cv_result = self.comparison_results['cross_validation'][metric].get(model_name)
                    if cv_result:
                        row.append(f"{cv_result['mean']:.4f} ± {cv_result['std']:.4f}")
                    else:
                        row.append("N/A")
                table.add_row(*row)
            
            console.print(table)
        
        # Statistical tests
        if 'statistical_tests' in self.comparison_results:
            console.print("\n[bold cyan]Statistical Significance Tests[/bold cyan]")
            
            pairwise_tests = self.comparison_results['statistical_tests'].get('pairwise_t_tests', {})
            if pairwise_tests:
                table = Table()
                table.add_column("Comparison", style="cyan")
                table.add_column("P-Value", style="green")
                table.add_column("Significant", style="yellow")
                table.add_column("Effect Size", style="magenta")
                
                for test_name, test_result in pairwise_tests.items():
                    if 'error' not in test_result:
                        table.add_row(
                            test_name,
                            f"{test_result['p_value']:.4f}",
                            "✅" if test_result['significant'] else "❌",
                            f"{test_result['effect_size']:.3f}"
                        )
                
                console.print(table)
    
    def generate_comparison_report(self, output_path: str = "model_comparison_report.html") -> None:
        """Generate HTML report of the comparison results."""
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run compare_models() first.")
        
        # Create HTML report
        html_content = self._create_comparison_html_report()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        if self.verbose:
            logger.info(f"Comparison report saved to {output_path}")
    
    def _create_comparison_html_report(self) -> str:
        """Create HTML content for the comparison report."""
        # This is a simplified version - in practice, you'd want a more sophisticated template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }
                .metric { margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .winner { background-color: #d4edda; }
                .loser { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>Model Comparison Report</h1>
        """
        
        # Add sections based on comparison results
        for section_name, section_data in self.comparison_results.items():
            html += f"<div class='section'><h2>{section_name.replace('_', ' ').title()}</h2>"
            html += self._format_comparison_section_data(section_data)
            html += "</div>"
        
        html += "</body></html>"
        return html
    
    def _format_comparison_section_data(self, data: Any) -> str:
        """Format comparison section data for HTML display."""
        if isinstance(data, dict):
            if 'rankings' in data:
                # Handle rankings specially
                html = "<h3>Rankings</h3>"
                for metric, ranking in data['rankings'].items():
                    html += f"<p><strong>{metric}:</strong> {' > '.join(ranking)}</p>"
                return html
            else:
                # Regular table
                html = "<table><tr><th>Model</th><th>Value</th></tr>"
                for key, value in data.items():
                    if isinstance(value, dict):
                        value_str = "<br>".join([f"{k}: {v}" for k, v in value.items()])
                    else:
                        value_str = str(value)
                    html += f"<tr><td>{key}</td><td>{value_str}</td></tr>"
                html += "</table>"
                return html
        else:
            return f"<p>{str(data)}</p>" 
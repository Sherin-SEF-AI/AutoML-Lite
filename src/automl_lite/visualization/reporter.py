"""
Report generator for AutoML Lite.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from jinja2 import Template

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReportGenerator:
    """
    Generate comprehensive HTML reports with visualizations.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        self.template_path = Path(__file__).parent.parent / "templates"
        self.static_path = Path(__file__).parent.parent / "static"
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def generate_report(
        self,
        path: Union[str, Path],
        automl: Any,
        problem_type: str,
        leaderboard: Optional[List[Dict[str, Any]]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None,
        ensemble_info: Optional[Dict[str, Any]] = None,
        interpretability_results: Optional[Dict[str, Any]] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
        nas_result: Optional[Any] = None,
    ) -> None:
        """
        Generate a comprehensive HTML report.
        
        Args:
            path: Path to save the report
            automl: AutoMLite instance
            problem_type: Type of ML problem
            leaderboard: Model leaderboard
            feature_importance: Feature importance scores
            training_history: Training history
            ensemble_info: Ensemble information
            interpretability_results: Model interpretability results
            X_test: Test features
            y_test: Test labels
            nas_result: NAS search results (optional)
        """
        # Create visualizations
        plots = self._create_visualizations(
            automl, problem_type, leaderboard, feature_importance, training_history,
            X_test, y_test
        )
        
        # Create NAS visualizations if available
        nas_plots = {}
        nas_summary = {}
        if nas_result is not None:
            nas_plots, nas_summary = self._create_nas_visualizations(nas_result)
        
        # Generate HTML content
        html_content = self._generate_html_content(
            automl, problem_type, leaderboard, feature_importance, training_history, 
            ensemble_info, interpretability_results, plots, nas_plots, nas_summary
        )
        
        # Save report
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report generated at {path}")
    
    def _create_visualizations(
        self,
        automl: Any,
        problem_type: str,
        leaderboard: Optional[List[Dict[str, Any]]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """Create all visualizations and return as base64 encoded strings."""
        plots = {}
        
        # Leaderboard plot
        if leaderboard:
            plots['leaderboard'] = self._create_leaderboard_plot(leaderboard)
        
        # Feature importance plot
        if feature_importance:
            plots['feature_importance'] = self._create_feature_importance_plot(feature_importance)
        
        # Training history plot
        if training_history:
            plots['training_history'] = self._create_training_history_plot(training_history)
            plots['learning_curve'] = self._create_learning_curve_plot(training_history)
        
        # Model performance comparison
        if leaderboard:
            plots['performance_comparison'] = self._create_performance_comparison_plot(leaderboard)
        
        # Test set visualizations if available
        if X_test is not None and y_test is not None:
            try:
                y_pred = automl.predict(X_test)
                
                if problem_type == "classification":
                    # Confusion matrix
                    plots['confusion_matrix'] = self._create_confusion_matrix_plot(y_test, y_pred)
                    
                    # ROC curve (if predict_proba is available)
                    try:
                        y_pred_proba = automl.predict_proba(X_test)
                        plots['roc_curve'] = self._create_roc_curve_plot(y_test, y_pred_proba)
                    except:
                        pass  # Skip ROC if predict_proba not available
                
                elif problem_type == "regression":
                    # Residuals plot
                    plots['residuals'] = self._create_residuals_plot(y_test, y_pred)
                
                # Feature correlation (if features are available)
                if hasattr(automl, 'selected_features') and automl.selected_features:
                    try:
                        X_selected = X_test[automl.selected_features]
                        if isinstance(X_selected, pd.DataFrame):
                            plots['feature_correlation'] = self._create_feature_correlation_plot(X_selected)
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Failed to create test set visualizations: {str(e)}")
        
        return plots
    
    def _create_leaderboard_plot(self, leaderboard: List[Dict[str, Any]]) -> str:
        """Create leaderboard visualization."""
        df = pd.DataFrame(leaderboard)
        df = df.sort_values('score', ascending=False)
        
        fig = px.bar(
            df,
            x='model_name',
            y='score',
            title='Model Performance Leaderboard',
            labels={'score': 'Cross-Validation Score', 'model_name': 'Model'},
            color='score',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        return self._fig_to_base64(fig)
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float]) -> str:
        """Create feature importance visualization."""
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:20])  # Top 20 features
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Feature Importance (Top 20)',
            labels={'x': 'Importance', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(height=600, showlegend=False)
        
        return self._fig_to_base64(fig)
    
    def _create_training_history_plot(self, training_history: List[Dict[str, Any]]) -> str:
        """Create training history visualization."""
        df = pd.DataFrame(training_history)
        
        # Handle both 'time' and 'training_time' field names
        time_column = 'time' if 'time' in df.columns else 'training_time'
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Model Scores Over Time', 'Training Time per Model'),
            vertical_spacing=0.1
        )
        
        # Score over time
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['score'],
                mode='lines+markers',
                name='Score',
                text=df['model_name']
            ),
            row=1, col=1
        )
        
        # Time per model
        fig.add_trace(
            go.Bar(
                x=df['model_name'],
                y=df[time_column],
                name='Training Time (s)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="Training History",
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=-45, row=2, col=1)
        
        return self._fig_to_base64(fig)
    
    def _create_performance_comparison_plot(self, leaderboard: List[Dict[str, Any]]) -> str:
        """Create performance comparison visualization."""
        df = pd.DataFrame(leaderboard)
        df = df.sort_values('score', ascending=False)
        
        # Create box plot for score distribution
        fig = px.box(
            df,
            y='score',
            title='Model Performance Distribution',
            labels={'score': 'Cross-Validation Score'}
        )
        
        fig.update_layout(height=400)
        
        return self._fig_to_base64(fig)
    
    def _create_confusion_matrix_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Create confusion matrix visualization."""
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Convert to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=300)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    def _create_roc_curve_plot(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> str:
        """Create ROC curve visualization."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=500,
            showlegend=True
        )
        
        return self._fig_to_base64(fig)
    
    def _create_residuals_plot(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """Create residuals plot for regression."""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residuals Distribution', 
                          'Q-Q Plot', 'Residuals vs Index'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Residuals vs Predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
            row=1, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row="1", col="1")
        
        # Residuals Distribution
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals Dist'),
            row=1, col=2
        )
        
        # Q-Q Plot (simplified)
        sorted_residuals = np.sort(residuals)
        theoretical_quantiles = np.percentile(sorted_residuals, np.linspace(0, 100, len(sorted_residuals)))
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q'),
            row=2, col=1
        )
        
        # Residuals vs Index
        fig.add_trace(
            go.Scatter(x=list(range(len(residuals))), y=residuals, mode='markers', name='Residuals vs Index'),
            row=2, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row="2", col="2")
        
        fig.update_layout(height=600, showlegend=False)
        
        return self._fig_to_base64(fig)
    
    def _create_feature_correlation_plot(self, X: pd.DataFrame) -> str:
        """Create feature correlation heatmap."""
        corr_matrix = X.corr()
        
        fig = px.imshow(
            corr_matrix,
            title='Feature Correlation Matrix',
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        fig.update_layout(height=600)
        
        return self._fig_to_base64(fig)
    
    def _create_learning_curve_plot(self, training_history: List[Dict[str, Any]]) -> str:
        """Create learning curve from training history."""
        if not training_history:
            return ""
        
        df = pd.DataFrame(training_history)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Model Scores Over Time', 'Training Time vs Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Scores over time
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['score'],
                mode='lines+markers',
                name='Score',
                text=df['model_name']
            ),
            row=1, col=1
        )
        
        # Training time vs score
        time_column = 'time' if 'time' in df.columns else 'training_time'
        fig.add_trace(
            go.Scatter(
                x=df[time_column],
                y=df['score'],
                mode='markers',
                name='Time vs Score',
                text=df['model_name']
            ),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 encoded string."""
        img_bytes = fig.to_image(format="png")
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _sanitize_for_template(self, obj):
        import numpy as np
        if isinstance(obj, dict):
            return {str(k): self._sanitize_for_template(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_template(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            if obj.size == 1:
                return float(obj)
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)

    def _create_nas_visualizations(self, nas_result: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """
        Create NAS-specific visualizations.
        
        Args:
            nas_result: NAS search results
        
        Returns:
            Tuple of (plots dict, summary dict)
        """
        try:
            from ..nas.visualization import NASVisualizer
        except ImportError:
            logger.warning("NAS visualization module not available")
            return {}, {}
        
        visualizer = NASVisualizer()
        plots = {}
        
        # Architecture diagram for best architecture
        if hasattr(nas_result, 'best_architecture') and nas_result.best_architecture:
            try:
                plots['best_architecture'] = visualizer.render_architecture_diagram(
                    nas_result.best_architecture,
                    format='base64'
                )
            except Exception as e:
                logger.warning(f"Failed to create architecture diagram: {e}")
        
        # Search progress plot
        if hasattr(nas_result, 'search_history') and nas_result.search_history:
            try:
                plots['search_progress'] = visualizer.create_search_progress_plot(
                    nas_result.search_history,
                    format='base64'
                )
            except Exception as e:
                logger.warning(f"Failed to create search progress plot: {e}")
        
        # Pareto front plot
        if hasattr(nas_result, 'pareto_front') and nas_result.pareto_front:
            try:
                plots['pareto_front'] = visualizer.create_pareto_front_plot(
                    nas_result.pareto_front,
                    format='base64'
                )
            except Exception as e:
                logger.warning(f"Failed to create Pareto front plot: {e}")
        
        # Create summary
        summary = {
            'search_strategy': getattr(nas_result, 'search_strategy', 'Unknown'),
            'search_space_type': getattr(nas_result, 'search_space_type', 'Unknown'),
            'total_architectures_evaluated': getattr(nas_result, 'total_architectures_evaluated', 0),
            'search_time': getattr(nas_result, 'search_time', 0.0),
            'best_accuracy': getattr(nas_result, 'best_accuracy', 0.0),
            'best_latency': getattr(nas_result, 'best_latency', 0.0),
            'best_model_size': getattr(nas_result, 'best_model_size', 0.0),
        }
        
        return plots, summary
    
    def _generate_html_content(
        self,
        automl: Any,
        problem_type: str,
        leaderboard: Optional[List[Dict[str, Any]]] = None,
        feature_importance: Optional[Dict[str, float]] = None,
        training_history: Optional[List[Dict[str, Any]]] = None,
        ensemble_info: Optional[Dict[str, Any]] = None,
        interpretability_results: Optional[Dict[str, Any]] = None,
        plots: Optional[Dict[str, str]] = None,
        nas_plots: Optional[Dict[str, str]] = None,
        nas_summary: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate HTML content using template."""
        
        # Prepare data for template
        template_data = {
            'problem_type': str(problem_type),
            'best_model_name': str(getattr(automl, 'best_model_name', 'Unknown')),
            'best_score': float(getattr(automl, 'best_score', 0.0)),
            'total_models_tried': int(len(leaderboard) if leaderboard else 0),
            'training_time': float(self._calculate_total_training_time(training_history)),
            'leaderboard': self._format_leaderboard(leaderboard),
            'feature_importance': self._format_feature_importance(feature_importance),
            'training_history': self._format_training_history(training_history),
            'ensemble_info': ensemble_info or {},
            'interpretability_results': interpretability_results or {},
            'plots': plots or {},
            'nas_plots': nas_plots or {},
            'nas_summary': nas_summary or {},
            'generation_time': str(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')),
        }
        template_data = self._sanitize_for_template(template_data)
        # Load and render template
        template = self._load_template()
        # Ensure all keys are strings for template rendering
        if isinstance(template_data, dict):
            template_data = {str(k): v for k, v in template_data.items()}
        return template.render(**template_data)
    
    def _load_template(self) -> Template:
        """Load HTML template."""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Lite Report</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #007bff;
        }
        .header h1 {
            color: #007bff;
            margin-bottom: 10px;
        }
        .summary {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        .summary-item {
            text-align: center;
            padding: 15px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .summary-item h3 {
            margin: 0;
            color: #007bff;
            font-size: 1.5em;
        }
        .summary-item p {
            margin: 5px 0 0 0;
            color: #666;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h2 {
            color: #333;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
        }
        .plot-container {
            text-align: center;
            margin: 20px 0;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #007bff;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AutoML Lite Report</h1>
            <p>Automated Machine Learning Analysis Report</p>
        </div>
        
        <div class="summary">
            <h2>📊 Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>{{ problem_type.title() }}</h3>
                    <p>Problem Type</p>
                </div>
                <div class="summary-item">
                    <h3>{{ best_model_name }}</h3>
                    <p>Best Model</p>
                </div>
                <div class="summary-item">
                    <h3>{{ "%.4f"|format(best_score) }}</h3>
                    <p>Best Score</p>
                </div>
                <div class="summary-item">
                    <h3>{{ total_models_tried }}</h3>
                    <p>Models Tried</p>
                </div>
                <div class="summary-item">
                    <h3>{{ "%.2f"|format(training_time) }}s</h3>
                    <p>Total Time</p>
                </div>
            </div>
        </div>
        
        {% if leaderboard %}
        <div class="section">
            <h2>🏆 Model Leaderboard</h2>
            <div class="plot-container">
                <img src="{{ plots.leaderboard }}" alt="Model Leaderboard">
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Model</th>
                        <th>Score</th>
                        <th>Parameters</th>
                    </tr>
                </thead>
                <tbody>
                    {% for model in leaderboard %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ model.model_name }}</td>
                        <td>{{ "%.4f"|format(model.score) }}</td>
                        <td><code>{{ model.params|tojson }}</code></td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if feature_importance %}
        <div class="section">
            <h2>🎯 Feature Importance</h2>
            <div class="plot-container">
                <img src="{{ plots.feature_importance }}" alt="Feature Importance">
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Feature</th>
                        <th>Importance</th>
                    </tr>
                </thead>
                <tbody>
                    {% for feature in feature_importance %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ feature.feature }}</td>
                        <td>{{ "%.4f"|format(feature.importance) }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        
        {% if training_history %}
        <div class="section">
            <h2>⏱️ Training History</h2>
            <div class="plot-container">
                <img src="{{ plots.training_history }}" alt="Training History">
            </div>
            {% if plots.learning_curve %}
            <div class="plot-container">
                <h3>📈 Learning Curves</h3>
                <img src="{{ plots.learning_curve }}" alt="Learning Curves">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if ensemble_info and ensemble_info.ensemble_method %}
        <div class="section">
            <h2>🎯 Ensemble Information</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <h3>{{ ensemble_info.ensemble_method.title() }}</h3>
                    <p>Ensemble Method</p>
                </div>
                <div class="summary-item">
                    <h3>{{ ensemble_info.top_k_models }}</h3>
                    <p>Top K Models</p>
                </div>
                {% if ensemble_info.ensemble_score is not none %}
                <div class="summary-item">
                    <h3>{{ "%.4f"|format(ensemble_info.ensemble_score) }}</h3>
                    <p>Ensemble Score</p>
                </div>
                {% endif %}
            </div>
            <div class="plot-container">
                <h3>🏆 Ensemble vs Individual Models</h3>
                <p>Comparison of ensemble performance against individual model performances.</p>
                {% if plots.performance_comparison %}
                <img src="{{ plots.performance_comparison }}" alt="Performance Comparison">
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        {% if interpretability_results %}
        <div class="section">
            <h2>🔍 Model Interpretability</h2>
            {% if interpretability_results.shap_values %}
            <div class="plot-container">
                <h3>SHAP Values Analysis</h3>
                <p>SHAP (SHapley Additive exPlanations) values provide insights into feature contributions.</p>
            </div>
            {% endif %}
            {% if interpretability_results.feature_effects %}
            <div class="plot-container">
                <h3>Feature Effects</h3>
                <p>Analysis of how individual features affect model predictions.</p>
            </div>
            {% endif %}
            {% if interpretability_results.model_complexity %}
            <div class="plot-container">
                <h3>Model Complexity</h3>
                <p>Complexity metrics and model transparency analysis.</p>
            </div>
            {% endif %}
            {% if plots.feature_correlation %}
            <div class="plot-container">
                <h3>Feature Correlation Matrix</h3>
                <p>Correlation between selected features to understand feature relationships.</p>
                <img src="{{ plots.feature_correlation }}" alt="Feature Correlation">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if plots.confusion_matrix or plots.roc_curve or plots.residuals %}
        <div class="section">
            <h2>📊 Model Performance Analysis</h2>
            {% if plots.confusion_matrix %}
            <div class="plot-container">
                <h3>Confusion Matrix</h3>
                <p>Detailed breakdown of model predictions vs actual values.</p>
                <img src="{{ plots.confusion_matrix }}" alt="Confusion Matrix">
            </div>
            {% endif %}
            {% if plots.roc_curve %}
            <div class="plot-container">
                <h3>ROC Curve</h3>
                <p>Receiver Operating Characteristic curve showing model discrimination ability.</p>
                <img src="{{ plots.roc_curve }}" alt="ROC Curve">
            </div>
            {% endif %}
            {% if plots.residuals %}
            <div class="plot-container">
                <h3>Residuals Analysis</h3>
                <p>Analysis of prediction errors to assess model quality.</p>
                <img src="{{ plots.residuals }}" alt="Residuals Analysis">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        {% if nas_summary and nas_summary.total_architectures_evaluated > 0 %}
        <div class="section">
            <h2>🧠 Neural Architecture Search</h2>
            <div class="summary">
                <h3>NAS Summary</h3>
                <div class="summary-grid">
                    <div class="summary-item">
                        <h3>{{ nas_summary.search_strategy.title() }}</h3>
                        <p>Search Strategy</p>
                    </div>
                    <div class="summary-item">
                        <h3>{{ nas_summary.total_architectures_evaluated }}</h3>
                        <p>Architectures Evaluated</p>
                    </div>
                    <div class="summary-item">
                        <h3>{{ "%.2f"|format(nas_summary.search_time) }}s</h3>
                        <p>Search Time</p>
                    </div>
                    <div class="summary-item">
                        <h3>{{ "%.4f"|format(nas_summary.best_accuracy) }}</h3>
                        <p>Best Accuracy</p>
                    </div>
                    {% if nas_summary.best_latency > 0 %}
                    <div class="summary-item">
                        <h3>{{ "%.2f"|format(nas_summary.best_latency) }}ms</h3>
                        <p>Best Latency</p>
                    </div>
                    {% endif %}
                    {% if nas_summary.best_model_size > 0 %}
                    <div class="summary-item">
                        <h3>{{ "%.2f"|format(nas_summary.best_model_size) }}MB</h3>
                        <p>Best Model Size</p>
                    </div>
                    {% endif %}
                </div>
            </div>
            
            {% if nas_plots.best_architecture %}
            <div class="plot-container">
                <h3>🏗️ Best Architecture</h3>
                <p>Network diagram of the best discovered architecture.</p>
                <img src="{{ nas_plots.best_architecture }}" alt="Best Architecture">
            </div>
            {% endif %}
            
            {% if nas_plots.search_progress %}
            <div class="plot-container">
                <h3>📈 Search Progress</h3>
                <p>Performance improvement over the course of the architecture search.</p>
                <img src="{{ nas_plots.search_progress }}" alt="Search Progress">
            </div>
            {% endif %}
            
            {% if nas_plots.pareto_front %}
            <div class="plot-container">
                <h3>⚖️ Pareto Front</h3>
                <p>Multi-objective optimization results showing trade-offs between accuracy, latency, and model size.</p>
                <img src="{{ nas_plots.pareto_front }}" alt="Pareto Front">
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>Report generated on {{ generation_time }}</p>
            <p>Powered by AutoML Lite 🤖</p>
        </div>
    </div>
</body>
</html>
        """
        
        return Template(template_content)
    
    def _calculate_total_training_time(self, training_history: Optional[List[Dict[str, Any]]]) -> float:
        """Calculate total training time."""
        if not training_history:
            return 0.0
        
        # Handle both 'time' and 'training_time' field names
        total_time = 0.0
        for item in training_history:
            time_value = item.get('time', item.get('training_time', 0))
            total_time = max(total_time, time_value)
        
        return total_time
    
    def _format_leaderboard(self, leaderboard: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format leaderboard for template."""
        if not leaderboard:
            return []
        
        formatted = []
        for i, item in enumerate(leaderboard, 1):
            formatted.append({
                'rank': i,
                'model_name': item.get('model_name', 'Unknown'),
                'score': item.get('score', 0.0),
                'params': item.get('params', {})
            })
        
        return formatted
    
    def _format_feature_importance(self, feature_importance: Optional[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Format feature importance for template."""
        if not feature_importance:
            return []
        
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        formatted = []
        
        for feature, importance in sorted_features:
            formatted.append({
                'feature': feature,
                'importance': importance
            })
        
        return formatted
    
    def _format_training_history(self, training_history: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Format training history for template."""
        if not training_history:
            return []
        
        return training_history 
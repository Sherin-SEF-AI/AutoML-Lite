"""
Visualization utilities for Neural Architecture Search.

This module provides visualization functions for NAS architectures, search progress,
and multi-objective optimization results.
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple
import json

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .architecture import Architecture, LayerConfig
from ..utils.logger import get_logger

logger = get_logger(__name__)


class NASVisualizer:
    """
    Visualizer for NAS architectures and search results.
    """
    
    def __init__(self):
        """Initialize the NAS visualizer."""
        self.color_map = {
            'dense': '#4CAF50',
            'conv2d': '#2196F3',
            'conv1d': '#03A9F4',
            'lstm': '#9C27B0',
            'gru': '#673AB7',
            'dropout': '#FF9800',
            'batchnormalization': '#FFC107',
            'maxpooling2d': '#00BCD4',
            'maxpooling1d': '#00ACC1',
            'flatten': '#795548',
            'input': '#607D8B',
            'output': '#F44336',
        }
    
    def render_architecture_diagram(
        self,
        architecture: Architecture,
        format: str = 'base64'
    ) -> str:
        """
        Generate a network diagram for an architecture.
        
        Args:
            architecture: Architecture to visualize
            format: Output format ('base64', 'svg', 'png')
        
        Returns:
            Diagram as base64 encoded string or SVG/PNG data
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not installed, using matplotlib fallback")
            return self._render_architecture_matplotlib(architecture, format)
        
        # Create directed graph
        dot = graphviz.Digraph(comment=f'Architecture {architecture.id[:8]}')
        dot.attr(rankdir='TB', size='8,10')
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        
        # Add input node
        dot.node('input', 'Input', fillcolor=self.color_map.get('input', '#607D8B'))
        
        # Add layer nodes
        for i, layer in enumerate(architecture.layers):
            layer_type = layer.layer_type.lower()
            color = self.color_map.get(layer_type, '#9E9E9E')
            
            # Format layer label
            label = self._format_layer_label(layer)
            node_id = f'layer_{i}'
            
            dot.node(node_id, label, fillcolor=color)
            
            # Connect to previous layer (sequential connection)
            if i == 0:
                dot.edge('input', node_id)
            else:
                dot.edge(f'layer_{i-1}', node_id)
        
        # Add skip connections
        for from_idx, to_idx in architecture.connections:
            dot.edge(
                f'layer_{from_idx}',
                f'layer_{to_idx}',
                style='dashed',
                color='red',
                label='skip'
            )
        
        # Add output node
        last_layer_id = f'layer_{len(architecture.layers)-1}'
        dot.node('output', 'Output', fillcolor=self.color_map.get('output', '#F44336'))
        dot.edge(last_layer_id, 'output')
        
        # Render to format
        if format == 'base64':
            # Render to PNG and encode as base64
            png_data = dot.pipe(format='png')
            img_base64 = base64.b64encode(png_data).decode()
            return f"data:image/png;base64,{img_base64}"
        elif format == 'svg':
            return dot.pipe(format='svg').decode()
        elif format == 'png':
            return dot.pipe(format='png')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_layer_label(self, layer: LayerConfig) -> str:
        """Format layer label for diagram."""
        layer_type = layer.layer_type
        params = layer.params
        
        # Build label with key parameters
        label_parts = [layer_type.upper()]
        
        if layer_type.lower() == 'dense':
            if 'units' in params:
                label_parts.append(f"units={params['units']}")
            if 'activation' in params:
                label_parts.append(f"act={params['activation']}")
        
        elif layer_type.lower() in ['conv2d', 'conv1d']:
            if 'filters' in params:
                label_parts.append(f"filters={params['filters']}")
            if 'kernel_size' in params:
                label_parts.append(f"kernel={params['kernel_size']}")
        
        elif layer_type.lower() in ['lstm', 'gru']:
            if 'units' in params:
                label_parts.append(f"units={params['units']}")
            if 'return_sequences' in params:
                label_parts.append(f"seq={params['return_sequences']}")
        
        elif layer_type.lower() == 'dropout':
            if 'rate' in params:
                label_parts.append(f"rate={params['rate']:.2f}")
        
        elif layer_type.lower() in ['maxpooling2d', 'maxpooling1d']:
            if 'pool_size' in params:
                label_parts.append(f"pool={params['pool_size']}")
        
        return '\n'.join(label_parts)
    
    def _render_architecture_matplotlib(
        self,
        architecture: Architecture,
        format: str = 'base64'
    ) -> str:
        """
        Fallback renderer using matplotlib when graphviz is not available.
        
        Args:
            architecture: Architecture to visualize
            format: Output format ('base64', 'png')
        
        Returns:
            Diagram as base64 encoded string or PNG data
        """
        fig, ax = plt.subplots(figsize=(10, max(8, len(architecture.layers) * 0.8)))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(architecture.layers) + 2)
        ax.axis('off')
        
        # Draw input
        self._draw_layer_box(ax, 5, len(architecture.layers) + 1, 'Input', '#607D8B')
        
        # Draw layers
        for i, layer in enumerate(architecture.layers):
            y_pos = len(architecture.layers) - i
            layer_type = layer.layer_type.lower()
            color = self.color_map.get(layer_type, '#9E9E9E')
            label = self._format_layer_label(layer)
            
            self._draw_layer_box(ax, 5, y_pos, label, color)
            
            # Draw connection to previous layer
            if i == 0:
                ax.arrow(5, len(architecture.layers) + 0.7, 0, -0.4,
                        head_width=0.3, head_length=0.2, fc='black', ec='black')
            else:
                ax.arrow(5, y_pos + 1.3, 0, -0.4,
                        head_width=0.3, head_length=0.2, fc='black', ec='black')
        
        # Draw skip connections
        for from_idx, to_idx in architecture.connections:
            from_y = len(architecture.layers) - from_idx
            to_y = len(architecture.layers) - to_idx
            ax.plot([6, 7, 7, 6], [from_y, from_y, to_y, to_y],
                   'r--', linewidth=2, label='skip' if from_idx == 0 else '')
        
        # Draw output
        self._draw_layer_box(ax, 5, 0, 'Output', '#F44336')
        ax.arrow(5, 1.3, 0, -0.4,
                head_width=0.3, head_length=0.2, fc='black', ec='black')
        
        plt.title(f'Architecture {architecture.id[:8]}', fontsize=14, fontweight='bold')
        
        # Convert to base64
        if format == 'base64':
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            return f"data:image/png;base64,{img_base64}"
        elif format == 'png':
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
            img_buffer.seek(0)
            plt.close()
            return img_buffer.getvalue()
        else:
            plt.close()
            raise ValueError(f"Unsupported format: {format}")
    
    def _draw_layer_box(self, ax, x, y, label, color):
        """Draw a layer box on matplotlib axes."""
        from matplotlib.patches import FancyBboxPatch
        
        box = FancyBboxPatch(
            (x - 1.5, y - 0.4), 3, 0.8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    def create_search_progress_plot(
        self,
        search_history: List[Dict[str, Any]],
        format: str = 'base64'
    ) -> str:
        """
        Create search progress visualization.
        
        Args:
            search_history: List of architecture evaluations with metrics
            format: Output format ('base64', 'html')
        
        Returns:
            Plot as base64 encoded string or HTML
        """
        if not search_history:
            return ""
        
        # Extract data
        iterations = list(range(1, len(search_history) + 1))
        accuracies = [h.get('accuracy', h.get('score', 0)) for h in search_history]
        
        # Calculate best so far
        best_so_far = []
        current_best = -float('inf')
        for acc in accuracies:
            current_best = max(current_best, acc)
            best_so_far.append(current_best)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Best Performance Over Time', 'Number of Architectures Evaluated'),
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4]
        )
        
        # Plot 1: Best performance over time
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=best_so_far,
                mode='lines',
                name='Best So Far',
                line=dict(color='green', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,255,0,0.1)'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=accuracies,
                mode='markers',
                name='Individual Architectures',
                marker=dict(color='blue', size=6, opacity=0.5)
            ),
            row=1, col=1
        )
        
        # Plot 2: Cumulative count
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=iterations,
                mode='lines',
                name='Architectures Evaluated',
                line=dict(color='purple', width=2),
                fill='tozeroy',
                fillcolor='rgba(128,0,128,0.1)'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_xaxes(title_text="Iteration", row=1, col=1)
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Performance", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="Neural Architecture Search Progress",
            showlegend=True,
            hovermode='x unified'
        )
        
        # Return in requested format
        if format == 'base64':
            img_bytes = fig.to_image(format="png", width=1000, height=700)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        elif format == 'html':
            return fig.to_html(include_plotlyjs='cdn')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def create_pareto_front_plot(
        self,
        architectures: List[Architecture],
        objectives: List[str] = ['accuracy', 'latency', 'model_size'],
        format: str = 'base64',
        highlight_pareto: bool = True
    ) -> str:
        """
        Create Pareto front visualization for multi-objective optimization.
        
        Args:
            architectures: List of evaluated architectures
            objectives: List of objective names to plot
            format: Output format ('base64', 'html')
            highlight_pareto: Whether to highlight Pareto-optimal solutions
        
        Returns:
            Plot as base64 encoded string or HTML
        """
        if not architectures:
            return ""
        
        # Extract metrics
        metrics_data = []
        for arch in architectures:
            metrics = arch.metadata.get('metrics', {})
            if all(obj in metrics for obj in objectives[:2]):  # At least 2 objectives needed
                metrics_data.append({
                    'id': arch.id[:8],
                    **{obj: metrics[obj] for obj in objectives if obj in metrics}
                })
        
        if not metrics_data:
            logger.warning("No architectures with required metrics found")
            return ""
        
        # Determine if 2D or 3D plot
        if len(objectives) >= 3 and all(obj in metrics_data[0] for obj in objectives[:3]):
            return self._create_3d_pareto_plot(metrics_data, objectives[:3], format, highlight_pareto)
        else:
            return self._create_2d_pareto_plot(metrics_data, objectives[:2], format, highlight_pareto)
    
    def _create_2d_pareto_plot(
        self,
        metrics_data: List[Dict[str, Any]],
        objectives: List[str],
        format: str,
        highlight_pareto: bool
    ) -> str:
        """Create 2D Pareto front plot."""
        obj1, obj2 = objectives[0], objectives[1]
        
        # Determine if objectives should be minimized or maximized
        # Accuracy is maximized, latency and model_size are minimized
        maximize_obj1 = 'accuracy' in obj1.lower() or 'score' in obj1.lower()
        maximize_obj2 = 'accuracy' in obj2.lower() or 'score' in obj2.lower()
        
        # Identify Pareto front if requested
        pareto_indices = set()
        if highlight_pareto:
            pareto_indices = self._compute_pareto_front_2d(
                metrics_data, obj1, obj2, maximize_obj1, maximize_obj2
            )
        
        # Create scatter plot
        fig = go.Figure()
        
        # Non-Pareto points
        non_pareto_data = [m for i, m in enumerate(metrics_data) if i not in pareto_indices]
        if non_pareto_data:
            fig.add_trace(go.Scatter(
                x=[m[obj1] for m in non_pareto_data],
                y=[m[obj2] for m in non_pareto_data],
                mode='markers',
                name='Non-Pareto',
                marker=dict(size=10, color='lightblue', opacity=0.6),
                text=[m['id'] for m in non_pareto_data],
                hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<extra></extra>'
            ))
        
        # Pareto points
        if pareto_indices:
            pareto_data = [metrics_data[i] for i in sorted(pareto_indices)]
            fig.add_trace(go.Scatter(
                x=[m[obj1] for m in pareto_data],
                y=[m[obj2] for m in pareto_data],
                mode='markers+lines',
                name='Pareto Front',
                marker=dict(size=15, color='red', symbol='star'),
                line=dict(color='red', width=2, dash='dash'),
                text=[m['id'] for m in pareto_data],
                hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Pareto Front: {obj1.title()} vs {obj2.title()}',
            xaxis_title=obj1.replace('_', ' ').title(),
            yaxis_title=obj2.replace('_', ' ').title(),
            height=600,
            hovermode='closest',
            showlegend=True
        )
        
        # Return in requested format
        if format == 'base64':
            img_bytes = fig.to_image(format="png", width=800, height=600)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        elif format == 'html':
            return fig.to_html(include_plotlyjs='cdn')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _create_3d_pareto_plot(
        self,
        metrics_data: List[Dict[str, Any]],
        objectives: List[str],
        format: str,
        highlight_pareto: bool
    ) -> str:
        """Create 3D Pareto front plot."""
        obj1, obj2, obj3 = objectives[0], objectives[1], objectives[2]
        
        # Determine if objectives should be minimized or maximized
        maximize_obj1 = 'accuracy' in obj1.lower() or 'score' in obj1.lower()
        maximize_obj2 = 'accuracy' in obj2.lower() or 'score' in obj2.lower()
        maximize_obj3 = 'accuracy' in obj3.lower() or 'score' in obj3.lower()
        
        # Identify Pareto front if requested
        pareto_indices = set()
        if highlight_pareto:
            pareto_indices = self._compute_pareto_front_3d(
                metrics_data, obj1, obj2, obj3,
                maximize_obj1, maximize_obj2, maximize_obj3
            )
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Non-Pareto points
        non_pareto_data = [m for i, m in enumerate(metrics_data) if i not in pareto_indices]
        if non_pareto_data:
            fig.add_trace(go.Scatter3d(
                x=[m[obj1] for m in non_pareto_data],
                y=[m[obj2] for m in non_pareto_data],
                z=[m[obj3] for m in non_pareto_data],
                mode='markers',
                name='Non-Pareto',
                marker=dict(size=6, color='lightblue', opacity=0.6),
                text=[m['id'] for m in non_pareto_data],
                hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<br>{obj3}: %{{z}}<extra></extra>'
            ))
        
        # Pareto points
        if pareto_indices:
            pareto_data = [metrics_data[i] for i in sorted(pareto_indices)]
            fig.add_trace(go.Scatter3d(
                x=[m[obj1] for m in pareto_data],
                y=[m[obj2] for m in pareto_data],
                z=[m[obj3] for m in pareto_data],
                mode='markers',
                name='Pareto Front',
                marker=dict(size=10, color='red', symbol='diamond'),
                text=[m['id'] for m in pareto_data],
                hovertemplate=f'<b>%{{text}}</b><br>{obj1}: %{{x}}<br>{obj2}: %{{y}}<br>{obj3}: %{{z}}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D Pareto Front: {obj1.title()} vs {obj2.title()} vs {obj3.title()}',
            scene=dict(
                xaxis_title=obj1.replace('_', ' ').title(),
                yaxis_title=obj2.replace('_', ' ').title(),
                zaxis_title=obj3.replace('_', ' ').title()
            ),
            height=700,
            hovermode='closest',
            showlegend=True
        )
        
        # Return in requested format
        if format == 'base64':
            img_bytes = fig.to_image(format="png", width=900, height=700)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        elif format == 'html':
            return fig.to_html(include_plotlyjs='cdn')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _compute_pareto_front_2d(
        self,
        metrics_data: List[Dict[str, Any]],
        obj1: str,
        obj2: str,
        maximize_obj1: bool,
        maximize_obj2: bool
    ) -> set:
        """Compute Pareto front for 2D objectives."""
        pareto_indices = set()
        
        for i, point_i in enumerate(metrics_data):
            is_dominated = False
            
            for j, point_j in enumerate(metrics_data):
                if i == j:
                    continue
                
                # Check if point_i is dominated by point_j
                better_obj1 = (
                    (point_j[obj1] > point_i[obj1] if maximize_obj1 else point_j[obj1] < point_i[obj1])
                )
                better_obj2 = (
                    (point_j[obj2] > point_i[obj2] if maximize_obj2 else point_j[obj2] < point_i[obj2])
                )
                
                equal_obj1 = abs(point_j[obj1] - point_i[obj1]) < 1e-9
                equal_obj2 = abs(point_j[obj2] - point_i[obj2]) < 1e-9
                
                # point_j dominates point_i if it's better in at least one objective
                # and not worse in any objective
                if (better_obj1 or equal_obj1) and (better_obj2 or equal_obj2) and (better_obj1 or better_obj2):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.add(i)
        
        return pareto_indices
    
    def _compute_pareto_front_3d(
        self,
        metrics_data: List[Dict[str, Any]],
        obj1: str,
        obj2: str,
        obj3: str,
        maximize_obj1: bool,
        maximize_obj2: bool,
        maximize_obj3: bool
    ) -> set:
        """Compute Pareto front for 3D objectives."""
        pareto_indices = set()
        
        for i, point_i in enumerate(metrics_data):
            is_dominated = False
            
            for j, point_j in enumerate(metrics_data):
                if i == j:
                    continue
                
                # Check if point_i is dominated by point_j
                better_obj1 = (
                    (point_j[obj1] > point_i[obj1] if maximize_obj1 else point_j[obj1] < point_i[obj1])
                )
                better_obj2 = (
                    (point_j[obj2] > point_i[obj2] if maximize_obj2 else point_j[obj2] < point_i[obj2])
                )
                better_obj3 = (
                    (point_j[obj3] > point_i[obj3] if maximize_obj3 else point_j[obj3] < point_i[obj3])
                )
                
                equal_obj1 = abs(point_j[obj1] - point_i[obj1]) < 1e-9
                equal_obj2 = abs(point_j[obj2] - point_i[obj2]) < 1e-9
                equal_obj3 = abs(point_j[obj3] - point_i[obj3]) < 1e-9
                
                # point_j dominates point_i if it's better in at least one objective
                # and not worse in any objective
                if ((better_obj1 or equal_obj1) and (better_obj2 or equal_obj2) and 
                    (better_obj3 or equal_obj3) and (better_obj1 or better_obj2 or better_obj3)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_indices.add(i)
        
        return pareto_indices

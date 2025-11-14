"""
Multi-objective optimization for Neural Architecture Search.

This module provides multi-objective optimization capabilities for NAS,
including Pareto dominance checking, Pareto front calculation, objective
weighting, constraint satisfaction, and visualization.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any
import numpy as np
import logging
import re

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class Objective:
    """Represents an optimization objective.
    
    Attributes:
        name: Name of the objective (e.g., 'accuracy', 'latency', 'model_size')
        direction: 'maximize' or 'minimize'
        weight: Weight for scalarization (default: 1.0)
    """
    name: str
    direction: str  # 'maximize' or 'minimize'
    weight: float = 1.0
    
    def __post_init__(self):
        if self.direction not in ['maximize', 'minimize']:
            raise ValueError(f"Direction must be 'maximize' or 'minimize', got {self.direction}")
        if self.weight < 0:
            raise ValueError(f"Weight must be non-negative, got {self.weight}")


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for NAS architectures.
    
    This class handles optimization of multiple competing objectives such as
    accuracy, latency, and model size. It provides Pareto dominance checking,
    Pareto front calculation, objective weighting, and constraint satisfaction.
    
    Args:
        objectives: List of Objective instances defining what to optimize
        constraints: Optional list of constraint expressions
        
    Example:
        >>> objectives = [
        ...     Objective('accuracy', 'maximize', weight=0.6),
        ...     Objective('latency', 'minimize', weight=0.3),
        ...     Objective('model_size', 'minimize', weight=0.1)
        ... ]
        >>> optimizer = MultiObjectiveOptimizer(objectives)
        >>> pareto_front = optimizer.compute_pareto_front(architectures)
    """
    
    def __init__(
        self,
        objectives: List[Objective],
        constraints: Optional[List[str]] = None
    ):
        """Initialize the multi-objective optimizer.
        
        Args:
            objectives: List of Objective instances
            constraints: Optional list of constraint expressions
        """
        if not objectives:
            raise ValueError("At least one objective must be specified")
            
        self.objectives = objectives
        self.constraints = constraints or []
        self._validate_objectives()
        
        logger.info(f"Initialized MultiObjectiveOptimizer with {len(objectives)} objectives")
        for obj in objectives:
            logger.info(f"  - {obj.name}: {obj.direction} (weight={obj.weight})")
    
    def _validate_objectives(self):
        """Validate that objectives are properly configured."""
        names = [obj.name for obj in self.objectives]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate objective names found")
        
        total_weight = sum(obj.weight for obj in self.objectives)
        if total_weight == 0:
            raise ValueError("Total objective weight cannot be zero")
    
    def dominates(
        self,
        arch1_metrics: Dict[str, float],
        arch2_metrics: Dict[str, float]
    ) -> bool:
        """Check if arch1 dominates arch2 (Pareto dominance).
        
        Architecture A dominates architecture B if:
        - A is better than or equal to B in all objectives
        - A is strictly better than B in at least one objective
        
        Args:
            arch1_metrics: Metrics dictionary for architecture 1
            arch2_metrics: Metrics dictionary for architecture 2
            
        Returns:
            True if arch1 dominates arch2, False otherwise
            
        Example:
            >>> metrics1 = {'accuracy': 0.95, 'latency': 50}
            >>> metrics2 = {'accuracy': 0.90, 'latency': 60}
            >>> optimizer.dominates(metrics1, metrics2)
            True
        """
        better_in_any = False
        worse_in_any = False
        
        for obj in self.objectives:
            if obj.name not in arch1_metrics or obj.name not in arch2_metrics:
                logger.warning(f"Objective {obj.name} not found in metrics, skipping")
                continue
            
            val1 = arch1_metrics[obj.name]
            val2 = arch2_metrics[obj.name]
            
            # Normalize comparison based on direction
            if obj.direction == 'maximize':
                if val1 > val2:
                    better_in_any = True
                elif val1 < val2:
                    worse_in_any = True
            else:  # minimize
                if val1 < val2:
                    better_in_any = True
                elif val1 > val2:
                    worse_in_any = True
        
        # Dominates if better in at least one and not worse in any
        return better_in_any and not worse_in_any
    
    def compute_pareto_front(
        self,
        architectures: List[Any],
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> List[Any]:
        """Compute the Pareto front from a list of architectures.
        
        The Pareto front consists of all non-dominated solutions. An architecture
        is in the Pareto front if no other architecture dominates it.
        
        Args:
            architectures: List of architecture objects
            get_metrics: Optional function to extract metrics from architecture.
                        If None, assumes architecture has a 'metrics' attribute.
                        
        Returns:
            List of architectures in the Pareto front
            
        Example:
            >>> pareto_front = optimizer.compute_pareto_front(architectures)
            >>> print(f"Found {len(pareto_front)} non-dominated solutions")
        """
        if not architectures:
            return []
        
        # Default metrics extractor
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        pareto_front = []
        
        for arch in architectures:
            arch_metrics = get_metrics(arch)
            
            # Check if this architecture is dominated by any other
            is_dominated = False
            for other_arch in architectures:
                if arch is other_arch:
                    continue
                
                other_metrics = get_metrics(other_arch)
                if self.dominates(other_metrics, arch_metrics):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(arch)
        
        logger.info(f"Computed Pareto front: {len(pareto_front)} out of {len(architectures)} architectures")
        return pareto_front
    
    def compute_pareto_rank(
        self,
        architectures: List[Any],
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> Dict[Any, int]:
        """Compute Pareto rank for each architecture.
        
        Rank 0 = Pareto front (non-dominated)
        Rank 1 = Dominated only by rank 0
        Rank 2 = Dominated only by rank 0 and 1, etc.
        
        Args:
            architectures: List of architecture objects
            get_metrics: Optional function to extract metrics from architecture
            
        Returns:
            Dictionary mapping architecture to its Pareto rank
        """
        if not architectures:
            return {}
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        ranks = {}
        remaining = list(architectures)
        current_rank = 0
        
        while remaining:
            # Find non-dominated architectures in remaining set
            front = []
            for arch in remaining:
                arch_metrics = get_metrics(arch)
                is_dominated = False
                
                for other_arch in remaining:
                    if arch is other_arch:
                        continue
                    other_metrics = get_metrics(other_arch)
                    if self.dominates(other_metrics, arch_metrics):
                        is_dominated = True
                        break
                
                if not is_dominated:
                    front.append(arch)
            
            # Assign rank to front
            for arch in front:
                ranks[arch] = current_rank
                remaining.remove(arch)
            
            current_rank += 1
        
        return ranks
    
    def scalarize(
        self,
        metrics: Dict[str, float],
        normalize: bool = True,
        reference_metrics: Optional[List[Dict[str, float]]] = None
    ) -> float:
        """Convert multi-objective metrics to a single scalar value.
        
        Uses weighted sum scalarization. Optionally normalizes objectives
        to [0, 1] range based on reference metrics.
        
        Args:
            metrics: Dictionary of objective values
            normalize: Whether to normalize objectives before weighting
            reference_metrics: Optional list of metrics for normalization
            
        Returns:
            Scalar value representing weighted combination of objectives
            
        Example:
            >>> metrics = {'accuracy': 0.95, 'latency': 50, 'model_size': 10}
            >>> score = optimizer.scalarize(metrics)
        """
        if normalize and reference_metrics:
            metrics = self._normalize_metrics(metrics, reference_metrics)
        
        score = 0.0
        total_weight = sum(obj.weight for obj in self.objectives)
        
        for obj in self.objectives:
            if obj.name not in metrics:
                logger.warning(f"Objective {obj.name} not found in metrics")
                continue
            
            value = metrics[obj.name]
            
            # For minimization objectives, use negative value
            if obj.direction == 'minimize':
                value = -value
            
            score += obj.weight * value
        
        # Normalize by total weight
        if total_weight > 0:
            score /= total_weight
        
        return score
    
    def _normalize_metrics(
        self,
        metrics: Dict[str, float],
        reference_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Normalize metrics to [0, 1] range based on reference metrics.
        
        Args:
            metrics: Metrics to normalize
            reference_metrics: List of reference metrics for computing min/max
            
        Returns:
            Normalized metrics dictionary
        """
        normalized = {}
        
        for obj in self.objectives:
            if obj.name not in metrics:
                continue
            
            # Collect all values for this objective
            values = [m[obj.name] for m in reference_metrics if obj.name in m]
            if not values:
                normalized[obj.name] = metrics[obj.name]
                continue
            
            min_val = min(values)
            max_val = max(values)
            
            if max_val == min_val:
                normalized[obj.name] = 0.5
            else:
                # Normalize to [0, 1]
                normalized[obj.name] = (metrics[obj.name] - min_val) / (max_val - min_val)
        
        return normalized
    
    def compute_hypervolume(
        self,
        architectures: List[Any],
        reference_point: Optional[Dict[str, float]] = None,
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> float:
        """Compute hypervolume indicator for a set of architectures.
        
        Hypervolume measures the volume of objective space dominated by the
        Pareto front. Higher is better. Only works for 2-3 objectives.
        
        Args:
            architectures: List of architecture objects
            reference_point: Reference point for hypervolume calculation
            get_metrics: Optional function to extract metrics from architecture
            
        Returns:
            Hypervolume value
        """
        if len(self.objectives) > 3:
            logger.warning("Hypervolume calculation only supported for 2-3 objectives")
            return 0.0
        
        if not architectures:
            return 0.0
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        # Extract objective values
        points = []
        for arch in architectures:
            metrics = get_metrics(arch)
            point = []
            for obj in self.objectives:
                if obj.name not in metrics:
                    logger.warning(f"Objective {obj.name} not found in metrics")
                    return 0.0
                val = metrics[obj.name]
                # For maximization, negate to convert to minimization
                if obj.direction == 'maximize':
                    val = -val
                point.append(val)
            points.append(point)
        
        # Set reference point if not provided
        if reference_point is None:
            ref_point = []
            for i, obj in enumerate(self.objectives):
                max_val = max(p[i] for p in points)
                ref_point.append(max_val + 1.0)  # Slightly worse than worst point
        else:
            ref_point = []
            for obj in self.objectives:
                val = reference_point.get(obj.name, 0)
                if obj.direction == 'maximize':
                    val = -val
                ref_point.append(val)
        
        # Simple hypervolume calculation for 2D case
        if len(self.objectives) == 2:
            return self._hypervolume_2d(points, ref_point)
        elif len(self.objectives) == 3:
            return self._hypervolume_3d(points, ref_point)
        
        return 0.0
    
    def _hypervolume_2d(self, points: List[List[float]], ref_point: List[float]) -> float:
        """Compute 2D hypervolume using sweep algorithm."""
        if not points:
            return 0.0
        
        # Sort points by first objective (ascending)
        sorted_points = sorted(points, key=lambda p: p[0])
        
        hypervolume = 0.0
        prev_y = ref_point[1]
        
        for i, point in enumerate(sorted_points):
            # Only consider points that dominate the reference point
            if point[0] < ref_point[0] and point[1] < ref_point[1]:
                # Width: distance to next point (or reference point)
                if i + 1 < len(sorted_points):
                    next_x = sorted_points[i + 1][0]
                else:
                    next_x = ref_point[0]
                
                width = next_x - point[0]
                height = prev_y - point[1]
                
                if width > 0 and height > 0:
                    hypervolume += width * height
                    prev_y = min(prev_y, point[1])
        
        return hypervolume
    
    def _hypervolume_3d(self, points: List[List[float]], ref_point: List[float]) -> float:
        """Compute 3D hypervolume using slicing algorithm."""
        if not points:
            return 0.0
        
        # Sort points by first objective
        sorted_points = sorted(points, key=lambda p: p[0])
        
        hypervolume = 0.0
        
        for i, point in enumerate(sorted_points):
            # Create 2D slice
            slice_points = []
            for other_point in sorted_points[i:]:
                if other_point[0] >= point[0]:
                    slice_points.append([other_point[1], other_point[2]])
            
            # Compute 2D hypervolume for slice
            if slice_points:
                slice_ref = [ref_point[1], ref_point[2]]
                next_x = sorted_points[i + 1][0] if i + 1 < len(sorted_points) else ref_point[0]
                width = next_x - point[0]
                slice_hv = self._hypervolume_2d(slice_points, slice_ref)
                hypervolume += width * slice_hv
        
        return hypervolume
    
    def set_objective_weights(self, weights: Dict[str, float]):
        """Update objective weights.
        
        Args:
            weights: Dictionary mapping objective names to weights
            
        Example:
            >>> optimizer.set_objective_weights({
            ...     'accuracy': 0.7,
            ...     'latency': 0.2,
            ...     'model_size': 0.1
            ... })
        """
        for obj in self.objectives:
            if obj.name in weights:
                if weights[obj.name] < 0:
                    raise ValueError(f"Weight for {obj.name} must be non-negative")
                obj.weight = weights[obj.name]
        
        # Validate total weight
        total_weight = sum(obj.weight for obj in self.objectives)
        if total_weight == 0:
            raise ValueError("Total objective weight cannot be zero")
        
        logger.info("Updated objective weights:")
        for obj in self.objectives:
            logger.info(f"  - {obj.name}: {obj.weight}")
    
    def get_objective_weights(self) -> Dict[str, float]:
        """Get current objective weights.
        
        Returns:
            Dictionary mapping objective names to weights
        """
        return {obj.name: obj.weight for obj in self.objectives}
    
    def normalize_weights(self):
        """Normalize objective weights to sum to 1.0."""
        total_weight = sum(obj.weight for obj in self.objectives)
        if total_weight > 0:
            for obj in self.objectives:
                obj.weight /= total_weight
        
        logger.info("Normalized objective weights to sum to 1.0")
    
    def select_best_architecture(
        self,
        architectures: List[Any],
        preferences: Optional[Dict[str, float]] = None,
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> Optional[Any]:
        """Select the best architecture based on preferences.
        
        If preferences are provided, temporarily updates weights and uses
        scalarization. Otherwise, returns the architecture with best scalarized
        score using current weights.
        
        Args:
            architectures: List of architecture objects
            preferences: Optional dictionary of objective weights
            get_metrics: Optional function to extract metrics from architecture
            
        Returns:
            Best architecture according to preferences, or None if empty list
            
        Example:
            >>> best = optimizer.select_best_architecture(
            ...     pareto_front,
            ...     preferences={'accuracy': 0.8, 'latency': 0.2}
            ... )
        """
        if not architectures:
            return None
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        # Temporarily update weights if preferences provided
        original_weights = None
        if preferences:
            original_weights = self.get_objective_weights()
            self.set_objective_weights(preferences)
        
        try:
            # Rank architectures using scalarization
            ranked = self.rank_architectures(architectures, method='scalarize', get_metrics=get_metrics)
            best_arch = ranked[0][0] if ranked else None
            
            if best_arch:
                metrics = get_metrics(best_arch)
                logger.info(f"Selected best architecture with metrics: {metrics}")
            
            return best_arch
        
        finally:
            # Restore original weights
            if original_weights:
                self.set_objective_weights(original_weights)
    
    def rank_architectures(
        self,
        architectures: List[Any],
        method: str = 'scalarize',
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> List[Tuple[Any, float]]:
        """Rank architectures by a single score.
        
        Args:
            architectures: List of architecture objects
            method: Ranking method ('scalarize' or 'pareto_rank')
            get_metrics: Optional function to extract metrics from architecture
            
        Returns:
            List of (architecture, score) tuples sorted by score (descending)
        """
        if not architectures:
            return []
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        if method == 'scalarize':
            # Collect all metrics for normalization
            all_metrics = [get_metrics(arch) for arch in architectures]
            
            # Compute scalar scores
            scores = []
            for arch in architectures:
                metrics = get_metrics(arch)
                score = self.scalarize(metrics, normalize=True, reference_metrics=all_metrics)
                scores.append((arch, score))
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
        
        elif method == 'pareto_rank':
            # Compute Pareto ranks
            ranks = self.compute_pareto_rank(architectures, get_metrics)
            
            # Convert to scores (lower rank = higher score)
            max_rank = max(ranks.values()) if ranks else 0
            scores = [(arch, max_rank - ranks[arch]) for arch in architectures]
            
            # Sort by score (descending)
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores
        
        else:
            raise ValueError(f"Unknown ranking method: {method}")
    
    def parse_constraint(self, constraint_expr: str) -> Callable[[Dict[str, float]], bool]:
        """Parse a constraint expression into a callable function.
        
        Supports expressions like:
        - "accuracy > 0.9"
        - "latency < 100"
        - "accuracy > 0.9 AND latency < 100"
        - "accuracy >= 0.85 OR model_size <= 10"
        
        Args:
            constraint_expr: Constraint expression string
            
        Returns:
            Function that takes metrics dict and returns True if satisfied
            
        Example:
            >>> constraint_fn = optimizer.parse_constraint("accuracy > 0.9 AND latency < 100")
            >>> metrics = {'accuracy': 0.95, 'latency': 80}
            >>> constraint_fn(metrics)
            True
        """
        # Normalize expression
        expr = constraint_expr.strip()
        
        # Split by AND/OR operators
        if ' AND ' in expr.upper():
            parts = re.split(r'\s+AND\s+', expr, flags=re.IGNORECASE)
            sub_constraints = [self.parse_constraint(part) for part in parts]
            return lambda metrics: all(fn(metrics) for fn in sub_constraints)
        
        elif ' OR ' in expr.upper():
            parts = re.split(r'\s+OR\s+', expr, flags=re.IGNORECASE)
            sub_constraints = [self.parse_constraint(part) for part in parts]
            return lambda metrics: any(fn(metrics) for fn in sub_constraints)
        
        # Parse single constraint
        # Match patterns like "metric_name operator value"
        pattern = r'(\w+)\s*([<>=!]+)\s*([\d.]+)'
        match = re.match(pattern, expr)
        
        if not match:
            raise ValueError(f"Invalid constraint expression: {constraint_expr}")
        
        metric_name = match.group(1)
        operator = match.group(2)
        threshold = float(match.group(3))
        
        # Create comparison function
        def constraint_fn(metrics: Dict[str, float]) -> bool:
            if metric_name not in metrics:
                logger.warning(f"Metric {metric_name} not found in architecture metrics")
                return False
            
            value = metrics[metric_name]
            
            if operator == '>':
                return value > threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<':
                return value < threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return abs(value - threshold) < 1e-9
            elif operator == '!=':
                return abs(value - threshold) >= 1e-9
            else:
                raise ValueError(f"Unknown operator: {operator}")
        
        return constraint_fn
    
    def check_constraints(
        self,
        metrics: Dict[str, float],
        constraints: Optional[List[str]] = None
    ) -> bool:
        """Check if metrics satisfy all constraints.
        
        Args:
            metrics: Dictionary of metric values
            constraints: Optional list of constraint expressions.
                        If None, uses self.constraints.
                        
        Returns:
            True if all constraints are satisfied, False otherwise
            
        Example:
            >>> constraints = ["accuracy > 0.9", "latency < 100"]
            >>> metrics = {'accuracy': 0.95, 'latency': 80}
            >>> optimizer.check_constraints(metrics, constraints)
            True
        """
        constraint_list = constraints if constraints is not None else self.constraints
        
        if not constraint_list:
            return True
        
        for constraint_expr in constraint_list:
            try:
                constraint_fn = self.parse_constraint(constraint_expr)
                if not constraint_fn(metrics):
                    logger.debug(f"Constraint not satisfied: {constraint_expr}")
                    return False
            except Exception as e:
                logger.error(f"Error evaluating constraint '{constraint_expr}': {e}")
                return False
        
        return True
    
    def filter_by_constraints(
        self,
        architectures: List[Any],
        constraints: Optional[List[str]] = None,
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None
    ) -> List[Any]:
        """Filter architectures that satisfy all constraints.
        
        Args:
            architectures: List of architecture objects
            constraints: Optional list of constraint expressions
            get_metrics: Optional function to extract metrics from architecture
            
        Returns:
            List of architectures that satisfy all constraints
            
        Example:
            >>> constraints = ["accuracy > 0.9", "latency < 100"]
            >>> valid_archs = optimizer.filter_by_constraints(architectures, constraints)
        """
        if not architectures:
            return []
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        constraint_list = constraints if constraints is not None else self.constraints
        
        if not constraint_list:
            return architectures
        
        filtered = []
        for arch in architectures:
            metrics = get_metrics(arch)
            if self.check_constraints(metrics, constraint_list):
                filtered.append(arch)
        
        logger.info(f"Filtered {len(filtered)} out of {len(architectures)} architectures by constraints")
        return filtered
    
    def add_constraint(self, constraint_expr: str):
        """Add a constraint to the optimizer.
        
        Args:
            constraint_expr: Constraint expression string
            
        Example:
            >>> optimizer.add_constraint("accuracy > 0.9")
            >>> optimizer.add_constraint("latency < 100 AND model_size < 10")
        """
        # Validate constraint by parsing it
        try:
            self.parse_constraint(constraint_expr)
            self.constraints.append(constraint_expr)
            logger.info(f"Added constraint: {constraint_expr}")
        except Exception as e:
            raise ValueError(f"Invalid constraint expression: {e}")
    
    def remove_constraint(self, constraint_expr: str):
        """Remove a constraint from the optimizer.
        
        Args:
            constraint_expr: Constraint expression string to remove
        """
        if constraint_expr in self.constraints:
            self.constraints.remove(constraint_expr)
            logger.info(f"Removed constraint: {constraint_expr}")
        else:
            logger.warning(f"Constraint not found: {constraint_expr}")
    
    def clear_constraints(self):
        """Remove all constraints."""
        self.constraints = []
        logger.info("Cleared all constraints")
    
    def visualize_pareto_front(
        self,
        architectures: List[Any],
        pareto_front: Optional[List[Any]] = None,
        get_metrics: Optional[Callable[[Any], Dict[str, float]]] = None,
        backend: str = 'matplotlib',
        save_path: Optional[str] = None,
        show: bool = True
    ) -> Optional[Any]:
        """Visualize the Pareto front with 2D or 3D scatter plots.
        
        Creates scatter plots showing the trade-offs between objectives.
        Non-dominated solutions (Pareto front) are highlighted.
        
        Args:
            architectures: List of all architecture objects
            pareto_front: Optional list of Pareto front architectures.
                         If None, will be computed automatically.
            get_metrics: Optional function to extract metrics from architecture
            backend: Visualization backend ('matplotlib' or 'plotly')
            save_path: Optional path to save the figure
            show: Whether to display the plot
            
        Returns:
            Figure object (matplotlib Figure or plotly Figure)
            
        Example:
            >>> fig = optimizer.visualize_pareto_front(
            ...     architectures,
            ...     backend='plotly',
            ...     save_path='pareto_front.html'
            ... )
        """
        if not architectures:
            logger.warning("No architectures to visualize")
            return None
        
        if get_metrics is None:
            get_metrics = lambda arch: getattr(arch, 'metrics', {})
        
        # Compute Pareto front if not provided
        if pareto_front is None:
            pareto_front = self.compute_pareto_front(architectures, get_metrics)
        
        # Extract metrics for all architectures
        all_metrics = [get_metrics(arch) for arch in architectures]
        pareto_metrics = [get_metrics(arch) for arch in pareto_front]
        
        # Determine number of objectives to plot
        n_objectives = len(self.objectives)
        
        if n_objectives < 2:
            logger.warning("Need at least 2 objectives for visualization")
            return None
        
        if backend == 'matplotlib':
            return self._visualize_matplotlib(
                all_metrics, pareto_metrics, save_path, show
            )
        elif backend == 'plotly':
            return self._visualize_plotly(
                all_metrics, pareto_metrics, save_path, show
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _visualize_matplotlib(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create visualization using matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("Matplotlib not available. Install with: pip install matplotlib")
            return None
        
        n_objectives = len(self.objectives)
        
        if n_objectives == 2:
            return self._plot_2d_matplotlib(all_metrics, pareto_metrics, save_path, show)
        elif n_objectives == 3:
            return self._plot_3d_matplotlib(all_metrics, pareto_metrics, save_path, show)
        else:
            logger.warning("Matplotlib visualization only supports 2-3 objectives")
            return None
    
    def _plot_2d_matplotlib(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create 2D scatter plot using matplotlib."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        obj1, obj2 = self.objectives[0], self.objectives[1]
        
        # Extract values
        all_x = [m[obj1.name] for m in all_metrics if obj1.name in m]
        all_y = [m[obj2.name] for m in all_metrics if obj2.name in m]
        pareto_x = [m[obj1.name] for m in pareto_metrics if obj1.name in m]
        pareto_y = [m[obj2.name] for m in pareto_metrics if obj2.name in m]
        
        # Plot all architectures
        ax.scatter(all_x, all_y, c='lightblue', s=50, alpha=0.6, label='All Architectures')
        
        # Plot Pareto front
        ax.scatter(pareto_x, pareto_y, c='red', s=100, alpha=0.8, 
                  label='Pareto Front', edgecolors='darkred', linewidths=2)
        
        # Connect Pareto front points
        if len(pareto_x) > 1:
            # Sort by first objective
            sorted_indices = np.argsort(pareto_x)
            sorted_x = [pareto_x[i] for i in sorted_indices]
            sorted_y = [pareto_y[i] for i in sorted_indices]
            ax.plot(sorted_x, sorted_y, 'r--', alpha=0.5, linewidth=1)
        
        # Labels and title
        ax.set_xlabel(f"{obj1.name} ({obj1.direction})", fontsize=12)
        ax.set_ylabel(f"{obj2.name} ({obj2.direction})", fontsize=12)
        ax.set_title("Pareto Front Visualization", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_3d_matplotlib(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create 3D scatter plot using matplotlib."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        obj1, obj2, obj3 = self.objectives[0], self.objectives[1], self.objectives[2]
        
        # Extract values
        all_x = [m[obj1.name] for m in all_metrics if obj1.name in m]
        all_y = [m[obj2.name] for m in all_metrics if obj2.name in m]
        all_z = [m[obj3.name] for m in all_metrics if obj3.name in m]
        pareto_x = [m[obj1.name] for m in pareto_metrics if obj1.name in m]
        pareto_y = [m[obj2.name] for m in pareto_metrics if obj2.name in m]
        pareto_z = [m[obj3.name] for m in pareto_metrics if obj3.name in m]
        
        # Plot all architectures
        ax.scatter(all_x, all_y, all_z, c='lightblue', s=50, alpha=0.4, 
                  label='All Architectures')
        
        # Plot Pareto front
        ax.scatter(pareto_x, pareto_y, pareto_z, c='red', s=100, alpha=0.8,
                  label='Pareto Front', edgecolors='darkred', linewidths=2)
        
        # Labels and title
        ax.set_xlabel(f"{obj1.name} ({obj1.direction})", fontsize=10)
        ax.set_ylabel(f"{obj2.name} ({obj2.direction})", fontsize=10)
        ax.set_zlabel(f"{obj3.name} ({obj3.direction})", fontsize=10)
        ax.set_title("Pareto Front Visualization (3D)", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _visualize_plotly(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create interactive visualization using plotly."""
        if not PLOTLY_AVAILABLE:
            logger.error("Plotly not available. Install with: pip install plotly")
            return None
        
        n_objectives = len(self.objectives)
        
        if n_objectives == 2:
            return self._plot_2d_plotly(all_metrics, pareto_metrics, save_path, show)
        elif n_objectives == 3:
            return self._plot_3d_plotly(all_metrics, pareto_metrics, save_path, show)
        else:
            logger.warning("Plotly visualization only supports 2-3 objectives")
            return None
    
    def _plot_2d_plotly(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create 2D interactive scatter plot using plotly."""
        obj1, obj2 = self.objectives[0], self.objectives[1]
        
        # Extract values
        all_x = [m[obj1.name] for m in all_metrics if obj1.name in m]
        all_y = [m[obj2.name] for m in all_metrics if obj2.name in m]
        pareto_x = [m[obj1.name] for m in pareto_metrics if obj1.name in m]
        pareto_y = [m[obj2.name] for m in pareto_metrics if obj2.name in m]
        
        # Create hover text
        all_hover = [
            f"{obj1.name}: {m[obj1.name]:.4f}<br>{obj2.name}: {m[obj2.name]:.4f}"
            for m in all_metrics if obj1.name in m and obj2.name in m
        ]
        pareto_hover = [
            f"{obj1.name}: {m[obj1.name]:.4f}<br>{obj2.name}: {m[obj2.name]:.4f}"
            for m in pareto_metrics if obj1.name in m and obj2.name in m
        ]
        
        # Create traces
        trace_all = go.Scatter(
            x=all_x, y=all_y,
            mode='markers',
            name='All Architectures',
            marker=dict(size=8, color='lightblue', opacity=0.6),
            hovertext=all_hover,
            hoverinfo='text'
        )
        
        trace_pareto = go.Scatter(
            x=pareto_x, y=pareto_y,
            mode='markers+lines',
            name='Pareto Front',
            marker=dict(size=12, color='red', opacity=0.8,
                       line=dict(width=2, color='darkred')),
            line=dict(color='red', width=1, dash='dash'),
            hovertext=pareto_hover,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[trace_all, trace_pareto])
        
        fig.update_layout(
            title="Pareto Front Visualization",
            xaxis_title=f"{obj1.name} ({obj1.direction})",
            yaxis_title=f"{obj2.name} ({obj2.direction})",
            hovermode='closest',
            width=900,
            height=700
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        if show:
            fig.show()
        
        return fig
    
    def _plot_3d_plotly(
        self,
        all_metrics: List[Dict[str, float]],
        pareto_metrics: List[Dict[str, float]],
        save_path: Optional[str],
        show: bool
    ):
        """Create 3D interactive scatter plot using plotly."""
        obj1, obj2, obj3 = self.objectives[0], self.objectives[1], self.objectives[2]
        
        # Extract values
        all_x = [m[obj1.name] for m in all_metrics if obj1.name in m]
        all_y = [m[obj2.name] for m in all_metrics if obj2.name in m]
        all_z = [m[obj3.name] for m in all_metrics if obj3.name in m]
        pareto_x = [m[obj1.name] for m in pareto_metrics if obj1.name in m]
        pareto_y = [m[obj2.name] for m in pareto_metrics if obj2.name in m]
        pareto_z = [m[obj3.name] for m in pareto_metrics if obj3.name in m]
        
        # Create hover text
        all_hover = [
            f"{obj1.name}: {m[obj1.name]:.4f}<br>"
            f"{obj2.name}: {m[obj2.name]:.4f}<br>"
            f"{obj3.name}: {m[obj3.name]:.4f}"
            for m in all_metrics 
            if obj1.name in m and obj2.name in m and obj3.name in m
        ]
        pareto_hover = [
            f"{obj1.name}: {m[obj1.name]:.4f}<br>"
            f"{obj2.name}: {m[obj2.name]:.4f}<br>"
            f"{obj3.name}: {m[obj3.name]:.4f}"
            for m in pareto_metrics
            if obj1.name in m and obj2.name in m and obj3.name in m
        ]
        
        # Create traces
        trace_all = go.Scatter3d(
            x=all_x, y=all_y, z=all_z,
            mode='markers',
            name='All Architectures',
            marker=dict(size=5, color='lightblue', opacity=0.4),
            hovertext=all_hover,
            hoverinfo='text'
        )
        
        trace_pareto = go.Scatter3d(
            x=pareto_x, y=pareto_y, z=pareto_z,
            mode='markers',
            name='Pareto Front',
            marker=dict(size=8, color='red', opacity=0.8,
                       line=dict(width=2, color='darkred')),
            hovertext=pareto_hover,
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(data=[trace_all, trace_pareto])
        
        fig.update_layout(
            title="Pareto Front Visualization (3D)",
            scene=dict(
                xaxis_title=f"{obj1.name} ({obj1.direction})",
                yaxis_title=f"{obj2.name} ({obj2.direction})",
                zaxis_title=f"{obj3.name} ({obj3.direction})"
            ),
            hovermode='closest',
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        if show:
            fig.show()
        
        return fig

"""
Logging utilities for Neural Architecture Search.

This module provides verbose logging capabilities for NAS operations including
architecture generation, evaluation, and search progress tracking.
"""

import time
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from ..utils.logger import get_logger

logger = get_logger(__name__)


class NASLogger:
    """
    Logger for NAS operations with verbose output and progress tracking.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize NAS logger.
        
        Args:
            verbose: Whether to enable verbose logging
        """
        self.verbose = verbose
        self.start_time = None
        self.architectures_evaluated = 0
        self.best_score = -float('inf')
        self.best_architecture_id = None
        self.search_history = []
    
    def log_search_start(
        self,
        search_strategy: str,
        search_space_type: str,
        time_budget: float,
        max_architectures: int
    ):
        """
        Log the start of NAS search.
        
        Args:
            search_strategy: Name of search strategy
            search_space_type: Type of search space
            time_budget: Time budget in seconds
            max_architectures: Maximum number of architectures to evaluate
        """
        self.start_time = time.time()
        self.architectures_evaluated = 0
        self.best_score = -float('inf')
        self.best_architecture_id = None
        self.search_history = []
        
        if self.verbose:
            logger.info("=" * 80)
            logger.info("🧠 Starting Neural Architecture Search")
            logger.info("=" * 80)
            logger.info(f"Search Strategy: {search_strategy}")
            logger.info(f"Search Space: {search_space_type}")
            logger.info(f"Time Budget: {time_budget:.0f}s ({time_budget/60:.1f} minutes)")
            logger.info(f"Max Architectures: {max_architectures}")
            logger.info("=" * 80)
    
    def log_architecture_generation(self, architecture_id: str, generation_method: str):
        """
        Log architecture generation.
        
        Args:
            architecture_id: Unique identifier for the architecture
            generation_method: Method used to generate architecture
        """
        if self.verbose:
            logger.info(f"🔨 Generated architecture {architecture_id[:8]} using {generation_method}")
    
    def log_architecture_evaluation_start(
        self,
        architecture_id: str,
        architecture_summary: str
    ):
        """
        Log the start of architecture evaluation.
        
        Args:
            architecture_id: Unique identifier for the architecture
            architecture_summary: Brief summary of architecture structure
        """
        if self.verbose:
            logger.info(f"⚙️  Evaluating architecture {architecture_id[:8]}")
            logger.info(f"   Structure: {architecture_summary}")
    
    def log_architecture_evaluation_complete(
        self,
        architecture_id: str,
        metrics: Dict[str, float],
        evaluation_time: float
    ):
        """
        Log completion of architecture evaluation.
        
        Args:
            architecture_id: Unique identifier for the architecture
            metrics: Performance metrics (accuracy, latency, model_size, etc.)
            evaluation_time: Time taken to evaluate in seconds
        """
        self.architectures_evaluated += 1
        
        # Update best score
        score = metrics.get('accuracy', metrics.get('score', 0))
        if score > self.best_score:
            self.best_score = score
            self.best_architecture_id = architecture_id
        
        # Record in history
        self.search_history.append({
            'architecture_id': architecture_id,
            'metrics': metrics,
            'evaluation_time': evaluation_time,
            'timestamp': time.time()
        })
        
        if self.verbose:
            # Format metrics
            metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                                    for k, v in metrics.items()])
            
            logger.info(f"✅ Architecture {architecture_id[:8]} evaluated in {evaluation_time:.2f}s")
            logger.info(f"   Metrics: {metrics_str}")
            
            # Show progress
            self._log_progress()
    
    def log_architecture_evaluation_failed(
        self,
        architecture_id: str,
        error: str
    ):
        """
        Log failed architecture evaluation.
        
        Args:
            architecture_id: Unique identifier for the architecture
            error: Error message
        """
        if self.verbose:
            logger.warning(f"❌ Architecture {architecture_id[:8]} evaluation failed: {error}")
    
    def log_architecture_skipped(
        self,
        architecture_id: str,
        reason: str
    ):
        """
        Log skipped architecture.
        
        Args:
            architecture_id: Unique identifier for the architecture
            reason: Reason for skipping
        """
        if self.verbose:
            logger.info(f"⏭️  Skipped architecture {architecture_id[:8]}: {reason}")
    
    def log_best_architecture_update(
        self,
        architecture_id: str,
        score: float,
        improvement: float
    ):
        """
        Log update to best architecture.
        
        Args:
            architecture_id: Unique identifier for the architecture
            score: New best score
            improvement: Improvement over previous best
        """
        if self.verbose:
            logger.info("=" * 80)
            logger.info(f"🏆 New best architecture found!")
            logger.info(f"   Architecture: {architecture_id[:8]}")
            logger.info(f"   Score: {score:.4f} (+{improvement:.4f})")
            logger.info("=" * 80)
    
    def log_search_progress(
        self,
        current_iteration: int,
        total_iterations: Optional[int] = None
    ):
        """
        Log search progress.
        
        Args:
            current_iteration: Current iteration number
            total_iterations: Total iterations (if known)
        """
        if self.verbose and current_iteration % 5 == 0:  # Log every 5 iterations
            self._log_progress(current_iteration, total_iterations)
    
    def _log_progress(
        self,
        current_iteration: Optional[int] = None,
        total_iterations: Optional[int] = None
    ):
        """Internal method to log progress with ETA."""
        if not self.start_time:
            return
        
        elapsed_time = time.time() - self.start_time
        
        # Calculate ETA if we have total iterations
        eta_str = ""
        if total_iterations and current_iteration:
            progress = current_iteration / total_iterations
            if progress > 0:
                estimated_total_time = elapsed_time / progress
                remaining_time = estimated_total_time - elapsed_time
                eta_str = f", ETA: {self._format_time(remaining_time)}"
        
        # Format progress message
        progress_msg = f"📊 Progress: {self.architectures_evaluated} architectures evaluated"
        if current_iteration and total_iterations:
            progress_msg += f" ({current_iteration}/{total_iterations})"
        
        progress_msg += f", Elapsed: {self._format_time(elapsed_time)}{eta_str}"
        
        logger.info(progress_msg)
        
        # Show best so far
        if self.best_architecture_id:
            logger.info(f"   Best so far: {self.best_architecture_id[:8]} (score: {self.best_score:.4f})")
    
    def log_search_complete(
        self,
        total_architectures: int,
        best_architecture_id: str,
        best_score: float,
        pareto_front_size: Optional[int] = None
    ):
        """
        Log completion of NAS search.
        
        Args:
            total_architectures: Total number of architectures evaluated
            best_architecture_id: ID of best architecture
            best_score: Best score achieved
            pareto_front_size: Size of Pareto front (for multi-objective)
        """
        if not self.start_time:
            return
        
        total_time = time.time() - self.start_time
        
        if self.verbose:
            logger.info("=" * 80)
            logger.info("🎉 Neural Architecture Search Complete!")
            logger.info("=" * 80)
            logger.info(f"Total Architectures Evaluated: {total_architectures}")
            logger.info(f"Total Time: {self._format_time(total_time)}")
            logger.info(f"Average Time per Architecture: {total_time/max(total_architectures, 1):.2f}s")
            logger.info(f"Best Architecture: {best_architecture_id[:8]}")
            logger.info(f"Best Score: {best_score:.4f}")
            
            if pareto_front_size:
                logger.info(f"Pareto Front Size: {pareto_front_size} architectures")
            
            logger.info("=" * 80)
    
    def log_checkpoint_saved(self, checkpoint_path: str, architectures_evaluated: int):
        """
        Log checkpoint save.
        
        Args:
            checkpoint_path: Path to checkpoint file
            architectures_evaluated: Number of architectures evaluated so far
        """
        if self.verbose:
            logger.info(f"💾 Checkpoint saved: {checkpoint_path} ({architectures_evaluated} architectures)")
    
    def log_checkpoint_loaded(self, checkpoint_path: str, architectures_evaluated: int):
        """
        Log checkpoint load.
        
        Args:
            checkpoint_path: Path to checkpoint file
            architectures_evaluated: Number of architectures in checkpoint
        """
        if self.verbose:
            logger.info(f"📂 Checkpoint loaded: {checkpoint_path} ({architectures_evaluated} architectures)")
    
    def log_hardware_constraint_check(
        self,
        architecture_id: str,
        constraints: Dict[str, Any],
        satisfied: bool
    ):
        """
        Log hardware constraint check.
        
        Args:
            architecture_id: Unique identifier for the architecture
            constraints: Hardware constraints checked
            satisfied: Whether constraints were satisfied
        """
        if self.verbose:
            status = "✅ satisfied" if satisfied else "❌ violated"
            constraints_str = ", ".join([f"{k}={v}" for k, v in constraints.items()])
            logger.info(f"🔍 Hardware constraints {status} for {architecture_id[:8]}: {constraints_str}")
    
    def log_transfer_learning(
        self,
        num_similar_architectures: int,
        similarity_threshold: float
    ):
        """
        Log transfer learning initialization.
        
        Args:
            num_similar_architectures: Number of similar architectures found
            similarity_threshold: Similarity threshold used
        """
        if self.verbose:
            logger.info(f"🔄 Transfer learning: Found {num_similar_architectures} similar architectures "
                       f"(threshold: {similarity_threshold:.2f})")
    
    def log_search_strategy_update(
        self,
        strategy_name: str,
        update_info: Dict[str, Any]
    ):
        """
        Log search strategy update.
        
        Args:
            strategy_name: Name of search strategy
            update_info: Information about the update
        """
        if self.verbose:
            info_str = ", ".join([f"{k}={v}" for k, v in update_info.items()])
            logger.info(f"🔄 {strategy_name} updated: {info_str}")
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in human-readable format.
        
        Args:
            seconds: Time in seconds
        
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_search_summary(self) -> Dict[str, Any]:
        """
        Get summary of search progress.
        
        Returns:
            Dictionary with search summary statistics
        """
        if not self.start_time:
            return {}
        
        elapsed_time = time.time() - self.start_time
        
        return {
            'architectures_evaluated': self.architectures_evaluated,
            'elapsed_time': elapsed_time,
            'best_score': self.best_score,
            'best_architecture_id': self.best_architecture_id,
            'avg_time_per_architecture': elapsed_time / max(self.architectures_evaluated, 1),
            'search_history': self.search_history
        }


def create_architecture_summary(architecture: Any) -> str:
    """
    Create a brief summary of an architecture for logging.
    
    Args:
        architecture: Architecture object
    
    Returns:
        Brief summary string
    """
    if not hasattr(architecture, 'layers'):
        return "Unknown architecture"
    
    # Count layer types
    layer_counts = {}
    for layer in architecture.layers:
        layer_type = layer.layer_type.lower()
        layer_counts[layer_type] = layer_counts.get(layer_type, 0) + 1
    
    # Format summary
    summary_parts = []
    for layer_type, count in sorted(layer_counts.items()):
        if count > 1:
            summary_parts.append(f"{count}x{layer_type}")
        else:
            summary_parts.append(layer_type)
    
    summary = " → ".join(summary_parts)
    
    # Add skip connections info
    if hasattr(architecture, 'connections') and architecture.connections:
        summary += f" + {len(architecture.connections)} skip connections"
    
    return summary

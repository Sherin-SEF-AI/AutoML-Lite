"""
Experiment Tracking for AutoML Lite.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import warnings

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Install with: pip install mlflow")

try:
    import wandb
    WANDB_AVAILABLE = True
except (ImportError, Exception):
    WANDB_AVAILABLE = False
    warnings.warn("Weights & Biases not available. Install with: pip install wandb")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install torch")

from ..utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """
    Comprehensive experiment tracking for AutoML Lite.
    """
    
    def __init__(
        self,
        tracking_backend: str = "mlflow",
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        log_artifacts: bool = True,
        log_metrics: bool = True,
        log_params: bool = True,
        log_models: bool = True,
        tracking_uri: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Experiment Tracker.
        
        Args:
            tracking_backend: Backend to use ('mlflow', 'wandb', 'tensorboard', 'local')
            experiment_name: Name of the experiment
            run_name: Name of the current run
            log_artifacts: Whether to log artifacts
            log_metrics: Whether to log metrics
            log_params: Whether to log parameters
            log_models: Whether to log models
            tracking_uri: URI for tracking server (MLflow)
            **kwargs: Additional backend-specific arguments
        """
        self.tracking_backend = tracking_backend.lower()
        self.experiment_name = experiment_name or "automl_lite_experiment"
        self.run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_artifacts_flag = log_artifacts
        self.log_metrics_flag = log_metrics
        self.log_params_flag = log_params
        self.log_models_flag = log_models
        self.tracking_uri = tracking_uri
        self.kwargs = kwargs
        
        # Initialize tracking
        self._initialize_tracking()
        
        # Run state
        self.is_active = False
        self.start_time = None
        self.metrics_buffer = []
        self.params_buffer = []
        self.artifacts_buffer = []
        
    def _initialize_tracking(self):
        """Initialize the tracking backend."""
        if self.tracking_backend == "mlflow":
            self._initialize_mlflow()
        elif self.tracking_backend == "wandb":
            self._initialize_wandb()
        elif self.tracking_backend == "tensorboard":
            self._initialize_tensorboard()
        elif self.tracking_backend == "local":
            self._initialize_local()
        else:
            raise ValueError(f"Unsupported tracking backend: {self.tracking_backend}")
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")
        
        # Set tracking URI
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(self.experiment_name)
        
        logger.info(f"MLflow tracking initialized for experiment: {self.experiment_name}")
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases tracking."""
        if not WANDB_AVAILABLE:
            raise ImportError("Weights & Biases not available. Install with: pip install wandb")
        
        # Initialize wandb
        wandb.init(
            project=self.experiment_name,
            name=self.run_name,
            config=self.kwargs.get('config', {}),
            **{k: v for k, v in self.kwargs.items() if k != 'config'}
        )
        
        logger.info(f"Weights & Biases tracking initialized for project: {self.experiment_name}")
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard tracking."""
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available. Install with: pip install torch")
        
        # Create log directory
        log_dir = Path("runs") / self.experiment_name / self.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize writer
        self.tensorboard_writer = SummaryWriter(str(log_dir))
        
        logger.info(f"TensorBoard tracking initialized at: {log_dir}")
    
    def _initialize_local(self):
        """Initialize local file-based tracking."""
        # Create experiment directory
        self.local_dir = Path("experiments") / self.experiment_name / self.run_name
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.local_dir / "metrics").mkdir(exist_ok=True)
        (self.local_dir / "params").mkdir(exist_ok=True)
        (self.local_dir / "artifacts").mkdir(exist_ok=True)
        (self.local_dir / "models").mkdir(exist_ok=True)
        
        logger.info(f"Local tracking initialized at: {self.local_dir}")
    
    def start_run(self):
        """Start a new experiment run."""
        self.is_active = True
        self.start_time = time.time()
        
        if self.tracking_backend == "mlflow":
            mlflow.start_run(run_name=self.run_name)
        elif self.tracking_backend == "wandb":
            # Already started in initialization
            pass
        elif self.tracking_backend == "tensorboard":
            # Already initialized
            pass
        elif self.tracking_backend == "local":
            # Create run metadata
            run_metadata = {
                "run_name": self.run_name,
                "start_time": datetime.now().isoformat(),
                "experiment_name": self.experiment_name
            }
            with open(self.local_dir / "run_metadata.json", "w") as f:
                json.dump(run_metadata, f, indent=2)
        
        logger.info(f"Started experiment run: {self.run_name}")
    
    def end_run(self):
        """End the current experiment run."""
        if not self.is_active:
            return
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        if self.tracking_backend == "mlflow":
            mlflow.end_run()
        elif self.tracking_backend == "wandb":
            wandb.finish()
        elif self.tracking_backend == "tensorboard":
            self.tensorboard_writer.close()
        elif self.tracking_backend == "local":
            # Update run metadata
            run_metadata = {
                "run_name": self.run_name,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat(),
                "duration_seconds": duration,
                "experiment_name": self.experiment_name
            }
            with open(self.local_dir / "run_metadata.json", "w") as f:
                json.dump(run_metadata, f, indent=2)
        
        self.is_active = False
        logger.info(f"Ended experiment run: {self.run_name} (duration: {duration:.2f}s)")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if not self.log_params_flag or not self.is_active:
            return
        
        if self.tracking_backend == "mlflow":
            mlflow.log_params(params)
        elif self.tracking_backend == "wandb":
            wandb.config.update(params)
        elif self.tracking_backend == "tensorboard":
            # TensorBoard doesn't have a direct params logging method
            # We'll store them in a text file
            pass
        elif self.tracking_backend == "local":
            # Save parameters to file
            params_file = self.local_dir / "params" / f"params_{int(time.time())}.json"
            with open(params_file, "w") as f:
                json.dump(params, f, indent=2)
        
        self.params_buffer.append(params)
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if not self.log_metrics_flag or not self.is_active:
            return
        
        if self.tracking_backend == "mlflow":
            mlflow.log_metrics(metrics, step=step)
        elif self.tracking_backend == "wandb":
            wandb.log(metrics, step=step)
        elif self.tracking_backend == "tensorboard":
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step or 0)
        elif self.tracking_backend == "local":
            # Save metrics to file
            metrics_data = {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "metrics": metrics
            }
            metrics_file = self.local_dir / "metrics" / f"metrics_{int(time.time())}.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)
        
        self.metrics_buffer.append(metrics)
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(self, local_path: Union[str, Path], artifact_path: Optional[str] = None):
        """Log an artifact."""
        if not self.log_artifacts_flag or not self.is_active:
            return
        
        local_path = Path(local_path)
        if not local_path.exists():
            logger.warning(f"Artifact not found: {local_path}")
            return
        
        if self.tracking_backend == "mlflow":
            mlflow.log_artifact(str(local_path), artifact_path)
        elif self.tracking_backend == "wandb":
            wandb.save(str(local_path), base_path=str(local_path.parent))
        elif self.tracking_backend == "tensorboard":
            # TensorBoard doesn't support artifact logging
            pass
        elif self.tracking_backend == "local":
            # Copy artifact to local directory
            import shutil
            dest_path = self.local_dir / "artifacts" / (artifact_path or local_path.name)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_path)
        
        self.artifacts_buffer.append(str(local_path))
        logger.debug(f"Logged artifact: {local_path}")
    
    def log_model(self, model, model_name: str = "model"):
        """Log a model."""
        if not self.log_models_flag or not self.is_active:
            return
        
        if self.tracking_backend == "mlflow":
            mlflow.sklearn.log_model(model, model_name)
        elif self.tracking_backend == "wandb":
            # Save model to temporary file and log as artifact
            import joblib
            temp_path = f"/tmp/{model_name}.pkl"
            joblib.dump(model, temp_path)
            wandb.save(temp_path)
        elif self.tracking_backend == "tensorboard":
            # TensorBoard doesn't support model logging
            pass
        elif self.tracking_backend == "local":
            # Save model to local directory
            import joblib
            model_path = self.local_dir / "models" / f"{model_name}.pkl"
            joblib.dump(model, model_path)
        
        logger.debug(f"Logged model: {model_name}")
    
    def log_automl_results(self, automl_results: Dict[str, Any]):
        """Log AutoML-specific results."""
        if not self.is_active:
            return
        
        # Log basic metrics
        if 'best_score' in automl_results:
            self.log_metrics({'best_score': automl_results['best_score']})
        
        if 'training_time' in automl_results:
            self.log_metrics({'training_time': automl_results['training_time']})
        
        if 'n_models_tried' in automl_results:
            self.log_metrics({'n_models_tried': automl_results['n_models_tried']})
        
        # Log parameters
        if 'config' in automl_results:
            self.log_params(automl_results['config'])
        
        # Log leaderboard as artifact
        if 'leaderboard' in automl_results:
            import pandas as pd
            leaderboard_df = pd.DataFrame(automl_results['leaderboard'])
            leaderboard_path = self.local_dir / "leaderboard.csv" if self.tracking_backend == "local" else "/tmp/leaderboard.csv"
            leaderboard_df.to_csv(leaderboard_path, index=False)
            self.log_artifact(leaderboard_path)
        
        # Log feature importance as artifact
        if 'feature_importance' in automl_results:
            import pandas as pd
            importance_df = pd.DataFrame(list(automl_results['feature_importance'].items()), 
                                       columns=['feature', 'importance'])
            importance_path = self.local_dir / "feature_importance.csv" if self.tracking_backend == "local" else "/tmp/feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            self.log_artifact(importance_path)
        
        logger.info("Logged AutoML results")
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of the current experiment."""
        if not self.is_active:
            return {}
        
        summary = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_backend": self.tracking_backend,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "n_metrics_logged": len(self.metrics_buffer),
            "n_params_logged": len(self.params_buffer),
            "n_artifacts_logged": len(self.artifacts_buffer)
        }
        
        return summary
    
    def __enter__(self):
        """Context manager entry."""
        self.start_run()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.end_run()


class ExperimentManager:
    """
    Manager for multiple experiments.
    """
    
    def __init__(self, base_experiment_name: str = "automl_lite"):
        """
        Initialize Experiment Manager.
        
        Args:
            base_experiment_name: Base name for experiments
        """
        self.base_experiment_name = base_experiment_name
        self.experiments = {}
    
    def create_experiment(
        self,
        experiment_name: str,
        tracking_backend: str = "mlflow",
        **kwargs
    ) -> ExperimentTracker:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            tracking_backend: Tracking backend to use
            **kwargs: Additional arguments for ExperimentTracker
            
        Returns:
            Experiment tracker
        """
        full_experiment_name = f"{self.base_experiment_name}_{experiment_name}"
        
        tracker = ExperimentTracker(
            tracking_backend=tracking_backend,
            experiment_name=full_experiment_name,
            **kwargs
        )
        
        self.experiments[experiment_name] = tracker
        return tracker
    
    def get_experiment(self, experiment_name: str) -> Optional[ExperimentTracker]:
        """Get an existing experiment."""
        return self.experiments.get(experiment_name)
    
    def list_experiments(self) -> List[str]:
        """List all experiments."""
        return list(self.experiments.keys())
    
    def close_all(self):
        """Close all experiments."""
        for tracker in self.experiments.values():
            if tracker.is_active:
                tracker.end_run()
        self.experiments.clear() 
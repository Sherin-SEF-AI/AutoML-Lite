"""
REST API for Model Serving using FastAPI

Provides endpoints for:
- Batch predictions
- Single predictions
- Model info
- Health checks
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import pickle
import joblib
import logging
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[List[float]] = Field(..., description="Feature matrix as list of lists")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    predictions: List[Union[float, int, str]]
    probabilities: Optional[List[List[float]]] = None
    model_name: str
    timestamp: str
    prediction_time_ms: float


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    model_type: str
    feature_names: List[str]
    n_features: int
    problem_type: str
    created_at: str
    version: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: Optional[str]
    uptime_seconds: float


class ModelServer:
    """
    Model serving wrapper for AutoML models

    Provides FastAPI application for serving predictions
    """

    def __init__(
        self,
        model: Any,
        model_name: str = "automl_model",
        feature_names: Optional[List[str]] = None,
        problem_type: str = "classification",
        version: str = "1.0"
    ):
        """
        Initialize Model Server

        Parameters:
        -----------
        model : Any
            Trained model
        model_name : str
            Name of the model
        feature_names : List[str], optional
            Feature names
        problem_type : str
            'classification' or 'regression'
        version : str
            Model version
        """
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names or []
        self.problem_type = problem_type
        self.version = version
        self.created_at = datetime.now().isoformat()
        self.start_time = datetime.now()

    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Make predictions

        Parameters:
        -----------
        features : np.ndarray
            Feature matrix

        Returns:
        --------
        result : Dict[str, Any]
            Prediction results with probabilities if available
        """
        import time
        start_time = time.time()

        # Make predictions
        predictions = self.model.predict(features)

        # Get probabilities if classification
        probabilities = None
        if self.problem_type == 'classification' and hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features).tolist()

        prediction_time = (time.time() - start_time) * 1000  # ms

        return {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'probabilities': probabilities,
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'prediction_time_ms': prediction_time
        }

    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'model_type': str(type(self.model).__name__),
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'problem_type': self.problem_type,
            'created_at': self.created_at,
            'version': self.version
        }

    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            'status': 'healthy',
            'model_loaded': self.model is not None,
            'model_name': self.model_name,
            'uptime_seconds': uptime
        }

    def create_fastapi_app(self):
        """
        Create FastAPI application

        Returns:
        --------
        app : FastAPI
            FastAPI application
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

        app = FastAPI(
            title=f"{self.model_name} API",
            description=f"Model serving API for {self.model_name}",
            version=self.version
        )

        # CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "message": f"Welcome to {self.model_name} API",
                "endpoints": {
                    "/predict": "POST - Make predictions",
                    "/info": "GET - Model information",
                    "/health": "GET - Health check"
                }
            }

        @app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """Prediction endpoint"""
            try:
                features = np.array(request.features)

                if len(self.feature_names) > 0 and features.shape[1] != len(self.feature_names):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected {len(self.feature_names)} features, got {features.shape[1]}"
                    )

                result = self.predict(features)
                return PredictionResponse(**result)

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/info", response_model=ModelInfo)
        async def info():
            """Model info endpoint"""
            return ModelInfo(**self.get_info())

        @app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint"""
            return HealthResponse(**self.health_check())

        return app

    def serve(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """
        Start serving the model

        Parameters:
        -----------
        host : str
            Host address
        port : int
            Port number
        reload : bool
            Whether to enable auto-reload
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn not installed. Install with: pip install uvicorn")

        app = self.create_fastapi_app()

        logger.info(f"Starting model server at http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=reload)

    def save(self, filepath: str):
        """Save model server to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> 'ModelServer':
        """Load model server from file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_model_server(
    model_path: str,
    model_name: str = "automl_model",
    **kwargs
) -> ModelServer:
    """
    Create model server from saved model

    Parameters:
    -----------
    model_path : str
        Path to saved model
    model_name : str
        Name for the model
    **kwargs
        Additional parameters for ModelServer

    Returns:
    --------
    server : ModelServer
        Model server instance
    """
    # Load model
    try:
        model = joblib.load(model_path)
    except Exception:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

    return ModelServer(model=model, model_name=model_name, **kwargs)


# Example CLI for serving
def main():
    """CLI for model serving"""
    import argparse

    parser = argparse.ArgumentParser(description="Serve AutoML model via REST API")
    parser.add_argument("model_path", type=str, help="Path to saved model")
    parser.add_argument("--name", type=str, default="automl_model", help="Model name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--problem-type", type=str, default="classification",
                       choices=['classification', 'regression'], help="Problem type")

    args = parser.parse_args()

    # Create and start server
    server = create_model_server(
        model_path=args.model_path,
        model_name=args.name,
        problem_type=args.problem_type
    )

    server.serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

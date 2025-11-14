"""
AutoML Lite - A simplified automated machine learning package for non-experts.

This package provides end-to-end ML automation with intelligent preprocessing,
model selection, and hyperparameter optimization.
"""

__version__ = "0.2.0"
__author__ = "Sherin Joseph Roy"
__email__ = "sherin@deepmost.ai"

from .core.automl import AutoMLite

__all__ = [
    "AutoMLite",
] 
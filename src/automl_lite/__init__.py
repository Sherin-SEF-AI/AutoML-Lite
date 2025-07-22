"""
AutoML Lite - A simplified automated machine learning package for non-experts.

This package provides end-to-end ML automation with intelligent preprocessing,
model selection, and hyperparameter optimization.
"""

__version__ = "0.1.1"
__author__ = "sherin joseph roy"
__email__ = "sherin.joseph2217@gmail.com"

from .core.automl import AutoMLite

__all__ = [
    "AutoMLite",
] 
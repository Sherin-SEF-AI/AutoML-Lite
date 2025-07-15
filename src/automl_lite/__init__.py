"""
AutoML Lite - A simplified automated machine learning package for non-experts.

This package provides end-to-end ML automation with intelligent preprocessing,
model selection, and hyperparameter optimization.
"""

__version__ = "0.1.0"
__author__ = "AutoML Lite Team"
__email__ = "team@automl-lite.org"

from .core.automl import AutoMLite

__all__ = [
    "AutoMLite",
] 
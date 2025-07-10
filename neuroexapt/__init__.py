"""
Neuro Exapt: A revolutionary neural network framework based on information theory
for dynamic architecture optimization.

This framework integrates information bottleneck principles with adaptive entropy
control to enable dynamic structural evolution of neural networks during training.
"""

__version__ = "0.1.0"
__author__ = "Neuro Exapt Team"
__email__ = "team@neuroexapt.ai"

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("Neuro Exapt requires Python 3.8 or later")

# Import main classes when available
try:
    from .neuroexapt import NeuroExapt
    from .trainer import Trainer
    
    __all__ = [
        "NeuroExapt",
        "Trainer",
        "__version__",
    ]
except ImportError:
    # During initial setup, these imports might fail
    __all__ = ["__version__"] 
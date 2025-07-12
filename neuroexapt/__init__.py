"""
NeuroExapt: Information-Theoretic Dynamic Architecture Optimization

A revolutionary neural network framework that enables adaptive architecture evolution
during training using information theory principles.

Features:
- Every-epoch architecture checking with intelligent evolution
- Information-theoretic analysis for layer importance
- Dynamic architecture optimization during training
- Automatic rollback on performance degradation
"""

__version__ = "1.0.0"
__author__ = "NeuroExapt Development Team"
__email__ = "team@neuroexapt.ai"

# Version check
import sys
if sys.version_info < (3, 8):
    raise RuntimeError("NeuroExapt requires Python 3.8 or later")

# Import main components
from .neuroexapt import NeuroExapt
from .trainer import Trainer, train_with_neuroexapt

# Try to import optional mathematical utilities
try:
    from .math import (
        calculate_entropy,
        calculate_mutual_information,
        calculate_network_complexity,
        DiscreteParameterOptimizer
    )
    _math_available = True
except ImportError:
    _math_available = False

_main_available = True

# Export list
__all__ = [
    "NeuroExapt",
    "Trainer", 
    "train_with_neuroexapt",
    "__version__",
]

# Add mathematical exports if available
if _math_available:
    __all__.extend([
        "calculate_entropy",
        "calculate_mutual_information", 
        "calculate_network_complexity",
        "DiscreteParameterOptimizer",
    ])

# Package info
__description__ = "Information-Theoretic Dynamic Architecture Optimization"
__url__ = "https://github.com/neuroexapt/neuroexapt"
__license__ = "MIT"

# Configuration
import os
import warnings

# Suppress warnings for cleaner output (can be overridden)
if os.environ.get('NEUROEXAPT_VERBOSE', '0') == '0':
    warnings.filterwarnings('ignore', category=UserWarning, module='neuroexapt')

# Suppress all welcome messages to keep output clean
import os

# Only show welcome message if explicitly requested
if os.environ.get('NEUROEXAPT_SHOW_WELCOME', '0') == '1':
    print("🧠 NeuroExapt: Information-Theoretic Neural Network Optimization") 